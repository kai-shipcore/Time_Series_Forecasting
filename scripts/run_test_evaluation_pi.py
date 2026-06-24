#!/usr/bin/env python3
"""
End-to-end test-set evaluation.

Phase 1  Backtest   — CV on training data with deseasonalization + holiday flag
Phase 2  Select     — Pick best model per SKU by MASE over CV windows
Phase 3  Refit      — Retrain selected model on full training data
Phase 4  Predict    — Forecast the held-out test weeks (TEST_WEEKS=10)
Phase 5  Score      — Compare against test actuals and V1 formula

Config from config.py (current values):
  USE_SEASONAL_ADJUSTMENT = True   smooth SKUs deseasonalized before fitting
  USE_HOLIDAY_FLAG        = True   Nov 20 – Dec 31 window × 1.26
  N_CV_SPLITS             = 6      full-history SKUs
  TEST_WEEKS              = 10     held-out evaluation window
  TRIM_TRAILING_WEEKS     = 3      noisy tail dropped
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import copy
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.utils import ConformalIntervals

from config import (
    FREQUENCY, TEST_WEEKS, TRIM_TRAILING_WEEKS,
    USE_SEASONAL_ADJUSTMENT, HOLIDAY_START, HOLIDAY_END, HOLIDAY_MULTIPLIER,
    ROUTE_SHORT_SMOOTH_TO_V1,
    OUTPUTS_REPORTS,
)

# ── Conformal interval settings (smooth SKUs only) ───────────────────────────
# n_windows : internal CV folds for residual calibration.
#             More = tighter (better quantile estimate), but slower and needs
#             more history.  5 needs ~50 + training weeks; safe for full-history
#             SKUs (92+ weeks).
# level     : conformal percentile.  Conformal tends to slightly overcover,
#             so 85 → ~88-92% actual coverage.  Adjust ±5 based on results.
CONFORMAL_N_WINDOWS = 5
CONFORMAL_LEVEL = 70

from src.models import get_models
from src.baselines import get_baselines
from src.deseasonalize import deseasonalize, reseasonalize
from src.backtest import backtest, _trim_to_train_start
from src.selector import select

PROCESSED_DIR = ROOT / "data" / "processed"


# ── Phase 3 helpers: refit + predict ─────────────────────────────────────────

def _dedupe_models(bucket: str, hist: str) -> list:
    """Candidates + baselines without duplicates — mirrors backtest.py exactly."""
    candidates     = get_models(bucket, hist)
    candidate_names = {type(m).__name__ for m in candidates}
    baselines       = [b for b in get_baselines(bucket, hist)
                       if type(b).__name__ not in candidate_names]
    return candidates + baselines


def _forecast_group(train_g: pd.DataFrame, bucket: str, hist: str) -> pd.DataFrame:
    """Fit all models on full training data; return TEST_WEEKS ahead predictions.
    Smooth SKUs get conformal prediction intervals; other buckets get point-only.

    In statsforecast 2.x, prediction_intervals must be passed to forecast(),
    not the constructor — so smooth SKUs use the combined forecast() call.

    n_windows is capped per group so conformal calibration folds never leave
    AutoETS with fewer than 20 training rows (its 'tiny datasets' threshold)."""
    models     = copy.deepcopy(_dedupe_models(bucket, hist))
    use_deseas = USE_SEASONAL_ADJUSTMENT and bucket == "smooth" and hist != "short"
    use_pi     = bucket == "smooth" and hist == "full"

    fit_data = deseasonalize(train_g) if use_deseas else train_g
    fit_data  = fit_data[["unique_id", "ds", "y"]]

    sf = StatsForecast(models=models, freq=FREQUENCY, n_jobs=-1)

    if use_pi:
        min_series = train_g.groupby("unique_id")["ds"].count().min()
        # Each conformal window removes TEST_WEEKS rows from the front of training.
        # Keep enough rows for AutoETS: series_len - n_windows*h >= 20.
        AUTOETS_MIN = 20
        n_windows = max(1, min(CONFORMAL_N_WINDOWS,
                               (min_series - AUTOETS_MIN) // TEST_WEEKS))
        pi = ConformalIntervals(h=TEST_WEEKS, n_windows=n_windows)
        fcast = sf.forecast(df=fit_data, h=TEST_WEEKS, level=[CONFORMAL_LEVEL],
                            prediction_intervals=pi)
    else:
        sf.fit(fit_data)
        fcast = sf.predict(h=TEST_WEEKS)

    fcast["ds"] = pd.to_datetime(fcast["ds"])

    if use_deseas:
        fcast = reseasonalize(fcast)

    return fcast


def _pick_yhat(fcast: pd.DataFrame, model_name: str) -> pd.Series:
    """Extract the selected model's predictions. Handles Ensemble:A+B names."""
    if model_name.startswith("Ensemble:"):
        parts = model_name.replace("Ensemble:", "").split("+")
        cols  = [c for c in parts if c in fcast.columns]
        if cols:
            return fcast[cols].mean(axis=1)
    if model_name in fcast.columns:
        return fcast[model_name]
    # Fallback to first non-id column
    avail = [c for c in fcast.columns if c not in {"unique_id", "ds"}]
    return fcast[avail[0]] if avail else pd.Series([np.nan] * len(fcast))


def _pick_intervals(fcast: pd.DataFrame, model_name: str) -> dict:
    """Extract lo/hi interval columns for the selected model.
    For ensembles, averages the component intervals (conservative:
    ensemble variance < component variance, so this slightly overestimates
    width — safe for replenishment)."""
    lo_suffix = f"-lo-{CONFORMAL_LEVEL}"
    hi_suffix = f"-hi-{CONFORMAL_LEVEL}"

    if model_name.startswith("Ensemble:"):
        parts = model_name.replace("Ensemble:", "").split("+")
        lo_cols = [f"{p}{lo_suffix}" for p in parts if f"{p}{lo_suffix}" in fcast.columns]
        hi_cols = [f"{p}{hi_suffix}" for p in parts if f"{p}{hi_suffix}" in fcast.columns]
        lo = fcast[lo_cols].mean(axis=1) if lo_cols else pd.Series([np.nan] * len(fcast))
        hi = fcast[hi_cols].mean(axis=1) if hi_cols else pd.Series([np.nan] * len(fcast))
    else:
        lo_col = f"{model_name}{lo_suffix}"
        hi_col = f"{model_name}{hi_suffix}"
        lo = fcast[lo_col] if lo_col in fcast.columns else pd.Series([np.nan] * len(fcast))
        hi = fcast[hi_col] if hi_col in fcast.columns else pd.Series([np.nan] * len(fcast))

    return {"lo": lo, "hi": hi}


def refit_and_predict(
    weekly: pd.DataFrame,
    profiles: pd.DataFrame,
    selection: pd.DataFrame,
) -> pd.DataFrame:
    all_weeks     = sorted(weekly["ds"].unique())
    trimmed       = all_weeks[:-TRIM_TRAILING_WEEKS] if TRIM_TRAILING_WEEKS else all_weeks
    test_start    = trimmed[-TEST_WEEKS]
    cutoff        = pd.Timestamp(trimmed[-(TEST_WEEKS + 1)])
    train_full    = weekly[weekly["ds"].isin(trimmed) & (weekly["ds"] < test_start)].copy()
    train_trimmed = _trim_to_train_start(train_full, profiles)

    # Build V1 index once if any SKUs route to V1
    v1_index = None
    if ROUTE_SHORT_SMOOTH_TO_V1:
        sys.path.insert(0, str(ROOT / "scripts"))
        from compare_v1 import build_cumsum_index, v1_forecast as _v1_forecast
        raw_path = PROCESSED_DIR / "orders_raw.parquet"
        if raw_path.exists():
            raw = pd.read_parquet(raw_path)
            raw["order_date"] = pd.to_datetime(raw["order_date"])
            v1_index = build_cumsum_index(raw)

    sel_map = selection.set_index("unique_id")["model"].to_dict()
    rows    = []

    for bucket in ("smooth", "intermittent", "low_volume"):
        for hist in ("full", "medium", "short"):
            skus = profiles.loc[
                (profiles["bucket"] == bucket) & (profiles["history_length"] == hist),
                "unique_id",
            ].tolist()
            if not skus:
                continue

            # V1 routing: smooth/short SKUs get V1 forecast directly
            if ROUTE_SHORT_SMOOTH_TO_V1 and bucket == "smooth" and hist == "short":
                t0 = time.time()
                n_ok = 0
                for uid in skus:
                    yhat = np.nan
                    if v1_index is not None:
                        try:
                            yhat = _v1_forecast(v1_index, uid, cutoff)
                            n_ok += 1
                        except Exception:
                            pass
                    rows.append({
                        "unique_id":      uid,
                        "yhat_total":     float(yhat),
                        "selected_model": "V1",
                        "bucket":         bucket,
                        "history_length": hist,
                    })
                print(f"  {bucket}/{hist}: {len(skus)} SKUs → V1  "
                      f"({n_ok} computed, {time.time()-t0:.1f}s)")
                continue

            train_g = train_trimmed[train_trimmed["unique_id"].isin(skus)].copy()
            if train_g.empty:
                continue

            t0      = time.time()
            fcast_g = _forecast_group(train_g, bucket, hist)
            elapsed = time.time() - t0
            print(f"  {bucket}/{hist}: {len(skus)} SKUs  ({elapsed:.1f}s)")

            for uid, uid_fcast in fcast_g.groupby("unique_id"):
                uid_fcast  = uid_fcast.sort_values("ds")
                model_name = sel_map.get(uid, "")
                preds      = _pick_yhat(uid_fcast, model_name)
                yhat_total = float(preds.sum())
                row = {
                    "unique_id":      uid,
                    "yhat_total":     yhat_total,
                    "selected_model": model_name,
                    "bucket":         bucket,
                    "history_length": hist,
                }
                # Conformal intervals (smooth/full only — medium lacks history for calibration)
                if bucket == "smooth" and hist == "full":
                    bounds = _pick_intervals(uid_fcast, model_name)
                    row["yhat_lo_90"] = max(0, float(bounds["lo"].clip(lower=0).sum()))
                    row["yhat_hi_90"] = max(0, float(bounds["hi"].clip(lower=0).sum()))
                rows.append(row)

    forecasts = pd.DataFrame(rows)
    return forecasts.merge(
        selection[["unique_id", "forecast_confidence", "MASE", "WAPE"]],
        on="unique_id", how="left",
    )


# ── V1 comparison (best-effort) ───────────────────────────────────────────────

def _try_v1(cutoff: pd.Timestamp, uids: list) -> dict:
    try:
        raw_path = PROCESSED_DIR / "orders_raw.parquet"
        if not raw_path.exists():
            print("  V1: orders_raw.parquet not found — skipping")
            return {}
        sys.path.insert(0, str(ROOT / "scripts"))
        from compare_v1 import build_cumsum_index, v1_forecast
        raw = pd.read_parquet(raw_path)
        raw["order_date"] = pd.to_datetime(raw["order_date"])
        index = build_cumsum_index(raw)
        v1 = {}
        for uid in uids:
            try:
                v1[uid] = v1_forecast(index, uid, cutoff)
            except Exception:
                pass
        print(f"  V1: computed for {len(v1)}/{len(uids)} SKUs")
        return v1
    except Exception as e:
        print(f"  V1: skipped ({e})")
        return {}


# ── Phase 5: Score + report ───────────────────────────────────────────────────

def score_and_report(
    forecasts: pd.DataFrame,
    test_set: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    actuals = test_set.groupby("unique_id")["y"].sum().reset_index(name="actual_total")
    results = forecasts.merge(actuals, on="unique_id", how="left")
    results["actual_total"] = results["actual_total"].fillna(0)
    results["ae"]   = (results["actual_total"] - results["yhat_total"]).abs()
    results["bias"] = results["yhat_total"] - results["actual_total"]

    # Fetch V1 only for smooth SKUs that use a statistical model (not those already routed to V1)
    v1_compare_uids = results.loc[
        (results["bucket"] == "smooth") & (results["selected_model"] != "V1"),
        "unique_id",
    ].tolist()
    print(f"\nFetching V1 forecasts ({len(v1_compare_uids)} statistical-model smooth SKUs)...")
    v1_map = _try_v1(cutoff, v1_compare_uids)
    if v1_map:
        results["v1_total"] = results["unique_id"].map(v1_map)
        results["ae_v1"]    = (results["actual_total"] - results["v1_total"]).abs()

    print(f"\n{'='*68}")
    print(f"TEST SET RESULTS   cutoff={cutoff.date()}   window={TEST_WEEKS} weeks")
    print(f"Deseas=ON  Holiday={HOLIDAY_START}–{HOLIDAY_END} ×{HOLIDAY_MULTIPLIER}")
    print(f"{'='*68}")

    for bucket in ("smooth", "intermittent", "low_volume"):
        sub = results[results["bucket"] == bucket]
        if sub.empty:
            continue
        n      = len(sub)
        mae    = sub["ae"].mean()
        wape   = sub["ae"].sum() / max(sub["actual_total"].sum(), 1e-6)
        bias   = sub["bias"].mean()
        within = (sub["ae"] / sub["actual_total"].clip(lower=1) < 0.25).sum()

        print(f"\n── {bucket.upper()}  ({n} SKUs) ──────────────────────────")
        print(f"  MAE        {mae:>10.2f}  units per SKU")
        print(f"  WAPE       {wape:>10.4f}")
        print(f"  Bias       {bias:>+10.2f}  units (+ = over-forecast)")
        print(f"  Within 25% {within:>10} / {n} SKUs")

        if "ae_v1" in results.columns and bucket == "smooth":
            # V1 comparison only applies to statistical-model SKUs; V1-routed SKUs are excluded
            sv = sub[sub["ae_v1"].notna()]
            n_v1_routed = (sub["selected_model"] == "V1").sum()
            if n_v1_routed:
                print(f"\n  Routing: {n_v1_routed} SKUs → V1 directly (short-history)")
            if not sv.empty:
                mae_v1  = sv["ae_v1"].mean()
                wape_v1 = sv["ae_v1"].sum() / max(sv["actual_total"].sum(), 1e-6)
                delta   = (mae_v1 - sv["ae"].mean()) / mae_v1 * 100
                print(f"  vs V1 ({len(sv)} statistical-model SKUs):")
                print(f"  V1 MAE     {mae_v1:>10.2f}   our model {delta:+.1f}%")
                print(f"  V1 WAPE    {wape_v1:>10.4f}")

        # Model breakdown
        print(f"\n  Model selection:")
        for model, grp in sub.groupby("selected_model"):
            print(f"    {model:<32} {len(grp):>5} SKUs  "
                  f"MAE={grp['ae'].mean():>7.2f}  bias={grp['bias'].mean():>+7.2f}")

        # Confidence breakdown for smooth
        if bucket == "smooth" and "forecast_confidence" in results.columns:
            print(f"\n  Confidence:")
            for conf, grp in sub.groupby("forecast_confidence"):
                print(f"    {conf:<12}  {len(grp):>4} SKUs  "
                      f"MAE={grp['ae'].mean():>7.2f}  "
                      f"MASE={grp['MASE'].mean():.3f}")

        # Prediction interval coverage (conformal, smooth only)
        lo_col, hi_col = "yhat_lo_90", "yhat_hi_90"
        if lo_col in sub.columns and hi_col in sub.columns:
            sub_pi = sub[sub[hi_col].notna()]
            if not sub_pi.empty:
                covered = ((sub_pi["actual_total"] >= sub_pi[lo_col]) &
                           (sub_pi["actual_total"] <= sub_pi[hi_col])).sum()
                avg_width = (sub_pi[hi_col] - sub_pi[lo_col]).mean()
                avg_hi = sub_pi[hi_col].mean()
                print(f"\n  Prediction interval ({len(sub_pi)} SKUs, "
                      f"conformal level={CONFORMAL_LEVEL}, n_windows={CONFORMAL_N_WINDOWS}):")
                print(f"    Coverage     {covered}/{len(sub_pi)}  "
                      f"({covered/len(sub_pi)*100:.0f}% — target ~90%)")
                print(f"    Avg width    {avg_width:>10.1f} units")
                print(f"    Avg upper    {avg_hi:>10.1f} units  "
                      f"(replenishment safety level)")

        # Top 10 worst SKUs
        worst = sub.nlargest(10, "ae")[["unique_id", "actual_total", "yhat_total", "ae", "bias", "selected_model"]]
        print(f"\n  Top 10 worst errors:")
        print(f"  {'SKU':<20} {'Actual':>8} {'Forecast':>10} {'AE':>8} {'Bias':>8}  Model")
        for _, row in worst.iterrows():
            print(f"  {str(row['unique_id']):<20} {row['actual_total']:>8.1f} "
                  f"{row['yhat_total']:>10.1f} {row['ae']:>8.1f} {row['bias']:>+8.1f}  "
                  f"{row['selected_model']}")

    total_actual = results["actual_total"].sum()
    total_yhat   = results["yhat_total"].sum()
    print(f"\n── PORTFOLIO TOTALS ─────────────────────────────────────────────")
    print(f"  Actual demand  {total_actual:>12,.0f} units")
    print(f"  Forecast total {total_yhat:>12,.0f} units")
    print(f"  Portfolio bias {total_yhat - total_actual:>+12,.0f} units  "
          f"({(total_yhat/total_actual - 1)*100:+.1f}%)")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    weekly   = pd.read_parquet(PROCESSED_DIR / "sales_clean.parquet")
    profiles = pd.read_csv(PROCESSED_DIR / "sku_profiles.csv")
    weekly["ds"]            = pd.to_datetime(weekly["ds"])
    profiles["train_start"] = pd.to_datetime(profiles["train_start"])
    print(f"  {weekly['unique_id'].nunique():,} SKUs | {len(weekly):,} rows\n")

    # Compute cutoff before backtest mutates nothing (read-only here)
    all_weeks  = sorted(weekly["ds"].unique())
    trimmed    = all_weeks[:-TRIM_TRAILING_WEEKS] if TRIM_TRAILING_WEEKS else all_weeks
    test_start = pd.Timestamp(trimmed[-TEST_WEEKS])
    cutoff     = pd.Timestamp(trimmed[-(TEST_WEEKS + 1)])

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    print("── Phase 1: Cross-validation backtest ───────────────────────────────")
    cv_df, test_set = backtest(weekly, profiles)

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    print("\n── Phase 2: Model selection ──────────────────────────────────────────")
    selection = select(weekly, profiles)

    # ── Phase 3 + 4 ───────────────────────────────────────────────────────────
    print("\n── Phase 3 + 4: Refit on full training data → predict test period ────")
    print(f"  Training through {cutoff.date()} | Test: {test_start.date()} + {TEST_WEEKS} weeks")
    print(f"  Conformal intervals: level={CONFORMAL_LEVEL}, n_windows={CONFORMAL_N_WINDOWS} (smooth only)")
    forecasts = refit_and_predict(weekly, profiles, selection)

    # ── Phase 5 ───────────────────────────────────────────────────────────────
    print("\n── Phase 5: Score ────────────────────────────────────────────────────")
    results = score_and_report(forecasts, test_set, cutoff)

    out = OUTPUTS_REPORTS / "test_evaluation.csv"
    results.to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
