"""
Regenerate shipcore.fc_forward_forecasts_test by running the REAL forward-forecast
pipeline with a cutoff of 13 weeks ago.

This is identical to run_forward_forecast.py except:
  - last_complete_monday is forced to 13 weeks before today's last Monday
  - results are written to fc_forward_forecasts_test instead of fc_forward_forecasts
  - --skip-ingest is the default (reuses existing sales_clean.parquet)

Run:  python scripts/seed_backtest_test.py
      python scripts/seed_backtest_test.py --ingest   # re-pull raw data first
"""

import sys, time, copy, argparse, shutil, tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sqlalchemy import text
from statsforecast import StatsForecast
from statsforecast.utils import ConformalIntervals

from config import FREQUENCY, TRIM_TRAILING_WEEKS, USE_SEASONAL_ADJUSTMENT, OUTPUTS_REPORTS
from src.ingest import ingest
from src.clean import clean
from src.profile import profile
from src.backtest import backtest, _trim_to_train_start
from src.selector import select
from src.models import get_models
from src.baselines import get_baselines
from src.deseasonalize import deseasonalize, reseasonalize
from src.db import get_engine
from src.v1 import load_raw_for_v1, build_index, compute_v1_per_week

FORWARD_WEEKS       = 13
CONFORMAL_LEVEL     = 70
CONFORMAL_N_WINDOWS = 5
PROCESSED_DIR       = ROOT / "data" / "processed"


def _dedupe_models(bucket, hist):
    candidates      = get_models(bucket, hist)
    candidate_names = {type(m).__name__ for m in candidates}
    baselines       = [b for b in get_baselines(bucket, hist) if type(b).__name__ not in candidate_names]
    return candidates + baselines


def _pick_yhat(fcast, model_name):
    if model_name.startswith("Ensemble:"):
        parts = model_name.replace("Ensemble:", "").split("+")
        cols  = [c for c in parts if c in fcast.columns]
        if cols:
            return fcast[cols].mean(axis=1)
    if model_name in fcast.columns:
        return fcast[model_name]
    avail = [c for c in fcast.columns if c not in {"unique_id", "ds"}]
    return fcast[avail[0]] if avail else pd.Series([np.nan] * len(fcast))


def _pick_intervals(fcast, model_name):
    lo_suf = f"-lo-{CONFORMAL_LEVEL}"
    hi_suf = f"-hi-{CONFORMAL_LEVEL}"
    if model_name.startswith("Ensemble:"):
        parts   = model_name.replace("Ensemble:", "").split("+")
        lo_cols = [f"{p}{lo_suf}" for p in parts if f"{p}{lo_suf}" in fcast.columns]
        hi_cols = [f"{p}{hi_suf}" for p in parts if f"{p}{hi_suf}" in fcast.columns]
        lo = fcast[lo_cols].mean(axis=1) if lo_cols else pd.Series([np.nan] * len(fcast))
        hi = fcast[hi_cols].mean(axis=1) if hi_cols else pd.Series([np.nan] * len(fcast))
    else:
        lo_col, hi_col = f"{model_name}{lo_suf}", f"{model_name}{hi_suf}"
        lo = fcast[lo_col] if lo_col in fcast.columns else pd.Series([np.nan] * len(fcast))
        hi = fcast[hi_col] if hi_col in fcast.columns else pd.Series([np.nan] * len(fcast))
    return lo.reset_index(drop=True), hi.reset_index(drop=True)


def refit_and_forecast(weekly, profiles, selection, cutoff, horizon_weeks=FORWARD_WEEKS):
    all_weeks  = sorted(weekly["ds"].unique())
    trimmed    = all_weeks[:-TRIM_TRAILING_WEEKS] if TRIM_TRAILING_WEEKS else all_weeks
    train_full = weekly[weekly["ds"].isin(trimmed)].copy()
    train_trimmed = _trim_to_train_start(train_full, profiles)

    sel_map  = selection.set_index("unique_id")["model"].to_dict()
    conf_map = selection.set_index("unique_id")["forecast_confidence"].to_dict()
    rows = []

    for bucket in ("smooth", "low_volume"):
        for hist in ("full", "medium", "short"):
            skus = profiles.loc[
                (profiles["bucket"] == bucket) & (profiles["history_length"] == hist),
                "unique_id",
            ].tolist()
            if not skus:
                continue
            train_g = train_trimmed[train_trimmed["unique_id"].isin(skus)].copy()
            if train_g.empty:
                continue

            # short included since Jul 2026 — keep in sync with run_forward_forecast.py
            use_deseas = USE_SEASONAL_ADJUSTMENT and bucket == "smooth"
            fit_data   = deseasonalize(train_g) if use_deseas else train_g
            fit_data   = fit_data[["unique_id", "ds", "y"]]

            model_min  = 20 if bucket == "smooth" else 8
            min_series = train_g.groupby("unique_id")["ds"].count().min()
            n_windows  = max(0, min(CONFORMAL_N_WINDOWS, (min_series - model_min) // horizon_weeks))
            use_pi     = n_windows >= 1

            t0     = time.time()
            models = copy.deepcopy(_dedupe_models(bucket, hist))

            if use_pi:
                pi    = ConformalIntervals(h=horizon_weeks, n_windows=n_windows)
                sf    = StatsForecast(models=models, freq=FREQUENCY, n_jobs=-1)
                fcast = sf.forecast(df=fit_data, h=horizon_weeks,
                                    level=[CONFORMAL_LEVEL], prediction_intervals=pi)
            else:
                sf = StatsForecast(models=models, freq=FREQUENCY, n_jobs=-1)
                sf.fit(fit_data)
                fcast = sf.predict(h=horizon_weeks)

            fcast["ds"] = pd.to_datetime(fcast["ds"])
            if use_deseas:
                fcast = reseasonalize(fcast)

            pi_label = f"n_windows={n_windows}" if use_pi else "no PI"
            print(f"  {bucket}/{hist}: {len(skus)} SKUs  [{pi_label}]  ({time.time()-t0:.1f}s)")

            for uid, uid_fcast in fcast.groupby("unique_id"):
                uid_fcast  = uid_fcast.sort_values("ds").reset_index(drop=True)
                model_name = sel_map.get(uid, "")
                preds      = _pick_yhat(uid_fcast, model_name)
                if use_pi:
                    lo_s, hi_s = _pick_intervals(uid_fcast, model_name)
                else:
                    lo_s = pd.Series([np.nan] * len(uid_fcast))
                    hi_s = pd.Series([np.nan] * len(uid_fcast))

                for ds_val, yhat_val, lo_val, hi_val in zip(
                    uid_fcast["ds"].values, preds.values, lo_s.values, hi_s.values
                ):
                    rows.append({
                        "unique_id":      uid,
                        "ds":             pd.Timestamp(ds_val),
                        "yhat":           max(0.0, float(yhat_val)) if pd.notna(yhat_val) else 0.0,
                        "yhat_lo":        max(0.0, float(lo_val))  if pd.notna(lo_val)  else None,
                        "yhat_hi":        max(0.0, float(hi_val))  if pd.notna(hi_val)  else None,
                        "bucket":         bucket,
                        "history_length": hist,
                        "selected_model": model_name,
                        "confidence":     conf_map.get(uid, "standard"),
                    })

    return pd.DataFrame(rows)


def write_test_forecasts(df: pd.DataFrame, forecast_date) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("""
            DROP TABLE IF EXISTS shipcore.fc_forward_forecasts_test;
            CREATE TABLE shipcore.fc_forward_forecasts_test (
                unique_id      TEXT             NOT NULL,
                forecast_date  DATE             NOT NULL,
                ds             DATE             NOT NULL,
                yhat           DOUBLE PRECISION NOT NULL,
                yhat_lo        DOUBLE PRECISION,
                yhat_hi        DOUBLE PRECISION,
                bucket         TEXT,
                history_length TEXT,
                selected_model TEXT,
                confidence     TEXT,
                v1_yhat        DOUBLE PRECISION
            );
        """))
        records = df.to_dict("records")
        CHUNK = 500
        for i in range(0, len(records), CHUNK):
            conn.execute(text("""
                INSERT INTO shipcore.fc_forward_forecasts_test
                    (unique_id, forecast_date, ds, yhat, yhat_lo, yhat_hi,
                     bucket, history_length, selected_model, confidence, v1_yhat)
                VALUES
                    (:unique_id, :forecast_date, :ds, :yhat, :yhat_lo, :yhat_hi,
                     :bucket, :history_length, :selected_model, :confidence, :v1_yhat)
            """), records[i:i + CHUNK])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action="store_true",
                        help="Re-pull raw data from DB (default: reuse sales_clean.parquet)")
    args = parser.parse_args()

    # ── Cutoff: 13 weeks before today's last Monday ───────────────────────────
    today             = pd.Timestamp.today().normalize()
    days_since_monday = today.dayofweek or 7
    last_monday       = today - pd.Timedelta(days=days_since_monday)
    cutoff            = last_monday - pd.Timedelta(weeks=13)
    print(f"Cutoff (last_complete_monday): {cutoff.date()}  (13 weeks before {last_monday.date()})")

    # ── Step 0: Ingest (optional) ─────────────────────────────────────────────
    if args.ingest:
        print("\n── Step 0: Ingest fresh data from DB ───────────────────────────")
        raw = ingest()
        print(f"  {len(raw):,} rows | {raw['link_master_sku'].nunique():,} SKUs"
              f" | {raw['order_date'].min().date()} → {raw['order_date'].max().date()}")
        weekly_all = clean(raw)
        print(f"  Cleaned → {len(weekly_all):,} weekly rows")
    else:
        print("\n── Step 0: Reusing existing sales_clean.parquet ────────────────")

    # ── Step 1: Load and cap to cutoff ────────────────────────────────────────
    print("\n── Step 1: Load + cap to cutoff ────────────────────────────────")
    weekly_all = pd.read_parquet(PROCESSED_DIR / "sales_clean.parquet")
    weekly_all["ds"] = pd.to_datetime(weekly_all["ds"])
    weekly = weekly_all[weekly_all["ds"] <= cutoff].copy()
    print(f"  {weekly['unique_id'].nunique():,} SKUs | {len(weekly):,} rows"
          f" | up to {weekly['ds'].max().date()}")

    # ── Snapshot shared artifact files ────────────────────────────────────────
    # profile() / backtest() / select() below overwrite the shared local
    # artifacts with as-of-cutoff versions. Without restoring them, the API
    # (which reads sku_profiles.csv live for active_weeks / graduation) and any
    # analysis on cv_results/selection silently use stale capped data until the
    # next full pipeline run. Snapshot here, restore in `finally` (crash-safe).
    artifacts = [
        PROCESSED_DIR / "sku_profiles.csv",
        OUTPUTS_REPORTS / "cv_results.parquet",
        OUTPUTS_REPORTS / "test_set.parquet",
        OUTPUTS_REPORTS / "selection.csv",
        OUTPUTS_REPORTS / "cv_metrics.csv",
    ]
    backup_dir = Path(tempfile.mkdtemp(prefix="seed_artifact_backup_"))
    backed_up = [p for p in artifacts if p.exists()]
    for p in backed_up:
        shutil.copy2(p, backup_dir / p.name)

    try:
        # ── Step 1b: Profile ──────────────────────────────────────────────────
        print("\n── Step 1b: Profile ────────────────────────────────────────────")
        profiles = profile(weekly)
        profiles["train_start"] = pd.to_datetime(profiles["train_start"])

        # ── Step 2: Backtest (CV for model selection) ─────────────────────────
        print("\n── Step 2: Backtest (CV for model selection) ───────────────────")
        backtest(weekly, profiles)

        # ── Step 3: Select ────────────────────────────────────────────────────
        print("\n── Step 3: Select ──────────────────────────────────────────────")
        selection = select(weekly, profiles)

        # ── Step 4: Refit on full data + forecast forward from cutoff ─────────
        print("\n── Step 4: Refit + forecast ─────────────────────────────────────")
        print(f"  Horizon: {FORWARD_WEEKS} weeks ahead of {cutoff.date()}")
        forecasts = refit_and_forecast(weekly, profiles, selection, cutoff)
        print(f"  {len(forecasts):,} rows | {forecasts['unique_id'].nunique()} SKUs")

        # ── Step 4b: V1 baseline ──────────────────────────────────────────────
        print("\n── Step 4b: Compute V1 baseline ────────────────────────────────")
        v1_raw       = load_raw_for_v1()
        v1_index     = build_index(v1_raw)
        smooth_ids   = forecasts["unique_id"].unique().tolist()
        v1_map       = compute_v1_per_week(smooth_ids, cutoff, FORWARD_WEEKS, v1_index)
        forecasts["v1_yhat"] = forecasts["unique_id"].map(v1_map)
        print(f"  V1 computed for {len(v1_map)} SKUs")

        # ── Step 5: Write to test table ───────────────────────────────────────
        print("\n── Step 5: Write to fc_forward_forecasts_test ──────────────────")
        forecasts["forecast_date"] = cutoff.date()
        write_test_forecasts(forecasts, cutoff.date())
        print(f"  Inserted {len(forecasts):,} rows  (forecast_date={cutoff.date()})")

        # ── Verify ────────────────────────────────────────────────────────────
        engine = get_engine()
        with engine.connect() as conn:
            check = conn.execute(text("""
                SELECT forecast_date, MIN(ds) AS h_start, MAX(ds) AS h_end,
                       COUNT(DISTINCT ds) AS weeks, COUNT(DISTINCT unique_id) AS skus,
                       COUNT(v1_yhat) AS v1_rows
                FROM shipcore.fc_forward_forecasts_test
                GROUP BY forecast_date ORDER BY forecast_date
            """)).fetchall()

        print("\n── Result ──────────────────────────────────────────────────────")
        for r in check:
            print(f"  {r[0]}  {r[1]} → {r[2]}  ({r[3]}W, {r[4]} SKUs, {r[5]} V1 values)")
        eligible = sum(1 for r in check if pd.Timestamp(str(r[2])) <= last_monday)
        print(f"  Cycles visible in /backtest-cycles: {eligible}")
    finally:
        for p in backed_up:
            shutil.copy2(backup_dir / p.name, p)
        shutil.rmtree(backup_dir, ignore_errors=True)
        print(f"\n  Restored {len(backed_up)} shared artifact file(s) (full-data versions preserved)")


if __name__ == "__main__":
    main()
