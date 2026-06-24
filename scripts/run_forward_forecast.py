#!/usr/bin/env python3
"""
Forward forecast job.

Full pipeline on every run:
  ingest → clean → profile → backtest (CV) → select → refit on ALL data → predict forward → write to DB

Run whenever new data arrives (weekly cron or manually):
  python3 scripts/run_forward_forecast.py
"""
import sys, time, copy
from datetime import date
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.utils import ConformalIntervals

from config import FREQUENCY, TEST_WEEKS, TRIM_TRAILING_WEEKS, USE_SEASONAL_ADJUSTMENT
from src.profile import profile
from src.backtest import backtest, _trim_to_train_start
from src.selector import select
from src.models import get_models
from src.baselines import get_baselines
from src.deseasonalize import deseasonalize, reseasonalize
from src.db import write_forward_forecasts

FORWARD_WEEKS      = 13
CONFORMAL_LEVEL    = 70
CONFORMAL_N_WINDOWS = 5


def _dedupe_models(bucket: str, hist: str) -> list:
    candidates      = get_models(bucket, hist)
    candidate_names = {type(m).__name__ for m in candidates}
    baselines       = [b for b in get_baselines(bucket, hist)
                       if type(b).__name__ not in candidate_names]
    return candidates + baselines


def _pick_yhat(fcast: pd.DataFrame, model_name: str) -> pd.Series:
    if model_name.startswith("Ensemble:"):
        parts = model_name.replace("Ensemble:", "").split("+")
        cols  = [c for c in parts if c in fcast.columns]
        if cols:
            return fcast[cols].mean(axis=1)
    if model_name in fcast.columns:
        return fcast[model_name]
    avail = [c for c in fcast.columns if c not in {"unique_id", "ds"}]
    return fcast[avail[0]] if avail else pd.Series([np.nan] * len(fcast))


def _pick_intervals(fcast: pd.DataFrame, model_name: str):
    lo_suf = f"-lo-{CONFORMAL_LEVEL}"
    hi_suf = f"-hi-{CONFORMAL_LEVEL}"
    if model_name.startswith("Ensemble:"):
        parts   = model_name.replace("Ensemble:", "").split("+")
        lo_cols = [f"{p}{lo_suf}" for p in parts if f"{p}{lo_suf}" in fcast.columns]
        hi_cols = [f"{p}{hi_suf}" for p in parts if f"{p}{hi_suf}" in fcast.columns]
        lo = fcast[lo_cols].mean(axis=1) if lo_cols else pd.Series([np.nan] * len(fcast))
        hi = fcast[hi_cols].mean(axis=1) if hi_cols else pd.Series([np.nan] * len(fcast))
    else:
        lo_col = f"{model_name}{lo_suf}"
        hi_col = f"{model_name}{hi_suf}"
        lo = fcast[lo_col] if lo_col in fcast.columns else pd.Series([np.nan] * len(fcast))
        hi = fcast[hi_col] if hi_col in fcast.columns else pd.Series([np.nan] * len(fcast))
    return lo.reset_index(drop=True), hi.reset_index(drop=True)


def refit_and_forecast(
    weekly: pd.DataFrame,
    profiles: pd.DataFrame,
    selection: pd.DataFrame,
) -> pd.DataFrame:
    """Refit selected models on ALL trimmed data; predict FORWARD_WEEKS ahead."""
    all_weeks  = sorted(weekly["ds"].unique())
    trimmed    = all_weeks[:-TRIM_TRAILING_WEEKS] if TRIM_TRAILING_WEEKS else all_weeks
    train_full = weekly[weekly["ds"].isin(trimmed)].copy()
    train_trimmed = _trim_to_train_start(train_full, profiles)

    sel_map  = selection.set_index("unique_id")["model"].to_dict()
    conf_map = selection.set_index("unique_id")["forecast_confidence"].to_dict()
    rows = []

    for bucket in ("smooth", "low_volume"):   # intermittent excluded
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

            use_deseas = USE_SEASONAL_ADJUSTMENT and bucket == "smooth" and hist != "short"

            fit_data = deseasonalize(train_g) if use_deseas else train_g
            fit_data = fit_data[["unique_id", "ds", "y"]]

            # Minimum training rows needed by the heaviest model in this set:
            # smooth uses AutoETS (needs ~20 obs); low_volume uses WindowAverage (needs ~8)
            model_min  = 20 if bucket == "smooth" else 8
            min_series = train_g.groupby("unique_id")["ds"].count().min()
            n_windows  = max(0, min(CONFORMAL_N_WINDOWS,
                                    (min_series - model_min) // FORWARD_WEEKS))
            use_pi = n_windows >= 1

            t0     = time.time()
            models = copy.deepcopy(_dedupe_models(bucket, hist))

            if use_pi:
                pi    = ConformalIntervals(h=FORWARD_WEEKS, n_windows=n_windows)
                sf    = StatsForecast(models=models, freq=FREQUENCY, n_jobs=-1)
                fcast = sf.forecast(df=fit_data, h=FORWARD_WEEKS,
                                    level=[CONFORMAL_LEVEL], prediction_intervals=pi)
            else:
                sf    = StatsForecast(models=models, freq=FREQUENCY, n_jobs=-1)
                sf.fit(fit_data)
                fcast = sf.predict(h=FORWARD_WEEKS)

            fcast["ds"] = pd.to_datetime(fcast["ds"])
            if use_deseas:
                fcast = reseasonalize(fcast)

            pi_label = f"n_windows={n_windows}" if use_pi else "no PI (too short)"
            print(f"  {bucket}/{hist}: {len(skus)} SKUs  [{pi_label}]  ({time.time()-t0:.1f}s)")

            for uid, uid_fcast in fcast.groupby("unique_id"):
                uid_fcast  = uid_fcast.sort_values("ds").reset_index(drop=True)
                model_name = sel_map.get(uid, "")

                # smooth/short SKUs may be mapped to "V1" — fall back to best available
                preds = _pick_yhat(uid_fcast, model_name)

                if use_pi:
                    lo_s, hi_s = _pick_intervals(uid_fcast, model_name)
                else:
                    lo_s = pd.Series([np.nan] * len(uid_fcast))
                    hi_s = pd.Series([np.nan] * len(uid_fcast))

                for ds_val, yhat_val, lo_val, hi_val in zip(
                    uid_fcast["ds"].values,
                    preds.values,
                    lo_s.values,
                    hi_s.values,
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


PROCESSED_DIR = ROOT / "data" / "processed"


def main():
    # Load from processed parquet — the DB only holds recent weeks, not full history.
    # Re-run src/ingest.py + src/clean.py separately when you need to refresh the base data.
    print("── Step 1: Load processed data ─────────────────────────────────")
    weekly = pd.read_parquet(PROCESSED_DIR / "sales_clean.parquet")
    weekly["ds"] = pd.to_datetime(weekly["ds"])
    print(f"  {weekly['unique_id'].nunique():,} SKUs | {len(weekly):,} rows"
          f" | {weekly['ds'].min().date()} → {weekly['ds'].max().date()}")

    print("\n── Step 1b: Profile ────────────────────────────────────────────")
    profiles = profile(weekly)
    profiles["train_start"] = pd.to_datetime(profiles["train_start"])

    print("\n── Step 2: Backtest (CV for model selection) ───────────────────")
    backtest(weekly, profiles)

    print("\n── Step 3: Select ──────────────────────────────────────────────")
    selection = select(weekly, profiles)

    print("\n── Step 4: Refit on full data + forward forecast ───────────────")
    print(f"  Horizon: {FORWARD_WEEKS} weeks ahead")
    # Find the last fully completed week. W-MON labels each week with the Monday
    # it ends on, so the last complete week = the most recent Monday before today.
    # If today IS Monday, that Monday is just starting so we go back 7 days.
    today = pd.Timestamp.today().normalize()
    days_back = today.dayofweek or 7   # Mon=0 → 7, Tue=1 → 1, ..., Sun=6 → 6
    last_complete_monday = today - pd.Timedelta(days=days_back)
    weekly_capped = weekly[weekly["ds"] <= last_complete_monday].copy()
    dropped = weekly[weekly["ds"] > last_complete_monday]["ds"].nunique()
    print(f"  Training through {last_complete_monday.date()}"
          f"{f' (dropped {dropped} incomplete week(s))' if dropped else ''}")
    forecasts = refit_and_forecast(weekly_capped, profiles, selection)
    print(f"  {len(forecasts):,} rows | {forecasts['unique_id'].nunique()} SKUs")

    print("\n── Step 5: Write to DB ─────────────────────────────────────────")
    forecasts["forecast_date"] = date.today()
    write_forward_forecasts(forecasts)
    print(f"  Done — forecast_date={date.today()}")


if __name__ == "__main__":
    main()
