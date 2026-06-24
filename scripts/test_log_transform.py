#!/usr/bin/env python3
"""
Log-transform CV test — smooth SKUs only.

Runs the same StatsForecast cross-validation as the main backtest but on
log1p(y). Predictions are inverse-transformed (expm1, clipped to 0) before
computing errors, so MASE/WAPE are on the original demand scale and are
directly comparable to cv_metrics.csv.

Does NOT write to any production outputs — purely diagnostic.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from statsforecast import StatsForecast

from config import FREQUENCY, N_CV_SPLITS, OUTPUTS_REPORTS, TRIM_TRAILING_WEEKS, TEST_WEEKS
from src.models import get_models
from src.baselines import get_baselines

PROCESSED = ROOT / "data" / "processed"
CV_METRICS = OUTPUTS_REPORTS / "cv_metrics.csv"


def _naive_mae(train_df: pd.DataFrame) -> pd.DataFrame:
    def _mae(y):
        d = np.abs(np.diff(y))
        return float(d.mean()) if len(d) > 0 else 1.0
    result = (
        train_df.sort_values(["unique_id", "ds"])
        .groupby("unique_id")["y"]
        .apply(lambda s: _mae(s.values))
        .reset_index(name="naive_mae")
    )
    result["naive_mae"] = result["naive_mae"].clip(lower=1e-6)
    return result


def run_log_cv(weekly: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    # Mirror backtest.py trimming / split exactly
    all_weeks = sorted(weekly["ds"].unique())
    trimmed_weeks = all_weeks[:-TRIM_TRAILING_WEEKS] if TRIM_TRAILING_WEEKS else all_weeks
    weekly = weekly[weekly["ds"].isin(trimmed_weeks)].copy()
    test_start = trimmed_weeks[-TEST_WEEKS]

    train_df = weekly[weekly["ds"] < test_start].copy()
    train_starts = pd.to_datetime(profiles.set_index("unique_id")["train_start"])
    train_df["_ts"] = train_df["unique_id"].map(train_starts)
    train_trimmed = (
        train_df[train_df["ds"] >= train_df["_ts"]]
        .drop(columns="_ts")
        [["unique_id", "ds", "y"]]
    )

    smooth_full = profiles.loc[
        (profiles["bucket"] == "smooth") & (profiles["history_length"] == "full"),
        "unique_id",
    ].tolist()

    df_group = train_trimmed[train_trimmed["unique_id"].isin(smooth_full)].copy()
    # Save original y for denominator, then transform
    orig_y = df_group[["unique_id", "ds", "y"]].copy()
    df_group["y"] = np.log1p(df_group["y"])

    candidates = get_models("smooth", "full")
    cand_names = {type(m).__name__ for m in candidates}
    baselines = [b for b in get_baselines("smooth", "full") if type(b).__name__ not in cand_names]
    models = candidates + baselines
    print(f"Models: {[type(m).__name__ for m in models]}")
    print(f"SKUs:   {df_group['unique_id'].nunique()}  |  n_windows={N_CV_SPLITS}  |  h={TEST_WEEKS}")

    sf = StatsForecast(models=models, freq=FREQUENCY, n_jobs=-1)
    cv_log = sf.cross_validation(df=df_group, h=TEST_WEEKS, n_windows=N_CV_SPLITS, step_size=TEST_WEEKS)

    # Inverse-transform all prediction columns (not y — we keep that in log space for now)
    meta = {"unique_id", "ds", "cutoff", "y"}
    model_cols = [c for c in cv_log.columns if c not in meta]
    for col in model_cols:
        cv_log[col] = np.expm1(cv_log[col]).clip(lower=0)
    # Restore original-scale y for error computation
    cv_log = cv_log.drop(columns="y").merge(
        weekly[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="left"
    )

    # naive_mae on original scale
    naive_mae_df = _naive_mae(orig_y)

    # Compute per-SKU × model MASE / WAPE on original scale
    long = cv_log.melt(
        id_vars=["unique_id", "ds", "y"],
        value_vars=model_cols,
        var_name="model",
        value_name="yhat",
    ).dropna(subset=["yhat"]).merge(naive_mae_df, on="unique_id", how="left")

    long["abs_err"]    = (long["y"] - long["yhat"]).abs()
    long["scaled_err"] = long["abs_err"] / long["naive_mae"]

    m = (
        long.groupby(["unique_id", "model"])
        .agg(
            MASE_log   =("scaled_err", "mean"),
            _sum_y     =("y",          "sum"),
            _sum_ae    =("abs_err",    "sum"),
        )
        .reset_index()
    )
    m["WAPE_log"] = (m["_sum_ae"] / m["_sum_y"].clip(lower=1e-6)).round(4)
    return m.drop(columns=["_sum_y", "_sum_ae"])


def main():
    weekly   = pd.read_parquet(PROCESSED / "sales_clean.parquet")
    profiles = pd.read_csv(PROCESSED / "sku_profiles.csv")
    profiles["train_start"] = pd.to_datetime(profiles["train_start"])
    weekly["ds"] = pd.to_datetime(weekly["ds"])

    print("Running log-transform CV on smooth/full SKUs...\n")
    log_metrics = run_log_cv(weekly, profiles)

    if not CV_METRICS.exists():
        print("cv_metrics.csv not found — run selector.py first for comparison.")
        print(log_metrics.groupby("model")[["MASE_log", "WAPE_log"]].median().round(3).to_string())
        return

    baseline = pd.read_csv(CV_METRICS)
    baseline = baseline[
        (baseline["bucket"] == "smooth") & (baseline["history_length"] == "full")
        & ~baseline["model"].str.startswith("Ensemble:")
    ][["unique_id", "model", "MASE", "WAPE"]].rename(columns={"MASE": "MASE_orig", "WAPE": "WAPE_orig"})

    merged = baseline.merge(log_metrics, on=["unique_id", "model"], how="inner")
    merged["MASE_delta"] = merged["MASE_log"] - merged["MASE_orig"]
    merged["WAPE_delta"] = merged["WAPE_log"] - merged["WAPE_orig"]

    print("\n── Per-model median metrics (smooth/full, original vs log-transformed y) ──")
    summary = (
        merged.groupby("model")
        .agg(
            MASE_orig=("MASE_orig", "median"),
            MASE_log =("MASE_log",  "median"),
            MASE_delta=("MASE_delta", "median"),
            WAPE_orig=("WAPE_orig", "median"),
            WAPE_log =("WAPE_log",  "median"),
            WAPE_delta=("WAPE_delta", "median"),
        )
        .round(3)
    )
    print(summary.to_string())

    wins_log  = (merged["MASE_log"]  < merged["MASE_orig"]).sum()
    wins_orig = (merged["MASE_log"]  > merged["MASE_orig"]).sum()
    ties      = len(merged) - wins_log - wins_orig
    print(f"\nLog better (MASE↓): {wins_log}  |  Original better: {wins_orig}  |  Tie: {ties}")
    print("(Negative MASE_delta = log transform helps for that SKU × model pair)")

    out = OUTPUTS_REPORTS / "log_transform_test.csv"
    merged.to_csv(out, index=False)
    print(f"\nFull comparison saved: {out}")


if __name__ == "__main__":
    main()
