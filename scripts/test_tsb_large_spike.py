#!/usr/bin/env python3
"""
TSB vs Croston/IMAPA/HistoricAverage on large-spike intermittent SKUs.

'Large spike' = max single-week demand >= 8 units (109 SKUs).
These are intermittent items that occasionally receive meaningful bulk orders
rather than purely 1-unit sporadic demand.

TSB (Teunter-Syntetos-Babai) separately tracks demand probability and demand
size, allowing the probability to decay toward zero — better suited to items
with bursty/clustered demand than Croston's fixed inter-arrival assumption.

Tests three TSB alpha pairs alongside the current intermittent candidate set.
Does NOT modify any production outputs.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    CrostonOptimized, IMAPA, HistoricAverage, TSB,
)

from config import FREQUENCY, N_CV_SPLITS, OUTPUTS_REPORTS, TRIM_TRAILING_WEEKS, TEST_WEEKS

PROCESSED   = ROOT / "data" / "processed"
POLICY_PATH = ROOT / "outputs" / "forecasts" / "intermittent_policy.csv"
SPIKE_MAX_THRESHOLD = 8   # large spike = max_demand >= this

MODELS = [
    CrostonOptimized(),
    IMAPA(),
    HistoricAverage(),
    TSB(alpha_d=0.1, alpha_p=0.1, alias="TSB_010_010"),
    TSB(alpha_d=0.2, alpha_p=0.1, alias="TSB_020_010"),
    TSB(alpha_d=0.1, alpha_p=0.2, alias="TSB_010_020"),
    TSB(alpha_d=0.3, alpha_p=0.3, alias="TSB_030_030"),
]


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


def main():
    weekly   = pd.read_parquet(PROCESSED / "sales_clean.parquet")
    profiles = pd.read_csv(PROCESSED / "sku_profiles.csv")
    policy   = pd.read_csv(POLICY_PATH)
    profiles["train_start"] = pd.to_datetime(profiles["train_start"])
    weekly["ds"] = pd.to_datetime(weekly["ds"])

    # Large-spike intermittent SKUs
    spike_skus = set(policy.loc[policy["max_demand"] >= SPIKE_MAX_THRESHOLD, "unique_id"])
    print(f"Large-spike SKUs (max_demand >= {SPIKE_MAX_THRESHOLD}): {len(spike_skus)}")

    # Mirror backtest trimming / split
    all_weeks    = sorted(weekly["ds"].unique())
    trimmed      = all_weeks[:-TRIM_TRAILING_WEEKS] if TRIM_TRAILING_WEEKS else all_weeks
    weekly       = weekly[weekly["ds"].isin(trimmed)].copy()
    test_start   = trimmed[-TEST_WEEKS]
    train_df     = weekly[weekly["ds"] < test_start].copy()

    # Trim to train_start per SKU
    train_starts = pd.to_datetime(profiles.set_index("unique_id")["train_start"])
    train_df["_ts"] = train_df["unique_id"].map(train_starts)
    train_trimmed = (
        train_df[train_df["ds"] >= train_df["_ts"]]
        .drop(columns="_ts")
        [["unique_id", "ds", "y"]]
    )

    df = train_trimmed[train_trimmed["unique_id"].isin(spike_skus)].copy()

    # Drop series too short for N_CV_SPLITS windows
    min_len = N_CV_SPLITS * TEST_WEEKS + 1
    lengths = df.groupby("unique_id")["ds"].count()
    too_short = lengths[lengths < min_len].index.tolist()
    if too_short:
        print(f"  Skipping {len(too_short)} SKUs with < {min_len} weeks of history")
        df = df[~df["unique_id"].isin(too_short)]

    print(f"SKUs with sufficient CV history: {df['unique_id'].nunique()}")
    print(f"Models: {[type(m).__name__ if not hasattr(m,'alias') else getattr(m,'alias',type(m).__name__) for m in MODELS]}")
    print()

    sf = StatsForecast(models=MODELS, freq=FREQUENCY, n_jobs=-1)
    cv = sf.cross_validation(
        df=df, h=TEST_WEEKS, n_windows=N_CV_SPLITS, step_size=TEST_WEEKS
    )

    naive_mae_df = _naive_mae(df)

    meta_cols = {"unique_id", "ds", "cutoff", "y"}
    model_cols = [c for c in cv.columns if c not in meta_cols]

    long = (
        cv.melt(id_vars=["unique_id", "ds", "y"], value_vars=model_cols,
                var_name="model", value_name="yhat")
        .dropna(subset=["yhat"])
        .merge(naive_mae_df, on="unique_id", how="left")
    )
    long["abs_err"]    = (long["y"] - long["yhat"]).abs()
    long["scaled_err"] = long["abs_err"] / long["naive_mae"]

    metrics = (
        long.groupby(["unique_id", "model"])
        .agg(
            MASE    =("scaled_err", "mean"),
            _sum_y  =("y",          "sum"),
            _sum_ae =("abs_err",    "sum"),
            Bias    =("yhat",       "mean"),
        )
        .reset_index()
    )
    metrics["WAPE"] = (metrics["_sum_ae"] / metrics["_sum_y"].clip(lower=1e-6)).round(4)

    print("── Median metrics across large-spike SKUs ──")
    summary = metrics.groupby("model")[["MASE", "WAPE"]].median().round(3)
    print(summary.sort_values("MASE").to_string())

    print()
    print("── Win count (lowest MASE per SKU) ──")
    winners = metrics.loc[metrics.groupby("unique_id")["MASE"].idxmin(), "model"]
    print(winners.value_counts().to_string())

    print()
    print("── Margin: winner vs runner-up MASE ──")
    wide = metrics.pivot_table(index="unique_id", columns="model", values="MASE")
    margin = wide.apply(lambda r: r.dropna().sort_values().iloc[1] - r.dropna().sort_values().iloc[0]
                        if r.dropna().shape[0] >= 2 else np.nan, axis=1)
    print(margin.describe(percentiles=[.25, .5, .75, .9]).round(3).to_string())
    print(f"Margin < 0.05 (essentially tied): {(margin < 0.05).sum()} / {margin.notna().sum()} SKUs")

    # Save detailed results
    out = OUTPUTS_REPORTS / "tsb_spike_test.csv"
    metrics.to_csv(out, index=False)
    print(f"\nDetailed results saved: {out}")


if __name__ == "__main__":
    main()
