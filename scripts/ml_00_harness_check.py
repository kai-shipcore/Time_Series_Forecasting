#!/usr/bin/env python3
"""ML experiment 00 — validate the harness, no ML involved.

Runs the production default for short-smooth SKUs — WindowAverage(12),
a flat forecast at the mean of the last 12 training weeks — through the
new src/ml split + scoring pipeline.

Purpose: if this prints pooled WAPEs in the ballpark of the numbers we
already know from run_test_evaluation.py / the July backtests (~0.22 for
smooth/short), the ruler works and every later LightGBM comparison on
this harness is trustworthy. Note: exact figures WILL differ from the
Apr–Jun backtest (different window, no deseasonalization here) — we are
checking plausibility, not equality.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ml.dataset import dev_splits, load_weekly
from src.ml.evaluate import score, score_table


def wa_forecast(train: pd.DataFrame, test: pd.DataFrame, window: int) -> pd.DataFrame:
    """WindowAverage(window): flat forecast = mean of last `window` train weeks."""
    level = (
        train.sort_values("ds")
        .groupby("unique_id")["y"]
        .apply(lambda s: s.tail(window).mean())
        .rename("yhat")
        .reset_index()
    )
    grid = test[["unique_id", "ds"]].drop_duplicates()
    return grid.merge(level, on="unique_id", how="left").fillna({"yhat": 0.0})


def main() -> None:
    weekly, profiles = load_weekly()
    smooth = profiles.loc[profiles["bucket"] == "smooth", "unique_id"]
    weekly = weekly[weekly["unique_id"].isin(smooth)]
    print(f"Scoring {smooth.nunique()} smooth SKUs "
          f"({weekly['ds'].min().date()} → {weekly['ds'].max().date()})\n")

    for split in dev_splits(weekly, n=3):
        print(split)
        results = {
            "WA12": score(wa_forecast(split.train, split.test, 12), split, profiles),
            "WA8":  score(wa_forecast(split.train, split.test, 8),  split, profiles),
            "naive_last_week": score(
                wa_forecast(split.train, split.test, 1), split, profiles
            ),
        }
        print(score_table(results).to_string(), "\n")
        # n_skus / demand context once per split
        print(results["WA12"][["segment", "n_skus", "actual_units"]]
              .to_string(index=False), "\n")


if __name__ == "__main__":
    main()
