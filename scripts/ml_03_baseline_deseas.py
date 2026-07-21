#!/usr/bin/env python3
"""ML experiment 03: the restart baseline (design Section 4.10, step 0).

Zero-feature structural baseline: deseasonalize history with the production
seasonal factors, take each SKU's trailing 12-week deseasonalized mean as its
level, forecast level x target-week factor. This is "predict ratio 1.0 on the
deseasonalized scale". No LightGBM involved.

Purpose:
  1. Establish the floor every future feature must beat.
  2. Measure how much structural seasonality alone fixes the post-holiday
     window, where raw WA12 over-forecast badly (bias +22%).

Compared against raw WA12 (no seasonal round-trip) on the three dev windows,
with the ramp cohort and bootstrap significance.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ml.seasonal import ml_factors as _factors
from src.ml.dataset import dev_splits, load_weekly
from src.ml.evaluate import (
    bootstrap_delta, cohort_score, ramp_cohort, score, score_table,
)


def wa12_raw(train, test):
    lvl = (
        train.sort_values("ds").groupby("unique_id")["y"]
        .apply(lambda s: s.tail(12).mean()).rename("yhat").reset_index()
    )
    grid = test[["unique_id", "ds"]].drop_duplicates()
    return grid.merge(lvl, on="unique_id", how="left").fillna({"yhat": 0.0})


def wa12_deseas(train, test):
    """Deseasonalized WA12: level from factor-adjusted history, forecast
    re-scaled by each target week's factor."""
    tr = train.sort_values(["unique_id", "ds"]).copy()
    tr["y_flat"] = tr["y"] / _factors(tr["ds"]).to_numpy()
    lvl = tr.groupby("unique_id")["y_flat"].apply(lambda s: s.tail(12).mean())

    grid = test[["unique_id", "ds"]].drop_duplicates().copy()
    grid["yhat"] = (
        grid["unique_id"].map(lvl).fillna(0.0) * _factors(grid["ds"]).to_numpy()
    )
    return grid


def main():
    weekly, profiles = load_weekly()
    smooth = profiles.loc[profiles["bucket"] == "smooth", "unique_id"]
    weekly = weekly[weekly["unique_id"].isin(smooth)]

    for split in dev_splits(weekly, n=3):
        print(f"\n{'=' * 62}\n{split}")
        raw = wa12_raw(split.train, split.test)
        des = wa12_deseas(split.train, split.test)

        results = {
            "WA12_raw": score(raw, split, profiles),
            "WA12_deseas": score(des, split, profiles),
        }
        print("\npooled WAPE by segment:")
        print(score_table(results).to_string())
        print("\nbias% by segment:")
        bias = {n: t.set_index("segment")["bias_pct"] for n, t in results.items()}
        print(pd.DataFrame(bias).to_string())

        for seg in ("short", "long"):
            bd = bootstrap_delta(des, raw, split, profiles, segment=seg)
            print(f"bootstrap deseas-vs-raw [{seg}]: {bd}")

        ramps = ramp_cohort(split, profiles)
        print(f"ramp cohort (n={len(ramps)}):")
        for label, p in [("raw", raw), ("deseas", des)]:
            c = cohort_score(p, split, ramps)
            print(f"    {label:<7} wape={c['pooled_wape']:.4f} bias={c['bias_pct']:+.1f}%")


if __name__ == "__main__":
    main()
