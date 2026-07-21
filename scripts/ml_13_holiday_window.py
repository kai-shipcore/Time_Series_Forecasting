#!/usr/bin/env python3
"""ML experiment 13: v6 stage 1 — holiday window ends mid-December.

Compares the current ML holiday window (Nov 20 - Dec 31) against one ending
mid-December (Nov 20 - Dec 15), at BASELINE level: deseasonalized WA12 under
each window, no model, so attribution is unambiguous.

Basis and pre-registered pass criteria: design doc Section 6, v6 stage 1. The
change rests on business knowledge (promotions run late Nov to mid Dec; Dec 2024
predates that practice), not on fitting the two observed Decembers, which belong
to different regimes.

Only ML_* settings are touched, so the statistical prototype and V1 are
unaffected by construction (src/ml/seasonal.py).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

import src.ml.seasonal as seas
from src.ml.dataset import dev_splits, load_weekly
from src.ml.evaluate import bootstrap_delta, is_significant, score, score_table

CURRENT = (12, 31)
CANDIDATE = (12, 15)


def wa12(train, test, end_md) -> pd.DataFrame:
    """Deseasonalized WA12 with the holiday window ending at `end_md`."""
    saved = seas.ML_HOLIDAY_END
    seas.ML_HOLIDAY_END = end_md
    try:
        adj = train.copy()
        adj["y_adj"] = adj["y"] / seas.ml_factors(adj["ds"]).to_numpy()
        lvl = (adj.sort_values("ds").groupby("unique_id")["y_adj"]
               .apply(lambda s: s.tail(12).mean()))
        grid = test[["unique_id", "ds"]].drop_duplicates().copy()
        grid["yhat"] = (grid["unique_id"].map(lvl).fillna(0.0).to_numpy()
                        * seas.ml_factors(grid["ds"]).to_numpy())
        return grid
    finally:
        seas.ML_HOLIDAY_END = saved


def main():
    weekly, profiles = load_weekly()
    smooth = profiles.loc[profiles["bucket"] == "smooth", "unique_id"]
    weekly = weekly[weekly["unique_id"].isin(set(smooth))]

    for split, name in zip(dev_splits(weekly, n=3), ["Mar-May", "Dec-Feb", "Oct-Dec"]):
        print(f"\n{'=' * 66}\n{name}  {split}")
        cur = wa12(split.train, split.test, CURRENT)
        cand = wa12(split.train, split.test, CANDIDATE)

        results = {"end_12-31": score(cur, split, profiles),
                   "end_12-15": score(cand, split, profiles)}
        print("\npooled WAPE:")
        print(score_table(results).to_string())
        print("\nbias% by segment:")
        print(pd.DataFrame(
            {n: t.set_index("segment")["bias_pct"] for n, t in results.items()}
        ).to_string())
        for seg in ("short", "long"):
            bd = bootstrap_delta(cand, cur, split, profiles, segment=seg)
            print(f"bootstrap candidate-vs-current [{seg}]: {bd}  "
                  f"significant={is_significant(bd)}")


if __name__ == "__main__":
    main()
