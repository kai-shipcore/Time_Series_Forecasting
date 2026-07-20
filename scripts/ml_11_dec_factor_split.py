#!/usr/bin/env python3
"""ML experiment 11: v5 stage 1 — per-segment December/holiday multiplier.

The `seasonal_fit` diagnostic (src/ml/diagnostics.py) measured post-correction
December residuals of 0.733 (long) versus 1.494 (short): the shared 1.26
holiday factor over-corrects mature SKUs and under-corrects young ones at the
same time. This tests splitting that one factor by segment, at BASELINE level
(deseasonalized WA12 for both segments), so no model confounds attribution.

Pre-registered design and pass criteria: design doc Section 6, v5 stage 1.
  - Empirical factor estimated per window from training data only (leak-free),
    as 1.26 x demand-weighted holiday residual, shrunk w=0.5 toward 1.26.
  - Long: must improve Dec-Feb AND Oct-Dec, no significant Mar-May regression.
  - Short: must improve Dec-Feb vs shared-factor deseas WA12, no significant
    Mar-May regression.

Internal cross-check: the shared-factor variant must reproduce the known
prototype-short figures (~0.2014 Mar-May, ~0.2863 Dec-Feb) and v-base-long
figures (0.1321 / 0.2764 / 0.1209).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config import HOLIDAY_MULTIPLIER
from src.deseasonalize import _factors, _is_holiday
from src.ml.dataset import dev_splits, load_weekly
from src.ml.evaluate import bootstrap_delta, is_significant, score, score_table
from src.ml.model import long_sku_set

SHRINK_W = 0.5   # pre-registered; not searched


def seg_factors(ds: pd.Series, f_holiday: float) -> pd.Series:
    """Shared monthly factors, with the holiday window at `f_holiday`."""
    base = _factors(ds)
    return base.where(~_is_holiday(ds), f_holiday)


def estimate_holiday_factor(train: pd.DataFrame, uids: set) -> float:
    """Leak-free empirical holiday factor for one segment, shrunk toward 1.26.

    On the segment's training history, deseasonalized with the CURRENT shared
    factors: ratio of each week's adjusted demand to the SKU's trailing
    12-week adjusted level (prior weeks only). The demand-weighted mean ratio
    over holiday-window weeks, divided by the all-weeks mean (cancelling
    growth), says how far 1.26 is off; shrinkage halves the step.
    """
    df = train[train["unique_id"].isin(uids)].sort_values(["unique_id", "ds"]).copy()
    df["y_adj"] = df["y"] / _factors(df["ds"]).to_numpy()
    g = df.groupby("unique_id")["y_adj"]
    level = g.rolling(12, min_periods=4).mean().reset_index(level=0, drop=True)
    df["level"] = level.groupby(df["unique_id"]).shift(1)   # prior weeks only
    df = df.dropna(subset=["level"])
    df = df[df["level"] > 0]
    df["r"] = df["y_adj"] / df["level"]

    hol = df[_is_holiday(df["ds"])]
    if hol.empty:
        return HOLIDAY_MULTIPLIER
    r_hol = np.average(hol["r"], weights=hol["level"])
    r_all = np.average(df["r"], weights=df["level"])
    implied = HOLIDAY_MULTIPLIER * (r_hol / r_all)
    return HOLIDAY_MULTIPLIER + SHRINK_W * (implied - HOLIDAY_MULTIPLIER)


def wa12_forecast(train, test, uids_factor: dict) -> pd.DataFrame:
    """Deseasonalized WA12 with a per-SKU holiday factor from `uids_factor`."""
    rows = []
    for uid, f_hol in uids_factor.items():
        hist = train.loc[train["unique_id"] == uid].sort_values("ds")
        if hist.empty:
            continue
        adj = hist["y"] / seg_factors(hist["ds"], f_hol).to_numpy()
        level = adj.tail(12).mean()
        tw = test.loc[test["unique_id"] == uid, "ds"].drop_duplicates()
        f = seg_factors(tw, f_hol).to_numpy()
        rows.append(pd.DataFrame({"unique_id": uid, "ds": tw, "yhat": level * f}))
    return pd.concat(rows, ignore_index=True)


def main():
    weekly, profiles = load_weekly()
    smooth = set(profiles.loc[profiles["bucket"] == "smooth", "unique_id"])
    weekly = weekly[weekly["unique_id"].isin(smooth)]

    for split, name in zip(dev_splits(weekly, n=3), ["Mar-May", "Dec-Feb", "Oct-Dec"]):
        print(f"\n{'=' * 66}\n{name}  {split}")
        uids = set(split.train["unique_id"].unique())
        longs = long_sku_set(profiles, split.cutoff) & uids
        shorts = uids - longs

        f_long = estimate_holiday_factor(split.train, longs)
        f_short = estimate_holiday_factor(split.train, shorts)
        print(f"  estimated holiday factor (w={SHRINK_W}): "
              f"long {f_long:.3f}, short {f_short:.3f}  (shared: {HOLIDAY_MULTIPLIER})")

        shared = wa12_forecast(split.train, split.test,
                               {u: HOLIDAY_MULTIPLIER for u in uids})
        split_f = wa12_forecast(split.train, split.test,
                                {**{u: f_long for u in longs},
                                 **{u: f_short for u in shorts}})

        results = {
            "shared_1.26": score(shared, split, profiles),
            "split_factor": score(split_f, split, profiles),
        }
        print("\npooled WAPE:")
        print(score_table(results).to_string())
        print("\nbias% by segment:")
        print(pd.DataFrame(
            {n_: t.set_index("segment")["bias_pct"] for n_, t in results.items()}
        ).to_string())

        for seg in ("short", "long"):
            bd = bootstrap_delta(split_f, shared, split, profiles, segment=seg)
            print(f"bootstrap split-vs-shared [{seg}]: {bd}  "
                  f"significant={is_significant(bd)}")


if __name__ == "__main__":
    main()
