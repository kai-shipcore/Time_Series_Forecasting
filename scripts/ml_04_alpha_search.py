#!/usr/bin/env python3
"""ML experiment 04: grid search for the age-damping exponent alpha.

Short-as-of SKUs get seasonal factor^alpha; long SKUs always get the full
factor (alpha=1). alpha=0 reproduces raw WA12 for short SKUs; alpha=1
reproduces full deseasonalization. Scored on the SHORT segment only, over
Dev 1 and Dev 2 (Dev 3 is excluded from short-segment decisions, design
Section 4.16; its numbers are printed for reference).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.deseasonalize import _factors
from src.ml.dataset import asof_history_length, dev_splits, load_weekly
from src.ml.evaluate import bootstrap_delta, score


def wa12_damped(train, test, profiles, cutoff, alpha):
    """WA12 with age-damped deseasonalization: short-as-of SKUs use
    factor^alpha, long SKUs use factor^1."""
    seg = asof_history_length(profiles, cutoff).astype("object")
    is_short = seg == "short"

    tr = train.sort_values(["unique_id", "ds"]).copy()
    f = _factors(tr["ds"]).to_numpy()
    a = np.where(tr["unique_id"].map(is_short).fillna(True), alpha, 1.0)
    tr["y_flat"] = tr["y"] / (f ** a)
    lvl = tr.groupby("unique_id")["y_flat"].apply(lambda s: s.tail(12).mean())

    grid = test[["unique_id", "ds"]].drop_duplicates().copy()
    fg = _factors(grid["ds"]).to_numpy()
    ag = np.where(grid["unique_id"].map(is_short).fillna(True), alpha, 1.0)
    grid["yhat"] = grid["unique_id"].map(lvl).fillna(0.0) * (fg ** ag)
    return grid


def main():
    weekly, profiles = load_weekly()
    smooth = profiles.loc[profiles["bucket"] == "smooth", "unique_id"]
    weekly = weekly[weekly["unique_id"].isin(smooth)]

    splits = dev_splits(weekly, n=3)
    alphas = [round(a, 1) for a in np.arange(0, 1.01, 0.1)]

    rows = []
    preds_cache = {}
    for a in alphas:
        row = {"alpha": a}
        for name, s in zip(["dev1", "dev2", "dev3"], splits):
            p = wa12_damped(s.train, s.test, profiles, s.cutoff, a)
            preds_cache[(a, name)] = p
            tbl = score(p, s, profiles).set_index("segment")
            row[f"{name}_wape"] = tbl.loc["smooth/short", "pooled_wape"]
            row[f"{name}_bias"] = tbl.loc["smooth/short", "bias_pct"]
        row["mean_dev12"] = round((row["dev1_wape"] + row["dev2_wape"]) / 2, 4)
        rows.append(row)

    out = pd.DataFrame(rows)
    print("SHORT segment, factor^alpha for short-as-of SKUs")
    print("(decision windows: dev1+dev2; dev3 reference only)\n")
    print(out.to_string(index=False))

    best = out.loc[out["mean_dev12"].idxmin(), "alpha"]
    print(f"\nbest alpha by mean(dev1, dev2): {best}")
    for name, s in zip(["dev1", "dev2"], splits[:2]):
        bd = bootstrap_delta(
            preds_cache[(best, name)], preds_cache[(0.0, name)], s, profiles,
            segment="short",
        )
        print(f"  bootstrap alpha={best} vs alpha=0 [{name}]: {bd}")


if __name__ == "__main__":
    main()
