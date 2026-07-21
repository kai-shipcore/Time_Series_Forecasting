#!/usr/bin/env python3
"""ML experiment 17: v10 — hyperparameter search on the validation slice only.

Scores candidates by the weighted L1 that early stopping already computes on the
held-out validation SKUs (Section 2.3), which by Section 4.6 equals the
pooled-WAPE numerator. No test-window data is touched during the search, so the
three development windows are not spent.

Grid fixed in advance: min_child_samples x num_leaves, 12 configurations.
Protocol and pass criteria: design doc Section 6, v10.

Usage:
    python scripts/ml_17_tune.py            # stage 1, the grid
    python scripts/ml_17_tune.py --verify   # stage 2, winner vs v9 on dev windows
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ml.dataset import dev_splits, load_weekly, stratified_val_skus
from src.ml.evaluate import bootstrap_delta, is_significant, score, score_table
from src.ml.model import FEATURES_V1, RatioLGBM, structural_baseline

MIN_CHILD = [20, 50, 100, 200]
LEAVES = [15, 31, 63]
CURRENT = (200, 31)


def load():
    weekly, profiles = load_weekly()
    smooth = profiles.loc[profiles["bucket"] == "smooth", "unique_id"]
    return weekly[weekly["unique_id"].isin(set(smooth))], profiles


def fit(split, profiles, val, **params):
    m = RatioLGBM(split.horizon, FEATURES_V1, deseas_features=True, deseas_all=True)
    m.PARAMS = {**RatioLGBM.PARAMS, **params}
    return m.fit(split.train, profiles, split.cutoff, val)


def search():
    weekly, profiles = load()
    splits = list(dev_splits(weekly, n=3))
    vals = [stratified_val_skus(s.train, profiles) for s in splits]

    rows = []
    for mcs in MIN_CHILD:
        for lv in LEAVES:
            scores, trees = [], []
            for s, v in zip(splits, vals):
                m = fit(s, profiles, v, min_child_samples=mcs, num_leaves=lv)
                scores.append(float(m.model.best_score_["valid_0"]["l1"]))
                trees.append(m.model.best_iteration_)
            rows.append({"min_child_samples": mcs, "num_leaves": lv,
                         "val_l1_mean": sum(scores) / len(scores),
                         "trees": "/".join(str(t) for t in trees),
                         "current": (mcs, lv) == CURRENT})
    df = pd.DataFrame(rows).sort_values("val_l1_mean").reset_index(drop=True)
    print("Validation-slice weighted L1 (lower is better). No test data used.\n")
    print(df.to_string(index=False))
    best = df.iloc[0]
    cur = df[df["current"]].iloc[0]
    print(f"\n  current  : min_child_samples={CURRENT[0]}, num_leaves={CURRENT[1]}, "
          f"L1={cur.val_l1_mean:.5f}")
    print(f"  winner   : min_child_samples={best.min_child_samples}, "
          f"num_leaves={best.num_leaves}, L1={best.val_l1_mean:.5f}")
    print(f"  reduction: {(1 - best.val_l1_mean / cur.val_l1_mean) * 100:.2f}%")
    print(f"\n  Verify with: python scripts/ml_17_tune.py --verify "
          f"--mcs {best.min_child_samples} --leaves {best.num_leaves}")


def verify(mcs, leaves):
    weekly, profiles = load()
    print(f"Stage 2: min_child_samples={mcs}, num_leaves={leaves} versus v9.\n")
    for split, name in zip(dev_splits(weekly, n=3), ["Mar-May", "Dec-Feb", "Oct-Dec"]):
        val = stratified_val_skus(split.train, profiles)
        p9 = fit(split, profiles, val).predict(split.train, profiles, split.cutoff)
        p10 = fit(split, profiles, val, min_child_samples=mcs,
                  num_leaves=leaves).predict(split.train, profiles, split.cutoff)
        results = {
            "baseline": score(structural_baseline(
                split.train, split.test, profiles, split.cutoff), split, profiles),
            "v9": score(p9, split, profiles),
            "v10": score(p10, split, profiles),
        }
        print(f"\n{'=' * 60}\n{name}")
        print(score_table(results).to_string())
        for seg in ("short", "long"):
            bd = bootstrap_delta(p10, p9, split, profiles, segment=seg)
            print(f"  v10-vs-v9 [{seg}]: {bd}  significant={is_significant(bd)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--mcs", type=int, default=CURRENT[0])
    ap.add_argument("--leaves", type=int, default=CURRENT[1])
    a = ap.parse_args()
    verify(a.mcs, a.leaves) if a.verify else search()
