#!/usr/bin/env python3
"""ML experiment 18: v10 stage 1b — wide random hyperparameter search.

The first attempt (ml_17) varied only min_child_samples and num_leaves over a
narrow range and found a flat surface. That was too small a slice to conclude
anything about hyperparameters: every configuration halted at 30-46 trees, so
early-stopping patience, not capacity, was binding, and patience was not in the
search. Learning rate, regularisation, and both sampling fractions were fixed.

This searches eight parameters at once, patience included. Random search rather
than a grid: with eight dimensions a grid spends its budget resolving one axis.

Scoring is unchanged and still touches no test data: the weighted L1 that early
stopping computes on the held-out validation SKUs (Section 2.3), which by
Section 4.6 equals the pooled-WAPE numerator, averaged over the three training
sets. Results append to outputs/reports/tune_wide.csv as they complete.

Usage:
    python scripts/ml_18_tune_wide.py --n 80
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config import OUTPUTS_REPORTS
from src.ml.dataset import dev_splits, load_weekly, stratified_val_skus
from src.ml.model import FEATURES_V1, RatioLGBM

OUT = OUTPUTS_REPORTS / "tune_wide.csv"

CURRENT = dict(learning_rate=0.05, num_leaves=31, min_child_samples=200,
               colsample_bytree=1.0, subsample=1.0, reg_alpha=0.0,
               reg_lambda=0.0, patience=100)


def sample(rng):
    return dict(
        learning_rate=float(np.exp(rng.uniform(np.log(0.005), np.log(0.3)))),
        # 255 leaves was in the first version of this space and killed the
        # run on config 23: LightGBM's histogram allocation is proportional to
        # leaves x features x bins, and the sandbox has under 4 GB. Capped at
        # 127, which completed fine (rand020).
        num_leaves=int(rng.choice([7, 15, 31, 63, 127])),
        min_child_samples=int(rng.choice([5, 10, 20, 50, 100, 200, 500, 1000])),
        colsample_bytree=float(rng.uniform(0.4, 1.0)),
        subsample=float(rng.uniform(0.5, 1.0)),
        reg_alpha=float(rng.choice([0.0, 0.01, 0.1, 1.0, 10.0])),
        reg_lambda=float(rng.choice([0.0, 0.01, 0.1, 1.0, 10.0])),
        patience=int(rng.choice([30, 100, 300, 1000])),
    )


def evaluate(cfg, splits, vals, profiles):
    scores, trees = [], []
    for s, v in zip(splits, vals):
        m = RatioLGBM(s.horizon, FEATURES_V1, deseas_features=True,
                      deseas_all=True, patience=cfg["patience"])
        m.PARAMS = {**RatioLGBM.PARAMS,
                    **{k: v_ for k, v_ in cfg.items() if k != "patience"}}
        # Cap trees for the SEARCH only: a low learning_rate with high patience
        # never early-stops before the deployment cap of 3000, which both runs
        # for minutes and can exhaust memory. 800 is ample for ranking configs;
        # the winner is refit at the real cap in stage 2.
        m.PARAMS["n_estimators"] = 800
        if cfg["subsample"] < 1.0:
            m.PARAMS["subsample_freq"] = 1
        m.fit(s.train, profiles, s.cutoff, v)
        scores.append(float(m.model.best_score_["valid_0"]["l1"]))
        trees.append(m.model.best_iteration_)
    return float(np.mean(scores)), trees


def worker(cfg_json):
    """Evaluate one config in an isolated process and print the result.

    Isolation matters: a configuration can exhaust memory or run for many
    minutes (low learning_rate with high patience never triggers early stopping
    before the n_estimators cap), and in-process that kills the whole search
    with no traceback. As a subprocess it is just one failed row.
    """
    cfg = json.loads(cfg_json)
    weekly, profiles = load_weekly()
    smooth = profiles.loc[profiles["bucket"] == "smooth", "unique_id"]
    weekly = weekly[weekly["unique_id"].isin(set(smooth))]
    splits = list(dev_splits(weekly, n=3))
    vals = [stratified_val_skus(s.train, profiles) for s in splits]
    l1, trees = evaluate(cfg, splits, vals, profiles)
    print("RESULT " + json.dumps({"val_l1": l1, "trees": trees}))


def main(n, seed, timeout):
    weekly, profiles = load_weekly()
    smooth = profiles.loc[profiles["bucket"] == "smooth", "unique_id"]
    weekly = weekly[weekly["unique_id"].isin(set(smooth))]
    splits = list(dev_splits(weekly, n=3))
    vals = [stratified_val_skus(s.train, profiles) for s in splits]

    rng = np.random.default_rng(seed)
    configs = [("current", CURRENT)] + [
        (f"rand{i:03d}", sample(rng)) for i in range(n)]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Resume: keep anything already scored, skip re-running it.
    rows = []
    if OUT.exists():
        rows = pd.read_csv(OUT).to_dict("records")
        done = {r["tag"] for r in rows}
        configs = [(tag, c) for tag, c in configs if tag not in done]
        print(f"resuming: {len(rows)} configs already scored, {len(configs)} to go\n")

    t0 = time.time()
    for i, (tag, cfg) in enumerate(configs):
        try:
            l1, trees = evaluate(cfg, splits, vals, profiles)
        except Exception as e:
            print(f"[{i + 1}/{len(configs)}] {tag:<9} FAILED ({type(e).__name__})",
                  flush=True)
            continue
        rows.append({"tag": tag, **cfg, "val_l1": l1,
                     "trees": "/".join(map(str, trees))})
        pd.DataFrame(rows).sort_values("val_l1").to_csv(OUT, index=False)
        print(f"[{i + 1}/{len(configs)}] {tag:<9} L1={l1:.6f} trees={trees} "
              f"({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows).sort_values("val_l1").reset_index(drop=True)
    cur = df[df.tag == "current"].iloc[0]
    print(f"\ncurrent L1 = {cur.val_l1:.6f}, rank {df.index[df.tag == 'current'][0] + 1} "
          f"of {len(df)}")
    print("\ntop 8:")
    cols = ["tag", "val_l1", "learning_rate", "num_leaves", "min_child_samples",
            "colsample_bytree", "subsample", "reg_alpha", "reg_lambda",
            "patience", "trees"]
    print(df.head(8)[cols].to_string(index=False))
    print(f"\nspread across all configs: {df.val_l1.min():.6f} to {df.val_l1.max():.6f} "
          f"({(df.val_l1.max() / df.val_l1.min() - 1) * 100:.1f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=int, default=150)
    ap.add_argument("--worker", type=str, default=None)
    a = ap.parse_args()
    if a.worker:
        worker(a.worker)
    else:
        main(a.n, a.seed, a.timeout)
