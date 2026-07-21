#!/usr/bin/env python3
"""ML experiment 14: v7 — per-segment sample weighting.

v7 = v4 (segment indicator) with training weights partially rebalanced so the
two segments contribute more equally: each segment's weight is multiplied by
(0.5 / its current share) ** BALANCE, BALANCE fixed at 0.5 in advance.

Hypothesis (design Sections 4.23, 6): v4 showed the model CAN separate segments
once given is_long, and then specialises on the segment carrying 72-98% of the
training weight, wrecking the other. Rebalancing removes the incentive while
keeping the ability. Weighting without the indicator (v7-ref below) cannot test
this, since the model then has no way to treat the segments differently.

Pre-registered pass criteria: design doc Section 6, v7.
  1. Dec-Feb long improves significantly against v3 (0.3145).
  2. Short holds v3's qualification: <= prototype on all three windows
     (0.2014 / 0.2863 / 0.4251).
  3. No significant regression against v3 in any other decision cell.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ml.dataset import dev_splits, load_weekly, stratified_val_skus
from src.ml.evaluate import bootstrap_delta, is_significant, score, score_table
from src.ml.model import FEATURES_V1, FEATURES_V4, RatioLGBM, structural_baseline

BALANCE = 0.5
PROTOTYPE = {"Mar-May": (0.2014, 0.1411), "Dec-Feb": (0.2863, 0.2737),
             "Oct-Dec": (0.4251, 0.0911)}


def main():
    weekly, profiles = load_weekly()
    smooth = profiles.loc[profiles["bucket"] == "smooth", "unique_id"]
    weekly = weekly[weekly["unique_id"].isin(set(smooth))]

    for split, name in zip(dev_splits(weekly, n=3), PROTOTYPE):
        print(f"\n{'=' * 66}\n{name}  {split}")
        val = stratified_val_skus(split.train, profiles)

        def fit(features, balance):
            return RatioLGBM(split.horizon, features, deseas_features=True,
                             deseas_all=True, balance=balance).fit(
                split.train, profiles, split.cutoff, val)

        m3 = fit(FEATURES_V1, 0.0)
        m4 = fit(FEATURES_V4, 0.0)
        m7 = fit(FEATURES_V4, BALANCE)
        mref = fit(FEATURES_V1, BALANCE)
        preds = {"v3": m3.predict(split.train, profiles, split.cutoff),
                 "v4": m4.predict(split.train, profiles, split.cutoff),
                 "v7": m7.predict(split.train, profiles, split.cutoff),
                 "v7ref": mref.predict(split.train, profiles, split.cutoff)}

        results = {"baseline": score(structural_baseline(
                       split.train, split.test, profiles, split.cutoff),
                       split, profiles)}
        results.update({k: score(v, split, profiles) for k, v in preds.items()})

        print(f"\n  weight scaling: long x{m7.seg_scale[0]}, short x{m7.seg_scale[1]}"
              f"   (v3 iters={m3.model.best_iteration_}, v7 iters={m7.model.best_iteration_})")
        print(f"  prototype (the bar): short {PROTOTYPE[name][0]:.4f}, "
              f"long {PROTOTYPE[name][1]:.4f}")
        print("\npooled WAPE:")
        print(score_table(results).to_string())
        print("\nbias% by segment:")
        print(pd.DataFrame(
            {n: t.set_index("segment")["bias_pct"] for n, t in results.items()}
        ).to_string())

        for seg in ("short", "long"):
            bd = bootstrap_delta(preds["v7"], preds["v3"], split, profiles, segment=seg)
            print(f"bootstrap v7-vs-v3 [{seg}]: {bd}  significant={is_significant(bd)}")


if __name__ == "__main__":
    main()
