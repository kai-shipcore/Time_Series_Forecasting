#!/usr/bin/env python3
"""ML experiment 09: v4 versus v3 versus baseline.

v4 = v3 with one feature added, `is_long`: a binary segment indicator taken
as of the forecast cutoff (1 = medium/full history, 0 = short). Everything
else is v3 unchanged, deseas_all=True included.

Hypothesis (design Section 4.20): the Dec-Feb long segment has oscillated
across v1/v2/v3 while long SKUs' own features never changed, implicating the
shared global trees. Without a segment feature the trees cannot separate the
two populations, so splits driven by short SKUs also apply to long ones. An
explicit indicator lets the model condition on segment.

Pre-registered pass criteria (design Section 6):
  1. Dec-Feb long holds the baseline (no significant regression vs v-base).
     This is the one criterion v3 failed.
  2. The Mar-May short win survives (still significantly better than v-base).
  3. No new significant regression vs v-base in any other decision cell.

Disconfirming evidence, also pre-registered: if `is_long` carries near-zero
gain, the hypothesis is wrong regardless of the WAPE movement.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ml.dataset import dev_splits, load_weekly, stratified_val_skus
from src.ml.evaluate import (
    bootstrap_delta, cohort_score, is_significant, ramp_cohort, score, score_table,
)
from src.ml.model import FEATURES_V1, FEATURES_V4, RatioLGBM, structural_baseline


def main():
    weekly, profiles = load_weekly()
    smooth = profiles.loc[profiles["bucket"] == "smooth", "unique_id"]
    weekly = weekly[weekly["unique_id"].isin(smooth)]

    for split in dev_splits(weekly, n=3):
        print(f"\n{'=' * 62}\n{split}")
        val_uids = stratified_val_skus(split.train, profiles)

        base = structural_baseline(split.train, split.test, profiles, split.cutoff)
        # Both flags stated explicitly on both models so the comparison cannot
        # drift if a constructor default changes later. v4 differs from v3 in
        # exactly one way: the feature list.
        m3 = RatioLGBM(split.horizon, FEATURES_V1,
                       deseas_features=True, deseas_all=True).fit(
            split.train, profiles, split.cutoff, val_uids)
        m4 = RatioLGBM(split.horizon, FEATURES_V4,
                       deseas_features=True, deseas_all=True).fit(
            split.train, profiles, split.cutoff, val_uids)
        p3 = m3.predict(split.train, profiles, split.cutoff)
        p4 = m4.predict(split.train, profiles, split.cutoff)

        results = {
            "baseline": score(base, split, profiles),
            "v3": score(p3, split, profiles),
            "v4": score(p4, split, profiles),
        }
        print(f"\npooled WAPE (v3 iters={m3.model.best_iteration_}, "
              f"v4 iters={m4.model.best_iteration_}):")
        print(score_table(results).to_string())
        print("\nbias% by segment:")
        print(pd.DataFrame(
            {n: t.set_index("segment")["bias_pct"] for n, t in results.items()}
        ).to_string())

        for seg in ("short", "long"):
            for label, a, b in [("v4-vs-baseline", p4, base), ("v4-vs-v3", p4, p3)]:
                bd = bootstrap_delta(a, b, split, profiles, segment=seg)
                print(f"bootstrap {label:<15}[{seg}]: {bd}  "
                      f"significant={is_significant(bd)}")

        ramps = ramp_cohort(split, profiles)
        print(f"ramp cohort (n={len(ramps)}):")
        for label, p in [("baseline", base), ("v3", p3), ("v4", p4)]:
            c = cohort_score(p, split, ramps)
            print(f"    {label:<9} wape={c['pooled_wape']:.4f} bias={c['bias_pct']:+.1f}%")

        # Did the model actually use the indicator? Pre-registered as the
        # disconfirming check: near-zero gain means the hypothesis is wrong.
        imp = m4.importance()
        print("v4 feature importance (gain):")
        print(imp.to_string(index=False))
        tot = imp["gain"].sum()
        share = imp.loc[imp.feature == "is_long", "gain"].iloc[0] / max(tot, 1e-9)
        print(f"  is_long share of total gain: {share:.2%}")


if __name__ == "__main__":
    main()
