#!/usr/bin/env python3
"""ML experiment 08: v3 versus v2 versus baseline.

v3 = v2 with the ENTIRE short-SKU path made seasonally consistent: levels,
training targets, and output scaling all use the factor-adjusted series for
every SKU (deseas_all=True). Hypothesis: full deseasonalization becomes
viable for short SKUs once the learned growth response can offset the
January over-cut that sank it at baseline level (Section 4.17 context).
The structural baseline is untouched.

Pre-registered pass criterion (design Section 6): hold the baseline on
Dec-Feb short while keeping the Mar-May short win, with no long-segment
regression.

Watch item: residual UNDER-forecast on Dec-Feb short would signal that the
mature-SKU factors over-swing for young SKUs (muted seasonality), pointing
at short-specific damped factors inside the ML path as the v4 candidate.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ml.dataset import dev_splits, load_weekly, stratified_val_skus
from src.ml.evaluate import (
    bootstrap_delta, cohort_score, ramp_cohort, score, score_table,
)
from src.ml.model import FEATURES_V1, RatioLGBM, structural_baseline


def main():
    weekly, profiles = load_weekly()
    smooth = profiles.loc[profiles["bucket"] == "smooth", "unique_id"]
    weekly = weekly[weekly["unique_id"].isin(smooth)]

    for split in dev_splits(weekly, n=3):
        print(f"\n{'=' * 62}\n{split}")
        val_uids = stratified_val_skus(split.train, profiles)

        base = structural_baseline(split.train, split.test, profiles, split.cutoff)
        # Both flags are stated explicitly on both models so this comparison
        # cannot drift if a constructor default changes later.
        m2 = RatioLGBM(split.horizon, FEATURES_V1,
                       deseas_features=True, deseas_all=False).fit(
            split.train, profiles, split.cutoff, val_uids)
        m3 = RatioLGBM(split.horizon, FEATURES_V1,
                       deseas_features=True, deseas_all=True).fit(
            split.train, profiles, split.cutoff, val_uids)
        p2 = m2.predict(split.train, profiles, split.cutoff)
        p3 = m3.predict(split.train, profiles, split.cutoff)

        results = {
            "baseline": score(base, split, profiles),
            "v2": score(p2, split, profiles),
            "v3": score(p3, split, profiles),
        }
        print(f"\npooled WAPE (v2 iters={m2.model.best_iteration_}, "
              f"v3 iters={m3.model.best_iteration_}):")
        print(score_table(results).to_string())
        print("\nbias% by segment:")
        bias = {n: t.set_index("segment")["bias_pct"] for n, t in results.items()}
        print(pd.DataFrame(bias).to_string())

        for seg in ("short", "long"):
            bb = bootstrap_delta(p3, base, split, profiles, segment=seg)
            b2 = bootstrap_delta(p3, p2, split, profiles, segment=seg)
            print(f"bootstrap v3-vs-baseline [{seg}]: {bb}")
            print(f"bootstrap v3-vs-v2       [{seg}]: {b2}")

        ramps = ramp_cohort(split, profiles)
        print(f"ramp cohort (n={len(ramps)}):")
        for label, p in [("baseline", base), ("v2", p2), ("v3", p3)]:
            c = cohort_score(p, split, ramps)
            print(f"    {label:<9} wape={c['pooled_wape']:.4f} bias={c['bias_pct']:+.1f}%")


if __name__ == "__main__":
    main()
