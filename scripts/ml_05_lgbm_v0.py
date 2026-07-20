#!/usr/bin/env python3
"""ML experiment 05: LightGBM v0 (lead-only) versus the structural baseline.

The model's only feature is the forecast lead, so the only thing it can
learn is the average ratio per horizon: a global growth-drift calibration.
Question under test: does even that minimal learning beat the fixed
ratio-1.0 baseline of Section 3?

Decision windows: all three for the long segment, Dev 1 + Dev 2 for short
(design Section 4.16). Bootstrap on every segment-window pair.
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
from src.ml.model import FEATURES_V0, RatioLGBM, structural_baseline


def main():
    weekly, profiles = load_weekly()
    smooth = profiles.loc[profiles["bucket"] == "smooth", "unique_id"]
    weekly = weekly[weekly["unique_id"].isin(smooth)]

    for split in dev_splits(weekly, n=3):
        print(f"\n{'=' * 62}\n{split}")

        base = structural_baseline(split.train, split.test, profiles, split.cutoff)

        val_uids = stratified_val_skus(split.train, profiles)
        # FEATURES_V0 is pinned explicitly: v0 is the lead-only model, and
        # relying on the constructor default silently turned this into a v1
        # run once the ramp block landed.
        model = RatioLGBM(split.horizon, FEATURES_V0).fit(
            split.train, profiles, split.cutoff, val_uids
        )
        lgbm = model.predict(split.train, profiles, split.cutoff)

        results = {
            "baseline": score(base, split, profiles),
            "LGBM_v0": score(lgbm, split, profiles),
        }
        print(f"\npooled WAPE by segment (best_iter={model.model.best_iteration_}, "
              f"clip_hi={model.clip_hi:.2f}):")
        print(score_table(results).to_string())
        print("\nbias% by segment:")
        bias = {n: t.set_index("segment")["bias_pct"] for n, t in results.items()}
        print(pd.DataFrame(bias).to_string())

        for seg in ("short", "long"):
            bd = bootstrap_delta(lgbm, base, split, profiles, segment=seg)
            print(f"bootstrap LGBM-vs-baseline [{seg}]: {bd}")

        ramps = ramp_cohort(split, profiles)
        print(f"ramp cohort (n={len(ramps)}):")
        for label, p in [("baseline", base), ("LGBM_v0", lgbm)]:
            c = cohort_score(p, split, ramps)
            print(f"    {label:<9} wape={c['pooled_wape']:.4f} bias={c['bias_pct']:+.1f}%")

        # What did the model actually learn? Ratio per lead (its whole brain).
        probe = pd.DataFrame({"lead": range(1, split.horizon + 1)})
        probe["learned_ratio"] = model.model.predict(probe[["lead"]]).round(4)
        print("learned ratio by lead:", probe.set_index("lead")["learned_ratio"].to_dict())


if __name__ == "__main__":
    main()
