#!/usr/bin/env python3
"""ML experiment 06: v1 (lead + ramp block) versus v0 (lead) versus baseline.

Ramp block hypothesis: a SKU running above its own 12-week level keeps
running above it in the near future, so the growth correction should attach
to SKUs that are actually growing rather than to everyone (the v0 failure).

Pre-registered pass criterion (design Section 6): hold the baseline in the
post-holiday window (Dev 2) while keeping v0's spring (Dev 1) gain.
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
from src.ml.model import FEATURES_V0, FEATURES_V1, RatioLGBM, structural_baseline


def main():
    weekly, profiles = load_weekly()
    smooth = profiles.loc[profiles["bucket"] == "smooth", "unique_id"]
    weekly = weekly[weekly["unique_id"].isin(smooth)]

    for split in dev_splits(weekly, n=3):
        print(f"\n{'=' * 62}\n{split}")
        val_uids = stratified_val_skus(split.train, profiles)

        base = structural_baseline(split.train, split.test, profiles, split.cutoff)
        # deseas_features=False pins this experiment to v1's original
        # feature computation (segment-native series), so it reproduces the
        # Section 4.19 result even now that v2 exists.
        m0 = RatioLGBM(split.horizon, FEATURES_V0).fit(
            split.train, profiles, split.cutoff, val_uids)
        m1 = RatioLGBM(split.horizon, FEATURES_V1).fit(
            split.train, profiles, split.cutoff, val_uids)
        p0 = m0.predict(split.train, profiles, split.cutoff)
        p1 = m1.predict(split.train, profiles, split.cutoff)

        results = {
            "baseline": score(base, split, profiles),
            "v0": score(p0, split, profiles),
            "v1": score(p1, split, profiles),
        }
        print(f"\npooled WAPE (v0 iters={m0.model.best_iteration_}, "
              f"v1 iters={m1.model.best_iteration_}):")
        print(score_table(results).to_string())
        print("\nbias% by segment:")
        bias = {n: t.set_index("segment")["bias_pct"] for n, t in results.items()}
        print(pd.DataFrame(bias).to_string())

        for seg in ("short", "long"):
            b1 = bootstrap_delta(p1, base, split, profiles, segment=seg)
            b0 = bootstrap_delta(p1, p0, split, profiles, segment=seg)
            print(f"bootstrap v1-vs-baseline [{seg}]: {b1}")
            print(f"bootstrap v1-vs-v0       [{seg}]: {b0}")

        ramps = ramp_cohort(split, profiles)
        print(f"ramp cohort (n={len(ramps)}):")
        for label, p in [("baseline", base), ("v0", p0), ("v1", p1)]:
            c = cohort_score(p, split, ramps)
            print(f"    {label:<9} wape={c['pooled_wape']:.4f} bias={c['bias_pct']:+.1f}%")

        print("v1 feature importance (gain):")
        print(m1.importance().to_string(index=False))

        # Learned response probe: predicted ratio across ramp states at three
        # leads. y_last_r and lag_1_r are set equal to the ramp value (a
        # coherent "steadily ramping" scenario).
        rows = []
        for ramp in (0.6, 0.8, 1.0, 1.2, 1.5, 2.0):
            for lead in (1, 5, 10):
                rows.append({"lead": lead, "ramp_4_12": ramp,
                             "y_last_r": ramp, "lag_1_r": ramp})
        probe = pd.DataFrame(rows)
        probe["pred_ratio"] = m1.model.predict(probe[FEATURES_V1]).round(3)
        print("learned response (rows=ramp state, cols=lead):")
        print(probe.pivot(index="ramp_4_12", columns="lead",
                          values="pred_ratio").to_string())


if __name__ == "__main__":
    main()
