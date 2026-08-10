#!/usr/bin/env python3
"""ML experiment 15: v8 — separate models per segment.

Two independent RatioLGBM fits, one on long SKUs and one on short, each on the
v3 configuration. Predictions concatenated and scored as one frame.

Last candidate from design Section 4.20, and the only one that structurally
cannot trade one segment against the other. Pre-registered criteria and known
limitations: design doc Section 6, v8.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ml.dataset import dev_splits, load_weekly, stratified_val_skus
from src.ml.evaluate import bootstrap_delta, is_significant, score, score_table
from src.ml.model import FEATURES_V1, RatioLGBM, long_sku_set, structural_baseline

from src.ml.reference import PROTOTYPE, warn_if_stale  # noqa: E402

warn_if_stale()


def main():
    weekly, profiles = load_weekly()
    smooth = profiles.loc[profiles["bucket"] == "smooth", "unique_id"]
    weekly = weekly[weekly["unique_id"].isin(set(smooth))]

    for split, name in zip(dev_splits(weekly, n=3), PROTOTYPE):
        print(f"\n{'=' * 66}\n{name}  {split}")
        val = stratified_val_skus(split.train, profiles)
        uids = set(split.train["unique_id"].unique())
        longs = long_sku_set(profiles, split.cutoff) & uids
        shorts = uids - longs

        def fit(restrict):
            return RatioLGBM(split.horizon, FEATURES_V1, deseas_features=True,
                             deseas_all=True, uids=restrict).fit(
                split.train, profiles, split.cutoff, val)

        m3 = RatioLGBM(split.horizon, FEATURES_V1, deseas_features=True,
                       deseas_all=True).fit(split.train, profiles, split.cutoff, val)
        mL, mS = fit(longs), fit(shorts)
        p3 = m3.predict(split.train, profiles, split.cutoff)
        p8 = pd.concat([mL.predict(split.train, profiles, split.cutoff),
                        mS.predict(split.train, profiles, split.cutoff)],
                       ignore_index=True)

        results = {
            "baseline": score(structural_baseline(
                split.train, split.test, profiles, split.cutoff), split, profiles),
            "v3": score(p3, split, profiles),
            "v8": score(p8, split, profiles),
        }
        print(f"\n  long model : {mL.n_train_rows:,} rows, {len(val & longs)} val SKUs, "
              f"{mL.model.best_iteration_} trees")
        print(f"  short model: {mS.n_train_rows:,} rows, {len(val & shorts)} val SKUs, "
              f"{mS.model.best_iteration_} trees")
        print(f"  prototype (the bar): short {PROTOTYPE[name][0]:.4f}, "
              f"long {PROTOTYPE[name][1]:.4f}")
        print("\npooled WAPE:")
        print(score_table(results).to_string())
        print("\nbias% by segment:")
        print(pd.DataFrame(
            {n: t.set_index("segment")["bias_pct"] for n, t in results.items()}
        ).to_string())
        for seg in ("short", "long"):
            bd = bootstrap_delta(p8, p3, split, profiles, segment=seg)
            print(f"bootstrap v8-vs-v3 [{seg}]: {bd}  significant={is_significant(bd)}")


if __name__ == "__main__":
    main()
