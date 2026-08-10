#!/usr/bin/env python3
"""ML experiment 23: v12 — SKU age in the shared (short) model.

v12 = v11 hybrid with sku_age added to the shared model's feature set. The
dedicated long model is unchanged, so long predictions are identical and only
short can move. Pre-registered design and criteria: design doc Section 6, v12.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ml.dataset import dev_splits, load_weekly, stratified_val_skus
from src.ml.evaluate import bootstrap_delta, is_significant, score, score_table
from src.ml.model import (FEATURES_V1, FEATURES_SHORT_AGE, FEATURES_V11_LONG,
                          RatioLGBM, long_sku_set, structural_baseline)

from src.ml.reference import PROTOTYPE, warn_if_stale  # noqa: E402

warn_if_stale()


def main():
    weekly, profiles = load_weekly()
    smooth = profiles.loc[profiles["bucket"] == "smooth", "unique_id"]
    weekly = weekly[weekly["unique_id"].isin(set(smooth))]

    for split, name in zip(dev_splits(weekly, n=3), PROTOTYPE):
        print(f"\n{'=' * 66}\n{name}")
        longs = long_sku_set(profiles, split.cutoff) & set(split.train["unique_id"])
        val_all = stratified_val_skus(split.train, profiles)
        val_long = stratified_val_skus(
            split.train[split.train["unique_id"].isin(longs)], profiles)

        # dedicated long model (v11), used by both v11 and v12
        mL = RatioLGBM(split.horizon, FEATURES_V11_LONG, deseas_features=True,
                       deseas_all=True, uids=longs).fit(
            split.train, profiles, split.cutoff, val_long)
        long_pred = mL.predict(split.train, profiles, split.cutoff)

        def hybrid(short_feats):
            m = RatioLGBM(split.horizon, short_feats, deseas_features=True,
                          deseas_all=True).fit(split.train, profiles, split.cutoff, val_all)
            sp = m.predict(split.train, profiles, split.cutoff)
            return pd.concat([sp[~sp["unique_id"].isin(longs)], long_pred],
                             ignore_index=True)

        p11 = hybrid(FEATURES_V1)
        p12 = hybrid(FEATURES_SHORT_AGE)
        base = structural_baseline(split.train, split.test, profiles, split.cutoff)
        results = {"baseline": score(base, split, profiles),
                   "v11": score(p11, split, profiles),
                   "v12": score(p12, split, profiles)}
        print(f"  prototype: short {PROTOTYPE[name][0]}, long {PROTOTYPE[name][1]}")
        print(score_table(results).to_string())
        print("\nbias% by segment:")
        print(pd.DataFrame(
            {n: t.set_index("segment")["bias_pct"] for n, t in results.items()}
        ).to_string())
        for seg in ("short", "long"):
            bd = bootstrap_delta(p12, p11, split, profiles, segment=seg)
            print(f"  {seg:<5}: v12-vs-v11 {bd['delta']:+.4f} "
                  f"{'SIG' if is_significant(bd) else ''}")


if __name__ == "__main__":
    main()
