#!/usr/bin/env python3
"""ML experiment 24: v13 — acceleration, two independent tests.

v13-short: accel added to the shared model (only short moves).
v13-long:  accel added to the dedicated long model (only long moves).
Pre-registered design and criteria: design doc Section 6, v13.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ml.dataset import dev_splits, load_weekly, stratified_val_skus
from src.ml.evaluate import bootstrap_delta, is_significant, score, score_table
from src.ml.model import (FEATURES_V1, FEATURES_SHORT_ACCEL, FEATURES_V11_LONG,
                          FEATURES_V11_LONG_ACCEL, RatioLGBM, long_sku_set,
                          structural_baseline)

PROTOTYPE = {"Mar-May": (0.2014, 0.1411), "Dec-Feb": (0.2863, 0.2737),
             "Oct-Dec": (0.4251, 0.0911)}


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

        def short_model(feats):
            m = RatioLGBM(split.horizon, feats, deseas_features=True,
                          deseas_all=True).fit(split.train, profiles, split.cutoff, val_all)
            sp = m.predict(split.train, profiles, split.cutoff)
            return sp[~sp["unique_id"].isin(longs)]

        def long_model(feats):
            m = RatioLGBM(split.horizon, feats, deseas_features=True, deseas_all=True,
                          uids=longs).fit(split.train, profiles, split.cutoff, val_long)
            return m.predict(split.train, profiles, split.cutoff)

        short_v11 = short_model(FEATURES_V1)
        short_a = short_model(FEATURES_SHORT_ACCEL)
        long_v11 = long_model(FEATURES_V11_LONG)
        long_a = long_model(FEATURES_V11_LONG_ACCEL)

        p11 = pd.concat([short_v11, long_v11], ignore_index=True)
        p_sa = pd.concat([short_a, long_v11], ignore_index=True)     # v13-short
        p_la = pd.concat([short_v11, long_a], ignore_index=True)     # v13-long

        base = structural_baseline(split.train, split.test, profiles, split.cutoff)
        results = {"baseline": score(base, split, profiles),
                   "v11": score(p11, split, profiles),
                   "v13-short": score(p_sa, split, profiles),
                   "v13-long": score(p_la, split, profiles)}
        print(f"  prototype: short {PROTOTYPE[name][0]}, long {PROTOTYPE[name][1]}")
        print(score_table(results).to_string())
        bs = bootstrap_delta(p_sa, p11, split, profiles, segment="short")
        bl = bootstrap_delta(p_la, p11, split, profiles, segment="long")
        print(f"  v13-short short vs v11: {bs['delta']:+.4f} "
              f"{'SIG' if is_significant(bs) else ''}")
        print(f"  v13-long  long  vs v11: {bl['delta']:+.4f} "
              f"{'SIG' if is_significant(bl) else ''}")


if __name__ == "__main__":
    main()
