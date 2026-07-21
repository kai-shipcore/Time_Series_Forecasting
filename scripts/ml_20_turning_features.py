#!/usr/bin/env python3
"""ML experiment 20: v11 exploration — turning-point features for long SKUs.

The Section 4.18 growth drift is the last deficit and is invariant to
hyperparameters (4.26): the model learns ratios that rise with lead and cannot
see that a long SKU sitting far above its annual norm is about to revert. A
pre-check found elevation (4wk / 52wk) correlates -0.57 with next-window change
in Dec-Feb, beating the -0.46 of the ramp the model already uses.

This is an EXPLORATION matrix, not a single pre-registered test: it tries
several feature/mode combinations to see which direction helps. Trying N
variants and picking the best inflates significance, so the winner must be
re-run as a single pre-registered v11 and ultimately cleared on the quarantined
final test before adoption. All features are gated to long SKUs (neutral 1.0
for short), so short results should not move; that is the key safety check.

Variants (all on v9 = FEATURES_V1 + the (12,15) window):
  v9              baseline
  +elev           add elevation vs annual
  +accel          add acceleration
  +both           add both
  elev_replace    replace ramp with elevation for long SKUs (ramp kept for short)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ml.dataset import dev_splits, load_weekly, stratified_val_skus
from src.ml.evaluate import bootstrap_delta, is_significant, score
from src.ml.model import (FEATURES_V1, FEATURES_ELEV, FEATURES_ACCEL,
                          FEATURES_BOTH, FEATURES_ELEV_REP, RatioLGBM,
                          structural_baseline)

VARIANTS = {"v9": FEATURES_V1, "+elev": FEATURES_ELEV, "+accel": FEATURES_ACCEL,
            "+both": FEATURES_BOTH, "elev_replace": FEATURES_ELEV_REP}


def main():
    weekly, profiles = load_weekly()
    smooth = profiles.loc[profiles["bucket"] == "smooth", "unique_id"]
    weekly = weekly[weekly["unique_id"].isin(set(smooth))]

    for split, name in zip(dev_splits(weekly, n=3), ["Mar-May", "Dec-Feb", "Oct-Dec"]):
        print(f"\n{'=' * 70}\n{name}")
        val = stratified_val_skus(split.train, profiles)
        base = structural_baseline(split.train, split.test, profiles, split.cutoff)
        preds = {}
        for tag, feats in VARIANTS.items():
            m = RatioLGBM(split.horizon, feats, deseas_features=True,
                          deseas_all=True).fit(split.train, profiles, split.cutoff, val)
            preds[tag] = m.predict(split.train, profiles, split.cutoff)

        rows = []
        for tag, pr in preds.items():
            t = score(pr, split, profiles)
            rows.append({"variant": tag,
                         "short": float(t[t.segment == "smooth/short"].pooled_wape.iloc[0]),
                         "long": float(t[t.segment == "smooth/long"].pooled_wape.iloc[0]),
                         "long_bias": float(t[t.segment == "smooth/long"].bias_pct.iloc[0])})
        tb = score(base, split, profiles)
        vb_long = float(tb[tb.segment == "smooth/long"].pooled_wape.iloc[0])
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        print(f"  v-base long = {vb_long:.4f}  (the number the long segment must beat)")
        for tag in ("+elev", "+accel", "+both", "elev_replace"):
            for seg in ("long", "short"):
                bd = bootstrap_delta(preds[tag], preds["v9"], split, profiles, segment=seg)
                mark = "  <-- SIG" if is_significant(bd) else ""
                print(f"    {tag:<13} vs v9 [{seg:<5}] delta={bd['delta']:+.4f}{mark}")


if __name__ == "__main__":
    main()
