#!/usr/bin/env python3
"""ML experiment 21: v11 hybrid — shared short model + dedicated long model.

The synthesis of the whole arc from v4 onward:
  - Section 4.24 / v11 exploration: long-targeted features in a SHARED model
    reorganize the shared trees and hurt short SKUs.
  - v8: a long-ONLY model forecasts long fine; only the short-only model
    suffered, because short depends on cross-segment transfer.
  - v11 exploration: the elevation feature (4wk / 52wk) is a genuine long-SKU
    breakthrough, finally beating v-base on Dec-Feb long.

Hybrid architecture:
  SHORT SKUs -> predictions from the shared v9 model (FEATURES_V1, all SKUs).
                Identical to v9 by construction, so short qualification cannot
                break.
  LONG SKUs  -> predictions from a model trained on LONG SKUs ONLY, with the
                turning-point features added. Not distracted by short SKUs;
                keeps the elevation signal; no shared trees to leak into short.

Tests three long-model feature sets. Exploration still: the winner must be
re-run as a single pre-registered version and cleared on the final test.

Known limitation, carried from v8: the long-only model has 6-7 validation SKUs
for early stopping, so its stopping point is noisy.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ml.dataset import dev_splits, load_weekly, stratified_val_skus
from src.ml.evaluate import bootstrap_delta, is_significant, score
from src.ml.model import (FEATURES_V1, FEATURES_ELEV, FEATURES_BOTH,
                          FEATURES_ELEV_REP, RatioLGBM, long_sku_set,
                          structural_baseline)

LONG_VARIANTS = {"hybrid+elev": FEATURES_ELEV, "hybrid+both": FEATURES_BOTH,
                 "hybrid_replace": FEATURES_ELEV_REP}


def main():
    weekly, profiles = load_weekly()
    smooth = profiles.loc[profiles["bucket"] == "smooth", "unique_id"]
    weekly = weekly[weekly["unique_id"].isin(set(smooth))]

    for split, name in zip(dev_splits(weekly, n=3), ["Mar-May", "Dec-Feb", "Oct-Dec"]):
        print(f"\n{'=' * 70}\n{name}")
        val = stratified_val_skus(split.train, profiles)
        longs = long_sku_set(profiles, split.cutoff) & set(split.train["unique_id"])

        # shared v9 model: its SHORT predictions feed every hybrid
        m9 = RatioLGBM(split.horizon, FEATURES_V1, deseas_features=True,
                       deseas_all=True).fit(split.train, profiles, split.cutoff, val)
        p9 = m9.predict(split.train, profiles, split.cutoff)
        short_pred = p9[~p9["unique_id"].isin(longs)]

        base = structural_baseline(split.train, split.test, profiles, split.cutoff)
        vb = score(base, split, profiles)
        vb_long = float(vb[vb.segment == "smooth/long"].pooled_wape.iloc[0])

        preds = {"v9": p9}
        for tag, feats in LONG_VARIANTS.items():
            mL = RatioLGBM(split.horizon, feats, deseas_features=True,
                           deseas_all=True, uids=longs).fit(
                split.train, profiles, split.cutoff, val)
            long_pred = mL.predict(split.train, profiles, split.cutoff)
            preds[tag] = pd.concat([short_pred, long_pred], ignore_index=True)

        rows = []
        for tag, pr in preds.items():
            t = score(pr, split, profiles)
            rows.append({"variant": tag,
                         "short": float(t[t.segment == "smooth/short"].pooled_wape.iloc[0]),
                         "long": float(t[t.segment == "smooth/long"].pooled_wape.iloc[0]),
                         "long_bias": float(t[t.segment == "smooth/long"].bias_pct.iloc[0])})
        print(pd.DataFrame(rows).to_string(index=False))
        print(f"  v-base long = {vb_long:.4f}  (the target the long segment must beat)")
        for tag in LONG_VARIANTS:
            bl = bootstrap_delta(preds[tag], preds["v9"], split, profiles, segment="long")
            bs = bootstrap_delta(preds[tag], preds["v9"], split, profiles, segment="short")
            print(f"    {tag:<15} long delta={bl['delta']:+.4f}"
                  f"{'  SIG' if is_significant(bl) else '':<5} "
                  f"short delta={bs['delta']:+.4f}"
                  f"{'  SIG' if is_significant(bs) else ''}")


if __name__ == "__main__":
    main()
