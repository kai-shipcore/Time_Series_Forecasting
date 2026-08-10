#!/usr/bin/env python3
"""ML experiment 22: v11 — hybrid short/long model, formalized.

SHORT: shared v9 model (FEATURES_V1, all SKUs). Identical to v9 by construction.
LONG:  long-only model, FEATURES_V11_LONG = [lead, y_last_r, lag_1_r, elev_long],
       early-stopping validation re-stratified WITHIN the long segment.

Pre-registered design and criteria: design doc Section 6, v11.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ml.dataset import dev_splits, load_weekly, stratified_val_skus
from src.ml.evaluate import bootstrap_delta, is_significant, score, score_table
from src.ml.model import (FEATURES_V1, FEATURES_V11_LONG, RatioLGBM,
                          long_sku_set, structural_baseline)

from src.ml.reference import PROTOTYPE, warn_if_stale  # noqa: E402

warn_if_stale()


def main():
    weekly, profiles = load_weekly()
    smooth = profiles.loc[profiles["bucket"] == "smooth", "unique_id"]
    weekly = weekly[weekly["unique_id"].isin(set(smooth))]

    for split, name in zip(dev_splits(weekly, n=3), PROTOTYPE):
        print(f"\n{'=' * 66}\n{name}  {split}")
        longs = long_sku_set(profiles, split.cutoff) & set(split.train["unique_id"])
        val_all = stratified_val_skus(split.train, profiles)
        # early-stopping SKUs re-stratified within the long segment
        val_long = stratified_val_skus(
            split.train[split.train["unique_id"].isin(longs)], profiles)

        m9 = RatioLGBM(split.horizon, FEATURES_V1, deseas_features=True,
                       deseas_all=True).fit(split.train, profiles, split.cutoff, val_all)
        p9 = m9.predict(split.train, profiles, split.cutoff)

        mL = RatioLGBM(split.horizon, FEATURES_V11_LONG, deseas_features=True,
                       deseas_all=True, uids=longs).fit(
            split.train, profiles, split.cutoff, val_long)
        long_pred = mL.predict(split.train, profiles, split.cutoff)
        p11 = pd.concat([p9[~p9["unique_id"].isin(longs)], long_pred], ignore_index=True)

        base = structural_baseline(split.train, split.test, profiles, split.cutoff)
        results = {"baseline": score(base, split, profiles),
                   "v9": score(p9, split, profiles),
                   "v11": score(p11, split, profiles)}
        vb_long = float(results["baseline"][
            results["baseline"].segment == "smooth/long"].pooled_wape.iloc[0])
        print(f"\n  long model: {mL.n_train_rows:,} rows, {len(val_long)} val SKUs, "
              f"{mL.model.best_iteration_} trees")
        print(f"  prototype: short {PROTOTYPE[name][0]}, long {PROTOTYPE[name][1]}")
        print(score_table(results).to_string())
        print("\nbias% by segment:")
        print(pd.DataFrame(
            {n: t.set_index("segment")["bias_pct"] for n, t in results.items()}
        ).to_string())
        for seg in ("short", "long"):
            v9d = bootstrap_delta(p11, p9, split, profiles, segment=seg)
            vbd = bootstrap_delta(p11, base, split, profiles, segment=seg)
            print(f"  {seg:<5}: v11-vs-v9 {v9d['delta']:+.4f} "
                  f"{'SIG' if is_significant(v9d) else '   '}   "
                  f"v11-vs-vbase {vbd['delta']:+.4f} "
                  f"{'SIG' if is_significant(vbd) else ''}")


if __name__ == "__main__":
    main()
