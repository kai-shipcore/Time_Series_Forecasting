#!/usr/bin/env python3
"""ML experiment 32: v17 — trailing 12-week Amazon-FBA share in the long model.

    .venv/bin/python scripts/ml_32_v17_channel_mix.py

Requires data/snapshots/<ML_DATA_SNAPSHOT>/channel_mix.parquet, written by
scripts/ml_31_export_channel_mix.py. No database access.

ARMS
    v11  shared short model (FEATURES_V1) + long model on FEATURES_V11_LONG
    v17  identical, except the long model gets FEATURES_V17_LONG
         (= FEATURES_V11_LONG + fba_share_12w)

The short arm is the SAME fitted model object in both, not a refit, so short
is identical by construction and any short difference is a bug. The script
asserts that rather than trusting it.

Pre-registered design and criteria: design doc Section 6, v17. Judged on
smooth/long only, by the Section 1.5 adoption rule (consistent sign across the
three windows, three-window mean improvement >= 0.01).

ONE VALIDATION DRAW PER WINDOW, SHARED BY BOTH ARMS
---------------------------------------------------
The early-stopping SKUs are drawn once per window and handed to both long
models. Drawing per arm was a real bug in the week-boundary work: the two arms
got 72% to 95% different validation SKUs, so the draw's noise landed entirely
on one arm and the measured delta was partly the draw. The draw is a nuisance
parameter here, and a nuisance parameter has to be held fixed across the thing
being compared.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.ml.dataset import dev_splits, load_weekly, stratified_val_skus  # noqa: E402
from src.ml.evaluate import (bootstrap_delta, is_significant, score,  # noqa: E402
                             score_table)
from src.ml.model import (FEATURES_V1, FEATURES_V11_LONG,  # noqa: E402
                          FEATURES_V17_LONG, RatioLGBM, add_channel_share,
                          long_sku_set, structural_baseline)

WINDOWS = ["Mar-May", "Dec-Feb", "Oct-Dec"]


def main():
    weekly, profiles = load_weekly()
    smooth = profiles.loc[profiles["bucket"] == "smooth", "unique_id"]
    weekly = weekly[weekly["unique_id"].isin(set(smooth))]

    deltas = {}
    for split, name in zip(dev_splits(weekly, n=3), WINDOWS):
        print(f"\n{'=' * 72}\n{name}  {split}")
        longs = long_sku_set(profiles, split.cutoff) & set(split.train["unique_id"])
        val_all = stratified_val_skus(split.train, profiles)
        val_long = stratified_val_skus(
            split.train[split.train["unique_id"].isin(longs)], profiles)

        # Is the feature actually varying at this cutoff? A feature that is
        # constant cannot produce a delta, and a null result from a constant
        # column would be reported as evidence about channel mix when it is
        # evidence about a broken join. Checked per window, because "it varied
        # on the full snapshot" is not the same claim.
        feat = add_channel_share(
            split.train[split.train["unique_id"].isin(longs)]
            .sort_values(["unique_id", "ds"]).reset_index(drop=True)
        )
        at_cutoff = feat[feat["ds"] == split.cutoff]["fba_share_12w"]
        print(f"  fba_share_12w at the cutoff over {len(at_cutoff)} long SKUs: "
              f"mean {at_cutoff.mean():.3f}  sd {at_cutoff.std():.3f}  "
              f"min {at_cutoff.min():.3f}  max {at_cutoff.max():.3f}  "
              f"exactly zero: {(at_cutoff == 0).sum()}")
        if at_cutoff.std() < 1e-6:
            raise ValueError(
                "fba_share_12w is constant across long SKUs at this cutoff. "
                "The join or the export is wrong; a null result here would be "
                "meaningless."
            )

        # SHORT arm: fitted once, reused by both arms untouched.
        m_short = RatioLGBM(split.horizon, FEATURES_V1, deseas_features=True,
                            deseas_all=True).fit(
            split.train, profiles, split.cutoff, val_all)
        p_short = m_short.predict(split.train, profiles, split.cutoff)
        short_only = p_short[~p_short["unique_id"].isin(longs)]

        preds, models = {}, {}
        for arm, feats in (("v11", FEATURES_V11_LONG), ("v17", FEATURES_V17_LONG)):
            m = RatioLGBM(split.horizon, feats, deseas_features=True,
                          deseas_all=True, uids=longs).fit(
                split.train, profiles, split.cutoff, val_long)
            models[arm] = m
            preds[arm] = pd.concat(
                [short_only, m.predict(split.train, profiles, split.cutoff)],
                ignore_index=True)

        base = structural_baseline(split.train, split.test, profiles, split.cutoff)
        results = {"baseline": score(base, split, profiles),
                   "v11": score(preds["v11"], split, profiles),
                   "v17": score(preds["v17"], split, profiles)}

        for arm in ("v11", "v17"):
            m = models[arm]
            print(f"  {arm} long model: {m.n_train_rows:,} rows, "
                  f"{len(val_long)} val SKUs, {m.model.best_iteration_} trees")
        print("\n  RAW per-segment results:")
        print(score_table(results).to_string())
        print("\n  bias% by segment:")
        print(pd.DataFrame(
            {n: t.set_index("segment")["bias_pct"] for n, t in results.items()}
        ).to_string())

        print("\n  v17 long-model feature importance (gain):")
        print(models["v17"].importance().to_string(index=False))

        # Criterion 2: the short arm is the same object in both, so this must
        # be exactly zero. Reported, not assumed.
        s11 = preds["v11"][~preds["v11"]["unique_id"].isin(longs)]["yhat"].sum()
        s17 = preds["v17"][~preds["v17"]["unique_id"].isin(longs)]["yhat"].sum()
        print(f"\n  short-arm check: identical predictions "
              f"({'PASS' if abs(s11 - s17) < 1e-6 else f'FAIL, diff {s11 - s17:+.4f}'})")

        d = bootstrap_delta(preds["v17"], preds["v11"], split, profiles, segment="long")
        deltas[name] = d
        print(f"  smooth/long  v17 - v11: {d['delta']:+.4f}  se {d['se']:.4f}  "
              f"{'SIG' if is_significant(d) else 'not distinguishable from noise'}")

    print(f"\n{'=' * 72}\nSECTION 1.5 VERDICT, smooth/long")
    vals = [deltas[w]["delta"] for w in WINDOWS]
    for w in WINDOWS:
        print(f"  {w:<8} {deltas[w]['delta']:+.4f}  (se {deltas[w]['se']:.4f})")
    mean = sum(vals) / len(vals)
    # Negative delta = v17 has lower WAPE = improvement.
    consistent = all(v < 0 for v in vals) or all(v > 0 for v in vals)
    print(f"\n  mean {mean:+.4f}   sign consistent across all three: {consistent}")
    print(f"  adoption needs consistent sign AND mean improvement >= 0.0100")
    if consistent and mean <= -0.01:
        print("  -> criteria MET on the primary rule. Check the short-arm and "
              "importance checks above before writing this up.")
    else:
        print("  -> criteria NOT met. v17 is rejected. Per the pre-registered "
              "guard, the three-column version is not run.")


if __name__ == "__main__":
    main()
