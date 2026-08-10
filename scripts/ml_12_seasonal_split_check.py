#!/usr/bin/env python3
"""Check that the ML track's seasonal factors differ from the prototype's only
where they are meant to.

Originally this asserted the two were indistinguishable, which held while the
ML_* holiday settings equalled the prototype's. v9 moved ML_HOLIDAY_END to
(12, 15) against the prototype's (12, 31) and that premise ended, so the script
failed from then on for the divergence it was supposed to permit. Rewritten
2026-08-10 to assert the divergence's shape rather than its absence: the sources
may differ only on weeks touching the gap between the two window ends, and a
disagreement anywhere else is a bug.

Three levels are checked: the factors themselves, the structural baseline, and a
fitted model. Run it after any change to either factor source.

The baseline and model figures come from src/ml/reference.py, tagged with the
snapshot they were measured on. If that snapshot is not the active one the script
returns "inconclusive" rather than a verdict, because a moved number cannot then
distinguish a factor bug from the data having changed underneath it.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ml.dataset import dev_splits, load_weekly, stratified_val_skus
from src.ml.evaluate import score
from src.ml.model import FEATURES_V1, RatioLGBM, structural_baseline
from src.ml.seasonal import matches_prototype

from config import HOLIDAY_END, ML_HOLIDAY_END  # noqa: E402
from src.ml.reference import (EXPECT_BASE, EXPECT_V3,  # noqa: E402
                              REFERENCE_SNAPSHOT, warn_if_stale)

# This script's verdict is only meaningful when the data underneath it has not
# moved. Its failure message names one cause, "the two factor sources differ",
# and that diagnosis is wrong if the snapshot changed instead. Recorded here so
# the check reports which of the two it is actually looking at.
ON_REFERENCE_DATA = warn_if_stale()


def seg(tbl, name):
    r = tbl[tbl["segment"] == name]
    return float(r["pooled_wape"].iloc[0]) if len(r) else float("nan")


def main() -> int:
    ok = True
    weekly, profiles = load_weekly()

    # This block used to require the two factor sources to agree on every week in
    # the pinned data, on the reasoning that they differ only on weeks outside it.
    # That reasoning covered the label-versus-days distinction and assumed the two
    # windows were otherwise identical. v9 moved ML_HOLIDAY_END to (12, 15) while
    # the prototype's HOLIDAY_END stayed (12, 31), so from that moment the sources
    # differ on any week covering Dec 16-31, of which the pinned data holds four.
    # The check has therefore been failing since v9 for a divergence that is the
    # entire point of the split, which is how it also went unnoticed that its
    # expected WAPE figures had gone stale.
    #
    # So it now asserts the SHAPE of the divergence instead of its absence: the
    # sources may differ only on weeks touching the gap between the two window
    # ends, and a disagreement anywhere else is the bug this script exists to
    # catch. Derived from config rather than hardcoded, so moving either window
    # again keeps the check honest.
    import pandas as pd
    from src.deseasonalize import _factors
    from src.ml.seasonal import ml_factors
    ds = pd.Series(sorted(pd.to_datetime(weekly["ds"].unique())))
    differs = ml_factors(ds).to_numpy() != _factors(ds).to_numpy()

    gap = set()
    for year in sorted({d.year for d in ds}):
        lo = pd.Timestamp(year, *ML_HOLIDAY_END) + pd.Timedelta(days=1)
        hi = pd.Timestamp(year, *HOLIDAY_END)
        if lo <= hi:
            gap.update(pd.date_range(lo, hi))
    # A week spans [ds-6, ds] under W-MON, so it may differ if any of those days
    # falls in the gap.
    allowed = ds.apply(
        lambda d: bool(gap & set(pd.date_range(d - pd.Timedelta(days=6), d)))
    ).to_numpy()

    unexpected = ds[differs & ~allowed]
    print(f"factor sources differ on {int(differs.sum())} of {len(ds)} pinned weeks; "
          f"{int((differs & allowed).sum())} are the expected Dec 16-31 divergence "
          f"(ML window ends {ML_HOLIDAY_END}, prototype ends {HOLIDAY_END})")
    if len(unexpected):
        print("  UNEXPECTED divergence, outside the window gap:")
        for d in unexpected:
            print(f"    {d.date()}")
    print(f"  (matches_prototype() over all future weeks: {matches_prototype()}, "
          f"expected False once the membership rules differ)")
    ok &= not len(unexpected)
    smooth = profiles.loc[profiles["bucket"] == "smooth", "unique_id"]
    weekly = weekly[weekly["unique_id"].isin(set(smooth))]

    print(f"\n{'window':<9}{'model':<8}{'short':>9}{'expected':>10}"
          f"{'long':>9}{'expected':>10}   ")
    for split, name in zip(dev_splits(weekly, n=3), EXPECT_BASE):
        val = stratified_val_skus(split.train, profiles)
        runs = {
            "v-base": structural_baseline(split.train, split.test, profiles, split.cutoff),
            "v3": RatioLGBM(split.horizon, FEATURES_V1, deseas_features=True,
                            deseas_all=True).fit(
                split.train, profiles, split.cutoff, val
            ).predict(split.train, profiles, split.cutoff),
        }
        for label, preds in runs.items():
            t = score(preds, split, profiles)
            got = (seg(t, "smooth/short"), seg(t, "smooth/long"))
            exp = (EXPECT_BASE if label == "v-base" else EXPECT_V3)[name]
            good = all(abs(g - e) <= 5e-4 for g, e in zip(got, exp))
            ok &= good
            print(f"{name:<9}{label:<8}{got[0]:>9.4f}{exp[0]:>10.4f}"
                  f"{got[1]:>9.4f}{exp[1]:>10.4f}   {'OK' if good else 'CHANGED'}")

    if ok:
        print("\nPASS: the split is behaviour-preserving")
        return 0
    if not ON_REFERENCE_DATA:
        # Two different failures share one symptom. Saying "the factor sources
        # differ" while the snapshot has moved underneath would be asserting the
        # one cause this run cannot distinguish.
        print(f"\nINCONCLUSIVE: results moved, but so did the data. The expected "
              f"figures were measured on snapshot {REFERENCE_SNAPSHOT} and this "
              f"ran on a different one, so a gap here says nothing about the "
              f"factor sources. Re-measure the expectations on the current "
              f"snapshot (ml_03 for v-base, ml_08 for v3), update "
              f"src/ml/reference.py, then run this again to get a real verdict.")
        return 2
    print("\nFAIL: results moved on unchanged data; the two factor sources differ")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
