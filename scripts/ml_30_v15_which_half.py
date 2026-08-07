#!/usr/bin/env python3
"""Which half of v15 causes the smooth/long drift?

v15 changes two things in one move and the design doc records the consequence
without attributing it: smooth/long regressed in all three windows, +0.0059,
+0.0090 and +0.0027, none individually beyond noise but all one direction.

The two changes are separable:

  monthly   The monthly index is averaged over the seven days a week covers,
            instead of read off the label's month. This is the fix for a week
            that is six-sevenths July taking August's multiplier. It touches
            the 22 of 110 weeks that straddle a month boundary.
  holiday   The 4-of-7 majority vote is replaced by a proportional lift, so a
            week with three holiday days gets three sevenths of 1.26 rather
            than nothing. It touches the 3 partial weeks, and it is why the
            count of weeks at exactly 1.26 falls from 8 to 5.

This runs a third arm with ONLY the monthly half blended, keeping the week-level
holiday majority override exactly as v11 has it. Comparing the three:

  v15 regresses on long and monthly-only does not  ->  the holiday cliff removal
  v15 and monthly-only both regress               ->  the month-boundary blend
  neither reproduces it                           ->  it was noise

Diagnostic only. Nothing here is a candidate version, and the monthly-only arm
is not proposed for adoption; it exists to attribute the effect.

    .venv/bin/python scripts/ml_30_v15_which_half.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import src.ml.seasonal as seas  # noqa: E402
from ml_26_week_boundary_ab import WINDOW_NAMES, draw_val, fit_v11  # noqa: E402
from src.ml.dataset import dev_splits, load_weekly  # noqa: E402
from src.ml.evaluate import bootstrap_delta, score  # noqa: E402


def blended_monthly_only(ds: pd.Series) -> pd.Series:
    """Monthly index averaged over the week's days; holiday left as a step.

    Deliberately mirrors `seas.ml_factors_blended` except for the holiday
    treatment, so the difference between the two arms is exactly one change.
    """
    ds_dt = pd.to_datetime(ds)
    acc = np.zeros(len(ds_dt), dtype=float)
    for k in range(7):
        d = pd.DatetimeIndex(ds_dt - pd.Timedelta(days=k))
        acc += pd.Series(d.month).map(seas.ML_SEASONAL).astype(float).to_numpy()
    monthly = pd.Series(acc / 7.0, index=ds.index)
    if seas.ML_USE_HOLIDAY_FLAG:
        monthly = monthly.where(~seas.ml_is_holiday(ds), seas.ML_HOLIDAY_MULTIPLIER)
    return monthly


def main() -> int:
    weekly, profiles = load_weekly()
    smooth = set(profiles.loc[profiles["bucket"] == "smooth", "unique_id"])
    weekly = weekly[weekly["unique_id"].isin(smooth)]

    real_blend = seas.ml_factors_blended
    rows = []
    for i, split in enumerate(dev_splits(weekly, n=3)):
        win = WINDOW_NAMES[i]
        val = draw_val(split, profiles)
        preds = {}

        seas.ML_SEASONAL_BLEND = False
        preds["v11"] = fit_v11(split, profiles, val=val)

        seas.ML_SEASONAL_BLEND = True
        try:
            preds["v15"] = fit_v11(split, profiles, val=val)
            seas.ml_factors_blended = blended_monthly_only
            preds["monthly_only"] = fit_v11(split, profiles, val=val)
        finally:
            seas.ml_factors_blended = real_blend
            seas.ML_SEASONAL_BLEND = False

        for arm, p in preds.items():
            for _, r in score(p[["unique_id", "ds", "yhat"]], split,
                              profiles).iterrows():
                rows.append({"arm": arm, "window": win, "segment": r["segment"],
                             "pooled_wape": r["pooled_wape"]})

        for arm in ("v15", "monthly_only"):
            b = bootstrap_delta(preds[arm], preds["v11"], split, profiles,
                                segment="long")
            rows.append({"arm": f"{arm}_vs_v11_long", "window": win,
                         "segment": "boot", "pooled_wape": b["delta"],
                         "se": b["se"]})
        print(f"  {win}: three arms fitted")

    d = pd.DataFrame(rows)
    tab = d[d.segment != "boot"].pivot_table(
        index=["segment", "window"], columns="arm", values="pooled_wape")
    tab["v15-v11"] = tab["v15"] - tab["v11"]
    tab["monthly-v11"] = tab["monthly_only"] - tab["v11"]
    print(f"\n{'=' * 78}\npooled WAPE by arm, and each arm's delta against v11")
    print(tab.round(4).to_string())

    print(f"\n{'=' * 78}\nsmooth/long only, the segment that drifted")
    lo = tab.loc["smooth/long"]
    print(f"  {'window':<9} {'v15-v11':>10} {'monthly-v11':>14}   "
          f"share of the v15 drift explained by the monthly half")
    for win in WINDOW_NAMES:
        a, b = lo.loc[win, "v15-v11"], lo.loc[win, "monthly-v11"]
        share = f"{b / a * 100:5.0f}%" if abs(a) > 1e-9 else "   n/a"
        print(f"  {win:<9} {a:>+10.4f} {b:>+14.4f}   {share}")
    print(f"  {'MEAN':<9} {lo['v15-v11'].mean():>+10.4f} "
          f"{lo['monthly-v11'].mean():>+14.4f}")

    print("\n  Reading it: if the monthly-only column reproduces most of the v15")
    print("  drift, the month-boundary blend is responsible. If it is near zero")
    print("  while v15 is not, the holiday cliff removal is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
