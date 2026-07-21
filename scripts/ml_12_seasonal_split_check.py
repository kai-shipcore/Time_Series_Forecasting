#!/usr/bin/env python3
"""Check that giving the ML track its own seasonal factors changed nothing.

While config's ML_* holiday settings equal the prototype's, src/ml/seasonal.py
must be indistinguishable from src/deseasonalize.py everywhere it is used.
This asserts that at three levels: the factors themselves, the structural
baseline, and a fitted model. Run it after any change to either factor source;
a failure means the two have diverged, which is fine when intended and a bug
when not.
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

EXPECT_BASE = {"Mar-May": (0.2097, 0.1321), "Dec-Feb": (0.1788, 0.2764),
               "Oct-Dec": (0.4861, 0.1209)}
EXPECT_V3 = {"Mar-May": (0.1863, 0.1345), "Dec-Feb": (0.1943, 0.3145),
             "Oct-Dec": (0.1826, 0.1011)}


def seg(tbl, name):
    r = tbl[tbl["segment"] == name]
    return float(r["pooled_wape"].iloc[0]) if len(r) else float("nan")


def main() -> int:
    ok = True
    print(f"factors identical to prototype: {matches_prototype()}")
    ok &= matches_prototype()

    weekly, profiles = load_weekly()
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

    print("\n" + ("PASS: the split is behaviour-preserving"
                  if ok else "FAIL: results moved; the two factor sources differ"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
