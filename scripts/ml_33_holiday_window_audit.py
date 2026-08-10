#!/usr/bin/env python3
"""Which recorded figures were measured before v9 moved the holiday window?

    .venv/bin/python scripts/ml_33_holiday_window_audit.py

Runs v-base and v3 on the OLD pinned snapshot under both holiday windows and
compares each against the figure written in the design doc. No database.

What this is for
----------------
The Section 6 v-base table did not reproduce. Four of its six cells matched a
fresh run on the snapshot it claims to be measured on; Mar-May long and Dec-Feb
long did not. Setting ML_HOLIDAY_END back to (12, 31), its value before v9,
reproduces all six exactly including the bias percentages.

v-base applies the seasonal round-trip to long SKUs, so the holiday window is
one of its inputs. When v9 moved ML_HOLIDAY_END from (12, 31) to (12, 15), the
baseline moved with it and the recorded table was not re-measured. Every
"versus v-base" comparison since then has been against a baseline built on
different seasonal factors from the model it was judging.

The signature is specific, which is why it is worth confirming rather than
assuming. Only long figures can move, because short SKUs get no round-trip in
v-base. Only windows containing weeks that COVER Dec 16-31 can move, because
that is where the two windows differ; under W-MON those are the weeks labelled
Dec 22 and Dec 29. That predicts Oct-Dec is untouched (its test window ends
Dec 15), Dec-Feb moves most (it holds both weeks), and Mar-May moves only
through its training data. All three held for v-base.

This script asks whether v3's recorded figures have the same problem. If they
do, the fix is the same: re-measure, and record that the old numbers were a
different seasonal specification rather than a different model.

Reads the recorded values from src/ml/reference.py so there is one copy of them.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import src.ml.seasonal as seas  # noqa: E402
from config import ML_FINAL_TEST_CUTOFF  # noqa: E402
from src.ml.dataset import (data_dir, make_splits,  # noqa: E402
                            stratified_val_skus)
from src.ml.evaluate import score  # noqa: E402
from src.ml.model import (FEATURES_V1, RatioLGBM,  # noqa: E402
                          structural_baseline)
from src.ml.reference import EXPECT_BASE, EXPECT_V3  # noqa: E402

OLD_SNAPSHOT = "2026-07-20"
WINDOWS = ["FINAL TEST", "Mar-May", "Dec-Feb", "Oct-Dec"]
JUDGED = ["Mar-May", "Dec-Feb", "Oct-Dec"]


def measure(snapshot: str) -> dict:
    d = data_dir(snapshot)
    weekly = pd.read_parquet(d / "sales_clean.parquet")
    weekly["ds"] = pd.to_datetime(weekly["ds"])
    profiles = pd.read_csv(d / "sku_profiles.csv")
    weekly = weekly[weekly["unique_id"].isin(
        set(profiles.loc[profiles["bucket"] == "smooth", "unique_id"]))]

    out: dict = {}
    splits = make_splits(weekly, n_splits=4, anchor=ML_FINAL_TEST_CUTOFF)
    for split, name in zip(splits, WINDOWS):
        if name not in JUDGED:
            continue
        # One validation draw, used by the single fitted model. Nothing is being
        # compared across arms here, so there is no cross-arm draw to share.
        val = stratified_val_skus(split.train, profiles)
        runs = {
            "v-base": structural_baseline(split.train, split.test, profiles, split.cutoff),
            "v3": RatioLGBM(split.horizon, FEATURES_V1, deseas_features=True,
                            deseas_all=True).fit(
                split.train, profiles, split.cutoff, val
            ).predict(split.train, profiles, split.cutoff),
        }
        for label, preds in runs.items():
            t = score(preds, split, profiles).set_index("segment")
            out[(name, label)] = (float(t.loc["smooth/short", "pooled_wape"]),
                                  float(t.loc["smooth/long", "pooled_wape"]))
    return out


def main() -> int:
    print(f"snapshot under test: {OLD_SNAPSHOT} (the one the figures claim to be from)\n")
    results = {}
    for end in ((12, 15), (12, 31)):
        seas.ML_HOLIDAY_END = end
        results[end] = measure(OLD_SNAPSHOT)

    print(f"{'window':<9}{'model':<8}{'seg':<7}{'recorded':>10}"
          f"{'(12,15)':>10}{'(12,31)':>10}   reproduces under")
    verdict = {}
    for name in JUDGED:
        for label, table in (("v-base", EXPECT_BASE), ("v3", EXPECT_V3)):
            for i, seg in enumerate(("short", "long")):
                rec = table[name][i]
                a = results[(12, 15)][(name, label)][i]
                b = results[(12, 31)][(name, label)][i]
                hits = [w for w, v in (("(12,15)", a), ("(12,31)", b))
                        if abs(v - rec) < 5e-4]
                where = " and ".join(hits) if hits else "NEITHER"
                verdict.setdefault(label, []).append(where)
                print(f"{name:<9}{label:<8}{seg:<7}{rec:>10.4f}{a:>10.4f}{b:>10.4f}   {where}")
        print()

    print("Reading this table")
    print("  'both'      the cell cannot distinguish the two windows, which is")
    print("              expected for every short cell (no seasonal round-trip")
    print("              in v-base) and for Oct-Dec (test ends Dec 15).")
    print("  '(12,31)'   the figure was measured before v9 and is stale.")
    print("  '(12,15)'   the figure is current.")
    print("  'NEITHER'   something else moved as well; do not attribute this one")
    print("              to the holiday window without a further check.")
    for label, wheres in verdict.items():
        stale = sum(1 for w in wheres if w == "(12,31)")
        never = sum(1 for w in wheres if w == "NEITHER")
        print(f"\n  {label}: {stale} cell(s) reproduce only under the pre-v9 window, "
              f"{never} reproduce under neither.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
