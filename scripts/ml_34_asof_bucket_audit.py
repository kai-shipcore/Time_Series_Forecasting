#!/usr/bin/env python3
"""How much does evaluating on AS-OF buckets change v11's recorded numbers?

    .venv/bin/python scripts/ml_34_asof_bucket_audit.py

No database. Reads the pinned snapshot. Writes nothing outside a temp directory.

The question
------------
`src/ml/dataset.py` takes a SKU's history length as-of the cutoff, deliberately
and for stated reasons, but takes its BUCKET (smooth versus intermittent) from
the present-day profile. The design doc calls the as-of bucket recompute "a
separate, deeper change" and has never done it.

That is not only a coverage gap. It is future information deciding who gets
scored. Measured on the pinned snapshot: at the 2025-10-06 cutoff, 121 SKUs
were classifiable as smooth from data available then, while the harness scores
against today's 467. So 349 SKUs enter the Oct-Dec window on the strength of
behaviour that had not happened yet, and 3 that WERE smooth at the time are
excluded because they are not smooth now.

Both directions are wrong in the same way, and they do not obviously cancel:
SKUs admitted with hindsight are ones that went on to sell smoothly, which is
the easier population to forecast.

What this measures
------------------
v11 refitted and rescored per window on the AS-OF population, against the same
model on today's population. Everything else is held identical: same cutoffs,
same test weeks, same feature sets, same validation-draw discipline.

As-of means as-of throughout, not just at scoring time:
  - the bucket, from `src.profile.profile` run on sales truncated at the cutoff
  - the training population, being the SKUs smooth as-of the cutoff
  - `train_start` and the ramp trim, from the same as-of profile
  - segment labels and eligibility, since `score()` derives both from whichever
    profile frame it is handed

Scoring a SKU the model was never trained on would produce a zero forecast and
a meaningless number, so the two have to move together.

What this does NOT do
---------------------
It does not change the harness. It is a measurement that informs whether that
change is worth making, and it deliberately leaves every recorded figure alone
so the comparison stays honest.

Reading the result
------------------
A small delta means the leak is real but immaterial, the recorded figures stand,
and this becomes a documented limitation. A large delta means the development
numbers are partly an artifact of hindsight selection, which would matter more
than any model version in this log, and the final test should not be read
without it.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import src.profile as P  # noqa: E402

# profile() writes sku_profiles.csv into PROCESSED_DIR as a side effect. Point it
# at a temp directory BEFORE any call, so an audit can never overwrite the live
# profile or the pinned snapshot. Same guard scripts/promoted_sku_accuracy.py
# uses, for the same reason.
P.PROCESSED_DIR = pathlib.Path(tempfile.mkdtemp())

from config import ML_DATA_SNAPSHOT, ML_FINAL_TEST_CUTOFF, TEST_WEEKS  # noqa: E402
from src.ml.dataset import Split, data_dir, stratified_val_skus  # noqa: E402
from src.ml.evaluate import bootstrap_delta, score  # noqa: E402
from src.ml.model import (FEATURES_V1, FEATURES_V11_LONG, RatioLGBM,  # noqa: E402
                          long_sku_set, structural_baseline)

WINDOWS = ["FINAL TEST", "Mar-May", "Dec-Feb", "Oct-Dec"]


def build_split(raw: pd.DataFrame, profiles: pd.DataFrame, cutoff: pd.Timestamp,
                test_weeks: list[pd.Timestamp], smooth: set[str]) -> Split:
    """A Split whose training frame is trimmed by the profile it is given.

    Mirrors load_weekly's trim rather than reusing it, because load_weekly reads
    the snapshot's profile from disk and the whole point here is to substitute a
    different one.
    """
    w = raw[raw["unique_id"].isin(smooth)].copy()
    starts = profiles.set_index("unique_id")["train_start"]
    w["_ts"] = w["unique_id"].map(starts)
    w = w[w["_ts"].isna() | (w["ds"] >= w["_ts"])].drop(columns="_ts")
    return Split(
        cutoff=cutoff,
        train=w[w["ds"] <= cutoff].sort_values(["unique_id", "ds"]).reset_index(drop=True),
        test=raw[raw["ds"].isin(test_weeks)].copy(),
        horizon=TEST_WEEKS,
    )


def fit_v11(split: Split, profiles: pd.DataFrame) -> pd.DataFrame:
    """v11 exactly as scripts/ml_22 builds it: shared short model, dedicated long."""
    longs = long_sku_set(profiles, split.cutoff) & set(split.train["unique_id"])
    val_all = stratified_val_skus(split.train, profiles)
    short = RatioLGBM(split.horizon, FEATURES_V1, deseas_features=True,
                      deseas_all=True).fit(split.train, profiles, split.cutoff, val_all)
    preds = short.predict(split.train, profiles, split.cutoff)
    if longs:
        val_long = stratified_val_skus(
            split.train[split.train["unique_id"].isin(longs)], profiles)
        mL = RatioLGBM(split.horizon, FEATURES_V11_LONG, deseas_features=True,
                       deseas_all=True, uids=longs).fit(
            split.train, profiles, split.cutoff, val_long)
        preds = pd.concat([preds[~preds["unique_id"].isin(longs)],
                           mL.predict(split.train, profiles, split.cutoff)],
                          ignore_index=True)
    return preds


def seg(tbl: pd.DataFrame, name: str) -> float:
    r = tbl[tbl["segment"] == name]
    return float(r["pooled_wape"].iloc[0]) if len(r) else float("nan")


def main() -> int:
    src = data_dir(ML_DATA_SNAPSHOT)
    raw = pd.read_parquet(src / "sales_clean.parquet")
    raw["ds"] = pd.to_datetime(raw["ds"])
    today = pd.read_csv(src / "sku_profiles.csv")
    today["train_start"] = pd.to_datetime(today["train_start"])

    weeks = sorted(raw["ds"].unique())
    anchor = weeks.index(pd.Timestamp(ML_FINAL_TEST_CUTOFF))
    print(f"snapshot {ML_DATA_SNAPSHOT}\n")

    rows = []
    for i, name in enumerate(WINDOWS):
        ci = anchor - i * TEST_WEEKS
        cutoff = weeks[ci]
        test_weeks = weeks[ci + 1: ci + 1 + TEST_WEEKS]

        asof = P.profile(raw[raw["ds"] <= cutoff].copy())
        asof["train_start"] = pd.to_datetime(asof["train_start"])
        sm_asof = set(asof.loc[asof["bucket"] == "smooth", "unique_id"])
        sm_now = set(today.loc[today["bucket"] == "smooth", "unique_id"])

        print(f"{'=' * 74}\n{name}  cutoff {cutoff.date()}  "
              f"test {test_weeks[0].date()}..{test_weeks[-1].date()}")
        print(f"  smooth as-of {len(sm_asof)}, smooth today {len(sm_now)}, "
              f"shared {len(sm_asof & sm_now)}, admitted only by hindsight {len(sm_now - sm_asof)}")

        out = {}
        for tag, prof, smooth in (("today", today, sm_now), ("as-of", asof, sm_asof)):
            split = build_split(raw, prof, cutoff, test_weeks, smooth)
            preds = fit_v11(split, prof)
            base = structural_baseline(split.train, split.test, prof, split.cutoff)
            t = score(preds, split, prof)
            b = score(base, split, prof)
            out[tag] = (split, preds, prof, t, b)
            print(f"  {tag:<6} scored {int(t[t.segment == 'TOTAL']['n_skus'].iloc[0]) if 'n_skus' in t else -1:>4}"
                  f"   short {seg(t, 'smooth/short'):.4f}  long {seg(t, 'smooth/long'):.4f}"
                  f"  TOTAL {seg(t, 'TOTAL'):.4f}"
                  f"   (v-base TOTAL {seg(b, 'TOTAL'):.4f})")

        rows.append({
            "window": name,
            "short_today": seg(out["today"][3], "smooth/short"),
            "short_asof": seg(out["as-of"][3], "smooth/short"),
            "long_today": seg(out["today"][3], "smooth/long"),
            "long_asof": seg(out["as-of"][3], "smooth/long"),
            "total_today": seg(out["today"][3], "TOTAL"),
            "total_asof": seg(out["as-of"][3], "TOTAL"),
            "base_total_today": seg(out["today"][4], "TOTAL"),
            "base_total_asof": seg(out["as-of"][4], "TOTAL"),
        })

    d = pd.DataFrame(rows)
    print(f"\n{'=' * 74}\nv11 pooled WAPE, today's buckets versus as-of buckets\n")
    for c in ("short", "long", "total"):
        d[f"{c}_delta"] = d[f"{c}_asof"] - d[f"{c}_today"]
    print(d[["window", "short_today", "short_asof", "short_delta",
             "long_today", "long_asof", "long_delta",
             "total_today", "total_asof", "total_delta"]].round(4).to_string(index=False))

    print("\nv11's MARGIN over the structural baseline, which is what the project claims\n")
    d["margin_today"] = d["total_today"] - d["base_total_today"]
    d["margin_asof"] = d["total_asof"] - d["base_total_asof"]
    print(d[["window", "margin_today", "margin_asof"]].round(4).to_string(index=False))
    print("\nNegative is v11 ahead of the baseline. If the margin survives the as-of")
    print("population, the hindsight selection is not what the result rests on.")
    print("If it shrinks materially, it partly is, and that belongs in the writeup")
    print("ahead of any model version.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
