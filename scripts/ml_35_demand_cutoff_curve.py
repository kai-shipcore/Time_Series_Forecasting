#!/usr/bin/env python3
"""Where does forecasting stop being worth it, as a function of weekly demand?

    .venv/bin/python scripts/ml_35_demand_cutoff_curve.py

No database. Reads the pinned snapshot. Writes nothing outside a temp directory.
Uses the three DEVELOPMENT windows only; the final test window is not touched.

The question this exists to settle
----------------------------------
`classify()` sends a SKU to `intermittent` when its mean is below
MEAN_INTERMITTENT_CUTOFF, currently 3.0 units a week, regardless of how regular
it is. The promotion override then readmits some of them at a mean of 2.0. Two
different bars for the same judgement, and a SKU can fail one and pass the other.

Reconciling them needs a number, not an argument. 3.0 and 2.0 were both chosen
rather than measured, so this measures the thing they are proxies for: at what
weekly demand does a forecast stop beating the trivial alternative?

Why "beats the baseline" and not "WAPE below X"
-----------------------------------------------
Absolute WAPE rises at low volume for reasons that have nothing to do with model
quality. One unit of integer noise on a SKU selling 2 a week is a 50% error and
on one selling 40 a week is 2.5%, so any fixed WAPE threshold is really a demand
threshold wearing a disguise, which is the assumption under test.

The honest comparison is against what the business gets WITHOUT a forecast. An
unforecast SKU appears in the Action List's Not-forecast section with a trailing
actual-sales rate. The structural baseline is a trailing 12-week mean, so it is
that same alternative made scoreable. Where the model cannot beat it, classifying
the SKU as smooth adds a forecast that is no better than the rate already shown,
while implying more precision than it has.

Method
------
Classification is widened to every SKU with zero_pct < ZERO_PCT_INTERMITTENT,
with NO mean test at all, so the low-demand bands are populated and can be
measured instead of assumed. Promotion and demotion are disabled while profiling
(both bars pushed out of range), so `train_start`, `active_weeks` and
`history_length` are the honest pre-override values and the 13-week truncation
cannot confound the bands it most affects.

v11 is then fitted per window on that widened population, exactly as ml_22 builds
it, and scored per SKU. SKUs are binned by their trailing 13-week mean AT THE
CUTOFF, because that is the quantity a threshold would actually be applied to at
run time.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import src.profile as P  # noqa: E402

P.PROCESSED_DIR = pathlib.Path(tempfile.mkdtemp())
# Disable both overrides so profiling returns pre-override values. Promotion is
# what truncates a SKU to 13 weeks, and it fires hardest in exactly the demand
# bands under study, so leaving it on would measure the bug rather than the
# question.
P.RECENT_MEAN_UPGRADE = float("inf")
P.RECENT_MEAN_DOWNGRADE = 0.0

from config import (ML_DATA_SNAPSHOT, ML_FINAL_TEST_CUTOFF,  # noqa: E402
                    TEST_WEEKS, ZERO_PCT_INTERMITTENT)
from src.ml.dataset import Split, data_dir, stratified_val_skus  # noqa: E402
from src.ml.evaluate import per_sku_totals  # noqa: E402
from src.ml.model import (FEATURES_V1, FEATURES_V11_LONG, RatioLGBM,  # noqa: E402
                          long_sku_set, structural_baseline)

# Development windows only. The final test window is quarantined and is not
# evaluated here; it is not in this list and must not be added to it.
DEV_WINDOWS = ["Mar-May", "Dec-Feb", "Oct-Dec"]
BINS = [0, 1, 2, 2.5, 3, 4, 6, 10, np.inf]
LABELS = ["<1", "1-2", "2-2.5", "2.5-3", "3-4", "4-6", "6-10", "10+"]


def main() -> int:
    src = data_dir(ML_DATA_SNAPSHOT)
    raw = pd.read_parquet(src / "sales_clean.parquet")
    raw["ds"] = pd.to_datetime(raw["ds"])
    weeks = sorted(raw["ds"].unique())
    anchor = weeks.index(pd.Timestamp(ML_FINAL_TEST_CUTOFF))

    parts = []
    for i, name in enumerate(DEV_WINDOWS, start=1):
        ci = anchor - i * TEST_WEEKS
        cutoff = weeks[ci]
        test_weeks = weeks[ci + 1: ci + 1 + TEST_WEEKS]

        prof = P.profile(raw[raw["ds"] <= cutoff].copy())
        prof["train_start"] = pd.to_datetime(prof["train_start"])
        # The widening: regular enough, with no mean test whatsoever.
        prof["bucket"] = np.where(
            prof["zero_pct"] < ZERO_PCT_INTERMITTENT, "smooth", "intermittent")
        smooth = set(prof.loc[prof["bucket"] == "smooth", "unique_id"])

        w = raw[raw["unique_id"].isin(smooth)].copy()
        starts = prof.set_index("unique_id")["train_start"]
        w["_ts"] = w["unique_id"].map(starts)
        w = w[w["_ts"].isna() | (w["ds"] >= w["_ts"])].drop(columns="_ts")
        split = Split(
            cutoff=cutoff,
            train=w[w["ds"] <= cutoff].sort_values(["unique_id", "ds"]).reset_index(drop=True),
            # Test restricted to the scored population. Leaving the full catalogue
            # here would score every intermittent SKU as a zero forecast and make
            # every pooled figure meaningless.
            test=raw[raw["ds"].isin(test_weeks) & raw["unique_id"].isin(smooth)].copy(),
            horizon=TEST_WEEKS,
        )

        longs = long_sku_set(prof, cutoff) & set(split.train["unique_id"])
        val_all = stratified_val_skus(split.train, prof)
        m = RatioLGBM(split.horizon, FEATURES_V1, deseas_features=True,
                      deseas_all=True).fit(split.train, prof, cutoff, val_all)
        preds = m.predict(split.train, prof, cutoff)
        if longs:
            val_long = stratified_val_skus(
                split.train[split.train["unique_id"].isin(longs)], prof)
            mL = RatioLGBM(split.horizon, FEATURES_V11_LONG, deseas_features=True,
                           deseas_all=True, uids=longs).fit(
                split.train, prof, cutoff, val_long)
            preds = pd.concat([preds[~preds["unique_id"].isin(longs)],
                               mL.predict(split.train, prof, cutoff)], ignore_index=True)
        base = structural_baseline(split.train, split.test, prof, cutoff)

        mt = per_sku_totals(preds, split, prof)[["unique_id", "ae", "y_total"]]
        bt = per_sku_totals(base, split, prof)[["unique_id", "ae", "y_total"]]
        d = mt.merge(bt, on="unique_id", suffixes=("_model", "_base"))

        # The quantity a threshold would be applied to at run time.
        recent = split.train[split.train["ds"] > cutoff - pd.Timedelta(weeks=13)]
        d["mean_weekly"] = d["unique_id"].map(recent.groupby("unique_id")["y"].mean())
        d["window"] = name
        parts.append(d)
        print(f"{name}: {len(smooth)} SKUs classified smooth by regularity alone, "
              f"{len(d)} scored")

    d = pd.concat(parts, ignore_index=True)
    d["band"] = pd.cut(d["mean_weekly"], bins=BINS, labels=LABELS, right=False)

    g = d.groupby("band", observed=True).apply(lambda x: pd.Series({
        "SKU-windows": len(x),
        "units": x["y_total_model"].sum(),
        "model": x["ae_model"].sum() / max(x["y_total_model"].sum(), 1),
        "baseline": x["ae_base"].sum() / max(x["y_total_base"].sum(), 1),
    }), include_groups=False)
    g["model beats baseline by"] = g["baseline"] - g["model"]

    print("\npooled WAPE by trailing weekly demand, three development windows combined\n")
    print(g.round(4).to_string())
    print("\nPositive in the last column means the model is better than a trailing")
    print("average. Where it is near zero or negative, a forecast adds nothing over")
    print("the rate the Not-forecast section already shows, and the cutoff belongs")
    print("at or above that band.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
