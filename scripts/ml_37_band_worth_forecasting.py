#!/usr/bin/env python3
"""Is a forecast worth making, by weekly demand band? Now that it is measurable.

    .venv/bin/python scripts/ml_37_band_worth_forecasting.py

No database. Development windows only; the final test window is not touched.
Requires ML_DATA_SNAPSHOT to point at a snapshot profiled with the onset fix.

Why this can be answered now and could not be before
----------------------------------------------------
The promotion override used to pin `train_start` to the start of the trailing
13-week window. That date sits in the future relative to every development
cutoff, so every promoted SKU had negative history at every window and was
silently excluded from scoring. 190 SKUs, 41% of the smooth set, and the ones
in the disputed 2-3 unit demand band were almost all of them.

An earlier attempt at this measurement (scripts/ml_35) had 8 and 4 SKU-windows
in those bands and was pure noise. With onset detection those SKUs carry real
historical history, so the bands populate: 87 SKU-windows between 2.0 and 3.0
units a week, against 116 above 3.0.

The question, and why it is not "which band has lower WAPE"
-----------------------------------------------------------
Absolute WAPE rises at low volume for arithmetic reasons. One unit of integer
noise is a 50% error on a SKU selling 2 a week and 2.5% on one selling 40, so
ranking bands by WAPE just rediscovers that small numbers are small.

What matters is whether the forecast beats what the business gets WITHOUT one.
An unforecast SKU appears in the Action List's Not-forecast section carrying a
trailing actual-sales rate. The structural baseline is a trailing 12-week mean,
which is that same alternative made scoreable. Where the model cannot beat it,
classifying the SKU as smooth buys nothing and implies a precision it does not
have.

The bootstrap is on the paired per-SKU difference within each band, because a
band of 87 SKU-windows can show a gap of 0.02 from sampling alone.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import ML_DATA_SNAPSHOT, ML_FINAL_TEST_CUTOFF  # noqa: E402
from src.ml.dataset import (dev_splits, load_weekly,  # noqa: E402
                            stratified_val_skus)
from src.ml.evaluate import per_sku_totals  # noqa: E402
from src.ml.model import (FEATURES_V1, FEATURES_V11_LONG, RatioLGBM,  # noqa: E402
                          long_sku_set, structural_baseline)

DEV_WINDOWS = ["Mar-May", "Dec-Feb", "Oct-Dec"]
BINS = [0, 2, 2.5, 3, 4, 6, 10, np.inf]
LABELS = ["<2", "2-2.5", "2.5-3", "3-4", "4-6", "6-10", "10+"]
N_BOOT = 2000


def main() -> int:
    print(f"snapshot {ML_DATA_SNAPSHOT}  (must be onset-profiled for this to mean anything)\n")
    weekly, profiles = load_weekly()
    smooth = set(profiles.loc[profiles["bucket"] == "smooth", "unique_id"])
    ws = weekly[weekly["unique_id"].isin(smooth)]

    parts = []
    for split, name in zip(dev_splits(ws, n=3), DEV_WINDOWS):
        longs = long_sku_set(profiles, split.cutoff) & set(split.train["unique_id"])
        val_all = stratified_val_skus(split.train, profiles)
        m = RatioLGBM(split.horizon, FEATURES_V1, deseas_features=True,
                      deseas_all=True).fit(split.train, profiles, split.cutoff, val_all)
        preds = m.predict(split.train, profiles, split.cutoff)
        if longs:
            val_long = stratified_val_skus(
                split.train[split.train["unique_id"].isin(longs)], profiles)
            mL = RatioLGBM(split.horizon, FEATURES_V11_LONG, deseas_features=True,
                           deseas_all=True, uids=longs).fit(
                split.train, profiles, split.cutoff, val_long)
            preds = pd.concat([preds[~preds["unique_id"].isin(longs)],
                               mL.predict(split.train, profiles, split.cutoff)],
                              ignore_index=True)
        base = structural_baseline(split.train, split.test, profiles, split.cutoff)

        mt = per_sku_totals(preds, split, profiles)[["unique_id", "ae", "y_total"]]
        bt = per_sku_totals(base, split, profiles)[["unique_id", "ae"]]
        d = mt.merge(bt, on="unique_id", suffixes=("_model", "_base"))
        recent = split.train[split.train["ds"] > split.cutoff - pd.Timedelta(weeks=13)]
        d["mean_weekly"] = d["unique_id"].map(recent.groupby("unique_id")["y"].mean())
        d["window"] = name
        parts.append(d)
        print(f"{name}: {len(d)} SKUs scored")

    d = pd.concat(parts, ignore_index=True)
    d["band"] = pd.cut(d["mean_weekly"], bins=BINS, labels=LABELS, right=False)

    rng = np.random.default_rng(0)
    rows = []
    for band, g in d.groupby("band", observed=True):
        ae_m, ae_b, y = (g["ae_model"].to_numpy(), g["ae_base"].to_numpy(),
                         g["y_total"].to_numpy())
        if y.sum() == 0:
            continue
        delta = ae_m.sum() / y.sum() - ae_b.sum() / y.sum()
        idx = rng.integers(0, len(g), size=(N_BOOT, len(g)))
        boots = (ae_m[idx].sum(1) - ae_b[idx].sum(1)) / y[idx].sum(1)
        se = float(boots.std())
        rows.append({
            "band": band, "SKU-windows": len(g), "units": int(y.sum()),
            "model": ae_m.sum() / y.sum(), "baseline": ae_b.sum() / y.sum(),
            "model - baseline": delta, "se": se,
            "verdict": ("model better" if delta < -2 * se else
                        "baseline better" if delta > 2 * se else
                        "indistinguishable"),
        })
    r = pd.DataFrame(rows)
    print("\npooled WAPE by trailing weekly demand, three development windows\n")
    print(r.round(4).to_string(index=False))
    print("\n'model - baseline' negative means the forecast beats a trailing average.")
    print("'indistinguishable' means the band cannot tell them apart at two standard")
    print("errors, which for a decision about whether to forecast at all is itself")
    print("an answer: a forecast that cannot be shown to beat a trailing mean is not")
    print("worth the precision it implies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
