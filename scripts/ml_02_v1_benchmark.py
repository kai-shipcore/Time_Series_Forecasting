#!/usr/bin/env python3
"""ML experiment 02 — benchmark the V1 formula (legacy sheet math) on the
same splits as ml_00/ml_01.

Reuses the exact production V1 implementation from scripts/compare_v1.py
(6-window weighted velocity blend + dampening per West/East stream, flat
30-day rate for FBA, × 70 days × proportional seasonal modifier).

V1 predicts a 70-day TOTAL per SKU; our 10-week (=70-day) test windows line
up exactly, and evaluate.score() accepts one total row per SKU.

WA12 is shown alongside as the reference point — the known result to
reproduce is V1 ≈ 0.29 vs WA12 ≈ 0.22 pooled WAPE on smooth/short, with a
strong V1 negative bias.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from compare_v1 import build_cumsum_index, v1_forecast
from config import ML_DATA_SNAPSHOT
from src.ml.dataset import dev_splits, load_weekly
from src.ml.evaluate import score, score_table

# Prefer the pinned copy beside the snapshot inputs. data/processed is rewritten
# by the weekly ingest, so reading it there made this benchmark's V1 figures drift
# with the calendar while everything they were compared against stayed pinned.
_PINNED = ROOT / "data" / "snapshots" / ML_DATA_SNAPSHOT / "orders_raw.parquet"
RAW_PATH = _PINNED if _PINNED.exists() else ROOT / "data" / "processed" / "orders_raw.parquet"


def wa_forecast(train, test, window=12):
    level = (
        train.sort_values("ds").groupby("unique_id")["y"]
        .apply(lambda s: s.tail(window).mean()).rename("yhat").reset_index()
    )
    grid = test[["unique_id", "ds"]].drop_duplicates()
    return grid.merge(level, on="unique_id", how="left").fillna({"yhat": 0.0})


def v1_predictions(index: dict, split, skus: list[str]) -> pd.DataFrame:
    """One row per SKU: yhat = V1's 70-day total, ds = first test week.

    As-of is the cutoff itself, because a week labelled `ds` covers Tuesday
    `ds-6` through Monday `ds` (src/clean.py uses closed="right", label="right").
    So a cutoff label of 2026-05-04 means training history ends ON Monday
    2026-05-04, and the test span is Tuesday 2026-05-05 through Monday
    2026-07-13. v1_forecast treats its argument as the last day of available
    history and forecasts the following HORIZON_DAYS, so passing the cutoff
    aligns V1's 70 days exactly with the 70 days being scored.

    This read `cutoff - 1 day` until 2026-08-13, which was correct under the
    Monday-to-Sunday convention where a label covered [ds-7, ds-1]. That
    convention was reverted on 2026-08-06 (Section 4.30, BACKLOG 16) and this
    was not updated with it, so V1 was discarding one real day of history and
    forecasting a span shifted one day early. Measured on the development
    windows, the fix moves V1 by -0.011 to +0.017 pooled WAPE depending on
    segment and window, improving it in Mar-May and Dec-Feb and costing it in
    Oct-Dec. It had been systematically handicapping V1 in two windows of three.
    """
    asof = split.cutoff
    first_week = split.test["ds"].min()
    rows = [
        {"unique_id": uid, "ds": first_week,
         "yhat": v1_forecast(index, uid, asof)}
        for uid in skus
    ]
    return pd.DataFrame(rows)


def main():
    weekly, profiles = load_weekly()
    smooth = profiles.loc[profiles["bucket"] == "smooth", "unique_id"]
    weekly = weekly[weekly["unique_id"].isin(smooth)]

    raw = pd.read_parquet(RAW_PATH)
    raw["order_date"] = pd.to_datetime(raw["order_date"])
    raw = raw[raw["unique_id"].isin(set(smooth))]
    print(f"orders_raw: {len(raw):,} rows through {raw['order_date'].max().date()} "
          f"(smooth SKUs only)")
    index = build_cumsum_index(raw)

    for split in dev_splits(weekly, n=3):
        print(f"\n{'='*62}\n{split}")
        skus = sorted(split.train["unique_id"].unique())

        results = {
            "V1": score(v1_predictions(index, split, skus), split, profiles),
            "WA12": score(wa_forecast(split.train, split.test), split, profiles),
        }
        print("\npooled WAPE by segment:")
        print(score_table(results).to_string())
        print("\nbias% by segment:")
        bias = {n: t.set_index("segment")["bias_pct"] for n, t in results.items()}
        print(pd.DataFrame(bias).to_string())


if __name__ == "__main__":
    main()
