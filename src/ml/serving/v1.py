"""V1 spreadsheet forecast, computed fresh and decomposed onto the model's
weekly forward grid, plus V1's accuracy baseline on the development windows.

V1's daily rate is a snapshot at one as-of date; only the seasonal modifier
varies week to week across the horizon. This module never reimplements the V1
formula itself (compare_v1.v1_daily_current / proportional_seasonal_modifier /
v1_forecast) -- it only calls it and reshapes the result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=True)

from compare_v1 import build_cumsum_index, load_raw, proportional_seasonal_modifier, v1_daily_current, v1_forecast  # noqa: E402

from src.ml.dataset import dev_splits  # noqa: E402
from src.ml.evaluate import per_sku_totals, score  # noqa: E402


def v1_forward(grid: pd.DataFrame, refresh: bool = True) -> pd.DataFrame:
    """V1 forecast on the same (unique_id, ds) grid as the model forecast.

    `grid` needs unique_id, ds, week_of (the model forward-forecast
    table works directly). V1's daily rate is computed once per SKU, at
    week_of - 1 day; each grid week gets that rate x 7 x that week's own
    seasonal modifier, since a 13-week horizon spans multiple months. SKUs
    absent from the velocity pull are dropped (no row), not zero-filled.
    """
    raw = load_raw(refresh=refresh)
    index = build_cumsum_index(raw)
    available = {uid for uid, _stream in index.keys()}

    week_of = pd.Timestamp(grid["week_of"].iloc[0])
    asof = week_of - pd.Timedelta(days=1)

    rows = []
    for uid, uid_grid in grid.groupby("unique_id"):
        if uid not in available:
            continue
        daily = v1_daily_current(index, uid, asof)
        for ds in sorted(pd.to_datetime(uid_grid["ds"]).unique()):
            ds = pd.Timestamp(ds)
            modifier = proportional_seasonal_modifier(ds - pd.Timedelta(days=6), ds)
            rows.append({
                "unique_id": uid,
                "week_of": week_of,
                "ds": ds,
                "v1_yhat": daily * 7 * modifier,
            })
    return pd.DataFrame(rows, columns=["unique_id", "week_of", "ds", "v1_yhat"])


def validate_v1(n_windows: int = 3, weekly=None, profiles=None) -> pd.DataFrame:
    """Score V1 on the development windows, same as-of rule as ml_02_v1_benchmark.

    One 70-day-total row per SKU at cutoff - 1 day (v1_forecast's native
    horizon), scored through the shared harness. Returns one row per
    (window, segment) in the same shape as serving.forecast.validate_version,
    with model_version="v1". Never touches the quarantined final test window
    (dev_splits excludes it by construction).
    """
    if weekly is None or profiles is None:
        from src.ml.dataset import load_weekly
        weekly, profiles = load_weekly()
    smooth = set(profiles.loc[profiles["bucket"] == "smooth", "unique_id"])
    w = weekly[weekly["unique_id"].isin(smooth)]

    raw = load_raw(refresh=False)
    index = build_cumsum_index(raw)

    rows = []
    for split in dev_splits(w, n=n_windows):
        skus = sorted(split.train["unique_id"].unique())
        asof = split.cutoff - pd.Timedelta(days=1)
        first_week = split.test["ds"].min()
        preds = pd.DataFrame([
            {"unique_id": uid, "ds": first_week, "yhat": v1_forecast(index, uid, asof)}
            for uid in skus
        ])
        label = f"{split.test['ds'].min():%b}-{split.test['ds'].max():%b}"
        tbl = score(preds, split, profiles)
        for _, r in tbl.iterrows():
            rows.append({
                "model_version": "v1",
                "window": label,
                "cutoff": split.cutoff.date().isoformat(),
                "segment": r["segment"],
                "n_skus": r["n_skus"],
                "actual_units": r["actual_units"],
                "pooled_wape": r["pooled_wape"],
                "bias_pct": r["bias_pct"],
            })
    return pd.DataFrame(rows)


def validate_v1_detail(n_windows: int = 3, weekly=None, profiles=None) -> pd.DataFrame:
    """Per-SKU detail behind validate_v1(): one row per SKU per window, before
    segment aggregation. Same V1 predictions and per-SKU totals
    (src.ml.evaluate.per_sku_totals) validate_v1() aggregates, so summing
    these rows by segment reproduces its numbers exactly. Never touches the
    quarantined final test window.

    Returns unique_id, model_version, window, cutoff, bucket, history_length,
    yhat_total, y_total, ae, bias.
    """
    if weekly is None or profiles is None:
        from src.ml.dataset import load_weekly
        weekly, profiles = load_weekly()
    smooth = set(profiles.loc[profiles["bucket"] == "smooth", "unique_id"])
    w = weekly[weekly["unique_id"].isin(smooth)]

    raw = load_raw(refresh=False)
    index = build_cumsum_index(raw)

    parts = []
    for split in dev_splits(w, n=n_windows):
        skus = sorted(split.train["unique_id"].unique())
        asof = split.cutoff - pd.Timedelta(days=1)
        first_week = split.test["ds"].min()
        preds = pd.DataFrame([
            {"unique_id": uid, "ds": first_week, "yhat": v1_forecast(index, uid, asof)}
            for uid in skus
        ])
        label = f"{split.test['ds'].min():%b}-{split.test['ds'].max():%b}"
        detail = per_sku_totals(preds, split, profiles)
        detail["model_version"] = "v1"
        detail["window"] = label
        detail["cutoff"] = split.cutoff.date().isoformat()
        parts.append(detail)
    out = pd.concat(parts, ignore_index=True)
    return out[["unique_id", "model_version", "window", "cutoff", "bucket",
                "history_length", "yhat_total", "y_total", "ae", "bias"]]
