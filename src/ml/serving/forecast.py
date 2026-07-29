"""Forward forecasting and validation for the servable model versions.

forward_forecast() trains the chosen version on all available history and
predicts a horizon of future weeks, returning a tidy per-SKU table for the
dashboard. validate_version() re-scores a version on the development windows and
returns per-window, per-segment pooled WAPE; it is both the reproduction check
for an existing version and the "is my new version better" tool for a new one.

Both read the pinned ML snapshot by default (config.ML_DATA_SNAPSHOT) so results
are reproducible. Pass snapshot=None to forecast from the live weekly refresh.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from config import MIN_SIM_HISTORY_WEEKS, ML_DATA_SNAPSHOT
from src.ml.dataset import asof_history_length, dev_splits, eligible_skus, load_weekly
from src.ml.evaluate import per_sku_totals, score
from src.ml.serving.models import CURRENT_BEST, get_model

FORWARD_COLUMNS = [
    "unique_id", "forecast_date", "ds", "yhat", "bucket",
    "history_length", "segment", "model_version", "served_by", "run_at",
]

_UNSET = object()


def _smooth_only(weekly: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """Only smooth SKUs are modeled; intermittent SKUs get no forecast."""
    smooth = set(profiles.loc[profiles["bucket"] == "smooth", "unique_id"])
    return weekly[weekly["unique_id"].isin(smooth)].copy()


def _load(snapshot, weekly, profiles):
    if weekly is None or profiles is None:
        kwargs = {} if snapshot is _UNSET else {"snapshot": snapshot}
        weekly, profiles = load_weekly(**kwargs)
    return weekly, profiles


def forward_forecast(
    version: str | None = None,
    horizon: int = 13,
    snapshot=_UNSET,
    weekly: pd.DataFrame | None = None,
    profiles: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, object]:
    """Train `version` on all history and forecast `horizon` weeks forward.

    Returns (forecast_table, fitted_model). The table has one row per SKU per
    forecast week with the columns in FORWARD_COLUMNS.
    """
    version = version or CURRENT_BEST
    weekly, profiles = _load(snapshot, weekly, profiles)
    w = _smooth_only(weekly, profiles)
    cutoff = w["ds"].max()

    model = get_model(version, horizon).fit(w, profiles, cutoff)
    preds = model.predict(w, profiles, cutoff)  # unique_id / ds / yhat

    asof = asof_history_length(profiles, cutoff)
    seg = asof.astype("object").replace({"medium": "long", "full": "long"})
    run_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    out = preds.copy()
    out["forecast_date"] = pd.Timestamp(cutoff)
    out["bucket"] = "smooth"
    out["history_length"] = out["unique_id"].map(asof).astype("object")
    out["segment"] = out["unique_id"].map(seg)
    out["model_version"] = version
    out["served_by"] = [model.served_by(u) for u in out["unique_id"]]
    out["run_at"] = run_at
    return out[FORWARD_COLUMNS].sort_values(["unique_id", "ds"]).reset_index(drop=True), model


def validate_version(
    version: str | None = None,
    n_windows: int = 3,
    snapshot=_UNSET,
    weekly: pd.DataFrame | None = None,
    profiles: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Re-score `version` on the development windows.

    Returns one row per (window, segment) with pooled WAPE and bias, for the
    reproduction check and for the dashboard's accuracy view. Never touches the
    quarantined final test window.
    """
    version = version or CURRENT_BEST
    weekly, profiles = _load(snapshot, weekly, profiles)
    w = _smooth_only(weekly, profiles)

    rows = []
    for split in dev_splits(w, n=n_windows):
        model = get_model(version, split.horizon).fit(split.train, profiles, split.cutoff)
        preds = model.predict(split.train, profiles, split.cutoff)
        label = f"{split.test['ds'].min():%b}-{split.test['ds'].max():%b}"
        tbl = score(preds, split, profiles)
        for _, r in tbl.iterrows():
            rows.append({
                "model_version": version,
                "window": label,
                "cutoff": split.cutoff.date().isoformat(),
                "segment": r["segment"],
                "n_skus": r["n_skus"],
                "actual_units": r["actual_units"],
                "pooled_wape": r["pooled_wape"],
                "bias_pct": r["bias_pct"],
            })
    return pd.DataFrame(rows)


def validate_version_detail(
    version: str | None = None,
    n_windows: int = 3,
    snapshot=_UNSET,
    weekly: pd.DataFrame | None = None,
    profiles: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-SKU detail behind validate_version(): one row per SKU per window,
    before segment aggregation. Uses the same fit/predict loop and the same
    per-SKU totals (src.ml.evaluate.per_sku_totals) validate_version()
    aggregates, so summing these rows by segment reproduces its numbers
    exactly. Never touches the quarantined final test window.

    Returns unique_id, model_version, window, cutoff, bucket, history_length,
    yhat_total, y_total, ae, bias.
    """
    version = version or CURRENT_BEST
    weekly, profiles = _load(snapshot, weekly, profiles)
    w = _smooth_only(weekly, profiles)

    parts = []
    for split in dev_splits(w, n=n_windows):
        model = get_model(version, split.horizon).fit(split.train, profiles, split.cutoff)
        preds = model.predict(split.train, profiles, split.cutoff)
        label = f"{split.test['ds'].min():%b}-{split.test['ds'].max():%b}"
        detail = per_sku_totals(preds, split, profiles)
        detail["model_version"] = version
        detail["window"] = label
        detail["cutoff"] = split.cutoff.date().isoformat()
        parts.append(detail)
    out = pd.concat(parts, ignore_index=True)
    return out[["unique_id", "model_version", "window", "cutoff", "bucket",
                "history_length", "yhat_total", "y_total", "ae", "bias"]]


def validate_version_weekly(
    version: str | None = None,
    n_windows: int = 3,
    snapshot=_UNSET,
    weekly: pd.DataFrame | None = None,
    profiles: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-week detail behind validate_version_detail(): one row per SKU per
    test week, before the window totals are formed.

    Same fit/predict loop and the same eligibility filter
    (dataset.eligible_skus at MIN_SIM_HISTORY_WEEKS) that
    src.ml.evaluate.per_sku_totals applies, so summing `yhat` over a
    (unique_id, window) reproduces that function's `yhat_total` exactly and the
    same for `y`. That equality is the contract: this function must never become
    a second, subtly different measurement path. Section 2.4 of the design doc
    requires one source of truth for measurement, and this is a view of it, not
    an alternative to it.

    Exists because the stored accuracy report keeps window totals only, which is
    enough to say how large a miss was but not when inside the window it
    happened. The dashboard's backtest chart needs the shape.

    Eligible SKUs with no prediction are emitted with yhat = 0, matching how
    per_sku_totals scores them, rather than being dropped.

    Returns unique_id, model_version, window, cutoff, ds, lead (1-based week
    index into the window), yhat, y. Never touches the quarantined final test
    window.
    """
    version = version or CURRENT_BEST
    weekly, profiles = _load(snapshot, weekly, profiles)
    w = _smooth_only(weekly, profiles)

    parts = []
    for split in dev_splits(w, n=n_windows):
        model = get_model(version, split.horizon).fit(split.train, profiles, split.cutoff)
        preds = model.predict(split.train, profiles, split.cutoff)
        label = f"{split.test['ds'].min():%b}-{split.test['ds'].max():%b}"

        keep = eligible_skus(profiles, split.cutoff, MIN_SIM_HISTORY_WEEKS)
        test = split.test[split.test["unique_id"].isin(keep)]
        preds = preds[preds["unique_id"].isin(keep)]

        # Outer join on the actuals: an eligible SKU-week with no prediction is
        # scored as zero upstream, so it must appear here as zero too, or the
        # totals would not reconcile.
        detail = test[["unique_id", "ds", "y"]].merge(
            preds[["unique_id", "ds", "yhat"]], on=["unique_id", "ds"], how="left"
        )
        detail["yhat"] = detail["yhat"].fillna(0.0)

        weeks = sorted(test["ds"].unique())
        lead = {d: i + 1 for i, d in enumerate(weeks)}
        detail["lead"] = detail["ds"].map(lead).astype(int)
        detail["model_version"] = version
        detail["window"] = label
        detail["cutoff"] = split.cutoff.date().isoformat()
        parts.append(detail)

    out = pd.concat(parts, ignore_index=True)
    return out[["unique_id", "model_version", "window", "cutoff", "ds",
                "lead", "yhat", "y"]].sort_values(
        ["unique_id", "cutoff", "lead"]).reset_index(drop=True)
