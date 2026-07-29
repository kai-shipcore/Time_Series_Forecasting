"""Accumulating record of what the model predicted, and what happened.

Every forward run writes its horizon to `data/processed/ml_forward_forecasts.parquet`
and overwrites the previous one. That file answers "what is the current
forecast". It cannot answer "is the model getting better", because the evidence
needed for that is discarded weekly.

This module keeps the evidence. Each run appends, so the store grows into a
record of every prediction the project has served, scoreable against actuals as
the weeks it covered complete.

Deliberately not named after a model version. The version lives in a column, so
a new one coexists with its predecessors rather than replacing them, and the
comparison between them is a query rather than an archaeology exercise. Nothing
here knows or cares which version is current.

Storage is a single parquet file for now, matching the rest of the ML track
while the model is still moving. The read and write paths are the only places
that know that, so moving to a `shipcore` table later is a change to two
functions rather than to every caller. That move is the prerequisite for serving
this from an API that does not share the filesystem.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
HISTORY_PATH = REPO_ROOT / "data" / "processed" / "ml_forecast_history.parquet"

#: A row is identified by which model made the prediction, when it made it, for
#: which SKU, and for which target week. Re-running the same version on the same
#: date replaces those rows rather than adding a second copy, which mirrors the
#: production pipeline's own rule that a re-run within a training week replaces.
KEY = ["model_version", "forecast_date", "unique_id", "ds"]

COLUMNS = KEY + ["yhat", "bucket", "history_length", "segment", "served_by", "run_at"]


def load() -> pd.DataFrame:
    """Every stored prediction. Empty frame with the right columns if none yet."""
    if not HISTORY_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_parquet(HISTORY_PATH)
    for col in ("forecast_date", "ds"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def append(forecast: pd.DataFrame) -> dict:
    """Add a run to the history, replacing any rows sharing its key.

    Returns a small summary so the caller can report what happened rather than
    claiming success blindly: how many rows arrived, how many replaced earlier
    ones, and how large the store now is.

    Read-modify-write on one file is safe here because the pipeline is a weekly
    cron with a single writer. It would not be safe under concurrent runs, which
    is one of the reasons this belongs in a database eventually.
    """
    if forecast is None or forecast.empty:
        return {"added": 0, "replaced": 0, "total": len(load()), "runs": 0}

    incoming = forecast.copy()
    for col in ("forecast_date", "ds"):
        incoming[col] = pd.to_datetime(incoming[col])
    missing = [c for c in KEY if c not in incoming.columns]
    if missing:
        raise ValueError(f"forecast is missing key columns: {missing}")
    keep = [c for c in COLUMNS if c in incoming.columns]
    incoming = incoming[keep]

    existing = load()
    replaced = 0
    if not existing.empty:
        # Anti-join on the key: drop what this run supersedes, keep the rest.
        marker = existing.merge(
            incoming[KEY].drop_duplicates(), on=KEY, how="left", indicator=True
        )["_merge"]
        replaced = int((marker == "both").sum())
        existing = existing[marker.to_numpy() == "left_only"]

    # Concatenating onto an empty frame is deprecated in pandas and changes the
    # resulting dtypes, so the first write skips it rather than relying on
    # behaviour that is going away.
    out = incoming if existing.empty else pd.concat([existing, incoming], ignore_index=True)
    out = out.sort_values(["model_version", "forecast_date", "unique_id", "ds"])
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(HISTORY_PATH, index=False)

    return {
        "added": len(incoming),
        "replaced": replaced,
        "total": len(out),
        "runs": int(out.groupby(["model_version", "forecast_date"]).ngroups),
    }


def runs() -> pd.DataFrame:
    """One row per stored run: version, date, SKUs covered, weeks, units."""
    df = load()
    if df.empty:
        return pd.DataFrame(
            columns=["model_version", "forecast_date", "n_skus", "n_weeks", "forecast_units"]
        )
    return (
        df.groupby(["model_version", "forecast_date"], as_index=False)
        .agg(
            n_skus=("unique_id", "nunique"),
            n_weeks=("ds", "nunique"),
            forecast_units=("yhat", "sum"),
        )
        .sort_values(["model_version", "forecast_date"])
        .reset_index(drop=True)
    )


def last_complete_week(today: pd.Timestamp | None = None) -> pd.Timestamp:
    """The most recent week that has finished, as its W-MON label.

    Weeks are labelled by the Monday they END on, so the week labelled
    2026-07-27 ran Mon 21 July to Sun 26 July. The latest such label at or
    before today is the last complete week; if today is itself a Monday the week
    it labels only starts today, so the answer is the Monday before.

    Derived from the calendar rather than from `max(ds)` in the sales file. Those
    usually agree, but not always: an ingest running mid-week can emit a partial
    week under next Monday's label, and scoring that would read two days of
    orders as a full week of demand and call the model wildly over-forecast.
    Trusting the calendar makes the boundary independent of when the pipeline
    happened to run.
    """
    today = pd.Timestamp(today or pd.Timestamp.today().normalize()).normalize()
    days_since_monday = today.dayofweek  # Monday == 0
    return today - pd.Timedelta(days=days_since_monday or 7)


def score_against_actuals(sales: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join stored predictions to what actually sold, where the week has settled.

    Returns one row per model version, run, SKU and target week, with `yhat`,
    `y`, and the lead in weeks between the run and the week it predicted.

    Scores every week up to and including the last complete one, and no further.
    A week still running is unresolved rather than wrong, and counting it would
    read a partial week of orders as a full week of demand, making the newest
    run look like it wildly over-forecast. See `last_complete_week`.

    Knows nothing about model versions, so a new one is scored the moment its
    first run lands without this function changing.
    """
    hist = load()
    if hist.empty:
        return pd.DataFrame(
            columns=["model_version", "forecast_date", "unique_id", "ds", "lead", "yhat", "y"]
        )
    if sales is None:
        from src.planning import data as D

        sales = D.load_sales()
    if sales.empty:
        return pd.DataFrame(
            columns=["model_version", "forecast_date", "unique_id", "ds", "lead", "yhat", "y"]
        )

    # Bounded by both: the calendar says which weeks have finished, the data
    # says which of those actually arrived. Taking the earlier of the two means
    # a late ingest cannot cause a week to be scored against absent actuals.
    settled_through = min(last_complete_week(), pd.to_datetime(sales["ds"]).max())
    closed = hist[hist["ds"] <= settled_through].copy()
    if closed.empty:
        return pd.DataFrame(
            columns=["model_version", "forecast_date", "unique_id", "ds", "lead", "yhat", "y"]
        )

    actual = sales[["unique_id", "ds", "y"]].copy()
    actual["ds"] = pd.to_datetime(actual["ds"])
    out = closed.merge(actual, on=["unique_id", "ds"], how="left")
    # A SKU-week absent from the sales grid sold nothing. The grid is complete
    # per the ingestion contract, so this is a true zero rather than a gap.
    out["y"] = out["y"].fillna(0.0)
    out["lead"] = ((out["ds"] - out["forecast_date"]).dt.days / 7).round().astype(int)
    return out[
        ["model_version", "forecast_date", "unique_id", "ds", "lead", "yhat", "y",
         "bucket", "history_length", "segment"]
    ].reset_index(drop=True)


def performance_by_run(sales: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pooled WAPE per run per segment, over the weeks that have closed.

    Pooled rather than averaged per SKU, matching the project's headline metric
    (design doc Section 1.3): errors summed across SKUs before dividing, so
    heavier-demand SKUs count more.

    `weeks_scored` is reported alongside, because a run whose horizon is mostly
    still open is measured on very little and its error should not be read with
    the same confidence as a fully closed one.

    `bias_pct` is in percentage points, matching `src/ml/evaluate.py`, so +1.0
    means the run over-forecast by one percent. The two functions feed the same
    API payload, and the same field name carrying a fraction in one place and a
    percentage in the other is a hundredfold error waiting for a reader who does
    not check which produced the row.
    """
    scored = score_against_actuals(sales)
    if scored.empty:
        return pd.DataFrame(
            columns=["model_version", "forecast_date", "segment", "n_skus",
                     "weeks_scored", "actual_units", "pooled_wape", "bias_pct"]
        )

    def agg(g: pd.DataFrame) -> pd.Series:
        y = g["y"].sum()
        err = (g["yhat"] - g["y"]).abs().sum()
        bias = (g["yhat"] - g["y"]).sum()
        return pd.Series({
            "n_skus": g["unique_id"].nunique(),
            "weeks_scored": g["ds"].nunique(),
            "actual_units": y,
            "pooled_wape": err / y if y else float("nan"),
            "bias_pct": round(bias / y * 100, 1) if y else float("nan"),
        })

    per_segment = (
        scored.groupby(["model_version", "forecast_date", "segment"])
        .apply(agg, include_groups=False)
        .reset_index()
    )
    total = (
        scored.assign(segment="TOTAL")
        .groupby(["model_version", "forecast_date", "segment"])
        .apply(agg, include_groups=False)
        .reset_index()
    )
    return (
        pd.concat([per_segment, total], ignore_index=True)
        .sort_values(["model_version", "forecast_date", "segment"])
        .reset_index(drop=True)
    )
