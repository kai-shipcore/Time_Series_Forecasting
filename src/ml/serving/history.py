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

Storage was a single parquet file, and that move to a `shipcore` table has now
happened: it was a change to `load` and `append` only, as this note predicted.
See `src/ml/serving/store.py` for why, in short that one file was both
irreplaceable and readable from exactly one machine.

Both stores are written. Reads prefer the table and fall back to the file, so a
machine with credentials sees the server's runs while a clone without any still
works from its own. The parquet is therefore still a real backup rather than a
vestige, and nothing here requires a database to function.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Moved to src/weeks so the ingest can apply the same rule. Re-exported here
# because callers of this module import it by name.
from src.weeks import last_complete_week  # noqa: F401

from . import store

REPO_ROOT = Path(__file__).resolve().parents[3]
HISTORY_PATH = REPO_ROOT / "data" / "processed" / "ml_forecast_history.parquet"

#: A row is identified by which model made the prediction, when it made it, for
#: which SKU, and for which target week. Re-running the same version on the same
#: date replaces those rows rather than adding a second copy, which mirrors the
#: production pipeline's own rule that a re-run within a training week replaces.
KEY = ["model_version", "week_of", "unique_id", "ds"]

COLUMNS = KEY + ["yhat", "bucket", "history_length", "segment", "served_by", "run_at"]


def canonical_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Put stored rows into the one segment vocabulary the project reports in.

    Two corrections, both applied on read rather than trusted from the writer.

    Intermittent rows are dropped. The model forecasts smooth SKUs and nothing
    else: serving/forecast.py filters to ``bucket == "smooth"`` and writes that
    literal, so a real run cannot produce another bucket. A stored row saying
    otherwise is bad data, and reporting it shows the model apparently
    predicting the SKUs it declines to predict.

    ``medium`` and ``full`` collapse to ``long``, matching src/ml/evaluate.py
    and design doc Section 4.4. Without this the performance table reports
    smooth/full and smooth/medium directly underneath a comparison grid that
    reports smooth/long, on the same page, for the same model.

    Where the bad rows come from: seed_forecast_history.py stamped the current
    profile onto fabricated historical runs, so SKUs demoted since the seed date
    arrived labelled intermittent, and it wrote raw history_length rather than
    the collapsed one. That script is fixed and real runs supersede the fixture,
    but the normalisation lives here rather than in the fixture because the
    store is written by a pipeline and read by several endpoints, and every one
    of those readers was otherwise free to disagree about what a segment is.

    Applied in load(), so a consumer cannot reach the store without it.
    """
    if df.empty:
        return df
    out = df
    if "bucket" in out.columns:
        out = out[out["bucket"].fillna("?") == "smooth"].copy()
    if "history_length" in out.columns:
        out["history_length"] = out["history_length"].replace(
            {"medium": "long", "full": "long"}
        )
    # Rebuilt from the corrected parts rather than string-patched, so segment
    # cannot survive as a stale concatenation of labels that have since moved.
    if "segment" in out.columns and {"bucket", "history_length"} <= set(out.columns):
        out["segment"] = out["bucket"].fillna("?") + "/" + out["history_length"].fillna("?")
    return out.reset_index(drop=True)


def _legacy_column_names(df):
    """Accept parquets written before the 2026-08-12 rename.

    Files on disk still carry `forecast_date`, which in the ML track always held
    the training week. Renamed on read rather than by rewriting the files: the
    dev_seed fixture is tracked in git and checksummed by its manifest, and the
    local history parquet is a backup nobody should have to migrate by hand.
    """
    if "forecast_date" in df.columns and "week_of" not in df.columns:
        df = df.rename(columns={"forecast_date": "week_of"})
    return df


def load() -> pd.DataFrame:
    """Every stored prediction. Empty frame with the right columns if none yet.

    The table wins when it can be read, because it is the one the weekly run on
    the server writes and therefore the only complete record. The parquet is
    whatever this machine happened to produce locally, which on a developer's
    laptop is a different and usually shorter history.

    Falls back rather than failing: a clone with no credentials is a supported
    way to run this project, and the seeded fixture is expected to work alone.

    Segment labels are normalised on the way out; see canonical_segments().
    """
    from_db = store.read(COLUMNS)
    if from_db is not None and not from_db.empty:
        return canonical_segments(from_db)

    if not HISTORY_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = _legacy_column_names(pd.read_parquet(HISTORY_PATH))
    for col in ("week_of", "ds"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return canonical_segments(df)


def _load_parquet() -> pd.DataFrame:
    """The local file alone, ignoring the table.

    `append` needs this. Merging incoming rows against the table and then
    writing the result to the parquet would copy the server's entire history
    into a laptop's file the first time anyone ran a forecast with credentials
    set, which is a surprising side effect of writing one run.
    """
    if not HISTORY_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = _legacy_column_names(pd.read_parquet(HISTORY_PATH))
    for col in ("week_of", "ds"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def append(forecast: pd.DataFrame) -> dict:
    """Add a run to the history, replacing any rows sharing its key.

    Returns a small summary so the caller can report what happened rather than
    claiming success blindly: how many rows arrived, how many replaced earlier
    ones, and how large the store now is.

    Writes both stores. The table is the durable, shared one; the parquet stays
    as a local backup and as the path a machine without credentials uses.

    Both writes assume a single writer, which the weekly cron provides. The
    table write clears this (model_version, week_of) before inserting, so two
    runs racing on the same week would have the second delete the first's rows
    mid-flight; the file write is read-modify-write and no safer. This was
    previously a plain upsert and was described here as safe under concurrent
    runs, which is no longer true and was never something the pipeline relied
    on. The reason for the change is in `store.upsert`: keying on unique_id
    could not remove SKUs a re-segmented run stopped producing.

    A failed table write does not fail the run. The forecast itself has already
    been produced and the parquet has it, so raising here would turn a
    recoverable bookkeeping problem into a failed Monday. It is reported in the
    returned summary instead, which the caller prints.
    """
    if forecast is None or forecast.empty:
        return {"added": 0, "replaced": 0, "total": len(load()), "runs": 0, "db_rows": 0}

    incoming = forecast.copy()
    for col in ("week_of", "ds"):
        incoming[col] = pd.to_datetime(incoming[col])
    missing = [c for c in KEY if c not in incoming.columns]
    if missing:
        raise ValueError(f"forecast is missing key columns: {missing}")
    keep = [c for c in COLUMNS if c in incoming.columns]
    incoming = incoming[keep]

    # The durable store first, so a crash between the two writes loses the
    # local copy rather than the shared one.
    db_rows = store.upsert(incoming)

    existing = _load_parquet()
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
    out = out.sort_values(["model_version", "week_of", "unique_id", "ds"])
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(HISTORY_PATH, index=False)

    return {
        "added": len(incoming),
        "replaced": replaced,
        "total": len(out),
        "runs": int(out.groupby(["model_version", "week_of"]).ngroups),
        # Rows written to the table: -1 means it could not be reached, which is
        # normal on a machine with no credentials and worth saying out loud on
        # the server, where it means this run is not backed up.
        "db_rows": db_rows,
    }


def runs() -> pd.DataFrame:
    """One row per stored run: version, date, SKUs covered, weeks, units."""
    df = load()
    if df.empty:
        return pd.DataFrame(
            columns=["model_version", "week_of", "n_skus", "n_weeks", "forecast_units"]
        )
    return (
        df.groupby(["model_version", "week_of"], as_index=False)
        .agg(
            n_skus=("unique_id", "nunique"),
            n_weeks=("ds", "nunique"),
            forecast_units=("yhat", "sum"),
        )
        .sort_values(["model_version", "week_of"])
        .reset_index(drop=True)
    )


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
            columns=["model_version", "week_of", "unique_id", "ds", "lead", "yhat", "y"]
        )
    if sales is None:
        from src.planning import data as D

        sales = D.load_sales()
    if sales.empty:
        return pd.DataFrame(
            columns=["model_version", "week_of", "unique_id", "ds", "lead", "yhat", "y"]
        )

    # Bounded by both: the calendar says which weeks have finished, the data
    # says which of those actually arrived. Taking the earlier of the two means
    # a late ingest cannot cause a week to be scored against absent actuals.
    settled_through = min(last_complete_week(), pd.to_datetime(sales["ds"]).max())
    closed = hist[hist["ds"] <= settled_through].copy()
    if closed.empty:
        return pd.DataFrame(
            columns=["model_version", "week_of", "unique_id", "ds", "lead", "yhat", "y"]
        )

    actual = sales[["unique_id", "ds", "y"]].copy()
    actual["ds"] = pd.to_datetime(actual["ds"])
    out = closed.merge(actual, on=["unique_id", "ds"], how="left")
    # A SKU-week absent from the sales grid sold nothing. The grid is complete
    # per the ingestion contract, so this is a true zero rather than a gap.
    out["y"] = out["y"].fillna(0.0)
    out["lead"] = ((out["ds"] - out["week_of"]).dt.days / 7).round().astype(int)
    return out[
        ["model_version", "week_of", "unique_id", "ds", "lead", "yhat", "y",
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
            columns=["model_version", "week_of", "segment", "n_skus",
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
        scored.groupby(["model_version", "week_of", "segment"])
        .apply(agg, include_groups=False)
        .reset_index()
    )
    total = (
        scored.assign(segment="TOTAL")
        .groupby(["model_version", "week_of", "segment"])
        .apply(agg, include_groups=False)
        .reset_index()
    )
    return (
        pd.concat([per_segment, total], ignore_index=True)
        .sort_values(["model_version", "week_of", "segment"])
        .reset_index(drop=True)
    )
