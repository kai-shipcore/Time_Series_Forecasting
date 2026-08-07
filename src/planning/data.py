"""Data access layer for the planning views.

Every loader returns a plain pandas DataFrame. Real model/sales/accuracy data is
read from the forecasting repo's ``data/processed`` and ``outputs/reports`` folders.
Inventory-side fields (on-hand, preorder backlog, confirmed inbound, ETA, product
name) do not exist anywhere in this repo, so they come from a separate inventory
snapshot file written by ``scripts/export_inventory_snapshot.py``. Without it a
clearly labelled SAMPLE snapshot is generated so the views are testable end to end.

This module lives in ``src/planning`` rather than under ``dashboard/`` because
both the Streamlit app and the FastAPI service read through it. Anything that
computes a number a user will act on belongs here, so the two hosts cannot
disagree; anything that renders belongs to the host.

Where the inputs live, which is no longer one answer:

- The forward forecast is read from ``shipcore.ml_forward_forecasts`` when the
  database is reachable, and from ``data/processed/ml_forward_forecasts.parquet``
  otherwise. Both are written by every run. The table is what lets a machine
  that did not produce the forecast show the current one; the file is what lets
  a clone with no credentials work at all.
- Inventory is a live query against two databases, falling back to an exported
  CSV and then to labelled sample data.
- Sales, the SKU profiles and the accuracy reports are still files only.

So the old note here, that everything is on disk and moving it is the
prerequisite for deploying the API away from this repo, is now half done. Sales
and profiles are what remain.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd

from src.planning import inventory
from src.planning._cache import cache as _cache

# ---------------------------------------------------------------------------
# Paths.  src/planning/data.py -> parents[2] is the repo root.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"
REPORTS = REPO_ROOT / "outputs" / "reports"
# Named from the repo root rather than relative to this file. It used to be
# `parents[1] / "data"`, which resolved correctly only while this module lived
# under dashboard/lib and would have silently become src/data after the move.
DASH_DATA = REPO_ROOT / "dashboard" / "data"

FORWARD_FORECAST = PROCESSED / "ml_forward_forecasts.parquet"
V1_FORWARD = PROCESSED / "v1_forward_forecasts.parquet"
SALES_CLEAN_PARQUET = PROCESSED / "sales_clean.parquet"
SALES_CLEAN_CSV = PROCESSED / "sales_clean.csv"
SKU_PROFILES = PROCESSED / "sku_profiles.csv"
ML_ACCURACY = REPORTS / "ml_accuracy.csv"
ML_ACCURACY_BY_SKU = REPORTS / "ml_accuracy_by_sku.csv"
BACKTEST_WEEKLY = REPORTS / "ml_backtest_weekly.csv"
INVENTORY_SNAPSHOT = DASH_DATA / "inventory_snapshot.csv"

# ---------------------------------------------------------------------------
# Real data loaders.
# ---------------------------------------------------------------------------
def _read_sales() -> pd.DataFrame:
    if SALES_CLEAN_PARQUET.exists():
        try:
            df = pd.read_parquet(SALES_CLEAN_PARQUET)
        except Exception:
            df = pd.read_csv(SALES_CLEAN_CSV)
    else:
        df = pd.read_csv(SALES_CLEAN_CSV)
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0)
    return df.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def _read_forecasts() -> pd.DataFrame:
    """LightGBM model forecast: unique_id, forecast_date, ds, yhat, bucket,
    history_length, segment, model_version, served_by, run_at.

    Prefers `shipcore.ml_forward_forecasts`, falling back to the parquet.

    The table is what the weekly run on the server writes, so preferring it is
    what lets a laptop show this week's horizon rather than whatever that
    machine last produced or was seeded with. The parquet remains the path for a
    clone with no credentials, which is a supported way to run this and the one
    the seeded fixture depends on.

    Only the newest training run is kept either way. That filter predates the
    table and is why the table can accumulate horizons without changing what any
    caller sees.
    """
    df = None
    try:
        from src.ml.serving import store

        df = store.read_forward()
    except Exception:
        df = None

    if df is None or df.empty:
        df = pd.read_parquet(FORWARD_FORECAST)

    df["ds"] = pd.to_datetime(df["ds"])
    df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    # Keep only the most recent training run's horizon.
    latest = df["forecast_date"].max()
    df = df[df["forecast_date"] == latest].copy()
    return df.reset_index(drop=True)


def _read_v1_forward() -> pd.DataFrame:
    """V1, decomposed onto the model's own (unique_id, ds) grid. A separate
    artifact, never merged into the model forecast table (see
    docs/V1_AND_DASHBOARD_WIRING_TASK.md). SKUs absent from the latest
    velocity pull simply have no rows here."""
    if not V1_FORWARD.exists():
        return pd.DataFrame(columns=["unique_id", "forecast_date", "ds", "v1_yhat"])
    df = pd.read_parquet(V1_FORWARD)
    df["ds"] = pd.to_datetime(df["ds"])
    df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    return df


def _read_profiles() -> pd.DataFrame:
    return pd.read_csv(SKU_PROFILES)


def _read_ml_accuracy() -> pd.DataFrame:
    """Precomputed model-vs-V1 backtest accuracy (scripts/ml_accuracy_report.py):
    model_version, window, cutoff, segment, n_skus, actual_units, pooled_wape,
    bias_pct. Never computed at dashboard load time (validate_* retrains)."""
    if ML_ACCURACY.exists():
        df = pd.read_csv(ML_ACCURACY)
        df["cutoff"] = pd.to_datetime(df["cutoff"])
        return df
    return pd.DataFrame()


def _read_ml_accuracy_by_sku() -> pd.DataFrame:
    """Per-SKU detail behind _read_ml_accuracy(): model_version, window,
    cutoff, unique_id, bucket, history_length, yhat_total, y_total, ae, bias.
    Summing these rows by (model_version, window, segment) reproduces
    ml_accuracy.csv exactly, since both come from the same detail."""
    if ML_ACCURACY_BY_SKU.exists():
        df = pd.read_csv(ML_ACCURACY_BY_SKU)
        df["cutoff"] = pd.to_datetime(df["cutoff"])
        return df
    return pd.DataFrame()


@_cache(show_spinner=False)
def load_sales() -> pd.DataFrame:
    return _read_sales()


@_cache(show_spinner=False)
def load_forecasts() -> pd.DataFrame:
    return _read_forecasts()


@_cache(show_spinner=False)
def load_v1_forward() -> pd.DataFrame:
    return _read_v1_forward()


@_cache(show_spinner=False)
def load_profiles() -> pd.DataFrame:
    return _read_profiles()


@_cache(show_spinner=False)
def load_ml_accuracy() -> pd.DataFrame:
    return _read_ml_accuracy()


@_cache(show_spinner=False)
def load_ml_accuracy_by_sku() -> pd.DataFrame:
    return _read_ml_accuracy_by_sku()


@_cache(show_spinner=False)
def load_backtest_weekly() -> pd.DataFrame:
    """Per-week backtest predictions: unique_id, model_version, window, cutoff,
    ds, lead, yhat, y.

    Written by scripts/ml_backtest_weekly.py, which refuses to write unless the
    totals reconcile exactly with ml_accuracy_by_sku.csv. Optional: the file is
    regenerated per model version and the dashboard degrades to the totals-only
    view when it is absent, so a missing file is a reduced chart rather than an
    error.
    """
    if not BACKTEST_WEEKLY.exists():
        return pd.DataFrame(
            columns=["unique_id", "model_version", "window", "cutoff",
                     "ds", "lead", "yhat", "y"]
        )
    df = pd.read_csv(BACKTEST_WEEKLY)
    df["ds"] = pd.to_datetime(df["ds"])
    return df


# ---------------------------------------------------------------------------
# Readiness.
# ---------------------------------------------------------------------------
#: Files the planning endpoints read, and whether their absence is fatal.
#:
#: ``data/processed`` and ``outputs/reports`` are both gitignored, so a fresh
#: clone has the code and none of the data. The server then starts, answers
#: /health, and raises on every real endpoint, which reaches the browser as a
#: bare "Internal Server Error" and sends the reader looking for a bug in the
#: application. Naming the missing file is the difference between a five minute
#: fix and an afternoon.
#:
#: Every required file, and two of the optional ones, can be put in place by
#: ``scripts/seed_dev_data.py`` from data already in the repository. So the
#: message a reader gets names that rather than the pipeline script that
#: originally produced the file: on a machine with no database and no
#: virtualenv, "run the forward forecast" is not an instruction anyone can
#: follow, and it was the reason this failure state read as a dead end.
SEED_SCRIPT = "scripts/seed_dev_data.py"

_DATA_FILES: list[tuple[str, Path, bool, str]] = [
    ("forecast", FORWARD_FORECAST, True,
     f"{SEED_SCRIPT}, scripts/ml_forward_forecast.py, or DB_* in .env "
     f"to read shipcore.ml_forward_forecasts"),
    ("sales", SALES_CLEAN_PARQUET, True,
     f"{SEED_SCRIPT}, or the weekly ingest"),
    ("profiles", SKU_PROFILES, True,
     f"{SEED_SCRIPT}, or scripts/ml_forward_forecast.py for a real run"),
    ("accuracy", ML_ACCURACY, False,
     "scripts/ml_evaluate.py (tracked in git; absent means a partial checkout)"),
    ("accuracy_by_sku", ML_ACCURACY_BY_SKU, False,
     "scripts/ml_evaluate.py (tracked in git; absent means a partial checkout)"),
    ("backtest_weekly", BACKTEST_WEEKLY, False,
     "scripts/ml_backtest_weekly.py (tracked in git; absent means a partial checkout)"),
    ("v1_forward", V1_FORWARD, False,
     f"{SEED_SCRIPT}, or scripts/v1_forward.py for a real run"),
    ("inventory", INVENTORY_SNAPSHOT, False,
     "scripts/export_inventory_snapshot.py (tracked in git; absent means a partial checkout)"),
]


def readiness() -> dict:
    """Which data files are present, and whether the service can actually serve.

    Cheap enough to call on every health check: it stats a handful of paths and
    reads nothing. Sales is satisfied by either the parquet or the CSV, matching
    what ``_read_sales`` will accept.

    The forecast is satisfied by either the parquet or
    ``shipcore.ml_forward_forecasts``, matching ``_read_forecasts``. Without
    this, a machine that reads the table but has no local file was reported as
    unable to serve while it was in fact serving, and the error card told the
    reader to run a seed script they did not need. The check has to describe
    what the service can do rather than which files happen to exist, and that
    stopped being the same question when the table was added.

    The table probe is the one part of this that is not free: it opens a
    connection. Only attempted when the file is absent, so the common paths, a
    seeded developer machine and the server, still stat and nothing more.
    """
    files = []
    missing_required = []
    for name, path, required, produced_by in _DATA_FILES:
        exists = path.exists()
        if name == "sales" and not exists:
            exists = SALES_CLEAN_CSV.exists()
        if name == "forecast" and not exists:
            try:
                from src.ml.serving import store

                exists = store.available(store.FORWARD_TABLE)
            except Exception:
                exists = False
        files.append({
            "name": name,
            "path": str(path.relative_to(REPO_ROOT)),
            "exists": exists,
            "required": required,
            "produced_by": produced_by,
        })
        if required and not exists:
            missing_required.append(name)

    return {
        "ready": not missing_required,
        "missing_required": missing_required,
        "missing_optional": [
            f["name"] for f in files if not f["exists"] and not f["required"]
        ],
        "files": files,
        "repo_root": str(REPO_ROOT),
    }


def sku_backtest_weekly(unique_id: str, version: str | None = None) -> pd.DataFrame:
    """Per-week backtest rows for one SKU, for the served model version."""
    df = load_backtest_weekly()
    if df.empty:
        return df
    out = df[df["unique_id"] == unique_id]
    if version is not None and "model_version" in out.columns:
        out = out[out["model_version"] == version]
    return out.sort_values(["cutoff", "lead"]).reset_index(drop=True)


def forecast_snapshot_date() -> pd.Timestamp | None:
    """Training date of the forecast horizon currently loaded."""
    fc = load_forecasts()
    return None if fc.empty else fc["forecast_date"].max()


# ---------------------------------------------------------------------------
# Derived sales aggregates.
# ---------------------------------------------------------------------------
def recent_sales(weeks: int = 4) -> pd.DataFrame:
    """Sum of the last ``weeks`` weeks of actual demand per SKU.

    Sales are weekly (W-MON), so 4 weeks is the ~30-day trailing window.
    Returns unique_id, recent_units, avg_daily_sales.
    """
    sales = load_sales()
    if sales.empty:
        return pd.DataFrame(columns=["unique_id", "recent_units", "avg_daily_sales"])
    last_week = sales["ds"].max()
    cutoff = last_week - pd.Timedelta(weeks=weeks)
    window = sales[sales["ds"] > cutoff]
    agg = window.groupby("unique_id")["y"].sum().rename("recent_units").reset_index()
    agg["avg_daily_sales"] = agg["recent_units"] / (weeks * 7.0)
    return agg


#: Display bands for the demand-trend label, on the 4-week-over-12-week
#: deseasonalized ramp. Nothing here changes what the model predicts or what is
#: ordered; these only decide which word appears beside the ratio.
#:
#: Reciprocal rather than additive. Ramp is a ratio, so 0.80 and 1.25 are the
#: same distance from 1.0 in the units it is measured in; 0.80 and 1.20 are not.
#:
#: These replaced 0.70 and 1.10 on 2026-08-04. The 0.70 was COLLAPSE_RAMP,
#: measured as the point where a plain 4-week average beat the model on the
#: development windows, 0.28 pooled WAPE against 1.49, and correct for that
#: question. But the flag that asked it moved off ramp deliberately: ramp
#: describes how history moved, not whether the forecast agrees with where
#: demand is now (see the note above `forecast_runs_high` in calc.py). That left
#: a model-reliability threshold labelling a descriptive band, where it called a
#: 30% fall "steady" and a 10% rise "rising". The old 1.10 and 0.40 were never
#: justified anywhere.
#:
#: Not narrowed further, and the measurement is why. Ramp deviation scales
#: inversely with volume: median |ramp - 1| is 0.28 for SKUs selling under 10
#: units in four weeks against 0.07 above 100, so half the smallest SKUs fall
#: outside this band on small-number variation alone. Tightening to 0.90/1.11
#: labels 60% of the catalogue as moving, which distinguishes nothing.
TREND_STEADY_LOW = 0.80
TREND_STEADY_HIGH = 1.25
#: Below this, "collapsing" rather than "falling". Inherited unjustified and
#: kept for now: its reciprocal (2.5x, a surge) has no state of its own, which
#: is a separate question about whether a breakout deserves its own label.
TREND_COLLAPSE = 0.40


@_cache(show_spinner=False)
def sku_ramp() -> pd.DataFrame:
    """Per-SKU ramp: the last 4 weeks over the last 12, on deseasonalized demand.

    This is the model's own ``ramp_4_12`` feature evaluated at the latest actual
    week. The seasonal factors are imported from ``src.ml.seasonal`` rather than
    reimplemented, so the dashboard's notion of "collapsing" cannot drift from
    the model's notion of the same thing.

    Returns unique_id, ramp, wa4 (mean units/week over the last 4 weeks, on raw
    demand, because that is what a planner would use as a fallback figure).
    """
    sales = load_sales()
    if sales.empty:
        return pd.DataFrame(columns=["unique_id", "ramp", "wa4"])
    try:
        import sys

        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from src.ml.seasonal import ml_factors
    except Exception:
        # The ML package is optional from the dashboard's point of view. Without
        # it the ramp is computed on raw demand, which is close but not the
        # model's variable, so the column is marked and the caller can say so.
        ml_factors = None

    s = sales.sort_values(["unique_id", "ds"]).copy()
    s["y_adj"] = s["y"] / ml_factors(s["ds"]).to_numpy() if ml_factors is not None else s["y"]
    g = s.groupby("unique_id")
    r4 = g["y_adj"].apply(lambda x: x.tail(4).mean())
    r12 = g["y_adj"].apply(lambda x: x.tail(12).mean())
    wa4 = g["y"].apply(lambda x: x.tail(4).mean())
    out = pd.DataFrame({"ramp": r4 / r12.clip(lower=1e-9), "wa4": wa4}).reset_index()
    out.attrs["deseasonalized"] = ml_factors is not None
    return out


def sku_sales_history(unique_id: str) -> pd.DataFrame:
    sales = load_sales()
    return sales[sales["unique_id"] == unique_id].copy()


def sku_forecast(unique_id: str) -> pd.DataFrame:
    """Model forecast for one SKU, with v1_yhat left-joined on (unique_id, ds)
    where available. V1 is a separate artifact (docs/V1_AND_DASHBOARD_WIRING_TASK.md);
    a SKU absent from the latest velocity pull simply has no v1_yhat (NaN)."""
    fc = load_forecasts()
    fc = fc[fc["unique_id"] == unique_id].sort_values("ds").copy()
    v1 = load_v1_forward()
    v1 = v1[v1["unique_id"] == unique_id][["unique_id", "ds", "v1_yhat"]]
    return fc.merge(v1, on=["unique_id", "ds"], how="left")


# ---------------------------------------------------------------------------
# Inventory adapter (sample until a real export is provided).
# ---------------------------------------------------------------------------
def inventory_columns() -> list[str]:
    """Schema expected of an inventory export written to inventory_snapshot.csv.

    ``available_inventory`` is available-to-sell, meaning physical stock less what
    is already allocated to unshipped orders. It is not on-hand, and the UI labels
    it "Available" for that reason.

    ``transit_stock`` is carried but not used in any calculation. The Commerce app
    adds it to its own stock total, so it is exported to keep the two comparable,
    but what it counts is not documented anywhere and it may overlap
    ``confirmed_inbound``. Until that is settled it is reported, not consumed.

    ``draft_inbound`` is units on containers still in draft: an order that exists
    but is not committed. Also reported and not consumed, but for a different and
    deliberate reason. A draft can be cancelled, so crediting it against the
    recommended quantity would under-order the SKUs someone has already acted on.
    It is shown beside the recommendation so a purchaser can see that an order
    exists, while the arithmetic continues to assume it does not.

    Absence of a draft row is a real zero, exactly as for ``confirmed_inbound``:
    both come from container line items, so no matching row means no units. An
    export written before these columns existed reads as zero too, which is
    correct enough, because how far the whole export can be trusted is already
    reported once by ``inventory_source`` rather than per column.
    """
    return [
        "unique_id",
        "available_inventory",
        "preorder_backlog",
        "confirmed_inbound",
        "inbound_eta",
        "draft_inbound",
        "draft_eta",
        "transit_stock",
    ]


# Product category, read from the SKU prefix. This is real: the prefix is part of
# the identifier, so Car cover and Seat cover are derived rather than invented.
#
# Three buckets only. Anything not confirmed as one of the two named families is
# "Other" rather than shown as its raw, uninterpreted prefix (CA-CL and the like).
FAMILY_CATEGORIES = {"CC": "Car cover"}          # every CC-* SKU is a car cover
PRODUCT_CATEGORIES = {"CA-SC": "Seat cover"}
OTHER_CATEGORY = "Other"


def product_category(unique_id: str) -> str:
    """Category for a SKU, read from its identifier: Car cover, Seat cover, or Other.

    The first token gives the family where that is enough to decide (every CC-*
    SKU is a car cover, whatever the second token). Otherwise the first two
    tokens are looked up against the one other confirmed prefix. Anything else
    is Other.
    """
    parts = str(unique_id).split("-")
    if parts and parts[0] in FAMILY_CATEGORIES:
        return FAMILY_CATEGORIES[parts[0]]
    prefix = "-".join(parts[:2]) if len(parts) >= 2 else str(unique_id)
    return PRODUCT_CATEGORIES.get(prefix, OTHER_CATEGORY)


def _build_sample_inventory() -> pd.DataFrame:
    """Deterministic, clearly-labelled sample inventory for the forecast SKUs.

    Values are seeded off recent real sales so the demo reads coherently, but they
    are NOT real stock positions. Replace by writing inventory_snapshot.csv.
    """
    fc = load_forecasts()
    if fc.empty:
        return pd.DataFrame(columns=inventory_columns() + ["is_sample"])

    ids = sorted(fc["unique_id"].unique())
    rng = np.random.RandomState(42)
    ads = (
        recent_sales(weeks=4)
        .set_index("unique_id")["avg_daily_sales"]
        .reindex(ids)
        .fillna(0.0)
    )

    today = pd.Timestamp(_dt.date.today())
    days_cover = rng.randint(0, 55, size=len(ids))
    available = np.round(ads.to_numpy() * days_cover).astype(int)
    # Force a realistic slice of out-of-stock SKUs.
    oos = rng.rand(len(ids)) < 0.18
    available[oos] = 0

    has_preorder = rng.rand(len(ids)) < 0.16
    preorder = np.where(
        has_preorder, np.round(ads.to_numpy() * rng.randint(5, 40, len(ids))), 0
    ).astype(int)

    has_inbound = rng.rand(len(ids)) < 0.35
    inbound = np.where(
        has_inbound, np.round(ads.to_numpy() * rng.randint(20, 90, len(ids))), 0
    ).astype(int)
    eta_days = rng.randint(7, 70, size=len(ids))
    eta = [
        (today + pd.Timedelta(days=int(d))).date().isoformat() if h else ""
        for d, h in zip(eta_days, has_inbound)
    ]

    df = pd.DataFrame(
        {
            "unique_id": ids,
            "available_inventory": available,
            "preorder_backlog": preorder,
            "confirmed_inbound": inbound,
            "inbound_eta": eta,
        }
    )
    df["is_sample"] = True
    return df


@_cache(show_spinner=False, ttl=300)
def load_inventory() -> pd.DataFrame:
    """Return the inventory position, preferring the live database tables.

    Three sources, tried in order, each a degradation of the one before:

    1. **The databases.** The inventory tables belong to the Commerce
       Integration application and are refreshed on its schedule, so reading
       them directly is both fresher than a copy and one fewer thing to
       remember to run. This is the normal path.
    2. **The exported CSV**, if the databases are unreachable. Real figures,
       but as old as the last time someone ran the export script.
    3. **Generated sample data**, if there is no export either. Coherent enough
       to develop against and labelled everywhere it is shown.

    The ``source`` column records which applied, and ``is_sample`` is kept as
    the boolean the UI already warns on. Falling back rather than raising is
    deliberate: a working copy without credentials should still start.
    """
    # Scoped to every profiled SKU, not only the forecastable ones. The
    # non-forecast section of the action list needs stock, backlog and inbound
    # for the intermittent tail, which is 87% of the SKU count and a fifth of
    # recent unit volume. Falls back to the forecast SKUs when no profile
    # snapshot is present, which is the only case where that list is all that
    # can be known.
    prof = load_profiles()
    fc = load_forecasts()
    if not prof.empty and "unique_id" in prof.columns:
        skus = sorted(prof["unique_id"].dropna().unique())
    else:
        skus = sorted(fc["unique_id"].unique()) if not fc.empty else []

    live = inventory.fetch_cached(skus) if skus else None
    if live is not None and not live.empty:
        for col in inventory_columns():
            if col not in live.columns:
                live[col] = np.nan
        live["is_sample"] = False
        live["source"] = "database"
        live.attrs.setdefault("snapshot_at", None)
        return live

    if INVENTORY_SNAPSHOT.exists():
        df = pd.read_csv(INVENTORY_SNAPSHOT)
        for col in inventory_columns():
            if col not in df.columns:
                df[col] = np.nan
        df["is_sample"] = False
        df["source"] = "export"
        return df

    sample = _build_sample_inventory()
    sample["source"] = "sample"
    return sample


def inventory_source() -> str:
    """Where the current inventory figures came from: database, export, sample."""
    inv = load_inventory()
    return str(inv["source"].iloc[0]) if len(inv) and "source" in inv.columns else "sample"


def inventory_is_sample() -> bool:
    inv = load_inventory()
    return bool(inv["is_sample"].iloc[0]) if len(inv) else True
