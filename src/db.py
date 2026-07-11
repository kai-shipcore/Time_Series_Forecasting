import os
from datetime import date, datetime
import pandas as pd
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from config import ROUTE_SHORT_SMOOTH_TO_V1

load_dotenv()

# Method shown for smooth/short SKUs — follows the actual routing in selector.py
_SHORT_METHOD = "V1" if ROUTE_SHORT_SMOOTH_TO_V1 else "WindowAverage"

_TABLE = "shipcore.fc_forward_forecasts"
_HIST_TABLE = "shipcore.fc_forecast_history"

_HIST_CREATE_SQL = f"""
CREATE TABLE IF NOT EXISTS {_HIST_TABLE} (
    unique_id         TEXT      NOT NULL,
    week_of           DATE      NOT NULL,
    run_date          TIMESTAMP NOT NULL,
    bucket            TEXT      NOT NULL,
    history_length    TEXT      NOT NULL,
    horizon_weeks     INTEGER   NOT NULL,
    yhat_total        FLOAT     NOT NULL,
    yhat_lo_40        FLOAT,
    yhat_hi_40        FLOAT,
    yhat_lo_60        FLOAT,
    yhat_hi_60        FLOAT,
    yhat_lo_70        FLOAT,
    yhat_hi_70        FLOAT,
    yhat_lo_80        FLOAT,
    yhat_hi_80        FLOAT,
    yhat_lo_90        FLOAT,
    yhat_hi_90        FLOAT,
    forecast_end_date DATE      NOT NULL,
    PRIMARY KEY (unique_id, week_of)
)
"""

_HIST_MIGRATE_PI_SQL = f"""
DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_schema='shipcore' AND table_name='fc_forecast_history' AND column_name='yhat_lo')
    THEN ALTER TABLE {_HIST_TABLE} RENAME COLUMN yhat_lo TO yhat_lo_70; END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_schema='shipcore' AND table_name='fc_forecast_history' AND column_name='yhat_hi')
    THEN ALTER TABLE {_HIST_TABLE} RENAME COLUMN yhat_hi TO yhat_hi_70; END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema='shipcore' AND table_name='fc_forecast_history' AND column_name='yhat_lo_40')
    THEN ALTER TABLE {_HIST_TABLE} ADD COLUMN yhat_lo_40 FLOAT; END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema='shipcore' AND table_name='fc_forecast_history' AND column_name='yhat_hi_40')
    THEN ALTER TABLE {_HIST_TABLE} ADD COLUMN yhat_hi_40 FLOAT; END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema='shipcore' AND table_name='fc_forecast_history' AND column_name='yhat_lo_60')
    THEN ALTER TABLE {_HIST_TABLE} ADD COLUMN yhat_lo_60 FLOAT; END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema='shipcore' AND table_name='fc_forecast_history' AND column_name='yhat_hi_60')
    THEN ALTER TABLE {_HIST_TABLE} ADD COLUMN yhat_hi_60 FLOAT; END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema='shipcore' AND table_name='fc_forecast_history' AND column_name='yhat_lo_70')
    THEN ALTER TABLE {_HIST_TABLE} ADD COLUMN yhat_lo_70 FLOAT; END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema='shipcore' AND table_name='fc_forecast_history' AND column_name='yhat_hi_70')
    THEN ALTER TABLE {_HIST_TABLE} ADD COLUMN yhat_hi_70 FLOAT; END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema='shipcore' AND table_name='fc_forecast_history' AND column_name='yhat_lo_80')
    THEN ALTER TABLE {_HIST_TABLE} ADD COLUMN yhat_lo_80 FLOAT; END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema='shipcore' AND table_name='fc_forecast_history' AND column_name='yhat_hi_80')
    THEN ALTER TABLE {_HIST_TABLE} ADD COLUMN yhat_hi_80 FLOAT; END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema='shipcore' AND table_name='fc_forecast_history' AND column_name='yhat_lo_90')
    THEN ALTER TABLE {_HIST_TABLE} ADD COLUMN yhat_lo_90 FLOAT; END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema='shipcore' AND table_name='fc_forecast_history' AND column_name='yhat_hi_90')
    THEN ALTER TABLE {_HIST_TABLE} ADD COLUMN yhat_hi_90 FLOAT; END IF;
END $$;
"""

_HIST_MIGRATE_SQL = f"""
ALTER TABLE IF EXISTS {_HIST_TABLE}
    ALTER COLUMN run_date TYPE TIMESTAMP USING run_date::timestamp
"""

_CREATE_SQL = f"""
CREATE TABLE IF NOT EXISTS {_TABLE} (
    unique_id      TEXT  NOT NULL,
    forecast_date  DATE  NOT NULL,
    ds             DATE  NOT NULL,
    yhat           FLOAT NOT NULL,
    yhat_lo_40     FLOAT,
    yhat_hi_40     FLOAT,
    yhat_lo_60     FLOAT,
    yhat_hi_60     FLOAT,
    yhat_lo_70     FLOAT,
    yhat_hi_70     FLOAT,
    yhat_lo_80     FLOAT,
    yhat_hi_80     FLOAT,
    yhat_lo_90     FLOAT,
    yhat_hi_90     FLOAT,
    bucket         TEXT,
    history_length TEXT,
    selected_model TEXT,
    confidence     TEXT,
    active_weeks   INTEGER,
    v1_yhat        FLOAT,
    PRIMARY KEY (unique_id, forecast_date, ds)
)
"""

_MIGRATE_PI_SQL = f"""
DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_schema='shipcore' AND table_name='fc_forward_forecasts' AND column_name='yhat_lo')
    THEN ALTER TABLE {_TABLE} RENAME COLUMN yhat_lo TO yhat_lo_70; END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_schema='shipcore' AND table_name='fc_forward_forecasts' AND column_name='yhat_hi')
    THEN ALTER TABLE {_TABLE} RENAME COLUMN yhat_hi TO yhat_hi_70; END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema='shipcore' AND table_name='fc_forward_forecasts' AND column_name='yhat_lo_40')
    THEN ALTER TABLE {_TABLE} ADD COLUMN yhat_lo_40 FLOAT; END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema='shipcore' AND table_name='fc_forward_forecasts' AND column_name='yhat_hi_40')
    THEN ALTER TABLE {_TABLE} ADD COLUMN yhat_hi_40 FLOAT; END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema='shipcore' AND table_name='fc_forward_forecasts' AND column_name='yhat_lo_60')
    THEN ALTER TABLE {_TABLE} ADD COLUMN yhat_lo_60 FLOAT; END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema='shipcore' AND table_name='fc_forward_forecasts' AND column_name='yhat_hi_60')
    THEN ALTER TABLE {_TABLE} ADD COLUMN yhat_hi_60 FLOAT; END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema='shipcore' AND table_name='fc_forward_forecasts' AND column_name='yhat_lo_70')
    THEN ALTER TABLE {_TABLE} ADD COLUMN yhat_lo_70 FLOAT; END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema='shipcore' AND table_name='fc_forward_forecasts' AND column_name='yhat_hi_70')
    THEN ALTER TABLE {_TABLE} ADD COLUMN yhat_hi_70 FLOAT; END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema='shipcore' AND table_name='fc_forward_forecasts' AND column_name='yhat_lo_80')
    THEN ALTER TABLE {_TABLE} ADD COLUMN yhat_lo_80 FLOAT; END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema='shipcore' AND table_name='fc_forward_forecasts' AND column_name='yhat_hi_80')
    THEN ALTER TABLE {_TABLE} ADD COLUMN yhat_hi_80 FLOAT; END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema='shipcore' AND table_name='fc_forward_forecasts' AND column_name='yhat_lo_90')
    THEN ALTER TABLE {_TABLE} ADD COLUMN yhat_lo_90 FLOAT; END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema='shipcore' AND table_name='fc_forward_forecasts' AND column_name='yhat_hi_90')
    THEN ALTER TABLE {_TABLE} ADD COLUMN yhat_hi_90 FLOAT; END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema='shipcore' AND table_name='fc_forward_forecasts' AND column_name='active_weeks')
    THEN ALTER TABLE {_TABLE} ADD COLUMN active_weeks INTEGER; END IF;
END $$;
"""


_engine = None

def get_engine():
    global _engine
    if _engine is None:
        url = "postgresql+psycopg2://{}:{}@{}:{}/{}".format(
            quote_plus(os.getenv("DB_USER")),
            quote_plus(os.getenv("DB_PASSWORD")),
            os.getenv("DB_HOST"),
            os.getenv("DB_PORT"),
            os.getenv("DB_NAME"),
        )
        _engine = create_engine(
            url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=1800,
            connect_args={
                "connect_timeout": 10,
                "sslmode": "require",
                # Kill any single statement running longer than 2 minutes.
                "options": "-c statement_timeout=120000",
            },
        )
    return _engine


def write_forward_forecasts(df: pd.DataFrame) -> None:
    """Create table if needed, migrate PI columns, insert fresh results.

    Replaces any prior run with the same horizon start: training always cuts at
    the last complete Monday, so runs within the same week produce identical
    forecasts and only the latest one is kept.
    """
    engine = get_engine()
    horizon_start = str(pd.Timestamp(df["ds"].min()).date())
    with engine.begin() as conn:
        conn.execute(text(_CREATE_SQL))
        conn.execute(text(_MIGRATE_PI_SQL))
        conn.execute(
            text(f"""
                DELETE FROM {_TABLE}
                WHERE forecast_date IN (
                    SELECT forecast_date FROM {_TABLE}
                    GROUP BY forecast_date
                    HAVING MIN(ds) = :hs
                )
            """),
            {"hs": horizon_start},
        )
        df.to_sql(
            "fc_forward_forecasts",
            conn,
            schema="shipcore",
            if_exists="append",
            index=False,
            method="multi",
            chunksize=500,
        )


def write_forecast_history(df: pd.DataFrame, run_date: datetime, horizon_weeks: int) -> None:
    """Aggregate per-week forecast rows into one summary row per SKU and upsert.

    Unique constraint is (unique_id, week_of) — one row per SKU per Monday week.
    Re-running within the same week overwrites the previous run.
    """
    def _sum_or_none(s: pd.Series):
        valid = s.dropna()
        return float(valid.sum()) if len(valid) > 0 else None

    today = pd.Timestamp(run_date.date())
    week_of = (today - pd.Timedelta(days=today.dayofweek)).date()

    _PI_LEVELS = [40, 60, 70, 80, 90]

    df = df[df["bucket"] == "smooth"].copy()

    agg_dict: dict = {"yhat_total": ("yhat", "sum"), "forecast_end_date": ("ds", "max")}
    for lvl in _PI_LEVELS:
        agg_dict[f"yhat_lo_{lvl}"] = (f"yhat_lo_{lvl}", _sum_or_none)
        agg_dict[f"yhat_hi_{lvl}"] = (f"yhat_hi_{lvl}", _sum_or_none)

    agg = (
        df.groupby(["unique_id", "bucket", "history_length"])
        .agg(**agg_dict)
        .reset_index()
    )
    agg["week_of"] = week_of
    agg["run_date"] = run_date
    agg["horizon_weeks"] = horizon_weeks
    agg["forecast_end_date"] = agg["forecast_end_date"].apply(
        lambda v: v.date() if hasattr(v, "date") else v
    )

    records = []
    for _, row in agg.iterrows():
        rec = {
            "unique_id":         row["unique_id"],
            "week_of":           week_of,
            "run_date":          run_date,
            "bucket":            row["bucket"],
            "history_length":    row["history_length"],
            "horizon_weeks":     horizon_weeks,
            "yhat_total":        float(row["yhat_total"]),
            "forecast_end_date": row["forecast_end_date"],
        }
        for lvl in _PI_LEVELS:
            rec[f"yhat_lo_{lvl}"] = row[f"yhat_lo_{lvl}"]
            rec[f"yhat_hi_{lvl}"] = row[f"yhat_hi_{lvl}"]
        records.append(rec)

    _pi_insert_cols = ", ".join(f"yhat_lo_{l}, yhat_hi_{l}" for l in _PI_LEVELS)
    _pi_insert_vals = ", ".join(f":yhat_lo_{l}, :yhat_hi_{l}" for l in _PI_LEVELS)
    _pi_upsert = "\n".join(
        f"                    yhat_lo_{l} = EXCLUDED.yhat_lo_{l},\n"
        f"                    yhat_hi_{l} = EXCLUDED.yhat_hi_{l},"
        for l in _PI_LEVELS
    ).rstrip(",")

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(_HIST_CREATE_SQL))
        try:
            conn.execute(text(_HIST_MIGRATE_SQL))
        except Exception:
            pass  # column already TIMESTAMP or table just created
        conn.execute(text(_HIST_MIGRATE_PI_SQL))
        conn.execute(
            text(f"""
                INSERT INTO {_HIST_TABLE}
                    (unique_id, week_of, run_date, bucket, history_length,
                     horizon_weeks, yhat_total, {_pi_insert_cols}, forecast_end_date)
                VALUES
                    (:unique_id, :week_of, :run_date, :bucket, :history_length,
                     :horizon_weeks, :yhat_total, {_pi_insert_vals}, :forecast_end_date)
                ON CONFLICT (unique_id, week_of) DO UPDATE SET
                    run_date          = EXCLUDED.run_date,
                    bucket            = EXCLUDED.bucket,
                    history_length    = EXCLUDED.history_length,
                    horizon_weeks     = EXCLUDED.horizon_weeks,
                    yhat_total        = EXCLUDED.yhat_total,
{_pi_upsert},
                    forecast_end_date = EXCLUDED.forecast_end_date
            """),
            records,
        )
    print(f"  Wrote {len(records)} rows to fc_forecast_history (week_of={week_of})")


def read_latest_forecast(sku_id: str) -> pd.DataFrame:
    engine = get_engine()
    query = f"""
        SELECT *
        FROM {_TABLE}
        WHERE unique_id = :uid
          AND forecast_date = (
              SELECT MAX(forecast_date) FROM {_TABLE} WHERE unique_id = :uid
          )
        ORDER BY ds
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params={"uid": sku_id})
    df["ds"] = pd.to_datetime(df["ds"])
    return df


_ALL_PRODUCT_TYPES = {"Car Cover", "Seat Cover", "Floor Mat"}


def _product_type_where(col: str, product_types: list[str] | None) -> str:
    """Return a SQL boolean expression to filter a SKU column by product type list."""
    if not product_types or _ALL_PRODUCT_TYPES.issubset(set(product_types)):
        return "TRUE"
    parts = []
    for pt in product_types:
        if pt == "Car Cover":
            parts.append(f"({col} LIKE 'CC%%' OR {col} = 'C-SJ-GR-7')")
        elif pt == "Seat Cover":
            parts.append(f"({col} LIKE 'CA-SC%%' OR {col} LIKE 'CL-SC%%')")
        elif pt == "Floor Mat":
            parts.append(f"{col} LIKE 'CA-FM%%'")
    return f"({' OR '.join(parts)})" if parts else "TRUE"


def read_segments(weeks: int = 10, product_types: list[str] | None = None) -> dict:
    """Return SKU counts and demand totals per segment for the last N complete weeks.

    Forecasted SKUs (smooth) come from fc_forward_forecasts.
    Everything else in the snapshot is treated as intermittent.
    """
    engine = get_engine()

    today = pd.Timestamp.today().normalize()
    days_back = today.dayofweek or 7
    last_monday = today - pd.Timedelta(days=days_back)
    period_start = last_monday - pd.Timedelta(weeks=weeks)

    pt_fcast  = _product_type_where("unique_id",       product_types)
    pt_snap   = _product_type_where("link_master_sku", product_types)

    with engine.connect() as conn:
        # Latest segment classification for every forecasted SKU
        forecast_df = pd.read_sql(text(f"""
            SELECT DISTINCT unique_id, bucket, history_length
            FROM {_TABLE}
            WHERE forecast_date = (SELECT MAX(forecast_date) FROM {_TABLE})
              AND {pt_fcast}
        """), conn)

        # All SKUs ever seen — so dormant SKUs (no recent sales) still count
        all_skus_df = pd.read_sql(text(f"""
            SELECT DISTINCT link_master_sku
            FROM shipcore.fc_velocity_link_snapshot_forecast
            WHERE {pt_snap}
        """), conn)

        # Demand per SKU for the last N complete weeks. Cap at last_monday so
        # the in-progress week is excluded — otherwise totals creep up daily
        # and disagree with the displayed period and the W-MON charts.
        demand_df = pd.read_sql(text(f"""
            SELECT link_master_sku, SUM(link_qty) AS demand
            FROM shipcore.fc_velocity_link_snapshot_forecast
            WHERE order_date > :start
              AND order_date <= :end
              AND {pt_snap}
            GROUP BY link_master_sku
        """), conn, params={"start": period_start, "end": last_monday})

    # Start from full SKU universe, attach recent demand (0 for dormant SKUs)
    merged = all_skus_df.merge(demand_df, on="link_master_sku", how="left")
    merged["demand"] = merged["demand"].fillna(0).astype(int)

    # Join segment classification — SKUs not in forecast table are intermittent
    merged = merged.merge(
        forecast_df, left_on="link_master_sku", right_on="unique_id", how="left"
    )

    def _segment(row):
        if pd.isna(row["bucket"]) or row["bucket"] == "low_volume":
            return "intermittent"
        if row["history_length"] == "short":
            return "smooth_short"
        return "smooth_full"

    merged["segment"] = merged.apply(_segment, axis=1)

    total_skus   = len(merged)
    total_demand = int(merged["demand"].sum())

    _DEFS = [
        ("smooth_full",  "Smooth",              "StatsForecast"),
        ("smooth_short", "Smooth / Short history", _SHORT_METHOD),
        ("intermittent", "Intermittent",         "Restock policy"),
    ]

    segments = []
    for key, name, method in _DEFS:
        sub = merged[merged["segment"] == key]
        demand = int(sub["demand"].sum())
        segments.append({
            "segment":    key,
            "name":       name,
            "method":     method,
            "sku_count":  len(sub),
            "demand":     demand,
            "demand_pct": round(demand / total_demand * 100, 1) if total_demand > 0 else 0.0,
        })

    forecasted = merged[merged["segment"].isin({"smooth_full", "smooth_short"})]

    # ── Pareto curve ──────────────────────────────────────────────────────────
    sorted_skus = merged.sort_values("demand", ascending=False).reset_index(drop=True)
    n_skus = len(sorted_skus)
    total_d = float(sorted_skus["demand"].sum())
    sorted_skus["sku_pct"] = (sorted_skus.index + 1) / n_skus * 100
    sorted_skus["cum_d_pct"] = (sorted_skus["demand"].cumsum() / total_d * 100) if total_d > 0 else 0.0

    pareto_x = sorted_skus["sku_pct"].round(2).tolist()
    pareto_y = sorted_skus["cum_d_pct"].round(2).tolist()

    n_fcast = len(forecasted)
    pareto_annotation = None
    if n_fcast > 0 and n_skus > 0 and total_d > 0:
        pareto_annotation = {
            "sku_pct":    round(n_fcast / n_skus * 100, 1),
            "demand_pct": round(float(forecasted["demand"].sum()) / total_d * 100, 1),
        }

    return {
        "total_skus":       total_skus,
        "forecasted_skus":  len(forecasted),
        "forecasted_pct":   round(len(forecasted) / total_skus * 100, 1) if total_skus > 0 else 0.0,
        "total_demand":     total_demand,
        "forecasted_demand": int(forecasted["demand"].sum()),
        "forecasted_demand_pct": round(forecasted["demand"].sum() / total_demand * 100, 1) if total_demand > 0 else 0.0,
        "weeks":        weeks,
        "period_start": str(period_start.date()),
        "period_end":   str(last_monday.date()),
        "segments":     segments,
        "pareto": {
            "x":          pareto_x,
            "y":          pareto_y,
            "annotation": pareto_annotation,
        },
    }


# ── Indexes ──────────────────────────────────────────────────────────────────
_ENSURE_INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_fc_vel_fc_sku_date ON shipcore.fc_velocity_link_snapshot_forecast (link_master_sku, order_date)",
    "CREATE INDEX IF NOT EXISTS idx_fc_vel_fc_date ON shipcore.fc_velocity_link_snapshot_forecast (order_date)",
    "CREATE INDEX IF NOT EXISTS idx_fc_fwd_forecast_date ON shipcore.fc_forward_forecasts (forecast_date)",
]


def ensure_indexes() -> None:
    engine = get_engine()
    with engine.begin() as conn:
        for stmt in _ENSURE_INDEX_SQL:
            conn.execute(text(stmt))


# ── Job persistence ──────────────────────────────────────────────────────────
import json
import uuid as _uuid

_JOBS_TABLE = "shipcore.fc_jobs"

_JOBS_CREATE_SQL = f"""
CREATE TABLE IF NOT EXISTS {_JOBS_TABLE} (
    job_id           TEXT PRIMARY KEY,
    job_type         TEXT NOT NULL,
    status           TEXT NOT NULL,
    lines            JSONB NOT NULL DEFAULT '[]'::jsonb,
    result           JSONB,
    exit_code        INTEGER,
    pgid             INTEGER,
    cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

# At most ONE running job per job_type ('forecast' | 'simulation').
# The partial unique index makes this race-free even across workers/processes.
_JOBS_INDEX_SQL = f"""
CREATE UNIQUE INDEX IF NOT EXISTS fc_jobs_one_running_per_type
ON {_JOBS_TABLE} (job_type)
WHERE status IN ('running', 'cancelling')
"""

_jobs_table_ready = False


def ensure_jobs_table() -> None:
    global _jobs_table_ready
    if _jobs_table_ready:
        return
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(_JOBS_CREATE_SQL))
        conn.execute(text(_JOBS_INDEX_SQL))
    _jobs_table_ready = True


def create_job(job_type: str) -> str | None:
    """Insert a new running job. Returns job_id, or None if a job of this
    type is already running (unique partial index violation)."""
    from sqlalchemy.exc import IntegrityError
    ensure_jobs_table()
    job_id = str(_uuid.uuid4())[:8]
    engine = get_engine()
    try:
        with engine.begin() as conn:
            conn.execute(
                text(f"INSERT INTO {_JOBS_TABLE} (job_id, job_type, status) "
                     f"VALUES (:jid, :jt, 'running')"),
                {"jid": job_id, "jt": job_type},
            )
    except IntegrityError:
        return None
    return job_id


def append_job_lines(job_id: str, lines: list[str]) -> None:
    """Append log lines to the job's JSONB array and bump the heartbeat."""
    if not lines:
        return
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(f"""UPDATE {_JOBS_TABLE}
                     SET lines = lines || CAST(:new AS jsonb), updated_at = now()
                     WHERE job_id = :jid"""),
            {"jid": job_id, "new": json.dumps(lines)},
        )


def touch_job(job_id: str) -> None:
    """Heartbeat: bump updated_at without changing anything else."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(f"UPDATE {_JOBS_TABLE} SET updated_at = now() WHERE job_id = :jid"),
            {"jid": job_id},
        )


def set_job_pgid(job_id: str, pgid: int) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(f"UPDATE {_JOBS_TABLE} SET pgid = :pgid, updated_at = now() "
                 f"WHERE job_id = :jid"),
            {"jid": job_id, "pgid": pgid},
        )


def finish_job(job_id: str, status: str, exit_code: int | None = None,
               result: dict | None = None) -> None:
    """Terminal transition: status must be 'done' | 'failed' | 'cancelled'."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(f"""UPDATE {_JOBS_TABLE}
                     SET status = :st, exit_code = :ec,
                         result = CAST(:res AS jsonb), updated_at = now()
                     WHERE job_id = :jid"""),
            {"jid": job_id, "st": status, "ec": exit_code,
             "res": json.dumps(result) if result is not None else None},
        )


def get_job(job_id: str) -> dict | None:
    ensure_jobs_table()
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text(f"""SELECT job_id, job_type, status, lines, result,
                            exit_code, pgid, cancel_requested
                     FROM {_JOBS_TABLE} WHERE job_id = :jid"""),
            {"jid": job_id},
        ).fetchone()
    if not row:
        return None
    return {
        "job_id": row[0], "job_type": row[1], "status": row[2],
        "lines": row[3] or [], "result": row[4],
        "exit_code": row[5], "pgid": row[6], "cancel_requested": bool(row[7]),
    }


def job_cancel_requested(job_id: str) -> bool:
    engine = get_engine()
    with engine.connect() as conn:
        val = conn.execute(
            text(f"SELECT cancel_requested FROM {_JOBS_TABLE} WHERE job_id = :jid"),
            {"jid": job_id},
        ).scalar()
    return bool(val)


def request_job_cancel(job_id: str) -> dict | None:
    """Flag a running job for cancellation. Returns the job row (or None)."""
    job = get_job(job_id)
    if not job:
        return None
    if job["status"] in ("running", "cancelling"):
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(
                text(f"""UPDATE {_JOBS_TABLE}
                         SET cancel_requested = TRUE, status = 'cancelling',
                             updated_at = now()
                         WHERE job_id = :jid AND status = 'running'"""),
                {"jid": job_id},
            )
        job = get_job(job_id)
    return job


def recover_orphaned_jobs() -> int:
    """Mark jobs left 'running'/'cancelling' by a previous process as failed.
    Called once at app startup. Safe under single-worker deployment (--workers 1).
    If ever scaled to multiple workers, replace with heartbeat-based recovery:
    only fail jobs where updated_at < now() - 2min, since running jobs heartbeat every 2s."""
    ensure_jobs_table()
    engine = get_engine()
    with engine.begin() as conn:
        res = conn.execute(text(f"""
            UPDATE {_JOBS_TABLE}
            SET status = 'failed', exit_code = -1,
                lines = lines || '["Error: job orphaned by server restart"]'::jsonb,
                updated_at = now()
            WHERE status IN ('running', 'cancelling')
        """))
    return res.rowcount or 0


def cleanup_old_jobs(days: int = 14) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(
            f"DELETE FROM {_JOBS_TABLE} "
            f"WHERE created_at < now() - (:d || ' days')::interval"
        ), {"d": days})


import time as _time
_GLOBAL_START: tuple[float, str] | None = None  # (expires_monotonic, value)
_GLOBAL_START_TTL = 3600.0  # refresh after 1 hour — covers data backfills without restart

def get_global_start() -> str:
    """Return the earliest order_date across all SKUs, cached with a 1-hour TTL."""
    global _GLOBAL_START
    now = _time.monotonic()
    if _GLOBAL_START is not None and _GLOBAL_START[0] > now:
        return _GLOBAL_START[1]
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT MIN(order_date) FROM shipcore.fc_velocity_link_snapshot_forecast"))
        row = result.scalar()
    value = str(row) if row else "2024-06-17"
    _GLOBAL_START = (now + _GLOBAL_START_TTL, value)
    return value


def read_actuals(
    sku_id: str,
    n_weeks: int | None = 26,
    start_date: str | None = None,
    pad_from: str | None = None,
) -> pd.DataFrame:
    """Pull weekly actuals aggregated in SQL using the W-MON convention.

    - start_date: anchor from a fixed date (overrides n_weeks).
    - pad_from: extend the series back to this date with 0s for missing weeks.
    """
    engine = get_engine()
    fetch_anchor = pad_from or start_date
    # W-MON week label: first Monday >= order_date. Matches pandas W-MON Grouper
    # and the expression used in /all-skus, /accuracy-history, /demand-trend.
    week_expr = "(order_date + ((8 - EXTRACT(ISODOW FROM order_date))::int % 7) * INTERVAL '1 day')::date"
    base = f"""
        SELECT {week_expr} AS ds, SUM(link_qty) AS y
        FROM shipcore.fc_velocity_link_snapshot_forecast
        WHERE link_master_sku = :uid
    """
    params: dict = {"uid": sku_id}
    if fetch_anchor:
        fetch_from = (pd.Timestamp(fetch_anchor) - pd.Timedelta(days=6)).strftime("%Y-%m-%d")
        base += " AND order_date >= :fetch_from"
        params["fetch_from"] = fetch_from
    base += " GROUP BY 1 ORDER BY 1"

    with engine.connect() as conn:
        raw = pd.read_sql(text(base), conn, params=params)

    if raw.empty and not pad_from:
        return pd.DataFrame(columns=["ds", "y"])

    weekly = raw.copy()
    if not weekly.empty:
        weekly["ds"] = pd.to_datetime(weekly["ds"])
        weekly["y"]  = weekly["y"].fillna(0).astype(int)
    else:
        weekly = pd.DataFrame(columns=["ds", "y"])

    if pad_from:
        # Build a complete weekly grid from pad_from to today, fill gaps with 0
        today = pd.Timestamp.today().normalize()
        full_idx = pd.date_range(start=pd.Timestamp(pad_from), end=today, freq="W-MON")
        weekly = (
            weekly.set_index("ds")
            .reindex(full_idx, fill_value=0)
            .reset_index()
            .rename(columns={"index": "ds"})
        )
        weekly["y"] = weekly["y"].fillna(0).astype(int)
    elif start_date is not None:
        weekly = weekly[weekly["ds"] >= pd.Timestamp(start_date)].reset_index(drop=True)
    elif n_weeks is not None:
        weekly = weekly.tail(n_weeks).reset_index(drop=True)
    return weekly
