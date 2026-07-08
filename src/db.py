import os
from datetime import date, datetime
import pandas as pd
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

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
            pool_size=5,
            max_overflow=5,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 10, "sslmode": "require"},
        )
    return _engine


def write_forward_forecasts(df: pd.DataFrame) -> None:
    """Create table if needed, migrate PI columns, delete today's rows, insert fresh results."""
    engine = get_engine()
    forecast_date = str(df["forecast_date"].iloc[0])
    with engine.begin() as conn:
        conn.execute(text(_CREATE_SQL))
        conn.execute(text(_MIGRATE_PI_SQL))
        conn.execute(
            text(f"DELETE FROM {_TABLE} WHERE forecast_date = :fd"),
            {"fd": forecast_date},
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

        # Demand per SKU for the last N complete weeks
        demand_df = pd.read_sql(text(f"""
            SELECT link_master_sku, SUM(link_qty) AS demand
            FROM shipcore.fc_velocity_link_snapshot_forecast
            WHERE order_date > :start
              AND {pt_snap}
            GROUP BY link_master_sku
        """), conn, params={"start": period_start})

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
        ("smooth_short", "Smooth / Short history", "V1"),
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


_GLOBAL_START: str | None = None

def get_global_start() -> str:
    """Return the earliest order_date across all SKUs, cached for the process lifetime."""
    global _GLOBAL_START
    if _GLOBAL_START is None:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT MIN(order_date) FROM shipcore.fc_velocity_link_snapshot_forecast"))
            row = result.scalar()
        _GLOBAL_START = str(row) if row else "2024-06-17"
    return _GLOBAL_START


def read_actuals(
    sku_id: str,
    n_weeks: int | None = 26,
    start_date: str | None = None,
    pad_from: str | None = None,
) -> pd.DataFrame:
    """Pull weekly actuals.

    - start_date: anchor from a fixed date (overrides n_weeks).
    - pad_from: extend the series back to this date with 0s for missing weeks.
    """
    engine = get_engine()
    fetch_anchor = pad_from or start_date
    if fetch_anchor:
        fetch_from = (pd.Timestamp(fetch_anchor) - pd.Timedelta(days=6)).strftime("%Y-%m-%d")
        query = """
            SELECT order_date, link_qty
            FROM shipcore.fc_velocity_link_snapshot_forecast
            WHERE link_master_sku = :uid AND order_date >= :fetch_from
        """
        params: dict = {"uid": sku_id, "fetch_from": fetch_from}
    else:
        query = """
            SELECT order_date, link_qty
            FROM shipcore.fc_velocity_link_snapshot_forecast
            WHERE link_master_sku = :uid
        """
        params = {"uid": sku_id}

    with engine.connect() as conn:
        raw = pd.read_sql(text(query), conn, params=params)

    if raw.empty and not pad_from:
        return pd.DataFrame(columns=["ds", "y"])

    if not raw.empty:
        raw["order_date"] = pd.to_datetime(raw["order_date"])
        weekly = (
            raw.groupby(pd.Grouper(key="order_date", freq="W-MON"))["link_qty"]
            .sum()
            .reset_index()
            .rename(columns={"order_date": "ds", "link_qty": "y"})
            .sort_values("ds")
            .reset_index(drop=True)
        )
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
