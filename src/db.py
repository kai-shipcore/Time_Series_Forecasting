import os
import pandas as pd
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

_TABLE = "shipcore.fc_forward_forecasts"

_CREATE_SQL = f"""
CREATE TABLE IF NOT EXISTS {_TABLE} (
    unique_id      TEXT  NOT NULL,
    forecast_date  DATE  NOT NULL,
    ds             DATE  NOT NULL,
    yhat           FLOAT NOT NULL,
    yhat_lo        FLOAT,
    yhat_hi        FLOAT,
    bucket         TEXT,
    history_length TEXT,
    selected_model TEXT,
    confidence     TEXT,
    PRIMARY KEY (unique_id, forecast_date, ds)
)
"""


def get_engine():
    url = "postgresql+psycopg2://{}:{}@{}:{}/{}".format(
        quote_plus(os.getenv("DB_USER")),
        quote_plus(os.getenv("DB_PASSWORD")),
        os.getenv("DB_HOST"),
        os.getenv("DB_PORT"),
        os.getenv("DB_NAME"),
    )
    return create_engine(url, connect_args={"connect_timeout": 10, "sslmode": "require"})


def write_forward_forecasts(df: pd.DataFrame) -> None:
    """Create table if needed, delete today's existing rows, insert fresh results."""
    engine = get_engine()
    forecast_date = str(df["forecast_date"].iloc[0])
    with engine.begin() as conn:
        conn.execute(text(_CREATE_SQL))
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


def read_segments(weeks: int = 10) -> dict:
    """Return SKU counts and demand totals per segment for the last N complete weeks.

    Forecasted SKUs (smooth) come from fc_forward_forecasts.
    Everything else in the snapshot is treated as intermittent.
    """
    engine = get_engine()

    today = pd.Timestamp.today().normalize()
    days_back = today.dayofweek or 7
    last_monday = today - pd.Timedelta(days=days_back)
    period_start = last_monday - pd.Timedelta(weeks=weeks)

    with engine.connect() as conn:
        # Latest segment classification for every forecasted SKU
        forecast_df = pd.read_sql(text(f"""
            SELECT DISTINCT unique_id, bucket, history_length
            FROM {_TABLE}
            WHERE forecast_date = (SELECT MAX(forecast_date) FROM {_TABLE})
        """), conn)

        # All SKUs ever seen — so dormant SKUs (no recent sales) still count
        all_skus_df = pd.read_sql(text("""
            SELECT DISTINCT link_master_sku
            FROM shipcore.fc_velocity_link_snapshot
        """), conn)

        # Demand per SKU for the last N complete weeks
        demand_df = pd.read_sql(text("""
            SELECT link_master_sku, SUM(link_qty) AS demand
            FROM shipcore.fc_velocity_link_snapshot
            WHERE order_date > :start
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
    if 0 < n_fcast <= n_skus:
        ann_idx = n_fcast - 1
        pareto_annotation = {
            "sku_pct":    round(pareto_x[ann_idx], 1),
            "demand_pct": round(pareto_y[ann_idx], 1),
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


def read_actuals(sku_id: str, n_weeks: int | None = 26, start_date: str | None = None) -> pd.DataFrame:
    """Pull weekly actuals. Pass start_date (YYYY-MM-DD) to anchor from a fixed date; otherwise tail n_weeks."""
    engine = get_engine()
    if start_date:
        start_ts = pd.Timestamp(start_date)
        # W-MON periods END on Monday, so the period labeled start_date spans the 6 days before it too.
        # Fetch 6 days earlier so the first period is complete, then filter after grouping.
        fetch_from = (start_ts - pd.Timedelta(days=6)).strftime("%Y-%m-%d")
        query = """
            SELECT order_date, link_qty
            FROM shipcore.fc_velocity_link_snapshot
            WHERE link_master_sku = :uid AND order_date >= :fetch_from
        """
        params: dict = {"uid": sku_id, "fetch_from": fetch_from}
    else:
        query = """
            SELECT order_date, link_qty
            FROM shipcore.fc_velocity_link_snapshot
            WHERE link_master_sku = :uid
        """
        params = {"uid": sku_id}

    with engine.connect() as conn:
        raw = pd.read_sql(text(query), conn, params=params)

    if raw.empty:
        return pd.DataFrame(columns=["ds", "y"])

    raw["order_date"] = pd.to_datetime(raw["order_date"])
    weekly = (
        raw.groupby(pd.Grouper(key="order_date", freq="W-MON"))["link_qty"]
        .sum()
        .reset_index()
        .rename(columns={"order_date": "ds", "link_qty": "y"})
        .sort_values("ds")
        .reset_index(drop=True)
    )
    if start_date is not None:
        weekly = weekly[weekly["ds"] >= pd.Timestamp(start_date)].reset_index(drop=True)
    elif n_weeks is not None:
        weekly = weekly.tail(n_weeks).reset_index(drop=True)
    return weekly
