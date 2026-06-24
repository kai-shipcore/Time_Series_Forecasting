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
