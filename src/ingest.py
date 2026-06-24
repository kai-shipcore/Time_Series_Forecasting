# Stage 1: Pull data from DB → raw DataFrame
import os
import pandas as pd
from urllib.parse import quote_plus
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

_QUERY = """
    SELECT
        order_date,
        link_master_sku,
        link_qty
    FROM shipcore.fc_velocity_link_snapshot
"""


def _engine():
    url = "postgresql+psycopg2://{}:{}@{}:{}/{}".format(
        quote_plus(os.getenv("DB_USER")),
        quote_plus(os.getenv("DB_PASSWORD")),
        os.getenv("DB_HOST"),
        os.getenv("DB_PORT"),
        os.getenv("DB_NAME"),
    )
    return create_engine(url, connect_args={"connect_timeout": 10, "sslmode": "require"})


def ingest() -> pd.DataFrame:
    engine = _engine()
    with engine.connect() as conn:
        df = pd.read_sql(_QUERY, conn, parse_dates=["order_date"])
    return df


if __name__ == "__main__":
    df = ingest()
    print(f"Rows: {len(df)}")
    print(f"SKUs: {df['link_master_sku'].nunique()}")
    print(f"Date range: {df['order_date'].min().date()} to {df['order_date'].max().date()}")
    print(df.head())
