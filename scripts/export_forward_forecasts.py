#!/usr/bin/env python3
"""Export the live forward-forecast table from shipcore for the dashboard.

Purpose: snapshot shipcore.fc_forward_forecasts to
data/processed/fc_forward_forecasts.parquet so the dashboard can be pointed at
current forecasts instead of the stale fc_forward_forecasts_test snapshot
(docs/PLANNING_REQUIREMENTS.md, "Known limitations").

Run locally (requires DB access):
    .venv/bin/python scripts/export_forward_forecasts.py

Output: data/processed/fc_forward_forecasts.parquet (full replace).

Notes:
- The column list is fixed to match fc_forward_forecasts_test.parquet exactly
  (see src/planning/data.py) so the file is a drop-in replacement. Conformal
  interval columns (yhat_lo_*/yhat_hi_*) are excluded, matching the exclusion
  in export_forecast_history.py.
- Unlike the test snapshot (one run), the live table accumulates one row set
  per weekly training run (write_forward_forecasts in src/db.py only deletes
  rows sharing the new run's horizon start). The dashboard filters to the
  latest forecast_date before use, so row/SKU counts are reported both for
  the full export and for that latest run.
"""
import os
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "data" / "processed"
TABLE = "fc_forward_forecasts"

COLUMNS = [
    "unique_id",
    "forecast_date",
    "ds",
    "yhat",
    "bucket",
    "history_length",
    "selected_model",
    "confidence",
    "active_weeks",
    "v1_yhat",
]

# override=True: the repo's .env is the source of truth for this script.
# Without it, stale DB_* variables exported in the user's shell silently
# take precedence (see scripts/export_forecast_history.py).
load_dotenv(ROOT / ".env", override=True)


def main() -> None:
    url = "postgresql+psycopg2://{}:{}@{}:{}/{}".format(
        quote_plus(os.getenv("DB_USER")),
        quote_plus(os.getenv("DB_PASSWORD")),
        os.getenv("DB_HOST"),
        os.getenv("DB_PORT"),
        os.getenv("DB_NAME"),
    )
    engine = create_engine(url, connect_args={"connect_timeout": 10, "sslmode": "require"})

    col_list = ", ".join(f'"{c}"' for c in COLUMNS)
    with engine.connect() as conn:
        df = pd.read_sql(f'SELECT {col_list} FROM shipcore."{TABLE}"', conn)

    if df.empty:
        print(f"{TABLE}: 0 rows — nothing exported")
        return

    df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    df["ds"] = pd.to_datetime(df["ds"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{TABLE}.parquet"
    df.to_parquet(out, index=False)

    print(f"saved {len(df):,} rows -> {out}")
    print(f"unique SKUs (all runs in file): {df['unique_id'].nunique():,}")

    by_run = df.groupby("forecast_date")["unique_id"].agg(rows="size", unique_skus="nunique")
    print(f"\nruns present ({len(by_run)}):")
    print(by_run.to_string())

    latest = df["forecast_date"].max()
    cur = df[df["forecast_date"] == latest]
    print(f"\nlatest run ({latest.date()}) — {len(cur):,} rows, "
          f"{cur['unique_id'].nunique():,} unique SKUs:")
    print(cur.groupby("bucket")["unique_id"].nunique().rename("unique_skus").to_string())


if __name__ == "__main__":
    main()
