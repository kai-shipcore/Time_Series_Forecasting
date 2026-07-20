#!/usr/bin/env python3
"""Export forecast tables from shipcore to local parquet files.

Purpose: make stored predictions available for offline analysis — in
particular, verifying the accuracy figures cited in
docs/ML_FORECAST_DESIGN.md (e.g., WA12 0.218 vs V1 0.291 pooled WAPE on
short-history SKUs, Apr–Jun 2026 backtest) against the database of record.

Run locally (requires DB access):
    python scripts/export_forecast_history.py                # default tables
    python scripts/export_forecast_history.py <table> ...    # specific tables

Default tables:
    fc_forecast_history        — one row per SKU per weekly run (13w totals)
    fc_forward_forecasts_test  — backtest/evaluation forecasts

Output: data/processed/<table>.parquet (one file per table, full replace).

Notes:
- Table schemas are discovered at runtime via information_schema, because
  the real schemas differ from the project documentation (e.g.,
  fc_forecast_history has no per-week "ds" column).
- Conformal interval columns (yhat_lo_*/yhat_hi_*) are excluded to keep
  exports small. Extend `keep` if interval analysis is ever needed.
"""
import os
import sys
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "data" / "processed"

DEFAULT_TABLES = ["fc_forecast_history", "fc_forward_forecasts_test"]

# override=True: the repo's .env is the source of truth for this script.
# Without it, stale DB_* variables exported in the user's shell silently
# take precedence (observed 2026-07-16: an outdated shell export caused
# password-authentication failures while .env held valid credentials).
load_dotenv(ROOT / ".env", override=True)

SCHEMA_QUERY = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = 'shipcore' AND table_name = %(table)s
    ORDER BY ordinal_position
"""


def export_table(conn, table: str) -> None:
    cols = pd.read_sql(SCHEMA_QUERY, conn, params={"table": table})["column_name"].tolist()
    if not cols:
        print(f"\n{table}: NOT FOUND in shipcore schema — skipped")
        return
    print(f"\n{table} — columns ({len(cols)}): {cols}")

    keep = [c for c in cols if not (c.startswith("yhat_lo") or c.startswith("yhat_hi"))]
    print(f"exporting ({len(keep)}): {keep}")

    col_list = ", ".join(f'"{c}"' for c in keep)
    df = pd.read_sql(f'SELECT {col_list} FROM shipcore."{table}"', conn)
    if df.empty:
        print(f"{table}: 0 rows — nothing exported")
        return

    # Parse date-like columns only; leave numerics (e.g., horizon_weeks) alone.
    for c in df.columns:
        if df[c].dtype == object and ("date" in c.lower() or "week" in c.lower() or c == "ds"):
            try:
                df[c] = pd.to_datetime(df[c])
            except (ValueError, TypeError):
                pass

    out = OUT_DIR / f"{table}.parquet"
    df.to_parquet(out, index=False)
    print(f"saved {len(df):,} rows → {out}")
    print(df.head(3).to_string())


def main() -> None:
    tables = sys.argv[1:] or DEFAULT_TABLES
    url = "postgresql+psycopg2://{}:{}@{}:{}/{}".format(
        quote_plus(os.getenv("DB_USER")),
        quote_plus(os.getenv("DB_PASSWORD")),
        os.getenv("DB_HOST"),
        os.getenv("DB_PORT"),
        os.getenv("DB_NAME"),
    )
    engine = create_engine(url, connect_args={"connect_timeout": 10, "sslmode": "require"})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with engine.connect() as conn:
        for table in tables:
            export_table(conn, table)


if __name__ == "__main__":
    main()
