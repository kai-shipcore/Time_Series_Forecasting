"""
Create and populate shipcore.fc_forward_forecasts_test with two synthetic
historical forecast runs whose full horizons have already passed.

Run once:  python scripts/seed_backtest_test.py
Re-running is idempotent (table is dropped and recreated).
"""

import sys
import random
from pathlib import Path
import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.db import get_engine

random.seed(42)

# ── Two synthetic past runs ────────────────────────────────────────────────
# Both must have MAX(ds) <= last_monday so the /backtest-cycles endpoint
# picks them up as eligible.

RUNS = [
    {"forecast_date": "2026-03-02", "horizon_weeks": 13},  # ds: Mar 9 → Jun 1
    {"forecast_date": "2026-04-06", "horizon_weeks": 11},  # ds: Apr 13 → Jun 22
]

engine = get_engine()

with engine.begin() as conn:
    # Recreate test table
    conn.execute(text("""
        DROP TABLE IF EXISTS shipcore.fc_forward_forecasts_test;
        CREATE TABLE shipcore.fc_forward_forecasts_test (
            unique_id      TEXT             NOT NULL,
            forecast_date  DATE             NOT NULL,
            ds             DATE             NOT NULL,
            yhat           DOUBLE PRECISION NOT NULL,
            yhat_lo        DOUBLE PRECISION,
            yhat_hi        DOUBLE PRECISION,
            bucket         TEXT,
            history_length TEXT,
            selected_model TEXT,
            confidence     TEXT
        );
    """))
    print("Created shipcore.fc_forward_forecasts_test")

    # Pull real smooth SKUs from the latest live run (up to 40 SKUs).
    # Deliberately includes any SKU whose unique_id doesn't match a known
    # product type pattern (e.g. CA-CL-AT-CBL) — that's a real edge case
    # worth exercising: it shows under "All" but disappears under any
    # single product type filter, since _product_type_where only matches
    # CC%/CA-SC%/CL-SC%/CA-FM%/C-SJ-GR-7.
    skus_df = pd.read_sql(text("""
        WITH latest AS (
            SELECT unique_id, MAX(forecast_date) AS fd
            FROM shipcore.fc_forward_forecasts
            GROUP BY unique_id
        )
        SELECT f.unique_id, f.yhat, f.yhat_lo, f.yhat_hi,
               f.bucket, f.history_length, f.selected_model, f.confidence,
               ROW_NUMBER() OVER (PARTITION BY f.unique_id ORDER BY f.ds) AS wk
        FROM shipcore.fc_forward_forecasts f
        JOIN latest l ON f.unique_id = l.unique_id AND f.forecast_date = l.fd
        WHERE f.bucket = 'smooth'
        ORDER BY f.unique_id, f.ds
    """), conn)

    # Build a per-SKU "typical weekly demand" from the real forecast
    sku_means = (
        skus_df.groupby("unique_id")["yhat"]
        .mean()
        .reset_index()
        .rename(columns={"yhat": "mean_yhat"})
        .head(40)
    )
    sku_meta = (
        skus_df[["unique_id", "bucket", "history_length", "selected_model", "confidence"]]
        .drop_duplicates("unique_id")
        .merge(sku_means, on="unique_id")
    )

    rows_inserted = 0
    for run in RUNS:
        forecast_date = pd.Timestamp(run["forecast_date"])
        # Weekly Mondays starting the day after forecast_date
        # (the forecast_date itself is the last training day)
        first_ds = forecast_date + pd.Timedelta(weeks=1)
        ds_dates = [first_ds + pd.Timedelta(weeks=i) for i in range(run["horizon_weeks"])]

        for _, sku_row in sku_meta.iterrows():
            uid = sku_row["unique_id"]
            mean = float(sku_row["mean_yhat"])

            # Give each SKU a fixed bias multiplier for this run so accuracy
            # metrics vary realistically across runs
            sku_bias = random.uniform(0.65, 1.35)
            # Only smooth_full (StatsForecast) produces P70 intervals; V1 (smooth_short) does not
            has_pi   = sku_row["history_length"] in ("full", "medium")

            for ds in ds_dates:
                # Add small week-to-week noise on top of the bias
                noise     = random.gauss(1.0, 0.15)
                yhat      = max(0.0, round(mean * sku_bias * noise, 2))
                yhat_lo   = max(0.0, round(yhat * random.uniform(0.55, 0.80), 2)) if has_pi else None
                yhat_hi   = max(yhat, round(yhat * random.uniform(1.20, 1.65), 2)) if has_pi else None

                conn.execute(text("""
                    INSERT INTO shipcore.fc_forward_forecasts_test
                        (unique_id, forecast_date, ds, yhat, yhat_lo, yhat_hi,
                         bucket, history_length, selected_model, confidence)
                    VALUES
                        (:uid, :fd, :ds, :yhat, :yhat_lo, :yhat_hi,
                         :bucket, :history_length, :selected_model, :confidence)
                """), {
                    "uid":            uid,
                    "fd":             forecast_date.date(),
                    "ds":             ds.date(),
                    "yhat":           yhat,
                    "yhat_lo":        yhat_lo,
                    "yhat_hi":        yhat_hi,
                    "bucket":         sku_row["bucket"],
                    "history_length": sku_row["history_length"],
                    "selected_model": sku_row["selected_model"],
                    "confidence":     sku_row["confidence"],
                })
                rows_inserted += 1

    print(f"Inserted {rows_inserted} rows across {len(RUNS)} test runs "
          f"({len(sku_meta)} SKUs each)")

# Verify what the backtest-cycles endpoint will see
with engine.connect() as conn:
    check = conn.execute(text("""
        SELECT forecast_date, MIN(ds) AS h_start, MAX(ds) AS h_end,
               COUNT(DISTINCT ds) AS weeks, COUNT(DISTINCT unique_id) AS skus
        FROM shipcore.fc_forward_forecasts_test
        GROUP BY forecast_date
        ORDER BY forecast_date
    """)).fetchall()
    print("\nTest cycles created:")
    for r in check:
        print(f"  {r[0]}  {r[1]} → {r[2]}  ({r[3]}W, {r[4]} SKUs)")
