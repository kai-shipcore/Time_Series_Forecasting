"""Backfill v1_yhat for a stored forecast run that predates the v1_yhat column.

V1 is deterministic given a cutoff date, so it can be reconstructed after the
fact: cutoff = the run's training cutoff (horizon start minus one week), same
value run_forward_forecast.py passes to compute_v1_per_week. Late-registered
orders can make the reconstruction differ marginally from what a live run
would have produced.

Usage: .venv/bin/python scripts/backfill_v1.py 2026-06-29
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from sqlalchemy import text

from src.db import get_engine
from src.v1 import load_raw_for_v1, build_index, compute_v1_per_week


def main(forecast_date: str) -> None:
    engine = get_engine()
    with engine.connect() as conn:
        run = pd.read_sql(text("""
            SELECT unique_id, ds, v1_yhat
            FROM shipcore.fc_forward_forecasts
            WHERE forecast_date = :fd
        """), conn, params={"fd": forecast_date}, parse_dates=["ds"])

    if run.empty:
        raise SystemExit(f"No stored forecast for forecast_date={forecast_date}")

    n_existing = run["v1_yhat"].notna().sum()
    if n_existing > 0:
        print(f"Run already has {n_existing} non-null v1_yhat rows — overwriting.")

    horizon_weeks = run["ds"].nunique()
    horizon_start = run["ds"].min()
    cutoff = horizon_start - pd.Timedelta(days=7)  # training cutoff (last complete Monday)
    unique_ids = run["unique_id"].unique().tolist()
    print(f"Run {forecast_date}: {len(unique_ids)} SKUs, horizon {horizon_weeks}w from {horizon_start.date()}, V1 cutoff {cutoff.date()}")

    raw = load_raw_for_v1()
    index = build_index(raw)
    v1_map = compute_v1_per_week(unique_ids, cutoff, horizon_weeks, index)

    with engine.begin() as conn:
        conn.execute(
            text("""
                UPDATE shipcore.fc_forward_forecasts
                SET v1_yhat = :v1
                WHERE forecast_date = :fd AND unique_id = :uid
            """),
            [{"v1": v1, "fd": forecast_date, "uid": uid} for uid, v1 in v1_map.items()],
        )
    print(f"Updated v1_yhat for {len(v1_map)} SKUs on run {forecast_date}.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
