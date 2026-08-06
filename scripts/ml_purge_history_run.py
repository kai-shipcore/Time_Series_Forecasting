#!/usr/bin/env python3
"""Remove one forecast run from the accumulating history.

The history store is append-only by design: it records what was predicted
before the outcome was known, which is the one claim a backtest cannot make.
Deleting from it is therefore not routine, and this script is deliberately
awkward -- explicit date, dry run by default, one run at a time.

Why it exists. On 2026-08-05 a run was triggered mid-week while
`clean()` still kept partial trailing buckets, so the model trained on a
three-day "week" and the run was stored under `forecast_date = 2026-08-10`.
That row is not a forecast anyone made a decision on and it is not a fair
record of the model: scored against actuals as those weeks close it will look
far worse than v11 deserves, and it would sit permanently in the section of the
validation page whose whole point is honest out-of-sample evidence.

Deleting it is the right call precisely because the store is evidence. A run
built on known-broken input is not evidence of anything except the bug, and the
bug is recorded in WORKLOG and BACKLOG where it belongs.

What this does NOT do, on purpose:

  - It does not touch ml_forward_forecasts. That file is the current forecast
    and is overwritten by the next good run anyway; deleting it would leave the
    planning screens with nothing to serve until then.
  - It does not re-run anything. Produce the replacement separately, after the
    binning fix, so the new run is stored under the cutoff it actually used.

Both stores are cleaned: the parquet on this machine and the shipcore table.
The table is the shared record, so a purge that only touched the file would be
undone the next time anything read from the database.

    # see what would go
    .venv/bin/python scripts/ml_purge_history_run.py --forecast-date 2026-08-10

    # actually remove it
    .venv/bin/python scripts/ml_purge_history_run.py --forecast-date 2026-08-10 --apply
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.ml.serving import history, store  # noqa: E402


def _rel(path: Path) -> str:
    """Path for display, shortened when it sits inside the repo.

    `relative_to` raises when it does not, which is a silly reason for a purge
    script to abort partway. Only ever used for printing.
    """
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _summarise(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        print(f"  {label}: nothing matches")
        return
    print(f"  {label}: {len(df):,} rows | {df['unique_id'].nunique()} SKUs "
          f"| versions {sorted(df['model_version'].unique())}")
    print(f"    target weeks {pd.to_datetime(df['ds']).min().date()} "
          f"-> {pd.to_datetime(df['ds']).max().date()}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--forecast-date", required=True,
                    help="the run's training cutoff, e.g. 2026-08-10")
    ap.add_argument("--version", default=None,
                    help="restrict to one model_version (default: every version on that date)")
    ap.add_argument("--apply", action="store_true",
                    help="actually delete; without this the script only reports")
    args = ap.parse_args()

    target = pd.Timestamp(args.forecast_date).normalize()

    # ---- the parquet -------------------------------------------------------
    # Read the file directly rather than through history.load(), which prefers
    # the table and normalises segments. This has to see the file exactly as it
    # is in order to rewrite it.
    local = pd.DataFrame()
    if history.HISTORY_PATH.exists():
        local = pd.read_parquet(history.HISTORY_PATH)
        local["forecast_date"] = pd.to_datetime(local["forecast_date"])
        hit = local["forecast_date"] == target
        if args.version:
            hit &= local["model_version"] == args.version
        print(f"\nparquet  {_rel(history.HISTORY_PATH)}")
        print(f"  holds {len(local):,} rows across "
              f"{local['forecast_date'].dt.date.nunique()} runs")
        _summarise(local[hit], "to remove")
    else:
        print(f"\nparquet  {_rel(history.HISTORY_PATH)} does not exist")
        hit = pd.Series(dtype=bool)

    # ---- the table ---------------------------------------------------------
    print(f"\ntable    {store.TABLE}")
    remote = store.read(history.COLUMNS)
    if remote is None:
        print("  unreachable (no credentials, or the database is down)")
        print("  NOTE: the table is the shared record. Purging only the file "
              "leaves the bad run in place for anyone reading from the database.")
    else:
        remote["forecast_date"] = pd.to_datetime(remote["forecast_date"])
        rhit = remote["forecast_date"] == target
        if args.version:
            rhit &= remote["model_version"] == args.version
        print(f"  holds {len(remote):,} rows across "
              f"{remote['forecast_date'].dt.date.nunique()} runs")
        _summarise(remote[rhit], "to remove")

    if not args.apply:
        print("\nDry run. Nothing was changed. Re-run with --apply to delete.")
        return 0

    # ---- delete ------------------------------------------------------------
    if not local.empty and hit.any():
        # Backup before touching it. The weekly cron already keeps dated copies
        # under data/history_backups for exactly this reason, and a script whose
        # whole job is deleting from an append-only store has less excuse than
        # the cron does for not taking one. Named for the purge rather than the
        # date so it cannot collide with the cron's own copies.
        backups = ROOT / "data" / "history_backups"
        backups.mkdir(parents=True, exist_ok=True)
        stamp = pd.Timestamp.now().strftime("%Y-%m-%dT%H%M%S")
        backup = backups / f"ml_forecast_history_before_purge_{stamp}.parquet"
        local.to_parquet(backup, index=False)
        print(f"\nbacked up {len(local):,} rows to {_rel(backup)}")

        kept = local[~hit]
        # Written whole rather than edited in place: the file is small and a
        # partial write on a store described as irreplaceable is not a risk
        # worth taking for the milliseconds it saves.
        kept.to_parquet(history.HISTORY_PATH, index=False)
        print(f"parquet: removed {int(hit.sum()):,} rows, {len(kept):,} remain")

    if remote is not None:
        from sqlalchemy import text
        eng = store.engine()
        sql = f"DELETE FROM {store.TABLE} WHERE forecast_date = :d"
        params = {"d": target.date()}
        if args.version:
            sql += " AND model_version = :v"
            params["v"] = args.version
        with eng.begin() as conn:
            deleted = conn.execute(text(sql), params).rowcount
        print(f"table:   removed {deleted:,} rows")

    print("\nDone. Produce the replacement run separately, after the binning fix, "
          "so it is stored under the cutoff it actually used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
