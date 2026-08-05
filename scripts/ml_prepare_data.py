"""Generate the ML track's data files from the database, for a machine that has none.

    .venv/bin/python scripts/ml_prepare_data.py
    .venv\\Scripts\\python.exe scripts\\ml_prepare_data.py     (Windows)

sync -> ingest -> clean -> profile -> forward forecast, which is the shortest
path from credentials to a machine that can serve the planning pages with
current data.

Also the on-demand path behind the Action List's Run Forecast panel, which
calls it with --force through POST /planning/run-forecast. That is why the step
lines are printed in a fixed "Step N/4" form and flushed: the API streams this
script's stdout and the browser recognises progress by that prefix.

Why this is a separate script from run_forward_forecast.py
---------------------------------------------------------
That one is the legacy statsforecast pipeline. It shares the first three steps
and then diverges: it selects per-SKU models by cross-validation and writes to
`shipcore.fc_forward_forecasts`, leaving `ml_forward_forecasts.parquet`
untouched. Running it to fix missing ML data would regenerate `sku_profiles.csv`
and so move segmentation underneath an ML forecast that did not change, which is
the failure `docs/BACKLOG.md` item 7 describes. This runs the ML side instead.

What it needs, and what it is not
---------------------------------
DB_* in .env. Every file it writes is derived from the orders table, so there is
no version of this that works without credentials; a machine without them wants
`scripts/seed_dev_data.py`, which copies committed fixtures in about a second.

It is also not the weekly run. That is `run_forecast_cron.sh` on the server. This
exists so a developer with credentials and an empty `data/processed/` can reach a
working state in one command instead of reading the pipeline to find out which
three scripts to run in which order.

Refuses to overwrite by default, for the same reason the seed does: on the
machine that runs the cron those files are live, and this pulls the entire order
history to rebuild them.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data" / "processed"
SALES = PROCESSED / "sales_clean.parquet"
PROFILES = PROCESSED / "sku_profiles.csv"
FORECAST = PROCESSED / "ml_forward_forecasts.parquet"


def venv_python() -> str:
    """This interpreter. Subprocesses inherit it rather than guessing a path."""
    return sys.executable


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true",
                    help="rebuild even when the files are already present")
    ap.add_argument("--horizon", type=int, default=13)
    ap.add_argument("--no-sync", action="store_true",
                    help="skip the velocity sync and ingest whatever was last synced")
    args = ap.parse_args()

    present = [p.name for p in (SALES, PROFILES, FORECAST) if p.exists()]
    if present and not args.force:
        print("Already present, nothing to do:")
        for name in present:
            print(f"  {name}")
        print("\nRe-run with --force to rebuild from the database. On the machine that")
        print("runs the weekly cron these are live data, so that is not the default.")
        return 0

    # Imported here rather than at module scope: both pull in sqlalchemy and the
    # ML stack, and the --help and already-present paths above should not need
    # either to be installed.
    from src.clean import clean
    from src.ingest import ingest
    from src.profile import profile
    from src.velocity_sync import sync_velocity_snapshot

    t0 = time.time()

    # Ahead of the ingest, because ingest() reads
    # shipcore.fc_velocity_link_snapshot_forecast and the sync is what refreshes
    # it. Without this the script pulled whatever had last been synced and
    # reported it as fresh: on a machine that had never run the weekly cron,
    # that could be an empty table, and on one that had, it was silently as old
    # as the last Monday.
    #
    # Never fatal, matching run_forecast_cron.sh: a forecast on slightly stale
    # velocity beats no forecast, and sync_velocity_snapshot prints what
    # happened either way.
    if args.no_sync:
        print("Step 1/4  sync: skipped (--no-sync)", flush=True)
    else:
        print("Step 1/4  sync: refreshing the velocity snapshot …", flush=True)
        sync_velocity_snapshot()

    print("Step 2/4  ingest + clean: pulling orders from the database …", flush=True)
    raw = ingest()
    weekly = clean(raw)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(SALES, index=False)
    print(f"          {len(weekly):,} weekly rows, "
          f"{weekly['unique_id'].nunique():,} SKUs -> {SALES.name}", flush=True)

    print("Step 3/4  profile: classifying each SKU …", flush=True)
    profiles = profile(weekly)
    profiles.to_csv(PROFILES, index=False)
    counts = profiles["bucket"].value_counts().to_dict()
    print(f"          {len(profiles):,} SKUs -> {PROFILES.name}  {counts}", flush=True)

    # A subprocess rather than an import: it is a script with its own argument
    # handling and its own database writes, and shelling out keeps one
    # definition of what a forward run does.
    print("Step 4/4  forward forecast: training and predicting …", flush=True)
    proc = subprocess.run(
        [venv_python(), str(ROOT / "scripts" / "ml_forward_forecast.py"),
         "--snapshot", "live", "--horizon", str(args.horizon)],
        cwd=str(ROOT),
    )
    if proc.returncode != 0:
        print(f"\nForward forecast failed (exit {proc.returncode}). "
              "The sales and profile files above were written and are reusable.")
        return proc.returncode

    print(f"\nDone in {time.time() - t0:.0f}s. The service can now serve; "
          "reload the page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
