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

import os
import shutil
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

    # ---- Staged run (BACKLOG item 15) --------------------------------------
    # Everything below writes into a staging directory beside data/processed,
    # and the artifacts are moved into place only after every step has
    # succeeded. Until that moment the live files are untouched, so a cancel, a
    # crash, a dropped SSH session or a failed step leaves the previous run
    # intact instead of a half-updated set.
    #
    # This is what the Stop button needed. The panel's Stop was removed on
    # 2026-08-05 and replaced with a "cannot be interrupted" notice, because
    # cancelling between the sales file and the profile left segmentation
    # describing last week and sales describing this one, with nothing on any
    # screen saying so.
    #
    # Beside data/processed rather than in /tmp, deliberately: os.replace is
    # atomic only within a filesystem, and /tmp is frequently a different one.
    #
    # HONEST LIMIT. Each file moves atomically; the set of four does not. There
    # is a window of milliseconds where some are new and some are old, against
    # the minutes the old behaviour left open. Closing it completely means
    # swapping a directory symlink, which changes how every reader resolves its
    # paths, and that is a larger change than this problem justifies.
    # Sweep staging directories left by earlier runs.
    #
    # The name carries the pid, so the check below only ever matches this
    # process and a directory abandoned by any other run is invisible to it.
    # Interrupting during the sync leaks one every time, because that step runs
    # before the try block that calls _abandon, and nothing else on the machine
    # removes them. Harmless individually, they are empty, but they accumulate
    # in data/ and the next person to look there finds a scatter of hidden
    # directories with no idea which run left them or whether any is live.
    #
    # Matching the glob rather than tracking pids: a stale directory's owner is
    # gone by definition, and a directory belonging to a concurrently running
    # prepare-data would be a second run against the same output files, which
    # api/main.py already prevents by giving both entry points the same job type
    # and returning 409.
    for stale in PROCESSED.parent.glob(".staging_*"):
        shutil.rmtree(stale, ignore_errors=True)

    staging = PROCESSED.parent / f".staging_{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    os.environ["FORECAST_PROCESSED_DIR"] = str(staging)

    def _abandon(why: str) -> None:
        shutil.rmtree(staging, ignore_errors=True)
        print(f"\n{why}\nStaging discarded. The previous run's files are "
              f"untouched and still being served.", flush=True)

    def _commit() -> None:
        """Move the staged artifacts into place, atomically per file."""
        moved = []
        for src_path in sorted(staging.iterdir()):
            dst = PROCESSED / src_path.name
            os.replace(src_path, dst)          # atomic within a filesystem
            moved.append(dst.name)
        shutil.rmtree(staging, ignore_errors=True)
        print(f"          committed {len(moved)} artifacts: {', '.join(moved)}",
              flush=True)

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
        # Interrupting here is common and was not handled. The step blocks for
        # minutes with no output, so someone watching it reasonably concludes it
        # has hung, and the KeyboardInterrupt then escaped past the try block
        # below and left the staging directory behind. Caught here so a cancel
        # during the sync says the same thing a cancel anywhere else does.
        #
        # Worth saying out loud, because the obvious next move is wrong: the
        # POST has already reached the app, and the app finishes the upsert
        # whether or not anything is still listening. Ctrl-C stops the waiting,
        # not the sync. Re-running immediately therefore starts a second upsert
        # on top of the first; --no-sync is the flag for picking up after one
        # that was already triggered.
        try:
            sync_velocity_snapshot()
        except KeyboardInterrupt:
            _abandon("Interrupted during the velocity sync.")
            print("The sync itself is still running on the app server: the "
                  "request was delivered before the interrupt. Give it a few "
                  "minutes, then re-run with --no-sync to use its result "
                  "rather than starting a second one.", flush=True)
            return 130

    # clean() and profile() read their own PROCESSED_DIR, which follows
    # FORECAST_PROCESSED_DIR set above. Reassigned explicitly as well, because
    # both modules were imported before the variable was set and read it at
    # import time.
    import src.clean as _clean_mod
    import src.profile as _profile_mod
    _clean_mod.PROCESSED_DIR = staging
    _profile_mod.PROCESSED_DIR = staging

    try:
        print("Step 2/4  ingest + clean: pulling orders from the database …", flush=True)
        raw = ingest()
        weekly = clean(raw)
        weekly.to_parquet(staging / SALES.name, index=False)
        print(f"          {len(weekly):,} weekly rows, "
              f"{weekly['unique_id'].nunique():,} SKUs -> {SALES.name}", flush=True)

        print("Step 3/4  profile: classifying each SKU …", flush=True)
        profiles = profile(weekly)
        profiles.to_csv(staging / PROFILES.name, index=False)
        counts = profiles["bucket"].value_counts().to_dict()
        print(f"          {len(profiles):,} SKUs -> {PROFILES.name}  {counts}", flush=True)
    except KeyboardInterrupt:
        _abandon("Interrupted.")
        return 130
    except Exception as exc:
        _abandon(f"Failed during ingest/clean/profile: {exc}")
        raise

    # A subprocess rather than an import: it is a script with its own argument
    # handling and its own database writes, and shelling out keeps one
    # definition of what a forward run does.
    # The subprocess inherits FORECAST_PROCESSED_DIR, so `--snapshot live`
    # resolves to the staging directory and it reads the sales and profile files
    # this run just wrote rather than the previous run's. Its own outputs land
    # there too.
    print("Step 4/4  forward forecast: training and predicting …", flush=True)
    try:
        proc = subprocess.run(
            [venv_python(), str(ROOT / "scripts" / "ml_forward_forecast.py"),
             "--snapshot", "live", "--horizon", str(args.horizon)],
            cwd=str(ROOT),
        )
    except KeyboardInterrupt:
        _abandon("Interrupted during the forward forecast.")
        return 130
    if proc.returncode != 0:
        _abandon(f"Forward forecast failed (exit {proc.returncode}).")
        return proc.returncode

    PROCESSED.mkdir(parents=True, exist_ok=True)
    _commit()
    print(f"\nDone in {time.time() - t0:.0f}s. The service can now serve; "
          "reload the page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
