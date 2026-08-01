"""Fill data/processed/ from the repository, so a fresh clone can serve.

Why this exists
---------------
`data/processed/` is gitignored, so a clone has the code and none of the data.
The service starts, answers /health, and raises on every real endpoint, which
reaches the browser as a bare "Internal Server Error". DEPLOYMENT.md calls this
"the original problem", and the answer it gives is to use the deployed app.
That is right for a colleague who only wants to read a forecast, and wrong for
one who has to run the planning pages locally to work on them.

Everything needed is already in the repository, just not where readiness()
looks:

    sales_clean.parquet     data/snapshots/2026-07-20/   (tracked, pinned)
    sku_profiles.csv        data/snapshots/2026-07-20/   (tracked, pinned)
    ml_forward_forecasts    data/dev_seed/               (tracked fixture)
    v1_forward_forecasts    data/dev_seed/               (tracked fixture)
    ml_accuracy*.csv        outputs/reports/             (already in place)
    ml_backtest_weekly.csv  outputs/reports/             (already in place)
    inventory_snapshot.csv  dashboard/data/              (already in place)

So this copies four files and nothing else. No database, no .env, no pipeline
run, no data handover.

Usage
-----
    .venv/bin/python scripts/seed_dev_data.py           # seed, refusing to clobber
    .venv/bin/python scripts/seed_dev_data.py --force   # overwrite what is there
    .venv/bin/python scripts/seed_dev_data.py --check   # report, write nothing

What it will not do
-------------------
Overwrite. If any target already exists the script names it and stops, because
on the machine that runs the weekly cron `data/processed/` is live data that is
newer than this fixture, and silently replacing it with a frozen July copy would
be the worst thing this script could do. `--force` is the only way past that,
and it says what it replaced.

Consistency
-----------
The seeded history and the seeded forecast have to meet. The forecast's
`forecast_date` is checked against the last week in the snapshot's sales file,
and a mismatch stops the run rather than producing a chart with a hole in it.
That check is the thing to look at first if the pinned snapshot is ever advanced
without rebuilding the fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_PROCESSED, DATA_SNAPSHOTS, ML_DATA_SNAPSHOT  # noqa: E402

DEV_SEED = ROOT / "data" / "dev_seed"

#: (filename, source directory). The snapshot is read rather than copied into
#: dev_seed, so there is one copy of the 850 KB pair in git and no way for a
#: second to drift from it.
FROM_SNAPSHOT = ("sales_clean.parquet", "sku_profiles.csv")
FROM_DEV_SEED = ("ml_forward_forecasts.parquet", "v1_forward_forecasts.parquet")


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def snapshot_dir() -> Path:
    """Where the pinned sales and profile files are.

    Follows config.ML_DATA_SNAPSHOT rather than naming a date here, so advancing
    the snapshot moves the seed with it instead of leaving this file pointing at
    a directory nobody uses any more.
    """
    if not ML_DATA_SNAPSHOT:
        raise SystemExit(
            "config.ML_DATA_SNAPSHOT is None, so there is no pinned snapshot to seed from.\n"
            "Set it to a folder under data/snapshots/, or copy data/processed/ by hand."
        )
    path = DATA_SNAPSHOTS / ML_DATA_SNAPSHOT
    if not path.is_dir():
        raise SystemExit(
            f"config.ML_DATA_SNAPSHOT is {ML_DATA_SNAPSHOT!r} but {path} does not exist."
        )
    return path


def sources() -> list[tuple[str, Path]]:
    snap = snapshot_dir()
    pairs = [(name, snap / name) for name in FROM_SNAPSHOT]
    pairs += [(name, DEV_SEED / name) for name in FROM_DEV_SEED]

    missing = [str(p.relative_to(ROOT)) for _, p in pairs if not p.is_file()]
    if missing:
        raise SystemExit(
            "These are tracked files and should be in any clone, so their absence means\n"
            "a partial checkout or a deleted file rather than a missing pipeline run:\n"
            + "".join(f"  {m}\n" for m in missing)
        )
    return pairs


def verify_seed_manifest() -> None:
    """Check the dev_seed fixtures against their recorded md5s.

    Cheap, and it catches the one failure that would otherwise be silent: a
    fixture rebuilt from a different run and committed without regenerating the
    manifest, which would seed a forecast that does not match what the README
    and the design doc say is seeded.
    """
    mpath = DEV_SEED / "manifest.json"
    if not mpath.is_file():
        print(f"  ! no manifest at {mpath.relative_to(ROOT)}; skipping checksum check")
        return
    manifest = json.loads(mpath.read_text())
    bad = []
    for name, meta in manifest.get("files", {}).items():
        path = DEV_SEED / name
        if not path.is_file():
            bad.append(f"{name}: recorded in the manifest but not present")
        elif md5(path) != meta["md5"]:
            bad.append(f"{name}: checksum differs from the manifest")
    if bad:
        raise SystemExit(
            "data/dev_seed does not match its manifest:\n"
            + "".join(f"  {b}\n" for b in bad)
            + "Rebuild the manifest if the fixture was replaced on purpose."
        )


def check_consistency(pairs: list[tuple[str, Path]]) -> str:
    """The seeded history and the seeded forecast must meet at the same week.

    Both files are pinned, so this can only fail when one of them is advanced
    and the other is not. Failing here is much better than the alternative,
    which is a demand chart with a gap or an overlap in it and nothing on screen
    saying why.
    """
    by_name = dict(pairs)
    sales = pd.read_parquet(by_name["sales_clean.parquet"], columns=["ds"])
    last_week = pd.to_datetime(sales["ds"]).max().date()

    fc = pd.read_parquet(by_name["ml_forward_forecasts.parquet"], columns=["forecast_date", "ds"])
    trained_through = pd.to_datetime(fc["forecast_date"]).max().date()
    horizon_start = pd.to_datetime(fc["ds"]).min().date()
    horizon_end = pd.to_datetime(fc["ds"]).max().date()

    if trained_through != last_week:
        raise SystemExit(
            f"The seed is inconsistent and would produce a broken chart.\n"
            f"  snapshot sales end   {last_week}\n"
            f"  forecast trained thru {trained_through}\n"
            "These must be the same week. Rebuild data/dev_seed from a run trained "
            "through the snapshot's last week, or point config.ML_DATA_SNAPSHOT at "
            "the snapshot the fixture was built from."
        )

    return f"history through {last_week}, forecast {horizon_start} to {horizon_end}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true",
                    help="overwrite files already in data/processed/")
    ap.add_argument("--check", action="store_true",
                    help="report what would happen and write nothing")
    args = ap.parse_args()

    pairs = sources()
    verify_seed_manifest()
    summary = check_consistency(pairs)

    present = [name for name, _ in pairs if (DATA_PROCESSED / name).is_file()]

    if args.check:
        print(f"Seed is consistent: {summary}")
        print(f"Target: {DATA_PROCESSED.relative_to(ROOT)}")
        for name, src in pairs:
            state = "present, would be kept" if name in present else "absent, would be written"
            print(f"  {name:<32} {state}   <- {src.relative_to(ROOT)}")
        if present:
            print(f"\n{len(present)} file(s) already there. Without --force this run would stop.")
        return 0

    if present and not args.force:
        print("Refusing to overwrite. data/processed/ already has:")
        for name in present:
            print(f"  {name}")
        print(
            "\nOn the machine that runs the weekly cron these are live data and newer than\n"
            "the fixture, so replacing them with a frozen July copy is almost certainly not\n"
            "what you want. Re-run with --force if it is."
        )
        return 1

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, src in pairs:
        dest = DATA_PROCESSED / name
        replaced = dest.is_file()
        shutil.copy2(src, dest)
        # copy2 carries the snapshot's read-only mode across, which would make a
        # later cron refresh fail on a file it is entitled to rewrite.
        dest.chmod(0o644)
        verb = "replaced" if replaced else "wrote"
        print(f"  {verb} {dest.relative_to(ROOT)}  ({dest.stat().st_size:,} bytes)")

    print(f"\nSeeded: {summary}")
    print(
        "\nThis is a frozen development fixture, not the current forecast. For figures to\n"
        "act on, use the deployed app.\n"
        "\nNext: .venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000\n"
        "Then GET /health should report ready: true."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
