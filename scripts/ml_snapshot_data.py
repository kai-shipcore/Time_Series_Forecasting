"""Create a pinned data snapshot for the ML development track.

Why this exists
---------------
The weekly cron rewrites data/processed/ in place: it revises recent actuals as
late orders register and regenerates the SKU profile snapshot. ML_FINAL_TEST_CUTOFF
pins which WEEKS each evaluation window covers, but not the numbers inside them,
so two model versions evaluated a week apart are not measured on the same data.
The v3 entry in the design doc records exactly this (baseline figures moved in the
third decimal after the 2026-07-20 refresh).

A snapshot is an immutable copy of the two ML inputs. The ML harness
(src/ml/dataset.py) reads from it; the production pipeline keeps reading
data/processed/ and keeps following the cron. The two never share a file, so the
refresh cannot desync model training.

Usage
-----
  # create today's snapshot from the current processed files
  .venv/bin/python scripts/ml_snapshot_data.py

  # a specific label, and overwrite if it exists
  .venv/bin/python scripts/ml_snapshot_data.py --date 2026-07-27 --force

  # check a snapshot still matches its manifest
  .venv/bin/python scripts/ml_snapshot_data.py --verify 2026-07-20

After creating a snapshot, point config.ML_DATA_SNAPSHOT at it. That is a
deliberate step: advancing the snapshot changes every recorded number, so the
affected results have to be re-baselined (design doc Section 4.14).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_PROCESSED, DATA_SNAPSHOTS  # noqa: E402

FILES = ("sales_clean.parquet", "sku_profiles.csv")
READ_ONLY = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def describe(dest: Path) -> dict:
    """Content summary used to prove a snapshot has not changed."""
    weekly = pd.read_parquet(dest / "sales_clean.parquet")
    profiles = pd.read_csv(dest / "sku_profiles.csv")
    ds = pd.to_datetime(weekly["ds"])

    buckets = profiles["bucket"].value_counts().to_dict() if "bucket" in profiles else {}
    hist = (
        profiles["history_length"].value_counts().to_dict()
        if "history_length" in profiles
        else {}
    )
    return {
        "weekly_rows": int(len(weekly)),
        "weekly_skus": int(weekly["unique_id"].nunique()),
        "week_first": str(ds.min().date()),
        "week_last": str(ds.max().date()),
        "n_weeks": int(ds.nunique()),
        "profile_rows": int(len(profiles)),
        "buckets": {str(k): int(v) for k, v in buckets.items()},
        "history_length": {str(k): int(v) for k, v in hist.items()},
        "files": {f: {"md5": md5(dest / f), "bytes": (dest / f).stat().st_size} for f in FILES},
    }


def create(label: str, force: bool) -> Path:
    missing = [f for f in FILES if not (DATA_PROCESSED / f).is_file()]
    if missing:
        raise SystemExit(
            f"Cannot snapshot: {', '.join(missing)} not found in {DATA_PROCESSED}. "
            f"Run the ingest pipeline first."
        )

    dest = DATA_SNAPSHOTS / label
    if dest.exists():
        if not force:
            raise SystemExit(
                f"Snapshot '{label}' already exists at {dest}. Pass --force to "
                f"overwrite it, but note that overwriting a snapshot invalidates "
                f"every result recorded against it."
            )
        for f in FILES:
            target = dest / f
            if target.exists():
                target.chmod(stat.S_IWUSR | READ_ONLY)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Snapshot '{label}' → {dest}")
    for f in FILES:
        shutil.copy2(DATA_PROCESSED / f, dest / f)
        print(f"  copied {f}")

    manifest = {
        "snapshot": label,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": str(DATA_PROCESSED),
        "purpose": (
            "Pinned inputs for the ML development track (src/ml/dataset.py). "
            "Immutable: the weekly cron refreshes data/processed/ only."
        ),
        **describe(dest),
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("  wrote manifest.json")

    # Read-only so an accidental write fails loudly instead of desyncing training.
    for f in FILES:
        (dest / f).chmod(READ_ONLY)

    print(
        f"\n  {manifest['weekly_rows']:,} weekly rows | "
        f"{manifest['weekly_skus']:,} SKUs | "
        f"{manifest['week_first']} → {manifest['week_last']} "
        f"({manifest['n_weeks']} weeks)"
    )
    if manifest["buckets"]:
        print(f"  buckets: {manifest['buckets']}")
    print(f"\nTo use it, set ML_DATA_SNAPSHOT = \"{label}\" in config.py.")
    return dest


def verify(label: str) -> int:
    dest = DATA_SNAPSHOTS / label
    mpath = dest / "manifest.json"
    if not mpath.is_file():
        raise SystemExit(f"No manifest at {mpath}; cannot verify snapshot '{label}'.")

    manifest = json.loads(mpath.read_text())
    print(f"Verifying snapshot '{label}' (created {manifest.get('created_at', '?')})")

    failures = []
    for f, meta in manifest["files"].items():
        actual = md5(dest / f)
        ok = actual == meta["md5"]
        print(f"  {'OK  ' if ok else 'FAIL'} {f}  {actual}")
        if not ok:
            failures.append(f)

    if failures:
        print(f"\n{len(failures)} file(s) changed since the snapshot was created.")
        print("Any result recorded against this snapshot is no longer reproducible.")
        return 1
    print(
        f"\nUnchanged. {manifest['weekly_rows']:,} weekly rows | "
        f"{manifest['weekly_skus']:,} SKUs | "
        f"{manifest['week_first']} → {manifest['week_last']}"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--date", default=date.today().isoformat(),
                    help="Snapshot label, default today (YYYY-MM-DD).")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing snapshot with the same label.")
    ap.add_argument("--verify", metavar="LABEL",
                    help="Verify an existing snapshot against its manifest and exit.")
    args = ap.parse_args()

    if args.verify:
        return verify(args.verify)
    create(args.date, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
