#!/usr/bin/env python3
"""Re-profile an existing snapshot in place into a new one, same sales data.

    .venv/bin/python scripts/ml_36_reprofile_snapshot.py --from 2026-08-03 --to 2026-08-03-onset

No database. The point is isolation, not freshness.

Why this exists rather than scripts/ml_snapshot_data.py
------------------------------------------------------
That script snapshots whatever is currently in data/processed, which today also
contains a fresh weekly refresh. Using it now would bundle the profiling change
with three weeks of new and restated order data, and any figure that moved could
not be attributed to either.

This copies `sales_clean.parquet` byte for byte from the source snapshot and
regenerates `sku_profiles.csv` from that same file with the current profiling
code. The only difference between the two snapshots is the profiler, so a
re-baseline across them measures the profiler and nothing else.

That is the lesson from 2026-08-10: the v-base figures had been stale since v9
and it took a fresh run to notice, precisely because nobody could separate "the
code changed" from "the data changed".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from config import DATA_SNAPSHOTS  # noqa: E402

READ_ONLY = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--to", dest="dst", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src = DATA_SNAPSHOTS / args.src
    dst = DATA_SNAPSHOTS / args.dst
    if not (src / "sales_clean.parquet").is_file():
        print(f"no sales_clean.parquet in {src}")
        return 1
    if dst.exists() and not args.force:
        print(f"{dst} already exists. Pass --force to replace it.")
        return 1

    # Profiling writes sku_profiles.csv into its PROCESSED_DIR as a side effect.
    # Point it at the destination so the write lands where it is wanted and can
    # never touch data/processed or the source snapshot.
    import src.profile as P  # noqa: E402

    dst.mkdir(parents=True, exist_ok=True)
    P.PROCESSED_DIR = dst

    for f in dst.glob("*"):
        f.chmod(stat.S_IWUSR | stat.S_IRUSR)
    shutil.copy2(src / "sales_clean.parquet", dst / "sales_clean.parquet")

    weekly = pd.read_parquet(dst / "sales_clean.parquet")
    weekly["ds"] = pd.to_datetime(weekly["ds"])
    print(f"re-profiling {len(weekly):,} rows, {weekly['unique_id'].nunique():,} SKUs, "
          f"{weekly['ds'].nunique()} weeks")
    profiles = P.profile(weekly)

    same = md5(src / "sales_clean.parquet") == md5(dst / "sales_clean.parquet")
    manifest = {
        "snapshot": args.dst,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "derived_from": args.src,
        "purpose": (
            "Same sales data as the source snapshot, profiles regenerated with "
            "the current src/profile.py. Isolates a profiling change from any "
            "data change so a re-baseline attributes cleanly."
        ),
        "sales_clean_identical_to_source": same,
        "weekly_rows": len(weekly),
        "weekly_skus": int(weekly["unique_id"].nunique()),
        "week_first": str(weekly["ds"].min().date()),
        "week_last": str(weekly["ds"].max().date()),
        "buckets": profiles["bucket"].value_counts().to_dict(),
        "history_length": profiles["history_length"].value_counts().to_dict(),
        "files": {
            f: {"md5": md5(dst / f), "bytes": (dst / f).stat().st_size}
            for f in ("sales_clean.parquet", "sku_profiles.csv")
        },
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for f in ("sales_clean.parquet", "sku_profiles.csv", "manifest.json"):
        (dst / f).chmod(READ_ONLY)

    print(f"\nwrote {dst}")
    print(f"  sales_clean.parquet identical to source: {same}")
    if not same:
        print("  WARNING: it should be. Something rewrote it; do not use this "
              "snapshot for an attribution comparison.")
    print(f"  buckets: {manifest['buckets']}")
    print(f"  history_length: {manifest['history_length']}")
    print(f"\nSet ML_DATA_SNAPSHOT = \"{args.dst}\" to measure against it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
