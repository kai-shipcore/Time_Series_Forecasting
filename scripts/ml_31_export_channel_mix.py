#!/usr/bin/env python3
"""Export per-SKU per-week channel mix, aligned to the pinned snapshot.

Needs the database. Run on a machine with DB_* credentials:

    .venv/bin/python scripts/ml_31_export_channel_mix.py

Writes data/snapshots/<ML_DATA_SNAPSHOT>/channel_mix.parquet.

Why a SEPARATE file rather than a column on sales_clean
-------------------------------------------------------
The pinned snapshot is what every figure in the Version Log was measured on. Add
a column to sales_clean.parquet and the file's checksum changes, the manifest
stops matching, and any argument about whether v11 and v17 are comparable becomes
an argument about whether the data moved. Keeping the target file byte-identical
and joining the feature from beside it means the only difference between the two
arms is the feature, which is the whole point.

Week binning
------------
Identical to src/clean.py: W-MON, closed="right", label="right", so a week runs
Tuesday through Monday and is labelled by the Monday it ends on. Restricted to
the exact SKU set and week labels in the pinned snapshot, so the join is total
and cannot silently drop rows.

Channel groups
--------------
Grouped on `channel` itself, as specified by the business:

    amazon_fba   Amazon FBA
    amazon_fbm   Amazon FBM
    walmart      Walmart
    coverland    Coverland B2C + Coverland B2B + ICarCover
    parts        advance_parts + auto_armor

An earlier version of this script used `src.v1._assign_stream`, which is V1's
internal fulfilment routing (east/west, sales/preorder) and not the sales
channel at all. It produced a plausible-looking table answering the wrong
question. Recorded so nobody reuses that function for this purpose again.

Matching normalises case, spaces and underscores, so "Amazon FBA",
"amazon_fba" and "AMAZON  FBA" all land in the same group. Anything that
matches nothing is reported by name with its unit count and put in `other`
rather than dropped, because a channel nobody mapped is a fact about the data,
not a rounding error.

What is written
---------------
One row per SKU per week, with `units_<stream>` for each stream. Shares and
trailing windows are NOT computed here. This file is the raw material; the
feature derives the trailing share at matrix-build time, from training data only,
so the as-of rule is enforced in one place rather than baked into an export.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from config import ML_DATA_SNAPSHOT  # noqa: E402
from src.ml.dataset import data_dir  # noqa: E402
from sqlalchemy import text  # noqa: E402

from src.v1 import _engine  # noqa: E402

import re  # noqa: E402

GROUPS = ["amazon_fba", "amazon_fbm", "walmart", "coverland", "parts", "other"]


def _norm(c: str) -> str:
    """Lowercase, and collapse anything that is not a letter or digit."""
    return re.sub(r"[^a-z0-9]+", " ", str(c).lower()).strip()


def assign_group(channel: str) -> str:
    """Map a raw channel string to one of GROUPS.

    Order matters: the two Amazon variants are tested before anything else so a
    generic "amazon" rule can never swallow them.
    """
    n = _norm(channel)
    if "fba" in n:
        return "amazon_fba"
    if "fbm" in n:
        return "amazon_fbm"
    if "walmart" in n:
        return "walmart"
    if "coverland" in n or "icarcover" in n or "i car cover" in n:
        return "coverland"
    if "advance" in n or "auto armor" in n or "autoarmor" in n:
        return "parts"
    return "other"


def main() -> int:
    snap = data_dir(ML_DATA_SNAPSHOT)
    pinned = pd.read_parquet(snap / "sales_clean.parquet")
    pinned["ds"] = pd.to_datetime(pinned["ds"])
    skus = set(pinned["unique_id"].unique())
    weeks = pd.DatetimeIndex(sorted(pinned["ds"].unique()))
    print(f"pinned snapshot {ML_DATA_SNAPSHOT}: {len(skus):,} SKUs, {len(weeks)} weeks")

    # Aggregate in Postgres, not in pandas. The first version of this script
    # reused src.v1.load_raw_for_v1, which SELECTs every order line with two
    # extra text columns and pulls the lot over the network before grouping. It
    # was slow enough to look hung. Grouping server-side collapses many order
    # lines into one row per SKU per week per (channel, order_type) and moves
    # the work to the machine holding the data.
    #
    # The week expression is the same one already used by api/main.py and
    # src/db.py, and it matches pandas W-MON closed="right" exactly: Monday maps
    # to itself, Tuesday through Sunday map forward to the next Monday, so a
    # bucket runs Tuesday through Monday and is labelled by the Monday it ends
    # on. Verified against the pandas grouper over 122 consecutive days.
    print("querying, grouped server-side (expect tens of seconds, not minutes) ...")
    sql = text("""
        SELECT
            link_master_sku AS unique_id,
            (order_date + ((8 - EXTRACT(ISODOW FROM order_date))::int % 7)
                * INTERVAL '1 day')::date AS ds,
            channel,
            SUM(link_qty) AS units
        FROM shipcore.fc_velocity_link_snapshot_forecast
        GROUP BY 1, 2, 3
    """)
    with _engine().connect() as conn:
        raw = pd.read_sql(sql, conn, parse_dates=["ds"])
    print(f"  {len(raw):,} grouped rows")
    raw["link_qty"] = raw["units"]

    # Every distinct channel and where it landed, printed before anything is
    # aggregated. This is the check that the mapping is right, and it is cheap.
    print("\n  raw channel values, units, and the group each maps to:")
    inv = (raw.groupby("channel", dropna=False)["link_qty"].sum()
           .sort_values(ascending=False))
    for ch, u in inv.items():
        g = assign_group(ch)
        flag = "   <-- UNMAPPED" if g == "other" else ""
        print(f"    {str(ch)[:34]:<34} {u:>9,.0f}  ->  {g}{flag}")

    raw["stream"] = [assign_group(c) for c in raw["channel"]]
    other_units = raw.loc[raw["stream"] == "other", "link_qty"].sum()
    if other_units:
        print(f"\n  {other_units:,.0f} units ({other_units / raw['link_qty'].sum() * 100:.1f}%) "
              f"are in 'other'. If that is not near zero, fix assign_group before "
              f"building a feature on these shares.")

    # ds already carries the W-MON label from SQL, so this only sums the
    # (channel, order_type) combinations that map to the same stream.
    weekly = (
        raw.groupby(["unique_id", "stream", "ds"], as_index=False)["link_qty"]
        .sum().rename(columns={"link_qty": "units"})
    )
    weekly = weekly[weekly["unique_id"].isin(skus) & weekly["ds"].isin(weeks)]

    wide = (
        weekly.pivot_table(index=["unique_id", "ds"], columns="stream",
                           values="units", fill_value=0)
        .reindex(columns=GROUPS, fill_value=0)
        .reset_index()
    )
    wide.columns = ["unique_id", "ds"] + [f"units_{g}" for g in GROUPS]

    # Total grid, so a SKU-week with no orders is an explicit row of zeros rather
    # than a missing key the join would have to guess at.
    grid = pd.MultiIndex.from_product([sorted(skus), weeks], names=["unique_id", "ds"])
    wide = (
        wide.set_index(["unique_id", "ds"]).reindex(grid, fill_value=0).reset_index()
    )

    out = snap / "channel_mix.parquet"
    wide.to_parquet(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)}  ({len(wide):,} rows)")

    print("\nunits by channel group over the whole snapshot, and share of total:")
    tot = wide[[f"units_{g}" for g in GROUPS]].sum()
    for g in GROUPS:
        u = tot[f"units_{g}"]
        print(f"  {g:<16} {u:>10,.0f}  {u / tot.sum() * 100:5.1f}%")
    print("\nA stream with a very small share cannot support a per-SKU feature:")
    print("its share is mostly zero and occasionally one order, which is noise")
    print("rather than signal. Check these before choosing the feature set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
