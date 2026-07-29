#!/usr/bin/env python3
"""Export a real inventory snapshot for the dashboard from production tables.

Replaces dashboard/lib/data.py's generated sample with dashboard/data/inventory_snapshot.csv,
sourced from the same systems the Commerce Integration app uses (see docs/ML_FORECAST_DESIGN.md
Section 2.1 for what the forecast's training target contains, which is the background that
motivated this, and the stage-1 investigation this script follows for how the routing below
was established).

Sources (two separate databases):
- Commerce/Supabase, ecommerce_data.coverland_inventory_by_warehouse: the origin table for
  on-hand and backorder. Confirmed by stage 1: `available` = on_hand - allocated in every row,
  never net of backorder, so summing available and separately adding backorder does not
  double-count. Rawer and fresher than shipcore.sc_inventory_snapshot (last refreshed
  2026-05-01, effectively stale) or shipcore.fc_stats (a pivoted cache with `back` negated).
- Primary (shipcore): fc_container_items joined to fc_containers for confirmed inbound, and
  sc_products for product_name. fc_products.product_name looked like the better candidate at
  first read (447/447 forecast SKUs matched, vs 439/447 for sc_products) -- but every single row
  in fc_products has product_name == master_sku (10,458/10,458 in the whole table checked, not
  just this SKU set): it is an unpopulated placeholder, not real data. sc_products has genuine
  human-readable names for 371 of the 447 forecast SKUs; the rest (8 unmatched, 68 where it also
  just echoes the SKU) are treated as no-name and left blank, same as the true misses.

Inbound filter, matching src/app/api/planning/dashboard/route.ts in the Commerce repo exactly:
  WHERE c.status IN ('shipped', 'packing_received')   -- draft containers excluded
Plus one addition on top of production's filter: c.eta_date >= CURRENT_DATE. A container still
marked shipped/packing_received after its ETA has passed is stale bookkeeping (status not yet
flipped to arrived), not a real future arrival, and would otherwise inflate confirmed_inbound
with units that likely already landed.

Missing-data handling: a SKU absent from the inventory source is written with blank
available_inventory/preorder_backlog/product_name cells (pandas will read these back as NaN),
not 0 -- so "no stock record" stays distinguishable from "record says zero". confirmed_inbound
is different: absence there is queried directly (no container line items match), which is a real
zero, not a gap, so it is filled with 0. Category is deliberately not included here: it is
already derived correctly from the SKU prefix in dashboard/lib/data.py::product_category().

Run:
    .venv/bin/python scripts/export_inventory_snapshot.py

Safe to re-run: overwrites dashboard/data/inventory_snapshot.csv in place.
"""
import os
import sys
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import bindparam, create_engine, text

ROOT = Path(__file__).parent.parent
FORECAST_PARQUET = ROOT / "data" / "processed" / "ml_forward_forecasts.parquet"
OUT_CSV = ROOT / "dashboard" / "data" / "inventory_snapshot.csv"

# override=True: the repo's .env is the source of truth. Without it, stale DB_*
# variables exported in the user's shell silently take precedence (see CLAUDE.md).
load_dotenv(ROOT / ".env", override=True)


def _engine(prefix: str, label: str):
    host, port, name, user, pw = (
        os.getenv(f"{prefix}_HOST"),
        os.getenv(f"{prefix}_PORT"),
        os.getenv(f"{prefix}_NAME"),
        os.getenv(f"{prefix}_USER"),
        os.getenv(f"{prefix}_PASSWORD"),
    )
    if not all([host, port, name, user, pw]):
        print(f"Missing {prefix}_* variables in .env for {label} connection.")
        sys.exit(1)
    url = f"postgresql+psycopg2://{quote_plus(user)}:{quote_plus(pw)}@{host}:{port}/{name}"
    return create_engine(url, connect_args={"connect_timeout": 10, "sslmode": "require"})


PRIMARY_ENGINE = _engine("DB", "primary (shipcore)")
COMMERCE_ENGINE = _engine("COMMERCE_DB", "Commerce/Supabase (ecommerce_data)")

INVENTORY_QUERY = text("""
    SELECT master_sku,
           SUM(available) AS available_inventory,
           SUM(on_hand)   AS on_hand_physical,
           SUM(allocated) AS allocated,
           SUM(backorder) AS preorder_backlog
    FROM ecommerce_data.coverland_inventory_by_warehouse
    WHERE master_sku IN :skus
    GROUP BY master_sku
""").bindparams(bindparam("skus", expanding=True))

# Does any stock sit outside the four warehouses the Commerce app pivots into
# west/east? If it does, our SUM over all warehouses is wider than theirs, and the
# two screens will disagree for a reason that has nothing to do with transit.
WAREHOUSE_QUERY = text("""
    SELECT COALESCE(NULLIF(warehouse, ''), '(unspecified)') AS warehouse,
           SUM(available) AS available
    FROM ecommerce_data.coverland_inventory_by_warehouse
    WHERE master_sku IN :skus
    GROUP BY 1 ORDER BY 2 DESC
""").bindparams(bindparam("skus", expanding=True))

# transit_stock is a manually-keyed integer on fc_stats, set through
# PATCH /api/planning/sku/[sku]/transit-stock and preserved across inventory
# syncs. The Commerce planning grid adds it to west+east available to form its
# total_stock, so their screen shows it and ours did not.
#
# It is exported as its own column rather than folded into available_inventory,
# deliberately. Two unknowns make folding it in unsafe: nobody here can say what
# it counts, and if it overlaps fc_container_items (both plausibly meaning "units
# on the water") then adding it to a formula that already credits confirmed
# inbound would count the same units twice. Adding it to available_inventory
# would also feed it into the week-by-week depletion as though it were on the
# shelf today, making every stockout date optimistic. Carried, reported, unused
# until its size and meaning are known.
TRANSIT_QUERY = text("""
    SELECT master_sku, COALESCE(transit_stock, 0) AS transit_stock
    FROM shipcore.fc_stats
    WHERE COALESCE(transit_stock, 0) <> 0
      AND master_sku IN :skus
""").bindparams(bindparam("skus", expanding=True))

# Confirmed inbound: production's status filter, plus an ETA floor of today.
# A container whose ETA already passed but is still "shipped"/"packing_received"
# is a status that wasn't updated, not a unit genuinely still in transit.
INBOUND_QUERY = text("""
    SELECT ci.master_sku,
           SUM(ci.qty)     AS confirmed_inbound,
           MIN(c.eta_date) AS inbound_eta
    FROM shipcore.fc_container_items ci
    JOIN shipcore.fc_containers c ON c.id = ci.container_id
    WHERE c.status IN ('shipped', 'packing_received')
      AND c.eta_date >= CURRENT_DATE
      AND ci.master_sku IN :skus
    GROUP BY ci.master_sku
""").bindparams(bindparam("skus", expanding=True))


# fc_products.product_name is an unpopulated placeholder that always equals
# master_sku (confirmed 10,458/10,458 rows in the whole table) -- sc_products has
# real names, so it is used instead despite matching a few fewer forecast SKUs.
PRODUCT_NAME_QUERY = text("""
    SELECT master_sku, product_name
    FROM shipcore.sc_products
    WHERE product_name IS NOT NULL
      AND product_name <> master_sku
      AND master_sku IN :skus
""").bindparams(bindparam("skus", expanding=True))


def main() -> None:
    if not FORECAST_PARQUET.exists():
        print(f"Forecast file not found: {FORECAST_PARQUET}")
        sys.exit(1)

    forecast_skus = sorted(pd.read_parquet(FORECAST_PARQUET)["unique_id"].unique().tolist())
    print(f"Forecast SKUs: {len(forecast_skus)}")

    with COMMERCE_ENGINE.connect() as conn:
        inv = pd.read_sql(INVENTORY_QUERY, conn, params={"skus": forecast_skus})
        warehouses = pd.read_sql(WAREHOUSE_QUERY, conn, params={"skus": forecast_skus})
        snapshot_ts = conn.execute(text(
            "SELECT MAX(created_at) FROM ecommerce_data.coverland_inventory_by_warehouse"
        )).scalar()

    with PRIMARY_ENGINE.connect() as conn:
        inbound = pd.read_sql(INBOUND_QUERY, conn, params={"skus": forecast_skus})
        names = pd.read_sql(PRODUCT_NAME_QUERY, conn, params={"skus": forecast_skus})
        transit = pd.read_sql(TRANSIT_QUERY, conn, params={"skus": forecast_skus})
        # When no SKU has nonzero transit stock, this comes back with 0 rows, and
        # pd.read_sql has nothing to infer a numeric dtype from -- it defaults to
        # object. Coerce explicitly so the later fillna/astype below always acts
        # on a numeric column, not an object one (which pandas otherwise "silently
        # downcasts" and warns about).
        transit["transit_stock"] = pd.to_numeric(transit["transit_stock"], errors="coerce")

    # Defensive sign correction per the output contract. Stage 1 found backorder
    # is never negative in this source, so this is a no-op today, not a fix for
    # an observed problem.
    inv["preorder_backlog"] = inv["preorder_backlog"].abs()

    out = pd.DataFrame({"unique_id": forecast_skus})
    out = out.merge(names, left_on="unique_id", right_on="master_sku", how="left").drop(columns=["master_sku"])
    out = out.merge(inv, left_on="unique_id", right_on="master_sku", how="left").drop(columns=["master_sku"])
    out = out.merge(inbound, left_on="unique_id", right_on="master_sku", how="left").drop(columns=["master_sku"])
    out = out.merge(transit, left_on="unique_id", right_on="master_sku", how="left").drop(columns=["master_sku"])
    # Absence from the transit query means the value is zero or unset, both of
    # which mean "no transit stock recorded". Unlike the inventory columns, this
    # is not a missing record.
    out["transit_stock"] = out["transit_stock"].fillna(0).astype(int)

    # confirmed_inbound absence means the query found no matching container line
    # items -- a real zero, unlike the inventory columns above where absence means
    # "no record at all" and must stay blank.
    out["confirmed_inbound"] = out["confirmed_inbound"].fillna(0).astype(int)

    out["inbound_eta"] = pd.to_datetime(out["inbound_eta"]).dt.strftime("%Y-%m-%d")
    out["inbound_eta"] = out["inbound_eta"].where(out["inbound_eta"].notna(), "")

    diag = out[["unique_id", "on_hand_physical", "allocated"]].copy()
    out = out[["unique_id", "product_name", "available_inventory", "preorder_backlog",
               "confirmed_inbound", "inbound_eta", "transit_stock"]]

    missing_skus = out.loc[out["available_inventory"].isna(), "unique_id"].tolist()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    print(f"\nWrote {len(out)} rows -> {OUT_CSV}")
    print(f"SKUs matched against inventory source: {len(out) - len(missing_skus)}")
    print(f"SKUs missing from inventory source: {len(missing_skus)}")
    if missing_skus:
        print(f"Missing: {missing_skus}")
    print(f"Total available_inventory: {out['available_inventory'].sum(skipna=True):,.0f}")
    print(f"Total preorder_backlog:    {out['preorder_backlog'].sum(skipna=True):,.0f}")
    print(f"Total confirmed_inbound:   {out['confirmed_inbound'].sum():,.0f}")
    print(f"Source snapshot timestamp (ecommerce_data, MAX(created_at)): {snapshot_ts}")

    # ---- Why the dashboard's stock figure looks low, answered in numbers ----
    on_hand = diag["on_hand_physical"].sum(skipna=True)
    alloc = diag["allocated"].sum(skipna=True)
    avail = out["available_inventory"].sum(skipna=True)
    print("\n--- stock definition ---")
    print(f"physical on hand           {on_hand:>10,.0f}")
    print(f"less allocated to orders   {alloc:>10,.0f}")
    print(f"= available to sell        {avail:>10,.0f}   <- what the dashboard uses")
    hidden = int(((diag["on_hand_physical"] > 0)
                  & (out["available_inventory"] == 0)).sum())
    print(f"SKUs showing 0 available while physically holding stock: {hidden}")

    print("\n--- transit stock (Commerce adds this to its total_stock; we do not) ---")
    t = out["transit_stock"]
    print(f"SKUs with a transit value  {int((t != 0).sum()):>10}")
    print(f"total transit units        {int(t.sum()):>10,}")
    both = int(((t > 0) & (out["confirmed_inbound"] > 0)).sum())
    print(f"SKUs with BOTH transit and container inbound: {both}")
    if both:
        overlap = out.loc[(t > 0) & (out["confirmed_inbound"] > 0)]
        exact = int((overlap["transit_stock"] == overlap["confirmed_inbound"]).sum())
        print(f"  of those, transit == confirmed_inbound exactly: {exact}")
        print("  (a high count here means the two columns are the same units,")
        print("   and adding both to the order formula would double-count)")

    print("\n--- available by warehouse (Commerce pivots only the four named ones) ---")
    known = {"Fullerton", "Canary", "TTM Group", "TTM Group Jefferson"}
    for _, w in warehouses.iterrows():
        mark = "" if w["warehouse"] in known else "   <- outside Commerce's west/east pivot"
        print(f"  {w['warehouse']:<26} {w['available']:>9,.0f}{mark}")


if __name__ == "__main__":
    main()
