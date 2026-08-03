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
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.planning import inventory as INV  # noqa: E402

FORECAST_PARQUET = ROOT / "data" / "processed" / "ml_forward_forecasts.parquet"
OUT_CSV = ROOT / "dashboard" / "data" / "inventory_snapshot.csv"


def main() -> None:
    # Through the same loader the service uses, rather than reading the parquet
    # directly. That loader prefers shipcore.ml_forward_forecasts and falls back
    # to the file, so this script now works on a machine that has credentials
    # but no local forecast, and cannot disagree with the dashboard about which
    # SKUs are forecast.
    from src.planning import data as D

    forecast_skus = sorted(D.load_forecasts()["unique_id"].unique().tolist())
    if not forecast_skus:
        print("No forward forecast found, in the database or at "
              f"{FORECAST_PARQUET}. Run scripts/ml_forward_forecast.py, or "
              "scripts/seed_dev_data.py for the committed fixture.")
        sys.exit(1)
    print(f"Forecast SKUs: {len(forecast_skus)}")

    # Same query path the dashboard and the API use at request time. Keeping one
    # definition is the point: an export that disagreed with the live read would
    # be worse than having no export, because the difference would be invisible.
    full = INV.fetch(forecast_skus, diagnostics=True)
    if full is None:
        print("Could not reach the inventory databases. Check the DB_* and "
              "COMMERCE_DB_* variables in .env, and that psycopg2 is installed.")
        sys.exit(1)
    snapshot_ts = full.attrs.get("snapshot_at")
    warehouses = INV.available_by_warehouse(forecast_skus)

    diag = full[["unique_id", "on_hand_physical", "allocated"]].copy()
    out = full[["unique_id", "product_name", "available_inventory", "preorder_backlog",
                "confirmed_inbound", "inbound_eta", "transit_stock"]].copy()

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
