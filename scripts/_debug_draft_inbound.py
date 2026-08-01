"""Why is the draft figure not showing on the Action List.

Answers, in order, the three things that can be true:

  1. The planning table is being built from an export that predates the column,
     so every row reads zero and no sub-line is drawn.
  2. The database is being read, but no container matches the filter, either
     because nothing is in draft or because the ETA rule excludes them.
  3. Both are fine and the running FastAPI process is older than the code.

Run:
    .venv/bin/python scripts/_debug_draft_inbound.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.planning import calc, data as D, inventory as INV  # noqa: E402


def rule(title: str) -> None:
    print(f"\n{'-' * 70}\n{title}\n{'-' * 70}")


def main() -> int:
    rule("1. Which inventory source is in use")
    source = D.inventory_source()
    print(f"   source: {source}")
    if source != "database":
        print("   -> Not reading the database, so the draft query never runs.")
        print("      An export written before this change has no draft column and")
        print("      every row reads zero. Fix: .venv/bin/python scripts/export_inventory_snapshot.py")

    rule("2. What the planning table actually carries")
    plan = calc.build_planning_table(calc.DEFAULT_PARAMS)
    if "draft_inbound" not in plan.columns:
        print("   draft_inbound column: MISSING -> calc.py did not add it. Wrong checkout?")
        return 1
    d = plan["draft_inbound"]
    print(f"   rows: {len(plan)}")
    print(f"   rows with draft_inbound > 0: {int((d > 0).sum())}   <- rows that would show a sub-line")
    print(f"   total drafted units: {int(d.sum()):,}")

    rule("3. What the containers table holds, ignoring every filter")
    INV._load_env()
    primary = INV._engine("DB")
    if primary is None:
        print("   No primary DB connection (DB_* vars or psycopg2 missing).")
        print("   Steps 1 and 2 above are the whole answer in that case.")
        return 0

    from sqlalchemy import text

    with primary.connect() as conn:
        print("\n   Containers by status:")
        by_status = pd.read_sql(text("""
            SELECT status, COUNT(*) AS containers,
                   SUM(CASE WHEN eta_date IS NULL THEN 1 ELSE 0 END) AS no_eta,
                   SUM(CASE WHEN eta_date < CURRENT_DATE THEN 1 ELSE 0 END) AS eta_past,
                   SUM(CASE WHEN eta_date >= CURRENT_DATE THEN 1 ELSE 0 END) AS eta_future
            FROM shipcore.fc_containers GROUP BY status ORDER BY 2 DESC
        """), conn)
        print(by_status.to_string(index=False))
        print("\n   DRAFT_STATUSES in code:", INV.DRAFT_STATUSES)
        print("   -> If the status column uses some other spelling for an uncommitted")
        print("      container, that is the bug and DRAFT_STATUSES needs widening.")

        print("\n   Draft line items, with and without the ETA rule:")
        counts = pd.read_sql(text(f"""
            SELECT
              COUNT(DISTINCT ci.master_sku)                                     AS skus_any_eta,
              SUM(ci.qty)                                                       AS units_any_eta,
              COUNT(DISTINCT ci.master_sku) FILTER (
                  WHERE c.eta_date IS NULL OR c.eta_date >= CURRENT_DATE)       AS skus_after_rule,
              SUM(ci.qty) FILTER (
                  WHERE c.eta_date IS NULL OR c.eta_date >= CURRENT_DATE)       AS units_after_rule
            FROM shipcore.fc_container_items ci
            JOIN shipcore.fc_containers c ON c.id = ci.container_id
            WHERE c.status IN {INV._status_list(INV.DRAFT_STATUSES)}
        """), conn)
        print(counts.to_string(index=False))
        print("   -> A large drop from any_eta to after_rule means the drafts exist")
        print("      but carry ETAs already in the past, and the rule is too strict.")

        print("\n   How many of those SKUs are on the action list at all:")
        skus = set(plan["unique_id"])
        drafted = pd.read_sql(text(f"""
            SELECT DISTINCT ci.master_sku
            FROM shipcore.fc_container_items ci
            JOIN shipcore.fc_containers c ON c.id = ci.container_id
            WHERE c.status IN {INV._status_list(INV.DRAFT_STATUSES)}
              AND (c.eta_date IS NULL OR c.eta_date >= CURRENT_DATE)
        """), conn)
        overlap = skus & set(drafted["master_sku"])
        print(f"   drafted SKUs: {len(drafted)} | forecastable SKUs: {len(skus)} | overlap: {len(overlap)}")
        print("   -> Overlap is what can possibly show. The action list covers only")
        print("      forecastable SKUs, so drafts on intermittent ones never appear here.")

    rule("If all three look right")
    print("   The running FastAPI process is older than the code. It is started")
    print("   without --reload, so Python edits need a restart:")
    print("     pkill -f 'uvicorn api.main' ; then reload the page to let it restart")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
