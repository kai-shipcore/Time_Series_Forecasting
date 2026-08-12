#!/usr/bin/env python3
"""Check that writing a run replaces the whole run, not just the rows it shares.

Why this test exists
--------------------
On 2026-08-10 the smooth set went from 467 SKUs to 338. The forward table kept
all 467: `store.upsert` keyed on (model_version, week_of, unique_id, ds), so the
new run overwrote the 338 SKUs it produced and had no way to express that the
other 129 were gone. The week then described two segmentations at once, with
nothing on screen saying so, and the planning pages served stale forecasts for
SKUs the model had stopped forecasting.

That is a data-correctness bug with no visible symptom, which is the kind worth
a test rather than a comment. Any change to the segmentation rules reproduces
it, and the rules are still being tuned.

Runs against SQLite, not Postgres
---------------------------------
The point of the test is the sequence of statements inside one transaction,
which is engine-independent. SQLite has supported ON CONFLICT since 3.24 and
executemany DELETE always, so the same code path executes.

Two Postgres-only things are stubbed rather than tested here, and both are
exercised by simply running the pipeline:

  - CREATE SCHEMA, which SQLite has no concept of. An in-memory database is
    ATTACHed under the name `shipcore` so the qualified table names resolve.
  - `_migrate_forecast_date_to_week_of`, which reads information_schema.

Usage::

    .venv/bin/python scripts/test_store_replace_run.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402

from src.ml.serving import store  # noqa: E402

FAILURES: list[str] = []


def check(label: str, got, want) -> None:
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: got {got!r}, want {want!r}")
    if not ok:
        FAILURES.append(label)


def run(skus: list[str], week: str, yhat: float, version: str = "v11") -> pd.DataFrame:
    """One run's worth of rows: every SKU forecast for two target weeks."""
    rows = [
        {
            "model_version": version,
            "week_of": pd.Timestamp(week),
            "unique_id": sku,
            "ds": pd.Timestamp(week) + pd.Timedelta(weeks=h),
            "yhat": yhat,
            "bucket": "smooth",
            "history_length": "long",
            "segment": "smooth/long",
            "served_by": "lgbm",
            "run_at": "2026-08-12T00:00:00",
        }
        for sku in skus
        for h in (1, 2)
    ]
    return pd.DataFrame(rows)


def fresh_engine():
    """A SQLite engine with `shipcore` attached, standing in for Postgres.

    A single shared connection, because ATTACH applies per connection and a pool
    that opened a second one would not see the schema.
    """
    eng = create_engine("sqlite://", poolclass=None)
    with eng.connect() as conn:
        conn.exec_driver_sql("ATTACH DATABASE ':memory:' AS shipcore")
        conn.commit()
    return eng


def install(monkey_engine) -> None:
    """Point store at the test engine and neutralise the Postgres-only DDL."""
    store.engine = lambda: monkey_engine

    def ensure_table(ddl: str, table: str) -> bool:
        body = ddl.replace("CREATE SCHEMA IF NOT EXISTS shipcore;", "")
        # CREATE INDEX is spelled the other way round in SQLite: the schema goes
        # on the index name and the table must be bare, where Postgres wants the
        # table qualified and infers the index's schema from it. So the
        # qualifier moves across. Rewritten rather than skipped so the DDL still
        # runs as a whole.
        body = re.sub(
            r"(CREATE INDEX IF NOT EXISTS\s+)(\w+)(\s+ON\s+)shipcore\.",
            r"\1shipcore.\2\3",
            body,
        )
        with monkey_engine.begin() as conn:
            for stmt in [s.strip() for s in body.split(";") if s.strip()]:
                conn.exec_driver_sql(stmt)
        return True

    store.ensure_table = ensure_table


def table_state(eng, table: str) -> pd.DataFrame:
    with eng.connect() as conn:
        return pd.read_sql(text(f"SELECT * FROM {table}"), conn)


def main() -> int:
    eng = fresh_engine()
    install(eng)
    T = store.FORWARD_TABLE

    print("\n1. the 2026-08-10 shrink: 5 SKUs, then a re-run producing only 3")
    store.upsert(run([f"S{i}" for i in range(1, 6)], "2026-08-10", 10.0),
                 table=T, ddl=store.FORWARD_CREATE_SQL)
    check("SKUs after the first run", table_state(eng, T)["unique_id"].nunique(), 5)

    store.upsert(run(["S1", "S2", "S3"], "2026-08-10", 99.0),
                 table=T, ddl=store.FORWARD_CREATE_SQL)
    after = table_state(eng, T)
    check("SKUs after the re-run", sorted(after["unique_id"].unique()), ["S1", "S2", "S3"])
    check("dropped SKUs remaining", int(after["unique_id"].isin(["S4", "S5"]).sum()), 0)
    check("surviving rows carry the new values", sorted(after["yhat"].unique()), [99.0])

    print("\n2. other weeks are untouched")
    store.upsert(run(["S1", "S2"], "2026-08-03", 5.0), table=T, ddl=store.FORWARD_CREATE_SQL)
    store.upsert(run(["S1"], "2026-08-10", 77.0), table=T, ddl=store.FORWARD_CREATE_SQL)
    after = table_state(eng, T)
    check("2026-08-03 survives the 2026-08-10 write",
          int((after["week_of"] == "2026-08-03").sum()), 4)
    check("2026-08-10 now holds one SKU",
          sorted(after[after["week_of"] == "2026-08-10"]["unique_id"].unique()), ["S1"])

    print("\n3. a second model_version on the same week is untouched")
    store.upsert(run(["A1", "A2"], "2026-08-10", 1.0, version="v18"),
                 table=T, ddl=store.FORWARD_CREATE_SQL)
    store.upsert(run(["S1"], "2026-08-10", 88.0, version="v11"),
                 table=T, ddl=store.FORWARD_CREATE_SQL)
    after = table_state(eng, T)
    check("v18 rows survive a v11 write",
          int((after["model_version"] == "v18").sum()), 4)

    print("\n4. a failed insert rolls the delete back")
    before = len(table_state(eng, T))
    bad = run(["S1"], "2026-08-10", 1.0)
    # NOT NULL on unique_id: the DELETE succeeds, the INSERT then fails, and the
    # week must survive. This is the failure that would otherwise empty a week.
    bad.loc[0, "unique_id"] = None
    rc = store.upsert(bad, table=T, ddl=store.FORWARD_CREATE_SQL)
    check("failed write reports -1", rc, -1)
    check("row count unchanged after the failure", len(table_state(eng, T)), before)

    print("\n5. replace_run=False keeps the old upsert-only behaviour")
    store.upsert(run(["S9"], "2026-08-10", 3.0), table=T, ddl=store.FORWARD_CREATE_SQL,
                 replace_run=False)
    after = table_state(eng, T)
    check("S9 added without clearing the week",
          int((after[after["week_of"] == "2026-08-10"]["unique_id"] == "S1").sum()), 2)

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {', '.join(FAILURES)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
