#!/usr/bin/env python3
"""Say why the database write failed, instead of guessing.

`store.upsert` returns -1 for every failure and `ml_forward_forecast.py` prints
that as "no DB credentials, or it could not be reached". Those are two causes
out of at least six, and on 2026-08-12 that message was printed for a failure
nobody had diagnosed. This walks the same path and reports where it stops.

Reads only. Prints no secrets: passwords are shown as a length, and the host is
shown as given because it is already in .env in plain text and is the field most
likely to be wrong.

Usage::

    .venv/bin/python scripts/ml_check_db.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ml.serving import store  # noqa: E402

REQUIRED = ("DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_NAME")


def main() -> int:
    print("\n1. the driver")
    try:
        import psycopg2  # noqa: F401
        print("   ok   psycopg2 imports")
    except Exception as exc:
        print(f"   STOP psycopg2 will not import: {exc}")
        print("        .venv/bin/pip install -r requirements.txt")
        return 1

    print("\n2. the environment, after .env is loaded with override=True")
    store._load_env()
    missing = [k for k in REQUIRED if not os.getenv(k)]
    for k in REQUIRED:
        v = os.getenv(k)
        if not v:
            shown = "MISSING"
        elif "PASSWORD" in k:
            # Length only. A truncated password has cost this project a
            # debugging session before, and the length is what showed it.
            shown = f"set, {len(v)} characters"
        else:
            shown = v
        print(f"   {'ok  ' if v else 'STOP'} {k:<12} {shown}")
    if missing:
        print(f"\n   STOP {len(missing)} variable(s) missing, so engine() returns None")
        print("        This is the only case the script's message describes correctly.")
        return 1

    print("\n3. building the engine")
    eng = store.engine()
    if eng is None:
        print("   STOP engine() returned None even though the variables are set,")
        print("        which means create_engine itself raised. Check DB_PORT is numeric.")
        return 1
    print("   ok   engine built")

    print("\n4. connecting")
    try:
        from sqlalchemy import text
        with eng.connect() as conn:
            who = conn.execute(text("SELECT current_user, current_database()")).first()
        print(f"   ok   connected as {who[0]} to {who[1]}")
    except Exception as exc:
        print(f"   STOP {type(exc).__name__}: {exc}")
        print("\n        Read the message above rather than assuming credentials:")
        print("        authentication failed  -> wrong user or password")
        print("        no pg_hba.conf entry   -> this machine's IP is not allowed in")
        print("        timeout / no route     -> host, port, or a firewall")
        return 1

    print("\n5. the two tables")
    for table in (store.TABLE, store.FORWARD_TABLE):
        try:
            from sqlalchemy import text
            with eng.connect() as conn:
                n = conn.execute(text(f"SELECT count(*) FROM {table}")).scalar()
                cols = conn.execute(text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = :s AND table_name = :t"
                ), {"s": table.split(".")[0], "t": table.split(".")[1]}).scalars().all()
            legacy = "forecast_date" in cols
            print(f"   ok   {table}: {n:,} rows"
                  + ("   <-- still has forecast_date, not week_of" if legacy else ""))
            with eng.connect() as conn:
                runs = conn.execute(text(
                    f"SELECT model_version, week_of, count(DISTINCT unique_id) AS skus, "
                    f"count(*) AS rows FROM {table} "
                    f"GROUP BY 1, 2 ORDER BY 2 DESC, 1 LIMIT 6"
                )).all()
            for r in runs:
                print(f"          {r[0]:<12} {r[1]}  {r[2]:>5} SKUs  {r[3]:>7,} rows")
        except Exception as exc:
            print(f"   STOP {table}: {type(exc).__name__}: {exc}")
            return 1

    print("\n6. write permission, tested and rolled back")
    try:
        from sqlalchemy import text
        with eng.connect() as conn:
            trans = conn.begin()
            conn.execute(text(
                f"DELETE FROM {store.FORWARD_TABLE} WHERE model_version = '__probe__'"))
            trans.rollback()
        print("   ok   DELETE is permitted (rolled back, nothing changed)")
    except Exception as exc:
        print(f"   STOP cannot DELETE: {type(exc).__name__}: {exc}")
        print("        The write path now deletes a run before inserting it, so a")
        print("        read-only or INSERT-only grant fails where it used to work.")
        return 1

    print("\nEverything the write path needs is working.")
    print("If ml_forward_forecast.py still reports a failed write, run it again and")
    print("read store.LAST_ERROR, which now holds the actual exception.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
