"""Database backing for the ML serving artifacts, with the parquet as fallback.

Why this exists
---------------
`ml_forecast_history.parquet` accumulates one entry per weekly run and is the
only record of what the model predicted before the outcome was known. Every
other file the cron writes rebuilds from the database; this one cannot, because
re-running past versions against past cutoffs produces backtest figures, which
is a different and weaker claim.

It lived on one disk. The weekly run happens on the server, writing in place,
and `data/processed/` is gitignored and excluded from the deploy, so the store
existed in exactly one location with no backup and no way for anyone else to
read it. Two problems in one: it could be lost, and it could not be seen.

A table solves both. It is backed up with the rest of the database, and any
machine with credentials reads exactly what the server serves. The legacy track
already stores its equivalent this way, in `shipcore.fc_forecast_history`; the
ML track writing a parquet instead was the asymmetry.

Behaviour
---------
The parquet is not abandoned. Writes go to both while the table is young, so a
machine with no credentials still works and so the file remains a usable
backup. Reads prefer the table and fall back to the file, which is what makes a
laptop with a `.env` see the server's runs and a laptop without one still see
its own.

Nothing here raises on a missing or unreachable database. That is an ordinary
condition, not an error: a fresh clone has no credentials, and the planning
pages are expected to work from the seeded fixture alone.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]

#: A row in either table is identified by which model made the prediction, when
#: it made it, for which SKU, and for which target week. Re-running in the same
#: training week therefore replaces its own rows and nothing else, which is the
#: rule the parquet paths already follow.
#: `week_of` is the W-MON label of the TRAINING CUTOFF WEEK, not the calendar
#: date the run happened. It was called `forecast_date` until 2026-08-12, which
#: was a genuine trap: the legacy track's `shipcore.fc_*` tables use
#: `forecast_date` for the calendar date and `week_of` for the training week, so
#: the same name meant different things on the two sides of this project.
#:
#: That cost real time. Reading the cron output, `forecast_date=2026-08-12` from
#: the legacy run and `trained through 2026-08-10` from the ML run look like a
#: contradiction, and both the developer and the assistant concluded the ML
#: table was keyed on run date and therefore accumulating duplicate runs per
#: week. It never was. Renaming makes the two tracks agree, and `run_at` already
#: carries the wall-clock timestamp for anyone who wants it.
#: Why the last write failed, or None. Set by `upsert`, read by callers that
#: want to print a cause alongside the -1. Not an exception because a failed
#: write here is deliberately non-fatal; see `upsert`.
LAST_ERROR: str | None = None

KEY = ["model_version", "week_of", "unique_id", "ds"]

TABLE = "shipcore.ml_forecast_history"
FORWARD_TABLE = "shipcore.ml_forward_forecasts"

#: Both tables hold the same shape, because they hold the same rows for
#: different spans: the forward table is the current horizon, the history table
#: is every horizon ever served. Written as one definition rather than two so
#: they cannot drift into disagreeing about a column.
def _create_sql(table: str, index_name: str) -> str:
    return f"""
CREATE SCHEMA IF NOT EXISTS shipcore;
CREATE TABLE IF NOT EXISTS {table} (
    model_version   text        NOT NULL,
    week_of         date        NOT NULL,
    unique_id       text        NOT NULL,
    ds              date        NOT NULL,
    yhat            double precision,
    bucket          text,
    history_length  text,
    segment         text,
    served_by       text,
    run_at          text,
    PRIMARY KEY (model_version, week_of, unique_id, ds)
);
CREATE INDEX IF NOT EXISTS {index_name}
    ON {table} (model_version, week_of);
"""


def _migrate_forecast_date_to_week_of(conn, table: str) -> bool:
    """Rename forecast_date to week_of on a table created before 2026-08-12.

    Idempotent: checks information_schema first and does nothing once applied,
    or when the table does not exist yet. Run on every write, matching the
    `_MIGRATE_PI_SQL` pattern in src/db.py, because this project has no
    migration tool and a schema change therefore has to be safe to execute
    repeatedly.

    Postgres carries the primary key and the index across a column rename
    automatically, so one statement moves the constraint too. Values are
    untouched: the column already held the training week, which is the whole
    point of the rename.

    Done in Python rather than as a `DO $$ ... END $$` block on purpose.
    `ensure_table` splits its DDL on ";" and executes the fragments, and a DO
    block contains semicolons, so an anonymous block would have been shredded
    into invalid statements. Returns True when it renamed something.
    """
    schema, name = table.split(".", 1)
    found = conn.exec_driver_sql(
        "SELECT 1 FROM information_schema.columns "
        f"WHERE table_schema = '{schema}' AND table_name = '{name}' "
        "AND column_name = 'forecast_date'"
    ).first()
    if not found:
        return False
    conn.exec_driver_sql(f"ALTER TABLE {table} RENAME COLUMN forecast_date TO week_of")
    return True


CREATE_SQL = _create_sql(TABLE, "ml_forecast_history_run_idx")
FORWARD_CREATE_SQL = _create_sql(FORWARD_TABLE, "ml_forward_forecasts_run_idx")

#: Columns both tables carry, in the order the parquet writes them.
COLUMNS = KEY + ["yhat", "bucket", "history_length", "segment", "served_by", "run_at"]


def write_forward(df: pd.DataFrame) -> int:
    """Store the current horizon. Rows written, or -1 if the database is absent.

    The counterpart to `src.db.write_forward_forecasts` on the legacy track, and
    the reason the Action List can be read from a machine that did not produce
    the forecast.

    Accumulates across weeks rather than replacing wholesale. `_read_forecasts`
    already takes the latest `week_of` and ignores the rest, so older horizons
    cost a little space and are occasionally useful for asking what last week's
    recommendation was.

    Within one training week it replaces the run outright: see `upsert`, which
    clears the (model_version, week_of) pair first. Keying alone was not enough,
    because it left behind SKUs that a re-segmented run no longer produces.
    """
    return upsert(df, table=FORWARD_TABLE, ddl=FORWARD_CREATE_SQL)


def read_forward() -> pd.DataFrame | None:
    """The stored horizons, or None when the table cannot be read."""
    return read(COLUMNS, table=FORWARD_TABLE)


def _load_env() -> None:
    """Read the repo's .env, overriding the shell.

    override=True for the reason recorded in CLAUDE.md: stale DB_* values
    exported in a shell otherwise win over the file, which has already cost one
    debugging session over a truncated password.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env", override=True)
    except Exception:
        pass


def engine():
    """A connection to the primary database, or None.

    None in every failure mode, and the caller's job is to fall back. An
    unusable connection is ordinary here: a clone without credentials, or
    without psycopg2, is a supported way to run this project.

    Deliberately not `src.db.get_engine`, which assumes the variables are set
    and raises inside `quote_plus(None)` when they are not. This is called on
    read paths that must degrade rather than fail.
    """
    _load_env()
    required = ("DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_NAME")
    if not all(os.getenv(k) for k in required):
        return None
    try:
        from urllib.parse import quote_plus

        from sqlalchemy import create_engine

        url = (
            f"postgresql+psycopg2://{quote_plus(os.environ['DB_USER'])}:"
            f"{quote_plus(os.environ['DB_PASSWORD'])}@{os.environ['DB_HOST']}:"
            f"{os.environ['DB_PORT']}/{os.environ['DB_NAME']}"
        )
        return create_engine(
            url,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 10, "sslmode": "require"},
        )
    except Exception:
        return None


def available(table: str = TABLE) -> bool:
    """Whether the table can actually be read right now.

    Connects rather than checking that an engine could be built, because the
    two differ: credentials can be present and wrong, the driver can be absent,
    the host can refuse. Callers use this to decide which store is
    authoritative, so a hopeful answer is worse than none.
    """
    eng = engine()
    if eng is None:
        return False
    try:
        with eng.connect() as conn:
            conn.exec_driver_sql(f"SELECT 1 FROM {table} LIMIT 1")
        return True
    except Exception:
        return False


def ensure_table(ddl: str = CREATE_SQL, table: str = TABLE) -> bool:
    """Create the table if it is absent, migrating it first. True if it exists.

    The rename runs BEFORE the create, so a table written under the old column
    name is migrated in place rather than left beside a new empty one. CREATE
    TABLE IF NOT EXISTS then finds the migrated table and does nothing.
    """
    eng = engine()
    if eng is None:
        return False
    try:
        with eng.begin() as conn:
            if _migrate_forecast_date_to_week_of(conn, table):
                print(f"  migrated {table}: forecast_date -> week_of")
            for statement in ddl.strip().split(";"):
                if statement.strip():
                    conn.exec_driver_sql(statement)
        return True
    except Exception:
        return False


def read(columns: list[str], table: str = TABLE) -> pd.DataFrame | None:
    """Every stored row, or None when the table cannot be read."""
    eng = engine()
    if eng is None:
        return None
    try:
        with eng.connect() as conn:
            df = pd.read_sql(f"SELECT * FROM {table}", conn)
    except Exception:
        return None
    for col in ("week_of", "ds"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    for col in columns:
        if col not in df.columns:
            df[col] = None
    return df[columns]


def upsert(df: pd.DataFrame, table: str = TABLE, ddl: str = CREATE_SQL,
           replace_run: bool = True) -> int:
    """Replace whole runs. Rows written, or -1 when the database is unreachable.

    -1 rather than 0 on failure, because 0 is a real answer: an empty frame
    writes nothing and succeeds. A caller that cannot tell those apart would
    report a silent failure as an ordinary no-op.

    A run is one (model_version, week_of) pair, and `replace_run` clears it
    before inserting so that after the write the week holds exactly what this
    run produced.

    This used to be a plain upsert on the key, justified as replacing "its own
    rows and nothing else". That is the bug. Keying on unique_id means a re-run
    can only overwrite the SKUs it produces; it has no way to remove a SKU it no
    longer produces. When the smooth set went from 467 SKUs to 338 on
    2026-08-10, the 129 dropped SKUs stayed in the table at their old values and
    the week described two segmentations at once, with nothing on screen saying
    so. Any change to the segmentation rules does this, so it was going to
    recur.

    Deleting the run first is what the legacy track already does in
    `src.db.write_forward_forecasts`, and it is what makes "one stored run per
    training week" true rather than aspirational.

    Scoped to one model_version, so two versions can be stored against the same
    week for comparison without either clearing the other.

    Both statements share a transaction: a failed insert rolls the delete back,
    so the failure mode is the previous run surviving, not the week vanishing.
    That leaves one real constraint, which the pipeline already meets: a run has
    to write its week in a single call. Both callers do
    (`ml_forward_forecast.py` passes the whole frame once), and a future caller
    that wrote in pieces would have each piece delete the last. Pass
    replace_run=False to append to a run instead.
    """
    if df is None or df.empty:
        return 0
    eng = engine()
    if eng is None or not ensure_table(ddl, table):
        return -1

    out = df.copy()
    for col in ("week_of", "ds"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col]).dt.date
    if "run_at" in out.columns:
        out["run_at"] = out["run_at"].astype(str)

    cols = list(out.columns)
    updatable = [c for c in cols if c not in KEY]
    placeholders = ", ".join(f":{c}" for c in cols)
    assignments = ", ".join(f"{c} = EXCLUDED.{c}" for c in updatable) or "yhat = EXCLUDED.yhat"
    sql = (
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT ({', '.join(KEY)}) DO UPDATE SET {assignments}"
    )

    try:
        from sqlalchemy import text

        records = out.to_dict(orient="records")
        with eng.begin() as conn:
            if replace_run:
                # One DELETE per (model_version, week_of) in the frame. Normally
                # exactly one; more only if a caller ever batches runs together.
                runs = (out[["model_version", "week_of"]]
                        .drop_duplicates().to_dict(orient="records"))
                conn.execute(
                    text(f"DELETE FROM {table} "
                         "WHERE model_version = :model_version AND week_of = :week_of"),
                    runs,
                )
            # Chunked because a single execute of 40k parameter sets is slow to
            # build and holds one long transaction; the pipeline is weekly, so
            # throughput does not matter but a timeout would.
            #
            # ON CONFLICT is kept even though the delete above means it should
            # never fire: it costs nothing, and it keeps the write idempotent
            # for anyone calling with replace_run=False.
            for i in range(0, len(records), 1000):
                conn.execute(text(sql), records[i:i + 1000])
        return len(records)
    except Exception as exc:
        # Record why. The return value is still -1, because callers treat this
        # as non-fatal on purpose and should not start handling exceptions, but
        # -1 alone was actively misleading: `ml_forward_forecast.py` prints it
        # as "no DB credentials, or it could not be reached", and on 2026-08-12
        # that message was shown for a failure that had not been diagnosed at
        # all. A swallowed exception that gets reported as a specific,
        # confidently wrong cause is worse than no message.
        global LAST_ERROR
        LAST_ERROR = f"{type(exc).__name__}: {exc}"
        return -1
