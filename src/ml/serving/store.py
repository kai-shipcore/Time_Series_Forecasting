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
KEY = ["model_version", "forecast_date", "unique_id", "ds"]

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
    forecast_date   date        NOT NULL,
    unique_id       text        NOT NULL,
    ds              date        NOT NULL,
    yhat            double precision,
    bucket          text,
    history_length  text,
    segment         text,
    served_by       text,
    run_at          text,
    PRIMARY KEY (model_version, forecast_date, unique_id, ds)
);
CREATE INDEX IF NOT EXISTS {index_name}
    ON {table} (model_version, forecast_date);
"""


CREATE_SQL = _create_sql(TABLE, "ml_forecast_history_run_idx")
FORWARD_CREATE_SQL = _create_sql(FORWARD_TABLE, "ml_forward_forecasts_run_idx")

#: Columns both tables carry, in the order the parquet writes them.
COLUMNS = KEY + ["yhat", "bucket", "history_length", "segment", "served_by", "run_at"]


def write_forward(df: pd.DataFrame) -> int:
    """Store the current horizon. Rows written, or -1 if the database is absent.

    The counterpart to `src.db.write_forward_forecasts` on the legacy track, and
    the reason the Action List can be read from a machine that did not produce
    the forecast.

    Accumulates rather than replacing wholesale. `_read_forecasts` already takes
    the latest `forecast_date` and ignores the rest, so older horizons cost a
    little space and are occasionally useful for asking what last week's
    recommendation was. Re-running inside one training week still replaces its
    own rows, because the key includes `forecast_date`.
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


def ensure_table(ddl: str = CREATE_SQL) -> bool:
    """Create the table if it is absent. True if it exists afterwards."""
    eng = engine()
    if eng is None:
        return False
    try:
        with eng.begin() as conn:
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
    for col in ("forecast_date", "ds"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    for col in columns:
        if col not in df.columns:
            df[col] = None
    return df[columns]


def upsert(df: pd.DataFrame, table: str = TABLE, ddl: str = CREATE_SQL) -> int:
    """Insert rows, replacing any that share the primary key. Rows written, or -1.

    -1 rather than 0 on failure, because 0 is a real answer: an empty frame
    writes nothing and succeeds. A caller that cannot tell those apart would
    report a silent failure as an ordinary no-op.

    Upsert rather than delete-then-insert so a re-run within the same training
    week replaces its own rows and nothing else, which is the rule the parquet
    path already follows and the one the production pipeline uses.
    """
    if df is None or df.empty:
        return 0
    eng = engine()
    if eng is None or not ensure_table(ddl):
        return -1

    out = df.copy()
    for col in ("forecast_date", "ds"):
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
            # Chunked because a single execute of 40k parameter sets is slow to
            # build and holds one long transaction; the pipeline is weekly, so
            # throughput does not matter but a timeout would.
            for i in range(0, len(records), 1000):
                conn.execute(text(sql), records[i:i + 1000])
        return len(records)
    except Exception:
        return -1
