"""Live inventory for the planning views, read from the databases directly.

The inventory tables are maintained by the Commerce Integration application and
refreshed on their own schedule, so there is no reason for this project to hold a
copy. Reading them at request time removes the export step from the loop and
removes the class of bug where the dashboard is confidently wrong because nobody
re-ran a script.

The SQL lives here rather than in `scripts/export_inventory_snapshot.py` so that
the live path and the export produce the same figures by construction. The script
now imports from this module.

Two databases, because the figures live in two places:

- ``ecommerce_data.coverland_inventory_by_warehouse`` on the Commerce/Supabase
  connection holds on-hand, allocated, available and backorder per warehouse.
- ``shipcore.fc_container_items`` joined to ``fc_containers`` on the primary
  connection holds confirmed inbound and its ETA, and ``sc_products`` holds the
  product name.

Availability is not assumed. Every entry point returns None rather than raising
when the connections are absent or the query fails, so a dashboard run without
database access degrades to the exported CSV and then to clearly-labelled sample
data instead of failing to start.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Seconds a fetched snapshot is reused before the databases are asked again.
#: The underlying tables refresh on the Commerce app's schedule, not ours, so a
#: few minutes of staleness is invisible while a per-request query against two
#: databases would not be. Streamlit's own caching does not help here because
#: FastAPI shares this module and has none.
CACHE_TTL_SECONDS = 300

_lock = threading.Lock()
_cached: tuple[float, pd.DataFrame] | None = None

# Container statuses that count as confirmed inbound. Matches
# src/app/api/planning/dashboard/route.ts in the Commerce repo exactly, so the
# two screens agree on what is really on the water. 'shipped' means the final
# list was sent, 'packing_received' that the packing list arrived; drafts are
# deliberately excluded.
CONFIRMED_STATUSES = ("shipped", "packing_received")

# Containers where an order exists but is not yet committed. The Google Sheet
# import sets status from the header colour: blue 'shipped', orange
# 'packing_received', purple or uncoloured 'draft'. So this is where a container
# sits between someone deciding to order and the packing list arriving, which at
# an eight-week lead time is long enough to order the same units twice.
#
# Read from the same two tables the Container Planning screens use, so the two
# cannot disagree about what has been drafted. Reported separately and never
# added to confirmed inbound: see the note on the query below.
DRAFT_STATUSES = ("draft",)


def _status_list(statuses: tuple[str, ...]) -> str:
    """Render a status tuple as a SQL IN list.

    Not f-string interpolation of the tuple itself. A one-element Python tuple
    renders as ``('draft',)`` and that trailing comma is a syntax error in
    Postgres, so interpolating directly works for the two-element constant above
    and breaks silently the day either list has one entry.
    """
    return "(" + ", ".join(f"'{s}'" for s in statuses) + ")"


def _engine(prefix: str):
    """SQLAlchemy engine from ``{prefix}_*`` environment variables, or None.

    None rather than an exception, in every failure mode: the caller's job is to
    fall back, and an unusable connection is an ordinary condition in a working
    copy without credentials or without the Postgres driver installed.

    The whole body is guarded, not just the variable check. `create_engine`
    resolves the DBAPI eagerly and raises ModuleNotFoundError when psycopg2 is
    absent, which is a live case: the ML dependencies are pinned but the driver
    is not part of every environment that imports this package.
    """
    try:
        from sqlalchemy import create_engine

        host, port, name, user, pw = (
            os.getenv(f"{prefix}_HOST"), os.getenv(f"{prefix}_PORT"),
            os.getenv(f"{prefix}_NAME"), os.getenv(f"{prefix}_USER"),
            os.getenv(f"{prefix}_PASSWORD"),
        )
        if not all([host, port, name, user, pw]):
            return None
        url = f"postgresql+psycopg2://{quote_plus(user)}:{quote_plus(pw)}@{host}:{port}/{name}"
        return create_engine(url, connect_args={"connect_timeout": 10, "sslmode": "require"})
    except Exception:
        return None


def _load_env() -> None:
    """Read the repo's .env, overriding the shell.

    override=True is deliberate and documented in CLAUDE.md: stale DB_* values
    exported in a shell otherwise take precedence over the file, which has
    already cost one debugging session over a truncated password.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env", override=True)
    except Exception:
        pass


def _sql():
    from sqlalchemy import bindparam, text

    def q(s):
        return text(s).bindparams(bindparam("skus", expanding=True))

    return {
        # `available` is on_hand less allocated, and is NOT net of backorder;
        # confirmed against the source table, so summing available here and
        # adding backorder separately does not count the same units twice.
        "stock": q("""
            SELECT master_sku,
                   SUM(available)  AS available_inventory,
                   SUM(on_hand)    AS on_hand_physical,
                   SUM(allocated)  AS allocated,
                   SUM(backorder)  AS preorder_backlog
            FROM ecommerce_data.coverland_inventory_by_warehouse
            WHERE master_sku IN :skus
            GROUP BY master_sku
        """),
        # ETA floor of today on top of production's status filter: a container
        # still marked shipped after its ETA has passed is stale bookkeeping,
        # not a future arrival, and would otherwise inflate confirmed inbound
        # with units that have probably already landed.
        "inbound": q(f"""
            SELECT ci.master_sku,
                   SUM(ci.qty)     AS confirmed_inbound,
                   MIN(c.eta_date) AS inbound_eta
            FROM shipcore.fc_container_items ci
            JOIN shipcore.fc_containers c ON c.id = ci.container_id
            WHERE c.status IN {_status_list(CONFIRMED_STATUSES)}
              AND c.eta_date >= CURRENT_DATE
              AND ci.master_sku IN :skus
            GROUP BY ci.master_sku
        """),
        # Same two tables, same join, different status and a different ETA rule.
        #
        # The ETA rule is the part worth reading. Confirmed inbound floors at
        # today because a container still marked shipped after its ETA is stale
        # bookkeeping. A draft is the opposite case: it has often not been
        # scheduled yet, so a null ETA is the normal state of a container that
        # was drafted this week, and applying the same floor would drop exactly
        # the newest drafts, which are the ones that make a SKU look unordered.
        # So nulls are kept and only a date already in the past is excluded.
        #
        # These units are never added to confirmed inbound. A draft is not a
        # commitment and can be cancelled, so crediting it against the
        # recommendation would under-order the SKUs someone has already worried
        # about. It is reported alongside instead, and the screen shows the
        # disagreement rather than resolving it.
        "draft": q(f"""
            SELECT ci.master_sku,
                   SUM(ci.qty)     AS draft_inbound,
                   MIN(c.eta_date) AS draft_eta
            FROM shipcore.fc_container_items ci
            JOIN shipcore.fc_containers c ON c.id = ci.container_id
            WHERE c.status IN {_status_list(DRAFT_STATUSES)}
              AND (c.eta_date IS NULL OR c.eta_date >= CURRENT_DATE)
              AND ci.master_sku IN :skus
            GROUP BY ci.master_sku
        """),
        # fc_products.product_name is an unpopulated placeholder equal to
        # master_sku on every row, so sc_products is used despite matching a few
        # fewer SKUs. Rows where it merely echoes the SKU are treated as no name.
        "names": q("""
            SELECT master_sku, product_name
            FROM shipcore.sc_products
            WHERE product_name IS NOT NULL
              AND product_name <> master_sku
              AND master_sku IN :skus
        """),
        # Carried, not used. See docs: it may double-count confirmed inbound and
        # nobody has been able to say what it counts. Currently zero everywhere.
        "transit": q("""
            SELECT master_sku, COALESCE(transit_stock, 0) AS transit_stock
            FROM shipcore.fc_stats
            WHERE COALESCE(transit_stock, 0) <> 0
              AND master_sku IN :skus
        """),
        "snapshot_at": text(
            "SELECT MAX(created_at) FROM ecommerce_data.coverland_inventory_by_warehouse"
        ),
    }


def fetch(skus: list[str], diagnostics: bool = False) -> pd.DataFrame | None:
    """Inventory for `skus`, straight from the databases. None if unavailable.

    Returns one row per requested SKU. A SKU absent from the stock table keeps
    NaN in the stock columns rather than 0, so "no inventory record" stays
    distinguishable from "the record says zero"; confirmed inbound is different,
    since absence there is a real zero (no container line items matched).

    With diagnostics=True the physical on-hand and allocated columns are kept,
    which the export script reports on and the dashboard does not need.
    """
    if not skus:
        return None
    try:
        _load_env()
        primary, commerce = _engine("DB"), _engine("COMMERCE_DB")
    except Exception:
        return None
    if primary is None or commerce is None:
        return None

    try:
        sql = _sql()
    except Exception:
        return None
    try:
        with commerce.connect() as conn:
            stock = pd.read_sql(sql["stock"], conn, params={"skus": skus})
            snapshot_at = conn.execute(sql["snapshot_at"]).scalar()
        with primary.connect() as conn:
            inbound = pd.read_sql(sql["inbound"], conn, params={"skus": skus})
            draft = pd.read_sql(sql["draft"], conn, params={"skus": skus})
            names = pd.read_sql(sql["names"], conn, params={"skus": skus})
            transit = pd.read_sql(sql["transit"], conn, params={"skus": skus})
    except Exception:
        return None

    out = pd.DataFrame({"unique_id": list(skus)})
    for frame in (names, stock, inbound, draft, transit):
        out = out.merge(frame, left_on="unique_id", right_on="master_sku",
                        how="left").drop(columns=["master_sku"], errors="ignore")

    # Coerce before arithmetic. A query that matches nothing comes back with zero
    # rows, and read_sql then has no values to infer a dtype from, so the column
    # arrives as object; after the merge it is all-NaN object, and fillna/astype
    # on that either warns about silent downcasting or fails outright. Transit is
    # the live case, since it is currently zero for every SKU. Applied to all the
    # numeric columns rather than just that one, because the same holds for any
    # of them the moment a query returns nothing.
    for col in ("available_inventory", "on_hand_physical", "allocated",
                "preorder_backlog", "confirmed_inbound", "draft_inbound",
                "transit_stock"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Defensive: backorder has never been negative in this source, so this is a
    # contract statement rather than a fix for an observed problem.
    out["preorder_backlog"] = out["preorder_backlog"].abs()
    # Absence means the query found no matching rows, which is a real zero here,
    # unlike the stock columns above where absence means "no record at all" and
    # must stay null so it can be told apart from a recorded zero.
    out["confirmed_inbound"] = out["confirmed_inbound"].fillna(0).astype(int)
    # Same reasoning as confirmed inbound: the query ran and matched no line
    # items, which is a real zero. That is distinct from the column being absent
    # entirely, which is what an older exported CSV gives and which must stay
    # null so the screen can decline to claim that nothing is drafted.
    out["draft_inbound"] = out["draft_inbound"].fillna(0).astype(int)
    out["transit_stock"] = out["transit_stock"].fillna(0).astype(int)
    for col in ("inbound_eta", "draft_eta"):
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
        out[col] = out[col].where(out[col].notna(), "")

    keep = ["unique_id", "product_name", "available_inventory", "preorder_backlog",
            "confirmed_inbound", "inbound_eta", "draft_inbound", "draft_eta",
            "transit_stock"]
    if diagnostics:
        keep += ["on_hand_physical", "allocated"]
    out = out[keep]
    out.attrs["snapshot_at"] = str(snapshot_at) if snapshot_at is not None else None
    return out


def available_by_warehouse(skus: list[str]) -> pd.DataFrame:
    """Available stock per warehouse, for the export script's diagnostics.

    Answers whether any stock sits outside the four warehouses the Commerce app
    pivots into west and east. If it does, this project's sum over all
    warehouses is wider than theirs and the two screens will disagree for a
    reason that has nothing to do with inventory.
    """
    from sqlalchemy import bindparam, text

    commerce = _engine("COMMERCE_DB")
    if commerce is None or not skus:
        return pd.DataFrame(columns=["warehouse", "available"])
    q = text("""
        SELECT COALESCE(NULLIF(warehouse, ''), '(unspecified)') AS warehouse,
               SUM(available) AS available
        FROM ecommerce_data.coverland_inventory_by_warehouse
        WHERE master_sku IN :skus
        GROUP BY 1 ORDER BY 2 DESC
    """).bindparams(bindparam("skus", expanding=True))
    try:
        with commerce.connect() as conn:
            return pd.read_sql(q, conn, params={"skus": skus})
    except Exception:
        return pd.DataFrame(columns=["warehouse", "available"])


def fetch_cached(skus: list[str]) -> pd.DataFrame | None:
    """`fetch` behind a short TTL, shared across hosts and threads."""
    global _cached
    with _lock:
        if _cached is not None and time.monotonic() - _cached[0] < CACHE_TTL_SECONDS:
            return _cached[1].copy()
    got = fetch(skus)
    if got is None:
        return None
    with _lock:
        _cached = (time.monotonic(), got)
    return got.copy()


def clear_cache() -> None:
    """Drop the memo, so the next read goes to the databases."""
    global _cached
    with _lock:
        _cached = None
