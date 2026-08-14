"""V1 forecasting formula — stream-based blend with seasonal modifier.

Single source of truth for V1_WINDOWS and SEASONAL. Every file that uses the
V1 formula imports these constants from here, either directly or through
scripts/compare_v1.py. Changing them here changes the live pipeline, the
accuracy benchmarks, the final test runner, and the forward V1 baseline.

The live pipeline also reads from shipcore.fc_user_preferences (see
_load_v1_config), falling back to these constants when the DB row is absent
or unreachable. To change V1 for the live forecast only, edit the DB row; to
change it everywhere including benchmarks, edit the constants below.

Computes a total forecast for a given horizon in days, from the DB
snapshot (fc_velocity_link_snapshot_forecast with order_type and channel).
"""
import calendar
import os
import time
from datetime import timedelta
from urllib.parse import quote_plus

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# load_raw_for_v1 reads the velocity table; without override a stale
# shell DB_PASSWORD shadows .env and authentication fails. Documented in
# CLAUDE.md as a truncated-password incident; hit again on 2026-08-10 by
# scripts/ml_31_export_channel_mix.py, which imports from here.
load_dotenv(override=True)

# Module-level defaults — used as fallbacks when DB is unreachable.
SEASONAL = {
    1: 0.75, 2: 0.80, 3: 0.90, 4: 0.95,
    5: 1.00, 6: 1.00, 7: 1.00, 8: 1.00, 9: 1.00,
    10: 1.10, 11: 1.25, 12: 1.30,
}

V1_WINDOWS = [
    (90, 0.10, "sales"),
    (60, 0.15, "sales"),
    (30, 0.30, "sales"),
    (15, 0.20, "sales"),
    (7,  0.15, "sales"),
    (30, 0.10, "preorder"),
]

_MONTH_ABBR = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
               "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}

_v1_config_cache: dict = {"expires": 0.0, "seasonal": None, "windows": None}


def _load_v1_config() -> tuple[dict, list]:
    """Return (seasonal_dict, windows_list) from DB global preferences, cached 5 minutes."""
    now = time.monotonic()
    if now < _v1_config_cache["expires"] and _v1_config_cache["seasonal"] is not None:
        return _v1_config_cache["seasonal"], _v1_config_cache["windows"]
    try:
        with _engine().connect() as conn:
            rows = conn.execute(text("""
                SELECT key, value FROM shipcore.fc_user_preferences
                WHERE user_id = 'global'
                  AND key IN (
                    'planning-dashboard-seasonal-factors',
                    'planning-dashboard-sales-window-weights'
                  )
            """)).fetchall()
        row_map = {r.key: r.value for r in rows}

        sf_raw = row_map.get("planning-dashboard-seasonal-factors")
        seasonal = (
            {_MONTH_ABBR[k]: float(v) for k, v in sf_raw.items() if k in _MONTH_ABBR}
            if sf_raw else SEASONAL
        )

        ww_raw = row_map.get("planning-dashboard-sales-window-weights")
        windows = (
            [(int(w["days"]), float(w["weight"]), w["order_type"]) for w in ww_raw]
            if ww_raw else V1_WINDOWS
        )

        _v1_config_cache.update({"expires": now + 300.0, "seasonal": seasonal, "windows": windows})
        return seasonal, windows
    except Exception:
        return SEASONAL, V1_WINDOWS


_engine_instance = None

def _engine():
    global _engine_instance
    if _engine_instance is None:
        url = "postgresql+psycopg2://{}:{}@{}:{}/{}".format(
            quote_plus(os.getenv("DB_USER")),
            quote_plus(os.getenv("DB_PASSWORD")),
            os.getenv("DB_HOST"),
            os.getenv("DB_PORT"),
            os.getenv("DB_NAME"),
        )
        _engine_instance = create_engine(
            url,
            pool_size=3,
            max_overflow=3,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 10, "sslmode": "require"},
        )
    return _engine_instance


def load_raw_for_v1() -> pd.DataFrame:
    """Pull order_date, unique_id, link_qty, channel, order_type from the snapshot."""
    engine = _engine()
    with engine.connect() as conn:
        raw = pd.read_sql(text("""
            SELECT order_date, link_master_sku AS unique_id, link_qty, channel, order_type
            FROM shipcore.fc_velocity_link_snapshot_forecast
        """), conn, parse_dates=["order_date"])
    raw["order_date"] = pd.to_datetime(raw["order_date"]).dt.normalize()
    return raw


def _assign_stream(channel: str, order_type: str) -> str | None:
    if channel == "Amazon FBA":
        return "fba"
    if order_type in ("sales", "preorder"):
        return f"west_{order_type}"
    if order_type == "ttm":
        return "east_sales"
    if order_type == "ttm_preorder":
        return "east_preorder"
    return None


def build_index(raw: pd.DataFrame) -> dict:
    """Build {(uid, stream): cumsum Series} for fast window lookups."""
    raw = raw.copy()
    raw["stream"] = raw.apply(
        lambda r: _assign_stream(r["channel"], r["order_type"]), axis=1
    )
    raw = raw[raw["stream"].notna()].drop(columns=["channel", "order_type"])

    daily = (
        raw.groupby(["unique_id", "stream", "order_date"])["link_qty"]
        .sum()
        .reset_index()
    )
    full_range = pd.date_range(raw["order_date"].min(), raw["order_date"].max(), freq="D")

    index: dict = {}
    for (uid, stream), grp in daily.groupby(["unique_id", "stream"]):
        s = grp.set_index("order_date")["link_qty"].reindex(full_range, fill_value=0)
        index[(uid, stream)] = s.cumsum()
    return index


def _window_sum(index: dict, uid: str, stream: str, end: pd.Timestamp, days: int) -> float:
    cs = index.get((uid, stream))
    if cs is None:
        return 0.0
    start = end - timedelta(days=days)
    end_val   = float(cs.asof(end))   if end   >= cs.index[0] else 0.0
    start_val = float(cs.asof(start)) if start >= cs.index[0] else 0.0
    return max(0.0, end_val - start_val)


def _blend_rate(index: dict, uid: str, prefix: str, as_of: pd.Timestamp) -> float:
    _, windows = _load_v1_config()
    rate = 0.0
    for days, weight, kind in windows:
        stream = f"{prefix}_preorder" if kind == "preorder" else f"{prefix}_sales"
        rate += weight * (_window_sum(index, uid, stream, as_of, days) / days)
    return max(0.0, rate)


def _dampen(S: float, R: float) -> float:
    if R == 0:
        return S
    change = abs((S - R) / R)
    return 0.1 * R + 0.9 * S if change < 0.5 else 0.2 * R + 0.8 * S


def _daily_rate(index: dict, uid: str, cutoff: pd.Timestamp) -> float:
    prev = cutoff - timedelta(days=7)
    west = _dampen(_blend_rate(index, uid, "west", cutoff),
                   _blend_rate(index, uid, "west", prev))
    east = _dampen(_blend_rate(index, uid, "east", cutoff),
                   _blend_rate(index, uid, "east", prev))
    fba  = _window_sum(index, uid, "fba", cutoff, 30) / 30
    return west + east + fba


def _seasonal_modifier(start: pd.Timestamp, end: pd.Timestamp) -> float:
    seasonal, _ = _load_v1_config()
    total_days = (end - start).days + 1
    weighted = 0.0
    current = start
    while current <= end:
        last_of_month = pd.Timestamp(
            current.year, current.month,
            calendar.monthrange(current.year, current.month)[1],
        )
        chunk_end  = min(end, last_of_month)
        chunk_days = (chunk_end - current).days + 1
        weighted  += seasonal[current.month] * chunk_days
        current    = chunk_end + timedelta(days=1)
    return weighted / total_days


def forecast_total(index: dict, uid: str, cutoff: pd.Timestamp, horizon_days: int) -> float:
    """V1 forecast total for `horizon_days` calendar days starting the day after cutoff."""
    rate  = _daily_rate(index, uid, cutoff)
    start = cutoff + timedelta(days=1)
    end   = cutoff + timedelta(days=horizon_days)
    mod   = _seasonal_modifier(start, end)
    return max(0.0, rate * horizon_days * mod)


def compute_v1_per_week(
    unique_ids: list[str],
    cutoff: pd.Timestamp,
    horizon_weeks: int,
    index: dict,
) -> dict[str, float]:
    """Return {uid: v1_yhat_per_week} for all requested SKUs.

    v1_yhat_per_week = V1 horizon total / horizon_weeks
    so that SUM(v1_yhat) over all ds rows equals the full V1 horizon forecast.
    """
    horizon_days = horizon_weeks * 7
    result: dict[str, float] = {}
    for uid in unique_ids:
        total = forecast_total(index, uid, cutoff, horizon_days)
        result[uid] = round(total / horizon_weeks, 4)
    return result
