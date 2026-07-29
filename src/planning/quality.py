"""Data-quality exception checks.

Each check returns the offending rows so a reviewer can act on them. Checks that
require data not present in this repo (e.g. order-line level preorder tagging) are
returned with ``available=False`` and an explanation rather than silently passing.
"""

from __future__ import annotations

import pandas as pd

_COLS = [
    "unique_id",
    "product_name",
    "product_category",
    "history_group",
    "available_inventory",
    "confirmed_inbound",
    "recommended_order_qty",
    "recent_units",
    "active_weeks",
]


def _slice(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    cols = [c for c in _COLS if c in df.columns]
    return df.loc[mask, cols].copy()


# Short labels for showing a flag next to the SKU it concerns, rather than only
# on a dedicated screen. A warning is worth most at the moment a decision is
# being made, which is on the list and the detail page.
SHORT_LABELS = {
    "missing_mapping": "no segmentation profile",
    "no_incoming_container": "nothing inbound",
    "new_no_history": "no sales in 4 weeks",
    "forecast_runs_high": "forecast well above recent sales",
}


def flags_by_sku(plan: pd.DataFrame) -> dict[str, list[str]]:
    """Map each flagged SKU to the short labels of the checks it trips."""
    out: dict[str, list[str]] = {}
    for check in run_all(plan):
        if not check["available"] or check["rows"].empty:
            continue
        label = SHORT_LABELS.get(check["key"], check["title"])
        for uid in check["rows"]["unique_id"]:
            out.setdefault(str(uid), []).append(label)
    return out


def run_all(plan: pd.DataFrame) -> list[dict]:
    """Return a list of check results.

    Each result: {key, title, description, available, rows (DataFrame), count}.
    """
    results: list[dict] = []

    def add(key, title, description, mask=None, available=True, note=""):
        if available and mask is not None:
            rows = _slice(plan, mask)
        else:
            rows = pd.DataFrame(columns=_COLS)
        results.append(
            {
                "key": key,
                "title": title,
                "description": description if not note else f"{description} {note}",
                "available": available,
                "rows": rows,
                "count": int(len(rows)),
            }
        )

    # 1. Missing SKU mappings: forecast SKU with no segmentation profile.
    # This does NOT yet check "no match in the inventory system" -- build_planning_table
    # fills inventory columns with 0 right after the join, so an unmatched SKU is
    # currently indistinguishable from one with genuinely zero inventory. Revisit once
    # real inventory data is wired in (some forecasted SKUs are known not to match it).
    missing_map = plan["bucket"].isna()
    add(
        "missing_mapping",
        "Missing SKU mappings",
        "Forecasted SKUs with no segmentation profile.",
        missing_map,
    )

    # 2. Preorder classification: needs order-line data not in this repo.
    add(
        "preorder_classification",
        "Unclear / misclassified preorder transactions",
        "Requires the order-line export (preorder tagging) which is not available in this repo.",
        available=False,
    )

    # 4. Stocking out soon with no confirmed inbound.
    no_inbound = (
        plan.get("stockout_soon", pd.Series(False, index=plan.index))
        & (plan["confirmed_inbound"] <= 0)
        & (plan["recommended_order_qty"] > 0)
    )
    add(
        "no_incoming_container",
        "No incoming containers",
        "SKUs projected to stock out within the risk window with no confirmed inbound quantity.",
        no_inbound,
    )

    # 4b. The forecast stands well above the rate the SKU is currently selling at.
    # Not a data-quality defect in the usual sense: the data is fine and the
    # model is behaving as measured. It sits here because this is the channel
    # that puts a caveat next to the SKU it concerns, on both screens, and a
    # planner acting on the recommended quantity needs it at that moment.
    runs_high = plan.get("forecast_runs_high", pd.Series(False, index=plan.index)).fillna(False)
    add(
        "forecast_runs_high",
        "Forecast well above recent sales",
        "SKUs forecast at 1.5x or more of their recent 4-week rate, where the excess is "
        "also material in units over the horizon. Both conditions are required, since a "
        "large ratio on a SKU selling a fraction of a unit a week is not worth acting on. "
        "The recommended quantity still uses the model.",
        runs_high,
    )

    # 5. Short-history SKUs with no recent sales.
    new_no_history = (plan["history_group"] == "short") & (plan["recent_units"] <= 0)
    add(
        "new_no_history",
        "Short-history SKUs with no recent sales",
        "Short-history SKUs that have a forecast but no demand in the last 4 weeks.",
        new_no_history,
    )

    return results
