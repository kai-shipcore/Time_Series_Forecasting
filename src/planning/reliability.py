"""Per-SKU forecast reliability.

Answers "how far off has this SKU's forecast usually been", using the stored
backtest results rather than any new modelling. Each SKU's absolute errors and
actuals are summed across every development window it appeared in, then divided,
which is the same pooled-WAPE convention used everywhere else in the project
(sum the errors and the actuals before dividing, so a heavier window counts more).

Only the served model's own errors are counted. The accuracy export holds one row
per SKU per window for the model AND one for the V1 spreadsheet baseline, so it
must be filtered by model version first. Pooling both together mixes two methods'
errors and double-counts the actuals, which understates reliability across the
board and misplaces roughly a third of SKUs into the wrong tier.

Not every forecast SKU has a history. A SKU that was too new to be eligible in the
older backtest windows has no measured error, and that absence is reported as its
own state rather than as a blank or a zero, because "we have never checked this
one" is a different statement from "this one is accurate".
"""

from __future__ import annotations

import pandas as pd

from src.planning._cache import cache as _cache

from src.planning import data as D

# Tier boundaries on pooled WAPE, set against the observed distribution of the 260
# SKUs that have history for the served model (median 0.176, upper quartile 0.286,
# ninth decile 0.468). These cuts split the measured population roughly 107 good,
# 90 fair, 63 poor, so each tier is large enough to be worth filtering by and
# "poor" stays selective enough to mean something.
GOOD_MAX = 0.15
FAIR_MAX = 0.30

TIER_ORDER = ["good", "fair", "poor", "none"]
TIER_LABEL = {
    "good": f"good (≤{GOOD_MAX:.0%})",
    "fair": f"fair ({GOOD_MAX:.0%}–{FAIR_MAX:.0%})",
    "poor": f"poor (>{FAIR_MAX:.0%})",
    "none": "no history",
}

def tier(wape: float | None) -> str:
    """Bucket a pooled WAPE into good / fair / poor, or none when unmeasured."""
    if wape is None or pd.isna(wape):
        return "none"
    if wape <= GOOD_MAX:
        return "good"
    if wape <= FAIR_MAX:
        return "fair"
    return "poor"


def served_version() -> str | None:
    """The model version behind the forecasts currently on screen.

    Reliability is filtered to this, so the error shown for a SKU always
    describes the same model that produced its forecast.
    """
    fc = D.load_forecasts()
    if fc.empty or "model_version" not in fc.columns:
        return None
    return str(fc["model_version"].iloc[0])


@_cache(show_spinner=False)
def per_sku(version: str | None = None) -> pd.DataFrame:
    """Reliability for every SKU that has backtest history for `version`.

    Returns unique_id, wape, n_windows, tier. SKUs with no history are absent;
    join with how="left" and let `tier` fall back to "none".
    """
    acc = D.load_ml_accuracy_by_sku()
    if acc.empty:
        return pd.DataFrame(columns=["unique_id", "wape", "n_windows", "tier"])

    # The export interleaves the model and the V1 baseline. Score one method.
    version = version or served_version()
    if version is not None and "model_version" in acc.columns:
        acc = acc[acc["model_version"] == version]
    if acc.empty:
        return pd.DataFrame(columns=["unique_id", "wape", "n_windows", "tier"])

    g = acc.groupby("unique_id").agg(
        ae=("ae", "sum"), actual=("y_total", "sum"), n_windows=("window", "nunique")
    )
    g = g[g["actual"] > 0].copy()
    g["wape"] = g["ae"] / g["actual"]
    g["tier"] = [tier(w) for w in g["wape"]]
    return g.reset_index()[["unique_id", "wape", "n_windows", "tier"]]


def attach(plan: pd.DataFrame) -> pd.DataFrame:
    """Add wape / n_windows / tier columns to a planning table."""
    rel = per_sku()
    out = plan.merge(rel, on="unique_id", how="left")
    out["tier"] = out["tier"].fillna("none")
    out["n_windows"] = out["n_windows"].fillna(0).astype(int)
    return out


def display(wape: float | None) -> str:
    """Short label for a table cell, e.g. '±13%' or 'no history'."""
    if wape is None or pd.isna(wape):
        return "no history"
    return f"±{wape:.0%}"


def tier_counts(plan: pd.DataFrame) -> dict[str, int]:
    """Count of SKUs per tier, for the table legend."""
    if "tier" not in plan.columns:
        plan = attach(plan)
    counts = plan["tier"].value_counts().to_dict()
    return {t: int(counts.get(t, 0)) for t in TIER_ORDER}
