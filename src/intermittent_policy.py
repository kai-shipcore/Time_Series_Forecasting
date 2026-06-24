# Stage 9 (intermittent path): Reorder-point policy for intermittent SKUs.
#
# For SKUs where week-by-week point forecasting is unreliable (≥30% zero weeks),
# we derive an inventory policy directly from the empirical demand distribution:
#
#   reorder_point  — hold at least this many units; replenish when stock falls here
#   order_qty      — how much to order each time; scaled by spike_tier
#   safety_stock   — buffer above expected lead-time demand at the target service level
#   spike_tier     — slow / moderate / large / extreme (based on max single-week demand)
#
# Reorder point is the (SERVICE_LEVEL)th percentile of the empirical rolling
# lead-time demand distribution — no parametric assumption required.
#
# order_qty is calibrated per tier so that restocking covers the SPIKE level of
# demand, not just the average:
#   slow     (max ≤ 4):  mean demand size × inter-arrival  (standard)
#   moderate (max 5–7):  75th-pct demand size × inter-arrival
#   large    (max 8–14): 90th-pct demand size × inter-arrival
#   extreme  (max ≥ 15): 95th-pct demand size × inter-arrival
#
# Output:
#   outputs/forecasts/intermittent_policy.csv
import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    LEAD_TIME_WEEKS,
    OUTPUTS_FORECASTS,
    SERVICE_LEVEL,
    TRIM_TRAILING_WEEKS,
)

# Spike tier thresholds (max single-week demand)
TIER_MODERATE = 5
TIER_LARGE    = 8
TIER_EXTREME  = 15

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
OUT_PATH = OUTPUTS_FORECASTS / "intermittent_policy.csv"


def _spike_tier(max_demand: float) -> str:
    if max_demand >= TIER_EXTREME:
        return "extreme"
    if max_demand >= TIER_LARGE:
        return "large"
    if max_demand >= TIER_MODERATE:
        return "moderate"
    return "slow"


def _sku_stats(y: np.ndarray, lead_time: int, service_level: float) -> dict:
    """Compute policy parameters for one intermittent SKU."""
    n_weeks  = len(y)
    nonzero  = y[y > 0]
    n_nonzero = len(nonzero)
    max_demand = float(y.max()) if n_nonzero > 0 else 0.0
    tier = _spike_tier(max_demand)

    if n_nonzero == 0:
        return {
            "n_weeks":             n_weeks,
            "n_demand_weeks":      0,
            "demand_freq":         0.0,
            "mean_demand_size":    0.0,
            "std_demand_size":     0.0,
            "mean_inter_arrival":  float(n_weeks),
            "mean_weekly_demand":  0.0,
            "reorder_point":       0.0,
            "safety_stock":        0.0,
            "order_qty":           0.0,
            "max_demand":          0.0,
            "peak_to_mean":        0.0,
            "top10_concentration": 0.0,
            "spike_tier":          "slow",
            "dead_sku":            True,
        }

    demand_freq        = n_nonzero / n_weeks
    mean_demand_size   = float(nonzero.mean())
    std_demand_size    = float(nonzero.std(ddof=1)) if n_nonzero > 1 else 0.0
    mean_inter_arrival = n_weeks / n_nonzero
    mean_weekly_demand = float(y.mean())

    # Empirical lead-time demand distribution (rolling L-week sums)
    if n_weeks >= lead_time:
        windows = np.array([y[i : i + lead_time].sum()
                            for i in range(n_weeks - lead_time + 1)])
    else:
        windows = np.array([y.sum()])

    reorder_point = float(np.percentile(windows, service_level * 100))
    safety_stock  = max(0.0, reorder_point - lead_time * mean_weekly_demand)

    # Order quantity calibrated to spike tier
    # Higher tiers use a demand percentile rather than the mean so the restock
    # covers a spike-level event, not just an average event.
    if tier == "slow":
        base_demand = mean_demand_size
    elif tier == "moderate":
        base_demand = float(np.percentile(nonzero, 75)) if n_nonzero >= 4 else mean_demand_size
    elif tier == "large":
        base_demand = float(np.percentile(nonzero, 90)) if n_nonzero >= 4 else mean_demand_size
    else:  # extreme
        base_demand = float(np.percentile(nonzero, 95)) if n_nonzero >= 4 else mean_demand_size

    order_qty = base_demand * max(1.0, mean_inter_arrival)

    # Spike characterisation
    peak_to_mean = max_demand / mean_weekly_demand if mean_weekly_demand > 0 else 0.0
    top_n = max(1, int(np.ceil(n_weeks * 0.10)))
    top10_concentration = float(np.sort(y)[-top_n:].sum() / y.sum()) if y.sum() > 0 else 0.0

    return {
        "n_weeks":             n_weeks,
        "n_demand_weeks":      n_nonzero,
        "demand_freq":         round(demand_freq, 4),
        "mean_demand_size":    round(mean_demand_size, 2),
        "std_demand_size":     round(std_demand_size, 2),
        "mean_inter_arrival":  round(mean_inter_arrival, 1),
        "mean_weekly_demand":  round(mean_weekly_demand, 4),
        "reorder_point":       round(reorder_point, 2),
        "safety_stock":        round(safety_stock, 2),
        "order_qty":           round(order_qty, 2),
        "max_demand":          round(max_demand, 2),
        "peak_to_mean":        round(peak_to_mean, 2),
        "top10_concentration": round(top10_concentration, 3),
        "spike_tier":          tier,
        "dead_sku":            False,
    }


def compute_policy(
    weekly: pd.DataFrame,
    profiles: pd.DataFrame,
    lead_time: int = LEAD_TIME_WEEKS,
    service_level: float = SERVICE_LEVEL,
) -> pd.DataFrame:
    all_weeks     = sorted(weekly["ds"].unique())
    trimmed_weeks = all_weeks[:-TRIM_TRAILING_WEEKS] if TRIM_TRAILING_WEEKS else all_weeks
    weekly        = weekly[weekly["ds"].isin(trimmed_weeks)].copy()

    interm       = profiles[profiles["bucket"] == "intermittent"].copy()
    train_starts = pd.to_datetime(interm.set_index("unique_id")["train_start"])

    rows = []
    for uid, grp in weekly[weekly["unique_id"].isin(interm["unique_id"])].groupby("unique_id"):
        grp = grp.sort_values("ds")
        ts  = train_starts.get(uid)
        if ts is not None:
            grp = grp[grp["ds"] >= ts]
        y = grp["y"].values
        if len(y) == 0:
            continue
        stats = _sku_stats(y, lead_time, service_level)
        stats["unique_id"] = uid
        rows.append(stats)

    policy = pd.DataFrame(rows)[[
        "unique_id",
        "n_weeks",
        "n_demand_weeks",
        "demand_freq",
        "mean_demand_size",
        "std_demand_size",
        "mean_inter_arrival",
        "mean_weekly_demand",
        "reorder_point",
        "safety_stock",
        "order_qty",
        "max_demand",
        "peak_to_mean",
        "top10_concentration",
        "spike_tier",
        "dead_sku",
    ]]

    OUTPUTS_FORECASTS.mkdir(parents=True, exist_ok=True)
    policy.to_csv(OUT_PATH, index=False)

    active = policy[~policy["dead_sku"]]
    print(f"Intermittent SKUs processed : {len(policy)}")
    print(f"  Active (≥1 demand week)   : {len(active)}")
    print(f"  Dead   (zero demand ever) : {policy['dead_sku'].sum()}")
    print(f"\nPolicy parameters:")
    print(f"  Lead time    : {lead_time} weeks")
    print(f"  Service level: {service_level:.0%}")
    print(f"\nSKUs by spike tier:")
    print(active["spike_tier"].value_counts().reindex(["slow","moderate","large","extreme"]).to_string())
    print(f"\nMedian policy values by tier:")
    cols = ["mean_demand_size", "mean_inter_arrival", "reorder_point", "order_qty"]
    print(active.groupby("spike_tier")[cols].median()
          .reindex(["slow","moderate","large","extreme"]).round(2).to_string())
    print(f"\nSaved: {OUT_PATH}")

    return policy


if __name__ == "__main__":
    weekly   = pd.read_parquet(PROCESSED_DIR / "sales_clean.parquet")
    profiles = pd.read_csv(PROCESSED_DIR / "sku_profiles.csv")
    profiles["train_start"] = pd.to_datetime(profiles["train_start"])
    weekly["ds"] = pd.to_datetime(weekly["ds"])

    print(f"Loaded: {weekly['unique_id'].nunique()} SKUs\n")
    compute_policy(weekly, profiles)
