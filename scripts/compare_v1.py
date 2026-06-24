#!/usr/bin/env python3
"""
Head-to-head comparison: V1 formula vs our StatsForecast models.

V1 computes daily_sales_current as the SUM of three independent streams:
  West   : order_type IN ('sales','preorder') AND channel != 'Amazon FBA'
           → 6-window weighted blend + 7-day previous + dampening
  East   : order_type IN ('ttm','ttm_preorder')
           → same 6-window blend + previous + dampening
  FBA    : channel = 'Amazon FBA'
           → 30-day sum / 30, no previous, no dampening

Forecast = (west_current + east_current + fba_current) × 70 × proportional_seasonal_modifier

Equalization rules applied:
  - Same target   : all order types combined (matches our model's y)
  - Same horizon  : 70 days, summed to a single number per SKU per window
  - Same windows  : exact CV cutoffs from cv_results.parquet; strict no-leakage
  - No modifier   : V1's seasonal modifier applied to V1 only
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import calendar
import os
from datetime import timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from urllib.parse import quote_plus

load_dotenv()

PROCESSED    = ROOT / "data" / "processed"
RAW_PATH     = PROCESSED / "orders_raw.parquet"
CV_PATH      = ROOT / "outputs" / "reports" / "cv_results.parquet"
OUT_PATH     = ROOT / "outputs" / "reports" / "v1_comparison.csv"
HORIZON_DAYS = 70

SEASONAL = {
    1: 0.75, 2: 0.80, 3: 0.90, 4: 0.95,
    5: 1.00, 6: 1.00, 7: 1.00, 8: 1.00, 9: 1.00,
    10: 1.10, 11: 1.25, 12: 1.30,
}

# V1 blend weights — same for West and East streams
V1_WINDOWS = [
    (90, 0.10, "sales"),
    (60, 0.15, "sales"),
    (30, 0.30, "sales"),
    (15, 0.20, "sales"),
    (7,  0.15, "sales"),
    (30, 0.10, "preorder"),
]


# ── Data pull ─────────────────────────────────────────────────────────────────

def _engine():
    url = "postgresql+psycopg2://{}:{}@{}:{}/{}".format(
        quote_plus(os.getenv("DB_USER")),
        quote_plus(os.getenv("DB_PASSWORD")),
        os.getenv("DB_HOST"),
        os.getenv("DB_PORT"),
        os.getenv("DB_NAME"),
    )
    return create_engine(url, connect_args={"connect_timeout": 10, "sslmode": "require"})


def load_raw() -> pd.DataFrame:
    if RAW_PATH.exists():
        print(f"Loading cached raw orders from {RAW_PATH}")
        return pd.read_parquet(RAW_PATH)

    print("Pulling raw orders from DB...")
    engine = _engine()
    with engine.connect() as conn:
        raw = pd.read_sql("""
            SELECT order_date, link_master_sku, link_qty, order_type, channel
            FROM shipcore.fc_velocity_link_snapshot
        """, conn, parse_dates=["order_date"])

    raw = raw.rename(columns={"link_master_sku": "unique_id"})
    raw["order_date"] = pd.to_datetime(raw["order_date"]).dt.normalize()

    # Assign each row to one of three non-overlapping V1 streams
    def assign_stream(row):
        if row["channel"] == "Amazon FBA":
            return "fba"
        if row["order_type"] in ("sales", "preorder"):
            return f"west_{row['order_type']}"   # west_sales / west_preorder
        if row["order_type"] in ("ttm", "ttm_preorder"):
            # map ttm→east_sales, ttm_preorder→east_preorder to reuse the V1_WINDOWS logic
            return "east_sales" if row["order_type"] == "ttm" else "east_preorder"
        return None  # unexpected type — exclude

    raw["stream"] = raw.apply(assign_stream, axis=1)
    raw = raw[raw["stream"].notna()].drop(columns=["order_type", "channel"])

    PROCESSED.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(RAW_PATH, index=False)
    print(f"  Saved {len(raw):,} rows → {RAW_PATH}")
    print("  Stream breakdown:")
    print(raw.groupby("stream")[["link_qty"]].agg(rows=("link_qty","count"), qty=("link_qty","sum")).to_string())
    return raw


# ── Cumulative-sum index ───────────────────────────────────────────────────────

def build_cumsum_index(raw: pd.DataFrame) -> dict:
    """
    Returns {(uid, stream): pd.Series(cumsum, index=daily DatetimeIndex)}.
    cumsum[d] = total units in that stream on or before day d.
    """
    print("\nBuilding daily cumulative sums per stream...")
    daily = (
        raw.groupby(["unique_id", "stream", "order_date"])["link_qty"]
        .sum()
        .reset_index()
    )
    full_range = pd.date_range(raw["order_date"].min(), raw["order_date"].max(), freq="D")

    index = {}
    for (uid, stream), grp in daily.groupby(["unique_id", "stream"]):
        s = grp.set_index("order_date")["link_qty"].reindex(full_range, fill_value=0)
        index[(uid, stream)] = s.cumsum()

    print(f"  Index covers {len(set(k[0] for k in index))} SKUs × {len(set(k[1] for k in index))} streams")
    return index


def window_sum(index: dict, uid: str, stream: str, end: pd.Timestamp, days: int) -> float:
    """Sum of demand in (end - days, end] — exactly `days` calendar days ending at end."""
    cs = index.get((uid, stream))
    if cs is None:
        return 0.0
    start     = end - timedelta(days=days)
    end_val   = float(cs.asof(end))   if end   >= cs.index[0] else 0.0
    start_val = float(cs.asof(start)) if start >= cs.index[0] else 0.0
    return max(0.0, end_val - start_val)


# ── V1 formula ────────────────────────────────────────────────────────────────

def _blend_rate(index: dict, uid: str, prefix: str, as_of: pd.Timestamp) -> float:
    """6-window weighted blend for one stream (west or east)."""
    rate = 0.0
    for days, weight, kind in V1_WINDOWS:
        stream = f"{prefix}_preorder" if kind == "preorder" else f"{prefix}_sales"
        s = window_sum(index, uid, stream, as_of, days)
        rate += weight * (s / days)
    return max(0.0, rate)


def _dampen(S: float, R: float) -> float:
    if R == 0:
        return S
    change = abs((S - R) / R)
    return 0.1 * R + 0.9 * S if change < 0.5 else 0.2 * R + 0.8 * S


def v1_daily_current(index: dict, uid: str, cutoff: pd.Timestamp) -> float:
    prev = cutoff - timedelta(days=7)

    # West stream
    west_S = _blend_rate(index, uid, "west", cutoff)
    west_R = _blend_rate(index, uid, "west", prev)
    west   = _dampen(west_S, west_R)

    # East stream
    east_S = _blend_rate(index, uid, "east", cutoff)
    east_R = _blend_rate(index, uid, "east", prev)
    east   = _dampen(east_S, east_R)

    # FBA stream — 30-day average only, no dampening
    fba = window_sum(index, uid, "fba", cutoff, 30) / 30

    return west + east + fba


def proportional_seasonal_modifier(start: pd.Timestamp, end: pd.Timestamp) -> float:
    total_days = (end - start).days + 1
    weighted   = 0.0
    current    = start
    while current <= end:
        last_of_month = pd.Timestamp(
            current.year, current.month,
            calendar.monthrange(current.year, current.month)[1],
        )
        chunk_end  = min(end, last_of_month)
        chunk_days = (chunk_end - current).days + 1
        weighted  += SEASONAL[current.month] * chunk_days
        current    = chunk_end + timedelta(days=1)
    return weighted / total_days


def v1_forecast(index: dict, uid: str, cutoff: pd.Timestamp) -> float:
    daily = v1_daily_current(index, uid, cutoff)
    start = cutoff + timedelta(days=1)
    end   = cutoff + timedelta(days=HORIZON_DAYS)
    return daily * HORIZON_DAYS * proportional_seasonal_modifier(start, end)


# ── Scoring ───────────────────────────────────────────────────────────────────

def score(cv_df: pd.DataFrame, index: dict) -> pd.DataFrame:
    meta   = {"unique_id", "ds", "cutoff", "y", "bucket", "history_length"}
    m_cols = [c for c in cv_df.columns if c not in meta]

    sel     = pd.read_csv(ROOT / "outputs" / "reports" / "selection.csv")
    sel_map = sel.set_index("unique_id")["model"].to_dict()

    print("\nScoring CV windows...")
    rows = []
    for (uid, cutoff), grp in cv_df.groupby(["unique_id", "cutoff"]):
        grp        = grp.sort_values("ds")
        actual_70d = float(grp["y"].sum())

        # Our model: use selected model (handle Ensemble)
        sel_model = sel_map.get(uid)
        if sel_model and sel_model.startswith("Ensemble:"):
            parts      = sel_model.split(":")[1].split("+")
            cols       = [p for p in parts if p in grp.columns and grp[p].notna().any()]
            model_yhat = grp[cols].mean(axis=1).sum() if cols else np.nan
        elif sel_model and sel_model in grp.columns and grp[sel_model].notna().any():
            model_yhat = float(grp[sel_model].sum())
        else:
            available  = [c for c in m_cols if c in grp.columns and grp[c].notna().any()]
            model_yhat = grp[available].mean(axis=1).sum() if available else np.nan

        rows.append({
            "unique_id":      uid,
            "cutoff":         cutoff,
            "bucket":         grp["bucket"].iloc[0],
            "history_length": grp["history_length"].iloc[0],
            "actual_70d":     actual_70d,
            "model_yhat":     model_yhat,
            "v1_yhat":        v1_forecast(index, uid, pd.Timestamp(cutoff)),
        })

    results = pd.DataFrame(rows)
    results["model_ae"]   = (results["actual_70d"] - results["model_yhat"]).abs()
    results["v1_ae"]      = (results["actual_70d"] - results["v1_yhat"]).abs()
    results["model_bias"] = results["model_yhat"] - results["actual_70d"]
    results["v1_bias"]    = results["v1_yhat"]    - results["actual_70d"]

    # Naive MAE denominator (1-step weekly, scaled to 70d)
    naive_mae = (
        cv_df.sort_values(["unique_id", "ds"])
        .groupby("unique_id")["y"]
        .apply(lambda s: float(np.abs(np.diff(s.values)).mean()) if len(s) > 1 else 1.0)
        .reset_index(name="naive_mae_weekly")
    )
    naive_mae["naive_mae_70d"] = (naive_mae["naive_mae_weekly"] * 10).clip(lower=1e-6)
    results = results.merge(naive_mae[["unique_id", "naive_mae_70d"]], on="unique_id", how="left")
    results["model_mase"] = results["model_ae"] / results["naive_mae_70d"]
    results["v1_mase"]    = results["v1_ae"]    / results["naive_mae_70d"]

    return results


# ── Summary ───────────────────────────────────────────────────────────────────

def summarise(results: pd.DataFrame):
    smooth = results[results["bucket"] == "smooth"].dropna(subset=["model_ae", "v1_ae"])

    print(f"\n{'='*62}")
    print(f"V1 vs Our Models — SMOOTH SKUs ONLY")
    print(f"{smooth['unique_id'].nunique()} SKUs × {smooth['cutoff'].nunique()} CV windows "
          f"= {len(smooth)} SKU-window observations")
    print(f"{'='*62}")

    def metrics(df, label):
        n      = len(df)
        mae_m  = df["model_ae"].mean()
        mae_v  = df["v1_ae"].mean()
        wape_m = df["model_ae"].sum() / max(df["actual_70d"].sum(), 1e-6)
        wape_v = df["v1_ae"].sum()    / max(df["actual_70d"].sum(), 1e-6)
        bias_m = df["model_bias"].mean()
        bias_v = df["v1_bias"].mean()
        wins_m = (df["model_ae"] < df["v1_ae"]).sum()
        wins_v = (df["v1_ae"] < df["model_ae"]).sum()
        mase_m = df["model_mase"].mean()
        mase_v = df["v1_mase"].mean()
        pct    = (mae_v - mae_m) / mae_v * 100 if mae_v > 0 else 0
        print(f"\n{label} (n={n})")
        print(f"  {'':22} {'Our Model':>12} {'V1 Formula':>12}  {'delta':>8}")
        print(f"  {'MAE (70d units)':22} {mae_m:>12.2f} {mae_v:>12.2f}  {pct:>+7.1f}%")
        print(f"  {'Mean MASE':22} {mase_m:>12.3f} {mase_v:>12.3f}")
        print(f"  {'WAPE':22} {wape_m:>12.3f} {wape_v:>12.3f}")
        print(f"  {'Bias (units)':22} {bias_m:>12.2f} {bias_v:>12.2f}")
        print(f"  {'SKU-window wins':22} {wins_m:>12} {wins_v:>12}  (ties={n-wins_m-wins_v})")

    metrics(smooth, "All smooth")

    print()
    for hl in ("full", "medium", "short"):
        sub = smooth[smooth["history_length"] == hl]
        if not sub.empty:
            metrics(sub, f"  history={hl}")

    print(f"\n  MAE by CV cutoff (smooth only):")
    print(f"  {'Cutoff':14} {'Our Model':>12} {'V1 Formula':>12} {'Delta':>8} {'Winner':>10}")
    for co, grp in smooth.groupby("cutoff"):
        mae_m  = grp["model_ae"].mean()
        mae_v  = grp["v1_ae"].mean()
        pct    = (mae_v - mae_m) / mae_v * 100 if mae_v > 0 else 0
        winner = "OurModel" if mae_m < mae_v else ("V1" if mae_v < mae_m else "Tie")
        print(f"    {str(pd.Timestamp(co).date()):13} {mae_m:>12.2f} {mae_v:>12.2f} {pct:>+7.1f}%  {winner:>10}")

    # Top V1 misses on smooth SKUs (useful for debugging V1 bias)
    worst_v1 = (
        smooth.assign(v1_lead=smooth["v1_ae"] - smooth["model_ae"])
        .nlargest(10, "v1_lead")
        [["unique_id", "cutoff", "actual_70d", "model_yhat", "v1_yhat",
          "model_ae", "v1_ae", "history_length"]]
    )
    print(f"\n  Top 10 SKU-windows where V1 hurts most (v1_ae - model_ae):")
    print(worst_v1.to_string(index=False))


def main():
    raw     = load_raw()
    index   = build_cumsum_index(raw)
    cv_df   = pd.read_parquet(CV_PATH)
    cv_df["ds"]     = pd.to_datetime(cv_df["ds"])
    cv_df["cutoff"] = pd.to_datetime(cv_df["cutoff"])

    results = score(cv_df, index)
    results.to_csv(OUT_PATH, index=False)
    print(f"\nDetailed results → {OUT_PATH}")
    summarise(results)


if __name__ == "__main__":
    main()
