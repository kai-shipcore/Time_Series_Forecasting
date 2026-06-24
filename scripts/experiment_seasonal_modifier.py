#!/usr/bin/env python3
"""
EXPERIMENT: Apply V1's proportional seasonal modifier to our model predictions.

Hypothesis: our models capture current demand level well but may under/over-shoot
when the forecast window straddles a seasonal transition (e.g. post-Q4 Jan drop,
Q4 ramp). Applying the same scalar modifier V1 uses might close that gap.

Reads only:
  outputs/reports/cv_results.parquet
  outputs/reports/selection.csv
  outputs/reports/v1_comparison.csv      (for V1 baseline + actuals)

Writes nothing to the pipeline. Delete this file to remove the experiment.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import calendar
from datetime import timedelta

import numpy as np
import pandas as pd

# ── Seasonal modifier (identical to compare_v1.py) ────────────────────────────
SEASONAL = {
    1: 0.75, 2: 0.80, 3: 0.90, 4: 0.95,
    5: 1.00, 6: 1.00, 7: 1.00, 8: 1.00, 9: 1.00,
    10: 1.10, 11: 1.25, 12: 1.30,
}
HORIZON_DAYS = 70


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


def modifier_for_cutoff(cutoff: pd.Timestamp) -> float:
    start = cutoff + timedelta(days=1)
    end   = cutoff + timedelta(days=HORIZON_DAYS)
    return proportional_seasonal_modifier(start, end)


# ── Load data ─────────────────────────────────────────────────────────────────
cv  = pd.read_parquet(ROOT / "outputs/reports/cv_results.parquet")
sel = pd.read_csv(ROOT / "outputs/reports/selection.csv")
cmp = pd.read_csv(ROOT / "outputs/reports/v1_comparison.csv")

cv["ds"]      = pd.to_datetime(cv["ds"])
cv["cutoff"]  = pd.to_datetime(cv["cutoff"])
cmp["cutoff"] = pd.to_datetime(cmp["cutoff"])

sel_map = sel.set_index("unique_id")["model"].to_dict()

# Focus on smooth bucket only (where comparison is meaningful)
smooth_uids = set(sel[sel["bucket"] == "smooth"]["unique_id"])
cv_smooth   = cv[cv["unique_id"].isin(smooth_uids)].copy()
cmp_smooth  = cmp[cmp["bucket"] == "smooth"].copy()


# ── Reconstruct our model's 70d total per SKU-cutoff ─────────────────────────
def model_total(uid, cutoff, grp):
    m = sel_map.get(uid, "")
    if m.startswith("Ensemble:"):
        parts  = m.split(":")[1].split("+")
        cols   = [p for p in parts if p in grp.columns and grp[p].notna().any()]
        return float(grp[cols].mean(axis=1).sum()) if cols else np.nan
    elif m in grp.columns and grp[m].notna().any():
        return float(grp[m].sum())
    else:
        avail = [c for c in grp.columns
                 if c not in {"unique_id","ds","cutoff","y","bucket","history_length"}
                 and grp[c].notna().any()]
        return float(grp[avail[0]].sum()) if avail else np.nan


rows = []
for (uid, cutoff), grp in cv_smooth.groupby(["unique_id", "cutoff"]):
    grp   = grp.sort_values("ds")
    raw   = model_total(uid, cutoff, grp)
    mod   = modifier_for_cutoff(pd.Timestamp(cutoff))
    adj   = raw * mod if pd.notna(raw) else np.nan
    actual= float(grp["y"].sum())

    # V1 from comparison file
    v1_row = cmp_smooth[(cmp_smooth["unique_id"] == uid) &
                        (cmp_smooth["cutoff"] == cutoff)]
    v1     = float(v1_row["v1_yhat"].iloc[0]) if not v1_row.empty else np.nan

    rows.append({
        "unique_id":       uid,
        "cutoff":          cutoff,
        "history_length":  grp["history_length"].iloc[0],
        "actual_70d":      actual,
        "model_raw":       raw,
        "model_adjusted":  adj,
        "v1":              v1,
        "modifier":        mod,
    })

results = pd.DataFrame(rows)

for col, yhat in [("model_raw", "model_raw"), ("model_adjusted", "model_adjusted"), ("v1", "v1")]:
    results[f"ae_{col}"] = (results["actual_70d"] - results[yhat]).abs()

results["raw_beats_v1"]  = results["ae_model_raw"]      < results["ae_v1"]
results["adj_beats_v1"]  = results["ae_model_adjusted"] < results["ae_v1"]
results["adj_beats_raw"] = results["ae_model_adjusted"] < results["ae_model_raw"]


# ── Summary ───────────────────────────────────────────────────────────────────
def print_block(df, label):
    n      = len(df)
    mae_r  = df["ae_model_raw"].mean()
    mae_a  = df["ae_model_adjusted"].mean()
    mae_v  = df["ae_v1"].mean()
    bias_r = (df["model_raw"]      - df["actual_70d"]).mean()
    bias_a = (df["model_adjusted"] - df["actual_70d"]).mean()
    bias_v = (df["v1"]             - df["actual_70d"]).mean()
    wape_r = df["ae_model_raw"].sum()      / max(df["actual_70d"].sum(), 1e-6)
    wape_a = df["ae_model_adjusted"].sum() / max(df["actual_70d"].sum(), 1e-6)
    wape_v = df["ae_v1"].sum()             / max(df["actual_70d"].sum(), 1e-6)
    w_rb   = df["raw_beats_v1"].sum()
    w_ab   = df["adj_beats_v1"].sum()
    w_ar   = df["adj_beats_raw"].sum()

    print(f"\n{label}  (n={n})")
    print(f"  {'':26} {'Raw model':>12} {'Adj model':>12} {'V1':>12}")
    print(f"  {'MAE (70d units)':26} {mae_r:>12.2f} {mae_a:>12.2f} {mae_v:>12.2f}")
    print(f"  {'WAPE':26} {wape_r:>12.3f} {wape_a:>12.3f} {wape_v:>12.3f}")
    print(f"  {'Bias (units)':26} {bias_r:>12.2f} {bias_a:>12.2f} {bias_v:>12.2f}")
    print(f"  {'Wins vs V1':26} {w_rb:>12} {w_ab:>12}")
    print(f"  {'Adj wins vs Raw':26} {'':>12} {w_ar:>12}")
    d_ra = (mae_r - mae_a) / mae_r * 100
    d_rv = (mae_r - mae_v) / mae_r * 100
    d_av = (mae_a - mae_v) / mae_a * 100
    print(f"  Adj vs Raw:  {d_ra:+.1f}%  |  Raw vs V1: {d_rv:+.1f}%  |  Adj vs V1: {d_av:+.1f}%")


print("=" * 70)
print("EXPERIMENT: V1 Seasonal Modifier Applied to Our Model — Smooth SKUs")
print("=" * 70)

print_block(results, "ALL smooth CV windows")

print("\n--- by history length ---")
for hl in ("full", "medium"):
    sub = results[results["history_length"] == hl]
    if not sub.empty:
        print_block(sub, f"  history={hl}")

print("\n--- by CV cutoff ---")
print(f"  {'Cutoff':14} {'Modifier':>10} {'Raw MAE':>10} {'Adj MAE':>10} {'V1 MAE':>10} {'BestModel':>12}")
for cut, grp in results.groupby("cutoff"):
    mod    = grp["modifier"].mean()
    mae_r  = grp["ae_model_raw"].mean()
    mae_a  = grp["ae_model_adjusted"].mean()
    mae_v  = grp["ae_v1"].mean()
    best   = min([("Raw", mae_r), ("Adj", mae_a), ("V1", mae_v)], key=lambda x: x[1])
    print(f"  {str(pd.Timestamp(cut).date()):14} {mod:>10.3f} {mae_r:>10.2f} {mae_a:>10.2f} {mae_v:>10.2f} {best[0]:>12}")

# Per-SKU: does adjustment help or hurt?
sku_agg = results.groupby("unique_id").agg(
    raw_total_ae  =("ae_model_raw",      "sum"),
    adj_total_ae  =("ae_model_adjusted", "sum"),
    v1_total_ae   =("ae_v1",             "sum"),
).reset_index()
sku_agg["adj_helps"]  = sku_agg["adj_total_ae"] < sku_agg["raw_total_ae"]
sku_agg["adj_vs_v1"]  = sku_agg["adj_total_ae"] < sku_agg["v1_total_ae"]
sku_agg["raw_vs_v1"]  = sku_agg["raw_total_ae"] < sku_agg["v1_total_ae"]
sku_agg["delta"]      = sku_agg["raw_total_ae"] - sku_agg["adj_total_ae"]  # positive = adj better

print(f"\n--- SKU-level (aggregate across all folds) ---")
print(f"  Adj helps vs Raw : {sku_agg['adj_helps'].sum()} / {len(sku_agg)} SKUs")
print(f"  Adj beats V1     : {sku_agg['adj_vs_v1'].sum()} / {len(sku_agg)} SKUs")
print(f"  Raw beats V1     : {sku_agg['raw_vs_v1'].sum()} / {len(sku_agg)} SKUs")

print(f"\n  Top 10 SKUs where modifier helps most (delta = raw_ae - adj_ae, +ve = adj better):")
top_help = sku_agg.nlargest(10, "delta")[["unique_id","raw_total_ae","adj_total_ae","v1_total_ae","delta"]]
print(top_help.to_string(index=False))

print(f"\n  Top 10 SKUs where modifier hurts most:")
top_hurt = sku_agg.nsmallest(10, "delta")[["unique_id","raw_total_ae","adj_total_ae","v1_total_ae","delta"]]
print(top_hurt.to_string(index=False))
