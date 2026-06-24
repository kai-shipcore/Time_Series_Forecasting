#!/usr/bin/env python3
"""
EXPERIMENT: Deseasonalized (monthly index) vs Deseasonalized + Holiday flag.
Compared against V1 formula throughout.

Two approaches, same CV windows, same model set:
  Deseas only  : divide/multiply by monthly SEASONAL factors
                 (Nov=1.25, Dec=1.30 as before)
  Holiday flag : Nov/Dec monthly factors zeroed to 1.0; weeks inside the
                 holiday window (Nov-20 → Dec-14) use HOLIDAY_MULTIPLIER instead
                 (Black Friday week through first half of December)

Best model per SKU is selected independently for each approach.
Delete this file to remove the experiment.
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, WindowAverage, Naive, HistoricAverage

# ── Constants ─────────────────────────────────────────────────────────────────
N_CV_FULL   = 6
N_CV_MEDIUM = 3
TEST_WEEKS  = 10
TRIM_TAIL   = 3
FREQUENCY   = "W-MON"

HOLIDAY_START      = (11, 20)
HOLIDAY_END        = (12, 31)   # Black Friday week through end of December
HOLIDAY_MULTIPLIER = 1.26       # CV-optimised

# Monthly-only index (no holiday window)
SEASONAL_MONTHLY: dict[int, float] = {
    1: 0.75, 2: 0.80, 3: 0.90, 4: 0.95,
    5: 1.00, 6: 1.00, 7: 1.00, 8: 1.00, 9: 1.00,
    10: 1.10, 11: 1.25, 12: 1.30,
}

# Monthly index with Nov/Dec zeroed — holiday window takes over for those weeks.
# Nov 1-19 (pre-window) → 1.0 baseline; Dec 15-31 (post-window) → 1.0 baseline.
SEASONAL_HOLIDAY: dict[int, float] = {
    **SEASONAL_MONTHLY,
    11: 1.00,   # early Nov (before window) treated as baseline
    12: 1.00,   # late Dec (after window, Dec 15-31) treated as baseline
}

MODELS = [
    AutoETS(season_length=1, damped=True,  alias="AutoETS_D"),
    AutoETS(season_length=1, damped=False, alias="AutoETS_U"),
    AutoARIMA(season_length=1),
    WindowAverage(window_size=8),
    Naive(),
    HistoricAverage(),
]
Q4_CUTOFFS = [pd.Timestamp("2025-10-27"), pd.Timestamp("2026-01-05")]


# ── Factor helpers ─────────────────────────────────────────────────────────────
def _is_holiday(ds: pd.Series) -> pd.Series:
    h_m0, h_d0 = HOLIDAY_START
    h_m1, h_d1 = HOLIDAY_END
    m, d = ds.dt.month, ds.dt.day
    return (((m == h_m0) & (d >= h_d0)) |
            ((m >  h_m0) & (m <  h_m1)) |
            ((m == h_m1) & (d <= h_d1)))


def _factors(ds: pd.Series, use_holiday: bool) -> pd.Series:
    index = SEASONAL_HOLIDAY if use_holiday else SEASONAL_MONTHLY
    f = ds.dt.month.map(index)
    if use_holiday:
        f = f.where(~_is_holiday(ds), HOLIDAY_MULTIPLIER)
    return f


def deseasonalize(df: pd.DataFrame, use_holiday: bool) -> pd.DataFrame:
    df = df.copy()
    df["y"] = df["y"] / _factors(df["ds"], use_holiday)
    return df


def reseasonalize(df: pd.DataFrame, use_holiday: bool) -> pd.DataFrame:
    df = df.copy()
    f    = _factors(df["ds"], use_holiday)
    meta = {"unique_id", "ds", "cutoff", "y", "bucket", "history_length"}
    for col in ["y"] + [c for c in df.columns if c not in meta]:
        if col in df.columns:
            df[col] = df[col] * f
    return df


# ── CV runner ──────────────────────────────────────────────────────────────────
def run_cv(train_raw: pd.DataFrame, smooth_prof: pd.DataFrame,
           use_holiday: bool, label: str) -> pd.DataFrame:
    parts = []
    for hist_len, n_windows in [("full", N_CV_FULL), ("medium", N_CV_MEDIUM)]:
        uids = smooth_prof.loc[smooth_prof["history_length"] == hist_len, "unique_id"].tolist()
        df_g = deseasonalize(train_raw, use_holiday)
        df_g = df_g[df_g["unique_id"].isin(uids)][["unique_id", "ds", "y"]]

        min_len   = n_windows * TEST_WEEKS + 1
        too_short = df_g.groupby("unique_id")["ds"].count()
        too_short = too_short[too_short < min_len].index.tolist()
        if too_short:
            df_g = df_g[~df_g["unique_id"].isin(too_short)]

        t0 = time.time()
        sf = StatsForecast(models=MODELS, freq=FREQUENCY, n_jobs=-1)
        cv = sf.cross_validation(df=df_g, h=TEST_WEEKS, n_windows=n_windows,
                                 step_size=TEST_WEEKS)
        cv["history_length"] = hist_len
        parts.append(cv)
        print(f"  [{label}] {hist_len}: {len(uids)} SKUs → {len(cv):,} rows  ({time.time()-t0:.1f}s)")

    cv_out           = pd.concat(parts, ignore_index=True)
    cv_out["ds"]     = pd.to_datetime(cv_out["ds"])
    cv_out["cutoff"] = pd.to_datetime(cv_out["cutoff"])
    return reseasonalize(cv_out, use_holiday)


# ── Model selection ────────────────────────────────────────────────────────────
def select_models(cv_df: pd.DataFrame, train_raw: pd.DataFrame) -> dict:
    naive_mae = (
        train_raw.sort_values(["unique_id", "ds"])
        .groupby("unique_id")["y"]
        .apply(lambda s: float(np.abs(np.diff(s.values)).mean()) if len(s) > 1 else 1.0)
        .reset_index(name="naive_mae")
    )
    naive_mae["naive_mae"] = naive_mae["naive_mae"].clip(lower=1e-6)

    meta   = {"unique_id", "ds", "cutoff", "y", "bucket", "history_length"}
    m_cols = [c for c in cv_df.columns if c not in meta]

    long = (
        cv_df.melt(id_vars=["unique_id","ds","y"], value_vars=m_cols,
                   var_name="model", value_name="yhat")
        .dropna(subset=["yhat"])
        .merge(naive_mae, on="unique_id", how="left")
    )
    long["mase"] = (long["y"] - long["yhat"]).abs() / long["naive_mae"]

    return (
        long.groupby(["unique_id","model"])["mase"].mean()
        .reset_index()
        .sort_values(["unique_id","mase"])
        .groupby("unique_id").first()["model"]
        .to_dict()
    )


# ── Score ──────────────────────────────────────────────────────────────────────
def score(cv_df: pd.DataFrame, sel_map: dict, cmp_smooth: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (uid, cutoff), grp in cv_df.groupby(["unique_id","cutoff"]):
        grp    = grp.sort_values("ds")
        actual = float(grp["y"].sum())
        m      = sel_map.get(uid, "")
        if m in grp.columns and grp[m].notna().any():
            yhat = float(grp[m].sum())
        else:
            avail = [c for c in grp.columns
                     if c not in {"unique_id","ds","cutoff","y","bucket","history_length"}
                     and grp[c].notna().any()]
            yhat  = float(grp[avail[0]].sum()) if avail else np.nan

        v1_row = cmp_smooth[(cmp_smooth["unique_id"]==uid)&(cmp_smooth["cutoff"]==cutoff)]
        v1     = float(v1_row["v1_yhat"].iloc[0]) if not v1_row.empty else np.nan

        rows.append({
            "unique_id": uid, "cutoff": cutoff,
            "history_length": grp["history_length"].iloc[0],
            "actual": actual, "yhat": yhat, "v1": v1,
            "selected_model": m,
        })

    r = pd.DataFrame(rows)
    r["ae"]    = (r["actual"] - r["yhat"]).abs()
    r["ae_v1"] = (r["actual"] - r["v1"]).abs()
    r["bias"]  = r["yhat"] - r["actual"]
    r["bias_v1"] = r["v1"] - r["actual"]
    return r


# ── Print helpers ──────────────────────────────────────────────────────────────
def print_comparison(monthly_r: pd.DataFrame, holiday_r: pd.DataFrame, label: str):
    both = monthly_r.merge(
        holiday_r[["unique_id","cutoff","yhat","ae","bias","selected_model"]],
        on=["unique_id","cutoff"], suffixes=("_m","_h")
    )
    n = len(both)
    mae_m  = both["ae_m"].mean()
    mae_h  = both["ae_h"].mean()
    mae_v  = both["ae_v1"].mean()
    bias_m = both["bias_m"].mean()
    bias_h = both["bias_h"].mean()
    bias_v = both["bias_v1"].mean()
    wape_m = both["ae_m"].sum()  / max(both["actual"].sum(), 1e-6)
    wape_h = both["ae_h"].sum()  / max(both["actual"].sum(), 1e-6)
    wape_v = both["ae_v1"].sum() / max(both["actual"].sum(), 1e-6)
    wins_hm = (both["ae_h"] < both["ae_m"]).sum()
    wins_mh = (both["ae_m"] < both["ae_h"]).sum()
    wins_mv = (both["ae_m"] < both["ae_v1"]).sum()
    wins_hv = (both["ae_h"] < both["ae_v1"]).sum()
    d_hm = (mae_m - mae_h) / mae_m * 100
    d_mv = (mae_v - mae_m) / mae_v * 100
    d_hv = (mae_v - mae_h) / mae_v * 100

    print(f"\n{label}  (n={n})")
    print(f"  {'':26} {'V1':>10} {'Deseas':>12} {'+ Holiday':>12}")
    print(f"  {'MAE (70d units)':26} {mae_v:>10.2f} {mae_m:>12.2f} {mae_h:>12.2f}")
    print(f"  {'WAPE':26} {wape_v:>10.3f} {wape_m:>12.3f} {wape_h:>12.3f}")
    print(f"  {'Bias (units)':26} {bias_v:>10.2f} {bias_m:>12.2f} {bias_h:>12.2f}")
    print(f"  {'Wins vs V1':26} {'':>10} {wins_mv:>12} {wins_hv:>12}")
    print(f"  {'Holiday wins vs Deseas':26} {'':>10} {'':>12} {wins_hm:>12}  (deseas wins {wins_mh})")
    print(f"  Deseas vs V1: {d_mv:+.1f}%  |  Holiday vs V1: {d_hv:+.1f}%  |  Holiday vs Deseas: {d_hm:+.1f}%")


def main():
    print("Loading data...")
    hist = pd.read_parquet(ROOT / "data/processed/sales_clean.parquet")
    prof = pd.read_csv(ROOT / "data/processed/sku_profiles.csv")
    cmp  = pd.read_csv(ROOT / "outputs/reports/v1_comparison.csv")

    hist["ds"]          = pd.to_datetime(hist["ds"])
    prof["train_start"] = pd.to_datetime(prof["train_start"])
    cmp["cutoff"]       = pd.to_datetime(cmp["cutoff"])
    cmp_smooth          = cmp[cmp["bucket"] == "smooth"].copy()

    smooth_prof = prof[(prof["bucket"]=="smooth") &
                       (prof["history_length"].isin(["full","medium"]))].copy()
    smooth_uids = set(smooth_prof["unique_id"])

    all_weeks  = sorted(hist["ds"].unique())
    trimmed    = all_weeks[:-TRIM_TAIL] if TRIM_TAIL else all_weeks
    test_start = pd.Timestamp(trimmed[-TEST_WEEKS])

    hist_smooth = hist[hist["unique_id"].isin(smooth_uids)].merge(
        smooth_prof[["unique_id","train_start","history_length"]], on="unique_id"
    )
    train_raw = (
        hist_smooth[hist_smooth["ds"].isin(trimmed) & (hist_smooth["ds"] < test_start)]
        .pipe(lambda d: d[d["ds"] >= d["train_start"]])
        [["unique_id","ds","y","history_length"]].copy()
    )
    print(f"  {len(smooth_uids)} smooth CV SKUs | train ends {train_raw['ds'].max().date()}")

    # Run both CV variants
    print("\nRunning CV — monthly deseasonalize (no holiday flag):")
    cv_monthly = run_cv(train_raw, smooth_prof, use_holiday=False, label="monthly")

    print(f"\nRunning CV — deseasonalize + holiday flag (Nov 20–Dec 14 window × {HOLIDAY_MULTIPLIER}):")
    cv_holiday = run_cv(train_raw, smooth_prof, use_holiday=True,  label="holiday")

    # Model selection
    sel_monthly = select_models(cv_monthly, train_raw)
    sel_holiday = select_models(cv_holiday, train_raw)

    print(f"\nModel selection — monthly:  {pd.Series(sel_monthly).value_counts().to_dict()}")
    print(f"Model selection — holiday:  {pd.Series(sel_holiday).value_counts().to_dict()}")

    # Score both
    r_monthly = score(cv_monthly, sel_monthly, cmp_smooth)
    r_holiday = score(cv_holiday, sel_holiday, cmp_smooth)

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "="*68)
    print(f"COMPARISON: Deseas (monthly) vs Deseas + Holiday flag vs V1")
    print(f"Holiday window: {HOLIDAY_START[0]}/{HOLIDAY_START[1]}–{HOLIDAY_END[0]}/{HOLIDAY_END[1]}"
          f"  |  multiplier={HOLIDAY_MULTIPLIER}  |  Nov/Dec monthly→1.0")
    print("="*68)

    print_comparison(r_monthly, r_holiday, "ALL smooth CV windows")

    for hl in ("full", "medium"):
        rm = r_monthly[r_monthly["history_length"]==hl]
        rh = r_holiday[r_holiday["history_length"]==hl]
        if not rm.empty:
            print_comparison(rm, rh, f"  history={hl}")

    print(f"\n--- Per-cutoff ---")
    print(f"  {'Cutoff':14} {'V1 MAE':>9} {'Deseas MAE':>11} {'Holiday MAE':>12}"
          f" {'Best':>9} {'V1 bias':>9} {'D bias':>8} {'H bias':>8}")
    for cut in sorted(r_monthly["cutoff"].unique()):
        rm = r_monthly[r_monthly["cutoff"]==cut]
        rh = r_holiday[r_holiday["cutoff"]==cut]
        mae_v  = rm["ae_v1"].mean()
        mae_m  = rm["ae"].mean()
        mae_h  = rh["ae"].mean()
        bias_v = rm["bias_v1"].mean()
        bias_m = rm["bias"].mean()
        bias_h = rh["bias"].mean()
        best   = min([("V1",mae_v),("Deseas",mae_m),("Holiday",mae_h)], key=lambda x:x[1])[0]
        tag    = " ◄ Q4" if pd.Timestamp(cut) in Q4_CUTOFFS else ""
        print(f"  {str(pd.Timestamp(cut).date()):14} {mae_v:>9.2f} {mae_m:>11.2f}"
              f" {mae_h:>12.2f} {best:>9} {bias_v:>9.2f} {bias_m:>8.2f} {bias_h:>8.2f}{tag}")

    # SKU-level
    sku_m = r_monthly.groupby("unique_id")["ae"].sum().rename("ae_m")
    sku_h = r_holiday.groupby("unique_id")["ae"].sum().rename("ae_h")
    sku_v = r_monthly.groupby("unique_id")["ae_v1"].sum().rename("ae_v")
    sku   = pd.concat([sku_m, sku_h, sku_v], axis=1).reset_index()
    print(f"\n--- SKU-level (sum of AE across all folds) ---")
    print(f"  Holiday beats Deseas : {(sku['ae_h'] < sku['ae_m']).sum()} / {len(sku)} SKUs")
    print(f"  Deseas beats V1      : {(sku['ae_m'] < sku['ae_v']).sum()} / {len(sku)} SKUs")
    print(f"  Holiday beats V1     : {(sku['ae_h'] < sku['ae_v']).sum()} / {len(sku)} SKUs")


if __name__ == "__main__":
    main()
