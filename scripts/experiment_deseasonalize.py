#!/usr/bin/env python3
"""
EXPERIMENT: Deseasonalize → fit → reseasonalize for smooth SKUs.

Rather than multiplying the model output by a seasonal modifier (which
double-counts wherever the model already learned the pattern), this approach:

  1. Divide each week's demand by its monthly seasonal factor before fitting.
     The model sees a "flat" series and only learns level + trend.
  2. Fit models on the deseasonalized series using the same CV setup.
  3. Multiply each forecast week's prediction back by that week's seasonal factor.
     All seasonality now comes from the known index; the model contributes none.

Comparison: raw model (original pipeline) vs deseas+reseas model vs V1 formula.

Reads only (no writes to the main pipeline):
  data/processed/sales_clean.parquet
  data/processed/sku_profiles.csv
  outputs/reports/cv_results.parquet    (raw model baseline + cutoffs)
  outputs/reports/v1_comparison.csv     (V1 baseline)
  outputs/reports/selection.csv         (which model won per SKU)

Delete this file to remove the experiment entirely.
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA, AutoETS, WindowAverage, Naive, HistoricAverage,
)

SEASONAL = {
    1: 0.75, 2: 0.80, 3: 0.90, 4: 0.95,
    5: 1.00, 6: 1.00, 7: 1.00, 8: 1.00, 9: 1.00,
    10: 1.10, 11: 1.25, 12: 1.30,
}

N_CV_FULL   = 6
N_CV_MEDIUM = 3
TEST_WEEKS  = 10
TRIM_TAIL   = 3
FREQUENCY   = "W-MON"

MODELS = [
    AutoARIMA(season_length=1),
    AutoETS(season_length=1, damped=True),
    WindowAverage(window_size=8),
    Naive(),
    HistoricAverage(),
]
MODEL_NAMES = [type(m).__name__ for m in MODELS]


def block(comp, label):
    n      = len(comp)
    mae_r  = comp["ae_raw"].mean()
    mae_d  = comp["ae_deseas"].mean()
    mae_v  = comp["ae_v1"].mean()
    bias_r = (comp["raw_yhat"]    - comp["actual_70d"]).mean()
    bias_d = (comp["deseas_yhat"] - comp["actual_70d"]).mean()
    bias_v = (comp["v1_yhat"]     - comp["actual_70d"]).mean()
    wape_r = comp["ae_raw"].sum()    / max(comp["actual_70d"].sum(), 1e-6)
    wape_d = comp["ae_deseas"].sum() / max(comp["actual_70d"].sum(), 1e-6)
    wape_v = comp["ae_v1"].sum()     / max(comp["actual_70d"].sum(), 1e-6)
    w_dr   = (comp["ae_deseas"] < comp["ae_raw"]).sum()
    w_dv   = (comp["ae_deseas"] < comp["ae_v1"]).sum()
    w_rv   = (comp["ae_raw"]    < comp["ae_v1"]).sum()
    d_dr   = (mae_r - mae_d) / mae_r * 100
    d_dv   = (mae_v - mae_d) / mae_v * 100
    d_rv   = (mae_v - mae_r) / mae_v * 100
    print(f"\n{label}  (n={n})")
    print(f"  {'':24} {'Raw model':>12} {'Deseas+Reseas':>14} {'V1':>12}")
    print(f"  {'MAE (70d units)':24} {mae_r:>12.2f} {mae_d:>14.2f} {mae_v:>12.2f}")
    print(f"  {'WAPE':24} {wape_r:>12.3f} {wape_d:>14.3f} {wape_v:>12.3f}")
    print(f"  {'Bias (units)':24} {bias_r:>12.2f} {bias_d:>14.2f} {bias_v:>12.2f}")
    print(f"  {'Wins vs Raw':24} {'':>12} {w_dr:>14}")
    print(f"  {'Wins vs V1':24} {w_rv:>12} {w_dv:>14}")
    print(f"  Deseas vs Raw: {d_dr:+.1f}%  |  Deseas vs V1: {d_dv:+.1f}%  |  Raw vs V1: {d_rv:+.1f}%")


def main():
    print("Loading data...")
    hist = pd.read_parquet(ROOT / "data/processed/sales_clean.parquet")
    prof = pd.read_csv(ROOT / "data/processed/sku_profiles.csv")
    sel  = pd.read_csv(ROOT / "outputs/reports/selection.csv")
    cv0  = pd.read_parquet(ROOT / "outputs/reports/cv_results.parquet")
    cmp  = pd.read_csv(ROOT / "outputs/reports/v1_comparison.csv")

    hist["ds"]          = pd.to_datetime(hist["ds"])
    prof["train_start"] = pd.to_datetime(prof["train_start"])
    cv0["ds"]           = pd.to_datetime(cv0["ds"])
    cv0["cutoff"]       = pd.to_datetime(cv0["cutoff"])
    cmp["cutoff"]       = pd.to_datetime(cmp["cutoff"])

    smooth_prof = prof[(prof["bucket"] == "smooth") &
                       (prof["history_length"].isin(["full", "medium"]))].copy()
    smooth_uids = set(smooth_prof["unique_id"])
    print(f"  {len(smooth_uids)} smooth CV SKUs ({smooth_prof['history_length'].value_counts().to_dict()})")

    hist_smooth = hist[hist["unique_id"].isin(smooth_uids)].copy()
    hist_smooth = hist_smooth.merge(
        smooth_prof[["unique_id", "train_start", "history_length"]], on="unique_id"
    )

    # Trim tail + split
    all_weeks  = sorted(hist["ds"].unique())
    trimmed    = all_weeks[:-TRIM_TAIL] if TRIM_TAIL else all_weeks
    test_start = pd.Timestamp(trimmed[-TEST_WEEKS])

    train_raw = hist_smooth[
        hist_smooth["ds"].isin(trimmed) & (hist_smooth["ds"] < test_start)
    ].copy()
    train_raw = train_raw[train_raw["ds"] >= train_raw["train_start"]]
    train_raw = train_raw[["unique_id", "ds", "y", "history_length"]].copy()
    print(f"  Train ends: {train_raw['ds'].max().date()}  |  Test starts: {test_start.date()}")

    # Deseasonalize
    train_raw["factor"] = train_raw["ds"].dt.month.map(SEASONAL)
    train_deseas        = train_raw.copy()
    train_deseas["y"]   = train_deseas["y"] / train_deseas["factor"]

    # CV on deseasonalized data
    cv_parts = []
    for hist_len, n_windows in [("full", N_CV_FULL), ("medium", N_CV_MEDIUM)]:
        uids = smooth_prof.loc[smooth_prof["history_length"] == hist_len, "unique_id"].tolist()
        df_g = train_deseas[train_deseas["unique_id"].isin(uids)][["unique_id", "ds", "y"]]

        min_len   = n_windows * TEST_WEEKS + 1
        lengths   = df_g.groupby("unique_id")["ds"].count()
        too_short = lengths[lengths < min_len].index.tolist()
        if too_short:
            print(f"  WARNING: {len(too_short)} {hist_len} series too short — skipped")
            df_g = df_g[~df_g["unique_id"].isin(too_short)]

        print(f"  CV {hist_len}: {len(uids)} SKUs | n_windows={n_windows} | {MODEL_NAMES}")
        t0 = time.time()
        sf = StatsForecast(models=MODELS, freq=FREQUENCY, n_jobs=-1)
        cv = sf.cross_validation(df=df_g, h=TEST_WEEKS, n_windows=n_windows,
                                 step_size=TEST_WEEKS)
        cv["history_length"] = hist_len
        cv_parts.append(cv)
        print(f"    → {len(cv):,} rows in {time.time()-t0:.1f}s")

    cv_deseas           = pd.concat(cv_parts, ignore_index=True)
    cv_deseas["ds"]     = pd.to_datetime(cv_deseas["ds"])
    cv_deseas["cutoff"] = pd.to_datetime(cv_deseas["cutoff"])

    # Reseasonalize forecasts and actuals
    cv_deseas["reseas_factor"] = cv_deseas["ds"].dt.month.map(SEASONAL)
    for m in MODEL_NAMES:
        if m in cv_deseas.columns:
            cv_deseas[f"{m}_reseas"] = cv_deseas[m] * cv_deseas["reseas_factor"]
    cv_deseas["y_true"] = cv_deseas["y"] * cv_deseas["reseas_factor"]

    # Model selection on reseasonalized predictions
    reseas_cols = [f"{m}_reseas" for m in MODEL_NAMES if f"{m}_reseas" in cv_deseas.columns]

    naive_mae = (
        train_raw.sort_values(["unique_id", "ds"])
        .groupby("unique_id")["y"]
        .apply(lambda s: float(np.abs(np.diff(s.values)).mean()) if len(s) > 1 else 1.0)
        .reset_index(name="naive_mae")
    )
    naive_mae["naive_mae"] = naive_mae["naive_mae"].clip(lower=1e-6)

    long = cv_deseas.melt(
        id_vars=["unique_id", "ds", "y_true"],
        value_vars=reseas_cols,
        var_name="model_col", value_name="yhat"
    ).dropna(subset=["yhat"])
    long["model"] = long["model_col"].str.replace("_reseas", "")
    long = long.merge(naive_mae, on="unique_id", how="left")
    long["ae"]   = (long["y_true"] - long["yhat"]).abs()
    long["mase"] = long["ae"] / long["naive_mae"]

    metrics_deseas = (
        long.groupby(["unique_id", "model"])
        .agg(MAE=("ae","mean"), MASE=("mase","mean"))
        .reset_index()
    )
    best_deseas = (
        metrics_deseas.sort_values(["unique_id","MASE","MAE"])
        .groupby("unique_id").first().reset_index()
        .rename(columns={"model":"best_deseas_model","MASE":"deseas_MASE","MAE":"deseas_MAE"})
    )
    print(f"\nDeseasonalized model selection:")
    print(best_deseas.groupby("best_deseas_model").size().to_string())

    # Raw model 70d totals (from original CV)
    sel_map = sel.set_index("unique_id")["model"].to_dict()
    raw_rows = []
    for (uid, cutoff), grp in cv0[cv0["unique_id"].isin(smooth_uids)].groupby(["unique_id","cutoff"]):
        grp = grp.sort_values("ds")
        m   = sel_map.get(uid, "")
        if m.startswith("Ensemble:"):
            parts = m.split(":")[1].split("+")
            cols  = [p for p in parts if p in grp.columns and grp[p].notna().any()]
            yhat  = grp[cols].mean(axis=1).sum() if cols else np.nan
        elif m in grp.columns and grp[m].notna().any():
            yhat = float(grp[m].sum())
        else:
            avail = [c for c in grp.columns
                     if c not in {"unique_id","ds","cutoff","y","bucket","history_length"}
                     and grp[c].notna().any()]
            yhat  = float(grp[avail[0]].sum()) if avail else np.nan
        raw_rows.append({"unique_id": uid, "cutoff": cutoff,
                         "actual_70d": float(grp["y"].sum()), "raw_yhat": yhat,
                         "history_length": grp["history_length"].iloc[0]})
    raw_df = pd.DataFrame(raw_rows)

    # Deseas best-model 70d totals
    best_map = best_deseas.set_index("unique_id")["best_deseas_model"].to_dict()
    deseas_rows = []
    for (uid, cutoff), grp in cv_deseas.groupby(["unique_id","cutoff"]):
        grp  = grp.sort_values("ds")
        bm   = best_map.get(uid, MODEL_NAMES[0])
        col  = f"{bm}_reseas"
        yhat = float(grp[col].sum()) if col in grp.columns and grp[col].notna().any() else np.nan
        deseas_rows.append({"unique_id": uid, "cutoff": pd.Timestamp(cutoff),
                            "deseas_yhat": yhat, "deseas_model": bm})
    deseas_df = pd.DataFrame(deseas_rows)

    v1_df          = cmp[cmp["bucket"]=="smooth"][["unique_id","cutoff","v1_yhat"]].copy()
    v1_df["cutoff"] = pd.to_datetime(v1_df["cutoff"])

    comp = raw_df.merge(deseas_df, on=["unique_id","cutoff"], how="inner")
    comp = comp.merge(v1_df,       on=["unique_id","cutoff"], how="left")
    comp["ae_raw"]    = (comp["actual_70d"] - comp["raw_yhat"]).abs()
    comp["ae_deseas"] = (comp["actual_70d"] - comp["deseas_yhat"]).abs()
    comp["ae_v1"]     = (comp["actual_70d"] - comp["v1_yhat"]).abs()

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("EXPERIMENT: Deseasonalize → Fit → Reseasonalize  (Smooth CV SKUs)")
    print("="*70)

    block(comp, "ALL smooth CV windows")

    print("\n--- by history length ---")
    for hl in ("full", "medium"):
        sub = comp[comp["history_length"] == hl]
        if not sub.empty:
            block(sub, f"  history={hl}")

    print("\n--- by CV cutoff ---")
    print(f"  {'Cutoff':14} {'Modifier':>9} {'Raw MAE':>10} {'Deseas MAE':>11} {'V1 MAE':>9} {'Best':>12}")
    for cut, grp in comp.groupby("cutoff"):
        mod   = SEASONAL[pd.Timestamp(cut).month]
        mae_r = grp["ae_raw"].mean()
        mae_d = grp["ae_deseas"].mean()
        mae_v = grp["ae_v1"].mean()
        best  = min([("Raw",mae_r),("Deseas",mae_d),("V1",mae_v)], key=lambda x: x[1])
        print(f"  {str(pd.Timestamp(cut).date()):14} {mod:>9.3f} {mae_r:>10.2f} {mae_d:>11.2f} {mae_v:>9.2f} {best[0]:>12}")

    sku_agg = comp.groupby("unique_id").agg(
        raw_total_ae   =("ae_raw",    "sum"),
        deseas_total_ae=("ae_deseas", "sum"),
        v1_total_ae    =("ae_v1",     "sum"),
    ).reset_index()
    sku_agg["deseas_beats_raw"] = sku_agg["deseas_total_ae"] < sku_agg["raw_total_ae"]
    sku_agg["deseas_beats_v1"]  = sku_agg["deseas_total_ae"] < sku_agg["v1_total_ae"]
    sku_agg["raw_beats_v1"]     = sku_agg["raw_total_ae"]    < sku_agg["v1_total_ae"]
    sku_agg["delta"]            = sku_agg["raw_total_ae"]    - sku_agg["deseas_total_ae"]

    print(f"\n--- SKU-level (aggregate across all folds) ---")
    print(f"  Deseas beats Raw  : {sku_agg['deseas_beats_raw'].sum()} / {len(sku_agg)} SKUs")
    print(f"  Deseas beats V1   : {sku_agg['deseas_beats_v1'].sum()} / {len(sku_agg)} SKUs")
    print(f"  Raw beats V1      : {sku_agg['raw_beats_v1'].sum()} / {len(sku_agg)} SKUs")

    print(f"\n  Top 10 where deseas helps most (+ve delta = deseas better):")
    top_h = sku_agg.nlargest(10,"delta").merge(
        best_deseas[["unique_id","best_deseas_model"]], on="unique_id", how="left"
    )
    print(top_h[["unique_id","best_deseas_model","raw_total_ae",
                 "deseas_total_ae","v1_total_ae","delta"]].to_string(index=False))

    print(f"\n  Top 10 where deseas hurts most:")
    top_hu = sku_agg.nsmallest(10,"delta").merge(
        best_deseas[["unique_id","best_deseas_model"]], on="unique_id", how="left"
    )
    print(top_hu[["unique_id","best_deseas_model","raw_total_ae",
                  "deseas_total_ae","v1_total_ae","delta"]].to_string(index=False))

    v1_wins_raw = sku_agg[~sku_agg["raw_beats_v1"]]["unique_id"].tolist()
    print(f"\n--- Previously V1-winning SKUs ({len(v1_wins_raw)}) after deseas ---")
    sub = sku_agg[sku_agg["unique_id"].isin(v1_wins_raw)].merge(
        best_deseas[["unique_id","best_deseas_model"]], on="unique_id", how="left"
    )
    sub["now_beats_v1"] = sub["deseas_total_ae"] < sub["v1_total_ae"]
    print(sub[["unique_id","best_deseas_model","raw_total_ae","deseas_total_ae",
               "v1_total_ae","now_beats_v1"]].sort_values("deseas_total_ae").to_string(index=False))
    print(f"\n  Flipped (deseas now beats V1): {sub['now_beats_v1'].sum()} / {len(sub)}")


if __name__ == "__main__":
    main()
