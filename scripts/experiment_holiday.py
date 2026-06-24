#!/usr/bin/env python3
"""
EXPERIMENT: Holiday window flag + damped-vs-undamped AutoETS A/B test.

Candidates
──────────
  Baseline : current pipeline (deseasonalize with monthly index, no holiday flag,
             damped AutoETS).  Results loaded from cv_results.parquet.
  A        : holiday flag ON + damped AutoETS   (robust; flag helps but damping
             still protects against flag error)
  B        : holiday flag ON + undamped AutoETS  (bolder; better if flag is
             calibrated, riskier if it overshoots Q4)

Both A and B are extracted from a single CV run (both ETS variants compete);
per-SKU selection picks whichever is best for each candidate.

Focus windows: Oct 2025 (Q4 ramp) and Jan 2026 (post-Q4 cliff) — the two
windows where damped-vs-undamped and the holiday flag are expected to diverge.

Reads only — does not touch the main pipeline outputs.
Delete this file to remove the experiment.
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

# ── Config (copied from config.py for self-containment inside __main__ guard) ─
N_CV_FULL   = 6
N_CV_MEDIUM = 3
TEST_WEEKS  = 10
TRIM_TAIL   = 3
FREQUENCY   = "W-MON"

HOLIDAY_START      = (11, 20)
HOLIDAY_END        = (12, 31)
HOLIDAY_MULTIPLIER = 1.35      # from config.py; change here to test sensitivity

SEASONAL_HOLIDAY: dict[int, float] = {
    1: 0.75, 2: 0.80, 3: 0.90, 4: 0.95,
    5: 1.00, 6: 1.00, 7: 1.00, 8: 1.00, 9: 1.00,
    10: 1.10, 11: 1.00, 12: 1.00,   # Nov/Dec zeroed; holiday window handles them
}

Q4_CUTOFFS  = [pd.Timestamp("2025-10-27"), pd.Timestamp("2026-01-05")]
ALL_CUTOFFS = [
    pd.Timestamp("2025-01-20"), pd.Timestamp("2025-03-31"),
    pd.Timestamp("2025-06-09"), pd.Timestamp("2025-08-18"),
    pd.Timestamp("2025-10-27"), pd.Timestamp("2026-01-05"),
]


# ── Factor helpers (mirror src/deseasonalize.py logic) ────────────────────────
def _is_holiday(ds: pd.Series) -> pd.Series:
    h_m0, h_d0 = HOLIDAY_START
    h_m1, h_d1 = HOLIDAY_END
    m, d = ds.dt.month, ds.dt.day
    return (((m == h_m0) & (d >= h_d0)) |
            ((m > h_m0) & (m < h_m1)) |
            ((m == h_m1) & (d <= h_d1)))


def _factors(ds: pd.Series) -> pd.Series:
    f = ds.dt.month.map(SEASONAL_HOLIDAY)
    f = f.where(~_is_holiday(ds), HOLIDAY_MULTIPLIER)
    return f


def deseasonalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["y"] = df["y"] / _factors(df["ds"])
    return df


def reseasonalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    factor = _factors(df["ds"])
    meta = {"unique_id", "ds", "cutoff", "y", "bucket", "history_length"}
    for col in ["y"] + [c for c in df.columns if c not in meta]:
        if col in df.columns:
            df[col] = df[col] * factor
    return df


# ── Scoring helpers ────────────────────────────────────────────────────────────
def get_model_total(uid, cutoff, grp, sel_map, model_override=None):
    """Sum of selected (or overridden) model's weekly predictions over the horizon."""
    m = model_override or sel_map.get(uid, "")
    if m.startswith("Ensemble:"):
        parts = m.split(":")[1].split("+")
        cols  = [p for p in parts if p in grp.columns and grp[p].notna().any()]
        return float(grp[cols].mean(axis=1).sum()) if cols else np.nan
    elif m in grp.columns and grp[m].notna().any():
        return float(grp[m].sum())
    avail = [c for c in grp.columns
             if c not in {"unique_id","ds","cutoff","y","bucket","history_length"}
             and grp[c].notna().any()]
    return float(grp[avail[0]].sum()) if avail else np.nan


def build_results(cv_df, sel_map, cmp_smooth, label,
                  damped_col="AutoETS_D", undamped_col="AutoETS_U"):
    """Produce a comparison DataFrame from a CV result set."""
    rows = []
    for (uid, cutoff), grp in cv_df.groupby(["unique_id", "cutoff"]):
        grp    = grp.sort_values("ds")
        actual = float(grp["y"].sum())

        # Best-of-all-models (picking between damped and undamped included)
        m_best = get_model_total(uid, cutoff, grp, sel_map)

        # Forced A: best model excluding undamped ETS
        grp_a = grp.drop(columns=[undamped_col], errors="ignore")
        sel_a = {k: (v if v != undamped_col else damped_col) for k, v in sel_map.items()}
        m_a   = get_model_total(uid, cutoff, grp_a, sel_a)

        # Forced B: best model excluding damped ETS
        grp_b = grp.drop(columns=[damped_col], errors="ignore")
        sel_b = {k: (v if v != damped_col else undamped_col) for k, v in sel_map.items()}
        m_b   = get_model_total(uid, cutoff, grp_b, sel_b)

        # V1
        v1_row = cmp_smooth[(cmp_smooth["unique_id"] == uid) &
                             (cmp_smooth["cutoff"] == cutoff)]
        v1 = float(v1_row["v1_yhat"].iloc[0]) if not v1_row.empty else np.nan

        rows.append({
            "unique_id": uid, "cutoff": cutoff,
            "history_length": grp["history_length"].iloc[0],
            "actual": actual,
            "A_yhat": m_a, "B_yhat": m_b, "best_yhat": m_best, "v1": v1,
        })

    r = pd.DataFrame(rows)
    for c, yhat in [("A","A_yhat"), ("B","B_yhat"), ("best","best_yhat"), ("v1","v1")]:
        r[f"ae_{c}"] = (r["actual"] - r[yhat]).abs()
    return r


def block(df, label):
    """Print a comparison block."""
    n    = len(df)
    cols = [("A (damped+holiday)","ae_A"), ("B (undamped+holiday)","ae_B"), ("V1","ae_v1")]
    mae  = {k: df[v].mean() for k, v in cols}
    bias = {
        "A (damped+holiday)":   (df["A_yhat"]    - df["actual"]).mean(),
        "B (undamped+holiday)": (df["B_yhat"]    - df["actual"]).mean(),
        "V1":                   (df["v1"]         - df["actual"]).mean(),
    }
    wins_AB = (df["ae_A"] < df["ae_B"]).sum()
    wins_BA = (df["ae_B"] < df["ae_A"]).sum()
    wins_AV = (df["ae_A"] < df["ae_v1"]).sum()
    wins_BV = (df["ae_B"] < df["ae_v1"]).sum()

    w = 26
    print(f"\n{label}  (n={n})")
    print(f"  {'':>{w}} {'Cand A':>14} {'Cand B':>14} {'V1':>12}")
    for name, _ in cols:
        m = mae[name]; b = bias[name]
        print(f"  {'MAE  '+name:>{w}} {m:>14.2f}")
    print(f"  {'Bias':>{w}} {bias['A (damped+holiday)']:>14.2f} "
          f"{bias['B (undamped+holiday)']:>14.2f} {bias['V1']:>12.2f}")
    print(f"  {'A wins vs B':>{w}} {wins_AB:>14} {wins_BA:>14}")
    print(f"  {'Wins vs V1':>{w}} {wins_AV:>14} {wins_BV:>14}")


def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data...")
    hist = pd.read_parquet(ROOT / "data/processed/sales_clean.parquet")
    prof = pd.read_csv(ROOT / "data/processed/sku_profiles.csv")
    sel  = pd.read_csv(ROOT / "outputs/reports/selection.csv")
    cv0  = pd.read_parquet(ROOT / "outputs/reports/cv_results.parquet")   # baseline
    cmp  = pd.read_csv(ROOT / "outputs/reports/v1_comparison.csv")

    hist["ds"]          = pd.to_datetime(hist["ds"])
    prof["train_start"] = pd.to_datetime(prof["train_start"])
    cv0["ds"]           = pd.to_datetime(cv0["ds"])
    cv0["cutoff"]       = pd.to_datetime(cv0["cutoff"])
    cmp["cutoff"]       = pd.to_datetime(cmp["cutoff"])

    smooth_prof = prof[(prof["bucket"] == "smooth") &
                       (prof["history_length"].isin(["full", "medium"]))].copy()
    smooth_uids = set(smooth_prof["unique_id"])
    cmp_smooth  = cmp[cmp["bucket"] == "smooth"].copy()

    all_weeks  = sorted(hist["ds"].unique())
    trimmed    = all_weeks[:-TRIM_TAIL] if TRIM_TAIL else all_weeks
    test_start = pd.Timestamp(trimmed[-TEST_WEEKS])

    hist_smooth = hist[hist["unique_id"].isin(smooth_uids)].merge(
        smooth_prof[["unique_id", "train_start", "history_length"]], on="unique_id"
    )
    train_raw = (
        hist_smooth[hist_smooth["ds"].isin(trimmed) & (hist_smooth["ds"] < test_start)]
        .pipe(lambda d: d[d["ds"] >= d["train_start"]])
        [["unique_id", "ds", "y", "history_length"]].copy()
    )

    # Holiday-adjusted training data
    train_hol = deseasonalize(train_raw)

    # Model set: both ETS variants so we can extract A and B from one CV run
    # AutoETS_D = damped (Candidate A)  |  AutoETS_U = undamped (Candidate B)
    MODELS = [
        AutoETS(season_length=1, damped=True,  alias="AutoETS_D"),
        AutoETS(season_length=1, damped=False, alias="AutoETS_U"),
        AutoARIMA(season_length=1),
        WindowAverage(window_size=8),
        Naive(),
        HistoricAverage(),
    ]

    # ── CV with holiday-adjusted data ─────────────────────────────────────────
    cv_parts = []
    for hist_len, n_windows in [("full", N_CV_FULL), ("medium", N_CV_MEDIUM)]:
        uids = smooth_prof.loc[smooth_prof["history_length"] == hist_len, "unique_id"].tolist()
        df_g = train_hol[train_hol["unique_id"].isin(uids)][["unique_id", "ds", "y"]]

        min_len   = n_windows * TEST_WEEKS + 1
        too_short = df_g.groupby("unique_id")["ds"].count()
        too_short = too_short[too_short < min_len].index.tolist()
        if too_short:
            df_g = df_g[~df_g["unique_id"].isin(too_short)]

        print(f"  CV holiday {hist_len}: {len(uids)} SKUs | n_windows={n_windows}")
        t0 = time.time()
        sf = StatsForecast(models=MODELS, freq=FREQUENCY, n_jobs=-1)
        cv = sf.cross_validation(df=df_g, h=TEST_WEEKS, n_windows=n_windows,
                                 step_size=TEST_WEEKS)
        cv["history_length"] = hist_len
        cv_parts.append(cv)
        print(f"    → {len(cv):,} rows in {time.time()-t0:.1f}s")

    cv_hol           = pd.concat(cv_parts, ignore_index=True)
    cv_hol["ds"]     = pd.to_datetime(cv_hol["ds"])
    cv_hol["cutoff"] = pd.to_datetime(cv_hol["cutoff"])
    cv_hol           = reseasonalize(cv_hol)

    # ── Model selection for holiday run ───────────────────────────────────────
    # Use same naive-MAE denominator as main pipeline
    naive_mae = (
        train_raw.sort_values(["unique_id","ds"])
        .groupby("unique_id")["y"]
        .apply(lambda s: float(np.abs(np.diff(s.values)).mean()) if len(s)>1 else 1.0)
        .reset_index(name="naive_mae")
    )
    naive_mae["naive_mae"] = naive_mae["naive_mae"].clip(lower=1e-6)

    meta = {"unique_id","ds","cutoff","y","bucket","history_length"}
    m_cols = [c for c in cv_hol.columns if c not in meta]

    long = (cv_hol.melt(id_vars=["unique_id","ds","y"], value_vars=m_cols,
                        var_name="model", value_name="yhat")
            .dropna(subset=["yhat"])
            .merge(naive_mae, on="unique_id", how="left"))
    long["ae"]   = (long["y"] - long["yhat"]).abs()
    long["mase"] = long["ae"] / long["naive_mae"]

    sel_hol = (
        long.groupby(["unique_id","model"])
        .agg(MASE=("mase","mean"), MAE=("ae","mean"))
        .reset_index()
        .sort_values(["unique_id","MASE","MAE"])
        .groupby("unique_id").first().reset_index()
        .set_index("unique_id")["model"].to_dict()
    )
    print(f"\nHoliday model selection:")
    from collections import Counter
    print(pd.Series(Counter(sel_hol.values())).sort_values(ascending=False).to_string())

    # ── Score all three: Baseline / Candidate A / Candidate B ────────────────
    baseline_sel = sel.set_index("unique_id")["model"].to_dict()
    cmp_smooth["cutoff"] = pd.to_datetime(cmp_smooth["cutoff"])

    # Baseline results from original cv_results (no holiday, damped ETS)
    base_rows = []
    for (uid, cutoff), grp in cv0[cv0["unique_id"].isin(smooth_uids)].groupby(["unique_id","cutoff"]):
        grp = grp.sort_values("ds")
        yhat= get_model_total(uid, cutoff, grp, baseline_sel)
        v1_row = cmp_smooth[(cmp_smooth["unique_id"]==uid)&(cmp_smooth["cutoff"]==cutoff)]
        v1  = float(v1_row["v1_yhat"].iloc[0]) if not v1_row.empty else np.nan
        base_rows.append({"unique_id":uid, "cutoff":cutoff,
                          "history_length": grp["history_length"].iloc[0],
                          "actual": float(grp["y"].sum()), "yhat": yhat, "v1": v1})
    base_df = pd.DataFrame(base_rows)
    base_df["ae_base"] = (base_df["actual"] - base_df["yhat"]).abs()
    base_df["ae_v1"]   = (base_df["actual"] - base_df["v1"]).abs()

    # Holiday A/B results
    hol_r = build_results(cv_hol, sel_hol, cmp_smooth, "Holiday")

    # Merge baseline + holiday
    comp = base_df.merge(
        hol_r[["unique_id","cutoff","A_yhat","B_yhat","ae_A","ae_B"]],
        on=["unique_id","cutoff"], how="inner"
    )

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("EXPERIMENT: Holiday Flag  |  Damped (A) vs Undamped (B) AutoETS")
    print(f"Holiday window: {HOLIDAY_START[0]}/{HOLIDAY_START[1]} → "
          f"{HOLIDAY_END[0]}/{HOLIDAY_END[1]}  |  multiplier={HOLIDAY_MULTIPLIER}")
    print("="*72)

    def summary_block(df, label):
        n      = len(df)
        mae_bl = df["ae_base"].mean()
        mae_a  = df["ae_A"].mean()
        mae_b  = df["ae_B"].mean()
        mae_v  = df["ae_v1"].mean()
        bias_bl= (df["yhat"]   - df["actual"]).mean()
        bias_a = (df["A_yhat"] - df["actual"]).mean()
        bias_b = (df["B_yhat"] - df["actual"]).mean()
        bias_v = (df["v1"]     - df["actual"]).mean()
        d_a_bl = (mae_bl - mae_a) / mae_bl * 100
        d_b_bl = (mae_bl - mae_b) / mae_bl * 100
        d_ab   = (mae_a  - mae_b) / mae_a  * 100
        w_a_bl = (df["ae_A"]    < df["ae_base"]).sum()
        w_b_bl = (df["ae_B"]    < df["ae_base"]).sum()
        w_ab   = (df["ae_A"]    < df["ae_B"]).sum()
        w_ba   = (df["ae_B"]    < df["ae_A"]).sum()
        print(f"\n{label}  (n={n})")
        print(f"  {'':22} {'Baseline':>12} {'A (damped)':>12} {'B (undamp)':>12} {'V1':>10}")
        print(f"  {'MAE':22} {mae_bl:>12.2f} {mae_a:>12.2f} {mae_b:>12.2f} {mae_v:>10.2f}")
        print(f"  {'Bias':22} {bias_bl:>12.2f} {bias_a:>12.2f} {bias_b:>12.2f} {bias_v:>10.2f}")
        print(f"  {'vs Baseline (delta)':22} {'':>12} {d_a_bl:>+11.1f}% {d_b_bl:>+11.1f}%")
        print(f"  {'Wins vs Baseline':22} {'':>12} {w_a_bl:>12} {w_b_bl:>12}")
        print(f"  {'A wins vs B / B vs A':22} {'':>12} {w_ab:>12} {w_ba:>12}")

    summary_block(comp, "ALL smooth CV windows")
    for hl in ("full", "medium"):
        sub = comp[comp["history_length"] == hl]
        if not sub.empty:
            summary_block(sub, f"  history={hl}")

    print(f"\n--- Per-cutoff MAE ---")
    print(f"  {'Cutoff':14} {'Baseline':>10} {'A (damp)':>10} {'B (undamp)':>11} {'V1':>10} {'Best':>10}")
    for cut, grp in comp.groupby("cutoff"):
        mae_bl = grp["ae_base"].mean()
        mae_a  = grp["ae_A"].mean()
        mae_b  = grp["ae_B"].mean()
        mae_v  = grp["ae_v1"].mean()
        winner = min([("Baseline",mae_bl),("A",mae_a),("B",mae_b),("V1",mae_v)],
                     key=lambda x: x[1])
        q4_tag = " ◄ Q4" if pd.Timestamp(cut) in Q4_CUTOFFS else ""
        print(f"  {str(pd.Timestamp(cut).date()):14} {mae_bl:>10.2f} {mae_a:>10.2f} "
              f"{mae_b:>11.2f} {mae_v:>10.2f} {winner[0]:>10}{q4_tag}")

    # SKU-level
    sku = comp.groupby("unique_id").agg(
        bl_ae=("ae_base","sum"), A_ae=("ae_A","sum"),
        B_ae=("ae_B","sum"),    v1_ae=("ae_v1","sum"),
    ).reset_index()
    sku["A_beats_bl"] = sku["A_ae"] < sku["bl_ae"]
    sku["B_beats_bl"] = sku["B_ae"] < sku["bl_ae"]
    sku["A_beats_B"]  = sku["A_ae"] < sku["B_ae"]
    print(f"\n--- SKU-level ---")
    print(f"  A beats Baseline : {sku['A_beats_bl'].sum()} / {len(sku)}")
    print(f"  B beats Baseline : {sku['B_beats_bl'].sum()} / {len(sku)}")
    print(f"  A beats B        : {sku['A_beats_B'].sum()} / {len(sku)}")

    # Q4/Jan cutoff deep-dive
    print(f"\n--- Q4 + Jan deep-dive ---")
    q4 = comp[comp["cutoff"].isin(Q4_CUTOFFS)]
    summary_block(q4, "  Oct 2025 + Jan 2026 combined")
    for cut in Q4_CUTOFFS:
        summary_block(comp[comp["cutoff"]==cut], f"  {cut.strftime('%b %Y')}")


if __name__ == "__main__":
    main()
