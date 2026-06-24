#!/usr/bin/env python3
"""
Optimize the HOLIDAY_MULTIPLIER hyperparameter via cross-validation.

The multiplier is treated as a 1-D hyperparameter: for each candidate value M,
we deseasonalize the training data with that multiplier applied to the holiday
window, run CV, reseasonalize, and measure MAE/WAPE.

Scoring is on folds whose test period overlaps the holiday window — those are the
only folds where M has any meaningful effect on the forecast.  Overall metrics are
shown as a sanity check that we're not hurting non-holiday weeks.

Usage:
  python scripts/optimize_holiday_multiplier.py              # coarse grid (default)
  python scripts/optimize_holiday_multiplier.py --fine       # coarse then fine around winner
  python scripts/optimize_holiday_multiplier.py --multipliers 1.4 1.6 1.8  # custom list
  python scripts/optimize_holiday_multiplier.py --window-start 11-20 --window-end 12-14

Output:
  outputs/reports/holiday_multiplier_search.csv   — full grid results for plotting
  Printed ranked table with the recommended value marked.
"""
import sys, argparse, time
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, AutoARIMA, WindowAverage, Naive, HistoricAverage

# ── Pipeline settings ─────────────────────────────────────────────────────────
N_CV_WINDOWS = 3     # medium CV — enough signal, keeps each run ~30-45s
TEST_WEEKS   = 10
TRIM_TAIL    = 3
FREQUENCY    = "W-MON"

# Default holiday window: Black Friday week → first half of December
DEFAULT_WINDOW_START = (11, 20)
DEFAULT_WINDOW_END   = (12, 14)

# Monthly factors applied outside the holiday window.
# Nov 1-19 and Dec 15-31 fall back to 1.0 (treated as baseline).
# If you want Nov pre-window or Dec post-window to have a different factor,
# edit SEASONAL_OUTSIDE here; the optimizer's objective won't change.
SEASONAL_OUTSIDE: dict[int, float] = {
    1: 0.75, 2: 0.80, 3: 0.90, 4: 0.95,
    5: 1.00, 6: 1.00, 7: 1.00, 8: 1.00, 9: 1.00,
    10: 1.10,
    11: 1.00,   # Nov 1-19 (pre-window) → baseline
    12: 1.00,   # Dec 15-31 (post-window) → baseline
}

# Coarse grid: wide range to locate the rough optimum before the fine pass.
# With a 25-day window, the V1 prior implies ~1.65; grid covers 1.0–2.5.
COARSE_GRID = [1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 2.00, 2.20, 2.50]

MODELS = [
    AutoETS(season_length=1, damped=True,  alias="ETS_D"),
    AutoETS(season_length=1, damped=False, alias="ETS_U"),
    AutoARIMA(season_length=1,             alias="ARIMA"),
    WindowAverage(window_size=8,           alias="WA8"),
    Naive(                                 alias="Naive"),
    HistoricAverage(                       alias="HistAvg"),
]


# ── Factor helpers ─────────────────────────────────────────────────────────────
def _in_window(ds: pd.Series, ws: tuple, we: tuple) -> pd.Series:
    m, d = ds.dt.month, ds.dt.day
    m0, d0 = ws
    m1, d1 = we
    if m0 == m1:
        return (m == m0) & (d >= d0) & (d <= d1)
    return (
        ((m == m0) & (d >= d0)) |
        ((m > m0)  & (m < m1))  |
        ((m == m1) & (d <= d1))
    )


def _factors(ds: pd.Series, multiplier: float, ws: tuple, we: tuple) -> pd.Series:
    f = ds.dt.month.map(SEASONAL_OUTSIDE).astype(float)
    f = f.where(~_in_window(ds, ws, we), multiplier)
    return f


def _deseas(df: pd.DataFrame, multiplier: float, ws: tuple, we: tuple) -> pd.DataFrame:
    df = df.copy()
    df["y"] = df["y"] / _factors(df["ds"], multiplier, ws, we)
    return df


def _reseas(df: pd.DataFrame, multiplier: float, ws: tuple, we: tuple) -> pd.DataFrame:
    df   = df.copy()
    f    = _factors(df["ds"], multiplier, ws, we)
    meta = {"unique_id", "ds", "cutoff", "y", "history_length"}
    for col in ["y"] + [c for c in df.columns if c not in meta]:
        if col in df.columns:
            df[col] = df[col] * f
    return df


# ── CV ─────────────────────────────────────────────────────────────────────────
def run_cv(train_raw: pd.DataFrame, smooth_prof: pd.DataFrame,
           multiplier: float, ws: tuple, we: tuple) -> pd.DataFrame:
    parts = []
    for hist_len, n_win in [("full", N_CV_WINDOWS), ("medium", N_CV_WINDOWS)]:
        uids = smooth_prof.loc[smooth_prof["history_length"] == hist_len, "unique_id"].tolist()
        df_g = _deseas(
            train_raw[train_raw["unique_id"].isin(uids)][["unique_id", "ds", "y"]],
            multiplier, ws, we
        )
        min_len = n_win * TEST_WEEKS + 1
        short   = df_g.groupby("unique_id")["ds"].count()
        df_g    = df_g[~df_g["unique_id"].isin(short[short < min_len].index)]

        sf = StatsForecast(models=MODELS, freq=FREQUENCY, n_jobs=-1)
        cv = sf.cross_validation(df=df_g, h=TEST_WEEKS, n_windows=n_win,
                                 step_size=TEST_WEEKS)
        cv["history_length"] = hist_len
        parts.append(cv)

    cv_out = pd.concat(parts, ignore_index=True)
    cv_out["ds"]     = pd.to_datetime(cv_out["ds"])
    cv_out["cutoff"] = pd.to_datetime(cv_out["cutoff"])
    return _reseas(cv_out, multiplier, ws, we)


# ── Model selection ────────────────────────────────────────────────────────────
def select_best(cv_df: pd.DataFrame, train_raw: pd.DataFrame) -> dict:
    naive_step = (
        train_raw.sort_values(["unique_id", "ds"])
        .groupby("unique_id")["y"]
        .apply(lambda s: float(np.abs(np.diff(s.values)).mean()) if len(s) > 1 else 1.0)
        .reset_index(name="naive_mae")
    )
    naive_step["naive_mae"] = naive_step["naive_mae"].clip(lower=1e-6)

    meta   = {"unique_id", "ds", "cutoff", "y", "history_length"}
    m_cols = [c for c in cv_df.columns if c not in meta]
    long   = (
        cv_df.melt(id_vars=["unique_id", "ds", "y"], value_vars=m_cols,
                   var_name="model", value_name="yhat")
        .dropna(subset=["yhat"])
        .merge(naive_step, on="unique_id", how="left")
    )
    long["mase"] = (long["y"] - long["yhat"]).abs() / long["naive_mae"]
    return (
        long.groupby(["unique_id", "model"])["mase"].mean()
        .reset_index().sort_values(["unique_id", "mase"])
        .groupby("unique_id").first()["model"].to_dict()
    )


# ── Scoring ────────────────────────────────────────────────────────────────────
def score(cv_df: pd.DataFrame, sel_map: dict, ws: tuple, we: tuple) -> pd.DataFrame:
    rows = []
    meta = {"unique_id", "ds", "cutoff", "y", "history_length"}
    for (uid, cutoff), grp in cv_df.groupby(["unique_id", "cutoff"]):
        grp    = grp.sort_values("ds")
        actual = float(grp["y"].sum())
        m      = sel_map.get(uid, "")
        cols   = [c for c in grp.columns if c not in meta and grp[c].notna().any()]
        yhat   = float(grp[m].sum()) if m in grp.columns and grp[m].notna().any() \
                 else (float(grp[cols[0]].sum()) if cols else np.nan)
        # flag folds whose test period touches the holiday window
        holiday_fold = bool(_in_window(grp["ds"], ws, we).any())
        rows.append({"unique_id": uid, "cutoff": cutoff,
                     "actual": actual, "yhat": yhat,
                     "holiday_fold": holiday_fold})

    r = pd.DataFrame(rows)
    r["ae"]   = (r["actual"] - r["yhat"]).abs()
    r["bias"] = r["yhat"] - r["actual"]
    return r


def _metrics(r: pd.DataFrame, holiday_only: bool) -> dict:
    sub = r[r["holiday_fold"]] if holiday_only else r
    if sub.empty:
        return {"mae": np.nan, "wape": np.nan, "bias": np.nan, "n": 0}
    return {
        "mae":  float(sub["ae"].mean()),
        "wape": float(sub["ae"].sum() / max(sub["actual"].sum(), 1e-6)),
        "bias": float(sub["bias"].mean()),
        "n":    len(sub),
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--fine", action="store_true",
                        help="Run a fine grid (±0.1, step 0.02) around the coarse winner")
    parser.add_argument("--multipliers", nargs="+", type=float,
                        help="Custom multiplier list (overrides built-in grid)")
    parser.add_argument("--window-start", default="11-20",
                        help="Window start MM-DD (default: 11-20, Black Friday week)")
    parser.add_argument("--window-end", default="12-14",
                        help="Window end MM-DD (default: 12-14, first half of December)")
    args = parser.parse_args()

    ws = tuple(int(x) for x in args.window_start.split("-"))
    we = tuple(int(x) for x in args.window_end.split("-"))

    print(f"Holiday window: {ws[0]:02d}/{ws[1]:02d} → {we[0]:02d}/{we[1]:02d}")
    print("Loading data…")

    hist = pd.read_parquet(ROOT / "data/processed/sales_clean.parquet")
    prof = pd.read_csv(ROOT / "data/processed/sku_profiles.csv")
    hist["ds"]          = pd.to_datetime(hist["ds"])
    prof["train_start"] = pd.to_datetime(prof["train_start"])

    smooth_prof = prof[(prof["bucket"] == "smooth") &
                       (prof["history_length"].isin(["full", "medium"]))].copy()
    smooth_uids = set(smooth_prof["unique_id"])

    all_weeks  = sorted(hist["ds"].unique())
    trimmed    = all_weeks[:-TRIM_TAIL] if TRIM_TAIL else all_weeks
    test_start = pd.Timestamp(trimmed[-TEST_WEEKS])

    hist_sm = hist[hist["unique_id"].isin(smooth_uids)].merge(
        smooth_prof[["unique_id", "train_start", "history_length"]], on="unique_id"
    )
    train_raw = (
        hist_sm[hist_sm["ds"].isin(trimmed) & (hist_sm["ds"] < test_start)]
        .pipe(lambda d: d[d["ds"] >= d["train_start"]])
        [["unique_id", "ds", "y", "history_length"]].copy()
    )
    print(f"  {len(smooth_uids)} smooth SKUs (full+medium) | "
          f"train ends {train_raw['ds'].max().date()}\n")

    grid = list(args.multipliers) if args.multipliers else list(COARSE_GRID)
    results: list[dict] = []

    def _eval(m: float) -> dict:
        t0  = time.time()
        cv  = run_cv(train_raw, smooth_prof, m, ws, we)
        sel = select_best(cv, train_raw)
        r   = score(cv, sel, ws, we)
        mh  = _metrics(r, holiday_only=True)
        mo  = _metrics(r, holiday_only=False)
        elapsed = time.time() - t0
        print(f"  M={m:.3f}  holiday MAE={mh['mae']:.2f}  holiday WAPE={mh['wape']:.4f}"
              f"  bias={mh['bias']:+.2f}  overall MAE={mo['mae']:.2f}  ({elapsed:.1f}s)")
        return {
            "multiplier":      m,
            "mae_holiday":     mh["mae"],
            "wape_holiday":    mh["wape"],
            "bias_holiday":    mh["bias"],
            "n_holiday_folds": mh["n"],
            "mae_overall":     mo["mae"],
            "wape_overall":    mo["wape"],
        }

    print("── Coarse grid ──────────────────────────────────────────────────────")
    for m in sorted(grid):
        results.append(_eval(m))

    if args.fine:
        df_tmp   = pd.DataFrame(results)
        best_m   = float(df_tmp.loc[df_tmp["mae_holiday"].idxmin(), "multiplier"])
        lo, hi   = round(max(1.0, best_m - 0.12), 3), round(best_m + 0.12, 3)
        done     = {r["multiplier"] for r in results}
        fine_grid = sorted(
            round(v, 3) for v in np.arange(lo, hi + 0.001, 0.02) if round(v, 3) not in done
        )
        if fine_grid:
            print(f"\n── Fine grid around M={best_m:.3f} ({'–'.join([str(lo),str(hi)])})"
                  f" ──────────────────────")
            for m in fine_grid:
                results.append(_eval(m))

    # ── Output ─────────────────────────────────────────────────────────────────
    df_res = pd.DataFrame(results).sort_values("multiplier").reset_index(drop=True)
    out_path = ROOT / "outputs/reports/holiday_multiplier_search.csv"
    df_res.to_csv(out_path, index=False)

    best_idx_h = int(df_res["mae_holiday"].idxmin())
    best_idx_o = int(df_res["mae_overall"].idxmin())
    best_m_h   = float(df_res.loc[best_idx_h, "multiplier"])
    best_m_o   = float(df_res.loc[best_idx_o, "multiplier"])

    print(f"\n{'='*72}")
    print(f"HOLIDAY MULTIPLIER SEARCH  |  window {ws[0]:02d}/{ws[1]:02d}–{we[0]:02d}/{we[1]:02d}")
    print(f"{'='*72}")
    hdr = f"  {'M':>7}  {'holiday MAE':>12}  {'holiday WAPE':>13}  {'bias':>8}  {'overall MAE':>12}"
    print(hdr)
    print("  " + "-"*68)
    for _, row in df_res.iterrows():
        tag = ""
        if row["multiplier"] == best_m_h:
            tag += "  ◄ best holiday MAE"
        elif row["multiplier"] == best_m_o and best_m_o != best_m_h:
            tag += "  ◄ best overall MAE"
        print(f"  {row['multiplier']:>7.3f}  "
              f"{row['mae_holiday']:>12.2f}  "
              f"{row['wape_holiday']:>13.4f}  "
              f"{row['bias_holiday']:>+8.2f}  "
              f"{row['mae_overall']:>12.2f}"
              f"{tag}")

    print(f"\nRecommendation: HOLIDAY_MULTIPLIER = {best_m_h:.3f}")
    if best_m_h != best_m_o:
        print(f"  (overall-MAE optimum is {best_m_o:.3f} — prefer holiday MAE"
              f" as primary objective)")
    print(f"\nTo apply: set HOLIDAY_MULTIPLIER = {best_m_h:.3f} in config.py")
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
