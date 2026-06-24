#!/usr/bin/env python3
"""
Experiment: apply the full-history model treatment to short-history smooth SKUs.

Current baseline: WindowAverage(12), no deseasonalization, no CV.
This experiment: full model set (AutoARIMA, AutoETS-damped, WindowAverage(8),
Naive, HistoricAverage) + deseasonalization + CV model selection.

Compared against V1 on the same test period.
Also generates plots: outputs/reports/short_smooth_fullmodels_vs_v1.pdf

Delete this file to remove the experiment.
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
from statsforecast import StatsForecast

from config import FREQUENCY, TEST_WEEKS, TRIM_TRAILING_WEEKS, OUTPUTS_REPORTS
from src.models import get_models
from src.baselines import get_baselines
from src.deseasonalize import deseasonalize, reseasonalize
from src.backtest import _trim_to_train_start
from compare_v1 import build_cumsum_index, v1_forecast

PROCESSED_DIR = ROOT / "data/processed"
CONTEXT_WEEKS = 12
SKUS_PER_PAGE = 6
MIN_CV_WINDOWS = 1       # one fold for everyone
# AutoETS(damped=True) crashes when fold train ≤ 9 rows (n <= npars+4, npars=5)
# With h=10 and n_windows=1: fold train = total - 10, so need total ≥ 20
AUTOETS_MIN_FOLD_TRAIN = 10
AUTOETS_MIN_SERIES = MIN_CV_WINDOWS * TEST_WEEKS + AUTOETS_MIN_FOLD_TRAIN  # 20 weeks


def pick_weekly(uid_f: pd.DataFrame, model_name: str) -> np.ndarray:
    if model_name.startswith("Ensemble:"):
        cols = [c for c in model_name.replace("Ensemble:", "").split("+")
                if c in uid_f.columns]
        return uid_f[cols].mean(axis=1).values if cols else np.full(len(uid_f), np.nan)
    if model_name in uid_f.columns:
        return uid_f[model_name].values
    avail = [c for c in uid_f.columns if c not in {"unique_id", "ds"}]
    return uid_f[avail[0]].values if avail else np.full(len(uid_f), np.nan)


def select_by_mase(cv_df: pd.DataFrame, train_df: pd.DataFrame) -> dict:
    naive_mae = (
        train_df.sort_values(["unique_id", "ds"])
        .groupby("unique_id")["y"]
        .apply(lambda s: float(np.abs(np.diff(s.values)).mean()) if len(s) > 1 else 1.0)
        .reset_index(name="naive_mae")
    )
    naive_mae["naive_mae"] = naive_mae["naive_mae"].clip(lower=1e-6)
    meta   = {"unique_id", "ds", "cutoff", "y", "history_length"}
    m_cols = [c for c in cv_df.columns if c not in meta]
    long = (
        cv_df.melt(id_vars=["unique_id", "ds", "y"], value_vars=m_cols,
                   var_name="model", value_name="yhat")
        .dropna(subset=["yhat"])
        .merge(naive_mae, on="unique_id", how="left")
    )
    long["mase"] = (long["y"] - long["yhat"]).abs() / long["naive_mae"]
    return (
        long.groupby(["unique_id", "model"])["mase"].mean()
        .reset_index().sort_values(["unique_id", "mase"])
        .groupby("unique_id").first()["model"].to_dict()
    )


def main():
    weekly    = pd.read_parquet(PROCESSED_DIR / "sales_clean.parquet")
    profiles  = pd.read_csv(PROCESSED_DIR / "sku_profiles.csv")
    weekly["ds"] = pd.to_datetime(weekly["ds"])
    profiles["train_start"] = pd.to_datetime(profiles["train_start"])

    all_weeks  = sorted(weekly["ds"].unique())
    trimmed    = all_weeks[:-TRIM_TRAILING_WEEKS]
    test_start = pd.Timestamp(trimmed[-TEST_WEEKS])
    cutoff     = pd.Timestamp(trimmed[-(TEST_WEEKS + 1)])
    test_end   = pd.Timestamp(trimmed[-1])
    test_wks   = [w for w in trimmed if w >= test_start]

    short_uids = profiles.loc[
        (profiles["bucket"] == "smooth") & (profiles["history_length"] == "short"),
        "unique_id",
    ].tolist()
    print(f"{len(short_uids)} short-history smooth SKUs")
    print(f"Cutoff {cutoff.date()} | Test {test_start.date()} → {test_end.date()}")

    # ── Training data ──────────────────────────────────────────────────────────
    train_trimmed = _trim_to_train_start(
        weekly[weekly["ds"].isin(trimmed) & (weekly["ds"] < test_start)].copy(), profiles
    )
    train_g = train_trimmed[train_trimmed["unique_id"].isin(short_uids)][
        ["unique_id", "ds", "y"]
    ].copy()

    lengths = train_g.groupby("unique_id")["ds"].count()
    # Full model set needs fold train ≥ 10 → total ≥ 20 weeks
    full_uids    = lengths[lengths >= AUTOETS_MIN_SERIES].index.tolist()
    # Reduced set (no AutoETS) for the few very-short series
    reduced_uids = lengths[lengths < AUTOETS_MIN_SERIES].index.tolist()
    print(f"  Full model set (≥{AUTOETS_MIN_SERIES} weeks): {len(full_uids)} SKUs")
    print(f"  Reduced set   (<{AUTOETS_MIN_SERIES} weeks): {len(reduced_uids)} SKUs  (AutoETS excluded)")

    # ── Model sets ─────────────────────────────────────────────────────────────
    candidates      = get_models("smooth", "full")   # AutoARIMA, AutoETS-damped, WA(8), Naive, HistAvg
    candidate_names = {type(m).__name__ for m in candidates}
    baselines       = [b for b in get_baselines("smooth", "full")
                       if type(b).__name__ not in candidate_names]
    models_full    = candidates + baselines
    models_reduced = [m for m in models_full if type(m).__name__ != "AutoETS"]
    print(f"Full models:    {[type(m).__name__ for m in models_full]}")
    print(f"Reduced models: {[type(m).__name__ for m in models_reduced]}")

    # ── CV — no deseasonalization (short history = incomplete seasonal cycle) ───
    t0 = time.time()
    cv_parts = []
    for uid_group, mset, label in [
        (full_uids,    models_full,    "full"),
        (reduced_uids, models_reduced, "reduced"),
    ]:
        if not uid_group:
            continue
        grp_train = train_g[train_g["unique_id"].isin(uid_group)].copy()
        sf_cv = StatsForecast(models=mset, freq=FREQUENCY, n_jobs=-1)
        cv_part = sf_cv.cross_validation(
            df=grp_train[["unique_id", "ds", "y"]],
            h=TEST_WEEKS, n_windows=MIN_CV_WINDOWS, step_size=TEST_WEEKS,
        )
        cv_parts.append(cv_part)

    cv_df = pd.concat(cv_parts, ignore_index=True)
    print(f"CV done ({time.time()-t0:.1f}s)")

    sel_map = select_by_mase(cv_df, train_g)
    print(f"Model selection: {pd.Series(sel_map).value_counts().to_dict()}")

    # ── Final fit + predict — no deseasonalization ─────────────────────────────
    t0 = time.time()
    fcast_parts = []
    for uid_group, mset in [(full_uids, models_full), (reduced_uids, models_reduced)]:
        if not uid_group:
            continue
        grp_train = train_g[train_g["unique_id"].isin(uid_group)].copy()
        sf_fit = StatsForecast(models=mset, freq=FREQUENCY, n_jobs=-1)
        sf_fit.fit(grp_train[["unique_id", "ds", "y"]])
        part = sf_fit.predict(h=TEST_WEEKS)
        part["ds"] = pd.to_datetime(part["ds"])
        fcast_parts.append(part)

    fcast = pd.concat(fcast_parts, ignore_index=True)
    print(f"Fit+predict done ({time.time()-t0:.1f}s)")

    # ── V1 ─────────────────────────────────────────────────────────────────────
    raw = pd.read_parquet(PROCESSED_DIR / "orders_raw.parquet")
    raw["order_date"] = pd.to_datetime(raw["order_date"])
    index = build_cumsum_index(raw)
    v1_totals = {}
    for uid in short_uids:
        try:
            v1_totals[uid] = v1_forecast(index, uid, cutoff)
        except Exception:
            pass
    print(f"V1: {len(v1_totals)}/{len(short_uids)}")

    # ── Also load baseline WindowAverage(12) results from previous run ─────────
    baseline_path = OUTPUTS_REPORTS / "test_evaluation.csv"
    wa12_map = {}
    if baseline_path.exists():
        base_df = pd.read_csv(baseline_path)
        base_short = base_df[base_df["unique_id"].isin(short_uids)]
        wa12_map = base_short.set_index("unique_id")["yhat_total"].to_dict()

    # ── Actuals ────────────────────────────────────────────────────────────────
    test_df       = weekly[weekly["ds"].isin(test_wks) & weekly["unique_id"].isin(short_uids)]
    actual_totals = test_df.groupby("unique_id")["y"].sum()

    # ── Summary ────────────────────────────────────────────────────────────────
    rows = []
    for uid in short_uids:
        uid_f    = fcast[fcast["unique_id"] == uid].sort_values("ds")
        mn       = sel_map.get(uid, "HistoricAverage")
        new_fc   = float(pd.Series(pick_weekly(uid_f, mn)).sum()) if not uid_f.empty else np.nan
        actual   = float(actual_totals.get(uid, 0))
        v1       = v1_totals.get(uid, np.nan)
        wa12     = wa12_map.get(uid, np.nan)
        rows.append({
            "uid": uid, "actual": actual,
            "new_model": new_fc, "wa12": wa12, "v1": v1,
            "ae_new":  abs(actual - new_fc) if not np.isnan(new_fc) else np.nan,
            "ae_wa12": abs(actual - wa12)   if not np.isnan(wa12)  else np.nan,
            "ae_v1":   abs(actual - v1)     if not np.isnan(v1)    else np.nan,
        })

    df = pd.DataFrame(rows)
    valid = df[df["ae_v1"].notna() & df["ae_new"].notna()]

    print(f"\n{'='*62}")
    print(f"SHORT SMOOTH SKUs — full model treatment vs WA(12) vs V1")
    print(f"{'='*62}")
    print(f"  {'':28} {'V1':>8} {'WA(12)':>8} {'New':>8}")
    print(f"  {'MAE':28} {valid['ae_v1'].mean():>8.2f} "
          f"{valid['ae_wa12'].mean():>8.2f} {valid['ae_new'].mean():>8.2f}")
    print(f"  {'WAPE':28} "
          f"{valid['ae_v1'].sum()/max(valid['actual'].sum(),1e-6):>8.4f} "
          f"{valid['ae_wa12'].sum()/max(valid['actual'].sum(),1e-6):>8.4f} "
          f"{valid['ae_new'].sum()/max(valid['actual'].sum(),1e-6):>8.4f}")
    print(f"  {'Bias':28} "
          f"{(valid['v1']-valid['actual']).mean():>+8.2f} "
          f"{(valid['wa12']-valid['actual']).mean():>+8.2f} "
          f"{(valid['new_model']-valid['actual']).mean():>+8.2f}")
    new_beats_v1  = (valid["ae_new"] < valid["ae_v1"]).sum()
    new_beats_wa12= (valid["ae_new"] < valid["ae_wa12"]).sum()
    print(f"  {'New beats V1':28} {new_beats_v1:>8} / {len(valid)}")
    print(f"  {'New beats WA(12)':28} {new_beats_wa12:>8} / {len(valid)}")

    # ── Sort for plots: new-beats-V1 first, then V1-wins ─────────────────────
    df["wins_v1"] = df["ae_new"] < df["ae_v1"]
    sorted_uids = (
        df.assign(ws=df["wins_v1"].map({True: 0, False: 1, None: 2}))
        .sort_values(["ws", "actual"], ascending=[True, False])["uid"].tolist()
    )
    n_wins = int(df["wins_v1"].sum())

    # ── Plot ──────────────────────────────────────────────────────────────────
    out_path = OUTPUTS_REPORTS / "short_smooth_fullmodels_vs_v1.pdf"
    n_pages  = (len(sorted_uids) + SKUS_PER_PAGE - 1) // SKUS_PER_PAGE
    print(f"\nPlotting {len(sorted_uids)} SKUs → {out_path.name}")

    with pdf_backend.PdfPages(out_path) as pdf:
        for page_start in range(0, len(sorted_uids), SKUS_PER_PAGE):
            page_uids = sorted_uids[page_start:page_start + SKUS_PER_PAGE]
            fig, axes = plt.subplots(3, 2, figsize=(16, 12))
            axes = axes.flatten()

            for ax_i, uid in enumerate(page_uids):
                ax  = axes[ax_i]
                row = df[df["uid"] == uid].iloc[0]

                hist = weekly[weekly["unique_id"] == uid].sort_values("ds").set_index("ds")["y"]
                ctx  = hist[
                    (hist.index >= test_start - pd.Timedelta(weeks=CONTEXT_WEEKS)) &
                    (hist.index < test_start)
                ]
                t_actual = test_df[test_df["unique_id"] == uid].sort_values("ds").set_index("ds")["y"]
                uid_f    = fcast[fcast["unique_id"] == uid].sort_values("ds").set_index("ds")
                mn       = sel_map.get(uid, "HistoricAverage")
                new_wk   = pd.Series(pick_weekly(uid_f.reset_index(), mn), index=uid_f.index)

                v1_total  = row["v1"]
                v1_wk     = v1_total / TEST_WEEKS if not np.isnan(v1_total) else np.nan

                wins      = row["wins_v1"]
                win_color = "#2e7d32" if wins else ("#c62828" if wins is False else "#777")
                win_str   = "✓ new model" if wins else ("✗ V1 wins" if wins is False else "—")

                if not ctx.empty:
                    ax.plot(ctx.index, ctx.values, color="#ccc", lw=1.1, zorder=1)
                ax.axvline(test_start, color="#aaa", lw=0.8, ls="--", zorder=2)

                if not t_actual.empty:
                    ax.plot(t_actual.index, t_actual.values,
                            color="#1565C0", lw=2, marker="o", ms=5,
                            label=f"Actual   {row['actual']:.0f}", zorder=5)
                if not new_wk.empty:
                    ax.plot(new_wk.index, new_wk.values,
                            color="#2E7D32", lw=1.8, ls="--", marker="s", ms=4,
                            label=f"{mn[:16]}  {row['new_model']:.0f}  (Δ{row['new_model']-row['actual']:+.0f})",
                            zorder=4)
                if not np.isnan(v1_wk) and not t_actual.empty:
                    ax.plot(t_actual.index, [v1_wk]*len(t_actual),
                            color="#E53935", lw=1.6, ls=":", marker="^", ms=4,
                            label=f"V1       {v1_total:.0f}  (Δ{row['v1']-row['actual']:+.0f})",
                            zorder=4)

                ax.axvspan(test_start, test_end, alpha=0.07, color="#1565C0", zorder=0)

                # Show launch date if visible in context
                ts_row = profiles.loc[profiles["unique_id"]==uid, "train_start"]
                if not ts_row.empty:
                    ts = pd.Timestamp(ts_row.iloc[0])
                    if ts >= test_start - pd.Timedelta(weeks=CONTEXT_WEEKS):
                        ax.axvline(ts, color="#FF9800", lw=1, ls="-.", alpha=0.7, zorder=2)

                all_vals = []
                if not t_actual.empty:   all_vals += list(t_actual.values)
                if not new_wk.empty:     all_vals += list(new_wk.dropna().values)
                if not np.isnan(v1_wk):  all_vals += [v1_wk] * TEST_WEEKS
                if not ctx.empty:        all_vals += list(ctx.values)
                if all_vals:
                    ax.set_ylim(max(0, min(all_vals)*0.80), max(all_vals)*1.22)

                ax.set_title(f"{uid}   [{mn}]   {win_str}",
                             fontsize=7.5, color=win_color, pad=3)
                ax.legend(fontsize=7, loc="upper left", framealpha=0.85,
                          handlelength=1.4, borderpad=0.4)
                ax.tick_params(labelsize=7)
                ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
                ax.xaxis.set_major_locator(matplotlib.dates.WeekdayLocator(byweekday=0, interval=2))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right")
                ax.set_ylabel("Units / week", fontsize=7)
                ax.grid(axis="y", alpha=0.25)

            for i in range(len(page_uids), SKUS_PER_PAGE):
                axes[i].set_visible(False)

            fig.suptitle(
                f"Short-history smooth — full model set vs V1   "
                f"cutoff {cutoff.date()} | {test_start.date()}–{test_end.date()}   "
                f"New model wins: {n_wins}/{len(sorted_uids)}   "
                f"Page {page_start//SKUS_PER_PAGE+1}/{n_pages}   "
                f"(orange = launch date)",
                fontsize=9, y=1.005,
            )
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
