#!/usr/bin/env python3
"""
Plot full-history smooth SKUs: test-period actual vs our model vs V1.
12 weeks of training context shown; test window is the focus.
Model-win SKUs sorted first, then V1-win SKUs.
Output: outputs/reports/full_smooth_test_vs_v1.pdf
"""
import sys
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


def pick_weekly(uid_fcast: pd.DataFrame, model_name: str) -> np.ndarray:
    if model_name.startswith("Ensemble:"):
        cols = [c for c in model_name.replace("Ensemble:", "").split("+")
                if c in uid_fcast.columns]
        return uid_fcast[cols].mean(axis=1).values if cols else np.full(len(uid_fcast), np.nan)
    if model_name in uid_fcast.columns:
        return uid_fcast[model_name].values
    avail = [c for c in uid_fcast.columns if c not in {"unique_id", "ds"}]
    return uid_fcast[avail[0]].values if avail else np.full(len(uid_fcast), np.nan)


def main():
    weekly    = pd.read_parquet(PROCESSED_DIR / "sales_clean.parquet")
    profiles  = pd.read_csv(PROCESSED_DIR / "sku_profiles.csv")
    selection = pd.read_csv(OUTPUTS_REPORTS / "selection.csv")
    weekly["ds"] = pd.to_datetime(weekly["ds"])
    profiles["train_start"] = pd.to_datetime(profiles["train_start"])

    all_weeks  = sorted(weekly["ds"].unique())
    trimmed    = all_weeks[:-TRIM_TRAILING_WEEKS]
    test_start = pd.Timestamp(trimmed[-TEST_WEEKS])
    cutoff     = pd.Timestamp(trimmed[-(TEST_WEEKS + 1)])
    test_end   = pd.Timestamp(trimmed[-1])
    test_weeks = [w for w in trimmed if w >= test_start]

    full_uids = profiles.loc[
        (profiles["bucket"] == "smooth") & (profiles["history_length"] == "full"),
        "unique_id",
    ].tolist()
    print(f"{len(full_uids)} full-history smooth SKUs")
    print(f"Cutoff {cutoff.date()} | Test {test_start.date()} → {test_end.date()}")

    # ── Refit + predict ────────────────────────────────────────────────────────
    train_trimmed = _trim_to_train_start(
        weekly[weekly["ds"].isin(trimmed) & (weekly["ds"] < test_start)].copy(), profiles
    )
    train_g = train_trimmed[train_trimmed["unique_id"].isin(full_uids)].copy()

    candidates      = get_models("smooth", "full")
    candidate_names = {type(m).__name__ for m in candidates}
    baselines       = [b for b in get_baselines("smooth", "full")
                       if type(b).__name__ not in candidate_names]

    print("Fitting models...")
    fit_data = deseasonalize(train_g)
    sf = StatsForecast(models=candidates + baselines, freq=FREQUENCY, n_jobs=-1)
    sf.fit(fit_data[["unique_id", "ds", "y"]])
    fcast = sf.predict(h=TEST_WEEKS)
    fcast["ds"] = pd.to_datetime(fcast["ds"])
    fcast = reseasonalize(fcast)
    print("Done.")

    sel_map = selection.set_index("unique_id")["model"].to_dict()

    # ── V1 ─────────────────────────────────────────────────────────────────────
    raw = pd.read_parquet(PROCESSED_DIR / "orders_raw.parquet")
    raw["order_date"] = pd.to_datetime(raw["order_date"])
    index = build_cumsum_index(raw)
    v1_totals = {}
    for uid in full_uids:
        try:
            v1_totals[uid] = v1_forecast(index, uid, cutoff)
        except Exception:
            pass
    print(f"V1: {len(v1_totals)}/{len(full_uids)} SKUs")

    # ── Actuals ────────────────────────────────────────────────────────────────
    test_df       = weekly[weekly["ds"].isin(test_weeks) & weekly["unique_id"].isin(full_uids)]
    actual_totals = test_df.groupby("unique_id")["y"].sum()

    # ── Build summary table ────────────────────────────────────────────────────
    rows = []
    for uid in full_uids:
        uid_f  = fcast[fcast["unique_id"] == uid].sort_values("ds")
        mn     = sel_map.get(uid, "")
        our_fc = float(pd.Series(pick_weekly(uid_f, mn)).sum())
        actual = float(actual_totals.get(uid, np.nan))
        v1     = v1_totals.get(uid, np.nan)
        ae_o   = abs(actual - our_fc)
        ae_v   = abs(actual - v1) if not np.isnan(v1) else np.nan
        wins   = ae_o < ae_v if not np.isnan(ae_v) else None
        rows.append({"uid": uid, "actual": actual, "our": our_fc, "v1": v1,
                     "ae_ours": ae_o, "ae_v1": ae_v, "wins": wins})

    df_sum = pd.DataFrame(rows)
    n_wins = int(df_sum["wins"].sum())
    print(f"\nModel wins: {n_wins}/{len(df_sum)}")
    print(f"MAE — Ours: {df_sum['ae_ours'].mean():.2f}   V1: {df_sum['ae_v1'].mean():.2f}")

    # Sort: model-wins descending by actual demand, then V1-wins
    sorted_uids = (
        df_sum
        .assign(win_sort=df_sum["wins"].map({True: 0, False: 1, None: 2}))
        .sort_values(["win_sort", "actual"], ascending=[True, False])
        ["uid"].tolist()
    )

    # ── Plot ───────────────────────────────────────────────────────────────────
    out_path = OUTPUTS_REPORTS / "full_smooth_test_vs_v1.pdf"
    n_pages  = (len(sorted_uids) + SKUS_PER_PAGE - 1) // SKUS_PER_PAGE

    with pdf_backend.PdfPages(out_path) as pdf:
        for page_start in range(0, len(sorted_uids), SKUS_PER_PAGE):
            page_uids = sorted_uids[page_start:page_start + SKUS_PER_PAGE]
            fig, axes = plt.subplots(3, 2, figsize=(16, 12))
            axes = axes.flatten()

            for ax_i, uid in enumerate(page_uids):
                ax  = axes[ax_i]
                row = df_sum[df_sum["uid"] == uid].iloc[0]

                # Context (dimmed)
                hist = weekly[weekly["unique_id"] == uid].sort_values("ds").set_index("ds")["y"]
                ctx  = hist[
                    (hist.index >= test_start - pd.Timedelta(weeks=CONTEXT_WEEKS)) &
                    (hist.index < test_start)
                ]

                # Test actuals
                t_actual = test_df[test_df["unique_id"] == uid].sort_values("ds").set_index("ds")["y"]

                # Model weekly
                uid_f      = fcast[fcast["unique_id"] == uid].sort_values("ds").set_index("ds")
                mn         = sel_map.get(uid, "")
                our_weekly = pd.Series(pick_weekly(uid_f.reset_index(), mn), index=uid_f.index)

                # V1 flat
                v1_total  = row["v1"]
                v1_weekly = v1_total / TEST_WEEKS if not np.isnan(v1_total) else np.nan

                wins      = row["wins"]
                win_color = "#2e7d32" if wins else ("#c62828" if wins is False else "#777")
                win_str   = "✓ model" if wins else ("✗ V1 wins" if wins is False else "—")

                # Context line
                if not ctx.empty:
                    ax.plot(ctx.index, ctx.values, color="#ccc", lw=1.1, zorder=1)
                ax.axvline(test_start, color="#aaa", lw=0.8, ls="--", zorder=2)

                # Test lines
                ax.plot(t_actual.index, t_actual.values,
                        color="#1565C0", lw=2, marker="o", ms=5,
                        label=f"Actual   {row['actual']:.0f}", zorder=5)
                ax.plot(our_weekly.index, our_weekly.values,
                        color="#2E7D32", lw=1.8, ls="--", marker="s", ms=4,
                        label=f"Model    {row['our']:.0f}  (Δ{row['our']-row['actual']:+.0f})", zorder=4)
                if not np.isnan(v1_weekly):
                    v1_ser = pd.Series(v1_weekly, index=t_actual.index)
                    ax.plot(v1_ser.index, v1_ser.values,
                            color="#E53935", lw=1.6, ls=":", marker="^", ms=4,
                            label=f"V1       {v1_total:.0f}  (Δ{row['v1']-row['actual']:+.0f})", zorder=4)

                ax.axvspan(test_start, test_end, alpha=0.07, color="#1565C0", zorder=0)

                # Y zoom: test-period range + context, small padding
                all_vals = (list(t_actual.values) + list(our_weekly.dropna().values) +
                            ([v1_weekly] * TEST_WEEKS if not np.isnan(v1_weekly) else []))
                if not ctx.empty:
                    all_vals += list(ctx.values)
                ymin = max(0, min(all_vals) * 0.80)
                ymax = max(all_vals) * 1.22
                ax.set_ylim(ymin, ymax)

                ax.set_title(f"{uid}   [{mn}]   {win_str}",
                             fontsize=7.5, color=win_color, pad=3)
                ax.legend(fontsize=7.5, loc="upper left", framealpha=0.85,
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
                f"Full-history smooth SKUs — test vs V1   "
                f"cutoff {cutoff.date()} | test {test_start.date()}–{test_end.date()}   "
                f"Model wins: {n_wins}/{len(sorted_uids)}   "
                f"Page {page_start // SKUS_PER_PAGE + 1}/{n_pages}",
                fontsize=9, y=1.005,
            )
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"\nSaved → {out_path}")
    print(f"\nV1 wins on:")
    v1w = df_sum[df_sum["wins"] == False].sort_values("ae_v1")
    print(v1w[["uid", "actual", "our", "v1", "ae_ours", "ae_v1"]].to_string(index=False))


if __name__ == "__main__":
    main()
