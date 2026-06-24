#!/usr/bin/env python3
"""
PDF report: conformal prediction intervals for all 51 full-history smooth SKUs.
Page 1: Summary charts (4 charts, scatter plots have covered/missed legends)
Page 2: Full SKU table (all 51 rows, readable)
Pages 3+: 6 SKUs per page (2×3 grid)
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

PROC = ROOT / "data/processed"
REP  = ROOT / "outputs/reports"

CONFORMAL_LEVEL = 70
TRAIN_SHOW      = 30
SKUS_PER_PAGE   = 6


def load_data():
    weekly   = pd.read_parquet(PROC / "sales_clean.parquet")
    test_set = pd.read_parquet(REP  / "test_set.parquet")
    results  = pd.read_csv(REP / "test_evaluation.csv")
    weekly["ds"]   = pd.to_datetime(weekly["ds"])
    test_set["ds"] = pd.to_datetime(test_set["ds"])
    return weekly, test_set, results


def sku_fig(skus_batch, weekly, test_set, results, level):
    n     = len(skus_batch)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 4.2))
    axes = np.array(axes).flatten()

    lo_col = "yhat_lo_90"
    hi_col = "yhat_hi_90"

    for ax, uid in zip(axes, skus_batch):
        row   = results[results["unique_id"] == uid].iloc[0]
        model = row["selected_model"]

        hist = weekly[weekly["unique_id"] == uid].sort_values("ds").tail(TRAIN_SHOW + 10)
        act  = test_set[test_set["unique_id"] == uid].sort_values("ds")
        test_dates  = act["ds"].values
        test_actual = act["y"].values

        cutoff     = pd.Timestamp(hist[hist["ds"] < test_dates[0]]["ds"].max())
        train_hist = hist[hist["ds"] <= cutoff].tail(TRAIN_SHOW)

        h      = len(test_dates)
        pt_wk  = row["yhat_total"] / h
        has_pi = pd.notna(row.get(lo_col)) and pd.notna(row.get(hi_col))

        missed = (has_pi and (
            row["actual_total"] < row[lo_col] or
            row["actual_total"] > row[hi_col]
        ))
        cov_color = "#C44E52" if missed else "#55A868"

        # ── cumulative training history ───────────────────────────────────
        cum_train = np.cumsum(train_hist["y"].values)
        ax.plot(train_hist["ds"].values, cum_train, color="#4C72B0", lw=1.4)

        # ── cumulative test period, continuing from end of training ───────
        base        = cum_train[-1]
        t_idx       = np.arange(1, h + 1)
        cum_actual  = base + np.cumsum(test_actual)
        cum_fcst    = base + pt_wk * t_idx

        # prepend the cutoff anchor so lines connect without a gap
        anchor_date = train_hist["ds"].values[-1:]
        plot_dates  = np.concatenate([anchor_date, test_dates])

        ax.axvspan(test_dates[0], test_dates[-1], alpha=0.05, color="orange")

        if has_pi:
            cum_lo = base + row[lo_col] * t_idx / h
            cum_hi = base + row[hi_col] * t_idx / h
            ax.fill_between(plot_dates,
                            np.concatenate([[base], cum_lo]),
                            np.concatenate([[base], cum_hi]),
                            alpha=0.25, color="orange")
            ax.plot(plot_dates, np.concatenate([[base], cum_lo]),
                    color="darkorange", lw=0.8, ls=":")
            ax.plot(plot_dates, np.concatenate([[base], cum_hi]),
                    color="darkorange", lw=0.8, ls=":")

        ax.plot(plot_dates, np.concatenate([[base], cum_fcst]),
                color="#DD8452", lw=1.5, ls="--")
        ax.plot(plot_dates, np.concatenate([[base], cum_actual]),
                color=cov_color, lw=2, zorder=6)

        # ── covered/missed annotation ─────────────────────────────────────
        cov_txt = "✗ MISSED" if missed else "✓ covered"
        ax.annotate(cov_txt, xy=(0.98, 0.97), xycoords="axes fraction",
                    ha="right", va="top", fontsize=7.5,
                    color=cov_color, fontweight="bold")

        if has_pi:
            band = f"  [{row[lo_col]:.0f}–{row[hi_col]:.0f}]"
        else:
            band = ""
        sub = (f"Actual {row['actual_total']:.0f}  |  Fcst {row['yhat_total']:.0f}"
               f"{band}  |  {model[:36]}")

        ax.set_title(f"{uid}", fontsize=8.5, fontweight="bold", pad=3)
        ax.set_xlabel(sub, fontsize=6.5, labelpad=3)
        ax.tick_params(axis="x", labelsize=6, rotation=28)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_ylim(bottom=0)

    # hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    handles = [
        plt.Line2D([0], [0], color="#4C72B0", lw=1.4, label="Cumulative history"),
        plt.Line2D([0], [0], color="#DD8452", lw=1.5, ls="--", label="Cumulative forecast"),
        mpatches.Patch(color="orange", alpha=0.3, label=f"P{level} cumulative interval"),
        plt.Line2D([0], [0], color="#55A868", lw=2, label="Cumulative actual — covered"),
        plt.Line2D([0], [0], color="#C44E52", lw=2, label="Cumulative actual — missed"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7.5,
               frameon=True, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    return fig


def summary_fig(results, level):
    fs = results[
        (results["bucket"] == "smooth") &
        (results["history_length"] == "full")
    ].copy()
    lo_col, hi_col = "yhat_lo_90", "yhat_hi_90"
    fs["has_pi"] = fs[hi_col].notna()
    pi = fs[fs["has_pi"]].copy()
    pi["covered"] = (pi["actual_total"] >= pi[lo_col]) & (pi["actual_total"] <= pi[hi_col])
    pi["width"]   = pi[hi_col] - pi[lo_col]
    pi["bias"]    = pi["yhat_total"] - pi["actual_total"]
    pi["rel_err"] = pi["bias"] / pi["actual_total"].clip(lower=1)

    n_covered = pi["covered"].sum()
    n_total   = len(pi)
    avg_width = pi["width"].mean()
    avg_lo    = pi[lo_col].mean()
    avg_hi    = pi[hi_col].mean()

    fig = plt.figure(figsize=(18, 9))
    fig.suptitle(
        f"Full-History Smooth SKUs — Conformal Prediction Intervals (P{level})\n"
        f"10-week held-out test window  |  n_windows=5  |  "
        f"Coverage: {n_covered}/{n_total} ({n_covered/n_total*100:.0f}%)  |  "
        f"Avg lower: {avg_lo:.0f}  |  Avg upper: {avg_hi:.0f}  |  Avg width: {avg_width:.0f} units",
        fontsize=11, y=0.98,
    )

    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35,
                          top=0.88, bottom=0.08, left=0.07, right=0.97)

    cov_patch  = mpatches.Patch(color="#55A868", label="Covered")
    miss_patch = mpatches.Patch(color="#C44E52", label="Missed")
    colors = pi["covered"].map({True: "#55A868", False: "#C44E52"})

    # ── 1. Actual vs Forecast scatter ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(pi["actual_total"], pi["yhat_total"], c=colors, s=40, alpha=0.8)
    lim = max(pi["actual_total"].max(), pi["yhat_total"].max()) * 1.05
    ax1.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.4)
    ax1.set_xlabel("Actual (10-wk total)", fontsize=8)
    ax1.set_ylabel("Forecast", fontsize=8)
    ax1.set_title("Actual vs Forecast", fontsize=9)
    ax1.tick_params(labelsize=7)
    ax1.legend(handles=[cov_patch, miss_patch], fontsize=7, loc="upper left")

    # ── 2. Interval width vs actual demand ────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(pi["actual_total"], pi["width"], c=colors, s=40, alpha=0.8)
    ax2.set_xlabel("Actual demand (10-wk)", fontsize=8)
    ax2.set_ylabel("Interval width (units)", fontsize=8)
    ax2.set_title("Interval Width vs Demand", fontsize=9)
    ax2.tick_params(labelsize=7)
    ax2.legend(handles=[cov_patch, miss_patch], fontsize=7, loc="upper left")

    # ── 3. Relative bias distribution ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(pi["rel_err"] * 100, bins=20, color="#4C72B0", edgecolor="white", lw=0.4)
    ax3.axvline(0, color="k", lw=1, ls="--", alpha=0.5)
    ax3.axvline(pi["rel_err"].mean() * 100, color="#DD8452", lw=1.5,
                label=f"Mean {pi['rel_err'].mean()*100:+.1f}%")
    ax3.set_xlabel("Forecast bias (%)", fontsize=8)
    ax3.set_ylabel("SKU count", fontsize=8)
    ax3.set_title("Bias Distribution", fontsize=9)
    ax3.legend(fontsize=7)
    ax3.tick_params(labelsize=7)

    # ── 4. Coverage by model ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    model_cov = (
        pi.groupby("selected_model")
        .agg(n=("covered", "count"), covered=("covered", "sum"))
        .assign(pct=lambda d: d["covered"] / d["n"] * 100)
        .sort_values("pct")
    )
    bar_colors = ["#C44E52" if p < 80 else "#55A868" for p in model_cov["pct"]]
    ax4.barh(range(len(model_cov)), model_cov["pct"], color=bar_colors)
    ax4.set_yticks(range(len(model_cov)))
    ax4.set_yticklabels([m[:30] for m in model_cov.index], fontsize=6)
    ax4.axvline(84, color="k", lw=1, ls="--", alpha=0.5, label="84% overall")
    for i, (_, r) in enumerate(model_cov.iterrows()):
        ax4.text(r["pct"] + 0.5, i, f"{r['covered']:.0f}/{r['n']:.0f}",
                 va="center", fontsize=6)
    ax4.set_xlabel("Coverage (%)", fontsize=8)
    ax4.set_title("Coverage by Model", fontsize=9)
    ax4.set_xlim(0, 115)
    ax4.tick_params(axis="x", labelsize=7)
    ax4.legend(fontsize=7)

    return fig


def table_fig(results, level):
    """Dedicated full page for the per-SKU interval table."""
    fs = results[
        (results["bucket"] == "smooth") &
        (results["history_length"] == "full")
    ].copy()
    lo_col, hi_col = "yhat_lo_90", "yhat_hi_90"
    fs["has_pi"] = fs[hi_col].notna()
    pi = fs[fs["has_pi"]].copy()
    pi["covered"] = (pi["actual_total"] >= pi[lo_col]) & (pi["actual_total"] <= pi[hi_col])
    pi["bias"]    = pi["yhat_total"] - pi["actual_total"]

    table_df = pi.sort_values("bias")[
        ["unique_id", "actual_total", "yhat_total", lo_col, hi_col, "covered", "selected_model"]
    ].copy()
    table_df.columns = ["SKU", "Actual", "Forecast", "Lo", "Hi", "Covered", "Model"]
    table_df["Model"]    = table_df["Model"].str[:32]
    table_df["Covered"]  = table_df["Covered"].map({True: "✓", False: "✗"})
    table_df["Actual"]   = table_df["Actual"].map("{:.0f}".format)
    table_df["Forecast"] = table_df["Forecast"].map("{:.0f}".format)
    table_df["Lo"]       = table_df["Lo"].map("{:.0f}".format)
    table_df["Hi"]       = table_df["Hi"].map("{:.0f}".format)

    n_rows = len(table_df)
    fig_h  = max(11, n_rows * 0.33 + 2.5)
    fig = plt.figure(figsize=(18, fig_h))
    fig.suptitle(
        f"Full-History Smooth SKUs — All {n_rows} SKUs with P{level} Intervals\n"
        "Sorted by forecast bias: most under-forecast (left) → most over-forecast (right)",
        fontsize=11, y=0.99,
    )

    ax = fig.add_axes([0.03, 0.02, 0.94, 0.93])
    ax.axis("off")

    tbl = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center", loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    col_widths = [0.22, 0.09, 0.10, 0.08, 0.08, 0.09, 0.34]
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.4)
        cell.set_width(col_widths[c])
        if r == 0:
            cell.set_facecolor("#CCCCCC")
            cell.set_text_props(fontweight="bold", fontsize=9)
        elif table_df.iloc[r - 1]["Covered"] == "✗":
            cell.set_facecolor("#FFE0E0")
        elif r % 2 == 0:
            cell.set_facecolor("#F5F5F5")

    return fig


def main():
    weekly, test_set, results = load_data()

    fs = results[
        (results["bucket"] == "smooth") &
        (results["history_length"] == "full")
    ].copy()
    # Sort by actual demand descending so high-volume SKUs appear first
    fs = fs.sort_values("actual_total", ascending=False)
    uids = fs["unique_id"].tolist()

    out_path = REP / f"full_smooth_pi_level{CONFORMAL_LEVEL}.pdf"
    with PdfPages(out_path) as pdf:
        # Page 1 — summary charts
        fig_s = summary_fig(results, CONFORMAL_LEVEL)
        pdf.savefig(fig_s, bbox_inches="tight")
        plt.close(fig_s)

        # Page 2 — full SKU table
        fig_t = table_fig(results, CONFORMAL_LEVEL)
        pdf.savefig(fig_t, bbox_inches="tight")
        plt.close(fig_t)

        # SKU pages
        for i in range(0, len(uids), SKUS_PER_PAGE):
            batch = uids[i:i + SKUS_PER_PAGE]
            page  = (i // SKUS_PER_PAGE) + 1
            total_pages = (len(uids) + SKUS_PER_PAGE - 1) // SKUS_PER_PAGE
            print(f"  Page {page}/{total_pages}: {batch[0]} … {batch[-1]}")
            fig = sku_fig(batch, weekly, test_set, results, CONFORMAL_LEVEL)
            fig.suptitle(
                f"Full-History Smooth SKUs — P{CONFORMAL_LEVEL} Conformal Intervals  "
                f"(Page {page}/{total_pages})",
                fontsize=9, y=1.01,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
