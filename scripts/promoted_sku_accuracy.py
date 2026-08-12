#!/usr/bin/env python3
"""Measure how accurately the model forecasts SKUs promoted from intermittent.

Produces the number behind `PROMOTED_ERROR_FALLBACK` in dashboard/lib/calc.py,
which is the safety-stock error assumed for a promoted SKU that has no measured
error of its own.

Why this needs a script rather than a query. A promoted SKU cannot be identified
from the current profile snapshot at an old cutoff: `src/profile.py` rewrites
`train_start` and `active_weeks` on promotion, and those values move forward
every run (docs/BACKLOG.md item 2). So the profiler is re-run against sales data
truncated to each backtest cutoff, which recovers who was promoted at the time,
and that is joined to the per-SKU scores already recorded.

No model is refitted. This reads outputs/reports/ml_accuracy_by_sku.csv, so it
reports on whatever version is stored there.

Run:
    .venv/bin/python scripts/promoted_sku_accuracy.py
"""

import argparse
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import src.profile as P  # noqa: E402
from src.ml.serving.models import CURRENT_BEST  # noqa: E402

# profile() writes sku_profiles.csv into its PROCESSED_DIR as a side effect.
# Redirected to a temporary directory so that running this analysis can never
# overwrite the pinned snapshot the whole project is measured against.
P.PROCESSED_DIR = pathlib.Path(tempfile.mkdtemp())

CUTOFFS = {"Oct-Dec": "2025-10-06", "Dec-Feb": "2025-12-15", "Mar-May": "2026-02-23"}


def pooled(x: pd.DataFrame) -> float:
    return x["ae"].sum() / x["y_total"].sum() if x["y_total"].sum() else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # Defaults to whatever is currently served rather than a pinned string, so
    # this keeps measuring the right model after the next version lands.
    ap.add_argument("--version", default=CURRENT_BEST)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    sales = pd.read_parquet(ROOT / "data/processed/sales_clean.parquet")
    acc = pd.read_csv(ROOT / "outputs/reports/ml_accuracy_by_sku.csv")
    acc = acc[acc["model_version"] == args.version]

    parts = []
    for window, cut in CUTOFFS.items():
        prof = P.profile(sales[sales["ds"] <= pd.Timestamp(cut)].copy())
        # profile() now writes an explicit `promoted` column. The old test,
        # active_weeks == RECENT_WEEKS, only worked while promotion assigned that
        # constant to every promoted SKU; since onset detection landed it
        # identifies roughly one in ten of them. Falls back to the old test for a
        # profile frame produced by older code.
        if "promoted" not in prof.columns:
            prof["promoted"] = (
                prof["active_weeks"].eq(P.RECENT_WEEKS) & prof["bucket"].eq("smooth")
            )
        a = acc[acc["window"] == window].merge(
            prof[["unique_id", "promoted"]], on="unique_id", how="left"
        )
        a["window"] = window
        parts.append(a)

    d = pd.concat(parts, ignore_index=True)
    short = d[d["history_length"] == "short"]

    print(f"\nmodel version: {args.version}")
    print("smooth/short pooled WAPE, promoted vs the rest\n")
    print(f"{'window':<10}{'promoted n':>12}{'promoted':>11}{'other n':>10}{'other':>10}")
    for window in CUTOFFS:
        g = short[short["window"] == window]
        p, o = g[g["promoted"]], g[~g["promoted"].fillna(False)]
        pw = pooled(p) if len(p) else float("nan")
        print(f"{window:<10}{len(p):>12}{pw:>11.4f}{len(o):>10}{pooled(o):>10.4f}")

    p, o = short[short["promoted"]], short[~short["promoted"].fillna(False)]
    print(f"{'ALL':<10}{len(p):>12}{pooled(p):>11.4f}{len(o):>10}{pooled(o):>10.4f}")
    print(f"\n  -> PROMOTED_ERROR_FALLBACK = {pooled(p):.4f}  "
          f"(dashboard/lib/calc.py)")
    print(f"  promoted share of scored short-segment units: "
          f"{p['y_total'].sum() / short['y_total'].sum():.1%}")
    print(f"  median per-SKU WAPE: promoted {(p['ae'] / p['y_total']).median():.3f}, "
          f"other {(o['ae'] / o['y_total']).median():.3f}")

    rng = np.random.default_rng(0)

    def boot(x):
        idx = rng.integers(0, len(x), (args.boot, len(x)))
        ae = x["ae"].to_numpy()[idx].sum(1)
        y = x["y_total"].to_numpy()[idx].sum(1)
        return ae / np.where(y == 0, np.nan, y)

    print("\n  bootstrap, promoted minus other:")
    for window in CUTOFFS:
        g = short[short["window"] == window]
        gp, go = g[g["promoted"]], g[~g["promoted"].fillna(False)]
        if len(gp) < 2 or len(go) < 2:
            print(f"    {window:<10} too few SKUs")
            continue
        diff = boot(gp) - boot(go)
        m, se = np.nanmean(diff), np.nanstd(diff)
        verdict = "distinguishable" if abs(m) > 2 * se else "within noise"
        print(f"    {window:<10}{m:>+9.4f}  se {se:.4f}   {verdict}")


if __name__ == "__main__":
    main()
