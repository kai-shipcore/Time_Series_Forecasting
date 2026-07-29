#!/usr/bin/env python3
"""Export per-week backtest predictions for the dashboard's backtest chart.

The stored accuracy report (outputs/reports/ml_accuracy_by_sku.csv) keeps one
predicted TOTAL per SKU per window. That is enough to say how large a miss was,
but not when inside the window it happened, so the dashboard could only draw the
prediction as a flat level. This writes the per-week rows behind those totals.

The numbers come from src.ml.serving.forecast.validate_version_weekly, which
reuses the same fit/predict loop and the same eligibility filter as the recorded
totals. This script verifies that equality before writing: summing yhat and y per
(unique_id, window) must reproduce the stored accuracy report exactly. If it does
not, the export is refused rather than written, because a backtest chart that
disagrees with the accuracy figure beside it is worse than no chart.

Examples:
  .venv/bin/python scripts/ml_backtest_weekly.py
  .venv/bin/python scripts/ml_backtest_weekly.py --version v11 --windows 3

Output:
  outputs/reports/ml_backtest_weekly.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.ml.serving.forecast import validate_version_weekly  # noqa: E402
from src.ml.serving.models import CURRENT_BEST  # noqa: E402

TOL = 1e-6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default=CURRENT_BEST)
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--out", default="outputs/reports/ml_backtest_weekly.csv")
    ap.add_argument("--reference", default="outputs/reports/ml_accuracy_by_sku.csv",
                    help="stored totals to reconcile against")
    ap.add_argument("--skip-check", action="store_true",
                    help="write without reconciling (not recommended)")
    args = ap.parse_args()

    print(f"version   {args.version}")
    print(f"windows   {args.windows}")
    print("running the development windows, this refits per window...\n")

    wk = validate_version_weekly(args.version, n_windows=args.windows)
    print(f"rows      {len(wk):,}")
    print(f"SKUs      {wk['unique_id'].nunique()}")
    print(f"windows   {sorted(wk['window'].unique())}")
    print(f"leads     {wk['lead'].min()}..{wk['lead'].max()}")

    ref_path = ROOT / args.reference
    if not args.skip_check and ref_path.exists():
        ref = pd.read_csv(ref_path)
        ref = ref[ref["model_version"] == args.version]
        got = (wk.groupby(["unique_id", "window"], as_index=False)
                 .agg(yhat_total=("yhat", "sum"), y_total=("y", "sum")))
        cmp = ref.merge(got, on=["unique_id", "window"], how="outer",
                        suffixes=("_ref", "_new"), indicator=True)

        only_ref = int((cmp["_merge"] == "left_only").sum())
        only_new = int((cmp["_merge"] == "right_only").sum())
        both = cmp[cmp["_merge"] == "both"].copy()
        both["d_yhat"] = (both["yhat_total_ref"] - both["yhat_total_new"]).abs()
        both["d_y"] = (both["y_total_ref"] - both["y_total_new"]).abs()
        bad = both[(both["d_yhat"] > TOL) | (both["d_y"] > TOL)]

        print("\n--- reconciliation against the stored totals ---")
        print(f"matched (unique_id, window) pairs   {len(both):,}")
        print(f"only in stored report               {only_ref}")
        print(f"only in this export                 {only_new}")
        print(f"pairs disagreeing beyond {TOL:g}       {len(bad)}")
        if len(bad):
            print("\nworst disagreements:")
            print(bad.nlargest(5, "d_yhat")[
                ["unique_id", "window", "yhat_total_ref", "yhat_total_new",
                 "y_total_ref", "y_total_new"]].to_string(index=False))
        if len(bad) or only_ref or only_new:
            print("\nREFUSED: this export does not reproduce the stored accuracy "
                  "report. Not writing. Investigate before using it, since the "
                  "chart would contradict the reliability figure beside it.")
            sys.exit(1)
        print("OK: totals reproduce the stored report exactly.")
    elif not ref_path.exists():
        print(f"\n(no reference at {args.reference}, skipping reconciliation)")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    wk.to_csv(out, index=False)
    print(f"\nwrote     {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
