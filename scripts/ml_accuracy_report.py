#!/usr/bin/env python3
"""Write the precomputed accuracy comparisons the dashboard's Forecast
Accuracy page reads: the current-best served model vs V1, on the same
development windows and segments. Retrains on each of the three dev-split
windows (via validate_version_detail / validate_v1_detail), so this is a
separate, slower cadence from scripts/ml_forward_forecast.py -- refresh it
when the served model version changes, not on every forward run. Never
touches the quarantined final test window (src.ml.dataset.dev_splits
excludes it by construction).

Trains once per version (not once per output file): the per-SKU detail is
computed first, then aggregated locally with the same segment rollup
validate_version()/validate_v1() use (src.ml.evaluate.aggregate_by_segment),
so both files are consistent by construction without a second training pass.

Run:
  .venv/bin/python scripts/ml_accuracy_report.py

Output:
  outputs/reports/ml_accuracy.csv         one row per (model_version, window, segment)
  outputs/reports/ml_accuracy_by_sku.csv  one row per (model_version, window, unique_id)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ml.dataset import load_weekly  # noqa: E402
from src.ml.evaluate import aggregate_by_segment  # noqa: E402
from src.ml.serving import CURRENT_BEST, validate_version_detail  # noqa: E402
from src.ml.serving.v1 import validate_v1_detail  # noqa: E402

SUMMARY_PATH = ROOT / "outputs" / "reports" / "ml_accuracy.csv"
DETAIL_PATH = ROOT / "outputs" / "reports" / "ml_accuracy_by_sku.csv"


def _summarize(detail: pd.DataFrame) -> pd.DataFrame:
    """One row per (model_version, window, segment), matching
    validate_version()'s shape, built from already-computed detail."""
    parts = []
    for (mv, window, cutoff), g in detail.groupby(
        ["model_version", "window", "cutoff"], sort=False
    ):
        agg = aggregate_by_segment(g)
        agg.insert(0, "cutoff", cutoff)
        agg.insert(0, "window", window)
        agg.insert(0, "model_version", mv)
        parts.append(agg)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    weekly, profiles = load_weekly()

    model_detail = validate_version_detail(CURRENT_BEST, weekly=weekly, profiles=profiles)
    v1_detail = validate_v1_detail(weekly=weekly, profiles=profiles)
    detail = pd.concat([model_detail, v1_detail], ignore_index=True)

    summary = _summarize(detail)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    detail.to_csv(DETAIL_PATH, index=False)

    print(f"wrote {len(summary)} rows -> {SUMMARY_PATH.relative_to(ROOT)}")
    print(f"wrote {len(detail)} rows -> {DETAIL_PATH.relative_to(ROOT)}")
    print()
    print(summary.pivot_table(index=["window", "segment"], columns="model_version",
                               values="pooled_wape").to_string())


if __name__ == "__main__":
    main()
