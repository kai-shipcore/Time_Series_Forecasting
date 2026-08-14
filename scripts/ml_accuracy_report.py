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

When to refresh it, which the line above got wrong by omission. The inputs are
`load_weekly()` with its default argument, so both the actuals and the SKU
profiles come from the pinned snapshot named by `config.ML_DATA_SNAPSHOT`. That
has two consequences worth stating here, because the absence of the second one
cost two weeks:

- Running this weekly is pointless. The inputs do not move with the weekly cron,
  so a second run against an unchanged snapshot rewrites identical bytes after
  retraining three windows.
- Running it is required whenever the snapshot is re-cut, because a re-cut
  snapshot carries a re-profiled population. The report is a measurement of a
  model over a cohort, and changing the cohort changes the measurement even
  though the model and the weeks are untouched.

The 2026-08-11 re-profile into `2026-08-03-v2` moved smooth/short from 382 SKUs
to 247 and nothing re-ran this, so the page served figures for a cohort that no
longer existed. `ml_accuracy_meta.json` exists so that condition is detectable
rather than something a reader has to know to look for.

Run:
  .venv/bin/python scripts/ml_accuracy_report.py

Output:
  outputs/reports/ml_accuracy.csv         one row per (model_version, window, segment)
  outputs/reports/ml_accuracy_by_sku.csv  one row per (model_version, window, unique_id)
  outputs/reports/ml_accuracy_meta.json   what these were measured on, and when
"""
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from config import ML_DATA_SNAPSHOT, ML_FINAL_TEST_CUTOFF  # noqa: E402
from src.ml.dataset import load_weekly  # noqa: E402
from src.ml.evaluate import aggregate_by_segment  # noqa: E402
from src.ml.serving import CURRENT_BEST, validate_version_detail  # noqa: E402
from src.ml.serving.v1 import validate_v1_detail  # noqa: E402

SUMMARY_PATH = ROOT / "outputs" / "reports" / "ml_accuracy.csv"
DETAIL_PATH = ROOT / "outputs" / "reports" / "ml_accuracy_by_sku.csv"
#: Provenance for the two CSVs beside it. Read by api/main.py, which used the
#: summary file's mtime before this existed. An mtime is a property of the
#: filesystem rather than of the measurement: `git checkout`, `cp -r` and a
#: deploy all rewrite it, so the page dated the report by when the file last
#: moved rather than by when the figures were computed.
META_PATH = ROOT / "outputs" / "reports" / "ml_accuracy_meta.json"


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


def _commit() -> str | None:
    """The revision these figures are attributable to, or None outside a
    checkout. Not a hard requirement the way it is in ml_41_final_test.py: that
    script spends a single-use window and must be reproducible, this one can be
    re-run at will."""
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except Exception:
        return None


def _population(profiles: pd.DataFrame) -> dict:
    """The cohort these figures describe, counted the way the profiler labels it.

    This is the field the API compares against live profiling. The comparison it
    enables is the one nothing was making: whether the population the model was
    measured on is still the population it is being served for. Those two
    drifted apart for two weeks in August 2026 with every number on the page
    reading as though they had not.
    """
    if profiles.empty or "bucket" not in profiles.columns:
        return {"total": int(len(profiles)), "segments": {}}
    hist = profiles.get("history_length", pd.Series("?", index=profiles.index))
    key = profiles["bucket"].fillna("?") + "/" + hist.fillna("?")
    return {
        "total": int(len(profiles)),
        "segments": {str(k): int(v) for k, v in key.value_counts().sort_index().items()},
    }


def _windows(detail: pd.DataFrame, model_version: str) -> list[dict]:
    """Per-window scored counts, from the model's own rows.

    Taken from the model rather than from V1 because the two are scored on the
    same SKU-window pairs by construction, and if they ever are not, the model's
    population is the one the page reports.
    """
    d = detail[detail["model_version"] == model_version]
    out = []
    for (window, cutoff), g in d.groupby(["window", "cutoff"], sort=False):
        out.append({
            "window": str(window),
            "cutoff": str(pd.Timestamp(cutoff).date()),
            "n_skus": int(g["unique_id"].nunique()),
        })
    return sorted(out, key=lambda w: w["cutoff"])


def main() -> None:
    weekly, profiles = load_weekly()

    model_detail = validate_version_detail(CURRENT_BEST, weekly=weekly, profiles=profiles)
    v1_detail = validate_v1_detail(weekly=weekly, profiles=profiles)
    detail = pd.concat([model_detail, v1_detail], ignore_index=True)

    summary = _summarize(detail)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    detail.to_csv(DETAIL_PATH, index=False)

    # Written last, and only after both CSVs are on disk, so the manifest cannot
    # describe a report that failed halfway. A missing manifest beside present
    # CSVs is read by the API as "pre-manifest checkout"; a manifest describing
    # absent CSVs would be a state nothing could interpret.
    meta = {
        "run_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": _commit(),
        "snapshot": ML_DATA_SNAPSHOT,
        "final_test_cutoff": ML_FINAL_TEST_CUTOFF,
        "model_version": CURRENT_BEST,
        "versions": sorted(detail["model_version"].unique().tolist()),
        "windows": _windows(detail, CURRENT_BEST),
        "scored_skus": int(detail.loc[detail["model_version"] == CURRENT_BEST,
                                      "unique_id"].nunique()),
        "population": _population(profiles),
    }
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"wrote {len(summary)} rows -> {SUMMARY_PATH.relative_to(ROOT)}")
    print(f"wrote {len(detail)} rows -> {DETAIL_PATH.relative_to(ROOT)}")
    print(f"wrote provenance      -> {META_PATH.relative_to(ROOT)}")
    print(f"  snapshot {ML_DATA_SNAPSHOT} | {meta['scored_skus']} SKUs scored | "
          f"population {meta['population']['segments']}")
    print()
    print(summary.pivot_table(index=["window", "segment"], columns="model_version",
                               values="pooled_wape").to_string())


if __name__ == "__main__":
    main()
