# ML Stage 1: load data and build train/test splits.
#
# Design decisions (v0):
# - SAME inputs as the statistical pipeline (sales_clean.parquet +
#   sku_profiles.csv). We never re-ingest or re-aggregate — if the two model
#   families ever disagree, it must be the model, not the data.
# - SAME split conventions as src/backtest.py:
#     * drop TRIM_TRAILING_WEEKS from the tail (noisy pending orders)
#     * hold out the last TEST_WEEKS complete weeks as the test window
#     * trim ramp-up SKUs to their train_start (pre-launch zeros are not
#       demand history and would poison lag features)
# - Rolling-origin ready: make_splits(n_splits=3) gives 3 cutoffs stepping
#   back `horizon` weeks each time, so results never hinge on one lucky
#   window. v0 experiments use n_splits=1 == the existing test window.
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    DATA_PROCESSED,
    DATA_SNAPSHOTS,
    MEDIUM_HISTORY_WEEKS,
    MIN_SIM_HISTORY_WEEKS,
    ML_DATA_SNAPSHOT,
    ML_FINAL_TEST_CUTOFF,
    SHORT_HISTORY_WEEKS,
    TEST_WEEKS,
    TRIM_TRAILING_WEEKS,
)

_REQUIRED_FILES = ("sales_clean.parquet", "sku_profiles.csv")


def data_dir(snapshot: str | None = ML_DATA_SNAPSHOT) -> Path:
    """Where the ML track reads its inputs from.

    Returns the pinned snapshot directory when `snapshot` is set, otherwise
    the live data/processed directory. Pinning matters because the weekly cron
    rewrites data/processed in place: ML_FINAL_TEST_CUTOFF fixes which weeks a
    window covers, but not the actuals inside it, so unpinned results drift
    between runs and stop being comparable across model versions.

    Production code does not call this. It reads data/processed directly and
    keeps following the weekly refresh.
    """
    if snapshot is None:
        return DATA_PROCESSED

    path = DATA_SNAPSHOTS / str(snapshot)
    if not path.is_dir():
        available = (
            ", ".join(sorted(p.name for p in DATA_SNAPSHOTS.iterdir() if p.is_dir()))
            if DATA_SNAPSHOTS.is_dir()
            else "none"
        )
        raise FileNotFoundError(
            f"ML_DATA_SNAPSHOT is '{snapshot}' but {path} does not exist. "
            f"Available snapshots: {available}. Create one with "
            f"scripts/ml_snapshot_data.py, or set ML_DATA_SNAPSHOT = None in "
            f"config.py to use the live data in data/processed."
        )

    missing = [f for f in _REQUIRED_FILES if not (path / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Snapshot '{snapshot}' is incomplete: missing {', '.join(missing)} "
            f"in {path}. Recreate it with scripts/ml_snapshot_data.py."
        )
    return path


def load_weekly(snapshot: str | None = ML_DATA_SNAPSHOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (weekly, profiles).

    weekly:   unique_id / ds / y — zero-filled W-MON grid, trailing weeks
              trimmed, ramp-up SKUs trimmed to train_start.
    profiles: sku_profiles.csv with parsed dates (bucket, history_length, …).

    Reads from the pinned snapshot (config.ML_DATA_SNAPSHOT) by default so
    that every experiment is measured on identical data regardless of when it
    runs. Pass snapshot=None to read the live, weekly-refreshed data instead.
    """
    src = data_dir(snapshot)
    weekly = pd.read_parquet(src / "sales_clean.parquet")
    profiles = pd.read_csv(src / "sku_profiles.csv")
    weekly["ds"] = pd.to_datetime(weekly["ds"])
    profiles["train_start"] = pd.to_datetime(profiles["train_start"])

    # Drop noisy tail (same as backtest.py)
    if TRIM_TRAILING_WEEKS:
        keep = sorted(weekly["ds"].unique())[:-TRIM_TRAILING_WEEKS]
        weekly = weekly[weekly["ds"].isin(keep)]

    # Trim ramp-up SKUs to train_start (same as backtest._trim_to_train_start)
    starts = profiles.set_index("unique_id")["train_start"]
    weekly = weekly.copy()
    weekly["_ts"] = weekly["unique_id"].map(starts)
    weekly = weekly[weekly["_ts"].isna() | (weekly["ds"] >= weekly["_ts"])]
    weekly = weekly.drop(columns="_ts").sort_values(["unique_id", "ds"])

    return weekly.reset_index(drop=True), profiles


@dataclass
class Split:
    """One rolling-origin evaluation split.

    cutoff = last TRAINING week (inclusive). Test covers the `horizon`
    weeks strictly after cutoff. Everything a model may look at must have
    ds <= cutoff — evaluate.py re-checks this to catch leakage bugs.
    """

    cutoff: pd.Timestamp
    train: pd.DataFrame
    test: pd.DataFrame
    horizon: int

    def __repr__(self) -> str:  # readable experiment logs
        return (
            f"Split(cutoff={self.cutoff.date()}, "
            f"test={self.test['ds'].min().date()}→{self.test['ds'].max().date()}, "
            f"h={self.horizon})"
        )


def make_splits(
    weekly: pd.DataFrame,
    horizon: int = TEST_WEEKS,
    n_splits: int = 1,
    anchor: str | pd.Timestamp | None = None,
) -> list[Split]:
    """Rolling-origin splits, most recent first (split 0 = final test).

    `anchor` is the last TRAINING week of split 0. Windows are built by
    stepping the cutoff back `horizon` weeks per split, so the anchor pins
    every window to a fixed date and a data refresh cannot shift them. Any
    weeks in the data after split 0's test window are ignored.

    When `anchor` is None, it falls back to the latest complete window in
    the data (legacy behavior: split 0's test is the last `horizon` weeks).
    """
    all_weeks = [pd.Timestamp(w) for w in sorted(weekly["ds"].unique())]

    if anchor is None:
        anchor_idx = len(all_weeks) - horizon - 1
    else:
        anchor = pd.Timestamp(anchor)
        if anchor not in all_weeks:
            raise ValueError(
                f"Anchor {anchor.date()} is not a week present in the data "
                f"({all_weeks[0].date()} to {all_weeks[-1].date()})."
            )
        anchor_idx = all_weeks.index(anchor)

    splits: list[Split] = []
    for i in range(n_splits):
        cutoff_idx = anchor_idx - i * horizon
        start = cutoff_idx + 1          # first test week
        end = start + horizon           # one past the last test week
        if cutoff_idx < 0:
            raise ValueError(
                f"Not enough history for split {i + 1}/{n_splits}: the cutoff "
                f"would fall before the data starts ({all_weeks[0].date()})."
            )
        if end > len(all_weeks):
            raise ValueError(
                f"Split {i + 1}/{n_splits} needs {horizon} test weeks after "
                f"{all_weeks[cutoff_idx].date()}, but the data ends at "
                f"{all_weeks[-1].date()}."
            )
        cutoff = all_weeks[cutoff_idx]
        test_weeks = all_weeks[start:end]
        splits.append(
            Split(
                cutoff=cutoff,
                train=weekly[weekly["ds"] <= cutoff].copy(),
                test=weekly[weekly["ds"].isin(test_weeks)].copy(),
                horizon=horizon,
            )
        )
    return splits


def stratified_val_skus(
    weekly: pd.DataFrame,
    profiles: pd.DataFrame,
    frac: float = 0.15,
    n_tiers: int = 3,
    seed: int = 42,
    min_per_cell: int = 2,
) -> set[str]:
    """Choose the early-stopping validation SKUs, stratified so the slice
    represents the whole portfolio.

    Stratification cells = segment (short vs long, merging medium+full) x
    within-segment demand-volume tercile. A proportional `frac` is drawn from
    each cell, with at least `min_per_cell` SKUs per cell. A purely random
    draw of 15% of 432 SKUs can under-sample high-volume SKUs, which dominate
    the demand-weighted early-stopping signal; stratifying prevents that.

    Returns a set of unique_id strings to hold out from tree fitting.
    """
    vol = weekly.groupby("unique_id")["y"].mean()
    seg = (
        profiles.set_index("unique_id")["history_length"]
        .reindex(vol.index)
        .map({"short": "short", "medium": "long", "full": "long"})
        .fillna("short")
    )
    df = pd.DataFrame({"uid": vol.index, "vol": vol.to_numpy(), "seg": seg.to_numpy()})

    rng = np.random.default_rng(seed)
    chosen: list[str] = []
    for _, g in df.groupby("seg"):
        g = g.copy()
        if len(g) >= n_tiers:
            # rank before qcut so identical volumes don't collapse a tier
            g["tier"] = pd.qcut(g["vol"].rank(method="first"), n_tiers, labels=False)
        else:
            g["tier"] = 0
        for _, cell in g.groupby("tier"):
            n = min(len(cell), max(min_per_cell, round(len(cell) * frac)))
            chosen.extend(rng.choice(cell["uid"].to_numpy(), size=n, replace=False))
    return set(chosen)


def eligible_skus(
    profiles: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    min_weeks: int = MIN_SIM_HISTORY_WEEKS,
) -> set[str]:
    """SKUs that were forecastable at `cutoff`.

    A SKU is eligible only if it had at least `min_weeks` weeks of history by
    the cutoff, measured as (cutoff - train_start). This mirrors the
    statistical prototype's per-window inclusion rule (backtest.py trims to
    train_start and requires MIN_SIM_HISTORY_WEEKS of history), so the two
    tracks score the same population. Without it, a backtest window would
    score SKUs that had little or no history at the time, or had not launched
    yet, which distorts both the pooled WAPE and the segment composition.

    Note: `train_start` is a stable per-SKU property (launch or ramp-start
    week), so this is a valid as-of check. The bucket label (smooth vs
    intermittent) is still taken from the present-day snapshot; recomputing
    it per window is a separate, deeper change (see design doc).
    """
    cutoff = pd.Timestamp(cutoff)
    ts = pd.to_datetime(profiles.set_index("unique_id")["train_start"])
    weeks_at_cutoff = (cutoff - ts).dt.days / 7
    return set(ts.index[weeks_at_cutoff >= min_weeks])


def asof_history_length(profiles: pd.DataFrame, cutoff: str | pd.Timestamp) -> pd.Series:
    """History-length label (short/medium/full) as of `cutoff`, not today.

    A SKU's history length is the weeks between its train_start and the
    cutoff, thresholded at the same 50/104-week boundaries the profiler uses.
    This matters in backtests: a SKU that is "full" today may have had only a
    few weeks of history at an older cutoff, so scoring it under today's label
    would put it in the wrong segment. Returns a Series indexed by unique_id.
    (bucket, i.e. smooth vs intermittent, is a deeper as-of recompute and is
    still taken from the snapshot; see the design doc.)
    """
    cutoff = pd.Timestamp(cutoff)
    ts = pd.to_datetime(profiles.set_index("unique_id")["train_start"])
    weeks = (cutoff - ts).dt.days / 7
    return pd.cut(
        weeks,
        bins=[-float("inf"), SHORT_HISTORY_WEEKS, MEDIUM_HISTORY_WEEKS, float("inf")],
        labels=["short", "medium", "full"],
        right=False,
    )


def final_test_split(weekly: pd.DataFrame) -> Split:
    """The QUARANTINED final test window, pinned to ML_FINAL_TEST_CUTOFF.

    Protocol: never used for feature choices, hyperparameters, or any other
    design decision. Run a model here only as the final go/no-go gate,
    ideally once. All iteration happens on dev_splits().
    """
    return make_splits(weekly, n_splits=1, anchor=ML_FINAL_TEST_CUTOFF)[0]


def dev_splits(weekly: pd.DataFrame, n: int = 3) -> list[Split]:
    """Development/validation windows: the n 10-week windows BEFORE the
    final test window, pinned to ML_FINAL_TEST_CUTOFF. All tuning and
    feature iteration is judged on the average and spread across these
    (3 windows span 3 different seasons).
    """
    return make_splits(weekly, n_splits=n + 1, anchor=ML_FINAL_TEST_CUTOFF)[1:]


if __name__ == "__main__":
    weekly, profiles = load_weekly()
    print(f"weekly: {len(weekly):,} rows | {weekly['unique_id'].nunique():,} SKUs "
          f"| {weekly['ds'].min().date()} → {weekly['ds'].max().date()}")
    for s in make_splits(weekly, n_splits=3):
        print(" ", s)
