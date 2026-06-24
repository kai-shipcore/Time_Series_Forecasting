# Stage 7: Model-agnostic backtest engine.
# - Drops TRIM_TRAILING_WEEKS from tail (pending/unregistered orders make them noisy)
# - Holds out the next TEST_WEEKS as evaluation set
# - Trims ramp-up SKUs to train_start before any fitting
# - Runs StatsForecast cross_validation per bucket × history_length group
# - Short-history SKUs produce no CV rows; selector.py assigns their defaults
#
# Outputs:
#   outputs/reports/cv_results.parquet  — raw CV forecasts (medium + full SKUs)
#   outputs/reports/test_set.parquet    — held-out TEST_WEEKS evaluation window
import time
import pandas as pd
from pathlib import Path
from statsforecast import StatsForecast

from config import FORECAST_HORIZON, FREQUENCY, N_CV_SPLITS, OUTPUTS_REPORTS, TRIM_TRAILING_WEEKS, TEST_WEEKS, USE_SEASONAL_ADJUSTMENT
from src.models import get_models
from src.baselines import get_baselines
from src.deseasonalize import deseasonalize, reseasonalize

HORIZON_WEEKS = round(FORECAST_HORIZON / 7)  # 13 weeks — production forecast horizon

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
CV_PATH = OUTPUTS_REPORTS / "cv_results.parquet"
TEST_PATH = OUTPUTS_REPORTS / "test_set.parquet"

# n_windows per history length — "short" has no CV
_HIST_WINDOWS = {
    "full":   N_CV_SPLITS,  # 6 windows × 10 weeks — test spans Jan 2025 → Mar 2026; Q4 is 2/6 not 3/4
    "medium": 3,            # 3 windows; medium SKUs have 52–104 active weeks, min_len=31 so all qualify
}


def _trim_to_train_start(df: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """Drop rows before each SKU's train_start (affects ramp-up SKUs only)."""
    train_starts = pd.to_datetime(profiles.set_index("unique_id")["train_start"])
    df = df.copy()
    df["_ts"] = df["unique_id"].map(train_starts)
    return df[df["ds"] >= df["_ts"]].drop(columns="_ts")


def _min_length(n_windows: int, h: int) -> int:
    """Minimum series length so window 1 has at least 1 training observation."""
    return n_windows * h + 1


def backtest(weekly: pd.DataFrame, profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # ── 1. Trim noisy tail, then split test set ────────────────────────────────
    all_weeks = sorted(weekly["ds"].unique())

    # Drop trailing weeks with unregistered/pending orders
    trimmed_weeks = all_weeks[:-TRIM_TRAILING_WEEKS] if TRIM_TRAILING_WEEKS else all_weeks
    weekly = weekly[weekly["ds"].isin(trimmed_weeks)].copy()

    test_start = trimmed_weeks[-TEST_WEEKS]
    train_df = weekly[weekly["ds"] < test_start].copy()
    test_df = weekly[weekly["ds"] >= test_start].copy()

    print(f"Dropped    : last {TRIM_TRAILING_WEEKS} weeks (noisy tail, pending orders)")
    print(f"Test set   : {test_start.date()} → {weekly['ds'].max().date()} "
          f"({TEST_WEEKS} weeks, {len(test_df):,} rows)")
    print(f"Train data : up to {train_df['ds'].max().date()} "
          f"({len(trimmed_weeks) - TEST_WEEKS} weeks)")
    print()

    # ── 2. Trim ramp-up SKUs to train_start ───────────────────────────────────
    train_trimmed = _trim_to_train_start(train_df, profiles)

    # ── 3. CV per bucket × history_length ─────────────────────────────────────
    cv_parts = []

    for bucket in ("smooth", "low_volume", "intermittent"):
        for hist, n_windows in _HIST_WINDOWS.items():
            skus = profiles.loc[
                (profiles["bucket"] == bucket) & (profiles["history_length"] == hist),
                "unique_id",
            ].tolist()

            if not skus:
                continue

            df_group = (
                train_trimmed[train_trimmed["unique_id"].isin(skus)]
                [["unique_id", "ds", "y"]]
            )

            # Drop series too short for the requested n_windows
            min_len = _min_length(n_windows, TEST_WEEKS)
            lengths = df_group.groupby("unique_id")["ds"].count()
            too_short = lengths[lengths < min_len].index.tolist()
            if too_short:
                print(f"  WARNING: {len(too_short)} {bucket}/{hist} series have "
                      f"< {min_len} weeks after trimming — skipped from CV")
                df_group = df_group[~df_group["unique_id"].isin(too_short)]

            if df_group.empty:
                continue

            candidates = get_models(bucket, hist)
            candidate_names = {type(m).__name__ for m in candidates}
            baselines = [b for b in get_baselines(bucket, hist)
                         if type(b).__name__ not in candidate_names]
            models = candidates + baselines
            model_names = [type(m).__name__ for m in models]

            apply_deseas = USE_SEASONAL_ADJUSTMENT and bucket == "smooth"
            deseas_label = " [deseasonalized]" if apply_deseas else ""
            print(f"  {bucket}/{hist}: {len(skus)} SKUs | "
                  f"n_windows={n_windows} | {model_names}{deseas_label}")

            fit_data = deseasonalize(df_group) if apply_deseas else df_group

            t0 = time.time()
            sf = StatsForecast(models=models, freq=FREQUENCY, n_jobs=-1)
            cv = sf.cross_validation(
                df=fit_data,
                h=TEST_WEEKS,
                n_windows=n_windows,
                step_size=TEST_WEEKS,
            )

            if apply_deseas:
                cv = reseasonalize(cv)

            cv["bucket"] = bucket
            cv["history_length"] = hist
            cv_parts.append(cv)
            print(f"    → {len(cv):,} rows in {time.time() - t0:.1f}s")

    # ── 4. Save and return ────────────────────────────────────────────────────
    cv_df = pd.concat(cv_parts, ignore_index=True)

    OUTPUTS_REPORTS.mkdir(parents=True, exist_ok=True)
    cv_df.to_parquet(CV_PATH, index=False)
    test_df.to_parquet(TEST_PATH, index=False)

    n_skus_cv = cv_df["unique_id"].nunique()
    n_skus_short = profiles[profiles["history_length"] == "short"]["unique_id"].nunique()
    print(f"\nCV results  : {CV_PATH.name}  ({len(cv_df):,} rows, {n_skus_cv} SKUs)")
    print(f"Test set    : {TEST_PATH.name}  ({len(test_df):,} rows, {TEST_WEEKS} weeks)")
    print(f"Short-hist  : {n_skus_short} SKUs — no CV, selector.py assigns defaults")

    return cv_df, test_df


if __name__ == "__main__":
    weekly = pd.read_parquet(PROCESSED_DIR / "sales_clean.parquet")
    profiles = pd.read_csv(PROCESSED_DIR / "sku_profiles.csv")
    profiles["train_start"] = pd.to_datetime(profiles["train_start"])
    weekly["ds"] = pd.to_datetime(weekly["ds"])

    print(f"Loaded: {weekly['unique_id'].nunique()} SKUs, {len(weekly):,} rows\n")
    backtest(weekly, profiles)
