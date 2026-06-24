# Stage 8: Model selection (CV only — test set is never touched here).
#
# For medium/full SKUs: picks the model with lowest avg MASE across CV windows
#   (WAPE as tiebreaker). Reads cv_results.parquet written by backtest.py.
# For short-history SKUs: assigns fixed defaults (no CV available).
#
# Outputs:
#   outputs/reports/cv_metrics.csv  — per-SKU × model CV metrics
#   outputs/reports/selection.csv   — per-SKU: selected model + CV metrics
import numpy as np
import pandas as pd
from pathlib import Path

from config import OUTPUTS_REPORTS, TRIM_TRAILING_WEEKS, TEST_WEEKS, ROUTE_SHORT_SMOOTH_TO_V1

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
CV_PATH = OUTPUTS_REPORTS / "cv_results.parquet"

# Assigned without CV — safest models for SKUs with < 1 full seasonal cycle.
# smooth: V1 when routing flag is on (rolling rate wins on ramp-up products),
#         otherwise WindowAverage(12) as a simple trailing-average fallback.
SHORT_DEFAULTS = {
    "smooth":     "V1" if ROUTE_SHORT_SMOOTH_TO_V1 else "WindowAverage",
    "low_volume": "WindowAverage",
}

# SKUs where the model accuracy is genuinely poor and restocking needs human oversight.
# Short-history smooth = all ramp-up products (<1 yr active), best CV MASE ~1.03 on recent tail.
# No reliable seasonal signal, demand pattern still evolving — flag for manual review.
LOW_CONFIDENCE_BUCKETS = {("smooth", "short")}

# Smooth full/medium SKUs with MASE above this threshold get flagged low confidence.
# Directly measures forecast difficulty regardless of cause (noise, Q4 concentration, etc.).
LOW_CONFIDENCE_MASE_THRESHOLD = 1.3


# ── Helpers ───────────────────────────────────────────────────────────────────

def _naive_mae(train_df: pd.DataFrame) -> pd.DataFrame:
    """In-sample 1-step naive MAE per SKU — denominator for MASE."""
    def _mae(y: np.ndarray) -> float:
        d = np.abs(np.diff(y))
        return float(d.mean()) if len(d) > 0 else 1.0

    result = (
        train_df.sort_values(["unique_id", "ds"])
        .groupby("unique_id")["y"]
        .apply(lambda s: _mae(s.values))
        .reset_index(name="naive_mae")
    )
    result["naive_mae"] = result["naive_mae"].clip(lower=1e-6)
    return result


def _metrics_from_cv(cv_df: pd.DataFrame, naive_mae_df: pd.DataFrame) -> pd.DataFrame:
    """Per-SKU × model metrics averaged across all CV windows."""
    meta = {"unique_id", "ds", "cutoff", "y", "bucket", "history_length"}
    model_cols = [c for c in cv_df.columns if c not in meta]

    long = (
        cv_df.melt(
            id_vars=["unique_id", "ds", "y", "bucket", "history_length"],
            value_vars=model_cols,
            var_name="model",
            value_name="yhat",
        )
        .dropna(subset=["yhat"])
        .merge(naive_mae_df, on="unique_id", how="left")
    )

    long["abs_err"]    = (long["y"] - long["yhat"]).abs()
    long["scaled_err"] = long["abs_err"] / long["naive_mae"]
    long["bias_err"]   = long["yhat"] - long["y"]

    m = (
        long.groupby(["unique_id", "model", "bucket", "history_length"])
        .agg(
            MAE     =("abs_err",    "mean"),
            MASE    =("scaled_err", "mean"),
            Bias    =("bias_err",   "mean"),
            _sum_y  =("y",          "sum"),
            _sum_ae =("abs_err",    "sum"),
        )
        .reset_index()
    )
    m["WAPE"] = (m["_sum_ae"] / m["_sum_y"].clip(lower=1e-6)).round(4)
    return m.drop(columns=["_sum_y", "_sum_ae"])


def _ensemble_metrics(
    cv_df: pd.DataFrame, metrics: pd.DataFrame, naive_mae_df: pd.DataFrame, top_n: int = 2
) -> pd.DataFrame:
    """
    For each SKU, average the top-N models' CV predictions and compute metrics.
    Returns rows in the same shape as _metrics_from_cv output — can be concat'd
    directly into the metrics table before selection.
    """
    meta = {"unique_id", "ds", "cutoff", "y", "bucket", "history_length"}
    available_cols = set(cv_df.columns) - meta

    cv_with_naive = cv_df.merge(naive_mae_df, on="unique_id", how="left")

    rows = []
    for uid, uid_m in metrics.groupby("unique_id"):
        top = uid_m.nsmallest(top_n, ["MASE", "WAPE"])
        top_models = [m for m in top["model"].tolist() if m in available_cols]
        if len(top_models) < 2:
            continue

        uid_cv = cv_with_naive[cv_with_naive["unique_id"] == uid]
        usable = [m for m in top_models if uid_cv[m].notna().any()]
        if len(usable) < 2:
            continue

        ensemble_yhat = uid_cv[usable].mean(axis=1)
        naive_mae_val = uid_cv["naive_mae"].iloc[0]
        abs_err = (uid_cv["y"] - ensemble_yhat).abs()

        rows.append({
            "unique_id":      uid,
            "model":          "Ensemble:" + "+".join(sorted(usable)),
            "bucket":         uid_m["bucket"].iloc[0],
            "history_length": uid_m["history_length"].iloc[0],
            "MAE":            float(abs_err.mean()),
            "MASE":           float((abs_err / naive_mae_val).mean()),
            "WAPE":           float(abs_err.sum() / max(uid_cv["y"].sum(), 1e-6)),
            "Bias":           float((ensemble_yhat - uid_cv["y"]).mean()),
        })

    return pd.DataFrame(rows)


def _select_best(metrics: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """MASE-primary for CV SKUs; fixed defaults for short-history SKUs."""
    cv_sel = (
        metrics
        .sort_values(["unique_id", "MASE", "WAPE"])
        .groupby("unique_id")
        .first()
        .reset_index()
        [["unique_id", "model", "bucket", "history_length", "MASE", "WAPE", "MAE", "Bias"]]
    )

    short = profiles.loc[
        profiles["history_length"] == "short",
        ["unique_id", "bucket", "history_length"],
    ].copy()

    if not short.empty:
        short["model"] = short["bucket"].map(SHORT_DEFAULTS)
        short[["MASE", "WAPE", "MAE", "Bias"]] = np.nan
        short_sel = short[
            ["unique_id", "model", "bucket", "history_length", "MASE", "WAPE", "MAE", "Bias"]
        ]
        result = pd.concat([cv_sel, short_sel], ignore_index=True)
    else:
        result = cv_sel

    result["forecast_confidence"] = result.apply(
        lambda r: "low" if (
            (r["bucket"], r["history_length"]) in LOW_CONFIDENCE_BUCKETS
            or (r["bucket"] == "smooth"
                and r["history_length"] in ("full", "medium")
                and pd.notna(r["MASE"])
                and r["MASE"] > LOW_CONFIDENCE_MASE_THRESHOLD)
        ) else "standard",
        axis=1,
    )
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def select(weekly: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    cv_df = pd.read_parquet(CV_PATH)

    # Reconstruct the training boundary (mirrors backtest.py trimming logic)
    all_weeks = sorted(weekly["ds"].unique())
    trimmed_weeks = all_weeks[:-TRIM_TRAILING_WEEKS] if TRIM_TRAILING_WEEKS else all_weeks
    test_start = trimmed_weeks[-TEST_WEEKS]

    train_df = weekly[weekly["ds"].isin(trimmed_weeks) & (weekly["ds"] < test_start)].copy()
    train_starts = pd.to_datetime(profiles.set_index("unique_id")["train_start"])
    train_df["_ts"] = train_df["unique_id"].map(train_starts)
    train_trimmed = (
        train_df[train_df["ds"] >= train_df["_ts"]]
        .drop(columns="_ts")
        [["unique_id", "ds", "y"]]
    )

    print("Computing naive MAE denominators...")
    naive_mae_df = _naive_mae(train_trimmed)

    print("Computing CV metrics...")
    metrics = _metrics_from_cv(cv_df, naive_mae_df)

    print("Computing ensemble metrics...")
    ensemble = _ensemble_metrics(cv_df, metrics, naive_mae_df)
    all_metrics = pd.concat([metrics, ensemble], ignore_index=True)

    print("Selecting best model per SKU...")
    selection = _select_best(all_metrics, profiles)

    OUTPUTS_REPORTS.mkdir(parents=True, exist_ok=True)
    all_metrics.to_csv(OUTPUTS_REPORTS / "cv_metrics.csv", index=False)
    selection.to_csv(OUTPUTS_REPORTS / "selection.csv", index=False)

    print(f"\nSaved: cv_metrics.csv  ({len(all_metrics):,} rows)")
    print(f"Saved: selection.csv   ({len(selection)} SKUs)\n")

    print("Model selection breakdown:")
    print(selection.groupby(["bucket", "model"]).size().to_string())
    print()
    print("Median CV metrics by bucket (medium/full SKUs only):")
    cv_skus = selection[selection["MASE"].notna()]
    print(cv_skus.groupby("bucket")[["MASE", "WAPE", "MAE", "Bias"]].median().round(3).to_string())
    n_low = (selection["forecast_confidence"] == "low").sum()
    print(f"\nLow-confidence SKUs (manual restocking review recommended): {n_low}")

    return selection


if __name__ == "__main__":
    weekly   = pd.read_parquet(PROCESSED_DIR / "sales_clean.parquet")
    profiles = pd.read_csv(PROCESSED_DIR / "sku_profiles.csv")
    profiles["train_start"] = pd.to_datetime(profiles["train_start"])
    weekly["ds"] = pd.to_datetime(weekly["ds"])

    print(f"Loaded: {weekly['unique_id'].nunique()} SKUs\n")
    select(weekly, profiles)
