# Stage 8: Model selection (CV only — test set is never touched here).
#
# For medium/full SKUs: picks the model with lowest horizon-aggregate WAPE across
#   CV windows. Horizon WAPE = |sum(yhat) - sum(y)| / sum(y) per window, averaged.
#   This directly measures 13-week total demand accuracy, which is what matters for
#   restocking decisions — per-week errors that cancel out don't matter.
# For short-history SKUs: assigns fixed defaults (no CV available).
#
# Outputs:
#   outputs/reports/cv_metrics.csv  — per-SKU × model CV metrics
#   outputs/reports/selection.csv   — per-SKU: selected model + CV metrics
import pandas as pd
from pathlib import Path

from config import OUTPUTS_REPORTS, TRIM_TRAILING_WEEKS, TEST_WEEKS, ROUTE_SHORT_SMOOTH_TO_V1

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
CV_PATH = OUTPUTS_REPORTS / "cv_results.parquet"

# Assigned without CV — safest models for SKUs with < 1 full seasonal cycle.
SHORT_DEFAULTS = {
    "smooth":     "V1" if ROUTE_SHORT_SMOOTH_TO_V1 else "WindowAverage",
    "low_volume": "WindowAverage",
}

# SKUs where the model accuracy is genuinely poor and restocking needs human oversight.
LOW_CONFIDENCE_BUCKETS = {("smooth", "short")}

# Smooth full/medium SKUs where the best model's 13-week total is still off by more
# than this fraction of actual demand get flagged for manual review.
LOW_CONFIDENCE_HORIZON_WAPE = 0.35


# ── Helpers ───────────────────────────────────────────────────────────────────

def _metrics_from_cv(cv_df: pd.DataFrame) -> pd.DataFrame:
    """Per-SKU × model horizon-aggregate WAPE, averaged across CV windows.

    For each (SKU, model, cutoff) window: sum yhat and y over the horizon,
    compute |sum(yhat)-sum(y)| / sum(y). Then average across windows.
    This is the right metric for restocking — total demand accuracy, not per-week.
    """
    meta = {"unique_id", "ds", "cutoff", "y", "bucket", "history_length"}
    model_cols = [c for c in cv_df.columns if c not in meta]

    long = cv_df.melt(
        id_vars=["unique_id", "ds", "cutoff", "y", "bucket", "history_length"],
        value_vars=model_cols,
        var_name="model",
        value_name="yhat",
    ).dropna(subset=["yhat"])

    # Sum over each CV window
    window = (
        long.groupby(["unique_id", "model", "cutoff", "bucket", "history_length"])
        .agg(total_y=("y", "sum"), total_yhat=("yhat", "sum"))
        .reset_index()
    )
    window["abs_err"] = (window["total_yhat"] - window["total_y"]).abs()
    window["wape"]    = window["abs_err"] / window["total_y"].clip(lower=1e-6)
    window["bias"]    = (window["total_yhat"] - window["total_y"]) / window["total_y"].clip(lower=1e-6)

    # Average across windows
    return (
        window.groupby(["unique_id", "model", "bucket", "history_length"])
        .agg(
            HorizonWAPE=("wape",    "mean"),
            HorizonBias=("bias",    "mean"),
            HorizonMAE =("abs_err", "mean"),
        )
        .reset_index()
    )


def _ensemble_metrics(cv_df: pd.DataFrame, metrics: pd.DataFrame, top_n: int = 2) -> pd.DataFrame:
    """Average the top-N models' CV predictions per SKU; compute horizon metrics."""
    meta = {"unique_id", "ds", "cutoff", "y", "bucket", "history_length"}
    available_cols = set(cv_df.columns) - meta

    rows = []
    for uid, uid_m in metrics.groupby("unique_id"):
        top = uid_m.nsmallest(top_n, "HorizonWAPE")
        top_models = [m for m in top["model"].tolist() if m in available_cols]
        if len(top_models) < 2:
            continue

        uid_cv = cv_df[cv_df["unique_id"] == uid].copy()
        usable = [m for m in top_models if uid_cv[m].notna().any()]
        if len(usable) < 2:
            continue

        uid_cv["_ensemble"] = uid_cv[usable].mean(axis=1)

        window = (
            uid_cv.groupby("cutoff")
            .agg(total_y=("y", "sum"), total_yhat=("_ensemble", "sum"))
            .reset_index()
        )
        window["abs_err"] = (window["total_yhat"] - window["total_y"]).abs()
        window["wape"]    = window["abs_err"] / window["total_y"].clip(lower=1e-6)
        window["bias"]    = (window["total_yhat"] - window["total_y"]) / window["total_y"].clip(lower=1e-6)

        rows.append({
            "unique_id":      uid,
            "model":          "Ensemble:" + "+".join(sorted(usable)),
            "bucket":         uid_m["bucket"].iloc[0],
            "history_length": uid_m["history_length"].iloc[0],
            "HorizonWAPE":    float(window["wape"].mean()),
            "HorizonBias":    float(window["bias"].mean()),
            "HorizonMAE":     float(window["abs_err"].mean()),
        })

    return pd.DataFrame(rows)


def _select_best(metrics: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """HorizonWAPE-primary for CV SKUs; fixed defaults for short-history SKUs."""
    cv_sel = (
        metrics
        .sort_values(["unique_id", "HorizonWAPE", "HorizonBias"])
        .groupby("unique_id")
        .first()
        .reset_index()
        [["unique_id", "model", "bucket", "history_length", "HorizonWAPE", "HorizonBias", "HorizonMAE"]]
    )

    short = profiles.loc[
        profiles["history_length"] == "short",
        ["unique_id", "bucket", "history_length"],
    ].copy()

    if not short.empty:
        short["model"] = short["bucket"].map(SHORT_DEFAULTS)
        short[["HorizonWAPE", "HorizonBias", "HorizonMAE"]] = float("nan")
        short_sel = short[[
            "unique_id", "model", "bucket", "history_length",
            "HorizonWAPE", "HorizonBias", "HorizonMAE",
        ]]
        result = pd.concat([cv_sel, short_sel], ignore_index=True)
    else:
        result = cv_sel

    result["forecast_confidence"] = result.apply(
        lambda r: "low" if (
            (r["bucket"], r["history_length"]) in LOW_CONFIDENCE_BUCKETS
            or (r["bucket"] == "smooth"
                and r["history_length"] in ("full", "medium")
                and pd.notna(r["HorizonWAPE"])
                and r["HorizonWAPE"] > LOW_CONFIDENCE_HORIZON_WAPE)
        ) else "standard",
        axis=1,
    )
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def select(weekly: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    cv_df = pd.read_parquet(CV_PATH)

    print("Computing CV metrics (horizon-aggregate)...")
    metrics = _metrics_from_cv(cv_df)

    print("Computing ensemble metrics...")
    ensemble = _ensemble_metrics(cv_df, metrics)
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
    print("Median horizon metrics by bucket (medium/full SKUs only):")
    cv_skus = selection[selection["HorizonWAPE"].notna()]
    print(cv_skus.groupby("bucket")[["HorizonWAPE", "HorizonBias", "HorizonMAE"]].median().round(3).to_string())
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
