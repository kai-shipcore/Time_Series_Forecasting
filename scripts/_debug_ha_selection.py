#!/usr/bin/env python3
"""Debug: why does HistoricAverage win in the 78-week CV fold?"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from statsforecast import StatsForecast

from config import FREQUENCY, TEST_WEEKS, TRIM_TRAILING_WEEKS
from src.models import get_models
from src.baselines import get_baselines
from src.deseasonalize import deseasonalize, reseasonalize
from src.backtest import _trim_to_train_start

PROCESSED_DIR = ROOT / "data/processed"

def main():
    weekly   = pd.read_parquet(PROCESSED_DIR / "sales_clean.parquet")
    profiles = pd.read_csv(PROCESSED_DIR / "sku_profiles.csv")
    weekly["ds"] = pd.to_datetime(weekly["ds"])
    profiles["train_start"] = pd.to_datetime(profiles["train_start"])

    all_weeks  = sorted(weekly["ds"].unique())
    trimmed    = all_weeks[:-TRIM_TRAILING_WEEKS]
    test_start = pd.Timestamp(trimmed[-TEST_WEEKS])

    full_uids = profiles.loc[
        (profiles["bucket"] == "smooth") & (profiles["history_length"] == "full"),
        "unique_id",
    ].tolist()
    train_full = _trim_to_train_start(
        weekly[weekly["ds"].isin(trimmed) & (weekly["ds"] < test_start)].copy(), profiles
    )
    train_full = train_full[train_full["unique_id"].isin(full_uids)]

    candidates      = get_models("smooth", "full")
    candidate_names = {type(m).__name__ for m in candidates}
    baselines       = [b for b in get_baselines("smooth", "full")
                       if type(b).__name__ not in candidate_names]
    models = candidates + baselines

    for L in [65, 78]:
        train = (
            train_full.sort_values(["unique_id", "ds"])
            .groupby("unique_id", group_keys=False)
            .apply(lambda g: g.tail(L))
            .reset_index(drop=True)
        )

        print(f"\n{'='*60}")
        print(f"{L}-week window — CV fold")
        print(f"{'='*60}")

        sf = StatsForecast(models=models, freq=FREQUENCY, n_jobs=-1)
        cv = sf.cross_validation(
            df=deseasonalize(train)[["unique_id", "ds", "y"]],
            h=TEST_WEEKS, n_windows=1, step_size=TEST_WEEKS,
        )
        cv = reseasonalize(cv)

        print(f"CV fold test window: {cv['ds'].min().date()} → {cv['ds'].max().date()}")

        naive_mae = (
            train.sort_values(["unique_id", "ds"])
            .groupby("unique_id")["y"]
            .apply(lambda s: float(np.abs(np.diff(s.values)).mean()) if len(s) > 1 else 1.0)
            .reset_index(name="naive_mae")
        )
        naive_mae["naive_mae"] = naive_mae["naive_mae"].clip(lower=1e-6)

        meta   = {"unique_id", "ds", "cutoff", "y"}
        m_cols = [c for c in cv.columns if c not in meta]
        long   = (
            cv.melt(id_vars=["unique_id", "ds", "y"], value_vars=m_cols,
                    var_name="model", value_name="yhat")
            .dropna(subset=["yhat"])
            .merge(naive_mae, on="unique_id", how="left")
        )
        long["ae"]   = (long["y"] - long["yhat"]).abs()
        long["mase"] = long["ae"] / long["naive_mae"]
        long["bias"] = long["yhat"] - long["y"]

        agg = long.groupby("model").agg(
            mase=("mase", "mean"),
            mae=("ae", "mean"),
            bias=("bias", "mean"),
        ).sort_values("mase")
        print("\nModel scores in this CV fold:")
        print(f"  {'Model':<22} {'MASE':>8} {'MAE':>8} {'Bias':>10}")
        for m, row in agg.iterrows():
            print(f"  {m:<22} {row['mase']:>8.4f} {row['mae']:>8.2f} {row['bias']:>+10.2f}")

        # How many SKUs does each model win?
        winner = (
            long.groupby(["unique_id", "model"])["mase"].mean()
            .reset_index().sort_values(["unique_id", "mase"])
            .groupby("unique_id").first()["model"]
        )
        print(f"\nModel wins in CV: {winner.value_counts().to_dict()}")

        # CV fold demand level vs actual test period demand
        cv_level  = cv.merge(train[["unique_id","ds","y"]].rename(columns={"y":"train_y"}),
                             on=["unique_id","ds"], how="left")
        test_df   = weekly[weekly["ds"] >= test_start][["unique_id","ds","y"]]
        test_df   = test_df[test_df["unique_id"].isin(full_uids)]
        print(f"\nDemand level comparison:")
        print(f"  CV fold actual mean:    {cv['y'].mean():.2f} u/wk")
        print(f"  True test period mean:  {test_df['y'].mean():.2f} u/wk")

        # What would HistoricAverage predict vs what AutoETS predicts for the cv fold
        ha_bias  = long[long["model"] == "HistoricAverage"]["bias"].mean()
        ets_bias = long[long["model"] == "AutoETS"]["bias"].mean() if "AutoETS" in long["model"].values else float("nan")
        print(f"\n  HistoricAverage CV bias: {ha_bias:+.2f}  (+ = overforecast)")
        print(f"  AutoETS       CV bias: {ets_bias:+.2f}")


if __name__ == "__main__":
    main()
