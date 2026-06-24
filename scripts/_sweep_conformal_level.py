#!/usr/bin/env python3
"""Sweep conformal levels to find the narrowest intervals at acceptable coverage."""
import sys, warnings, copy
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.utils import ConformalIntervals

from config import FREQUENCY, TEST_WEEKS, TRIM_TRAILING_WEEKS
from src.models import get_models
from src.baselines import get_baselines
from src.deseasonalize import deseasonalize, reseasonalize
from src.backtest import _trim_to_train_start

PROC = ROOT / "data/processed"
REP  = ROOT / "outputs/reports"
LEVELS    = [75, 70, 60, 50, 40]
N_WINDOWS = 5


def main():
    weekly   = pd.read_parquet(PROC / "sales_clean.parquet")
    profiles = pd.read_csv(PROC / "sku_profiles.csv")
    test_set = pd.read_parquet(REP / "test_set.parquet")
    weekly["ds"] = pd.to_datetime(weekly["ds"])
    profiles["train_start"] = pd.to_datetime(profiles["train_start"])
    test_set["ds"] = pd.to_datetime(test_set["ds"])

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
    fit_data   = deseasonalize(train_full)[["unique_id", "ds", "y"]]

    actuals = (
        test_set[test_set["unique_id"].isin(full_uids)]
        .groupby("unique_id")["y"].sum()
    )

    candidates      = get_models("smooth", "full")
    candidate_names = {type(m).__name__ for m in candidates}
    baselines       = [b for b in get_baselines("smooth", "full")
                       if type(b).__name__ not in candidate_names]
    base_models = candidates + baselines

    selection = pd.read_csv(REP / "selection.csv")
    sel_map   = selection.set_index("unique_id")["model"].to_dict()

    avg_actual = actuals.mean()
    print(f"{'Level':>6}  {'Coverage':>10}  {'Avg width':>10}  "
          f"{'Rel width':>10}  {'Avg upper':>10}")
    print("─" * 58)

    for level in LEVELS:
        models = copy.deepcopy(base_models)
        pi  = ConformalIntervals(h=TEST_WEEKS, n_windows=N_WINDOWS)
        sf  = StatsForecast(models=models, freq=FREQUENCY, n_jobs=-1)
        fcast = sf.forecast(df=fit_data, h=TEST_WEEKS, level=[level],
                            prediction_intervals=pi)
        fcast["ds"] = pd.to_datetime(fcast["ds"])
        fcast = reseasonalize(fcast)

        lo_suf, hi_suf = f"-lo-{level}", f"-hi-{level}"
        rows = []
        for uid, uid_f in fcast.groupby("unique_id"):
            uid_f = uid_f.sort_values("ds")
            mn = sel_map.get(uid, "")
            if mn.startswith("Ensemble:"):
                parts = mn.replace("Ensemble:", "").split("+")
                lo_cols = [f"{p}{lo_suf}" for p in parts if f"{p}{lo_suf}" in uid_f.columns]
                hi_cols = [f"{p}{hi_suf}" for p in parts if f"{p}{hi_suf}" in uid_f.columns]
                lo = uid_f[lo_cols].mean(axis=1).clip(lower=0).sum() if lo_cols else np.nan
                hi = uid_f[hi_cols].mean(axis=1).clip(lower=0).sum() if hi_cols else np.nan
            else:
                lo_col, hi_col = f"{mn}{lo_suf}", f"{mn}{hi_suf}"
                lo = uid_f[lo_col].clip(lower=0).sum() if lo_col in uid_f.columns else np.nan
                hi = uid_f[hi_col].clip(lower=0).sum() if hi_col in uid_f.columns else np.nan
            rows.append({"uid": uid, "lo": float(lo), "hi": float(hi),
                         "actual": actuals.get(uid, np.nan)})

        df = pd.DataFrame(rows).dropna(subset=["lo", "hi"])
        covered   = ((df["actual"] >= df["lo"]) & (df["actual"] <= df["hi"])).sum()
        avg_width = (df["hi"] - df["lo"]).mean()
        avg_hi    = df["hi"].mean()
        rel_width = avg_width / avg_actual
        print(f"{level:>6}  {covered:>4}/{len(df):<5} "
              f"({covered/len(df)*100:.0f}%)  {avg_width:>10.1f}  "
              f"{rel_width:>9.0%}  {avg_hi:>10.1f}")


if __name__ == "__main__":
    main()
