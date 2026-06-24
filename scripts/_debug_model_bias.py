#!/usr/bin/env python3
"""
For a given training length, run ALL models on the actual test period and
compare their individual MAE and bias — not just the CV-selected winner.

This answers: are trend-extrapolating models (AutoARIMA, AutoETS) actually
closer to the truth on growing products, even if CV doesn't pick them?
"""
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
TRAIN_LENGTHS = [39, 52, 65, 78]
AUTOETS_MIN_LEN = 20


def main():
    weekly   = pd.read_parquet(PROCESSED_DIR / "sales_clean.parquet")
    profiles = pd.read_csv(PROCESSED_DIR / "sku_profiles.csv")
    weekly["ds"] = pd.to_datetime(weekly["ds"])
    profiles["train_start"] = pd.to_datetime(profiles["train_start"])

    all_weeks  = sorted(weekly["ds"].unique())
    trimmed    = all_weeks[:-TRIM_TRAILING_WEEKS]
    test_start = pd.Timestamp(trimmed[-TEST_WEEKS])
    test_wks   = [w for w in trimmed if w >= test_start]

    full_uids = profiles.loc[
        (profiles["bucket"] == "smooth") & (profiles["history_length"] == "full"),
        "unique_id",
    ].tolist()
    train_full = _trim_to_train_start(
        weekly[weekly["ds"].isin(trimmed) & (weekly["ds"] < test_start)].copy(), profiles
    )
    train_full = train_full[train_full["unique_id"].isin(full_uids)]

    test_df = weekly[weekly["ds"].isin(test_wks) & weekly["unique_id"].isin(full_uids)]
    actual_totals = test_df.groupby("unique_id")["y"].sum()

    # Weekly rate in the last 4 weeks of training (proxy for current run rate)
    recent_rate = (
        train_full.sort_values(["unique_id", "ds"])
        .groupby("unique_id")
        .apply(lambda g: g.tail(4)["y"].mean(), include_groups=False)
    )

    candidates      = get_models("smooth", "full")
    candidate_names = {type(m).__name__ for m in candidates}
    baselines       = [b for b in get_baselines("smooth", "full")
                       if type(b).__name__ not in candidate_names]
    models_full    = candidates + baselines
    models_no_ets  = [m for m in models_full if type(m).__name__ != "AutoETS"]

    model_names = [type(m).__name__ for m in models_full]

    print(f"{'='*72}")
    print(f"PER-MODEL ACTUAL TEST PERFORMANCE  (all models, not just CV winner)")
    print(f"{'='*72}")
    print(f"\nNote: bias < 0 = underforecast. Actual test mean = "
          f"{actual_totals.mean():.1f} units/10-wk window\n")

    for L in TRAIN_LENGTHS:
        n_folds = max(1, (L - AUTOETS_MIN_LEN) // TEST_WEEKS)
        train = (
            train_full.sort_values(["unique_id", "ds"])
            .groupby("unique_id", group_keys=False)
            .apply(lambda g: g.tail(L))
            .reset_index(drop=True)
        )

        # Fit all models, predict test period
        sf = StatsForecast(models=models_full, freq=FREQUENCY, n_jobs=-1)
        sf.fit(deseasonalize(train)[["unique_id", "ds", "y"]])
        fcast = sf.predict(h=TEST_WEEKS)
        fcast["ds"] = pd.to_datetime(fcast["ds"])
        fcast = reseasonalize(fcast)

        # Also run CV to get selected model per SKU
        sf_cv = StatsForecast(models=models_full, freq=FREQUENCY, n_jobs=-1)
        cv    = sf_cv.cross_validation(
            df=deseasonalize(train)[["unique_id", "ds", "y"]],
            h=TEST_WEEKS, n_windows=n_folds, step_size=TEST_WEEKS,
        )
        cv = reseasonalize(cv)

        naive_mae = (
            train.sort_values(["unique_id", "ds"])
            .groupby("unique_id")["y"]
            .apply(lambda s: float(np.abs(np.diff(s.values)).mean()) if len(s) > 1 else 1.0)
            .reset_index(name="naive_mae")
        ).set_index("unique_id")["naive_mae"].clip(lower=1e-6)

        meta   = {"unique_id", "ds", "cutoff", "y"}
        m_cols = [c for c in cv.columns if c not in meta]
        cv_long = (
            cv.melt(id_vars=["unique_id", "ds", "y"], value_vars=m_cols,
                    var_name="model", value_name="yhat")
            .dropna(subset=["yhat"])
        )
        cv_long["mase"] = (
            (cv_long["y"] - cv_long["yhat"]).abs()
            / cv_long["unique_id"].map(naive_mae)
        )
        sel_map = (
            cv_long.groupby(["unique_id", "model"])["mase"].mean()
            .reset_index().sort_values(["unique_id", "mase"])
            .groupby("unique_id").first()["model"].to_dict()
        )

        print(f"\n── {L} weeks  ({n_folds} CV fold{'s' if n_folds>1 else ''}) ─────────────")
        print(f"  {'Model':<22} {'TestMAE':>8} {'TestBias':>10} {'CV MASE':>9}  "
              f"{'Wins vs WA':>10}")

        # Per-model stats in actual test
        rows = []
        wa_totals = None
        for mn in model_names:
            if mn not in fcast.columns:
                continue
            per_sku = fcast.groupby("unique_id")[mn].sum()
            ae   = (actual_totals - per_sku).abs()
            bias = (per_sku - actual_totals)
            cv_mase = cv_long[cv_long["model"] == mn]["mase"].mean()
            rows.append({
                "model": mn, "mae": ae.mean(), "bias": bias.mean(),
                "cv_mase": cv_mase, "totals": per_sku,
            })
            if mn == "WindowAverage":
                wa_totals = per_sku

        rows.sort(key=lambda r: r["mae"])
        for r in rows:
            n_sel = sum(1 for v in sel_map.values() if v == r["model"])
            if wa_totals is not None and r["model"] != "WindowAverage":
                wins_vs_wa = (
                    (actual_totals - r["totals"]).abs() <
                    (actual_totals - wa_totals).abs()
                ).sum()
                w_str = f"{wins_vs_wa:>3}/51"
            else:
                w_str = "  ref "
            print(f"  {r['model']:<22} {r['mae']:>8.2f} {r['bias']:>+10.2f} "
                  f"{r['cv_mase']:>9.4f}  {w_str}   (CV selected: {n_sel})")

        # What's the actual average demand growth rate?
        cv_dates   = sorted(cv["ds"].unique())
        cv_actual  = test_df[test_df["ds"].isin(cv_dates)]["y"].mean() if cv_dates else float("nan")
        test_actual = test_df["y"].mean()
        last4_rate  = recent_rate[recent_rate.index.isin(full_uids)].mean() * TEST_WEEKS
        print(f"\n  Demand context:")
        print(f"    Last 4-week avg rate × 10:  {last4_rate:.1f} units (naive run-rate forecast)")
        print(f"    CV fold actual mean/wk:     {cv_actual:.2f}")
        print(f"    True test actual mean/wk:   {test_actual:.2f}")
        print(f"    Growth CV→test:             {test_actual - cv_actual:+.2f} u/wk")


if __name__ == "__main__":
    main()
