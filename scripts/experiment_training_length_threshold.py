#!/usr/bin/env python3
"""
Training-length threshold experiment.

Question: at what training length do our statistical models beat V1?

Design:
  - Fixed test window (same 10 held-out weeks for every truncation length)
  - Fixed SKU set (all 51 full-history smooth SKUs; all have ≥92 weeks, so
    the sample composition never changes across lengths)
  - Truncate from the recent end — keep the last L weeks before the cutoff
  - V1 also truncated to the same window (filter raw orders to cutoff - L*7 days)
  - n_windows=1 CV throughout for consistent model selection
  - Per-volume-tier breakdown

Training lengths tested: 26, 39, 52, 65, 78 weeks
"""
import sys, time
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
from compare_v1 import build_cumsum_index, v1_forecast

PROCESSED_DIR = ROOT / "data/processed"

TRAIN_LENGTHS = [26, 39, 52, 65, 78]   # weeks; max feasible is ~82 (92 total - 10 fold min)

# Volume tier boundaries (units/week average over full training period)
VOL_TIERS = [(0, 15, "low <15"), (15, 60, "med 15–60"), (60, 9999, "high >60")]

# AutoETS (damped) needs fold train ≥ 10; with n_windows=1, h=10 → series ≥ 20 weeks
AUTOETS_MIN_LEN = 20


def build_models():
    candidates      = get_models("smooth", "full")
    candidate_names = {type(m).__name__ for m in candidates}
    baselines       = [b for b in get_baselines("smooth", "full")
                       if type(b).__name__ not in candidate_names]
    full    = candidates + baselines
    no_ets  = [m for m in full if type(m).__name__ != "AutoETS"]
    return full, no_ets


def select_by_mase(cv_df: pd.DataFrame, train_df: pd.DataFrame) -> dict:
    naive_mae = (
        train_df.sort_values(["unique_id", "ds"])
        .groupby("unique_id")["y"]
        .apply(lambda s: float(np.abs(np.diff(s.values)).mean()) if len(s) > 1 else 1.0)
        .reset_index(name="naive_mae")
    )
    naive_mae["naive_mae"] = naive_mae["naive_mae"].clip(lower=1e-6)
    meta   = {"unique_id", "ds", "cutoff", "y"}
    m_cols = [c for c in cv_df.columns if c not in meta]
    long   = (
        cv_df.melt(id_vars=["unique_id", "ds", "y"], value_vars=m_cols,
                   var_name="model", value_name="yhat")
        .dropna(subset=["yhat"])
        .merge(naive_mae, on="unique_id", how="left")
    )
    long["mase"] = (long["y"] - long["yhat"]).abs() / long["naive_mae"]
    return (
        long.groupby(["unique_id", "model"])["mase"].mean()
        .reset_index().sort_values(["unique_id", "mase"])
        .groupby("unique_id").first()["model"].to_dict()
    )


def pick_total(fcast: pd.DataFrame, sel_map: dict) -> dict:
    totals = {}
    for uid, uid_f in fcast.groupby("unique_id"):
        uid_f = uid_f.sort_values("ds")
        mn    = sel_map.get(uid, "HistoricAverage")
        if mn in uid_f.columns:
            vals = uid_f[mn]
        else:
            avail = [c for c in uid_f.columns if c not in {"unique_id", "ds"}]
            vals  = uid_f[avail[0]] if avail else pd.Series([np.nan])
        totals[uid] = float(vals.sum())
    return totals


def run_one_length(train_full: pd.DataFrame, length: int,
                   models_full, models_no_ets) -> tuple[dict, dict]:
    """
    Truncate train_full to `length` most-recent weeks per SKU,
    run CV + fit + predict. Returns (yhat_totals_dict, sel_map).

    n_windows is adaptive: as many folds as the length allows while keeping
    the earliest fold's training set ≥ AUTOETS_MIN_LEN rows.
    """
    train = (
        train_full
        .sort_values(["unique_id", "ds"])
        .groupby("unique_id", group_keys=False)
        .apply(lambda g: g.tail(length))
        .reset_index(drop=True)
    )

    # Max folds = how many TEST_WEEKS-sized steps fit before the first fold
    # goes below AUTOETS_MIN_LEN training rows.
    n_windows = max(1, (length - AUTOETS_MIN_LEN) // TEST_WEEKS)

    lens        = train.groupby("unique_id")["ds"].count()
    ets_uids    = lens[lens >= AUTOETS_MIN_LEN].index.tolist()
    no_ets_uids = lens[lens < AUTOETS_MIN_LEN].index.tolist()

    cv_parts    = []
    fcast_parts = []

    for uid_group, mset in [(ets_uids, models_full), (no_ets_uids, models_no_ets)]:
        if not uid_group:
            continue
        grp     = train[train["unique_id"].isin(uid_group)].copy()
        grp_des = deseasonalize(grp)

        sf_cv   = StatsForecast(models=mset, freq=FREQUENCY, n_jobs=-1)
        cv      = sf_cv.cross_validation(
            df=grp_des[["unique_id", "ds", "y"]],
            h=TEST_WEEKS, n_windows=n_windows, step_size=TEST_WEEKS,
        )
        cv_parts.append(reseasonalize(cv))

        sf_fit  = StatsForecast(models=mset, freq=FREQUENCY, n_jobs=-1)
        sf_fit.fit(grp_des[["unique_id", "ds", "y"]])
        pred    = sf_fit.predict(h=TEST_WEEKS)
        pred["ds"] = pd.to_datetime(pred["ds"])
        fcast_parts.append(reseasonalize(pred))

    cv_df  = pd.concat(cv_parts, ignore_index=True)
    fcast  = pd.concat(fcast_parts, ignore_index=True)
    sel    = select_by_mase(cv_df, train)
    totals = pick_total(fcast, sel)
    return totals, sel


def main():
    print("Loading data...")
    weekly   = pd.read_parquet(PROCESSED_DIR / "sales_clean.parquet")
    profiles = pd.read_csv(PROCESSED_DIR / "sku_profiles.csv")
    raw_full = pd.read_parquet(PROCESSED_DIR / "orders_raw.parquet")
    weekly["ds"]            = pd.to_datetime(weekly["ds"])
    profiles["train_start"] = pd.to_datetime(profiles["train_start"])
    raw_full["order_date"]  = pd.to_datetime(raw_full["order_date"])

    all_weeks  = sorted(weekly["ds"].unique())
    trimmed    = all_weeks[:-TRIM_TRAILING_WEEKS]
    test_start = pd.Timestamp(trimmed[-TEST_WEEKS])
    cutoff     = pd.Timestamp(trimmed[-(TEST_WEEKS + 1)])
    test_wks   = [w for w in trimmed if w >= test_start]

    full_uids = profiles.loc[
        (profiles["bucket"] == "smooth") & (profiles["history_length"] == "full"),
        "unique_id",
    ].tolist()

    # Full training data (no length restriction yet) — trimmed to SKU train_start
    train_full = _trim_to_train_start(
        weekly[weekly["ds"].isin(trimmed) & (weekly["ds"] < test_start)].copy(), profiles
    )
    train_full = train_full[train_full["unique_id"].isin(full_uids)].copy()

    # Verify all SKUs have enough history for the max length
    lens = train_full.groupby("unique_id")["ds"].count()
    max_len = max(TRAIN_LENGTHS)
    eligible = lens[lens >= max_len].index.tolist()
    if len(eligible) < len(full_uids):
        dropped = set(full_uids) - set(eligible)
        print(f"  Dropping {len(dropped)} SKUs with < {max_len} weeks — sample stays fixed")
        full_uids = eligible
        train_full = train_full[train_full["unique_id"].isin(full_uids)]
    print(f"  Fixed SKU set: {len(full_uids)} full-history smooth SKUs")
    print(f"  Test window: {test_start.date()} → {pd.Timestamp(trimmed[-1]).date()} ({TEST_WEEKS} weeks)\n")

    # Volume tiers — based on FULL training period (stable across all lengths)
    weekly_rate = (
        train_full.groupby("unique_id")["y"].mean()  # avg units per week
    )
    def vol_tier(rate):
        for lo, hi, label in VOL_TIERS:
            if lo <= rate < hi:
                return label
        return VOL_TIERS[-1][2]
    tier_map = weekly_rate.apply(vol_tier).to_dict()
    print("Volume tier breakdown (of fixed SKU set):")
    for _, _, label in VOL_TIERS:
        n = sum(1 for v in tier_map.values() if v == label)
        rates = [r for uid, r in weekly_rate.items() if tier_map[uid] == label]
        print(f"  {label:12}  {n:>3} SKUs  avg rate {np.mean(rates):.1f} u/wk")
    print()

    # Actuals (fixed test window)
    test_df       = weekly[weekly["ds"].isin(test_wks) & weekly["unique_id"].isin(full_uids)]
    actual_totals = test_df.groupby("unique_id")["y"].sum().to_dict()

    models_full, models_no_ets = build_models()
    print(f"Models (full): {[type(m).__name__ for m in models_full]}")
    print(f"Models (no ETS): {[type(m).__name__ for m in models_no_ets]}\n")

    # ── Sweep ─────────────────────────────────────────────────────────────────
    all_rows = []   # (length, uid, actual, our, v1, model)

    for length in TRAIN_LENGTHS:
        t0 = time.time()

        # --- Model ---
        our_totals, sel_map = run_one_length(train_full, length, models_full, models_no_ets)

        # --- V1: rebuild index on truncated raw orders ---
        v1_start = cutoff - pd.Timedelta(weeks=length)
        raw_trunc = raw_full[raw_full["order_date"] >= v1_start].copy()
        v1_index  = build_cumsum_index(raw_trunc)
        v1_totals = {}
        for uid in full_uids:
            try:
                v1_totals[uid] = v1_forecast(v1_index, uid, cutoff)
            except Exception:
                pass

        n_folds = max(1, (length - AUTOETS_MIN_LEN) // TEST_WEEKS)
        elapsed = time.time() - t0
        print(f"Length {length:>3} weeks  folds={n_folds}  ({elapsed:.1f}s)  "
              f"V1: {len(v1_totals)}/{len(full_uids)}  "
              f"Models: {pd.Series(sel_map).value_counts().to_dict()}")

        for uid in full_uids:
            actual = actual_totals.get(uid, 0)
            our    = our_totals.get(uid, np.nan)
            v1     = v1_totals.get(uid, np.nan)
            all_rows.append({
                "length": length,
                "uid":    uid,
                "actual": actual,
                "our":    our,
                "v1":     v1,
                "model":  sel_map.get(uid, "?"),
                "vol_tier": tier_map.get(uid, "?"),
                "weekly_rate": float(weekly_rate.get(uid, 0)),
            })

    df = pd.DataFrame(all_rows)
    df["ae_our"] = (df["actual"] - df["our"]).abs()
    df["ae_v1"]  = (df["actual"] - df["v1"]).abs()
    df["bias_our"] = df["our"] - df["actual"]
    df["wins"]     = df["ae_our"] < df["ae_v1"]

    print()
    print("=" * 80)
    print("TRAINING LENGTH THRESHOLD — fixed test window, fixed 51-SKU set")
    print("=" * 80)
    print(f"\n{'Weeks':>6}  {'Our MAE':>8}  {'V1 MAE':>8}  {'Delta':>7}  "
          f"{'Wins':>8}  {'Bias':>8}  {'Top model'}")
    print("─" * 75)

    for length in TRAIN_LENGTHS:
        sub   = df[(df["length"] == length) & df["ae_our"].notna() & df["ae_v1"].notna()]
        mae_o = sub["ae_our"].mean()
        mae_v = sub["ae_v1"].mean()
        delta = mae_o - mae_v        # negative = our model wins
        wins  = sub["wins"].sum()
        bias  = sub["bias_our"].mean()
        top_m = sub["model"].value_counts().index[0] if not sub.empty else "?"
        marker = " ◀" if delta < 0 else ""
        print(f"{length:>6}  {mae_o:>8.2f}  {mae_v:>8.2f}  {delta:>+7.2f}  "
              f"{wins:>3}/{len(sub):>2}  {bias:>+8.2f}  {top_m}{marker}")

    # ── Per-volume-tier breakdown ──────────────────────────────────────────────
    print()
    print("─" * 80)
    print("PER-VOLUME-TIER  (Delta = Our MAE − V1 MAE; negative = our model wins)")
    print("─" * 80)

    for _, _, tier_label in VOL_TIERS:
        tier_df = df[df["vol_tier"] == tier_label]
        n_skus  = tier_df["uid"].nunique()
        if n_skus == 0:
            continue
        print(f"\n  {tier_label}  ({n_skus} SKUs)")
        print(f"  {'Weeks':>6}  {'Our MAE':>8}  {'V1 MAE':>8}  {'Delta':>7}  {'Wins':>8}  {'Bias':>8}")
        print(f"  {'─'*60}")
        for length in TRAIN_LENGTHS:
            sub   = tier_df[(tier_df["length"] == length) & tier_df["ae_our"].notna() & tier_df["ae_v1"].notna()]
            if sub.empty:
                continue
            mae_o = sub["ae_our"].mean()
            mae_v = sub["ae_v1"].mean()
            delta = mae_o - mae_v
            wins  = sub["wins"].sum()
            bias  = sub["bias_our"].mean()
            marker = " ◀" if delta < 0 else ""
            print(f"  {length:>6}  {mae_o:>8.2f}  {mae_v:>8.2f}  {delta:>+7.2f}  "
                  f"{wins:>3}/{len(sub):>2}  {bias:>+8.2f}{marker}")

    # ── Model selection evolution ─────────────────────────────────────────────
    print()
    print("─" * 80)
    print("MODEL SELECTION BY TRAINING LENGTH")
    print("─" * 80)
    model_names = ["AutoARIMA", "AutoETS", "WindowAverage", "Naive", "HistoricAverage", "SeasonalNaive"]
    header = f"{'Weeks':>6}  " + "".join(f"{m[:10]:>12}" for m in model_names)
    print(header)
    for length in TRAIN_LENGTHS:
        sub    = df[df["length"] == length]
        counts = sub.drop_duplicates("uid")["model"].value_counts()
        row    = f"{length:>6}  " + "".join(f"{counts.get(m, 0):>12}" for m in model_names)
        print(row)


if __name__ == "__main__":
    main()
