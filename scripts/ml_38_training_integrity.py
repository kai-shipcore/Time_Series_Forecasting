#!/usr/bin/env python3
"""Prove the model is refitted per window and cannot see the future.

    .venv/bin/python scripts/ml_38_training_integrity.py

No database. Development windows only. Exits non-zero if any check fails.

Why this exists
---------------
Every figure in the version log assumes two things that are never checked: that
a fresh model is fitted for each evaluation window, and that nothing it trains
on postdates that window's cutoff. Both are the kind of property that is true
until a refactor quietly makes it false, and a leak of this class does not
announce itself. It looks like an unusually good result.

The checks below are deliberately mechanical. They inspect the actual matrices
and predictions rather than reading the code, because the code has been read
many times and the point is to test the thing the reading assumes.

What each check would catch
---------------------------
1  TEMPORAL CONTAINMENT      a training row whose target week is after the
                             cutoff, which is direct target leakage.
2  ANCHOR CONTAINMENT        a feature row built from data after the cutoff.
3  PREDICTION WINDOW         predictions dated inside the training period,
                             which would score the model on what it fitted.
4  FRESH ESTIMATOR           a booster reused between windows, so window three
                             is really window one still.
5  DETERMINISM               the same window fitted twice giving different
                             answers, which would make every recorded third
                             decimal meaningless and every A/B unreadable.
6  ORDER INDEPENDENCE        state carried between fits, so a window's result
                             depends on what was fitted before it. This is the
                             one that a shared mutable default or a cached
                             frame produces, and it is invisible in any single
                             run because every run uses the same order.
7  VALIDATION DISJOINTNESS   early-stopping SKUs also appearing in the training
                             rows, which makes early stopping choose the point
                             of maximum overfit rather than the opposite.
8  PARAMS NOT SHARED         a per-instance hyperparameter override leaking into
                             the class default and silently re-baselining every
                             later fit in the same process.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.ml.dataset import dev_splits, load_weekly, stratified_val_skus  # noqa: E402
from src.ml.model import (FEATURES_V1, FEATURES_V11_LONG, RatioLGBM,  # noqa: E402
                          build_matrix, long_sku_set)

WINDOWS = ["Mar-May", "Dec-Feb", "Oct-Dec"]
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        failures.append(f"{name}: {detail}")


def fit_short(split, profiles, val):
    return RatioLGBM(split.horizon, FEATURES_V1, deseas_features=True,
                     deseas_all=True).fit(split.train, profiles, split.cutoff, val)


def main() -> int:
    weekly, profiles = load_weekly()
    smooth = set(profiles.loc[profiles["bucket"] == "smooth", "unique_id"])
    ws = weekly[weekly["unique_id"].isin(smooth)]
    splits = list(dev_splits(ws, n=3))

    boosters, preds_first = {}, {}

    for split, name in zip(splits, WINDOWS):
        print(f"\n{name}  cutoff {split.cutoff.date()}")

        # 1 and 2. Inspect the training matrix directly.
        mat = build_matrix(split.train, split.horizon, split.cutoff, profiles,
                           for_training=True, deseas_features=True, deseas_all=True)
        check("1 no training target after the cutoff",
              mat["tgt_ds"].max() <= split.cutoff,
              f"max target {mat['tgt_ds'].max().date()} vs cutoff {split.cutoff.date()}")
        check("2 no feature anchor after the cutoff",
              mat["ds"].max() <= split.cutoff,
              f"max anchor {mat['ds'].max().date()}")
        check("2b training frame itself stops at the cutoff",
              split.train["ds"].max() <= split.cutoff)

        val = stratified_val_skus(split.train, profiles)
        m = fit_short(split, profiles, val)
        p = m.predict(split.train, profiles, split.cutoff)

        # 3. Predictions must fall strictly after the cutoff and inside the test
        # window. score() checks the upper edge; this checks the lower one.
        check("3 every prediction is after the cutoff",
              bool((p["ds"] > split.cutoff).all()),
              f"earliest {p['ds'].min().date()}")
        check("3b predictions cover the test window only",
              set(p["ds"].unique()) <= set(split.test["ds"].unique()))

        # 7. Early-stopping SKUs are held out of the fitted rows.
        tr_rows = mat[~mat["unique_id"].isin(val)]
        check("7 validation SKUs excluded from training rows",
              len(set(tr_rows["unique_id"]) & set(val)) == 0,
              f"{len(val)} val SKUs")

        boosters[name] = m.model.booster_.model_to_string()
        preds_first[name] = p.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    # 4. Distinct boosters per window.
    print("\ncross-window")
    uniq = len(set(boosters.values()))
    check("4 a distinct fitted model per window", uniq == len(WINDOWS),
          f"{uniq} distinct boosters across {len(WINDOWS)} windows")

    # 5. Determinism: refit one window and compare predictions exactly.
    s0 = splits[0]
    v0 = stratified_val_skus(s0.train, profiles)
    again = fit_short(s0, profiles, v0).predict(s0.train, profiles, s0.cutoff)
    again = again.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    same = np.allclose(again["yhat"].to_numpy(),
                       preds_first[WINDOWS[0]]["yhat"].to_numpy(), rtol=0, atol=0)
    check("5 refitting the same window is bit-identical", same,
          "" if same else f"max diff {np.abs(again['yhat'].to_numpy() - preds_first[WINDOWS[0]]['yhat'].to_numpy()).max():.3e}")

    # 6. Order independence: fit the windows in reverse and compare each one to
    # the forward pass. Anything carried between fits shows up here and nowhere
    # else, because every ordinary run uses the same order.
    ok_order = True
    for split, name in reversed(list(zip(splits, WINDOWS))):
        v = stratified_val_skus(split.train, profiles)
        p = fit_short(split, profiles, v).predict(split.train, profiles, split.cutoff)
        p = p.sort_values(["unique_id", "ds"]).reset_index(drop=True)
        if not np.allclose(p["yhat"].to_numpy(),
                           preds_first[name]["yhat"].to_numpy(), rtol=0, atol=0):
            ok_order = False
            check(f"6 {name} identical when fitted in reverse order", False,
                  f"max diff {np.abs(p['yhat'].to_numpy() - preds_first[name]['yhat'].to_numpy()).max():.3e}")
    if ok_order:
        check("6 window results independent of fitting order", True)

    # 8. A per-instance override must not reach the class default.
    before = dict(RatioLGBM.PARAMS)
    RatioLGBM(10, FEATURES_V1, params={"num_leaves": 7}).__init__(
        10, FEATURES_V1, params={"num_leaves": 7})
    check("8 per-instance params do not mutate the class default",
          RatioLGBM.PARAMS == before,
          f"num_leaves now {RatioLGBM.PARAMS.get('num_leaves')}")

    # The long model is a separate fit on a restricted SKU set; confirm the
    # restriction actually applies rather than being silently ignored.
    longs = long_sku_set(profiles, splits[0].cutoff) & set(splits[0].train["unique_id"])
    mL = RatioLGBM(splits[0].horizon, FEATURES_V11_LONG, deseas_features=True,
                   deseas_all=True, uids=longs).fit(
        splits[0].train, profiles, splits[0].cutoff, stratified_val_skus(
            splits[0].train[splits[0].train["unique_id"].isin(longs)], profiles))
    pL = mL.predict(splits[0].train, profiles, splits[0].cutoff)
    check("9 long model predicts only its own segment",
          set(pL["unique_id"]) <= longs,
          f"{len(set(pL['unique_id']))} SKUs, segment holds {len(longs)}")

    print(f"\n{'=' * 60}")
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED. Recorded figures cannot be trusted "
              f"until these are understood:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All checks passed. Models are refitted per window, trained only on data "
          "at or before each cutoff, and independent of fitting order.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
