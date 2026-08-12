#!/usr/bin/env python3
"""v18: does the tuned hyperparameter set help the v11 HYBRID, and which arm?

    .venv/bin/python scripts/ml_39_v18_tune_hybrid.py

No database. Development windows only; the final test window is not touched.

Why re-open v10
---------------
v10 searched hyperparameters for the architecture of the time: ONE shared model
on FEATURES_V1 across all smooth SKUs. It was rejected and the defaults kept.
v11 then replaced that architecture with a hybrid, a shared short model plus a
dedicated long model on FEATURES_V11_LONG, and the search was never repeated
against it. A setting that does nothing for one global model can matter for a
model fitted to 55 long SKUs, which is a different bias-variance problem
entirely.

So this is not a re-run of v10. It is the same question asked of the model that
actually exists.

The search itself does not need repeating. `outputs/reports/tune_wide.csv` holds
81 configurations scored on the INTERNAL VALIDATION SLICE, which is SKUs held
out inside the training period and costs no development window. Those results
are sound: 81 configurations produced 81 distinct scores, which could not happen
if the settings had not been applied (Section 4.33 records why that needed
checking). This takes the winner and asks where, if anywhere, it belongs.

Arms
----
    v11          both models on the defaults. The incumbent.
    short-tuned  tuned settings on the shared short model only
    long-tuned   tuned settings on the dedicated long model only
    both-tuned   tuned settings on both

Separating the arms matters because the two models have opposite problems. The
short model sees thousands of SKU-weeks and can afford capacity; the long model
sees 55 to 62 SKUs and has degraded under every perturbation tried so far
(Sections 4.30, 4.32 and the v15 to v17 entries). A configuration that helps one
may well hurt the other, and testing them together would let the effects cancel
and report a tie.

Pass criteria, stated before running
------------------------------------
Section 1.5 in full, because this is a design change and not a correctness fix:
a consistent sign across all three development windows and a three-window mean
improvement of at least 0.01, with borderline calls going to twice the bootstrap
standard error of the paired difference.

Judged per segment against the arm that touches it. short-tuned is judged on
smooth/short, long-tuned on smooth/long. both-tuned is reported for completeness
and cannot by itself adopt anything, since a gain there that is absent from both
single-arm results would be the two effects cancelling rather than compounding.

Disconfirming evidence, also pre-registered: if the tuned arms halt at
materially the same tree counts as the defaults, the settings are not changing
the fit and a null result says nothing about tuning, only that this particular
configuration was inert on this data.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.ml.dataset import dev_splits, load_weekly, stratified_val_skus  # noqa: E402
from src.ml.evaluate import bootstrap_delta, is_significant, score, score_table  # noqa: E402
from src.ml.model import (FEATURES_V1, FEATURES_V11_LONG, RatioLGBM,  # noqa: E402
                          long_sku_set, structural_baseline)
from src.ml.reference import warn_if_stale  # noqa: E402

WINDOWS = ["Mar-May", "Dec-Feb", "Oct-Dec"]
TUNE_CSV = ROOT / "outputs" / "reports" / "tune_wide.csv"


def winner() -> tuple[dict, int]:
    """Best configuration by validation L1, and its patience."""
    d = pd.read_csv(TUNE_CSV).sort_values("val_l1")
    w = d.iloc[0]
    params = dict(
        learning_rate=float(w.learning_rate), num_leaves=int(w.num_leaves),
        min_child_samples=int(w.min_child_samples),
        colsample_bytree=float(w.colsample_bytree), subsample=float(w.subsample),
        reg_alpha=float(w.reg_alpha), reg_lambda=float(w.reg_lambda),
    )
    if params["subsample"] < 1.0:
        params["subsample_freq"] = 1
    return params, int(w.patience)


def build(split, profiles, longs, val_all, val_long, short_p, long_p, pat):
    """v11 exactly as ml_22 builds it, with optional per-arm overrides."""
    ms = RatioLGBM(split.horizon, FEATURES_V1, deseas_features=True, deseas_all=True,
                   patience=pat if short_p else 100, params=short_p)
    ms.fit(split.train, profiles, split.cutoff, val_all)
    preds = ms.predict(split.train, profiles, split.cutoff)
    ml = RatioLGBM(split.horizon, FEATURES_V11_LONG, deseas_features=True,
                   deseas_all=True, uids=longs,
                   patience=pat if long_p else 100, params=long_p)
    ml.fit(split.train, profiles, split.cutoff, val_long)
    preds = pd.concat([preds[~preds["unique_id"].isin(longs)],
                       ml.predict(split.train, profiles, split.cutoff)],
                      ignore_index=True)
    return preds, ms.model.best_iteration_, ml.model.best_iteration_


def main() -> int:
    warn_if_stale()
    params, pat = winner()
    print(f"tuned configuration (validation-slice winner), patience={pat}:")
    for k, v in params.items():
        print(f"    {k:<20}{v}")
    print()

    weekly, profiles = load_weekly()
    smooth = set(profiles.loc[profiles["bucket"] == "smooth", "unique_id"])
    ws = weekly[weekly["unique_id"].isin(smooth)]

    ARMS = {"v11": (None, None), "short-tuned": (params, None),
            "long-tuned": (None, params), "both-tuned": (params, params)}
    deltas: dict[str, dict[str, dict]] = {a: {} for a in ARMS if a != "v11"}

    for split, name in zip(dev_splits(ws, n=3), WINDOWS):
        print(f"{'=' * 72}\n{name}  {split}")
        longs = long_sku_set(profiles, split.cutoff) & set(split.train["unique_id"])
        # Drawn ONCE per window and shared by every arm. Drawing per arm lets the
        # draw's noise land on one side of the comparison, which is a mistake
        # this project has already made once.
        val_all = stratified_val_skus(split.train, profiles)
        val_long = stratified_val_skus(
            split.train[split.train["unique_id"].isin(longs)], profiles)

        preds, results = {}, {}
        for arm, (sp, lp) in ARMS.items():
            p, ts, tl = build(split, profiles, longs, val_all, val_long, sp, lp, pat)
            preds[arm] = p
            results[arm] = score(p, split, profiles)
            print(f"  {arm:<12} trees short={ts:<5} long={tl}")

        results["baseline"] = score(
            structural_baseline(split.train, split.test, profiles, split.cutoff),
            split, profiles)
        print("\n  RAW per-segment results:")
        print(score_table(results).to_string())

        for arm in deltas:
            for seg in ("short", "long"):
                d = bootstrap_delta(preds[arm], preds["v11"], split, profiles, segment=seg)
                deltas[arm].setdefault(seg, {})[name] = d
                print(f"  {arm:<12} {seg:<5} vs v11: {d['delta']:+.4f}  se {d['se']:.4f}"
                      f"  {'SIG' if is_significant(d) else ''}")
        print()

    print(f"{'=' * 72}\nSECTION 1.5 VERDICT (negative delta = tuned is better)\n")
    for arm in deltas:
        for seg in ("short", "long"):
            v = [deltas[arm][seg][w]["delta"] for w in WINDOWS]
            mean = sum(v) / len(v)
            consistent = all(x < 0 for x in v) or all(x > 0 for x in v)
            met = consistent and mean <= -0.01
            judged = ((arm == "short-tuned" and seg == "short")
                      or (arm == "long-tuned" and seg == "long"))
            tag = "" if judged else "   (reported, not judged)"
            print(f"  {arm:<12} {seg:<5} " + "  ".join(f"{x:+.4f}" for x in v)
                  + f"   mean {mean:+.4f}  consistent {str(consistent):<5} "
                  + ("ADOPT" if met else "reject") + tag)
    print("\nAdoption needs a consistent sign AND a mean improvement >= 0.0100 on the")
    print("segment the arm actually touches. both-tuned cannot adopt on its own.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
