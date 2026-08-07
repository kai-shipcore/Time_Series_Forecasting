#!/usr/bin/env python3
"""ML experiment 29: v15 — seasonal factor blended across the days a week covers.

Hypothesis, criteria and recorded expectation are in the design doc, Section 6,
v15, written before this ran. Read them there rather than here.

The criterion is NOT Section 1.5's, and that is deliberate. This is a
correctness fix: a week running Tuesday 28 July through Monday 3 August is
labelled 3 August and currently takes August's multiplier with six of its seven
days in July. The predicted effect is a factor change of about 0.02 on a fifth
of weeks, which is very unlikely to move pooled WAPE by the 0.01 the adoption
rule wants. Adopting a neutral performance change would be wrong; adopting a
neutral correctness fix is right, because the present behaviour cannot be
defended whatever it scores.

So: adopt unless a development window and segment REGRESSES by more than two
bootstrap standard errors. A tie adopts. An improvement is welcome and is not
required.

Both arms are scored against the same split, on the same data, with the same
early-stopping validation draw, so `evaluate.bootstrap_delta` applies directly
(unlike experiment 26, where the two arms had different actuals and needed a
paired cross-arm bootstrap).

    .venv/bin/python scripts/ml_29_v15_seasonal_blend.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

import src.ml.seasonal as seas  # noqa: E402
from ml_26_week_boundary_ab import (EXCLUDED_CELLS, WINDOW_NAMES,  # noqa: E402
                                    _rel, draw_val, fit_v11)
from src.ml.dataset import dev_splits, load_weekly  # noqa: E402
from src.ml.evaluate import bootstrap_delta, score  # noqa: E402

OUT_BY_MODE = {
    "full": ROOT / "outputs" / "reports" / "ml_v15_seasonal_blend.csv",
    "holiday": ROOT / "outputs" / "reports" / "ml_v16_holiday_blend.csv",
}

# Section 1.5's noise rule, used here as a floor rather than a bar: a cell fails
# only if it gets WORSE by more than this many standard errors.
SE_MULTIPLE = 2.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=("full", "holiday"), default="full",
                    help="'full' is v15 (monthly + holiday blended); "
                         "'holiday' is v16 (holiday only, monthly untouched)")
    args = ap.parse_args()
    arm_name = {"full": "v15", "holiday": "v16"}[args.mode]

    weekly, profiles = load_weekly()
    smooth = set(profiles.loc[profiles["bucket"] == "smooth", "unique_id"])
    weekly = weekly[weekly["unique_id"].isin(smooth)]

    if seas.ML_SEASONAL_BLEND != "off":
        print(f"config.ML_SEASONAL_BLEND is {seas.ML_SEASONAL_BLEND!r}. This "
              "experiment sets the mode itself and needs 'off' at rest, so the "
              "baseline arm is the recorded one. Aborting.")
        return 1

    rows, boots = [], []
    for i, split in enumerate(dev_splits(weekly, n=3)):
        win = WINDOW_NAMES[i]
        # One validation draw for both arms. The data is identical here so the
        # draw would come out the same anyway, but passing it explicitly means
        # the comparison cannot depend on that continuing to be true.
        val = draw_val(split, profiles)

        preds = {}
        for arm, mode in (("v11", "off"), (arm_name, args.mode)):
            seas.ML_SEASONAL_BLEND = mode
            try:
                preds[arm] = fit_v11(split, profiles, val=val)
            finally:
                seas.ML_SEASONAL_BLEND = "off"

        for arm, p in preds.items():
            for _, r in score(p[["unique_id", "ds", "yhat"]], split,
                              profiles).iterrows():
                rows.append({"arm": arm, "window": win, "segment": r["segment"],
                             "n_skus": r["n_skus"],
                             "actual_units": r["actual_units"],
                             "pooled_wape": r["pooled_wape"],
                             "bias_pct": r["bias_pct"]})

        for seg in ("short", "long", None):
            name = f"smooth/{seg}" if seg else "TOTAL"
            b = bootstrap_delta(preds[arm_name], preds["v11"], split, profiles,
                                segment=seg)
            b.update({"window": win, "segment": name,
                      "excluded": (win, name) in EXCLUDED_CELLS})
            boots.append(b)
        print(f"  {win}: both arms fitted and scored")

    out = pd.DataFrame(rows)
    OUT_BY_MODE[args.mode].parent.mkdir(parents=True, exist_ok=True)
    OUT = OUT_BY_MODE[args.mode]
    out.to_csv(OUT, index=False)

    wide = out.pivot_table(index=["segment", "window"], columns="arm",
                           values="pooled_wape")
    wide["delta"] = wide[arm_name] - wide["v11"]
    print(f"\n{'=' * 70}\npooled WAPE  (delta = {arm_name} - v11; negative is better)")
    print(wide.round(4).to_string())

    bt = pd.DataFrame(boots)
    print(f"\n{'=' * 70}\nbootstrap of the difference, 1000 paired SKU resamples")
    print(f"  {'window':<9} {'segment':<13} {'delta':>9} {'se':>7} "
          f"{'95% CI':>20}  verdict")
    worst = None
    for _, r in bt.iterrows():
        # delta = v15 - v11, so POSITIVE is a regression.
        regress = r["delta"] > SE_MULTIPLE * r["se"]
        tag = ("excluded" if r["excluded"]
               else "REGRESSION" if regress
               else "improves" if r["delta"] < -SE_MULTIPLE * r["se"]
               else "tie")
        if not r["excluded"] and regress:
            worst = r
        print(f"  {r['window']:<9} {r['segment']:<13} {r['delta']:>+9.4f} "
              f"{r['se']:>7.4f}  [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  {tag}")

    print(f"\n{'=' * 70}\nVERDICT against the pre-registered criterion")
    print(f"  Adopt unless a judged cell regresses by more than "
          f"{SE_MULTIPLE:.0f} standard errors.")
    if worst is None:
        print("\n  ADOPT. No judged cell regresses beyond noise.")
        print("  Set ML_SEASONAL_BLEND = True in config.py, then re-run the")
        print("  versions still quoted (v11, structural baseline, naive) and")
        print("  the phase sweep, which is v15's secondary criterion.")
    else:
        print(f"\n  REJECT. {worst['segment']} in {worst['window']} regresses "
              f"{worst['delta']:+.4f} against {SE_MULTIPLE * worst['se']:.4f}.")
        print("  Leave ML_SEASONAL_BLEND = False and record the result.")

    print(f"\nwrote {_rel(OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
