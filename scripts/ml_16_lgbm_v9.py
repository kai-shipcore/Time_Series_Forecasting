#!/usr/bin/env python3
"""ML experiment 16: v9 — holiday window ends mid-December, inside v3.

v9 = v3 with ML_HOLIDAY_END moved from (12,31) to (12,15). Two supporting
corrections land with it: holiday membership is decided on the days a week
covers rather than its label, and the ML factors are independent of the
prototype's, so neither the prototype nor V1 moves.

v6 tested the same window change at BASELINE level and was not adopted, failing
on the short segment. v3 succeeded once before precisely where the baseline
failed (Section 4.20), so the change deserves testing inside a model.

Basis is business knowledge, not fitting: promotions run late November to
mid-December and December 2024 predates that practice.

Pre-registered criteria: design doc Section 6, v9.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

import src.ml.seasonal as seas
from src.ml.dataset import dev_splits, load_weekly, stratified_val_skus
from src.ml.evaluate import bootstrap_delta, is_significant, score, score_table
from src.ml.model import FEATURES_V1, RatioLGBM, structural_baseline

from src.ml.reference import PROTOTYPE, warn_if_stale  # noqa: E402

warn_if_stale()


def run(split, profiles, val, end_md):
    """Fit and predict v3 under a given holiday window end."""
    saved = seas.ML_HOLIDAY_END
    seas.ML_HOLIDAY_END = end_md
    try:
        m = RatioLGBM(split.horizon, FEATURES_V1, deseas_features=True,
                      deseas_all=True).fit(split.train, profiles, split.cutoff, val)
        return m, m.predict(split.train, profiles, split.cutoff)
    finally:
        seas.ML_HOLIDAY_END = saved


def main():
    weekly, profiles = load_weekly()
    smooth = profiles.loc[profiles["bucket"] == "smooth", "unique_id"]
    weekly = weekly[weekly["unique_id"].isin(set(smooth))]

    for split, name in zip(dev_splits(weekly, n=3), PROTOTYPE):
        print(f"\n{'=' * 66}\n{name}  {split}")
        val = stratified_val_skus(split.train, profiles)
        m3, p3 = run(split, profiles, val, (12, 31))
        m9, p9 = run(split, profiles, val, (12, 15))

        results = {
            "baseline": score(structural_baseline(
                split.train, split.test, profiles, split.cutoff), split, profiles),
            "v3": score(p3, split, profiles),
            "v9": score(p9, split, profiles),
        }
        print(f"\n  v3 trees={m3.model.best_iteration_}, v9 trees={m9.model.best_iteration_}")
        print(f"  prototype (the bar): short {PROTOTYPE[name][0]:.4f}, "
              f"long {PROTOTYPE[name][1]:.4f}")
        print("\npooled WAPE:")
        print(score_table(results).to_string())
        print("\nbias% by segment:")
        print(pd.DataFrame(
            {n: t.set_index("segment")["bias_pct"] for n, t in results.items()}
        ).to_string())
        for seg in ("short", "long"):
            bd = bootstrap_delta(p9, p3, split, profiles, segment=seg)
            print(f"bootstrap v9-vs-v3 [{seg}]: {bd}  significant={is_significant(bd)}")


if __name__ == "__main__":
    main()
