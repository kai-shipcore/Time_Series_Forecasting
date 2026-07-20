#!/usr/bin/env python3
"""ML experiment 10: the statistical prototype, scored by the ML harness.

Closes backlog item 1 (design Section 5.4). Sections 1.2 and 1.6 name the
prototype as the accuracy bar the LightGBM track must clear, but every version
so far has been compared against v-base, which is the FLOOR (a 12-week moving
average) and not the bar. This runs the prototype on the same splits and scores
it through the same `evaluate.score`, so the version log finally has the number
that decides adoption.

Nothing here reimplements the prototype. It reuses the production pipeline:

    src.backtest.backtest              phase 1, CV on training data
    src.selector.select                phase 2, model choice per SKU
    run_test_evaluation.refit_and_predict   phases 3+4, refit and forecast

The only new logic is (a) restricting the frame per evaluation window and
(b) reshaping the output for `evaluate.score`.

How the window is controlled: `backtest()` and `refit_and_predict()` both derive
their split from the END of the frame they are given, holding out the last
TEST_WEEKS. So truncating `weekly` to a dev window's last test week makes them
reproduce exactly that window, with selection re-run on that window's training
data only. This matters: `outputs/reports/selection.csv` is regenerated with all
data through today, and reusing it for an older cutoff would let the model
choice see the future.

Side-effect safety: `backtest()` and `select()` write cv_results.parquet,
cv_metrics.csv and selection.csv into outputs/reports/. Those are production
artifacts, so this script redirects them to a temp directory and restores the
paths afterwards. It never writes to outputs/reports/.
"""
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

import src.backtest as bt
import src.selector as sel
from run_test_evaluation import refit_and_predict
from src.ml.dataset import (
    asof_history_length, dev_splits, eligible_skus, load_weekly,
)
from src.ml.evaluate import score, score_table
from src.ml.model import structural_baseline

WINDOWS = ["Mar-May", "Dec-Feb", "Oct-Dec"]


def asof_profiles(profiles: pd.DataFrame, cutoff) -> pd.DataFrame:
    """profiles with history_length recomputed as of `cutoff`.

    The pipeline routes SKUs by `history_length`: full/medium go through CV
    model selection, short gets a fixed default. The snapshot labels are
    present-day, so at an older cutoff a SKU that is "medium" today may have had
    only a few weeks of history, and sending it through CV is both anachronistic
    and unstable (AutoETS raises on series that short). This is the same as-of
    correction the ML harness applies when scoring, design Section 4.15.

    bucket is left as-is: recomputing smooth/intermittent as of a cutoff needs
    the full profiling statistics, and the ML harness has the same limitation.
    """
    out = profiles.copy()
    asof = asof_history_length(profiles, cutoff)
    out["history_length"] = out["unique_id"].map(asof).astype("object")
    return out


def _redirect_outputs(tmp: Path):
    """Point the pipeline's artifact writes at a temp dir; return an undo fn."""
    saved = (bt.CV_PATH, bt.TEST_PATH, sel.CV_PATH, sel.OUTPUTS_REPORTS)
    bt.CV_PATH = tmp / "cv_results.parquet"
    bt.TEST_PATH = tmp / "test_set.parquet"
    sel.CV_PATH = tmp / "cv_results.parquet"
    sel.OUTPUTS_REPORTS = tmp

    def undo():
        bt.CV_PATH, bt.TEST_PATH, sel.CV_PATH, sel.OUTPUTS_REPORTS = saved
    return undo


def prototype_predictions(weekly, profiles, split) -> pd.DataFrame:
    """Run the production pipeline for one window; return unique_id/ds/yhat.

    `refit_and_predict` yields one 10-week TOTAL per SKU. `evaluate.score` sums
    per SKU before scoring, so a single row carrying the total, dated inside the
    test window, scores identically to ten weekly rows. This is the same shape
    ml_02 uses for the V1 benchmark.
    """
    # Restrict to the SKUs the scorer will actually score. `evaluate.score`
    # drops anything with under MIN_SIM_HISTORY_WEEKS of history at the cutoff
    # (design Section 4.15), so forecasting the rest produces output that is
    # discarded. It also avoids handing AutoETS series of 1-2 weeks, which it
    # rejects outright; that never arises in production because by now every SKU
    # has enough history, but it does at older cutoffs.
    keep = eligible_skus(profiles, split.cutoff)
    frame = weekly[
        weekly["unique_id"].isin(keep) & (weekly["ds"] <= split.test["ds"].max())
    ].copy()
    profiles = asof_profiles(profiles[profiles["unique_id"].isin(keep)], split.cutoff)

    tmp = Path(tempfile.mkdtemp(prefix="proto_bench_"))
    undo = _redirect_outputs(tmp)
    try:
        bt.backtest(frame, profiles)
        selection = sel.select(frame, profiles)
        fc = refit_and_predict(frame, profiles, selection)
    finally:
        undo()
        shutil.rmtree(tmp, ignore_errors=True)

    out = fc[["unique_id", "yhat_total"]].rename(columns={"yhat_total": "yhat"})
    out = out.dropna(subset=["yhat"])
    out["ds"] = split.test["ds"].min()
    return out[["unique_id", "ds", "yhat"]]


def main():
    weekly, profiles = load_weekly()
    smooth = profiles.loc[profiles["bucket"] == "smooth", "unique_id"]
    weekly = weekly[weekly["unique_id"].isin(set(smooth))]

    for split, name in zip(dev_splits(weekly, n=3), WINDOWS):
        print(f"\n{'=' * 66}\n{name}  {split}")
        proto = prototype_predictions(weekly, profiles, split)
        base = structural_baseline(split.train, split.test, profiles, split.cutoff)

        results = {
            "prototype": score(proto, split, profiles),
            "v-base": score(base, split, profiles),
        }
        print(f"\npooled WAPE ({len(proto)} SKUs forecast by the prototype):")
        print(score_table(results).to_string())
        print("\nbias% by segment:")
        print(pd.DataFrame(
            {n: t.set_index("segment")["bias_pct"] for n, t in results.items()}
        ).to_string())


if __name__ == "__main__":
    main()
