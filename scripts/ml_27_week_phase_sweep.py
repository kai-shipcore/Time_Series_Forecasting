#!/usr/bin/env python3
"""ML experiment 27: all seven week phases, as a distribution rather than a choice.

Why this and not experiment 26
------------------------------
Experiment 26 compared two week boundaries and found v11's error higher on one
of them, consistently in direction but with a magnitude resting on a single cell
whose 95% interval spanned almost twelve to one. Two points cannot say whether
that is a property of the calendar, a property of the model, or one unlucky
draw. Seven can.

A week is seven consecutive days, and which day it starts on is a free choice.
There are exactly seven, and the project has now used two of them: Monday, by
decision, and Tuesday, by a pandas default nobody chose. Nothing distinguishes
the other five except that no one has looked at them.

What this produces, and what it must not be used for
----------------------------------------------------
The output is a SPREAD: for each model, the range of pooled WAPE across the
seven phases. That spread is the answer to "how much does this number depend on
a convention we invented", which is a question a reviewer will ask and which no
single figure can answer.

It is emphatically NOT a menu. Picking the best of seven phases would be taking
a maximum over seven noisy draws, which is a far worse version of the selection
problem than picking the better of two, and it would inflate the final test by
an amount nobody could estimate afterwards. The temptation is real and gets
stronger the wider the spread turns out to be, which is exactly why it is
written down here before the numbers exist.

The legitimate uses are three:

  1. Report the spread as a robustness figure alongside the headline WAPE.
  2. Compare v11's spread against the comparators'. They have no seasonal
     machinery and no lag features, so a model-specific sensitivity shows up as
     a wider band rather than as a single suspicious cell.
  3. Read the SHAPE of WAPE against phase. A smooth trend suggests something
     structural about how much of a week's demand sits at its edges. A spike at
     one or two phases suggests a calendar artifact, most obviously which bucket
     a holiday falls into, since a one-day move can put Christmas in a different
     week. Two points cannot distinguish these and seven can.

How the phases are constructed
------------------------------
Shifting the grouper's anchor day would change the week LABELS as well as the
spans, and the labels are what `dev_splits` anchors its windows to, so the
comparison would stop being like for like.

Instead the DATES move and the grouper stays fixed. For phase p, orders are
dated p days earlier and then bucketed with the usual W-MON, closed left,
labelled right. The bucket labelled L then holds real dates [L-7+p, L+p):

    p=0   Mon..Sun   the current production boundary
    p=1   Tue..Mon   the old pandas default, experiment 26's "old" arm
    p=2   Wed..Tue
    ...
    p=6   Sun..Sat

Every phase produces the identical set of week labels, so all seven get the same
evaluation windows, the same SKU population, the same pinned profiles and the
same early-stopping validation draw. Only the seven days inside each bucket
differ.

    .venv/bin/python scripts/ml_27_week_phase_sweep.py --orders-cache data/processed/_ab_orders.parquet

Writes outputs/reports/ml_week_phase_sweep.csv. Writes nothing else.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

from ml_01_naive_baseline import naive_mean12  # noqa: E402
from ml_26_week_boundary_ab import (WINDOW_NAMES, EXCLUDED_CELLS, _rel,  # noqa: E402
                                    draw_val, fit_v11, load_orders, prep)
from config import ML_DATA_SNAPSHOT  # noqa: E402
from src.ml.dataset import data_dir, dev_splits  # noqa: E402
from src.ml.evaluate import score  # noqa: E402

OUT = ROOT / "outputs" / "reports" / "ml_week_phase_sweep.csv"
# --reprofile writes elsewhere. Sending both runs to one path means the second
# silently destroys the first, and the two are meant to be read against each
# other. Identical mistake to the one already fixed in ml_26 with OUT_DIAG; it
# reappeared here because that fix was applied to the file it was found in
# rather than searched for.
OUT_REPROFILE = OUT.with_name("ml_week_phase_sweep_reprofiled.csv")

# Named by the weekday each phase's week starts on, which is the only thing
# that varies. Phase 0 is what production computes; phase 1 is what every
# recorded number in the Version Log was measured on.
PHASES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def weekly_for_phase(
    orders: pd.DataFrame, phase: int, skus: set, weeks: pd.DatetimeIndex
) -> pd.DataFrame:
    """Weekly totals with the week starting `phase` days after Monday.

    The dates move and the grouper stays put, so every phase emits the same
    week labels and the evaluation windows cannot drift between phases. See the
    module docstring for why that matters.
    """
    o = orders.copy()
    o["order_date"] = pd.to_datetime(o["order_date"]) - pd.Timedelta(days=phase)
    w = (
        o.groupby(
            ["link_master_sku",
             pd.Grouper(key="order_date", freq="W-MON", closed="left", label="right")],
            as_index=False,
        )["link_qty"]
        .sum()
        .rename(columns={"link_master_sku": "unique_id", "order_date": "ds",
                         "link_qty": "y"})
    )
    w = w[w["unique_id"].isin(skus) & w["ds"].isin(weeks)]
    grid = pd.MultiIndex.from_product([sorted(skus), weeks],
                                      names=["unique_id", "ds"])
    return (
        w.set_index(["unique_id", "ds"]).reindex(grid, fill_value=0)
        .reset_index().sort_values(["unique_id", "ds"]).reset_index(drop=True)
    )


def profiles_for_phase(weekly_all: pd.DataFrame, tmp: Path) -> pd.DataFrame:
    """Re-derive sku_profiles from one phase's own weekly series.

    Why this exists. The pinned `sku_profiles.csv` is held fixed across phases
    as the population control, and it was generated from Tuesday-binned data.
    `train_start`, `bucket`, `history_length` and the eligibility filter all
    encode phase 1's bucket edges, so phase 1 is scored against profiles that
    line up with its own buckets while every other phase is scored against
    profiles offset by p days. Holding profiles fixed controls the population
    and hands Tuesday a matched fit in the same gesture. This function is the
    test of whether that is where Tuesday's advantage comes from.

    `src.profile.profile` writes sku_profiles.csv into data/processed as a side
    effect, which would overwrite the live file once per phase. PROCESSED_DIR is
    rebound to a scratch directory for the duration. Restored in a finally, so a
    failure mid-sweep cannot leave the module pointing somewhere else.
    """
    import src.profile as prof_mod

    original = prof_mod.PROCESSED_DIR
    prof_mod.PROCESSED_DIR = tmp
    try:
        out = prof_mod.profile(weekly_all)
    finally:
        prof_mod.PROCESSED_DIR = original
    out["train_start"] = pd.to_datetime(out["train_start"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--orders-cache", type=Path, default=None)
    ap.add_argument("--skip-v11", action="store_true",
                    help="comparators only; no lightgbm required")
    ap.add_argument("--reprofile", action="store_true",
                    help="re-derive sku_profiles from each phase's own data "
                         "instead of reusing the pinned Tuesday-derived file. "
                         "Scores the SKUs classed smooth under EVERY phase, so "
                         "the population stays common while train_start and "
                         "history_length become phase-matched.")
    args = ap.parse_args()

    src = data_dir(ML_DATA_SNAPSHOT)
    pinned = pd.read_parquet(src / "sales_clean.parquet")
    pinned["ds"] = pd.to_datetime(pinned["ds"])
    profiles = pd.read_csv(src / "sku_profiles.csv")
    profiles["train_start"] = pd.to_datetime(profiles["train_start"])

    skus = set(pinned["unique_id"].unique())
    weeks = pd.DatetimeIndex(sorted(pinned["ds"].unique()))
    smooth = set(profiles.loc[profiles["bucket"] == "smooth", "unique_id"])

    orders = load_orders(args.orders_cache)
    orders["order_date"] = pd.to_datetime(orders["order_date"])

    raw = {name: weekly_for_phase(orders, p, skus, weeks)
           for p, name in enumerate(PHASES)}

    # `prof_by_phase` is what each phase is scored against. Without --reprofile
    # every phase shares the pinned file, which is the experiment 26 setup.
    prof_by_phase = {name: profiles for name in PHASES}
    if args.reprofile:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            for name in PHASES:
                prof_by_phase[name] = profiles_for_phase(raw[name], Path(td))
        smooth_by_phase = {
            n: set(p.loc[p["bucket"] == "smooth", "unique_id"])
            for n, p in prof_by_phase.items()
        }
        # Intersection, not each phase's own set. Letting the cohort move with
        # the phase would vary the population and the alignment together, and a
        # difference produced by both at once cannot be attributed to either --
        # the same mistake this experiment was built to avoid.
        smooth = set.intersection(*smooth_by_phase.values())
        print("\nre-profiled per phase; smooth counts")
        for n in PHASES:
            own = smooth_by_phase[n]
            print(f"  {n:<4} {len(own):>4} smooth   "
                  f"{len(own - smooth):>3} not in the common set")
        print(f"  scored population = {len(smooth)} SKUs smooth under all seven")

    arms = {
        name: prep(raw[name][raw[name]["unique_id"].isin(smooth)],
                   prof_by_phase[name])
        for name in PHASES
    }

    splits = {a: dev_splits(w, n=3) for a, w in arms.items()}

    rows = []
    for i, win in enumerate(WINDOW_NAMES):
        # One validation draw per window, shared by all seven phases, for the
        # reason recorded in ml_26.draw_val: letting it follow the data puts
        # refit noise on v11 that the comparators cannot have.
        shared = None if args.skip_v11 else draw_val(splits["Mon"][i],
                                                     prof_by_phase["Mon"])
        for phase in PHASES:
            split = splits[phase][i]
            profiles = prof_by_phase[phase]
            preds = {"naive_mean12": naive_mean12(split.train, split.test)}
            from src.ml.model import structural_baseline
            preds["baseline"] = structural_baseline(
                split.train, split.test, profiles, split.cutoff)
            if not args.skip_v11:
                preds["v11"] = fit_v11(split, profiles, val=shared)
            for model, p in preds.items():
                for _, r in score(p[["unique_id", "ds", "yhat"]], split,
                                  profiles).iterrows():
                    rows.append({"phase": phase, "window": win, "model": model,
                                 "segment": r["segment"], "n_skus": r["n_skus"],
                                 "actual_units": r["actual_units"],
                                 "pooled_wape": r["pooled_wape"]})
            print(f"  {win:<8} {phase}: {', '.join(preds)}")

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_path = OUT_REPROFILE if args.reprofile else OUT
    out.to_csv(out_path, index=False)

    judged = out[[(w, s) not in EXCLUDED_CELLS
                  for w, s in zip(out["window"], out["segment"])]]

    print(f"\n{'=' * 78}\npooled WAPE by phase (rows) and cell (columns)")
    for model in sorted(judged["model"].unique()):
        m = judged[judged["model"] == model]
        tab = m.pivot_table(index="phase", columns=["segment", "window"],
                            values="pooled_wape").reindex(PHASES)
        print(f"\n{model}")
        print(tab.round(4).to_string())

    print(f"\n{'=' * 78}\nspread across the seven phases, per cell")
    print("  the robustness figure: how much the number depends on a convention\n")
    sp = (judged.groupby(["model", "segment", "window"])["pooled_wape"]
          .agg(lo="min", hi="max", sd="std"))
    sp["range"] = sp["hi"] - sp["lo"]
    print(sp.round(4).to_string())

    print(f"\n{'=' * 78}\nmean range across cells, per model")
    per = sp.groupby(level="model")["range"].mean().sort_values()
    for model, v in per.items():
        print(f"  {model:<14} {v:.4f}")
    if len(per) > 1:
        least = per.index[0]
        print(f"\n  Relative to {least}, the least phase-dependent method:")
        for model, v in per.items():
            if model != least:
                print(f"    {model:<14} {v / per.iloc[0]:.1f}x the range")

    print("\n  This is a spread, not a menu. Choosing the phase with the lowest")
    print("  number would be a maximum over seven draws and would inflate the")
    print("  final test by an amount that cannot be recovered afterwards.")
    print(f"\nwrote {_rel(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
