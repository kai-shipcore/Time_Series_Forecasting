#!/usr/bin/env python3
"""ML experiment 26: does the W-MON bin boundary move the recorded numbers?

The question
------------
`src/clean.py` was corrected on 2026-08-05 from `closed="right"` to
`closed="left"`. Before the fix the bucket labelled Monday L spanned Tuesday
(L-6) through Monday L; after it, Monday (L-7) through Sunday (L-1). Same
labels, same demand, a window shifted one day.

The pinned snapshot (`data/snapshots/2026-07-20/`) is a frozen copy that
`clean()` does not touch, so no recorded figure moved when the fix landed. That
is convenient and it is also the problem: every number in the Version Log was
measured on the old boundary, while the production pipeline now trains on the
new one. Either that gap is too small to matter, or the snapshot has to be
rebuilt and every quoted version re-run before the final test.

Nobody should guess which. Monday is a seventh of the week's volume, and a
seventh of every bucket moving to its neighbour is easily enough to shift pooled
WAPE in the second decimal, which is coarser than the third decimal this project
compares at.

Pre-registered criteria (stated before the first run, per CLAUDE.md)
-------------------------------------------------------------------
Measured on v11, the served version, across the three development windows.

  PASS, keep the pinned snapshot:
      |delta| < 0.02 on every window and segment, AND
      |three-window mean delta| < 0.01, AND
      no comparison flips sign between the two arms.

  FAIL, re-snapshot before the final test and re-run every quoted version:
      any window or segment exceeding 0.02, OR
      a three-window mean exceeding 0.01, OR
      any sign flip.

Those two thresholds are not invented here. They are the numbers Section 1.5
already uses: a single-window difference below 0.02 is treated as inconclusive,
and a three-window mean below 0.01 is below the adoption bar. The argument for
keeping the snapshot is precisely that the boundary moves the results by less
than the amount the project already treats as indistinguishable from chance. If
that is not true, the argument fails and the numbers get rebuilt.

The sign clause is separate on purpose. The magnitude thresholds protect the
recorded figures; the sign protects the claim those figures are used to make. A
boundary that moved every number by 0.03 without changing who beats whom would
still invalidate the Version Log while leaving "v11 beats the baseline" intact,
and those are different failures with different costs.

Note what the sign clause does NOT test. It is not "v11 wins", which is settled
elsewhere and is not this experiment's business. It compares the sign of
(v11 - comparator) on the old arm against the same sign on the new arm. A cell
where v11 loses on both arms is not a flip; a cell where it wins on one and
loses on the other is. Only the second says the boundary changed a conclusion.

Section 1.5's exclusion is honoured: smooth/short in the Oct-Dec 2025 window had
only 14 eligible SKUs and is excluded from short-segment decisions there. Its
numbers are printed and marked, and take no part in the verdict. A criterion that
let an inadmissible cell decide this would be a criterion the project has already
rejected everywhere else.

What is held fixed, and why it matters
--------------------------------------
Only the bin boundary varies. Everything else is pinned:

  raw orders     One ingest, grouped twice. Re-pulling per variant would let
                 late-registering orders differ between the two arms.
  SKU set        The pinned snapshot's SKUs.
  week labels    The pinned snapshot's labels, so `dev_splits` (anchored to
                 ML_FINAL_TEST_CUTOFF) builds identical windows on both arms.
  sku_profiles   The pinned file, reused unchanged. This is the important one.
                 Re-profiling on shifted bins can flip SKUs across the
                 smooth/intermittent boundary and move `train_start`, which
                 changes the population being scored rather than the data under
                 it. A comparison where both the data and the cohort moved is
                 one you cannot attribute. What re-profiling would do to the
                 population is a separate question, and it only arises if the
                 verdict is FAIL, since that is the only branch where the
                 snapshot gets rebuilt at all.

The old arm rebuilt today will not reproduce the pinned snapshot exactly,
because orders have registered against those weeks since 2026-07-20. That drift
is measured and printed rather than assumed away: it is the floor under any
difference this experiment can resolve, and if it is large relative to the
old-vs-new delta then the experiment answers nothing and says so.

Usage
-----
    .venv/bin/pip install -r requirements.txt        # lightgbm is not installed
    .venv/bin/python scripts/ml_26_week_boundary_ab.py

    # reuse the raw pull instead of hitting the database again
    .venv/bin/python scripts/ml_26_week_boundary_ab.py --orders-cache data/processed/_ab_orders.parquet

    # cheap arms only, no lightgbm needed
    .venv/bin/python scripts/ml_26_week_boundary_ab.py --skip-v11

Writes outputs/reports/ml_week_boundary_ab.csv. Touches nothing else: no file
under data/processed/ or data/snapshots/ is written, so a run of this script
cannot change what production serves or what any recorded number was measured on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# scripts/ too, so naive_mean12 is imported from the file that defines it
# rather than copied. A metric with two definitions eventually has two values.
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

from config import ML_DATA_SNAPSHOT, TRIM_TRAILING_WEEKS  # noqa: E402
from ml_01_naive_baseline import naive_mean12  # noqa: E402
from src.ml.dataset import data_dir, dev_splits, stratified_val_skus  # noqa: E402
from src.ml.evaluate import per_sku_totals, score  # noqa: E402

OUT = ROOT / "outputs" / "reports" / "ml_week_boundary_ab.csv"
# The diagnostic writes somewhere else. Sending it to OUT would overwrite the
# v11 run it exists to be compared against, which is the one file you need to
# still have when reading it.
OUT_DIAG = OUT.with_name("ml_week_boundary_ab_nodeseas.csv")

# The two arms. "old" is what every recorded number was measured on; "new" is
# what src/clean.py produces now and what production trains on.
ARMS = {"old": "right", "new": "left"}

# Section 1.5 thresholds, restated as constants so the verdict printed at the
# bottom cannot drift from the criteria in the docstring.
MAX_WINDOW_DELTA = 0.02
MAX_MEAN_DELTA = 0.01

# Window names, in the order dev_splits returns them, matching the design doc
# and scripts/ml_22_v11_hybrid.py so a result here can be read against a result
# there without anyone having to map cutoffs to seasons by hand.
WINDOW_NAMES = ["Mar-May", "Dec-Feb", "Oct-Dec"]

# Section 1.5: only 14 short SKUs were eligible at the Oct-Dec cutoff, too few
# to mean anything, so that cell decides nothing. Printed and marked, excluded
# from the verdict. (window name, segment).
EXCLUDED_CELLS = {("Oct-Dec", "smooth/short")}


def weekly_from_orders(
    orders: pd.DataFrame, closed: str, skus: set, weeks: pd.DatetimeIndex
) -> pd.DataFrame:
    """Weekly totals under one bin boundary, on a fixed SKU x week grid.

    Mirrors src/clean.py's grouping, with `closed` as the variable. The grid is
    imposed from the pinned snapshot rather than derived from the data, so both
    arms come out the same shape and any difference between them is values.
    """
    w = (
        orders.groupby(
            [
                "link_master_sku",
                pd.Grouper(key="order_date", freq="W-MON", closed=closed, label="right"),
            ],
            as_index=False,
        )["link_qty"]
        .sum()
        .rename(columns={"link_master_sku": "unique_id", "order_date": "ds", "link_qty": "y"})
    )
    w = w[w["unique_id"].isin(skus) & w["ds"].isin(weeks)]
    grid = pd.MultiIndex.from_product(
        [sorted(skus), weeks], names=["unique_id", "ds"]
    )
    return (
        w.set_index(["unique_id", "ds"])
        .reindex(grid, fill_value=0)
        .reset_index()
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )


def prep(weekly: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """The post-load steps `dataset.load_weekly` applies, replicated exactly.

    Not imported from it because load_weekly reads a snapshot directory off
    disk and these arms are held in memory. Kept adjacent to it in behaviour:
    trailing trim, then trim each ramp-up SKU to its train_start.
    """
    if TRIM_TRAILING_WEEKS:
        keep = sorted(weekly["ds"].unique())[:-TRIM_TRAILING_WEEKS]
        weekly = weekly[weekly["ds"].isin(keep)]

    starts = profiles.set_index("unique_id")["train_start"]
    weekly = weekly.copy()
    weekly["_ts"] = weekly["unique_id"].map(starts)
    weekly = weekly[weekly["_ts"].isna() | (weekly["ds"] >= weekly["_ts"])]
    return weekly.drop(columns="_ts").sort_values(["unique_id", "ds"]).reset_index(drop=True)


def draw_val(split, profiles: pd.DataFrame, seed: int = 42) -> tuple[set, set]:
    """The two early-stopping validation draws v11 needs, for one window.

    Split out so the same draw can be handed to both arms. It must be, and the
    first version of this experiment did not do it.

    `stratified_val_skus` ranks SKUs by mean volume and cuts terciles, then
    samples within each cell. Both steps are sensitive to the values: a SKU
    whose mean shifts slightly can cross a tercile edge, which reorders the
    cell, which changes what `rng.choice` returns for every SKU in it. Deriving
    the draw separately per arm therefore produced two nearly disjoint
    validation sets, 72% to 95% different across the three windows, and with
    them two different early-stopping points and two different models.

    That noise sits only on v11. The structural baseline and the trailing mean
    have no validation set and no early stopping, so an experiment that let the
    draw follow the arm compared a stochastically refit model against two
    deterministic ones and read the difference as boundary sensitivity.
    """
    longs = long_sku_set_cached(profiles, split.cutoff, split.train)
    val_all = stratified_val_skus(split.train, profiles, seed=seed)
    val_long = stratified_val_skus(
        split.train[split.train["unique_id"].isin(longs)], profiles, seed=seed
    )
    return val_all, val_long


def long_sku_set_cached(profiles: pd.DataFrame, cutoff, train: pd.DataFrame) -> set:
    """`long_sku_set` intersected with the SKUs present in training.

    Depends only on profiles, the cutoff and the training SKU set, all three of
    which are identical across arms, so this returns the same set for both.
    """
    from src.ml.model import long_sku_set

    return long_sku_set(profiles, cutoff) & set(train["unique_id"])


def fit_v11(
    split,
    profiles: pd.DataFrame,
    deseas: bool = True,
    val: tuple[set, set] | None = None,
) -> pd.DataFrame:
    """v11 exactly as scripts/ml_22_v11_hybrid.py fits it.

    Shared short model over all SKUs, dedicated long model over the long
    segment, long predictions overwriting the shared ones. Any drift from ml_22
    here would be measured as a boundary effect, so this follows it line for
    line rather than paraphrasing it.

    `deseas=False` is a diagnostic, not a version. The first run of this
    experiment found v11 roughly five times more boundary-sensitive than either
    comparator (mean |delta| 0.0159 against 0.0030 and 0.0032), and the leading
    explanation is that the seasonal machinery is aligned to the old boundary:
    the monthly multipliers and the holiday window were fitted and CV-optimised
    against Tuesday-to-Monday buckets, and v11 applies them to the training
    targets and features rather than only as a final multiplier the way the
    structural baseline does. Refitting with deseasonalisation off on both arms
    isolates that. If the asymmetry collapses, the seasonal alignment is
    implicated; if it survives, it is not, and the cause is elsewhere.

    What comes out of a `deseas=False` run is not v11 and must never be recorded
    as a v11 number or compared against one.
    """
    from src.ml.model import FEATURES_V1, FEATURES_V11_LONG, RatioLGBM

    longs = long_sku_set_cached(profiles, split.cutoff, split.train)
    # `val` supplied means both arms share one draw, which is the controlled
    # comparison. Deriving it here from this arm's own data is the uncontrolled
    # one, kept reachable behind --per-arm-val so the size of that mistake can
    # be measured rather than argued about.
    val_all, val_long = val if val is not None else draw_val(split, profiles)

    m_short = RatioLGBM(
        split.horizon, FEATURES_V1, deseas_features=deseas, deseas_all=deseas
    ).fit(split.train, profiles, split.cutoff, val_all)
    p_short = m_short.predict(split.train, profiles, split.cutoff)

    m_long = RatioLGBM(
        split.horizon, FEATURES_V11_LONG, deseas_features=deseas,
        deseas_all=deseas, uids=longs,
    ).fit(split.train, profiles, split.cutoff, val_long)
    p_long = m_long.predict(split.train, profiles, split.cutoff)

    return pd.concat(
        [p_short[~p_short["unique_id"].isin(longs)], p_long], ignore_index=True
    )


def _rel(path: Path) -> str:
    """Path for display, shortened when it sits inside the repo.

    `relative_to` raises when it does not, and a path handed in on the command
    line often does not: a relative one resolves against the working directory,
    and an absolute one can point anywhere. Crashing on the way to printing a
    filename is a poor trade, and worse here than usual, because the line it
    crashed on ran immediately after a slow database pull. Same helper as
    scripts/ml_purge_history_run.py, which hit this first. Display only.
    """
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_orders(cache: Path | None) -> pd.DataFrame:
    """One raw pull, optionally cached so re-runs do not re-hit the database."""
    if cache and cache.exists():
        print(f"orders: reusing {_rel(cache)}")
        return pd.read_parquet(cache)

    from src.ingest import ingest

    print("orders: pulling from the database ...")
    orders = ingest()
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    if cache:
        # Written before anything else can fail. The pull is the slow part of
        # this script and losing it to a later error means paying for it twice.
        cache.parent.mkdir(parents=True, exist_ok=True)
        orders.to_parquet(cache, index=False)
        print(f"orders: cached to {_rel(cache)}")
    return orders


def report_drift(arms: dict[str, pd.DataFrame], pinned: pd.DataFrame) -> None:
    """How much moved, and how much of it is the boundary rather than time.

    Two quantities, and the comparison is only meaningful if the first is
    clearly larger than the second:

      boundary   old arm vs new arm, both pulled today. What is being tested.
      revision   old arm vs the pinned file. Late-registering orders against
                 weeks that closed before 2026-07-20. Noise floor.
    """
    old, new = arms["old"], arms["new"]
    merged = old.merge(new, on=["unique_id", "ds"], suffixes=("_old", "_new"))
    moved = (merged["y_old"] != merged["y_new"]).sum()
    units = merged["y_old"].sum()
    shifted = (merged["y_old"] - merged["y_new"]).abs().sum()

    print("\nboundary effect on the data itself")
    print(f"  {moved:,} of {len(merged):,} SKU-weeks differ "
          f"({moved / len(merged) * 100:.1f}%)")
    print(f"  {shifted:,.0f} units moved between adjacent weeks, "
          f"{shifted / max(units, 1) * 100:.1f}% of {units:,.0f} total")

    p = pinned.merge(old, on=["unique_id", "ds"], suffixes=("_pin", "_old"))
    rev = (p["y_pin"] - p["y_old"]).abs().sum()
    print("\nnoise floor: late-order revision since the snapshot was taken")
    print(f"  {(p['y_pin'] != p['y_old']).sum():,} SKU-weeks differ between the "
          f"pinned file and the old arm rebuilt today")
    print(f"  {rev:,.0f} units, {rev / max(units, 1) * 100:.1f}% of total")
    if rev >= shifted:
        print("  WARNING: revision is as large as the boundary effect. Any "
              "difference below reflects both, and this experiment cannot "
              "separate them.")


def bootstrap_arm_delta(
    old: pd.DataFrame, new: pd.DataFrame, segment: str | None = None,
    n_boot: int = 1000, seed: int = 0,
) -> dict:
    """Paired SKU bootstrap of the pooled-WAPE difference BETWEEN ARMS.

    `evaluate.bootstrap_delta` cannot do this. It compares two models against
    one split, so it carries a single `y` and a single denominator. The two
    arms here have different actuals, because different actuals is the whole
    point, so each side needs its own numerator and its own denominator:

        delta_b = sum(ae_new)/sum(y_new) - sum(ae_old)/sum(y_old)

    over a resampled SKU set. Paired on SKU, which is legitimate because the
    SKU population is held identical across arms by construction, so the same
    resample indexes the same products on both sides and the SKU draw cancels.

    This is the number the verdict needed and did not have. Three of the four
    magnitude criteria failed by less than 0.006, and the whole smooth/long
    result rests on a single cell at 0.0536. Section 1.5 puts the sampling
    noise of a single-window paired difference at 0.011 to 0.014, which is
    larger than most of what is being called a failure here, and it says in
    terms that borderline calls are settled by the bootstrap. Reporting the
    point estimates without it was reading differences the design doc had
    already said are not readable on their own.
    """
    import numpy as np

    df = old.merge(new, on="unique_id", suffixes=("_old", "_new"))
    if segment is not None:
        df = df[df["history_length_old"] == segment]
    if df.empty:
        return {"delta": float("nan"), "se": float("nan"), "n": 0}

    ae_o, y_o = df["ae_old"].to_numpy(), df["y_total_old"].to_numpy()
    ae_n, y_n = df["ae_new"].to_numpy(), df["y_total_new"].to_numpy()
    point = ae_n.sum() / y_n.sum() - ae_o.sum() / y_o.sum()

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(df), size=(n_boot, len(df)))
    deltas = ae_n[idx].sum(1) / y_n[idx].sum(1) - ae_o[idx].sum(1) / y_o[idx].sum(1)
    return {
        "delta": float(point),
        "se": float(deltas.std()),
        "ci_lo": float(np.percentile(deltas, 2.5)),
        "ci_hi": float(np.percentile(deltas, 97.5)),
        "n": int(len(df)),
    }


def report_series_shape(arms: dict[str, pd.DataFrame], profiles: pd.DataFrame) -> None:
    """Is one arm's series simply easier to forecast than the other's?

    Costs no model fits and asks the question the WAPE comparison cannot. v11
    reads lagged values of this series; the trailing mean does not care about
    order at all. So if one bucketing produces a more autocorrelated, less
    jumpy series, a lag-based model scores better on it for a reason that has
    nothing to do with being a better forecaster, and the comparators would
    barely notice. That is exactly the asymmetry the sensitivity table shows.

    Two descriptive statistics per arm, over smooth SKUs:

      lag-1 autocorrelation   how much this week predicts next week.
      relative week-to-week   mean |change| divided by mean level. Falls as the
      change                  series gets smoother.

    Reported for the whole series and for December and January alone, because
    the damage concentrates in the Dec-Feb window and a whole-series average
    would bury a seasonal effect.

    Descriptive only. Higher autocorrelation on the old arm would support the
    idea that the old numbers were flattered by the bucketing; it would not on
    its own prove it.
    """
    smooth = set(profiles.loc[profiles["bucket"] == "smooth", "unique_id"])

    def stats(df: pd.DataFrame) -> tuple[float, float]:
        ac, rel = [], []
        for _, g in df.groupby("unique_id"):
            y = g.sort_values("ds")["y"].astype(float)
            if len(y) < 12 or y.std() == 0:
                continue
            a = y.autocorr(lag=1)
            if pd.notna(a):
                ac.append(a)
            m = y.mean()
            if m > 0:
                rel.append(y.diff().abs().mean() / m)
        return (
            float(pd.Series(ac).median()) if ac else float("nan"),
            float(pd.Series(rel).median()) if rel else float("nan"),
        )

    print("\nseries shape by arm (smooth SKUs, median across SKUs)")
    print(f"  {'':<10} {'lag-1 autocorr':>16} {'rel wk-to-wk change':>22}")
    for label, sel in (("all weeks", None), ("Dec+Jan", (12, 1))):
        for arm, w in arms.items():
            d = w[w["unique_id"].isin(smooth)]
            if sel is not None:
                d = d[pd.to_datetime(d["ds"]).dt.month.isin(sel)]
            a, r = stats(d)
            print(f"  {label if arm == 'old' else '':<10} {arm:<4} {a:>11.4f} {r:>21.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--orders-cache", type=Path, default=None,
                    help="parquet path to cache the raw order pull in")
    ap.add_argument("--skip-v11", action="store_true",
                    help="baseline and naive arms only; no lightgbm required")
    ap.add_argument("--per-arm-val", action="store_true",
                    help="derive the early-stopping validation draw separately "
                         "per arm, as the first version of this experiment did. "
                         "Uncontrolled: the draws come out 72-95%% different and "
                         "the resulting noise lands only on v11. Kept to measure "
                         "the size of that mistake, not to run experiments with.")
    ap.add_argument("--no-deseas", action="store_true",
                    help="DIAGNOSTIC: refit with deseasonalisation off on both "
                         "arms, to test whether the seasonal machinery is what "
                         "makes the model boundary-sensitive. The result is not "
                         "v11 and must not be recorded as a v11 number.")
    args = ap.parse_args()

    # A diagnostic run carries a different label all the way through, into the
    # CSV as well as the printout, so its numbers cannot later be mistaken for
    # v11's. The verdict block refuses to rule on it for the same reason.
    v11_label = "v11-nodeseas" if args.no_deseas else "v11"
    if args.no_deseas:
        print("DIAGNOSTIC RUN: deseasonalisation off on both arms. The model "
              "fitted here is not v11 and its numbers are not v11 numbers.")

    src = data_dir(ML_DATA_SNAPSHOT)
    pinned = pd.read_parquet(src / "sales_clean.parquet")
    pinned["ds"] = pd.to_datetime(pinned["ds"])
    profiles = pd.read_csv(src / "sku_profiles.csv")
    profiles["train_start"] = pd.to_datetime(profiles["train_start"])

    skus = set(pinned["unique_id"].unique())
    weeks = pd.DatetimeIndex(sorted(pinned["ds"].unique()))
    print(f"pinned snapshot {ML_DATA_SNAPSHOT}: {len(skus):,} SKUs, "
          f"{len(weeks)} weeks, {weeks[0].date()} -> {weeks[-1].date()}")

    orders = load_orders(args.orders_cache)
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    print(f"orders: {len(orders):,} lines, "
          f"{orders['order_date'].min().date()} -> {orders['order_date'].max().date()}")

    arms = {
        name: weekly_from_orders(orders, closed, skus, weeks)
        for name, closed in ARMS.items()
    }
    report_drift(arms, pinned)

    # Smooth only, matching what the model forecasts and what ml_22 scores.
    smooth = set(profiles.loc[profiles["bucket"] == "smooth", "unique_id"])
    arms = {k: prep(v[v["unique_id"].isin(smooth)], profiles) for k, v in arms.items()}
    # AFTER prep, not before. Called on the raw grid this reported a median
    # lag-1 autocorrelation of 0.68 when the real figure is about 0.25: the
    # zero-filled grid gives every SKU a long run of pre-launch zeros, and a run
    # of zeros is perfectly autocorrelated. prep() trims each SKU to its
    # train_start, which is what makes the statistic describe demand rather than
    # describe the padding.
    report_series_shape(arms, profiles)

    # Window outer, arm inner, so one validation draw can be made per window and
    # handed to both arms. The reverse nesting is what let the draw follow the
    # arm in the first version.
    splits = {arm: dev_splits(weekly, n=3) for arm, weekly in arms.items()}

    rows = []
    totals: dict = {}
    for i in range(len(WINDOW_NAMES)):
        # Drawn from the old arm. The choice of arm is arbitrary and the point
        # is only that both get the same set; old is used because it is the
        # incumbent and so the new arm is the one being asked to prove itself.
        shared_val = None
        if not (args.skip_v11 or args.per_arm_val):
            shared_val = draw_val(splits["old"][i], profiles)

        for arm in ARMS:
            split = splits[arm][i]
            preds = {
                "naive_mean12": naive_mean12(split.train, split.test),
            }
            from src.ml.model import structural_baseline
            preds["baseline"] = structural_baseline(
                split.train, split.test, profiles, split.cutoff
            )
            if not args.skip_v11:
                preds[v11_label] = fit_v11(split, profiles,
                                           deseas=not args.no_deseas,
                                           val=shared_val)

            for model, p in preds.items():
                # Kept so the two arms can be bootstrapped against each other
                # afterwards. score() aggregates these away, and the aggregate
                # cannot be resampled.
                totals[(arm, WINDOW_NAMES[i], model)] = per_sku_totals(
                    p[["unique_id", "ds", "yhat"]], split, profiles
                )
                table = score(p[["unique_id", "ds", "yhat"]], split, profiles)
                for _, r in table.iterrows():
                    rows.append({
                        "arm": arm,
                        "window": WINDOW_NAMES[i],
                        "cutoff": str(split.cutoff.date()),
                        "model": model,
                        "segment": r["segment"],
                        "n_skus": r["n_skus"],
                        "actual_units": r["actual_units"],
                        "pooled_wape": r["pooled_wape"],
                        "bias_pct": r["bias_pct"],
                    })
            print(f"  {arm:<4} {WINDOW_NAMES[i]:<8} cutoff {split.cutoff.date()}: "
                  f"scored {', '.join(preds)}")

    out = pd.DataFrame(rows)
    out_path = OUT_DIAG if args.no_deseas else OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    # ---- the comparison ----------------------------------------------------
    wide = out.pivot_table(
        index=["model", "segment", "window"], columns="arm", values="pooled_wape"
    )
    wide["delta"] = wide["new"] - wide["old"]
    wide["excluded"] = [
        "yes" if (w, s) in EXCLUDED_CELLS else "" for _, s, w in wide.index
    ]
    print(f"\n{'=' * 72}\npooled WAPE by arm  (delta = new - old)")
    print(wide.round(4).to_string())
    print("\nprofiles held fixed at the pinned sku_profiles.csv on both arms, so "
          "the SKU population and every train_start are identical.")
    if (wide["excluded"] == "yes").any():
        print("excluded = Section 1.5: too few eligible SKUs to decide anything. "
              "Shown, not counted.")

    # ---- boundary sensitivity by model -------------------------------------
    # The comparison that carries the information, and it is not the verdict.
    # If every method moves by about the same amount, the boundary is a
    # property of the data and says nothing about any model. If one method
    # moves several times more than the others, that method depends on the
    # boundary in a way the others do not, which is a robustness fact about the
    # model rather than a bookkeeping fact about the snapshot.
    #
    # Printed automatically because the first run's asymmetry, v11 at roughly
    # five times either comparator, was only visible after adding the deltas up
    # by hand afterwards. The most informative number in the run should not
    # depend on someone thinking to compute it.
    print(f"\n{'=' * 72}\nboundary sensitivity by model (judged cells only)")
    sens = wide[wide["excluded"] != "yes"].copy()
    sens["abs"] = sens["delta"].abs()
    summary = sens.groupby(level="model")["abs"].agg(["mean", "max", "count"])
    print(summary.round(4).to_string())
    if len(summary) > 1:
        least = summary["mean"].idxmin()
        for model, r in summary.iterrows():
            if model != least and r["mean"] > 3 * summary.loc[least, "mean"]:
                print(f"\n  {model} moves {r['mean'] / summary.loc[least, 'mean']:.1f}x "
                      f"more than {least} under the same change of boundary.")
                print("  A shared shift would move every method alike. This one "
                      "does not, so it is not only the data that moved.")

    # ---- did the question change, or only the answer? ----------------------
    # The decisive number for whether the boundary is a hyperparameter or a
    # redefinition of the target, and it was missing from every earlier round.
    #
    # score() sums each SKU over the whole 10-week window before dividing, so
    # what is being predicted is a window total, not a weekly series. Shifting
    # the boundary by one day slides that window one day earlier: it drops one
    # day at the end and picks up one day at the start, and the other 69 are
    # the same demand either way. If the totals barely move, the two arms are
    # answering nearly the same question and a difference between them really
    # is about the model. If they move a lot, the arms are scored on different
    # targets and no comparison between them means much.
    print(f"\n{'=' * 72}\ndid the target change?  (10-week window totals per SKU)")
    print(f"  {'window':<9} {'units old':>12} {'units new':>12} "
          f"{'net':>8} {'sum |per-SKU change|':>22}")
    for win in WINDOW_NAMES:
        o = next((totals[k] for k in totals if k[0] == "old" and k[1] == win), None)
        n = next((totals[k] for k in totals if k[0] == "new" and k[1] == win), None)
        if o is None or n is None:
            continue
        j = o[["unique_id", "y_total"]].merge(
            n[["unique_id", "y_total"]], on="unique_id", suffixes=("_o", "_n"))
        so, sn = j["y_total_o"].sum(), j["y_total_n"].sum()
        gross = (j["y_total_o"] - j["y_total_n"]).abs().sum()
        print(f"  {win:<9} {so:>12,.0f} {sn:>12,.0f} "
              f"{(sn - so) / so * 100:>7.2f}% {gross / so * 100:>21.2f}%")
    print("\n  Compare the last column against the WAPE differences below. A target")
    print("  that moved as much as the error did cannot support a claim about either.")

    # ---- is any of this bigger than the noise? -----------------------------
    if not args.skip_v11:
        print(f"\n{'=' * 72}\nis the boundary difference bigger than SKU sampling noise?")
        print("  paired SKU bootstrap, 1000 resamples, same SKUs on both arms")
        print(f"  {'window':<9} {'segment':<13} {'delta':>8} {'se':>7} "
              f"{'95% CI':>18} {'n':>5}  beyond noise")
        for i, win in enumerate(WINDOW_NAMES):
            for seg in ("short", "long", None):
                name = f"smooth/{seg}" if seg else "TOTAL"
                if (win, name) in EXCLUDED_CELLS:
                    continue
                o = totals.get(("old", win, v11_label))
                n = totals.get(("new", win, v11_label))
                if o is None or n is None:
                    continue
                b = bootstrap_arm_delta(o, n, seg)
                if not b["n"]:
                    continue
                # Section 1.5's own rule for one window: |delta| > 2 * se.
                verdict = "yes" if abs(b["delta"]) > 2 * b["se"] else "NO - noise"
                print(f"  {win:<9} {name:<13} {b['delta']:>+8.4f} {b['se']:>7.4f} "
                      f"  [{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}] {b['n']:>5}  {verdict}")
        print("\n  A cell marked noise is one where this experiment cannot tell the")
        print("  two boundaries apart, whatever its point estimate looks like.")

    # ---- verdict -----------------------------------------------------------
    print(f"\n{'=' * 72}\nVERDICT against the pre-registered criteria")
    if args.no_deseas:
        print("  Diagnostic run: no verdict. The criteria are stated on v11, and")
        print("  the model fitted here is not v11. Read the sensitivity table")
        print("  above against the v11 run's, not against its thresholds.")
        print(f"\nwrote {_rel(out_path)}")
        return 0
    if "v11" not in wide.index.get_level_values("model"):
        print("  v11 was not run (--skip-v11), so no verdict. The criteria are")
        print("  stated on v11 because that is the version being served.")
        print(f"\nwrote {_rel(out_path)}")
        return 0

    judged = wide.loc["v11"]
    judged = judged[judged["excluded"] != "yes"]

    worst = judged["delta"].abs().max()
    means = judged.groupby("segment")["delta"].mean()
    print(f"  largest single window/segment |delta|: {worst:.4f} "
          f"(threshold {MAX_WINDOW_DELTA})")
    for seg, m in means.items():
        print(f"  mean delta across judged windows, {seg}: {m:+.4f} "
              f"(threshold {MAX_MEAN_DELTA})")

    # Sign stability. NOT "does v11 win" -- that is settled elsewhere and is not
    # what this experiment is for. The question is whether the boundary changes
    # WHO WINS: the sign of (v11 - comparator) on the old arm against the sign
    # of the same difference on the new arm. Same sign on both arms is no flip,
    # even in a cell where v11 loses on both.
    signs_ok = True
    for (seg, win), g in out.groupby(["segment", "window"]):
        if (win, seg) in EXCLUDED_CELLS:
            continue
        by_arm = {a: gg.set_index("model")["pooled_wape"] for a, gg in g.groupby("arm")}
        if not all("v11" in s.index for s in by_arm.values()):
            continue
        for comparator in ("baseline", "naive_mean12"):
            if not all(comparator in s.index for s in by_arm.values()):
                continue
            diffs = {a: s["v11"] - s[comparator] for a, s in by_arm.items()}
            if (diffs["old"] < 0) != (diffs["new"] < 0):
                print(f"  SIGN FLIP: v11 vs {comparator} on {seg}/{win} changes "
                      f"direction between arms (old {diffs['old']:+.4f}, "
                      f"new {diffs['new']:+.4f})")
                signs_ok = False

    passed = (
        worst < MAX_WINDOW_DELTA
        and means.abs().max() < MAX_MEAN_DELTA
        and signs_ok
    )
    print(f"\n  {'PASS - keep the pinned snapshot' if passed else 'FAIL - re-snapshot before the final test'}")
    if not passed:
        print("  Rebuild the snapshot, then re-run every version still quoted "
              "in the Version Log before the final test window is opened.")

    print(f"\nwrote {_rel(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
