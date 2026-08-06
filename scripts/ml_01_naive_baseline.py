#!/usr/bin/env python3
"""Naive baselines: the floor any forecast has to clear to be called a forecast.

Design Section 4.29. The project has always compared against V1, the incumbent
spreadsheet, which is the right business comparison and answers "should we
switch". It does not answer the question an outsider asks first, which is
whether the model has forecasting skill at all. V1 could be bad; beating it
proves nothing on its own.

A naive baseline answers that, and it is the one accuracy comparison that
travels between projects. Absolute WAPE does not: it moves with aggregation
level, horizon and how much intermittent demand is in scope, so "16% WAPE"
means nothing without all three. "Beats a trailing mean by a third at a
ten-week horizon" means roughly the same thing anywhere.

Three baselines, cheapest first:

  last        the last observed week, carried flat across the horizon. The
              textbook naive. On weekly retail demand it is mostly noise
              propagation, and it is here as a floor under the floor.
  mean12      the trailing 12-week mean, carried flat. The honest naive for
              this data, and near enough what V1 does with its velocity
              windows, so it doubles as a check on the V1 comparison.
  snaive      the same ten calendar weeks one year earlier. The standard
              seasonal baseline, and unusable on this data: only 18-20% of
              SKU-weeks have an observation 52 weeks back, so its headline
              figure is mostly zero-filled absence rather than error. The
              coverage percentage is printed for exactly that reason, and the
              number should not be quoted without it. Kept because that
              coverage IS the finding: a business with two years of data
              cannot lean on year-over-year repetition, which is independent
              support for imposing seasonality structurally (Section 4.10).

None of these are candidate models. They exist to be beaten, and to make the
size of the win legible.

Scored through src.ml.evaluate.score() rather than by hand, so the baselines
inherit the model's exact eligibility filter, as-of segment labels and pooling.
A baseline scored under looser rules than the model is not a baseline.

Reads the pinned snapshot and runs on dev_splits() only, which excludes the
final test window by construction (dataset.dev_splits slices split 0 off).

    .venv/bin/python scripts/ml_01_naive_baseline.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.ml.dataset import dev_splits, load_weekly  # noqa: E402
from src.ml.evaluate import score  # noqa: E402

OUT = ROOT / "outputs" / "reports" / "ml_naive_baseline.csv"


def _grid(test: pd.DataFrame) -> pd.DataFrame:
    """One row per SKU per test week, which is the shape score() expects."""
    return test[["unique_id", "ds"]].drop_duplicates()


def naive_last(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Last observed week, held flat."""
    lvl = (train.sort_values("ds").groupby("unique_id")["y"].last().rename("yhat"))
    return _grid(test).merge(lvl, on="unique_id", how="left").fillna({"yhat": 0.0})


def naive_mean12(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Trailing 12-week mean, held flat.

    12 weeks to match the denominator the model's own target is a ratio to, so
    the baseline and the model are anchored on the same notion of "level" and
    the comparison isolates the modelling rather than the window length.
    """
    lvl = (
        train.sort_values("ds").groupby("unique_id")["y"]
        .apply(lambda s: s.tail(12).mean()).rename("yhat").reset_index()
    )
    return _grid(test).merge(lvl, on="unique_id", how="left").fillna({"yhat": 0.0})


def naive_seasonal(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """The same calendar weeks one year earlier, week for week.

    Not held flat: each target week takes its own value from 52 weeks back, so
    this carries last year's shape as well as its level.

    Zero-filled where the SKU has no observation a year before a given week.
    That is the honest scoring of a method that cannot answer, and it is why
    the coverage count is printed: on a catalogue whose data starts in mid-2024
    this baseline is unavailable for a large share of SKU-weeks, which is
    itself a finding about how much history the project has.
    """
    grid = _grid(test).copy()
    hist = train[["unique_id", "ds", "y"]].copy()
    hist["ds"] = hist["ds"] + pd.Timedelta(weeks=52)
    out = grid.merge(hist, on=["unique_id", "ds"], how="left")
    out["covered"] = out["y"].notna()
    return out.rename(columns={"y": "yhat"}).fillna({"yhat": 0.0})


BASELINES = {
    "naive_last": naive_last,
    "naive_mean12": naive_mean12,
    "naive_seasonal": naive_seasonal,
}


def main() -> None:
    weekly, profiles = load_weekly()
    # Smooth only, matching what the model forecasts. Scoring a baseline on a
    # population the model never sees would compare two different problems.
    smooth = profiles.loc[profiles["bucket"] == "smooth", "unique_id"]
    weekly = weekly[weekly["unique_id"].isin(smooth)]

    rows = []
    for split in dev_splits(weekly, n=3):
        print(f"\n{'=' * 64}\n{split}")
        for name, fn in BASELINES.items():
            preds = fn(split.train, split.test)
            if "covered" in preds.columns:
                cov = preds["covered"].mean()
                preds = preds.drop(columns="covered")
                print(f"  {name}: {cov * 100:.0f}% of SKU-weeks had a value a year earlier")
            table = score(preds[["unique_id", "ds", "yhat"]], split, profiles)
            for _, r in table.iterrows():
                rows.append({
                    "baseline": name,
                    "cutoff": str(split.cutoff.date()),
                    "segment": r["segment"],
                    "n_skus": r["n_skus"],
                    "actual_units": r["actual_units"],
                    "pooled_wape": r["pooled_wape"],
                    "bias_pct": r["bias_pct"],
                })

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"\n{'=' * 64}\nTOTAL, pooled across the three dev windows")
    tot = out[out["segment"] == "TOTAL"]
    # Unit-weighted, because pooled WAPE is unit-weighted within a window and
    # averaging the three cells equally would let the smallest window count as
    # much as the largest. Same combination rule as the served comparison.
    for name, g in tot.groupby("baseline"):
        u = g["actual_units"].sum()
        w = (g["pooled_wape"] * g["actual_units"]).sum() / u
        print(f"  {name:<16} {w * 100:>6.2f}%")

    acc = ROOT / "outputs" / "reports" / "ml_accuracy.csv"
    if acc.exists():
        a = pd.read_csv(acc)
        a = a[a["segment"] == "TOTAL"]
        print()
        for v, g in a.groupby("model_version"):
            u = g["actual_units"].sum()
            w = (g["pooled_wape"] * g["actual_units"]).sum() / u
            print(f"  {v:<16} {w * 100:>6.2f}%   (from ml_accuracy.csv)")

    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
