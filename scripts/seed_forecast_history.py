"""Fabricate stored forecast runs so the demand-versus-forecast chart can be
looked at before real runs have accumulated.

The chart reads `ml_forecast_history.parquet`, which gains one entry per weekly
run. A fresh store is empty, so the chart shows demand and a forward horizon and
no predicted line, and there is no way to see the thing working until several
Mondays have passed. This writes plausible runs so the page can be reviewed now.

The rows are labelled with a model version ending in `-SAMPLE`, for two reasons.
They are visibly not real anywhere they surface, and the serving endpoint prefers
the version the current forward forecast came from, so these rows are ignored the
moment genuine runs land. The sample retires itself.

    python3 scripts/seed_forecast_history.py             # 6 weekly runs
    python3 scripts/seed_forecast_history.py --runs 10
    python3 scripts/seed_forecast_history.py --clear     # remove them again

Nothing here touches real rows. `--clear` deletes only versions ending in
`-SAMPLE`, and reports what it removed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ml.serving import history as hist  # noqa: E402
from src.ml.serving.models import CURRENT_BEST  # noqa: E402
from src.planning import data as D  # noqa: E402

SAMPLE_SUFFIX = "-SAMPLE"
HORIZON = 13


def _sample_version() -> str:
    return f"{CURRENT_BEST}{SAMPLE_SUFFIX}"


def clear() -> int:
    """Remove sample rows, leaving anything real untouched."""
    store = hist.load()
    if store.empty:
        print("History store is empty. Nothing to clear.")
        return 0

    is_sample = store["model_version"].astype(str).str.endswith(SAMPLE_SUFFIX)
    removed = int(is_sample.sum())
    if removed == 0:
        print(f"No sample rows found. Store holds {len(store):,} real rows, untouched.")
        return 0

    kept = store[~is_sample]
    if kept.empty:
        hist.HISTORY_PATH.unlink(missing_ok=True)
        print(f"Removed {removed:,} sample rows. Store was sample-only, so the file is gone.")
    else:
        kept.to_parquet(hist.HISTORY_PATH, index=False)
        print(f"Removed {removed:,} sample rows. {len(kept):,} real rows kept.")
    return removed


def seed(n_runs: int, seed_value: int) -> None:
    rng = np.random.default_rng(seed_value)

    sales = D.load_sales()
    forecasts = D.load_forecasts()
    if forecasts.empty:
        raise SystemExit("No forward forecast found; run the pipeline first.")

    profiles = D.load_profiles()[["unique_id", "bucket", "history_length"]]
    skus = sorted(set(forecasts["unique_id"]))
    version = _sample_version()

    # Runs land on consecutive Mondays ending one week before the last complete
    # week, so the most recent run still has closed weeks to be scored on.
    last_complete = hist.last_complete_week()
    run_dates = [last_complete - pd.Timedelta(weeks=n_runs - i) for i in range(n_runs)]

    truth = sales[["unique_id", "ds", "y"]].copy()
    truth["ds"] = pd.to_datetime(truth["ds"])

    # A real run predicts 13 weeks out, so most of its horizon is in the future
    # and unscored. Deriving those from actuals gives zero, because there are no
    # actuals yet, which made later runs look like they forecast almost nothing.
    # Weeks past the end of history fall back to each SKU's recent average, so
    # the stored horizon is shaped like a forecast rather than a cliff.
    last_actual_week = truth["ds"].max()
    recent = (
        truth[truth["ds"] > last_actual_week - pd.Timedelta(weeks=8)]
        .groupby("unique_id", as_index=False)["y"].mean()
        .rename(columns={"y": "recent_mean"})
    )

    frames = []
    for run_date in run_dates:
        weeks = pd.date_range(run_date + pd.Timedelta(weeks=1), periods=HORIZON, freq="W-MON")
        grid = pd.MultiIndex.from_product(
            [skus, weeks], names=["unique_id", "ds"]
        ).to_frame(index=False)
        grid = grid.merge(truth, on=["unique_id", "ds"], how="left")
        grid = grid.merge(recent, on="unique_id", how="left")
        # Inside history: anchor on the actual, so the scored part of the chart
        # tracks demand. Beyond it: anchor on the recent mean, so the unscored
        # part of the horizon still looks like a forecast.
        grid["base"] = np.where(
            grid["ds"] <= last_actual_week,
            grid["y"].fillna(0.0),
            grid["recent_mean"].fillna(0.0),
        )
        grid["lead"] = ((grid["ds"] - run_date).dt.days / 7).round().astype(int)

        # Error that widens with lead, plus a small run-level bias, so the chart
        # shows the near term tracking closely and the far term drifting. This
        # is decoration, not a model: it exists to make the shape legible.
        scale = 0.06 + 0.014 * grid["lead"]
        run_bias = rng.normal(0.0, 0.03)
        noise = rng.normal(run_bias, scale, size=len(grid))
        grid["yhat"] = (grid["base"] * (1.0 + noise)).clip(lower=0.0).round(2)

        grid["model_version"] = version
        grid["forecast_date"] = run_date
        grid["run_at"] = pd.Timestamp.now()
        frames.append(grid.drop(columns=["y", "lead", "base", "recent_mean"]))

    out = pd.concat(frames, ignore_index=True)
    out = out.merge(profiles, on="unique_id", how="left")
    out["bucket"] = out["bucket"].fillna("smooth")
    out["history_length"] = out["history_length"].fillna("short")
    out["segment"] = out["bucket"] + "/" + out["history_length"]
    out["served_by"] = "sample"

    summary = hist.append(out)
    scored = hist.score_against_actuals(sales)
    scored = scored[scored["model_version"] == version]

    print(f"Seeded {n_runs} runs as {version!r}")
    print(f"  rows written  : {summary['added']:,} (replaced {summary['replaced']:,})")
    print(f"  runs in store : {summary['runs']}")
    print(f"  SKUs per run  : {len(skus):,}")
    print(f"  run dates     : {run_dates[0].date()} to {run_dates[-1].date()}")
    if scored.empty:
        print("  scored weeks  : none yet")
    else:
        y = scored["y"].sum()
        wape = (scored["yhat"] - scored["y"]).abs().sum() / y if y else float("nan")
        print(f"  scored weeks  : {scored['ds'].nunique()} "
              f"(leads {scored['lead'].min()} to {scored['lead'].max()})")
        print(f"  sample WAPE   : {wape:.3f}  <- fabricated, means nothing")
    print()
    print("These rows are sample data. The serving endpoint prefers the version the")
    print("current forward forecast came from, so they are ignored as soon as a real")
    print("run lands. To remove them now:")
    print("  python3 scripts/seed_forecast_history.py --clear")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=6,
                    help="how many weekly runs to fabricate (default 6)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed, for repeatability")
    ap.add_argument("--clear", action="store_true",
                    help="remove sample rows and exit")
    args = ap.parse_args()

    if args.clear:
        clear()
        return
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")
    seed(args.runs, args.seed)


if __name__ == "__main__":
    main()
