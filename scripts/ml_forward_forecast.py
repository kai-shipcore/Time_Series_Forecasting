#!/usr/bin/env python3
"""Train the current-best ML model on all available history and write a per-SKU
forward forecast for the dashboard. The LightGBM-track counterpart to
scripts/run_forward_forecast.py.

Examples:
  .venv/bin/python scripts/ml_forward_forecast.py
  .venv/bin/python scripts/ml_forward_forecast.py --version v11 --horizon 13
  .venv/bin/python scripts/ml_forward_forecast.py --snapshot live   # data/processed

Outputs:
  data/processed/ml_forward_forecasts.parquet   the forecast table (overwritten)
  data/processed/ml_forecast_history.parquet    every run, appended
  outputs/models/<version>_<cutoff>.joblib       the fitted model (+ .json meta)
  data/processed/v1_forward_forecasts.parquet    V1, same grid, separate artifact
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.ml.serving import store  # noqa: E402
from src.ml.serving.forecast import forward_forecast  # noqa: E402
from src.ml.serving.history import append as append_history  # noqa: E402
from src.ml.serving.models import CURRENT_BEST  # noqa: E402
from src.ml.serving.persist import save_model  # noqa: E402
from src.ml.serving.v1 import v1_forward  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default=CURRENT_BEST, help="registered model version")
    ap.add_argument("--horizon", type=int, default=13, help="forecast weeks ahead")
    ap.add_argument(
        "--snapshot", default=None,
        help="snapshot name (default: config.ML_DATA_SNAPSHOT), or 'live' for data/processed",
    )
    ap.add_argument("--out", default="data/processed/ml_forward_forecasts.parquet")
    ap.add_argument("--models-dir", default="outputs/models")
    ap.add_argument("--no-v1", action="store_true", help="skip the V1 comparison forecast")
    ap.add_argument("--no-history", action="store_true",
                    help="skip appending this run to the accumulating forecast history")
    ap.add_argument("--v1-out", default="data/processed/v1_forward_forecasts.parquet")
    ap.add_argument(
        "--v1-refresh", action=argparse.BooleanOptionalAction, default=True,
        help="re-pull orders from the DB for V1 (default: refresh)",
    )
    args = ap.parse_args()

    # snapshot: unset -> config default; 'live' -> None (live data); else the name.
    if args.snapshot is None:
        snap_kwargs = {}
    elif args.snapshot == "live":
        snap_kwargs = {"snapshot": None}
    else:
        snap_kwargs = {"snapshot": args.snapshot}

    fc, model = forward_forecast(version=args.version, horizon=args.horizon, **snap_kwargs)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(out, index=False)

    # And to the database, so the horizon is readable from a machine that did
    # not produce it. The parquet stays: it is what a clone without credentials
    # reads, and the seeded fixture depends on that path continuing to work.
    #
    # Never fatal. The forecast is already written above, and a database that is
    # briefly unreachable should not turn a good run into a failed one.
    fwd_rows = store.write_forward(fc)
    if fwd_rows >= 0:
        print(f"forecast        {fwd_rows:,} rows written to {store.FORWARD_TABLE}")
    else:
        print(f"forecast        NOT written to {store.FORWARD_TABLE} "
              "(no DB credentials, or it could not be reached)")

    cutoff = fc["forecast_date"].iloc[0].date()
    model_path = ROOT / args.models_dir / f"{args.version}_{cutoff}.joblib"
    save_model(model, model_path)

    n_sku = fc["unique_id"].nunique()
    seg = fc.drop_duplicates("unique_id")["segment"].value_counts().to_dict()
    print(f"version         {args.version}")
    print(f"trained through {cutoff}  (horizon {args.horizon}w)")
    print(f"forecast weeks  {fc['ds'].min().date()} -> {fc['ds'].max().date()}")
    print(f"SKUs            {n_sku}  ({seg})")
    print(f"rows            {len(fc):,}")
    print(f"wrote           {out.relative_to(ROOT)}")
    print(f"saved model     {model_path.relative_to(ROOT)}")

    # Append to the accumulating record as well as overwriting the current-run
    # file. The current file answers "what is the forecast"; the history answers
    # "is the model getting better", and until now every weekly run destroyed
    # the evidence needed for the second question. Re-running in the same week
    # replaces its own rows rather than duplicating them.
    if not args.no_history:
        try:
            summary = append_history(fc)
            print(f"history         +{summary['added']:,} rows "
                  f"({summary['replaced']:,} replaced), {summary['total']:,} total "
                  f"across {summary['runs']} run(s)")
            # Said out loud rather than left to inference. On the server a
            # failed table write means this run exists on one disk only, which
            # is the condition the table was added to end; on a laptop with no
            # credentials it is expected and worth distinguishing from it.
            db_rows = summary.get("db_rows", -1)
            if db_rows >= 0:
                print(f"                {db_rows:,} rows also written to "
                      f"shipcore.ml_forecast_history")
            else:
                print("                NOT written to shipcore.ml_forecast_history "
                      "(no DB credentials, or it could not be reached)")
                print("                the parquet above is the only copy of this run")
        except Exception as e:  # noqa: BLE001 — history must not fail a good forecast
            print(f"history FAILED  {type(e).__name__}: {e}")
            print("                (the forecast above is written and unaffected)")

    # V1 is a separate concern from the model forecast above, which has already
    # been trained and written: a V1 failure (e.g. DB unreachable with no cache)
    # must not take down an otherwise-successful model run.
    if not args.no_v1:
        print()
        try:
            v1_fc = v1_forward(fc[["unique_id", "ds", "forecast_date"]], refresh=args.v1_refresh)
            v1_out = ROOT / args.v1_out
            v1_out.parent.mkdir(parents=True, exist_ok=True)
            v1_fc.to_parquet(v1_out, index=False)
            n_v1_sku = v1_fc["unique_id"].nunique()
            print(f"V1              {len(v1_fc):,} rows, {n_v1_sku} SKUs "
                  f"({n_sku - n_v1_sku} not in the velocity pull)")
            print(f"wrote           {v1_out.relative_to(ROOT)}")
        except Exception as e:  # noqa: BLE001 — V1 must never block the model forecast
            print(f"V1 FAILED       {type(e).__name__}: {e}")
            print("                (model forecast above is unaffected)")


if __name__ == "__main__":
    main()
