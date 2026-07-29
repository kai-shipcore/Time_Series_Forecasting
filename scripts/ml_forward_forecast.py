#!/usr/bin/env python3
"""Train the current-best ML model on all available history and write a per-SKU
forward forecast for the dashboard. The LightGBM-track counterpart to
scripts/run_forward_forecast.py.

Examples:
  .venv/bin/python scripts/ml_forward_forecast.py
  .venv/bin/python scripts/ml_forward_forecast.py --version v11 --horizon 13
  .venv/bin/python scripts/ml_forward_forecast.py --snapshot live   # data/processed

Outputs:
  data/processed/ml_forward_forecasts.parquet   the forecast table
  outputs/models/<version>_<cutoff>.joblib       the fitted model (+ .json meta)
  data/processed/v1_forward_forecasts.parquet    V1, same grid, separate artifact
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.ml.serving.forecast import forward_forecast  # noqa: E402
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
