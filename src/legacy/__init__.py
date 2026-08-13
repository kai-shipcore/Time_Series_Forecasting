"""The statsforecast prototype: retained as a record, no longer run.

This package holds the four modules that make up the original statistical
forecasting track: the per-SKU model menu, the cross-validation backtest, the
selector that picks a winner per SKU, and the reference baselines. It was the
first working forecaster in this project and the thing the LightGBM track was
built to beat.

STATUS, AS OF 2026-08-13
------------------------
Nothing calls this code. On that date the two screens it served were deleted,
the router in `api/legacy/` was unmounted, and the weekly cron was pointed at
`scripts/ml_prepare_data.py`, which never touched this track. No live path
reaches any module here.

Earlier the same day this file said the opposite, and the history is worth one
paragraph because it is the interesting part. This package was genuinely
load-bearing until that afternoon, and not for the reason anyone expected: the
weekly cron ran the statsforecast pipeline FIRST because it was the only thing
that produced `data/processed/sales_clean.parquet`, and the LightGBM run has no
ingest of its own. So production ran a full cross-validation and model selection
every Tuesday in order to obtain two files, and deleting this track would not
have raised an error. The LightGBM forecast would have carried on being served
and quietly stopped moving. That dependency is gone; the replacement script had
existed for weeks and the cron had simply never been pointed at it.

WHY IT IS KEPT
--------------
Because it is the accuracy bar. Every table in `docs/ML_FORECAST_DESIGN.md` has
a "prototype" column, and that column is this code. The success criterion for the
whole LightGBM project was never "beat a moving average", it was "beat this",
per segment. A reader checking a claim needs to be able to see what produced the
number it is compared against.

It is also a substantial part of what the project did. The model selection, the
cross-validation backtest and the conformal intervals were real work, and a
repository that records only the surviving approach misrepresents how the
conclusion was reached.

WHAT NOT TO DO WITH IT
----------------------
Do not develop it. Do not tune it. Any comparison it was used for is already
recorded, and changing it now would invalidate those records without producing
anything anyone reads.

Do not run `scripts/legacy/run_forward_forecast.py` casually on the machine that
serves production. Without `--skip-ingest` it rewrites `sku_profiles.csv`, which
moves segmentation underneath a LightGBM forecast that did not change. See
`scripts/legacy/README.md`.

RELATED
-------
- `api/legacy/` holds the sixteen API endpoints that served this track's output.
- `scripts/legacy/` holds its pipeline entry point.
- `docs/ML_FORECAST_DESIGN.md` records what it was measured against, and where it
  won and lost. It still wins one cell: smooth/long in the Oct-Dec window, where
  it scores 0.0918 against the LightGBM model's 0.1040. Why is not understood.
"""
