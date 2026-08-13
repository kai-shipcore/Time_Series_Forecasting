"""The statsforecast prototype: frozen, not dead.

This package holds the four modules that make up the original statistical
forecasting track: the per-SKU model menu, the cross-validation backtest, the
selector that picks a winner per SKU, and the reference baselines. It was the
first working forecaster in this project and the thing the LightGBM track was
built to beat. It is kept, and kept running, rather than deleted.

WHY IT IS STILL HERE, which is not the same as still being developed
---------------------------------------------------------------------
Three things depend on this code today. Read all three before removing any of
it; the second is the one that surprises people.

1. SKU Planning (`/planning/sku-forecasts/[sku]`) serves its per-SKU chart,
   backtest panel and sales history from the legacy API endpoints in
   `api/legacy.py`, which import this package. That page is deliberately out of
   scope for retirement, pending a wider website refactor. See BACKLOG item 6.

2. The weekly cron runs the legacy pipeline FIRST, because it performs the
   ingest. `scripts/run_forward_forecast.py` pulls fresh order lines from the
   database and writes `data/processed/sales_clean.parquet`. The LightGBM run
   has no ingest of its own: it reads the file the legacy run just refreshed.
   Deleting this track without first giving the ML pipeline its own ingest stops
   the LightGBM forecast getting fresh data, and it does so silently: the
   forecast keeps being served, it just stops moving. See
   `scripts/run_forecast_cron.sh`, which says the same thing at the call site.

3. The `shipcore.fc_*` tables are written by that run and read by SKU Planning.

WHAT "FROZEN" MEANS HERE
------------------------
No new work goes into this package. It is not the forecaster the project
recommends; that is the LightGBM track in `src/ml/`, evaluated in
`docs/ML_FORECAST_DESIGN.md`. Bug fixes to keep the two live consumers working
are in scope. Model changes, new candidate models and tuning are not: any
comparison this code was used for has already been recorded, and changing it now
would invalidate those records without producing anything anyone reads.

WHEN IT CAN ACTUALLY BE DELETED
-------------------------------
All three of these have to be true, in this order:

  a. SKU Planning is migrated off the legacy endpoints, closing dependency 1.
  b. The ML pipeline owns its own ingest, closing dependency 2. Until then the
     legacy run is load-bearing regardless of who reads its forecasts.
  c. The old Demand Forecast page is retired (BACKLOG 6), which additionally
     needs `ml_forecast_history` to hold several settled runs, or its accuracy
     chart is replaced by an empty one.

Only then does this package become genuinely unreferenced. Anything short of
that is a deletion with a live caller.
"""
