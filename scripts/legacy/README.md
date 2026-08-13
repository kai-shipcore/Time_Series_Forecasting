# Statsforecast pipeline scripts: kept as a record, not run

`run_forward_forecast.py` is the entry point of the original statistical
forecasting pipeline:

```
ingest -> clean -> profile -> backtest (CV) -> select -> refit -> forecast -> shipcore.fc_*
```

Nothing calls it. It was removed from the weekly cron on 2026-08-13, and the two
screens that read the forecasts it produced were deleted at the same time.

## Why it was in the weekly job long after its output stopped being used

The first three steps produce `data/processed/sales_clean.parquet` and
`data/processed/sku_profiles.csv`, which are the LightGBM pipeline's only inputs.
For a while this script was the only thing that wrote them, so the weekly job ran
a complete cross-validation and model selection in order to obtain two files.

That made the dependency dangerous rather than merely wasteful. Deleting this
script would not have produced an error. The LightGBM forecast would have carried
on being served and quietly stopped moving, because the sales file it reads would
never be rewritten again.

## What replaced it

`scripts/ml_prepare_data.py`, which already performed the same sync, ingest,
clean and profile steps followed by the LightGBM forecast, and had done since it
was written for the Action List's Run Forecast button. The cron now calls that.
It is also strictly safer: it stages every artifact in a sibling directory and
commits with `os.replace` only after the forecast succeeds, so an interrupted run
leaves the previous week's files intact instead of a half-updated set where
segmentation describes one week and sales describe another.

## If you ever need to run this

It still works, and it still writes the `shipcore.fc_*` tables. Nothing reads
them.

```
.venv/bin/python scripts/legacy/run_forward_forecast.py --skip-ingest
```

`--skip-ingest` matters here. Without it this script re-runs the ingest and
rewrites `sku_profiles.csv`, which moves segmentation underneath a LightGBM
forecast that did not change. That is the failure `docs/BACKLOG.md` item 7
describes, and it is the reason this is not a harmless script to run out of
curiosity on the machine that serves production.

## Related

- `src/legacy/` holds the models, selector, backtest and baselines this uses.
- `api/legacy/` holds the sixteen API endpoints that served its output.
- `docs/ML_FORECAST_DESIGN.md` records what it was measured against and where it
  won and lost. Its results are the "prototype" column in every table there, so
  this code is the evidence behind the accuracy bar the LightGBM work was judged
  against.
