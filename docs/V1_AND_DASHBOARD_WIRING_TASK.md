# Task: V1 in the forward pipeline, and wire the dashboard to the LightGBM model

For Claude Code, run on the user's machine (the database is reachable there, not in a
sandbox). This continues the LightGBM serving work. The model side already exists:
`src/ml/serving/` packages v11, and `scripts/ml_forward_forecast.py` writes
`data/processed/ml_forward_forecasts.parquet` plus a saved model. This task adds the V1
comparison as a separate, freshly computed artifact and points the dashboard at the
LightGBM outputs instead of the legacy statsforecast tables.

Work one part at a time. Run the checks at the end of each part before moving on.

## Design decisions already made (do not revisit)

- V1 is stored as its own artifact, keyed by `(unique_id, ds)`, and joined in at read time.
  The model forecast table stays model-pure; V1 is never written into it.
- V1 is always recomputed from the newest database pull at forecast time, never read from a
  stale cache when the database is reachable.
- V1 is computed with the existing production formula in `scripts/compare_v1.py`. Do not
  reimplement the formula.
- Segmentation terminology is short vs long everywhere.

## Part A: compute V1 in the forward pipeline, fresh from the database

Goal: every forward run produces a V1 forecast on the same SKU and week grid as the model,
from the latest order data.

Facts about the V1 code (`scripts/compare_v1.py`):
- `load_raw(refresh=True)` pulls the unbounded-history velocity table
  (`shipcore.fc_velocity_link_snapshot_forecast`) from the database, assigns the five
  streams, saves `data/processed/orders_raw.parquet`, and returns the dataframe. If the
  database is unreachable it falls back to the cached parquet with a warning.
- `build_cumsum_index(raw)` builds the per-SKU per-stream daily cumulative index.
- `v1_daily_current(index, uid, asof)` is V1's current daily rate. `HORIZON_DAYS = 70` is
  V1's native horizon, so do not call `v1_forecast` directly for a weekly curve.
- `proportional_seasonal_modifier(start, end)` is V1's seasonal factor for a date range.

As-of convention (from `scripts/ml_02_v1_benchmark.py`): the as-of date is `cutoff - 1 day`,
not the cutoff, because a W-MON week labelled `ds` covers `[ds-7, ds-1]`.

Per-week V1 (decompose V1's daily rate onto the model's weekly grid): for each forecast week
with label `ds` (covering `ds-7 .. ds-1`),

```
v1_week = v1_daily_current(index, uid, asof) * 7 * proportional_seasonal_modifier(ds-6, ds)
```

Use the model forecast's own `(unique_id, ds)` grid so the two align exactly.

Implementation:
1. Add `src/ml/serving/v1.py` with `v1_forward(grid, snapshot=..., refresh=True)` that takes
   the model forecast grid (the `unique_id`/`ds`/`forecast_date` frame), calls
   `compare_v1.load_raw(refresh=refresh)` and `build_cumsum_index`, computes `v1_yhat` per row
   by the formula above, and returns `unique_id, forecast_date, ds, v1_yhat`. Import
   `compare_v1` by adding `ROOT/"scripts"` to `sys.path`, as `ml_02` does.
2. In `scripts/ml_forward_forecast.py`, after writing the model forecast, call `v1_forward`
   on the same grid and write `data/processed/v1_forward_forecasts.parquet`. Add a
   `--no-v1` flag to skip it, and a `--v1-refresh/--no-v1-refresh` toggle (default refresh).
   Keep the model step independent of V1 so a database outage cannot block the model run.
3. Honor the repo's env rule: ensure the correct database URL wins. The ML CLAUDE.md notes
   the shell can carry stale `DB_*` exports, so confirm the pull uses the intended
   connection (add `load_dotenv(override=True)` if the pull resolves the wrong host).

Checks for Part A:
- `.venv/bin/python scripts/ml_forward_forecast.py` prints a V1 line and writes
  `v1_forward_forecasts.parquet`.
- The V1 file has the same `(unique_id, ds)` rows as `ml_forward_forecasts.parquet` for the
  SKUs present in the velocity pull; SKUs absent from the pull have no V1 rows (acceptable).
- `orders_raw.parquet` is refreshed to the newest `order_date` after the run.

## Part B: V1 accuracy baseline for the validation view

Goal: the accuracy view compares the model against V1 on the same development windows and
segments, without retraining at dashboard load time.

1. Add `validate_v1(n_windows=3, snapshot=...)` to `src/ml/serving/forecast.py` (or a sibling
   in `serving/v1.py`) that, for each `dev_splits` window, scores V1 with the same as-of rule
   as `ml_02` (`v1_predictions` builds one 70-day total row per SKU at `cutoff - 1 day`) and
   returns per-window per-segment pooled WAPE via `src.ml.evaluate.score`. Reuse
   `compare_v1.build_cumsum_index` and `v1_forecast`; do not reimplement.
2. Add a small script or extend the forward script to write
   `outputs/reports/ml_accuracy.csv` containing both the model rows (from
   `validate_version(CURRENT_BEST)`) and the V1 rows (from `validate_v1`), with columns
   `model_version, window, cutoff, segment, n_skus, actual_units, pooled_wape, bias_pct`.
   The dashboard reads this file; it must not call `validate_*` itself, because those retrain.

Checks for Part B:
- `ml_accuracy.csv` has model rows and `v1` rows for short and long in each window.
- The model rows reproduce the recorded v11 numbers (short 0.1961, 0.2000, 0.1783; long
  0.1355, 0.1380, 0.1000).

## Part C: point the dashboard at the LightGBM outputs

All in `dashboard/lib/data.py` and the accuracy page. Keep the sample-inventory design as is.

1. `FORWARD_FORECAST` should point at `data/processed/ml_forward_forecasts.parquet`. Update
   `_read_forecasts` for its schema: `unique_id, forecast_date, ds, yhat, bucket,
   history_length, segment, model_version, served_by, run_at` (no `selected_model`, no
   `v1_yhat`). Keep filtering to the latest `forecast_date`.
2. Add `load_v1_forward()` reading `data/processed/v1_forward_forecasts.parquet`, and make
   `sku_forecast(uid)` left-join `v1_yhat` on `(unique_id, ds)` so the SKU Detail chart's
   "Spreadsheet (V1)" line still works. Where V1 is missing, the line is simply absent.
3. For the SKU Detail header "selected model", show a label built from `model_version` and
   `served_by` (for example `v11 (long model)` / `v11 (shared model)`).
4. Accuracy page: read `outputs/reports/ml_accuracy.csv` instead of `test_evaluation.csv` and
   `v1_comparison.csv`. Compare `model_version == CURRENT_BEST` rows against `model_version ==
   "v1"` rows, by window and by segment. Keep the short/long framing.
5. Leave `history_group` and all inventory-side behavior unchanged.

Checks for Part C:
- All pages pass the AppTest smoke check (see `dashboard/PAGES_BUILD_SPEC.md` for the command).
- SKU Detail shows actual, model, and V1 lines for a SKU that has V1, and drops only the V1
  line for one that does not.
- The forecast SKU count matches the latest model run (447 for the 2026-07-20 snapshot).
- The accuracy view shows the model beating V1 in five of six window/segment cells, losing
  only long in the Oct-Dec window, consistent with the design doc.

## Acceptance criteria

- V1 is a separate artifact, recomputed from a fresh database pull each forward run, joined
  into the dashboard by `(unique_id, ds)`.
- No V1 logic is reimplemented; everything routes through `scripts/compare_v1.py`.
- The dashboard reads the LightGBM model outputs and the precomputed accuracy file, never the
  legacy statsforecast tables and never a live `validate_*` call.
- A database outage degrades gracefully: the model forecast still runs, and V1 falls back to
  the cached pull with a visible warning.
```
