# Development seed

The two forward-forecast files a fresh clone needs before the planning endpoints
can serve anything. Copied into `data/processed/` by `scripts/seed_dev_data.py`.

## Why this exists

`data/processed/` is gitignored, so a clone has the code and none of the data.
The service then starts, answers `/health`, and raises on every real endpoint.
`readiness()` requires three files, and two of them were already in the
repository under `data/snapshots/2026-07-20/`. The forward forecast was not in
the repository at all, which is the entire reason a new machine could not run
the service without a database, a pipeline run, or a copy of someone's working
tree.

## What is here, and what is not

`sales_clean.parquet` and `sku_profiles.csv` are deliberately absent. They live
in `data/snapshots/2026-07-20/`, which is tracked, manifest-verified and
immutable by design. Copying them here would put a second copy of the same
850 KB in git with nothing keeping the two in step. The seed script reads them
from the snapshot instead.

## These are frozen, and they are not the current forecast

The horizon is `2026-07-27` through `2026-10-19`, trained through `2026-07-20`,
model version `v11`. That training week is exactly the last sales week in the
pinned snapshot, so the seeded history and the seeded forecast meet with no gap
and no overlap. Live `data/processed/sales_clean.parquet` has since moved on to
`2026-07-27` and is now a worse match for these files than the snapshot is.

Nothing here is refreshed by the weekly cron and nothing here should be. The
figures on a seeded machine are the figures from the week of July 20th, and any
question about what to actually order today is answered by the deployed app,
not by a development clone.

## This never reaches the server

The deploy is `rsync --delete` with `--exclude "data/"`, so nothing under
`data/` is uploaded. That exclusion is what stops a code push from overwriting
the server's freshly generated Monday forecast with a stale copy from a
developer's working tree, and it covers this directory along with everything
else in `data/`.

## Refreshing the seed

Only when the pinned snapshot moves. The two have to stay in step: the
forecast's `forecast_date` must equal the snapshot's last sales week, and the
seed script checks this and refuses to run when it does not hold. To rebuild,
copy the current `ml_forward_forecasts.parquet` and `v1_forward_forecasts.parquet`
from a run trained through the new snapshot's last week, then regenerate
`manifest.json`.
