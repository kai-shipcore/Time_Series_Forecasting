# Data and pipeline

Where the numbers come from and what runs weekly.

## 1. Week convention

| Property | Value |
|---|---|
| Span | Tuesday through Monday inclusive |
| Label | The Monday the week ends on; pandas frequency `W-MON` |
| Example | `2026-07-13` covers Tuesday 7 July through Monday 13 July |

The span (`src/clean.py`, `closed="right"`), the `W-MON` label and the Tuesday cron are one decision and move together.

Warning: changing the convention without changing the cron day makes the pipeline silently lose a week. Two separate defects have come from this.

Demand covers all four order types: `sales`, `preorder`, `ttm`, `ttm_preorder`, attributed to the week the order was **placed**, not the week it shipped. Launch preorders can dominate a newly launched SKU's short history.

## 2. Inputs

| Source | Contents | Granularity |
|---|---|---|
| `shipcore.fc_velocity_link_snapshot_forecast` | Complete order history, all channels, no date cap. **Source of truth** | One row per order line |
| `data/processed/sales_clean.parquet` | The above, aggregated by the weekly ingest. Complete grid, zero-filled | One row per SKU per week |
| `data/processed/sku_profiles.csv` | Segment labels and training start date | One row per SKU |
| `ecommerce_data.coverland_inventory_by_warehouse` | On-hand, allocated, backorder | One row per SKU per warehouse |
| `shipcore.fc_container_items` joined to `shipcore.fc_containers` | Confirmed and drafted inbound with ETAs | One row per container line |

Inventory and container tables are read live for the Action List only. They are not forecast inputs.

Warning: two velocity tables exist and are not interchangeable. `..._snapshot_forecast` is uncapped and feeds the forecast. `..._snapshot` carries a 120-day cap and feeds the Velocity UI page only.

## 3. Outputs

| Destination | Contents |
|---|---|
| `shipcore.ml_forward_forecasts` | Current 13-week forecast, one row per SKU per future week |
| `shipcore.ml_forecast_history` | Every forecast served, accumulating, one row per SKU per target week per run |
| `data/processed/ml_forward_forecasts.parquet` | File copy, the read fallback |
| `data/processed/ml_forecast_history.parquet` | File copy of the accumulating history |
| `data/history_backups/` | Dated copies of the history, last 12 kept |

Both `ml_` tables share a column definition. The key is `(model_version, week_of, unique_id, ds)`: `week_of` is the training week the run was made from, `ds` the week being forecast. The model version lives in a column, never a filename, so versions coexist.

**Rationale.** The accumulating history cannot be rebuilt; everything else regenerates from the database. It records what was predicted before the outcome was known. Re-running an old model against an old cutoff yields a backtest, a weaker claim. Hence a table, a file copy and a weekly backup.

Note: `shipcore.fc_forward_forecasts` and `shipcore.fc_forecast_history` are frozen at their 2026-08-13 values and belong to the retired statsforecast track. They are not a current forecast.

## 4. The pipeline

`scripts/ml_prepare_data.py` runs the whole sequence.

```bash
.venv/bin/python scripts/ml_prepare_data.py --force --horizon 13
```

| Step | Action | Writes |
|---|---|---|
| 1/4 sync | Refreshes the velocity snapshot from the order source | `fc_velocity_link_snapshot_forecast` |
| 2/4 ingest + clean | Pulls orders, aggregates to the weekly grid | `sales_clean.parquet` |
| 3/4 profile | Classifies each SKU into bucket and history length | `sku_profiles.csv` |
| 4/4 forecast | Runs LightGBM v11, writes forward and history | `ml_forward_forecasts`, `ml_forecast_history` |

| Flag | Behaviour |
|---|---|
| `--force` | Required. The pipeline refuses to overwrite live files by default |
| `--snapshot live` | Required for a weekly run. Without it the ML scripts default to the pinned snapshot, so the run produces the same forecast every week and appears to work |
| `--horizon` | Floor of 13, enforced in the UI options and the endpoint signature (`Query(default=13, ge=13, le=104)`) but not in the script. Each run replaces the stored forecast for its training week, so a shorter run clobbers a full snapshot |

### 4.1 Failed runs are safe

Artifacts are written to a staging directory beside `data/processed/` and moved into place with `os.replace` only after the forecast succeeds. A crash, failed step or dropped SSH session leaves last week's files intact and still served. The script exits non-zero, so cron mails on failure.

Warning: `os.replace` is atomic only within a filesystem. The staging directory must be a sibling of `data/processed/`, never `/tmp`.

Note: a stale forecast is indistinguishable from a healthy one on screen. Check `trained_through` on the Action List against the calendar.

## 5. The weekly run

One cron entry on the server, as the `coverland` user:

```
0 10 * * 2 cd /opt/coverland-forecast-api && scripts/run_forecast_cron.sh >> logs/forecast_cron.log 2>&1
```

The script runs `ml_prepare_data.py --force`, calls `/health` and confirms `ready: true`, then copies the accumulating history into `data/history_backups/`, keeping the last twelve.

Warning: Tuesday (day 2) is required. The week labelled Monday L is still open for all of Monday L, so a Monday run can use only the week that closed the previous Monday.

Note: 10:00 UTC was 3am Pacific when set. The server stays on UTC, so Pacific wall-clock time drifts by an hour at each DST transition.

## 6. Two pins

| Pin | Location | Fixes |
|---|---|---|
| `ML_FINAL_TEST_CUTOFF` | `config.py:44`, currently `2026-05-04` | Which weeks each evaluation window covers |
| `ML_DATA_SNAPSHOT` | `config.py:67`, currently `2026-08-03-v2` | The values inside those weeks |

Both are required for a recorded result to reproduce: the weekly refresh revises recent actuals as late orders register. Advancing either re-baselines every number in the project; advancing one does not advance the other.

Evaluation inputs come from the snapshot named by `ML_DATA_SNAPSHOT`, not from `data/processed/`, so the weekly refresh cannot move recorded results.

## 7. Catching up after a missed run

Applies when `data_freshness` reports `ok: false`, or after any week the Tuesday cron did not complete. The procedure advances the live half only.

Warning: do not touch `ML_DATA_SNAPSHOT`. `outputs/reports/final_test.json` records the snapshot it was measured against, the runner refuses to overwrite it, and the window cannot be re-run.

1. Refresh the served data and forecast.

   ```bash
   cd ~/Documents/Time_Series_Forecasting
   .venv/bin/pip install -r requirements.txt     # lightgbm is not installed by default
   .venv/bin/python scripts/ml_prepare_data.py --force
   ```

   | Check | Expected |
   |---|---|
   | `Training through` in the output | The newest complete W-MON label |
   | SKUs forecast | Approximately 340 |
   | `sales_clean.parquet` newest `ds` | Matches `Training through` |

   The SKU count is decisive. A count far from 340 means the profiler did not run or the thresholds have changed.

2. Re-run the accuracy report, only when the snapshot has been re-cut or the profiler has changed. It reads the pinned snapshot, so step 1 neither affects it nor substitutes for it.

   ```bash
   .venv/bin/python scripts/ml_accuracy_report.py
   ```

   Check the printed grid against `OVERVIEW.md` §6 before committing.

3. Commit the files together. The manifest must travel with the CSVs or the drift check reports every deploy as undateable.

   ```bash
   git add outputs/reports/ml_accuracy.csv \
           outputs/reports/ml_accuracy_by_sku.csv \
           outputs/reports/ml_accuracy_meta.json
   git commit -m "Re-run the accuracy report against snapshot 2026-08-03-v2"
   ```

4. Confirm both checks clear.

   ```bash
   curl -s http://127.0.0.1:8000/health | python3 -m json.tool | grep -A3 -E "data_freshness|accuracy_report"
   ```

   Both read `"ok": true`, and on Forecast Validation the amber line above section 02 and the banner above section 01 both disappear. If either persists, the `detail` string names which step did not take effect.

## 8. Health checks

```bash
curl -s http://144.24.40.252:8000/health | python3 -m json.tool
```

`/health` sits outside the token check, so it answers even with a wrong token.

| Field | Meaning |
|---|---|
| `ready` | `true` means every required data file is present. `false` means the service is alive with nothing to serve, and every planning page returns 500 |
| `missing_required` | Which files are missing when `ready` is false |
| `commit` | Which revision is actually serving. Compare against the tip of `main`. Catches a push that never deployed and a deploy that never took the port |
| `data_freshness` | Whether the served week is the newest complete one. `ok: false` means the forecast on screen is stale |
| `accuracy_report` | Whether the pinned accuracy report still describes the served population. `ok: false` means Forecast Validation sections 01 and 05 describe a cohort that has moved |

**Rationale.** `data_freshness` and `accuracy_report` are excluded from `ready` deliberately: a stale forecast is a wrong answer, not an absent one, and a 503 would take working screens down and trigger the local auto-start path against a running server. `ready` is in the body because "no server" and "server with no data" have different fixes.

An unhealthy `data_freshness` names its own remedy:

```jsonc
"data_freshness":  { "ok": false, "detail": "served data ends 2026-08-03 but the last complete week is 2026-08-10, 1 week(s) behind",
                     "fix": "scripts/ml_prepare_data.py --force" }
```

Three watchers: an hourly GitHub Actions workflow, where a `200` on a token-protected endpoint is a failure because the API is then open to the internet; the weekly cron, mailing on failure; and the planning-page service status indicator.

After any change to the API modules:

```bash
.venv/bin/python scripts/check_route_parity.py --probe
```

It confirms the API still serves the routes it should, with no database and no network. The script walks the router tree and drives the app's real matching logic, with a negative control every run.

**Rationale.** FastAPI stores an included router as one opaque object, so route counts compare nothing.

Before pushing:

```bash
.venv/bin/python scripts/verify_repo.py          # 8 static checks
.venv/bin/python scripts/smoke_planning_api.py   # every planning endpoint, in-process
```

The smoke test catches planning-page failures the static checks cannot.

## 9. Troubleshooting

| Symptom | Likely cause | First check |
|---|---|---|
| Pages cannot reach the forecast server | Service down, or `AI_SERVICE_URL` wrong | `curl /health` **from a machine with no local server running** |
| `/health` answers but `ready` is false | The cron did not run or failed | `crontab -l \| grep run_forecast_cron`, then `logs/forecast_cron.log` |
| Every planning page 500s | Same; no data to serve | `missing_required` in `/health` |
| Forecast not moving week to week | The cron failed before committing. A failed run leaves last week's files in place by design, so this looks healthy from outside | `logs/forecast_cron.log`, and `trained_through` against the calendar |
| `commit` is not the tip of `main` | A push that failed to deploy, or a stale process holding the port | The GitHub Actions run for that push |
| Action List shows "SAMPLE inventory data" | A database credential is wrong | Both `DB_*` **and** `COMMERCE_DB_*` must be set; a partial set degrades silently |

Warning: Demand Pilot auto-starts a local forecast service when the configured one does not answer, so a misconfigured `AI_SERVICE_URL` appears to work. The settling test is `curl` from a machine with no local server.

Warning: `AI_SERVICE_URL` may come from either `.env` or `.env.local`. Confirm which before editing, and restart after.

## 10. Deploying

Full reference in `DEPLOYMENT.md`.

| Rule | Detail |
|---|---|
| Code deploys on push to `main` | Via GitHub Actions. No manual step |
| Data does not deploy | The deploy excludes `data/`, `outputs/` and `logs/`, so the cron keeps sole ownership of the server's data |
| Order matters when both repositories change | Deploy Commerce first, or both together. The Python side unmounts endpoints the old pages call |

After a deploy, `commit` in `/health` matches the tip of `main`.

## 11. Maintenance calendar

| When | Action |
|---|---|
| Weekly, automatic | The Tuesday cron. Check the log if cron mails a failure |
| After each deploy | `/health` `commit` matches the tip of `main` |
| After any API code change | `scripts/check_route_parity.py --probe` |
| When the served model version **or the population** changes | Regenerate `outputs/reports/ml_accuracy.csv` via `scripts/ml_accuracy_report.py` |
| Twice a year | The cron's UTC time drifts an hour against Pacific at each DST transition |
| When dependencies change | Pin from `pip freeze` **on the deploy host**, not from PyPI |

**Accuracy report refresh trigger.** The docstring names only a change of served model version. Refreshing is also needed when the population changes, which a profiling or threshold change does. See `SCREENS.md` §3.7.

**Dependency pinning.** Every version in `requirements.txt` is pinned exactly. The five ML pins exist because results are compared at the third decimal, finer than the drift an unpinned solver update introduces. The service pins followed an unverifiable change to route registration in an unidentified FastAPI version. Neither group is to be relaxed.

## 12. Coupled constraints

Changing one element of a coupling without the others breaks something quietly.

| Coupling | Constraint |
|---|---|
| The Tuesday cron, the `W-MON` label, the Tuesday-to-Monday span | One decision |
| `ML_FINAL_TEST_CUTOFF` and `ML_DATA_SNAPSHOT` | Both required for reproducibility |
| `ml_prepare_data.py` and the weekly cron | The cron's only job is to call it, and it is also what the Action List's Run Forecast button calls. One script, two triggers, no second copy of the sequence |
| `FORECAST_API_TOKEN` on both sides | The Python service and Demand Pilot must agree, or every request except `/health` returns 401 |
| `run-forecast.tsx`'s progress bar and `ml_prepare_data.py`'s stdout | The component regexes the script's output for `Step N/4`. Renaming those prefixes silently breaks the progress bar. This contract is undeclared in the code |
