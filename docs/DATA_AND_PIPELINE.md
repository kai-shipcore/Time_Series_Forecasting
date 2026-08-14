# Data and pipeline: where the numbers come from and what runs weekly

**Audience:** whoever keeps this running. No machine-learning knowledge assumed. If you can
read a log file, use SSH and edit an environment variable, this is written for you.

**Where this sits.** `OVERVIEW.md` explains what the system is for. `MODEL.md` explains what
happens to the data once it is prepared. This document covers everything either side of
that: where the data comes from, what the weekly job does, how to tell whether it worked,
and what to do when it did not.

---

## 1. The week convention

**Read this first. It has caused two separate bugs and was documented inconsistently for
months.**

- A week runs **Tuesday through Monday inclusive**.
- A week is **labelled by the Monday it ends on**. The pandas frequency is `W-MON`.
- So a label of `2026-07-13` means Tuesday 7 July through Monday 13 July.

Three things are one decision and move together:

1. The Tuesday-to-Monday span (`src/clean.py`, `closed="right"`).
2. The `W-MON` label.
3. The **Tuesday** cron.

If the week convention is ever changed, the cron day changes with it or the pipeline
silently loses a week. Reverted to this convention on 2026-08-06, and the V1 comparison
carried a stale off-by-one from that revert until 2026-08-13 because nobody updated it at the
same time.

**What the demand number contains.** All four order types: `sales`, `preorder`, `ttm` and
`ttm_preorder`. Demand is attributed to the week the order was **placed**, not the week it
shipped. That matters most for newly launched SKUs, whose launch preorders can dominate a
short history.

---

## 2. Inputs

| Source | What it is | Granularity |
|---|---|---|
| `shipcore.fc_velocity_link_snapshot_forecast` | The order-line table. Complete order history, all channels, no date cap. **The source of truth.** | One row per order line |
| `data/processed/sales_clean.parquet` | The above, aggregated. Produced by the weekly ingest. | **One row per SKU per week.** Complete grid: every SKU has a row for every week, zero-filled |
| `data/processed/sku_profiles.csv` | Each SKU's segment labels and training start date | One row per SKU |
| `ecommerce_data.coverland_inventory_by_warehouse` | On-hand, allocated, backorder | One row per SKU per warehouse |
| `shipcore.fc_container_items` joined to `shipcore.fc_containers` | Confirmed and drafted inbound with ETAs | One row per container line |

Inventory and container tables are read **live for the Action List only**. They are not
inputs to the forecast.

**A second, 120-day-capped velocity table exists**, `shipcore.fc_velocity_link_snapshot`.
It feeds the Velocity UI page only. The forecast uses the uncapped
`..._forecast` table. Do not swap them.

---

## 3. Outputs

| Destination | Contents | Granularity |
|---|---|---|
| `shipcore.ml_forward_forecasts` | The current thirteen-week forecast | One row per SKU per future week |
| `shipcore.ml_forecast_history` | Every forecast ever served, accumulating | One row per SKU per target week per run |
| `data/processed/ml_forward_forecasts.parquet` | File copy, the read fallback | Same |
| `data/processed/ml_forecast_history.parquet` | File copy of the accumulating history | Same |
| `data/history_backups/` | Dated copies of the history, last 12 kept | Weekly |

Both `ml_` tables share a column definition so they cannot drift apart. The key is
`(model_version, week_of, unique_id, ds)`, where `week_of` is the training week the run was
made from and `ds` is the week being forecast. The model version lives in a **column**, never
in a filename, so versions coexist.

**The accumulating history is the one artifact that cannot be rebuilt.** Everything else
regenerates from the database. The history records what was predicted *before the outcome
was known*; re-running an old model against an old cutoff produces a backtest, which is a
weaker and different claim. That is why it is written to both a table and a file, and why the
file is backed up weekly.

**Frozen, no longer written.** `shipcore.fc_forward_forecasts` and
`shipcore.fc_forecast_history` belong to the retired statsforecast track. They sit at the
values they held on 2026-08-13. Nothing updates them. Do not read them for a current
forecast.

---

## 4. The pipeline

One script does the whole sequence: **`scripts/ml_prepare_data.py`**.

```bash
.venv/bin/python scripts/ml_prepare_data.py --force --horizon 13
```

| Step | What it does | Writes |
|---|---|---|
| **1/4  sync** | Refreshes the velocity snapshot from the order source | `fc_velocity_link_snapshot_forecast` |
| **2/4  ingest + clean** | Pulls orders, aggregates to the weekly grid | `sales_clean.parquet` |
| **3/4  profile** | Classifies each SKU into bucket and history length | `sku_profiles.csv` |
| **4/4  forecast** | Runs LightGBM v11, writes forward and history | `ml_forward_forecasts`, `ml_forecast_history` |

**`--force` is required** because the pipeline refuses to overwrite live files by default,
which is the right default everywhere except here.

**`--snapshot live` is load-bearing** where it is used. Without it the ML scripts default to
the pinned snapshot that exists so recorded evaluation figures cannot drift. A weekly run
against a frozen snapshot would produce the same forecast every week and look like it was
working.

**Horizon has a floor of 13**, enforced in the UI options and in the endpoint signature
(`Query(default=13, ge=13, le=104)`) but **not** in the script. Each run replaces the stored
forecast for its training week, so a shorter run clobbers a full snapshot.

### 4.1 A failed run is safe, and this is the useful property

Every artifact is written into a staging directory **beside** `data/processed/`, not in
`/tmp`, and moved into place with `os.replace` only after the forecast has succeeded.
`os.replace` is atomic within a filesystem, which is why the staging directory has to be a
sibling.

A crash, a failed step or a dropped SSH session therefore leaves last week's files intact and
still being served. What you get is a forecast that is a week old, **not** a half-updated set
where segmentation describes one week and sales describe another. The script exits non-zero,
so cron mails on the Tuesday it breaks.

**The failure mode this creates.** A stale forecast looks identical to a healthy one from the
outside. Check `trained_through` on the Action List against the calendar, not just whether
the page loads.

---

## 5. The weekly run

One cron entry, on the server, as the `coverland` user:

```
0 10 * * 2 cd /opt/coverland-forecast-api && scripts/run_forecast_cron.sh >> logs/forecast_cron.log 2>&1
```

**Tuesday, day 2, and this is load-bearing.** A week is labelled by the Monday it ends on, so
the week labelled Monday L is still open for the whole of Monday L. A Monday run can only use
the week that closed the *previous* Monday, making every forecast a week staler than it needs
to be. A Tuesday run picks up the week that closed hours earlier.

**10:00 UTC was 3am Pacific when it was set.** The server stays on UTC, so the Pacific
wall-clock time drifts by an hour across each daylight-saving transition. Re-adjust twice a
year if that matters.

The script does three things:

1. **The pipeline**, `ml_prepare_data.py --force`.
2. **Readiness check.** Calls `/health` and confirms `ready: true`.
3. **History backup.** Copies the accumulating history into `data/history_backups/`, keeping
   the last twelve.

**What used to be here, and why it is worth knowing.** Until 2026-08-13 the cron ran the
statsforecast pipeline **first**, because it was the only thing producing
`sales_clean.parquet`. Production ran a full cross-validation and model selection every week
in order to get two files. That made deleting the old track dangerous in a way nothing
announced: it would not have raised an error. The forecast would have carried on being served
and simply stopped moving. The fix needed no new code, only pointing the cron at the script
that had done the ML-only sequence for weeks.

---

## 6. Two pins that must not move casually

| Pin | In | Fixes |
|---|---|---|
| `ML_FINAL_TEST_CUTOFF` | `config.py:44`, currently `2026-05-04` | Which weeks each evaluation window covers |
| `ML_DATA_SNAPSHOT` | `config.py:67`, currently `2026-08-03-v2` | The values inside those weeks |

Both are required for a recorded result to reproduce, because the weekly refresh revises
recent actuals as late orders register. **Advancing either re-baselines every number in the
project.** Do not advance one and assume the other followed.

The ML track reads its evaluation inputs from the snapshot named by `ML_DATA_SNAPSHOT`, not
from `data/processed/`, which is why the weekly refresh cannot move recorded results.

---

## 6b. Catching the pipeline up after a missed run

Run this when `data_freshness` reports `ok: false`, or after any week the Tuesday cron did
not complete. It advances the **live** half only. `ML_DATA_SNAPSHOT` is not touched, for the
reason in Section 6 and one more: `outputs/reports/final_test.json` records the snapshot it
was measured against, `scripts/ml_41_final_test.py` refuses to overwrite it, and the window
cannot be re-run. Moving the pin strands the strongest result the project has.

**Step 1. Refresh the served data and forecast.**

```bash
cd ~/Documents/Time_Series_Forecasting
.venv/bin/pip install -r requirements.txt     # lightgbm is not installed by default
.venv/bin/python scripts/ml_prepare_data.py --force
```

Expect, on a run made any day of the week beginning 2026-08-11 or later:

| Check | Expected |
|---|---|
| `Training through` in the output | the newest complete W-MON label, `2026-08-10` at time of writing |
| SKUs forecast | **340**, not 467. The pipeline re-profiles, and the current thresholds classify 127 of the previous 467 as intermittent |
| `data/processed/sales_clean.parquet` newest `ds` | matches `Training through` |

**The SKU count is the check that matters.** Before this run the served forecast covered 467
SKUs while the current profiler recognised 340, so 127 products were being forecast that the
model declines to forecast. A run that leaves the count at 467 means profiling did not
re-run and the output should not be trusted.

**Step 2. Re-run the accuracy report.** Only needed when the snapshot has been re-cut or the
profiler has changed, which is the case here and is not the case most weeks. It reads the
pinned snapshot, so it is unaffected by Step 1 and cannot be substituted by it.

```bash
.venv/bin/python scripts/ml_accuracy_report.py
```

Check the printed grid against `OVERVIEW.md` Section 6 before committing. The figures will
move, because the population moved: v11 smooth/short in Oct-Dec should go from **0.1783** to
approximately **0.2473**, and the Mar-May cells should no longer read `n_skus=206`.

**Step 3. Commit all four files together.**

```bash
git add outputs/reports/ml_accuracy.csv \
        outputs/reports/ml_accuracy_by_sku.csv \
        outputs/reports/ml_accuracy_meta.json
git commit -m "Re-run the accuracy report against snapshot 2026-08-03-v2"
```

The manifest travels with the CSVs or the drift check reports every deploy as undateable.

**Step 4. Confirm both checks clear.**

```bash
curl -s http://127.0.0.1:8000/health | python3 -m json.tool | grep -A3 -E "data_freshness|accuracy_report"
```

Both should read `"ok": true`. On the Forecast Validation page the amber line above section
02 and the banner above section 01 both disappear. If either persists, the corresponding
step did not take effect, and the `detail` string names which.

---

## 7. How to tell whether it is working

### The one-command check

```bash
curl -s http://144.24.40.252:8000/health | python3 -m json.tool
```

`/health` deliberately sits **outside** the token check, so it answers even if your token is
wrong.

| Field | Meaning |
|---|---|
| `ready` | `true` means every required data file is present. `false` means the service is alive but has nothing to serve, and every planning page returns 500 |
| `missing_required` | Which files are missing when `ready` is false |
| `commit` | **Which revision is actually serving.** Compare against the tip of `main` |
| `data_freshness` | Whether the served week is the newest complete one. `ok: false` means the pipeline has stopped delivering and the forecast on screen is stale |
| `accuracy_report` | Whether the pinned accuracy report still describes the served population. `ok: false` means Forecast Validation sections 01 and 05 describe a cohort that has moved |

**`data_freshness` and `accuracy_report` are the two fields that answer "is what I am
looking at true", as opposed to "is the server up".** Both were added on 2026-08-14 after
each of their failure modes had happened and gone unnoticed. Neither is part of `ready`,
deliberately: a stale forecast and a stale caption are both wrong answers rather than
absent ones, and returning 503 for either would take working screens down and, on a local
setup, trigger the auto-start path against a server that is already running.

```jsonc
// A healthy pair
"data_freshness":  { "ok": true,  "detail": "data through 2026-08-10, current" },
"accuracy_report": { "ok": true,  "detail": "accuracy report matches the pinned snapshot and the served population" }

// The state this checkout was actually in on 2026-08-14
"data_freshness":  { "ok": false, "detail": "served data ends 2026-08-03 but the last complete week is 2026-08-10, 1 week(s) behind",
                     "fix": "scripts/ml_prepare_data.py --force" }
```

**`commit` is the field people forget.** "The service is running from this directory" is also
true of a process left over from an earlier deploy. A push that never deployed and a deploy
that never took the port both show up here as a commit mismatch, and both were invisible
before this field existed.

`ready` is reported in the body rather than as a 503 on purpose: "no server" and "server with
no data" are different problems with different fixes.

### What already watches it

- An **hourly GitHub Actions workflow** checks external reachability and confirms the token
  is still enforced. A `200` on a token-protected endpoint is a **failure** there, because it
  means the API is open to the internet.
- The **weekly cron** mails on failure.
- The **planning pages** show service status and report when they cannot reach the server.

### After any change to the API modules

```bash
.venv/bin/python scripts/check_route_parity.py --probe
```

Confirms the API still serves the routes it should. Needs no database and no network. It
exists because the obvious check does not work: FastAPI stores an included router as a single
opaque object, so comparing route counts compares nothing. The script walks the router tree
and drives the app's real matching logic, and runs a **negative control** every time, so a
check that has gone blind says so rather than passing.

### Before pushing anything

```bash
.venv/bin/python scripts/verify_repo.py          # 8 static checks
.venv/bin/python scripts/smoke_planning_api.py   # every planning endpoint, in-process
```

The smoke test is the one that answers "do the Action List and Forecast Validation pages
work". It found a real bug the static checks could not: both pages had been returning 500
since 2026-08-12, because a column rename left `forecast_snapshot_date()` asking for a column
that no longer existed.

---

## 8. When something is wrong

| Symptom | Most likely cause | First thing to check |
|---|---|---|
| Pages report they cannot reach the forecast server | Service down, or `AI_SERVICE_URL` wrong | `curl /health` **from a machine with no local server running** |
| `/health` answers but `ready` is false | The cron did not run or failed | `crontab -l \| grep run_forecast_cron`, then `logs/forecast_cron.log` |
| Every planning page 500s | Same; no data to serve | `missing_required` in `/health` |
| The forecast is not moving week to week | The cron failed before committing. A failed run leaves last week's files in place by design, so this looks identical to health from outside | `logs/forecast_cron.log`, and `trained_through` against the calendar |
| `commit` is not the tip of `main` | A push that failed to deploy, or a stale process holding the port | The GitHub Actions run for that push |
| Action List shows "SAMPLE inventory data" | A database credential is wrong | Both `DB_*` **and** `COMMERCE_DB_*` must be set; a partial set degrades silently |

**The diagnostic mistake this project made twice.** A developer machine appeared to be
talking to the server and was not. The Next.js app falls back to starting a *local* forecast
service when the configured one does not answer, so a misconfigured `AI_SERVICE_URL` looks
like it works while something else answers. **The test that settles it is `curl` from a
machine with no local server.** Do that first, not last.

**Relatedly:** `Commerce_Integration` has both `.env` and `.env.local`, and **`.env.local`
wins**, which is Next.js precedence rather than a project convention. Two people once edited
`AI_SERVICE_URL` in `.env` on separate machines and neither edit had any effect. Check which
file is supplying the value before editing, and restart the dev server afterwards, because
environment variables are read at startup.

---

## 9. Deploying

**Code deploys on push to `main`,** via GitHub Actions: rsync over SSH, then restart the
`coverland-forecast-api` systemd unit. There is no manual step.

**Data does not deploy.** The deploy excludes `data/`, `outputs/` and `logs/`, which under
`rsync --delete` means both "do not upload" and "do not destroy". The cron owns the server's
data; the deploy declines to touch it. Without those excludes every deploy would wipe the
server's data and leave the API serving 500s until the following Tuesday.

Those excludes are **not** a restatement of `.gitignore`, and reading them that way is a
mistake this project's documentation used to make. Some paths under `data/` are tracked on
purpose. The rule is about **ownership**, not about what happens to be in git.

**Order matters when both repositories change.** Deploy **Commerce first, or both together.**
The Python side unmounts endpoints the old pages call; if the API ships first while the pages
still exist, those pages error.

After a deploy, check `commit` in `/health` matches the tip of `main`.

`DEPLOYMENT.md` is the full reference for server setup, secrets and the runbook, and is more
detailed than this summary.

---

## 10. Running it locally, with no database

This works and is the reason a clone is useful to someone without credentials:

```bash
pip install -r requirements.txt
python scripts/seed_dev_data.py                 # fills data/processed from tracked fixtures
python -m uvicorn api.main:app --port 8000      # /health then reports ready: true
```

The 4.5 MB that makes this possible is `data/dev_seed/`, the four snapshots under
`data/snapshots/`, and the reports under `outputs/reports/`.

---

## 11. Maintenance calendar

| When | What |
|---|---|
| Weekly, automatic | The Tuesday cron. Check the log if cron mails a failure |
| After each deploy | `/health` `commit` matches the tip of `main` |
| After any API code change | `scripts/check_route_parity.py --probe` |
| **When the served model version changes** | **Regenerate `outputs/reports/ml_accuracy.csv`** via `scripts/ml_accuracy_report.py`. See the warning below |
| Twice a year | The cron's UTC time drifts an hour against Pacific at each DST transition |
| When dependencies change | Pin from `pip freeze` **on the deploy host**, not from PyPI |

**The accuracy report has a refresh trigger nobody wrote down.** Its docstring says to
refresh it "when the served model version changes". That is incomplete: it also needs
refreshing when the **population** changes, which is what a profiling or threshold change
does. The file on disk is dated 2026-07-30 and predates three such changes, so the Forecast
Validation screen is currently showing stale figures. See `SCREENS.md` Section 3.7.

**On dependency pinning.** Every version in `requirements.txt` is pinned exactly. The five
machine-learning pins are there because results are compared at the third decimal, finer than
the drift an unpinned solver update can introduce. The service pins were added later, after
an unverifiable change to route registration in a FastAPI version nobody could name. Do not
relax either group.

---

## 12. Things that are only true together

Constraints that make sense only as a set. Changing one without the others breaks something
quietly.

- **The Tuesday cron, the `W-MON` label, and the Tuesday-to-Monday span.** One decision.
- **`ML_FINAL_TEST_CUTOFF` and `ML_DATA_SNAPSHOT`.** Both required for reproducibility.
- **`ml_prepare_data.py` and the weekly cron.** The cron's only job is to call it, and it is
  also what the Action List's Run Forecast button calls. One script, two triggers, no second
  copy of the sequence. That is deliberate: the two used to differ, and the difference was
  invisible from either screen.
- **`FORECAST_API_TOKEN` on both sides.** The Python service and the Next.js app must agree,
  or every request except `/health` returns 401.
- **`run-forecast.tsx`'s progress bar and `ml_prepare_data.py`'s stdout.** The component
  regexes the script's output for `Step N/4`. Renaming those prefixes silently breaks the
  progress bar. This contract is undeclared anywhere in the code.
