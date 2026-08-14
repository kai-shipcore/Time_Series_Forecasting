> **ARCHIVED 2026-08-14. Replaced by `docs/DATA_AND_PIPELINE.md`.** Its Section 1 population counts are stale, and its Section 4 provenance note is wrong about the V1 column: it cites the rebaseline log, which predates the 2026-08-13 as-of fix.

---

# Operations: running and maintaining the demand forecast

**Audience:** whoever keeps this running. No machine-learning knowledge is assumed. If you
can read a log file, use SSH and edit an environment variable, this document is written for
you.

**What this document is not.** It does not explain how the model works or why it was built
the way it was. That is `PROJECT_WRITEUP.md` for the reasoning and `ML_FORECAST_DESIGN.md`
for the full record. This document covers what the system does, what it touches, how to
tell whether it is working, and what to do when it is not.

**Companion documents.** `DEPLOYMENT.md` is the reference for server setup, secrets and the
deploy mechanics, and it is more detailed than the summary here. `CODEBASE_GUIDE.md` maps
the code. `BACKLOG.md` is the list of known work not done.

---

## 1. What this system does, in one page

The company sells vehicle seat covers and accessories, roughly 3,300 active SKUs. Somebody
has to decide how many of each to order and when. That decision needs an answer to "how
many of this will we sell per week over the next quarter", and this system produces that
answer.

It forecasts **weekly unit demand per SKU, thirteen weeks ahead**, and publishes the result
to the Demand Pilot web app, where the Action List turns it into recommended order
quantities.

**It does not forecast everything, on purpose.** About 3,000 of the 3,300 SKUs sell too
sporadically for a weekly number to mean anything: they have zero sales in most weeks, so
"expected units next week" is not a quantity a model can usefully estimate. Those are
labelled **intermittent** and no forecast is produced for them. The roughly 450 SKUs that do
sell regularly are labelled **smooth**, and although they are only about 13% of the
catalogue they carry about 83% of total units. That is why forecasting only the smooth SKUs
is a reasonable thing to do rather than a gap.

Smooth SKUs are split again by how much sales history exists for them, because a SKU that
launched four months ago cannot be forecast the same way as one with two years of history:

| Segment | Meaning | Roughly |
|---|---|---|
| smooth/short | fewer than 50 weeks of active sales | 360 SKUs |
| smooth/long | 50 or more weeks | 80 SKUs |

These labels are recalculated every week, so a SKU moves between them over time. That is
intended, and it is the reason a SKU can appear in or vanish from the forecast without
anyone changing anything.

### The number to know

The forecast's accuracy is quoted as **pooled WAPE**. Lower is better. A value of 0.20 means
the segment's forecasts were off by 20% of the units it actually sold. "Pooled" means big
SKUs count more than small ones, in proportion to their volume, which matches what an error
actually costs. Section 4 of this document gives the current figures.

---

## 2. One pipeline now, and the one that used to be here

The repository contains **two** forecasting systems, and only one of them runs.

**The LightGBM track** is the live one. It produces the forecast the Action List and
Forecast Validation pages serve, and it is what `PROJECT_WRITEUP.md` is about.

**The statsforecast track** came first and was retired on 2026-08-13. Its code is still in
the tree, under `api/legacy/`, `src/legacy/` and `scripts/legacy/`, kept as a record of the
work rather than as running code: it is the accuracy bar every figure in the design
document is compared against. Nothing calls it. Its API router is not mounted and its
pipeline is not scheduled.

**Three things went at the same time, and it is worth knowing what they were**, because
someone will find references to them:

| Deleted | Was at |
|---|---|
| The Demand Forecast page | `/planning/demand-forecast` |
| SKU Planning's Demand Forecast tab | `/planning/sku-forecasts/[sku]?tab=forecast` |
| Fourteen Next.js proxy routes | `/api/forecast/*` |

`?tab=forecast` still resolves rather than 404ing; it lands on Sales Analysis. The per-SKU
view of the served model is at `/planning/action-list/[sku]`.

### The trap that used to be here, now closed

Until that date the weekly cron ran the statsforecast pipeline **first**, because it was the
only thing that produced `sales_clean.parquet`, and the LightGBM run has no ingest of its
own. Production ran a full cross-validation and model selection every week in order to get
two files.

That made deleting the old track dangerous in a way nothing announced: it would not have
raised an error. The forecast would have carried on being served and simply stopped moving.

The fix needed no new code. `scripts/ml_prepare_data.py` had performed the whole ML-only
sequence for weeks, for the Action List's Run Forecast button, and the cron had never been
pointed at it. It now is. The `shipcore.fc_*` tables are no longer written by anything, so
the last statsforecast forecast sits there frozen at the date the track stopped running.

---

## 3. What it reads and what it writes

### Inputs

| Source | What it is | Granularity |
|---|---|---|
| `shipcore.fc_velocity_link_snapshot_forecast` | The order-line table. Complete order history, all sales channels, no date cap. | One row per order line |
| `data/processed/sales_clean.parquet` | The above, aggregated. Produced by the weekly ingest. | **One row per SKU per week.** Complete grid: every SKU has a row for every week, zero-filled where there were no sales |
| `data/processed/sku_profiles.csv` | Each SKU's segment labels and training start date. Produced by the profiling stage. | One row per SKU |
| Inventory and inbound tables | On-hand stock, preorder backlog, confirmed inbound. Read live for the Action List, not for the forecast itself. | One row per SKU per warehouse |

**On weeks.** Every week is labelled by the **Monday it ends on**, and the week runs
**Tuesday through Monday inclusive**. A label of `2026-07-13` means Tuesday 7 July through
Monday 13 July. Write that sentence down somewhere; it has caused two separate bugs in this
project, and the label semantics and the span were documented inconsistently for months.

**On what the demand number contains.** The target includes all four order types: `sales`,
`preorder`, `ttm` and `ttm_preorder`. Demand is attributed to the week the order was
**placed**, not the week it shipped. This matters for newly launched SKUs, whose launch
preorders can dominate their short history.

### Outputs

| Destination | Contents | Granularity |
|---|---|---|
| `shipcore.ml_forward_forecasts` | The current thirteen-week forecast | One row per SKU per future week |
| `shipcore.ml_forecast_history` | Every forecast ever served, accumulating | One row per SKU per target week per run |
| `shipcore.fc_forward_forecasts` | The statsforecast track's last forecast. **No longer written.** Frozen at the date that track was retired | One row per SKU per future week |
| `shipcore.fc_forecast_history` | The statsforecast track's stored predictions. **No longer written** | One row per SKU per run |
| `data/processed/ml_forecast_history.parquet` | File copy of the accumulating history | Same as the table |
| `data/history_backups/` | Dated copies of the above, last 12 kept | Weekly |

Both `ml_` tables share a column definition so they cannot drift apart. The key is
`(model_version, week_of, unique_id, ds)`, where `week_of` is the training week the run was
made from and `ds` is the week being forecast.

**The accumulating history is the one artifact that cannot be rebuilt.** Everything else
regenerates from the database. The history records what was predicted *before the outcome
was known*, and re-running an old model against an old cutoff produces a backtest, which is
a weaker and different claim. That is why it is written to both a table and a file, and why
the file is backed up weekly.

---

## 4. Current accuracy

Measured on the pinned `2026-08-03-v2` data snapshot. Pooled WAPE, lower is better. "v11"
is the current LightGBM model, "prototype" is the statsforecast track, and "V1" is the
legacy spreadsheet method the business ran on before any of this.

| Segment and window | v11 (current) | prototype | V1 (spreadsheet) |
|---|---|---|---|
| short, Mar-May | 0.1926 | 0.2028 | 0.3351 |
| short, Dec-Feb | 0.1994 | 0.2904 | 0.2240 |
| short, Oct-Dec (reference only) | 0.2473 | 0.4137 | 0.2210 |
| long, Mar-May | 0.1350 | 0.1437 | 0.2776 |
| long, Dec-Feb | 0.1389 | 0.2690 | 0.3928 |
| long, Oct-Dec | 0.1040 | 0.0918 | 0.0851 |

**Read this honestly.** The current model beats the legacy spreadsheet in **four of six**
cells, often by a wide margin, and loses both Oct-Dec cells to it by 0.02 to 0.04. Why the
velocity method is more robust in the Q4 window is **not understood** and is worth
someone's attention. Any document claiming five of six is out of date.

**Source of these figures.** `docs/rebaseline_2026-08-03-v2/ml_22_v11_hybrid.log` and
`ml_02_v1_benchmark.log`, which are the raw output of the runs, rather than a table
transcribed from them. The design doc's own v11 table had five of six prototype cells
transcribed wrong until 2026-08-13; if a number here ever disagrees with a table
elsewhere, the logs win.

The Oct-Dec short cell is marked reference-only because just 14 short SKUs were eligible at
that cutoff, too few to conclude anything from.

**The final test, run once on 2026-08-13.** On the quarantined window (cutoff 2026-05-04,
weeks 2026-05-11 to 2026-07-13, 303 SKUs):

| segment | v11 | V1 (spreadsheet) |
|---|---|---|
| smooth/short | 0.2061 | 0.3772 |
| smooth/long | 0.1324 | 0.1872 |
| TOTAL | 0.1784 | 0.3059 |

v11's error is 42% smaller than the spreadsheet's overall, and both segment differences are
statistically significant. On calibration, V1 came in 28% low over this window against v11's
0.0%, which for ordering is the difference between absorbing a shortfall by hand and not.
Note that V1's bias is season-dependent rather than always low: it ran 12.9% high in the
post-holiday window.

**Where the model earns its keep.** On this particular window v11 matches a plain
twelve-week moving average with a seasonal adjustment. That is expected: May to July is a
flat stretch. Across all four evaluation windows v11 is never significantly worse than that
simple method and is significantly better at the two seasonal turning points, which is where
ordering actually goes wrong:

| | moving average | v11 |
|---|---|---|
| Q4 ramp-up, new and growing SKUs | 46% under-forecast | 7.5% |
| after the holidays, established SKUs | 16.7% over-forecast | 4.3% |

A trailing average cannot turn a corner it has not yet seen. That is the gap the model
closes, and it is concentrated rather than spread evenly across the year. Full detail in
`ML_FORECAST_DESIGN.md` section 4.35, with the criteria fixed beforehand in 4.34.

---

## 5. The weekly run

One cron entry, on the server, as the `coverland` user:

```
0 10 * * 2 cd /opt/coverland-forecast-api && scripts/run_forecast_cron.sh >> logs/forecast_cron.log 2>&1
```

**Tuesday, not Monday, and this is load-bearing.** A week runs Tuesday to Monday and is
labelled by the Monday it ends on, so the week labelled Monday L is still open for the whole
of Monday L. A Monday run can only use the week that closed the *previous* Monday, making
every forecast a week staler than it needs to be. A Tuesday run picks up the week that
closed hours earlier. If the week convention is ever changed, this line changes with it or
the pipeline silently loses a week.

10:00 UTC was 3am Pacific when it was set. The server stays on UTC, so the Pacific
wall-clock time drifts by an hour across each daylight-saving transition. Re-adjust if that
matters.

### What the run does, in order

1. **The pipeline** (`ml_prepare_data.py --force`), which internally is four steps: sync the
   velocity snapshot, ingest and clean into `sales_clean.parquet`, profile into
   `sku_profiles.csv`, then forecast and write the `shipcore.ml_*` tables.
2. **Readiness check.** Calls `/health` and confirms the service reports `ready: true`.
3. **History backup.** Copies the accumulating history file into `data/history_backups/`,
   keeping the last twelve.

**A failed run is safe, and this is the useful property.** Every artifact is written into a
staging directory beside `data/processed/` and moved into place with `os.replace` only after
the forecast has succeeded. A crash, a failed step or a dropped SSH session therefore leaves
last week's files intact and still being served. What you get is a forecast that is a week
old, not a half-updated set where segmentation describes one week and sales describe
another. The script exits non-zero, so cron mails on the Tuesday it breaks.

Until 2026-08-13 the job ran two scripts writing directly into `data/processed/`, and had
none of that protection. It was the one path that never benefited from the staging work.

**`--force` is required** because the pipeline refuses to overwrite live files by default,
which is the right default everywhere except here.

**`--snapshot live` is load-bearing.** Without it the ML script defaults to the pinned
snapshot that exists so recorded evaluation figures cannot drift. A weekly run against a
frozen snapshot would produce the same forecast every week and look like it was working.

---

## 6. How to tell whether it is working

### The one-command check

```bash
curl -s -H "x-forecast-token: $FORECAST_API_TOKEN" http://144.24.40.252:8000/health | python3 -m json.tool
```

`/health` deliberately sits outside the token check, so it answers even if your token is
wrong. What to look at:

| Field | Meaning |
|---|---|
| `ready` | `true` means every required data file is present. `false` means the service is alive but has nothing to serve, and every planning page will return 500 |
| `missing_required` | Which files are missing when `ready` is false |
| `commit` | **Which revision is actually serving.** Compare against the tip of `main` |

**`commit` is the field people forget.** "The service is running from this directory" is
also true of a process left over from an earlier deploy. "The service started from this
commit" is not. A push that never deployed and a deploy that never took the port both show
up here as a commit mismatch, and both were invisible before this field existed.

`ready` is reported in the body rather than as a 503 on purpose: "no server" and "server
with no data" are different problems with different fixes, and conflating them wastes
time.

### What already watches it

- An hourly GitHub Actions workflow checks external reachability and confirms the token is
  still being enforced. A `200` on a token-protected endpoint is a failure there, because it
  means the API is open to the internet.
- The weekly cron mails on failure.
- The Next.js planning pages show service status and will report when they cannot reach the
  forecast server.

### After any change to the API modules

```bash
.venv/bin/python scripts/check_route_parity.py --probe
```

This confirms the API still serves the routes it is meant to. It needs no database and no
network. It exists because the statsforecast endpoints were moved into their own package,
and the obvious way to verify that split does not work: FastAPI stores an included router as
a single opaque object, so comparing route counts compares nothing. The script walks the
router tree and drives the app's real matching logic instead, and it runs a negative control
every time so that a check which has gone blind says so rather than passing.

---

## 7. When something is wrong

`DEPLOYMENT.md` has the full runbook under "the forecast API is not responding". The short
version:

| Symptom | Most likely cause | First thing to check |
|---|---|---|
| Planning pages report they cannot reach the forecast server | Service down, or `AI_SERVICE_URL` pointing somewhere wrong | `curl` `/health` from a machine with **no local server running** |
| `/health` answers but `ready` is false | The weekly cron did not run or failed | `crontab -l \| grep run_forecast_cron`, then `logs/forecast_cron.log` |
| Every planning page 500s | Same as above; no data to serve | `missing_required` in the `/health` body |
| The forecast is not moving week to week | The cron did not run, or it failed before committing. A failed run leaves the previous week's files in place by design, so a stale forecast looks identical to a healthy one from the outside | `logs/forecast_cron.log`, and `trained_through` on the Action List against the calendar |
| `commit` in `/health` is not the tip of `main` | A push that failed to deploy, or a stale process holding the port | The GitHub Actions run for that push |
| The Action List shows "SAMPLE inventory data" | A database credential is wrong | Both `DB_*` and `COMMERCE_DB_*` must be set; a partial set degrades silently rather than erroring |

**The diagnostic mistake this project made twice**, recorded so nobody makes it a third
time: a developer machine appeared to be talking to the server, and was not. The Next.js app
falls back to starting a *local* forecast service when the configured one does not answer,
so a misconfigured `AI_SERVICE_URL` looks like it works while something else answers. The
test that settles it is `curl` from a machine with no local server. Do that first, not last.

Relatedly: `Commerce_Integration` has both `.env` and `.env.local`, and **`.env.local`
wins**. Two people once edited `AI_SERVICE_URL` in `.env` on separate machines and neither
edit had any effect. Check which file is supplying the value before editing it, and restart
the dev server afterwards, because environment variables are read at startup.

---

## 8. Deploying

**Code deploys on push to `main`,** via GitHub Actions: rsync over SSH, then restart the
`coverland-forecast-api` systemd unit. There is no manual step.

**Data does not deploy.** The deploy excludes `data/`, `outputs/` and `logs/`, which under
`rsync --delete` means both "do not upload" and "do not destroy". The cron owns the server's
data; the deploy declines to touch it. Without those excludes every deploy would wipe the
server's data and leave the API serving 500s until the following Tuesday.

Those excludes are **not** a restatement of `.gitignore`, and reading them that way is a
mistake this project's own documentation used to make. Some paths under `data/` are tracked
on purpose. The rule is about ownership, not about what happens to be in git.

After a deploy, check `commit` in `/health` matches the tip of `main`.

---

## 9. Maintenance calendar

| When | What |
|---|---|
| Weekly, automatic | The Tuesday cron. Check the log if cron mails a failure |
| After each deploy | `/health` `commit` matches the tip of `main` |
| After any API code change | `scripts/check_route_parity.py --probe` |
| Twice a year | The cron's UTC time drifts an hour against Pacific at each daylight-saving transition. Re-adjust if the 3am slot matters |
| When dependencies change | Pin from `pip freeze` **on the deploy host**, not from PyPI, so the file records production rather than proposing an upgrade to it. The two environments have already drifted once |
| Ongoing | `ml_forecast_history` accumulates one run per week. Several settled runs are the precondition for retiring the old Demand Forecast page (BACKLOG 6) |

**On dependency pinning.** Every version in `requirements.txt` is pinned exactly. The five
machine-learning pins are there because results are compared at the third decimal, finer
than the drift an unpinned solver update can introduce. The service pins were added later,
after an unverifiable change to route registration in a FastAPI version nobody could name.
Do not relax either group.

---

## 10. Things that are true together

These are constraints that only make sense as a set. Changing one without the others breaks
something quietly.

- **The Tuesday cron, the `W-MON` week label, and the Tuesday-to-Monday span.** All three
  are one decision. Change the week convention and the cron day changes with it.
- **The pinned data snapshot and the pinned window anchor.** The anchor fixes which weeks
  each evaluation window covers; the snapshot fixes the values inside those weeks. Both are
  required for a recorded result to reproduce. Advancing either re-baselines every number in
  the design document.
- **`ml_prepare_data.py` and the weekly cron.** The cron's only job is to call it. If you
  change what that script does, you have changed the weekly run, and it is also what the
  Action List's Run Forecast button calls. One script, two triggers, no second copy of the
  sequence. That is deliberate: the two used to differ, and the difference was invisible
  from either screen.
- **`FORECAST_API_TOKEN` on both sides.** The Python service and the Next.js app must agree,
  or every request except `/health` returns 401.

---

## 11. Where to look next

| Question | Document |
|---|---|
| How does the model work, and why is it built this way? | `PROJECT_WRITEUP.md` |
| What was decided, and what was rejected? | `ML_FORECAST_DESIGN.md`, sections 4 and 6 |
| Where does this code live and what does it do? | `CODEBASE_GUIDE.md` |
| How do I set up a server or a local copy? | `DEPLOYMENT.md` |
| What is known to be wrong or unfinished? | `BACKLOG.md` |
| What should I be careful about? | `HANDOVER.md` |
| What happened and when? | `WORKLOG.md` |
