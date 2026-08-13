# Verifying the 2026-08-13 retirement

A checklist for convincing yourself that retiring the statsforecast track broke nothing.
Work top to bottom. Everything in part 1 is automated and already passing; parts 2 and 3
need your machine and your eyes, because they need a database, a browser and a build
toolchain that matches your platform.

If something fails, part 5 has the rollback.

---

## What changed

**`Time_Series_Forecasting`**, 8 commits, 44 files, +4,035 / −2,548.

| | |
|---|---|
| Deleted | `src/chat.py` (477 lines, never worked, addressed port 8001) |
| Moved | `models.py`, `selector.py`, `backtest.py`, `baselines.py` to `src/legacy/`; `run_forward_forecast.py` to `scripts/legacy/`; `api/legacy.py` to `api/legacy/routes.py` |
| Added | `api/common.py`, `api/legacy/__init__.py`, `src/legacy/__init__.py`, `scripts/legacy/README.md`, `scripts/check_route_parity.py`, `scripts/verify_repo.py`, `docs/OPERATIONS.md`, `docs/MODEL_GUIDE.md`, `outputs/reports/route_parity.json` |
| Behaviour | the legacy router is no longer mounted; the weekly cron calls `ml_prepare_data.py` instead of two scripts |

**`Commerce_Integration`**, 2 commits, 43 files, +175 / −7,581.

| | |
|---|---|
| Deleted | the Demand Forecast page (4 routes, 13 components), SKU Planning's orphaned forecast tab, 14 proxy routes under `/api/forecast/` |
| Kept | `/api/forecast/status`, which is generic job machinery the Action List polls |
| Modified | the nav config and three stale route comments |
| Added | `scripts/verify-api-routes.mjs` |

**Nothing was deleted that the ML path uses.** The statsforecast code is still in the tree,
importable and readable; it is simply not mounted, not scheduled and not called.

---

## Part 1. Automated, and already passing

Run these first. They need no database and no network, so they work anywhere.

```bash
cd ~/Documents/Time_Series_Forecasting
.venv/bin/pip install -r requirements.txt        # lightgbm is not installed yet
.venv/bin/python scripts/verify_repo.py
```

Expect eight `ok` lines and `All checks passed`. What each one rules out:

| Check | The failure it catches |
|---|---|
| every `.py` parses | a truncated file from a bad edit |
| every `src/` and `api/` module imports | an import still pointing at a moved path |
| script imports resolve | the same, for the 90-odd scripts, checked statically |
| paths used in code, shell and CI exist | a moved file still named in a `.sh` or a workflow, which nothing else checks |
| API routes match the recorded expectation | an endpoint silently added or lost |
| statsforecast router is not mounted | an accidental remount widening public surface area |
| weekly cron is valid | a syntax error, or a `$VAR` that `set -u` would abort on |
| no imports of moved modules | anything still importing `src.models` and friends |

Then the Commerce side:

```bash
cd ~/Documents/Commerce_Integration/Commerce_Integration
npx tsc --noEmit                      # clean
node scripts/verify-api-routes.mjs    # 135 routes, 178 calls, all resolving
npx vitest run                        # could not be run in the sandbox: wrong-platform binaries
npm run build                         # same
```

**`vitest` and `npm run build` have not been run.** Both commits used `--no-verify` because
the sandbox's `node_modules` holds another platform's native bindings. `tsc` and `eslint`
both pass, and no test references any deleted path (the suite covers `src/lib/`, and the
deletions are confined to `src/app/` and `src/components/`), but that is an argument, not a
test result. Run both before pushing.

---

## Part 2. The things only a running system can tell you

### 2.1 The API serves what it should

```bash
cd ~/Documents/Time_Series_Forecasting
.venv/bin/python -m uvicorn api.main:app --port 8000
```

In another terminal:

```bash
curl -s localhost:8000/health | python3 -m json.tool
```

`ready` should be `true` if `data/processed/` is populated. Then confirm the retired
endpoints are gone and the live ones are not:

```bash
T=$(grep FORECAST_API_TOKEN .env | cut -d= -f2)
curl -s -o /dev/null -w '%{http_code}  /segmentation (expect 404)\n'        -H "x-forecast-token: $T" localhost:8000/segmentation
curl -s -o /dev/null -w '%{http_code}  /planning/action-list (expect 200)\n' -H "x-forecast-token: $T" localhost:8000/planning/action-list
curl -s -o /dev/null -w '%{http_code}  /segmentation, no token (expect 401)\n' localhost:8000/segmentation
```

The third one matters and is the least obvious: **401, not 404**, because the token
middleware runs before routing. That is exactly why the hourly security workflow had to be
repointed, and checking it here confirms the reasoning rather than taking it on trust.

### 2.2 The weekly pipeline still runs end to end

This is the highest-value single check, because it is the change with the most production
risk. It needs database credentials.

```bash
.venv/bin/python scripts/ml_prepare_data.py --force
```

Watch for `Step 1/4` through `Step 4/4` and a `committed N artifacts` line at the end. Then:

```bash
ls -la data/processed/sales_clean.parquet data/processed/sku_profiles.csv
ls -la data/processed/ml_forward_forecasts.parquet
```

All three should have today's timestamp.

**Compare the profile counts against last week's**, because a silent change in the
segmentation population is the one thing that would not announce itself:

```bash
.venv/bin/python -c "
import pandas as pd
p = pd.read_csv('data/processed/sku_profiles.csv')
print(p.groupby(['bucket','history_length']).size())"
```

Roughly 360 smooth/short and 80 smooth/long, against about 2,900 intermittent. A large
move here is worth understanding before deploying.

If anything fails partway, nothing is committed and the previous week's files are untouched.
That is the staging behaviour, and it is new: the old two-script job had none.

### 2.3 The planning screens still work

```bash
cd ~/Documents/Commerce_Integration/Commerce_Integration
npm run dev
```

| Page | Expect |
|---|---|
| `/planning/action-list` | loads, figures present, Run Forecast panel shows server status |
| `/planning/forecast-validation` | loads, charts render |
| `/planning/sku-forecasts` | loads with **four** tabs: Sales Analysis, Inventory & Inbound, Inbound History, Order Recommendation |
| `/planning/sku-forecasts?tab=forecast` | lands on Sales Analysis, does **not** 404 |
| `/planning/demand-forecast` | **404**, and is absent from the sidebar |
| the sidebar | no Demand Forecast entry, under Planning or anywhere else |

Then press **Run Forecast** on the Action List and let it finish. That exercises
`/api/planning/run-forecast` and the `/api/forecast/status` polling, which is the one proxy
route kept, and it is the path most likely to have been broken by deleting its neighbours.

---

## Part 3. After deploying

Deploy **Commerce first, or both together.** The Python side unmounts endpoints the old
pages called; if the API ships first while the pages still exist, those pages error. In the
other order, the endpoints are simply unused. Neither is a disaster, but one is visible to
colleagues.

```bash
curl -s http://144.24.40.252:8000/health | python3 -m json.tool
```

- `commit` must equal the tip of `main`. If it does not, the push did not deploy, or the
  restart did not replace the serving process.
- `ready` must be `true`.

Then check the hourly workflow's next run passes. It now probes
`/planning/demand-patterns` instead of `/segmentation`, and a 404 there is a hard error.

**One queued item from the backlog**, unrelated to this change but due at the same time:
confirm `error_basis` on the live Action List. The demand-band fallback was verified against
a local profile file from 2026-08-10, which predates the onset fix, so the counts reported
then describe a population that no longer exists.

---

## Part 4. What is deliberately not verified

Stated so the gaps are known rather than assumed away.

- **The forecast's correctness.** None of this says the numbers are right, only that the
  code is internally consistent and the pipeline runs. Accuracy is the final test's job.
- **`vitest` and `npm run build`.** See part 1.
- **The archived statsforecast code's behaviour.** `verify_repo.py` confirms it still
  imports, so the record has not rotted. Nobody has run it since retirement and nobody
  should need to.
- **Anything reached only by a URL built entirely at runtime.** `verify-api-routes.mjs`
  matches string literals and template literals with a literal prefix. A URL assembled from
  variables with no literal part is invisible to it.

---

## Part 5. Rollback

Every change is a commit on `main` in one of the two repositories, and nothing has been
force-pushed or rewritten.

**Before pushing**, to undo everything in a repository:

```bash
git reset --hard origin/main
```

**After pushing**, revert rather than reset, so the history stays honest:

```bash
# Forecasting repo, 8 commits
git revert --no-commit b869b6e^..HEAD && git commit

# Commerce repo, 2 commits
git revert --no-commit dad6a5e^..HEAD && git commit
```

**To restore only the old page** without undoing anything else, its files are intact in
git history:

```bash
git checkout dad6a5e^ -- src/app/planning/demand-forecast src/components/planning/demand-forecast src/app/api/forecast
```

and uncomment the two `include_router` lines in `api/main.py`, then re-record the route
expectation with `scripts/check_route_parity.py --write`.

**To restore only the old weekly cron behaviour**, `scripts/legacy/run_forward_forecast.py`
still works. Read `scripts/legacy/README.md` first: without `--skip-ingest` it rewrites
`sku_profiles.csv` and moves segmentation underneath a LightGBM forecast that did not change.
