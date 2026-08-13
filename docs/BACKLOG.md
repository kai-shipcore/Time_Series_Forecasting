# Backlog

Deferred work that has been decided or identified but not yet built. Each item records
what the change is, why it matters, and what blocks it. Completed work goes in
`WORKLOG.md`; this file is only for things still ahead of us.

Closed items are kept in place with their reasoning rather than deleted, because half of
their value is stopping the same question being reopened from scratch. Read the status line
first: it says whether an item was solved, decided against, or ruled out of scope, and those
are three different things.

## Index, reviewed 2026-08-12

**Still open and actionable**

| # | Item | Size |
|---|---|---|
| 6 | Retire the old Demand Forecast page, and SKU Planning's forecast tab with it | see the 2026-08-13 note in that section |
| 29 | Codebase cleanup: dead scripts and stale files | explicitly last, only if there is time |

Item 28 closed on 2026-08-13: the cron now calls `scripts/ml_prepare_data.py`, which never
touched the statsforecast track, so no live path reaches it.
| 26 | Run the final test, once | pre-registered, half a day |
| 27 | Rewrite the three stale documents | a day |
| 24 | Take the personal copy of this repository | last thing before handover, see below |

Item 25 closed on 2026-08-13: the statsforecast endpoints are in `api/legacy.py` and its
models in `src/legacy/`, with `scripts/check_route_parity.py` to prove the API still
serves what it did. That was a move, so nothing was retired by it.

Item order in that table is not priority: 24 is last on purpose, and 26 should not wait,
because a pre-registered test that sits unrun invites the pre-registration being revised to
suit whatever is convenient later.

> **24 is last, and it has to happen before anything else is deleted.** The statsforecast
> half of `api/main.py`, plus `src/models.py`, `selector.py`, `backtest.py` and
> `baselines.py`, are still in the tree only because BACKLOG 6 has not closed. A copy taken
> after that page is retired does not contain them. Details in section 24.

**Not in this file, and the largest piece of documentation debt**

`docs/PROJECT_WRITEUP.md` carries v11 and V1 figures from the July snapshot and claims the
model "matches or beats the production spreadsheet in five of six cells" with one exception.
On the current snapshot it is four of six with two exceptions, and the V1 column does not
match any measurement taken since. It is the document most likely to be read by someone
outside the project, so it is the one most worth correcting.

**Done 2026-08-11 and 2026-08-12**

| # | Outcome |
|---|---|
| 2 | Resolved via the promotion path rather than the proposed `launch_week`; the proposed fix was measured and is wrong |
| 11 | Both `shipcore.ml_*` tables written by the Tuesday cron, 6,071 rows each; no longer single-copy |
| 15 | Pipeline runs staged and committed atomically; verified by killing a real run mid-write |
| 21 | Closed. Both causes identified: the deploy's own fallback branch, and manual testing during the port-8000 work. Nothing recurring |
| 23 | A stored run now replaces the whole week rather than only the SKUs it repeats. Covered by `scripts/test_store_replace_run.py` |
| — | Unmeasured-error fallback moved from a hardcoded promoted-cohort constant to demand-band medians computed each run |
| — | 14 dead files removed, Streamlit prototype retired, production data relocated out of `dashboard/` |

**Done 2026-08-10**

| # | Outcome |
|---|---|
| 5 | Both CSV exports name their columns, with human headers, a UTF-8 BOM and correct escaping |
| 8 | Dev machine moved to Node 22; deploy now asserts the server's Node version against `.nvmrc` before installing |
| 9 | Closed by decision: the drafted figure reads the same two tables Container Planning does, so there is no second source to compare against |
| 18 | `eval_X` / `eval_y`, verified bit-identical against the re-baseline logs |
| 20 | `/health` returns the deployed commit; the hourly check fails when it is not the tip of main |

**Verify after the next deploy, not build**

| # | Item |
|---|---|
| — | Confirm `error_basis` on the live Action List. The demand-band fallback was verified against a local profile file from 2026-08-10, which predates the onset fix, so the counts reported at the time describe a population that no longer exists. The logic is verified; the numbers were stale. |

**Closed**

| # | Outcome |
|---|---|
| 1, 4 | Out of scope, and blocked on data that does not exist: inventory history, stockout event dates |
| 3 | Done. Not-forecast section ships, on a trailing-13-week actual-sales basis |
| 7, 14, 16, 17, 19 | Done or resolved |
| 10, 12 | Out of scope / decided against |
| 13 | All five sub-items closed |

---

## 1. Stockout-aware demotion in `src/profile.py`

**Status:** CLOSED 2026-08-10 as out of scope for this project. Not solved, and not
solvable here: blocker 2 below needs inventory HISTORY, and only a current snapshot exists.
Left in full because the analysis is correct and whoever has the inventory history should
start from it rather than rediscover it. Original entry follows.

**The change.** The recent-activity demotion currently reads:

```python
downgrade = (
    profiles["bucket"].isin({"smooth", "low_volume"}) &
    (profiles["recent_mean"] < RECENT_MEAN_DOWNGRADE)      # recent 13-week mean < 2.0
)
```

Add an inventory condition, so a SKU is demoted to `intermittent` only when its recent mean
is below the threshold **and** it had stock available during that period. A SKU that sold
little because it was out of stock keeps its current classification. SKUs already classified
`intermittent` are left alone; this changes only the demotion path, not the promotion path.

**Why it matters.** Demotion to `intermittent` removes a SKU from the forecast entirely
(`_smooth_only` in `src/ml/serving/forecast.py` filters to smooth SKUs). Under the current
rule, a long enough stockout suppresses recent sales, which demotes the SKU, which silently
drops it out of the forecast at exactly the point it most needs reordering. The low sales are
a supply outcome, not a demand signal, and the classifier cannot currently tell the
difference.

**Blockers.**
1. ~~No inventory input.~~ **Partly cleared (July 2026).** Real inventory now exists at
   `data/inventory/inventory_snapshot.csv`, written by `scripts/export_inventory_snapshot.py`
   from the same tables the Commerce app uses. `profile.py` still reads only sales history
   (`unique_id, ds, y`) and would need that file, or the query behind it, passed in. What
   remains genuinely blocked is blocker 2 below, which the snapshot does not help with.
2. Correctness over time. Profiling is used in backtests, where every input must be as-of the
   cutoff (see `asof_history_length` and `eligible_skus` in `src/ml/dataset.py`). A current
   inventory snapshot answers "is it in stock today", not "was it in stock during that
   13-week window", so using it to decide a historical label would leak future information
   into past windows. Doing this properly needs inventory history, or an explicit scope of
   forward runs only with the limitation recorded.

**Next step.** Confirm whether the inventory tables carry history or only a current snapshot.
That answer decides whether this is the full fix or a forward-only version.

---

## 22. The service dependencies are unpinned while the model's are exact

**Status: DONE 2026-08-12.** All 18 packages pinned from `pip freeze` on the deploy host.
See the resolution at the end of this section; the problem statement below is as written
before it was fixed.

`requirements.txt` pins five packages exactly (`pandas`, `numpy`, `scikit-learn`,
`lightgbm`, `statsforecast`) and leaves twelve unpinned, including every one that serves the
API: `fastapi`, `uvicorn[standard]`, `sqlalchemy`, `psycopg2-binary`, `plotly`.

CLAUDE.md explains the pins: "ML dependencies are pinned to exact versions in
requirements.txt, because results are compared at the third decimal." That reasoning was
applied to the model and not to the service. So a rebuilt server, or any `pip install -r
requirements.txt` run today, gets whatever FastAPI is current rather than the one the code
was written against, and nothing records which that was.

**Found by tripping over it.** An attempt to move the legacy endpoints into their own module
could not be verified locally, because FastAPI registers an included router as a single
opaque `_IncludedRouter` object instead of copying its routes into `app.routes`. Whether the
server behaves the same way is unknown, because the server's version is unknown. That is the
whole problem in one sentence: a behavioural change in an unpinned framework, discovered by
accident, on a service with no test suite.

**The fix is to pin them**, from whatever the server currently has rather than from today's
PyPI, so the pins record reality instead of asserting a new one. `pip freeze` on the server
gives the true set. Doing it the other way round would upgrade production while claiming to
stabilise it.

**Not done because it needs the server.** Pinning to versions read off a laptop would be a
guess, and an unlucky guess is a failed deploy on a service with no staging environment.

**Resolved 2026-08-12** by pinning from `pip freeze` on the server. The two environments
had drifted, which is what the item predicted:

| package | server | this laptop |
|---|---|---|
| fastapi | 0.141.1 | 0.138.0 |
| uvicorn | 0.52.0 | 0.49.0 |
| plotly | 6.9.0 | 6.8.0 |
| pyarrow (transitive) | 25.0.0 | 24.0.0 |

The five ML pins matched exactly on both, so no recorded model result is affected.

**A wrong correction, recorded because it is the same mistake as the item.** Midway through
this work the entry's "FastAPI 0.141" was changed to "0.138.0" on the strength of the
laptop's `.venv`, where `_IncludedRouter` does exist at `fastapi/routing.py:1518`. Finding
the class there was treated as confirmation, when it only showed the class exists in both
versions. The original number was the server's and was correct. The evidence was local, the
claim was about production, and the gap between those two is the entire subject of this
item. Left in rather than quietly reverted.

**Consequence for anyone installing this file.** `pip install -r requirements.txt` on a
machine at 0.138.0 now upgrades FastAPI to 0.141.1. That is the intended direction: the
pins record production. The port-8000 verification difficulty that started this item should
be retried against 0.141.1 before concluding anything about how routes are registered.

---

## 2. `train_start` does two jobs, and promoted SKUs can never be backtested

**Status: RESOLVED 2026-08-11, by a different route than this item proposes. The fix below
was measured and is wrong; read that before acting on anything here.**

**The proposed fix does not work.** This item asks for a stable `launch_week` used for
as-of eligibility. Measured on the pinned snapshot, that admits 178 to 226 SKUs per window
of which 93% to 95% have NO training rows at that cutoff, because the ramp trim removes
their pre-onset history. They would be scored as zero forecasts and every pooled figure
would be destroyed.

**The failure it predicts does not occur either.** It warns that SKUs are scored without
usable history. Measured: zero SKUs in any window are scored with fewer than the minimum.

**The real defect was upstream, in promotion itself.** `src/profile.py` assigned three
constants on promotion: `train_start` to 13 weeks ago, `active_weeks` to 13,
`history_length` to `short`. Only 15 of 190 promoted SKUs genuinely had 13 weeks; the
median had 34 and the maximum 111. Because that `train_start` sits in the future relative
to every cutoff, all 190 had negative history and were absent from every recorded figure,
which is 41% of the smooth set.

Fixed by detecting each SKU's real smooth-history onset (`_smooth_onset`). `train_start`
becomes a fixed historical date, so eligibility stops being non-stationary, which is what
this item was really about. Scored population went from 266/251/66 to 356/327/103. Design
doc Section 4.32.

**One consumer broke and was fixed with it.** `src/planning/calc.py` identified promoted
SKUs by `active_weeks == 13` to size their safety stock. Onset detection destroyed that
signature; `src/profile.py` now writes an explicit `promoted` column.

Original entry follows.

**The problem.** `src/profile.py` promotes an intermittent SKU to smooth/short when its recent
13 weeks look smooth, and sets `train_start` to the first of those 13 weeks. That is right for
training: the earlier intermittent period should not be learned from. But `train_start` is also
what `eligible_skus` uses to decide whether a SKU had enough history at a backtest cutoff, and
for a promoted SKU the value is not a launch date. It advances with every profiling run while
the evaluation windows stay pinned, so those SKUs show negative history at every cutoff and are
never scored. On the current snapshot that is 187 of 447 served SKUs, 42%, carrying 14.8% of
forecast units and 20% of recommended units.

**Why it matters.** Excluding them from scoring is defensible on its own terms. The
consequences are what need attention. A large minority of what the dashboard serves has never
been measured and, under this design, cannot be. Their safety stock falls back to the segment
median error rather than anything observed about them, and the reliability tier displays as
"none", which reads as not-yet-measured rather than not-measurable. The docstring on
`eligible_skus` asserted that `train_start` is stable, which is how the issue stayed invisible;
that claim is now corrected in place.

**The fix.** Separate the two meanings: a stable `launch_week` (or first active week) that never
moves, used for as-of eligibility, and `train_start` for where usable training history begins,
which may move on promotion. Eligibility would then ask a question with a fixed answer.

**Blocker, and why this is not a quick change.** Changing eligibility changes the scored SKU
population, which re-baselines every recorded number in the version log. It needs the same
treatment as advancing the snapshot (design doc Section 4.21): a deliberate decision, a
re-run of the recorded versions, and the old figures kept for comparison. Worth pairing with
any other change that forces a re-baseline rather than spending one on its own.

---

## 3. A comprehensive action list, including SKUs the model does not forecast

**Status: DONE.** The Action List carries a Not-forecast section
(`not-forecast-section.tsx`, `not-forecast-table.tsx`) with its own filters, search,
sorting, pagination and CSV export.

**The basis chosen, which is what this item was actually asking for.** Trailing 13 weeks of
ACTUAL sales, not a forecast, from the candidates listed below. The screen and the manual
both say so in those words, and the column is labelled "13w demand … actual sales over the
trailing 13 weeks, not a forecast". That answers the sequencing note's concern directly:
the two halves of the screen are not mixing measured numbers with unmeasured ones under
shared headings, because the non-forecast half never claims a forecast. A dash and a zero
are also distinguished there, a dash meaning no inventory record and zero meaning the record
reports none, which is the same refusal to imply precision that is not there.

Original entry follows.

**The change.** The action list currently covers forecastable SKUs only: 432 of roughly 3,300.
Everything else is intermittent and has no forecast by design. The list should eventually cover
every SKU a planner is responsible for, not only the ones the model can speak to.

**Why it is not just a filter change.** Every quantity on that screen is derived from a
forecast: coverage demand, safety stock, the recommended order quantity, the projected stockout
date, the reliability tier. An intermittent SKU has none of them, so including it means either
leaving most of the row empty or deriving those figures another way. The second is the real
work, and it needs a stated basis. Candidates, none evaluated: a simple rate carried forward
from recent sales, a service-level rule on the historical inter-arrival distribution, which is
the usual treatment for intermittent demand, or a reorder point set from lead-time demand
without a forecast at all.

**Sequencing note.** Whatever basis is chosen has to be honest about its own accuracy the way
the forecast path is, or the screen ends up mixing measured numbers with unmeasured ones under
the same headings. That is the trap to avoid: the current list is trustworthy partly because
every number on it traces to something scored on the development windows.

**Related.** The forecast path already reconciles against current segmentation
(`build_planning_table` drops SKUs demoted to intermittent since the forecast ran, counted in
`df.attrs["demoted_since_forecast"]`). Those demoted SKUs are exactly the population this item
would pick up, so the two meet here.

---

## 4. Censored demand: stockout and pre-order periods misread as low demand

**Status:** CLOSED 2026-08-10 as out of scope for this project. Not solved. The blocking
fact is in "Data position" below and has not changed: nothing in this repo records when a
SKU went out of stock or was restocked, so the correction cannot be estimated, let alone
applied. Pre-order periods ARE identifiable today, so a partial version is possible for
whoever picks this up. Original entry follows.

**The problem.** The model trains on `sales_clean`, which is units sold. During a stockout, a
pre-order conversion, or the recovery period after a restock, units sold is capped by what was
available rather than by what customers wanted. The model reads those weeks as genuine demand
and learns that demand fell. This understates the forecast, which understates the recommended
order quantity, which makes the next stockout more likely.

**Where it shows up.**
- Training data: suppressed weeks are treated as ordinary observations.
- Segmentation: see backlog item 1 above, the most immediate consequence.
- Recommended order quantity: average daily sales is computed from the same suppressed weeks
  (`recent_sales` in `dashboard/lib/data.py`).

**Supporting analyses (from the forecasting team, provided as reference).**
- Shopify pre-order conversion decline: for each SKU, compare the normal-sales period before
  the stockout against an equal-length period after conversion to pre-order. Limited to the
  three Shopify channels, since pre-order is Shopify-specific.
- Marketplace restock recovery: treat pre-stockout sales as a 100% baseline and measure
  recovery at 30, 60 and 90 days after restocking on Amazon, eBay and Walmart. The stated
  goal is to estimate normalized demand and expected recovery periods.

Both measure the same underlying quantity: how far observed sales fall below true demand, and
for how long. That is the correction the model needs.

**Data position.**
- Pre-order is available. `data/processed/orders_raw.parquet` carries `west_preorder` and
  `east_preorder` streams (about 19,400 rows), so pre-order periods can be identified from
  order lines already in the repo. The source table carries the same distinction as an
  `order_type` column, which `src/ingest.py` does not filter on, so pre-order units are
  inside the training target and are attributed to the order date (design doc Section 2.1).
- Stockout and restock event dates are not available. Nothing in this repo records when a SKU
  went out of stock or was restocked, and per-channel marketplace detail is collapsed into the
  west/east/fba streams. This needs a new source before the recovery analysis is possible.

**Note on scope.** The reference screens the forecasting team designed are a way of measuring
this, not the goal. The goal is a demand estimate that is not distorted by supply, which then
flows into every number the dashboard shows.

---

## 5. The Action List CSV export ships the wire format, not the screen

**Status: DONE 2026-08-10.** Both CSV exports now build a fixed list of 19 columns with
human headers, a UTF-8 BOM so Excel reads accented characters, and RFC 4180 escaping.
`src/components/planning/action-list/csv-export.ts` in the Commerce app.

The problem statement below is kept as written, because it is the reason the column list is
fixed rather than derived from the row object, and a future change that reintroduces
`Object.keys` would undo it silently.

**The problem.** `exportCsv` in `action-list-content.tsx` builds its header from
`Object.keys(view[0])`, so the file contains every field on the row in whatever order the API
returned them: about forty columns including `forecast_over_recent`, `gap_closable_by_order`,
`n_windows`, `error_basis`, `demand_state` and `supply_gap_days`. The table above it shows ten,
chosen and ordered so a row reads as a sentence. Someone who filters the list and exports it
gets a different artefact from the one they were reading, with internal names as headings.

**Why it matters when it matters.** The export is the last step of the weekly cycle in
`docs/PLANNING_PLAN.md` §3.4: the purchaser works the list, then exports what they decided. A file
nobody can read without the schema in front of them does not close that loop, and a column
named `gap_closable_by_order` in a spreadsheet sent to a supplier is worse than absent.

**The fix, and the decision inside it.** Name the export columns explicitly rather than
deriving them. The decision is which set: the ten on screen is the obvious answer and the
wrong one, because a few fields that are deliberately not columns are useful in a spreadsheet,
the inbound ETA as a date and the estimated stockout date among them. Worth also settling
whether headings are the human labels the table uses or stable machine names, which depends on
whether the file is read by a person or loaded by something downstream.

**Related.** The non-forecast section has the same defect in its own `exportCsv`, and should be
fixed in the same pass so the two files are consistent with each other.

---

## 6. Retiring the old Demand Forecast page

**Status: decided 2026-08-13, scope widened, and the timing constraint below was
overruled deliberately.**

**What changed.** SKU Planning's Demand Forecast tab is now in scope and is to be deleted
outright, not migrated. The original entry below kept it out of scope pending a wider
website refactor. That reservation is withdrawn: the intent is that the statsforecast work
is retained as a record of the process and is not reachable from any live code path, and
leaving one page on it defeats that.

**Consequence.** With both screens gone, all sixteen endpoints in `api/legacy/routes.py`
lose their consumers and the router stops being mounted in `api/main.py`. The
`shipcore.fc_*` tables stop being read. What remains entangled is the ingest, which is why
BACKLOG 28 exists and has to land with this.

**The timing constraint below is real and is being accepted, not solved.**
`accuracy-trend.tsx` plots error per run from the legacy `fc_forecast_history`, which holds
genuinely accumulated history. Its replacement reads `ml_forecast_history`, which held two
stored runs as of 2026-08-12. Retiring now therefore trades a populated accuracy chart for
a nearly empty one that fills at one run per week. Whoever does this should say so to the
people who use that chart rather than let them discover it. It is a presentation loss, not
a data loss: the underlying history keeps accumulating either way.

### Original entry

**Status:** decided, mostly unblocked.

**The decision.** `/planning/demand-forecast` is replaced by Action List and Forecast Validation.
The AI assistant on it (`forecast-chat.tsx`) is retired rather than ported: it was a side project,
nobody uses it, and its tool calls had been failing silently for the whole life of the deployment
because `src/chat.py` addressed port 8001 where nothing listened.

**What is already covered.** Demand concentration and segment mix by the demand-patterns section.
Per-SKU trajectory, order quantity, reliability and both charts by the Action List SKU detail. The
model-versus-spreadsheet comparison, per-SKU outliers and demand-against-forecast by Forecast
Validation.

**What is not yet covered**, and should be checked before deleting rather than assumed:
`all-skus-table` (per-SKU demand, trend, year-on-year, CSV export), `segmentation-overview`
(segment counts with per-cell model and error), `segment-detail-table` (1,773 lines, the per-SKU
smooth and intermittent tables), and `model-details`.

**The one timing constraint.** `accuracy-trend.tsx` plots error per run over time from the legacy
`fc_forecast_history`, which has real accumulated history. Its replacement fills from
`ml_forecast_history`, which was empty until the first weekly run appended to it. Retiring before
that store has several settled runs swaps a working chart for an empty one.

**Scope.** The page's 13 components are self-contained: nothing outside the folder imports them.
The only cross-folder import is SKU Planning pulling its own `demand-forecast-tab`, which is a
different folder.

**Explicitly out of scope.** SKU Planning stays on the legacy statsforecast path, pending a wider
website refactor by a colleague. That keeps `run_forward_forecast.py`, the `shipcore.fc_*` tables
and the Monday 9am cron in service, so retiring this page does not retire the legacy track.

---

## 7. The Run Forecast button and who owns the data files

**Status:** resolved by the retirement above; the facts recorded so the resolution is checkable.

**What the button does.** `POST /run-forecast` spawns `scripts/run_forward_forecast.py`, which
writes its forecasts to the database (`shipcore.fc_forward_forecasts`), not to files. On the way it
runs ingest, clean and profile, and those do write to disk: `sales_clean.parquet` and
`sku_profiles.csv`, both of which the weekly data push also sends. Two writers, but both derived
from the same database, so the server's copy would be fresher rather than wrong.

**The real issue, and why the retirement settles it.** The button refreshes the legacy forecast the
old page reads. It does not regenerate `ml_forward_forecasts.parquet`, so Action List and Forecast
Validation do not move when it is pressed, while `sku_profiles.csv` does, shifting segmentation
underneath a forecast file that did not change. The button lives only on the page being retired, so
it goes with the page. If a manual trigger is ever wanted again, it should run the ML pipeline too
or say plainly which half it refreshes.

---

## 8. Node version disagrees with what the project declares

**Status: DONE 2026-08-10.** Machine moved to Node 22. `actions/setup-node` was considered
for the CI gap this entry itself flagged below, and rejected: the build runs over SSH on the
server via `appleboy/ssh-action`, not on the GitHub runner, so pinning the runner's Node
version would have been a no-op that looked like a fix.

**What was done.** Installed Homebrew's `node@22` (keg-only, so it does not conflict with the
existing unversioned `node` formula) and put it first on `PATH` via `~/.zshrc`, per Homebrew's
own caveat, rather than force-linking over the existing formula or introducing nvm onto a
machine that already standardises on Homebrew. `node -v` inside `Commerce_Integration` now
reports v22.x and `npm install` no longer prints `EBADENGINE`. Neither `engines.node` nor
`.nvmrc` changed; the machine moved to match the declaration, not the other way round.

For the actual gap, `.github/workflows/deploy.yml` now reads the major version from `.nvmrc`
and asserts it against the server's `node -v` before `npm install` runs, failing the deploy
with both versions named if they disagree, rather than silently building against whatever
Node the server happens to have. It does not install or switch Node on the server; a mismatch
is a decision to make, not a side effect of a deploy.

Original entry follows.

**Status:** decided 2026-07-31. Move the machine to Node 22. Neither repository changes.

`Commerce_Integration/package.json` declares `engines.node` as `>=20.9 <24`. The development
machine runs v24.2.0, which npm reports as `EBADENGINE` on every install.

**What settles it.** The repository also carries a `.nvmrc`, and it reads `22`. So the project
states its intended version twice, in two files, and they agree: 22, comfortably inside the
declared range. The laptop on 24.2.0 is the only thing disagreeing, and widening `engines` to
accommodate it would be editing the declaration to match an accident. That is the same move as
relaxing a version pin to make an install succeed.

**What to do.** In the Commerce checkout, `nvm use` reads `.nvmrc` and needs no argument;
`nvm install 22` first if it is not already there.

**Still worth confirming, and not blocking.** The CI workflow runs `npm install` with no
`setup-node` step, so the deployed app builds against whatever Node the server happens to have.
Unpinned rather than wrong, but it means a major-version gap between the server and a laptop stays
invisible until something breaks, which is the usual shape of "works locally, fails in production".
Pinning CI to 22 with `actions/setup-node` would close it, and is a small separate change.

---

## 9. A SKU already on a draft container still reads as needing an order

**Status:** BUILT and SHIPPED. Live on the Action List: an "already drafted" filter card, the
quantity as a sub-line under the recommended order with its ETA, and a tooltip stating that
draft units are not committed inbound and are not subtracted from the recommendation. The
design constraint that is the whole of this item is therefore satisfied on screen.

**CLOSED 2026-08-10 without the comparison, by decision.** The reasoning: the drafted figure
is read from `fc_containers` and `fc_container_items`, the same two tables the Container
Planning screens read, so there is no second source that could disagree. Comparing a number
against itself is not a verification.

**What that argument does not cover, recorded rather than left implied.** It establishes the
tables are shared; it does not establish the FILTER is. `DRAFT_STATUSES` assumes `draft` is
the only uncommitted status the sheet import ever writes, and the ETA rule is applied here
independently. If drafted totals ever look wrong, those two are the difference, not the data
source. The remaining checks at the end of this item stay written down for that reason.

**How this was reached, because the first answer was wrong.** This started as "let the purchaser
mark a SKU actioned", taken from `docs/PLANNING_PLAN.md` §3.4, where the walkthrough has the
purchaser accept a quantity and mark the SKU so the list holds their place. That framing was
rejected on review: the state it proposes to record by hand is already recorded by the container
system, and a manual flag would be a second, private, weaker copy of it. The real gap is
narrower and needs no new user action.

**What already works.** `confirmed_inbound` is `fc_container_items` joined to `fc_containers`
under `status IN ('shipped', 'packing_received')` with `eta_date >= CURRENT_DATE`, and
`build_planning_table` subtracts it as `inbound_in_window`. So a SKU whose container has shipped
already sees its recommended quantity fall and drops off the list without anyone marking
anything. That half of the problem is solved.

**The gap.** `draft` is excluded from that filter. Containers are created by the Google Sheets
import, which sets status from the header colour: blue `shipped`, orange `packing_received`,
purple or uncoloured `draft`. A container that has been decided and entered but not yet shipped
therefore contributes nothing, and the SKU keeps showing a full recommended quantity with
nothing on the row saying an order exists. At an eight-week lead time that window is long enough
to order the same units twice.

**Precedent.** `src/app/api/planning/dashboard/route.ts` and
`src/app/api/planning/sku-forecasts/inbound/route.ts` both accept `includeDrafts=1` and widen to
`('shipped', 'packing_received', 'draft')`. The question is already asked elsewhere in the
application; the action list is the screen that does not ask it.

**The design constraint, and it is the whole of the work.** Draft units must not be added to
`confirmed_inbound`. A draft is not a commitment, it can be cancelled, and crediting it against
the recommendation would under-order exactly the SKUs someone has already worried about. It
belongs on the row as its own signal, a separate figure reading "N units on a draft container",
so the purchaser sees that an order exists while the recommended quantity continues to assume it
does not. This is the same treatment the runs-high callout gets: show the disagreement rather
than resolve it silently.

**Known limit.** Draft coverage is only as current as the last sheet import, which is explicitly
not a full synchronisation: rows missing from the sheet are not deleted and a zero does not
reset an existing quantity. So this narrows the double-order window rather than closing it, and
how far depends on how promptly the sheet is maintained. Worth stating on the screen rather than
implying the figure is live.

**Dead end, recorded so it is not investigated twice.** `PurchaseOrder` and `POItem` exist in
`prisma/schema.prisma` but nothing in the application references them. A purchase order would be
an earlier signal than a draft container, but it is not how this system works: the real path is
Google Sheet to container.

**Not to be confused with item 5 or S5.** The CSV export is how a finished list leaves the
application. S5 tracks what was ordered against what was recommended and stays blocked on
purchase order outcomes, which as above do not exist here yet.

**What was built.** `src/planning/inventory.py` gained a `draft` query against the same two
tables, `data.py` carries `draft_inbound` and `draft_eta` in `inventory_columns()`, and
`calc.py` coerces them without letting either into the order formula. The Action List shows the
quantity as an italic sub-line under the recommended order, and gains an "already drafted"
filter. The SKU detail page carries it as a caveat above the order card, with the other caveats.

**What still needs checking against the live database, and why it cannot be checked here.**

1. That the numbers are right. Run `scripts/export_inventory_snapshot.py`, which now writes the
   two new columns, and compare the drafted totals against the Container Planning screens. They
   read the same tables, so they should agree exactly; if they do not, the status filter or the
   ETA rule is the place to look.
2. How many rows actually carry draft coverage. This decides whether the sub-line is the right
   shape or whether it should be promoted to its own sortable column. A sub-line was chosen on
   the expectation that most rows show nothing; if a large fraction carry drafts, "sort by what
   I have already drafted" becomes a real way to work the list and the column earns its width.
3. Whether `fc_containers.status` holds any value other than the three the sheet import writes.
   `DRAFT_STATUSES` assumes `draft` is the only uncommitted state.

**Known limit, unchanged by this work.** Draft coverage is only as current as the last sheet
import, which is not a full synchronisation: rows missing from the sheet are not deleted and a
zero does not reset an existing quantity. This narrows the double-order window rather than
closing it.

---

## 10. Nothing on the planning screens carries money

**Status:** CLOSED 2026-08-10 as out of scope, confirming the 2026-07-31 deferral rather
than revisiting it. Not solved. Still needs a decided cost basis (unit versus landed) and a
source for it before a column would mean anything, which is a data question rather than a
screen one. Original entry follows.

**The gap.** Every figure on the Action List and SKU Detail is in units. Purchasing does work to
a budget, so a recommended 1,117 units is a materially different decision at $200 a unit than at
$20, and a list of 447 SKUs cannot be triaged by spend.

**Why it is deferred rather than built.** Confirmed as real but explicitly set aside for now. It
is also not only a screen change: it needs a decided cost basis, unit cost against landed cost,
and a source for it, most likely SKU master or a related table in the Commerce database. That is
the same shape as the inventory export in §2.2, a data question that has to be settled before a
column means anything.

**Where it would surface if taken up.** Order value beside the recommended quantity on both
screens, and a value-sorted view of the list, so the largest commitments are visible without
opening 447 rows.

---

## 11. The forecast history exists on one disk and cannot be rebuilt

**Status:** BUILT 2026-07-31. The table, the dual write, the fallback read, the one-off migration
and the cron backup all exist. NOT YET RUN against a real database, which the assistant cannot
reach; see "What has to happen on a machine with credentials" at the end.

**In one sentence, for anyone who has not read the rest.** The weekly cron writes what the
model predicted before the outcome was known to a single gitignored file on one server disk,
that file is the only evidence any prediction was made in advance, and re-running the model
later cannot recreate it because that produces a backtest instead. The dual write to Postgres
exists to remove the single copy.

**Confirmed 2026-08-10 as the wanted design: live tables on the database, matching the legacy
track.** Two of them, defined together in `src/ml/serving/store.py` so they cannot drift into
disagreeing about a column:

| Table | Holds | Legacy counterpart |
|---|---|---|
| `shipcore.ml_forward_forecasts` | the current horizon, accumulating one set per `forecast_date` | `shipcore.fc_forward_forecasts` |
| `shipcore.ml_forecast_history` | every prediction ever served | `shipcore.fc_forecast_history` |

Both are created on first write by `CREATE TABLE IF NOT EXISTS`, so there is no migration
step. `scripts/ml_forward_forecast.py` writes both and prints whether each landed.

**Nothing needs building. It needs one run with credentials**, which the Tuesday cron does.
Read that run's output for `rows written to shipcore.ml_forward_forecasts` and `rows also
written to shipcore.ml_forecast_history`. If either says NOT written, the server's `.env` is
the place to look, and until then that artifact is still single-copy.

**One ordering constraint, and it is easy to miss.** The server runs whatever code was last
deployed. The `drop_leading_partial_week` fix and the `/health` commit stamp are both newer
than the last deploy, so a push has to land before the cron runs or Tuesday's run repeats the
partial-first-week bug and reports no commit.

**What is at risk.** `data/processed/ml_forecast_history.parquet` gains one entry per weekly run,
keyed by `model_version, forecast_date, unique_id, ds`. It is the only record of what the model
predicted before the outcome was known. Everything else the cron writes is regenerable from the
database by re-running the pipeline, and the accuracy reports are tracked in git; this one is not
recoverable by any means. Re-running past versions against past cutoffs would produce backtest
figures, which is a weaker and different claim: the value of this store is precisely that the
predictions were served in advance.

**Where it lives.** `/opt/coverland-forecast-api/data/processed/` on the server, written in place
by `scripts/run_forecast_cron.sh`. Gitignored, and excluded from the deploy's `rsync` by the same
rule that stops a code push overwriting the server's data. One copy, no backup.

**Why it matters more each week.** Forecast Validation's "Performance on forecasts actually
served" section and the demand-versus-forecast chart both read it, and both are empty until runs
accumulate. Backlog item 6 lists exactly this as the one timing constraint on retiring the old
Demand Forecast page. Losing the file resets that clock to zero.

**Why git is not the answer.** The server writes it, so git cannot be the transport back without
someone connecting weekly to commit the result, which is a manual step that will be forgotten. It
is also the arrangement `data/` is excluded from the deploy specifically to prevent.

**The fix the codebase already points at.** `src/ml/serving/history.py` says in its own docstring
that moving this behind functions rather than callers is "the prerequisite for serving this from an
API that does not share the filesystem". The legacy track already writes its equivalent to
`shipcore.fc_forecast_history`, a database table, which is backed up, shared, and reachable from
any machine. The ML track writing a parquet instead is the asymmetry, and closing it removes the
single-disk problem rather than mitigating it.

**A stopgap worth taking today either way.** Have `run_forecast_cron.sh` write a dated copy
somewhere durable after each successful run. Minutes of work, and it bounds the loss to one week
until the table exists.

**One thing to check first.** Both a laptop and the server have run forecasts at different times,
so two divergent copies of this file may exist. They would merge cleanly, since the key makes rows
idempotent, but nothing merges them today and the server's copy is the one being served. Worth
comparing before either is treated as authoritative.

**What was built.** `src/ml/serving/store.py` holds the table definition and a keyed upsert.
`history.py`'s `load` and `append` now use it, which is the two-function change the module's own
docstring predicted. Writes go to both stores: the table first, so a crash between them loses the
local copy rather than the shared one. Reads prefer the table and fall back to the parquet, so a
machine with credentials sees the server's runs and a clone without any still works from its own.
Nothing raises when the database is absent, because a credential-free clone is a supported way to
run this project.

`scripts/migrate_history_to_db.py` imports an existing parquet. It is idempotent on the key, which
is what lets it be run on both the laptop and the server to merge the two divergent copies rather
than having to choose one.

`run_forecast_cron.sh` also keeps twelve dated copies of the parquet under `data/history_backups/`.
That covers the case the table cannot: a run where the database was unreachable, which is exactly
when the file is the only copy.

**No backfill.** The tables start empty and fill from the next weekly run. Importing the existing
parquets was built and then removed: it solved a problem nobody had, and it would have merged two
divergent local histories into a shared store for no stated benefit. The parquets remain on disk
and remain readable as the fallback, so nothing is lost by not importing them.

**What has to happen on a machine with credentials.** None of the database path has been executed.

1. Run a forecast, or wait for Monday. Both tables are created on first write, so there is no
   separate migration step.
2. Watch the run's output for `rows written to shipcore.ml_forward_forecasts` and `rows also
   written to shipcore.ml_forecast_history`. If either says NOT written, that artifact is still
   single-copy and the server's credentials need looking at.
3. From a different machine with credentials, confirm the Action List shows that run's
   `trained_through` rather than the seeded 2026-07-20 fixture. That is the whole point of the
   forward table.
4. Confirm a clone with no `.env` still works from the seed, which is the path that must not
   regress.

---

## 12. Two loose ends from moving the ML artifacts into tables

**Status:** CLOSED 2026-08-10. The first was fixed on 2026-07-31; the remaining two are
decided against rather than deferred, so this item is not waiting on anything.

**V1 history table: decided against.** The reasoning in the entry already pointed this way
and is now the decision. The backtest grid answers the adoption question, V1 is retained for
comparison rather than as a candidate, and the Forecast Validation page states the limitation
in place rather than implying otherwise.

**Prediction interval columns: not applicable while v11 is the model.** v11 emits a point
forecast only. `_MIGRATE_PI_SQL` in `src/db.py` remains the pattern if that ever changes.

Original entry follows.

**~~`readiness()` still requires the forward-forecast parquet.~~ Fixed 2026-07-31.** The forecast
requirement is now satisfied by the parquet *or* `shipcore.ml_forward_forecasts`, matching what
`_read_forecasts` actually accepts, and the table is only probed when the file is absent so the
common paths still stat and nothing more. `scripts/export_inventory_snapshot.py` was reading the
parquet directly for its SKU list and now goes through `load_forecasts` for the same reason.

**V1 has no history table.** `v1_forward_forecasts` is recomputed per run and overwritten, so the
V1 comparison exists for the current horizon and for the pinned backtest windows, but not for
forecasts actually served over time. The Forecast Validation page says so in place rather than
implying otherwise. If model-versus-spreadsheet on served forecasts is ever wanted as a trend, V1
needs the same treatment the model forecast just received, which is one more table with the same
key. Not obviously worth it: the backtest grid already answers the adoption question, and V1 is
retained for comparison rather than as a candidate.

**Prediction intervals, when they arrive.** The ML tables have no `yhat_lo_*` / `yhat_hi_*`
columns, because v11 emits a point forecast only. The legacy track's `_MIGRATE_PI_SQL` in
`src/db.py` is the pattern for adding them later without a migration tool: `ALTER TABLE` guarded by
`information_schema` checks, run on every write.

---

## 13. UI changes on the planning screens

**Status: CLOSED 2026-08-10.** All five sub-items are resolved. They were grouped because they
are all presentation changes on screens that already have their numbers right.

13.1 to 13.4 were settled on 2026-08-05. 13.5 was sequenced behind item 14 and closed on
2026-08-10, resolved by the per-page manual rather than by inline text; the departure from
what it asked for is recorded in the sub-item rather than smoothed over. Each is marked below
rather than deleted, so the reasoning survives the decision.

**13.1 Reorder the Forecast Validation page.** *Done 2026-08-05.* Ordered as an argument rather
than by build date: the claim (model versus spreadsheet), its scope (how demand is shaped), the
claim drawn over time (demand vs forecast), the out-of-sample record (forecasts actually served),
where it is weakest (per-SKU outliers), and what is deliberately not claimed yet (final test).

Two sections moved, not one. Demand shape went from last to second, which was the point: it is the
context for every figure on the page and it arrived after all of them. Outliers went from third to
fifth, so per-SKU detail follows the aggregate evidence rather than preceding it; "where it
diverges from the pooled figure" needs the pooled figure to have been stated.

Demand shape was considered for first place, as the truest reading of "what is being validated".
Rejected because a reader opening this page wants to know whether the model is better, and burying
that under a context section risks them not reaching it.

A second reason was given at the time and was false: that demand patterns is the slowest of the
three requests, so leading with it would have opened the page on a spinner. It is in fact the
quickest. The claim came from a comment in validation-content.tsx inferring slowness from the fact
that it scans full sales history, and was repeated without being checked against the page. The
decision stands on the first reason alone.

**13.2 Change the table of contents on that page.** *Done 2026-08-05, by removing it.* The
objection recorded in this item turned out to be right: six sections listed one-for-one is
navigation the scrollbar already provides. What the bar was actually contributing was the
numbering, and the numbering belongs to the headings, where it survives.

`VALIDATION_SECTIONS` stays, because it is what assigns those numbers and keeping the order in one
place is what stops a heading claiming to be section three while sitting fourth. Verified that the
list and the render order agree.

Reopen only if this page starts being presented rather than read; a jump list earns its place when
someone needs to say "section three" and go there in front of an audience.

**13.3 Replace the demand table with a Pareto curve.** *Done 2026-08-05.* The table is gone and
the curve is in its place: SKUs ranked by demand on the x axis, cumulative share of demand on the
y, with a dotted diagonal for what perfectly even demand would look like. Without that reference a
cumulative curve looks steep whatever the distribution, since every one of them starts at zero and
ends at 100%.

`/planning/demand-patterns` gained `pareto`, the cumulative series downsampled to ~200 points,
sampled evenly by rank so the flat tail does not collapse, with both ends pinned. `concentration`
stays on the response and is what the annotation and the sentence below the chart read, because
the curve is downsampled and a figure taken off it could sit a sample interval from the truth.

Units, not revenue, and the axis says so; revenue is still item 10.

Recorded briefly on this date as declined, in favour of adding an interpretation line to the
table. That was wrong on the substance: the interpretation was worth adding and did not address
the complaint this item makes, which is that a distribution read from four rows has to be summed
in the reader's head. Both are in now. On the current 26-week window the top 5% of SKUs carry 63%
of demand, which is the kind of shape a curve shows at a glance and a table does not.

**13.4 Fit the Action List table to the screen.** *Done 2026-08-05.* Neither of the two directions
on its own, in the end.

The pinned SKU column turned out to be the largest single cost, which the note above got backwards:
it warned against shrinking that column, but the product name in it was uncapped, so one long name
set the width for every row. Capped at 15rem with the full text on hover; the SKU itself is never
truncated, so the anchor the horizontal scroll depends on is intact.

Dropping columns was rejected for the reason the item implies: every one of the nine was added for
a reason, and which three matter depends on the task. Someone checking coverage wants demand and
trend, someone placing an order wants position and quantity, and the screen cannot know which. So
the nine optional columns are individually hideable, grouped by band, remembered per reader in
localStorage, defaulting to all of them. SKU and Priority are not offered: one is the row's
identity and the scroll anchor, the other is the order the worklist is built on.

The band headers compute their `colSpan` from what is visible and disappear when their last column
goes, and the coloured rule that separates bands moves to the first column still showing rather
than vanishing with the one it was drawn on. Verified across five hiding patterns that the band row
and the column-name row always agree on width.

Not done: widening the page past the app layout's `container` cap. With the picker in place it is
no longer needed to make the table usable, and escaping the container affects how this page aligns
with every other screen. Worth doing only if the table still overflows in practice.

**13.5 Explain the priorities on the screen.** *Done. Resolved by the per-page manual rather
than by inline text, which is a departure from what this item asked for and is recorded as
such.*

`app-layout.tsx` builds a context-sensitive `manualHref` from the matched nav item, so every
screen has a manual button that opens its own page. The Action List's
(`content/manual/action-list.{en,ko}.html`) defines each label in plain language: Preorder as
SKUs carrying preorder backlog, No Stock as available inventory at or below zero, Routine as
neither, and states that the first match wins when conditions overlap.

It also does the job this item was sequenced after item 14 to make possible, and does it
explicitly: "The ★ beside it is not another priority; it marks a product in the group
generating half of four-week demand." That is the distinction item 14 created, written for a
reader rather than for the changelog. The card list separately warns that the No stock on hand
card and the No Stock priority differ, since a SKU with both no stock and backlog is badged
Preorder but belongs to that card, which is exactly the kind of near-miss the original entry
worried about.

**Where this falls short of what was asked, stated rather than glossed.** The item said the
explanation belongs where the labels appear, not in a legend elsewhere, and a manual page is a
legend elsewhere. The mitigation is that it is one click from the screen and opens on the
right page rather than at a table of contents. Reopen if anyone is seen hovering a badge
expecting a tooltip; that would be the evidence this resolution is wrong, and it costs little
to add then.

---

## 16. Weekly buckets were one day out from the documented convention

**Status:** CLOSED 2026-08-06, by changing the documentation rather than the code. The change
described below was made on 2026-08-05 and reverted the next day on evidence. Read the reversal
note at the end before the rest of this entry, which is left in its original wording.

**Reversal, 2026-08-06.** Tuesday-to-Monday was reinstated. Two reasons, in order of weight:

1. Experiment 27 swept all seven possible week phases. v11 scores best on Tue-Mon in seven of
   eight cells, consistently across three seasons, while both comparators' optima wander by
   window. Leave-one-window-out selection chooses Tuesday on every fold, for an honest
   out-of-sample gain of 0.0132 pooled WAPE against Monday, with selection optimism measured at
   0.0001. Design doc Section 4.30.
2. The SQL half of this stack has always been Tue-Mon. `api/main.py` (three queries) and
   `src/db.py` bucket with `(order_date + ((8 - ISODOW) % 7) days)`, which maps Monday to itself
   and Tuesday through Sunday forward to the next Monday. The Mon-Sun change put the Python
   ingest and the API's own queries into disagreement about which week a Monday's orders belong
   to, and nothing would have reported that. Verified after the reversal: the two agree on every
   one of 122 days tested.

So the code was right and the documentation was wrong, which is the opposite of what this entry
originally concluded. The docs have been corrected instead.

Three things encode the convention and are only correct together: `clean.py` closed="right",
`last_complete_week` stepping back an extra week on Mondays, and the cron running Tuesday. The
cron moved from Monday to Tuesday in the same change, because bucket L stays open for the whole
of Monday L and a Monday run would forecast from data seven days older than necessary.

**No recorded figure moved, in either direction.** The pinned snapshot was generated under
Tue-Mon and is once again consistent with production, so the Version Log stands and no
re-snapshot is needed. The re-snapshot that Section 4.30 called for is cancelled by the
reversal.

---

Original entry follows.

**The defect.** `src/clean.py` aggregated with `pd.Grouper(key="order_date", freq="W-MON")`, whose
defaults are `closed="right", label="right"`. That bins **Tuesday through Monday**. Everything
written about this project describes weeks as **Monday through Sunday**, labelled by the Monday they
end on.

    was      bucket 2026-08-10  <-  Tue 04 Aug .. Mon 10 Aug
    now      bucket 2026-08-10  <-  Mon 03 Aug .. Sun 09 Aug

**The fix.** `closed="left", label="right"` on the Grouper. One parameter.

`last_complete_week` in src/weeks.py needed the matching change and did not get it in the first
pass: it subtracted an extra seven days on Mondays, which was right under the old binning (bucket L
was still open on Monday L) and wrong under the new one (bucket L closed at midnight Sunday). Left
alone it would have discarded the most recent complete week on every cron run. Caught by a boundary
test rather than by reading, which is the argument for writing the test.

**No recorded figure moved.** The pinned snapshot at `data/snapshots/2026-07-20/` is a frozen copy,
and its manifest says so: "Immutable: the weekly cron refreshes data/processed/ only." Only live
data is regenerated by `clean()`, so the Version Log and Decision Log stand exactly as measured.
This was initially assessed as a re-baseline event; that was wrong, and the distinction is between
changing the data and changing the *pinned* data.

**What is now inconsistent, and is the reason this entry stays open in spirit.** Live data is binned
Monday-to-Sunday; the pinned snapshot still carries Tuesday-to-Monday. So the recorded accuracy
figures describe the method measured on one boundary while the served forecast trains on the other.
The mismatch is one day of bucket contents and does not invalidate the relative results, but it is
real and should be stated wherever those figures are quoted.

It closes at the next re-snapshot, which IS a re-baseline and should be treated as one (design
Section 4.21): a deliberate decision, a re-run of the recorded versions, and the old figures kept
for comparison. Worth pairing with backlog item 2, which forces a re-baseline anyway.

**Consequence for the cron: it stays on Monday.** Under the corrected binning a Monday run trains
through the week that ended the previous night, which is as fresh as weekly data can be. An earlier
recommendation to move it to Tuesday was a workaround for this bug and is withdrawn.

---

## 15. The on-demand pipeline is not safely interruptible

**Status: FIXED 2026-08-12.** `scripts/ml_prepare_data.py` now stages the whole run.

**How.** Everything is written into `data/.staging_<pid>/` and moved into `data/processed/`
only after every step has succeeded. A cancel, a crash, a dropped SSH session or a failed
step deletes the staging directory and leaves the previous run's files untouched and still
being served.

**The lever is one environment variable, `FORECAST_PROCESSED_DIR`, read by
`config.DATA_PROCESSED`.** It has to be an environment variable rather than an argument
because the pipeline spans processes: `ml_prepare_data` writes two artifacts itself and
shells out for the third, and that subprocess must READ the two just written rather than the
previous run's. `--snapshot live` resolves through the same config value, so the staged
inputs and outputs stay together. Arguments would have had to be threaded through every
reader as well as every writer.

`src/clean.py` and `src/profile.py` follow the same value while remaining module-level names
reassigned at call time, because `scripts/ml_36` and `promoted_sku_accuracy.py` point them at
a temp directory to stop an analysis overwriting live data, and that had to keep working.

**Verified against the real pipeline** by `scripts/_test_staged_pipeline.sh`. It checksums
`data/processed`, starts a real run, waits until the staging directory actually contains an
artifact rather than guessing a time, sends SIGKILL, and asserts the live files are
byte-identical. SIGKILL because it cannot be deferred, caught or ignored, and because a
crash, a dropped connection or a power loss is the failure worth protecting against; a
polite Ctrl-C is the easy case. An orphaned `.staging_*` directory after SIGKILL is expected
and inert, and the test says so rather than failing on it.

**The test caught a real bug in the first version of this fix, which is why it exists.**
`src/clean.py` had `OUTPUT_PATH = PROCESSED_DIR / "sales_clean.parquet"` as a module constant
computed at import. Redirecting `PROCESSED_DIR` therefore moved the CSV, which is built at
call time, and left the parquet being written straight into live `data/processed`. A kill
mid-run replaced the live sales file and not the profile: exactly the corruption staging
exists to prevent, with a staging directory beside it looking like protection. Every check I
had thought to run passed, because each confirmed the redirect reached a name rather than
that the write landed in the right place. Now resolved at call time.

**Third instance of one shape this week**, and the pattern is more useful than the instance:
a derived value stops tracking its source and nothing says so. `RatioLGBM.PARAMS` against
`self.params` (Section 4.33), the transcribed v-base figures against the recomputed ones
(Section 4.31), and `OUTPUT_PATH` against `PROCESSED_DIR`. In each case the fix is to resolve
at use rather than at import, and the way each was found was re-deriving a value rather than
reading the code.

**Honest limit, recorded rather than glossed.** Each file moves atomically, the set of four
does not. There is a window of milliseconds where some are new and some are old, against the
minutes the previous behaviour left open. Closing it entirely means swapping a directory
symlink, which changes how every reader resolves its paths and is a larger change than this
problem justifies.

**The Stop button can now come back.** It was removed on 2026-08-05 and replaced with a
"cannot be interrupted" indicator; the condition that made that necessary is gone. Re-adding
it is a Commerce-repo change and has not been done here.

Original entry follows.

**What happened.** The panel's Run button was pressed mid-week to test it. It looked stuck during
the velocity sync, Stop was pressed, and the run continued to completion regardless. That outcome
was correct by accident, and the reason is worth recording.

**Why stopping is unsafe.** `scripts/ml_prepare_data.py` writes three artifacts in sequence with no
transaction and no rollback:

    sales_clean.parquet  ->  sku_profiles.csv  ->  ml_forward_forecasts.parquet

Cancel between the first and second and the sales data describes this week while segmentation
describes last. Cancel between the second and third and both describe this week while the served
forecast describes last, which is precisely the drift `demoted_since_forecast` exists to detect,
manufactured deliberately by pressing a button. There is no path that restores the previous state.

**Interim.** The Stop button was removed 2026-08-05 and replaced with a "cannot be interrupted"
indicator. The panel no longer calls `/cancel-forecast`, though that endpoint still exists and is
still reachable from the legacy screen, so the cancelled branch is still handled and now warns that
the artifacts may be mid-sequence.

**The fix.** Write all three to temporary paths and move them into place together at the end. Moves
within a filesystem are atomic, so a cancelled or crashed run leaves the previous set untouched and
the next run starts clean. After that the Stop button can come back and mean what it says. Worth
pairing with the same treatment for the database writes, which have the same shape.

**Also unresolved, and separate.** Even with atomic writes, the velocity sync cannot be cancelled:
it is an HTTP request already delivered to the app, which completes it whether or not anything is
listening. That is a property of the remote call rather than of this pipeline, and the panel says so.

---

## 14. Best Seller is not a supply state and should leave the priority ladder

**Status:** DONE 2026-08-05. Kept in full below because it is the reasoning behind a visible
change in what the screen's counts mean, and because 13.5 has to explain the result.

**What shipped.** `_priority` in `src/planning/calc.py` keeps Preorder, No Stock and Routine.
`best_seller` stays as a column and became the middle sort key: priority, then best seller, then
recommended quantity. On the Action List the star is drawn beside the priority badge on every row
that earns it, and the summary card filters the attribute rather than the label.

**Measured against the predictions in this item.** The label counts moved exactly as expected and
nothing else did:

| | before | after |
|---|---|---|
| Preorder | 192 | 192 |
| No Stock | 16 | 16 |
| Best Seller | 35 | — |
| Routine | 189 | 224 |
| best-seller card | 35 (label) | 89 (attribute) |

54 SKUs now show a star that structurally could not before: 50 in Preorder and 4 in No Stock. That
is the cost the item described, measured. Total recommended quantity is unchanged at 2,426 units,
confirming nothing in the order arithmetic depended on `priority`.

The item predicted 86 for the attribute count and 27 for the label; the data has moved since it was
written and the figures are 89 and 35. The proportion is the same: fewer than 40% of top sellers
could carry the badge.

**`best_seller_at_risk` revisited, as this item asked.** It is still served and still not displayed,
but the reason changed. It was previously undisplayable because it cut across the priority labels;
that conflict is gone. What remains is that it is an intersection of two conditions the screen
already offers separately, and a card per intersection does not scale. The two filters compose,
which is the general answer. Recorded on the type.

**The threshold, settled 2026-08-05.** `best_seller_top_pct: 0.20` is replaced by
`best_seller_demand_share: 0.50`: best seller is now the smallest set of SKUs that together carry
half of recent units, rather than a fixed slice of the list.

A percentile describes the list, not the business. It named a fifth of the SKUs whatever demand
did, and its count moved whenever SKUs entered or left the forecastable set for reasons unrelated
to selling well. A demand share says something that survives the list changing size, and adjusts
in the right direction: concentration shrinks the set, dispersion grows it.

On the current data that is 46 of 432 SKUs, 10.6% of the list, carrying 50.2% of units. The set is
minimal by construction, and verified: drop its smallest member and the remaining 45 carry 49.66%.
The boundary is clean, 88 units in against 87 out. Ties resolve on `unique_id` so the membership is
deterministic across runs, since a star that appears and disappears without the data changing is
worse than no star.

One correction to the reasoning that prompted this. The Pareto curve was cited as evidence that a
flat 20% is arbitrary, on the grounds that the top 5% of SKUs carry 63% of demand. That figure is
for the WHOLE catalogue including the intermittent tail, which is mostly near-zero and makes
concentration look extreme. Within the forecastable set the distribution is far flatter: the top
5% carry 34%. The case against the percentile stands, but on the "describes the list, not the
business" argument rather than on the concentration figure.

Alternatives measured and not taken: an absolute cut (>= 50 units per 4 weeks gives 85 SKUs and
64.7% of demand) is the most stable and the easiest to explain, but a fixed unit count silently
becomes wrong as the business grows. Revisit if the demand-share set proves volatile week to week,
which has not been tested over time yet.

---

### Original entry

**Status:** decided 2026-08-04, unblocked. Small change, and it moves counts on a screen people
are already reading, so it wants announcing rather than slipping in.

**The criteria as they stand** (`_priority` in `src/planning/calc.py`). First match wins, so a SKU
carries one label even when several are true:

| Label | Criterion | Rank |
|-------|-----------|------|
| Preorder | `preorder_backlog > 0`, units already owed to customers | 1 |
| No Stock | `available_inventory <= 0`, nothing free to sell | 2 |
| Best Seller | in the top 20% of forecasted SKUs by `recent_units` | 3 |
| Routine | none of the above | 99 |

**The category error.** Preorder, No Stock and Routine all answer one question, what the supply
situation is. They are mutually exclusive states of a single variable, which is what makes a
precedence ladder the right shape for them. Best Seller answers a different question, how much the
SKU matters, and every SKU has both a supply state and an importance at all times. Putting them in
one slot means one of the two is discarded on every row.

**What that costs, in the three ways it shows.** The label is drained by its own membership: 86 of
432 SKUs carry the `best_seller` flag and 27 carry the badge, because a top seller is precisely the
kind of SKU that is out of stock or on preorder, so the badge thins out exactly where importance
would be most useful. Importance then vanishes from the queues that outrank it, and a best seller
with a preorder backlog is badged identically to a tail SKU with a preorder backlog, when working
the first one first is the whole reason for tracking best sellers. And because Routine is the
residual, the star ends up living in the one queue whose members by definition need nothing urgent
today. The badge currently means "top 20% by demand and nothing more pressing about it", which is
not a concept anybody asked for.

**The change.** Take it out of the ladder. Priority keeps Preorder, No Stock and Routine. Best
seller becomes an attribute on the row, its own marker and its own filter, and a tiebreaker inside
each queue: sort by priority, then by best seller, then by recommended quantity. The star then
means "top 20% by demand", full stop, and it appears on the preorder and no-stock rows where it
currently cannot.

**Consequences to expect rather than discover.** The best-seller count on the summary cards goes
from 27 to 86, and the card stops being a priority filter and becomes an attribute filter, which is
a change in kind and not only in number. Anyone reading the star today as a queue position loses
that reading. `best_seller_at_risk` already cuts across labels and should be revisited in the same
pass, since it is a third set again and the summary card was corrected once for confusing it with
the label. Nothing in the order quantity or the stockout projection depends on `priority`, so the
recommended numbers do not move.

**Still to decide.** Whether the 20% cut earns its place at all once it is an attribute. It is
`best_seller_top_pct` in the planning parameters, chosen rather than measured, and a relative cut
means the set stays a fifth of the list whatever demand does. A Pareto view (13.3) may be the
better statement of the same fact, in which case the star is a shortcut for reading it and should
be labelled as one.

---

## 17. Open port 8000 so the API is reachable without cloning this repo

**Status: DONE 2026-08-07.**

**The want.** A colleague working on the planning pages had to clone this repo and run the
service locally to see real data. Reaching the deployed API directly removes that.

**Result.** `http://144.24.40.252:8000` is reachable with `FORECAST_API_TOKEN` in the
`x-forecast-token` header. Set that plus `AI_SERVICE_URL=http://144.24.40.252:8000` and the
Planning pages work with no local Python at all.

**The blocker was never a firewall, which is the part worth remembering.** The systemd unit
ran uvicorn with `--host 127.0.0.1`, so nothing listened on the public interface and the
kernel answered arriving packets with a RST. That surfaces as "connection refused", which was
read as a closed port and written into DEPLOYMENT.md as "a per-port firewall rule". A firewall
DROP gives a timeout; a refusal means the packet arrived and no socket wanted it. Changing the
unit to `--host 0.0.0.0` was the entire fix. The host turned out not to filter at all
(`iptables` INPUT policy ACCEPT, no REJECT, firewalld and ufw inactive) and the Oracle VCN
already permitted 8000, so no firewall rule was added.

**Two things this leaves.** The host has no packet filtering, so the VCN security list is the
only network control and `FORECAST_API_TOKEN` is the entire application perimeter; it is
enforced on every path except `/health`, and only while the variable is set. And the work
uncovered a live incident, recorded as BACKLOG 21: the unit had been crash-looping since
00:01:11 UTC because an unmanaged uvicorn held the port, while every deploy reported green.

**Note on this entry.** It was referred to as "backlog 17" throughout the work before anyone
checked, and no such item existed; the file went 16, 18. Written afterwards, into the gap.

---

## 18. LightGBM 4.7 deprecates the argument every model fit uses

**Status: DONE 2026-08-10.** `src/ml/model.py:fit` now passes `eval_X` / `eval_y`.
`eval_sample_weight` is unchanged: it is not part of the deprecation, and the library still
indexes it per validation set alongside `eval_class_weight` and `eval_group`. Renaming it to
match would have raised.

**Confirmed behaviour-preserving two ways.** By construction first: both spellings meet in
the library's own `_validate_eval_set_Xy`, which returns `eval_set` unchanged on one path and
builds `[(eval_X, eval_y)]` on the other. That list is exactly what the old call passed, so
training receives the same object either way. Then empirically, because the first argument is
a reading of someone else's code: `scripts/_verify_backlog_18.sh` re-ran six experiments and
diffed them against the logs in `docs/rebaseline_2026-08-03/`. Every pooled WAPE, bias
figure, tree count, row count and bootstrap delta is identical.

**The verification failed first, for a reason worth recording.** Four of the six reported
DIFFERS. Every differing line was a `prototype:` reference figure or the stale-reference
banner, because `PROTOTYPE` had been re-measured on the 2026-08-03 snapshot AFTER those logs
were written. Nothing about the model had moved. The comparison now excludes reference
figures, which are orientation printed beside results and explicitly not pass criteria, while
retaining 28 to 66 substantive lines per script so it cannot pass vacuously.

That failure is the same shape as the one in item 20 and in `ml_12`: a check that goes red
for a reason unrelated to what it is testing gets read as noise, and then it is not a check.

Original entry follows.

`src/ml/model.py:fit` calls `self.model.fit(..., eval_set=[...])`. LightGBM 4.7.0 emits
`LGBMDeprecationWarning: The argument 'eval_set' is deprecated, use 'eval_X' and 'eval_y'
instead` on every fit, so a run prints one warning per model and a full experiment prints
a dozen.

**Not urgent, and deliberately not done yet.** 4.7.0 is what `requirements.txt` pins and
what is installed, so behaviour is unchanged and results are reproducible. The warning is
about a future release, not this one.

**Why it is worth writing down.** Two reasons. The version pin exists because results are
compared at the third decimal, which means the pin cannot be bumped casually, which in turn
means this will sit unnoticed until someone rebuilds the environment and the fit call stops
working rather than warning. And the noise is not free: twelve warnings on a run make real
warnings easy to miss, including the `per_sku_totals()` warning about eligible SKUs with no
predictions, which is a genuine bug signal.

**The change.** Switch to `eval_X` / `eval_y` and confirm on the development windows that
the numbers are bit-identical before and after. If they are not, the change is not
cosmetic and needs its own entry rather than being folded into a cleanup.

**Do it between model versions, not during one.** Touching the fit call while an experiment
is open makes any difference in that experiment ambiguous.

---

## 19. ml_is_holiday computes the week's days from the wrong span

**Status: FIXED 2026-08-06, same day it was logged.** `start = ds - 6`, `end = ds`. Verified
afterwards: still exactly 8 weeks flagged over every label in the data, so no recorded figure
moves, which is what the analysis below predicted. A second instance of the same off-by-one was
found and fixed in the Commerce app at the same time, in
`src/lib/forecast-metrics/repository.ts:getLastCompletedMonday`, where it was not harmless.
Original entry follows.

`src/ml/seasonal.py:ml_is_holiday` decides holiday membership from the days a week covers,
which is right, and then derives those days as `[ds - 7, ds - 1]`. That is the Monday-to-Sunday
span. The convention actually in force is Tuesday-to-Monday, so the correct derivation is
`[ds - 6, ds]`. The docstring states the intent correctly and the arithmetic is one day out.

**Currently harmless, and measured rather than assumed.** Over every W-MON label from 2024-06-17
to 2026-07-20, both derivations flag exactly 8 weeks and disagree on none. The window runs
Nov 20 to Dec 15 and membership needs 4 of 7 days inside it, so a one-day shift never crosses
the threshold for any week in the data.

**Why fix it anyway.** It is only harmless for this window and this date range. Moving
`ML_HOLIDAY_END`, extending the data by a year, or narrowing the window would each be enough to
make weeks start disagreeing, and the failure would appear as a small unexplained accuracy
change in whichever experiment happened to be running. The seasonal code is also exactly where
the phase sensitivity in Section 4.30 concentrates, so a known one-day error sitting there is
worth removing before anyone investigates that again.

**The change.** `start = ds - 6 days`, `end = ds`. Then confirm the flagged-week count is still 8
and that no development-window figure moves, which is the expected result given the above.

**Do it between model versions.** It cannot change a number today, but it is seasonal-adjustment
code, and touching that mid-experiment makes any difference ambiguous.

---

## 20. A push that fails to deploy is indistinguishable from one that deployed

**Status: DETECTION BUILT 2026-08-10. The cause of the original incident is still unknown
and probably always will be.**

**What was built.** `/health` now returns `commit`, read once at startup from
`.deployed_commit`, which `ci-cd.yml` writes after the rsync and before the restart.
`api-reachable.yml` compares it hourly against the tip of `main` and fails when the two
have disagreed for more than thirty minutes. The grace period is there because a deploy
takes a couple of minutes and a mismatch straight after a push is normal rather than a
fault.

This does not stop a push failing to deploy. It converts that from silent into a red
scheduled run within the hour, which is what the entry below asks for: the third and only
option that does not depend on a human remembering to look.

**Why the commit rather than a heartbeat.** It closes item 21 with the same mechanism. A
push that never deployed and a deploy that never took the port are both "the serving commit
is not the expected one", and both were invisible before.

Original entry follows.

Commit `e7a6665` was pushed to `main` at 15:02. `git ls-remote` confirms the remote branch
moved. No workflow run was created. The trigger is a plain `on: push: branches: [main]` with
no path filters, and the deploy job's only condition is `github.ref == 'refs/heads/main'`, so
nothing in `.github/workflows/ci-cd.yml` explains the miss. Two manual `workflow_dispatch`
runs on the same commit then succeeded, deploy job included, so the workflow itself is sound.

**Why this matters more than the one incident.** From the developer's side a push that
deploys and a push that does not look identical: the terminal reports success either way. The
consequence here was that the fix for a live bug, the partial-trailing-week guard, sat
undeployed for a day while work continued on top of it, and nobody noticed because nobody had
a reason to look. The next occurrence will be found the same way, by someone eventually
wondering why a change has no effect.

**Candidate causes, none confirmed.** A GitHub Actions incident at that moment; Actions
disabled or restricted at the repository or organisation level; a spending or concurrency
limit. The signed-in Actions tab shows a banner in the second and third cases, and that is
the first thing to read.

**The change.** Something that makes the absence visible rather than silent. Options, cheapest
first: check the Actions tab as a habit after pushing, which is free and unreliable; a branch
protection rule requiring the CI check to pass, which makes a missing run block rather than
pass quietly; or a scheduled job that compares the deployed commit against the tip of `main`
and reports a mismatch. The third is the only one that catches this without a human
remembering.

**Related.** `/health` already returns `repo_root`, and the deploy checks it. Adding the
deployed commit SHA to that response would make the comparison in option three a one-line
check rather than a new mechanism.

---

## 21. The deploy cannot tell whether the unit it restarted is the one serving

**Status: CAUSE FIXED 2026-08-07. MONITORING BUILT 2026-08-10. A SECOND ROUTE CLOSED and the
remaining unknown made self-identifying, 2026-08-12.**

**A second way in, closed 2026-08-12.** `src/lib/forecast-server.ts:resolveServerDir()` in the
Commerce repo read an explicit `FORECAST_SERVER_DIR` BEFORE checking `NODE_ENV`, so a value
set in production won and the app would auto-start uvicorn there. Its own docstring said the
opposite: that an unset variable is how the app declines to compete with systemd. The guard
described an intention the code did not enforce.

That is also exactly the mistake `DEPLOYMENT.md` warns about, and the hardest one to see:
`.env.local` overrides `.env` per variable, and under pm2 the value need not appear in any
file at all, which is why searching `/opt` for one found nothing on 2026-08-07. Production
now refuses first, before the variable is read, so no environment can re-enter the race.

**CLOSED 2026-08-12. There was no mystery process.** The squatter that appeared after the
deploy fix was started by hand, by the developer, from a terminal in
`/opt/coverland-forecast-api`, while doing the work that made port 8000 reachable in the
first place.

Every piece of evidence fits and none of it needed a further hypothesis. The command line
used a RELATIVE venv path, `nohup .venv/bin/python -m uvicorn ...`, where systemd's ExecStart
is absolute, so it came from a shell already sitting in that directory. The two sightings
were 00:00:06 and 00:01:11 UTC on consecutive workdays, which is 5:00pm and 5:01pm Pacific:
end of the working day, on exactly the days the port work was being done.

**So there is nothing recurring to fix.** The two causes were the deploy's own fallback branch
(fixed 2026-08-07) and a person doing manual testing. Neither will produce another squatter
unaided.

**What was built anyway, and is worth keeping.** `server-diagnostics.yml` walks each uvicorn's
parent chain to init and prints its cgroup, so systemd, a person over ssh, pm2 and cron are
told apart on sight rather than by inference. Cheap, and it means a recurrence names its own
source in seconds instead of costing another investigation. The Commerce-side guard closed a
route that was real regardless of whether anything had used it.

**Socket activation was considered and NOT done, deliberately.** uvicorn binds port 8000
itself, so the port is first-come-first-served and a stray process can win the race. A
`.socket` unit with `ListenStream=0.0.0.0:8000` would have systemd hold the port from boot
and hand the descriptor to uvicorn via `--fd`, making a second binder structurally impossible.
That is the real fix for the general problem. It is not taken because it changes how the
service starts, cannot be rehearsed anywhere but that server, and would leave the API down if
the handoff misbehaved. With both actual causes eliminated and a commit mismatch surfacing
within the hour, the exposure does not justify that risk. Recorded as the recommendation for
whoever has a maintenance window.

Original entry follows.

**The monitoring gap is closed, by both changes this entry asked for.** Change 2, the
`is-active` assertion, went in on 2026-08-07 and was hardened the same day after it passed
while the unit was crash-looping: `systemctl restart` returns before uvicorn reaches its
bind error, so the unit reads `active` for a second or two. It now waits 8 seconds, checks
twice 5 seconds apart, and requires `0.0.0.0:8000` to be bound, which is the condition that
actually matters and which `is-active` alone cannot see.

Change 1, the commit SHA, went in on 2026-08-10. This is the one that catches the specific
failure described below. `repo_root` proves the answering process started from the deploy
directory, and a process left over from an EARLIER deploy satisfies that exactly, which is
why the check passed throughout the incident. The commit does not: a stale process reports a
stale stamp, because the file is read once at startup rather than per request. The deploy now
fails on a mismatch, naming both commits.

**Still open, and it is the cause rather than the detection.** Nothing has yet identified
what starts the unmanaged uvicorn. The likely source is still the Next.js app's on-demand
start, which fires when `AI_SERVICE_URL` is localhost and port 8000 does not answer, exactly
as it would not during a restart. Until that is found this can recur; the difference is that
it now fails the deploy loudly instead of reporting green, and `scripts/_kill_squatter.sh`
clears it.

Original entry follows.

**The cause was the deploy itself.** `.github/workflows/ci-cd.yml` chose between systemd and a
fallback with:

```bash
if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files | grep -q '^coverland-forecast-api\.service'; then
```

under `set -euo pipefail`. `grep -q` exits at its first match and closes the pipe;
`systemctl list-unit-files` prints hundreds of units, is still writing, dies of SIGPIPE and
returns 141; `pipefail` makes 141 the status of the pipeline. So the condition was FALSE on
every deploy even though the unit was found. Reproduced in isolation: the same construct with
`seq 1 200000 | grep -q '^42$'` takes the else branch under `pipefail` and the if branch
without it.

The fallback then ran `pkill -f 'uvicorn api.main:app'`, killing the systemd-managed process,
and started `nohup .venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 &`.
That command line matches the observed squatter character for character, including the
relative venv path and the missing `--workers 1`. systemd then restarted under
`Restart=always`, could not bind, and crash-looped indefinitely.

**Fixed** by replacing the pipeline with `systemctl cat coverland-forecast-api.service`, which
takes a unit name directly, needs no pipe and returns non-zero when the unit is absent. The
fallback now binds `0.0.0.0` to match the unit rather than silently reverting the change that
made the API reachable. An `is-active` assertion was added after the restart, polling for 20
seconds and failing the build with the journal attached if the unit is not `active`.

**Still open: the monitoring gap that let this run for a day.**

On 2026-08-07 the systemd unit was found crash-looping with `[Errno 98] address already in
use`, having been in that state since 00:01:11 UTC. An unmanaged uvicorn process held port
8000, started five seconds after the deploy rsync, with a relative venv path and no
`--workers 1`, so not by systemd. Every CI run in between reported green.

**Why the existing check does not catch it.** The workflow polls `/health` and compares
`repo_root` against `DEPLOY_PATH`, which proves the answering process was started from the
deploy directory. A stale process started from that same directory satisfies it exactly. The
check distinguishes "some other service on this box" from "our directory", and does not
distinguish "the unit systemd just restarted" from "a process that has been there for hours".

**Two changes, both small.**

1. Return the deployed commit SHA from `/health`, written by the deploy into a file the app
   reads at startup, and have the workflow compare it to `github.sha`. That closes this and
   BACKLOG 20 with one mechanism: a push that never deployed and a deploy that never took the
   port both show up as a SHA mismatch.
2. Have the workflow assert `systemctl is-active coverland-forecast-api` after the restart.
   One line, and it would have caught this specific failure immediately, since the unit
   reports `activating (auto-restart)` rather than `active` while crash-looping.

**The cause is still open.** The likely source is the Next.js app's on-demand start, which
fires when `AI_SERVICE_URL` is localhost and port 8000 does not answer, exactly as it would
not during a restart. `DEPLOYMENT.md` already says to leave `FORECAST_SERVER_DIR` unset in
production for this reason. The search for it was inconclusive: the candidate paths do not
exist and the `sudo grep` ran without a TTY. Finding where that app is deployed and checking
its environment is the next step, and until it is done this recurs at every deploy.

---

## 23. A re-run replaced its own SKUs, not its own week

**Status: done, 2026-08-12.** `src/ml/serving/store.py`, covered by
`scripts/test_store_replace_run.py`.

**What was wrong.** `store.upsert` wrote with `ON CONFLICT (model_version, week_of,
unique_id, ds) DO UPDATE`. That replaces a row the incoming run also produces, and it has
no way to express that a SKU is gone. Removing a SKU from the forecast set is not an
update to any row; it is the absence of one, and an upsert cannot represent absence.

The function's own docstring recorded the intent as replacing "its own rows and nothing
else", which is exactly the behaviour and exactly the defect. It reads as correct until
the set of SKUs shrinks.

**What it would cost, and what it has not yet cost.** The smooth set goes from 467 SKUs to
338 when the promotion threshold moves to 3.0 units/week. Applied against a table already
holding the 467, the upsert would overwrite the 338 it produced and leave the other 129 at
the previous week's values, so the week would describe two segmentations at once. The
planning screens read the latest `week_of` and would serve all 467 with nothing indicating
that a quarter were no longer forecast.

That has not happened yet, and this entry said it had. Checked against the live tables on
2026-08-12: `shipcore.ml_forecast_history` holds two runs, 2026-08-10 and 2026-08-03, both
467 SKUs at 6,071 rows; `shipcore.ml_forward_forecasts` holds the same two weeks, 467 SKUs
at 6,071 rows and 468 at 24,336. Every stored run is the 467-SKU segmentation. The
threshold change has not reached the server, so the mixed week was a reconstruction from
local files rather than an observation.

The bug is real regardless: it is a property of the write, demonstrated by the test, and it
fires the first time the server runs with the new threshold. Recorded this way because
"this already happened" and "this happens on the next deploy" call for different urgency,
and only the second one is supported.

Silent is the important part either way. A forecast that is merely wrong is still a
forecast the model stands behind; these would be forecasts the model had stopped making.

**One thing worth a look before the next deploy.** The 2026-08-03 forward week holds 52
target weeks per SKU where 2026-08-10 holds 13, so a `--horizon 52` run is stored there.
Aggregates cannot tell a single 52-week run apart from a 13-week run written over one,
because the upsert replaces in place. `SELECT run_at, count(*) ... GROUP BY 1` on that week
distinguishes them. Either way the run-replacing write makes the next run for a week
correct by construction.

**The fix.** `upsert` deletes the `(model_version, week_of)` pair before inserting, both
statements in one transaction. After a write the week holds exactly what that run
produced. This is what `src.db.write_forward_forecasts` on the legacy track already did,
so the two tracks now agree.

Scoped to one `model_version`, so v11 and a candidate version can be stored against the
same week without either clearing the other.

**What it assumes.** One run writes its week in a single call. Both callers do. A future
caller that wrote in pieces would have each piece delete the last, which is why the
constraint is written into the docstring and why `replace_run=False` exists.

**Why there is a test rather than a comment.** The bug has no symptom: nothing throws,
nothing looks wrong on screen, and the row count goes up rather than down. It is
reproduced by any change to the segmentation rules, and those are still being tuned. The
test drives the real `upsert` against SQLite and was confirmed to fail against the
previous behaviour before being kept.

**Cleaning up the rows already stored.** No purge is needed. Re-running the forecast now
clears the week first, so `scripts/ml_forward_forecast.py --snapshot live` repairs the
table by itself, with no window in which the planning screens have nothing to serve.
`scripts/ml_purge_history_run.py --also-forward` remains for the case where a bad run must
be removed without a replacement, and its own bug (it read the parquet without the
`forecast_date` to `week_of` shim, and crashed) was fixed at the same time.

---

## 24. Take the personal copy of this repository

**Status: open, and deliberately last.** Do this after the documentation pass and before
handover.

**Why it is last.** The documentation is the part worth keeping, and it is being rewritten
now. A copy taken before it is finished carries the stale figures that BACKLOG's own
header calls the largest piece of documentation debt.

**Why it must come before any further deletion.** BACKLOG 6 retires the old Demand
Forecast page, and with it the statsforecast implementation: the legacy half of
`api/main.py`, `src/models.py`, `src/selector.py`, `src/backtest.py`, `src/baselines.py`.
That is the model-selection and backtesting work, and it is a substantial part of what the
project was. Once 6 closes it is gone from the tree. Take the copy first.

**The repository is clean and can be copied as it stands.** `.env` and
`.claude/settings.local.json` exist locally but have never been tracked, and
`git log --all` over both paths returns nothing. No credential has ever been committed
here. This is not true of the Commerce app, where `.claude/settings.local.json` has been
tracked since 2026-04-28; that repository needs separate thought if it is ever copied.

**It runs without a database.** `scripts/seed_dev_data.py` fills `data/processed/` from
files already tracked in git, which is why a clone works with no credentials at all:

```
pip install -r requirements.txt
python scripts/seed_dev_data.py
python -m uvicorn api.main:app --port 8000     # /health then reports ready: true
```

The 4.5 MB that makes this possible is `data/dev_seed/`, the four snapshots under
`data/snapshots/`, and the accuracy and backtest reports in `outputs/reports/`.

**Worth including.** The design doc's decision log and version log, including the
rejections. Six perturbations that each cost the long segment 0.005 to 0.012 is a better
account of the work than the changes that landed.

---

## 25. Extract the legacy statsforecast endpoints from `api/main.py`

**Status: DONE 2026-08-13, on the second attempt.** What landed, and how it was
verified, is at the end of this section. The original entry follows first, because the
reason the first attempt failed is the reason the second one is trustworthy.

`api/main.py` holds both tracks: the LightGBM serving endpoints and the older statsforecast
ones that the current Demand Forecast page uses. The legacy half is roughly 1,700 lines and
depends on `src/models.py`, `src/selector.py`, `src/backtest.py` and `src/baselines.py`.

Moving it into `api/legacy.py` makes BACKLOG 6 a deletion of one file and one import rather
than surgery on a 4,000-line module.

**Why the first attempt was reverted, so the next one does not repeat it.** Two problems,
in order:

1. The extraction truncated multi-line parenthesised imports, so the moved module did not
   import what it needed.
2. More importantly, the result could not be verified. FastAPI registers an included router
   as a single opaque object rather than copying its routes into `app.routes`, so the
   obvious check, comparing route lists before and after, compares nothing. The revert was
   verified instead: 35 routes, identical.

**What changes for the next attempt.** BACKLOG 22 pinned FastAPI at the server's 0.141.1,
so the verification difficulty can at least be reproduced against a known version. Compare
resolved paths by walking the router tree, or drive both versions with real requests, rather
than comparing `app.routes` lengths.

**Ordering.** This is a move, not a deletion, so it can happen before BACKLOG 24. Deleting
the code cannot.

### What landed, 2026-08-13

`api/main.py` went from 3,184 lines to 1,346. The sixteen statsforecast endpoints are in
`api/legacy.py`; `src/models.py`, `selector.py`, `backtest.py` and `baselines.py` are in
`src/legacy/`, moved with `git mv` so the history follows them. `JobLogger` is the only
helper both tracks needed, so it went to `api/common.py` rather than being duplicated or
imported across the two, which would have made them circular.

`_parse_product_types`, `_data_version`, `_cached_response` and `_VALID_LEVELS` moved
into `api/legacy.py` rather than into the shared module. They looked shared and were not:
`_data_version` reads `fc_forward_forecasts`, and nothing on the ML side calls any of
them. Putting them in the legacy module means retiring that track removes them too.

**Both failure modes from the first attempt were addressed rather than avoided.**

1. The imports were not guessed. `symtable` scope analysis on the extracted block listed
   every name it referenced but did not define. All 52 resolved to the original
   header, so the new import block was derived from the code rather than eyeballed.
2. The verification exists now, as `scripts/check_route_parity.py`, and it took three
   attempts to make it capable of failing. That history is worth keeping:
   - Walking `app.routes` missed all sixteen legacy routes. FastAPI 0.141.1 stores an
     included router as a `_IncludedRouter` with no `.path` and no `.routes`; its
     contents are reachable only via `.original_router`. The walker now raises on a route
     object it does not understand rather than skipping it.
   - Sending real requests and treating 404 as "not routed" passed while testing nothing.
     The token middleware answers 401 before routing, so a nonexistent path never
     returned 404.
   - Adding the token then produced false failures on `/planning/sku/{sku_id}`, whose
     handler legitimately raises 404 for an unknown SKU. A status code cannot distinguish
     "no route" from "route says not found".

   It now calls `BaseRoute.matches()`, which is the app's own resolution logic, with no
   middleware, handler or database in the way. A negative control runs on every
   invocation: if a deliberately nonexistent path matches, the script reports that it is
   blind instead of passing.

**Result:** 34 routes, equal to the previous 35 minus `/chat`, no shadowing, and the
probe confirms each one resolves. Verified with no database and no network.

**`/chat` was deleted, not moved.** BACKLOG 6 already recorded that the assistant's tool
calls had been failing silently for the whole life of the deployment, because
`src/chat.py` addressed port 8001 where nothing listened. It is the one piece of this
track that was genuinely unused, so it went, along with `src/chat.py` (477 lines). The
Next.js side of it is a separate change in the Commerce repository.

**What this does not do.** It does not retire anything. The legacy track still serves SKU
Planning and still performs the weekly ingest that the LightGBM run depends on;
`src/legacy/__init__.py` is now the authority on that and states what must be true before
deletion is possible. The extraction makes BACKLOG 6 a smaller job, which was its purpose.

---

## 26. Run the final test, once

**Status: open. Pre-registered 2026-08-12 in `ML_FORECAST_DESIGN.md` section 4.34.**

The final test window is quarantined by `ML_FINAL_TEST_CUTOFF` and has never been evaluated
against during development, which is the only thing that makes it a test rather than another
development window.

    .venv/bin/python scripts/ml_41_final_test.py

The runner writes `outputs/reports/final_test.json` and refuses to overwrite it, so a second
run is a deliberate act with a visible trace. It also refuses to run against a dirty tree,
so commit first.

**Read section 4.34 before running it, not after.** It states what result would count as a
pass and what would count as a failure, written before the numbers were known. The value of
the whole exercise is that those criteria were fixed in advance; reading them afterwards and
deciding they were roughly what you meant is how a pre-registration becomes decoration.

**Whatever it returns, it gets recorded.** A result worse than the development windows is a
real finding about generalisation, not a reason to look for a better window.

---

## 27. Rewrite the three stale documents

**Status: open.**

| document | state |
|---|---|
| `PROJECT_WRITEUP.md` | carries a SUPERSEDED banner. July figures, and a five-of-six claim that is four-of-six now |
| `HANDOVER.md` | carries a SUPERSEDED banner. Rewrite last, at actual handover, so it describes the state being handed over |
| `LEARNING_NOTES.md` | no banner, and largely fine. It is conceptual and does not repeat the numbers that moved |

`LEARNING_NOTES.md` is listed to record that it was checked rather than assumed: it was
described as stale earlier, and on inspection it contains none of the figures the 2026-08-11
and 2026-08-12 work changed. It needs a pass for the promotion threshold and the onset fix
as concepts, not a figure sweep.

The design doc and `CODEBASE_GUIDE.md` are current, having been updated alongside the work.

---

## 28. Separate the ingest from the statsforecast run

**Status: DONE 2026-08-13, and it needed no new code.** What was actually wrong, and the
one-line fix, are recorded after the original entry. The analysis below was right about the
problem and wrong about the solution, which is worth keeping rather than overwriting.

**Original status: open, and it became necessary rather than optional on 2026-08-13**, when the
decision was taken to delete SKU Planning's forecast tab as well as the Demand Forecast
page. Until then the statsforecast track had live consumers and the entanglement below was
merely untidy.

**The problem.** `scripts/run_forward_forecast.py` does two unrelated jobs in one script:

```
Step 0a  sync velocity snapshot          }
Step 0   ingest from DB, clean           }  shared: the ML track needs these
Step 1b  profile, write sku_profiles.csv }
------------------------------------------------------------------
Step 2   backtest (cross-validation)     }
Step 3   select model per SKU            }  statsforecast only: no consumer
Step 4   refit, forecast forward         }  once both pages are deleted
Step 5   write shipcore.fc_* tables      }
```

The first three steps write `sales_clean.parquet` and `sku_profiles.csv`, which are the
LightGBM run's only inputs. `scripts/ml_forward_forecast.py` has no ingest of its own and
reads the files this script produces. That is why `run_forecast_cron.sh` runs the
statsforecast pipeline first, and why deleting it silently freezes the ML forecast rather
than breaking it.

**The change.** Extract steps 0a to 1b into `scripts/run_ingest.py`. The cron then reads:

```
scripts/run_ingest.py            # shared, owns sales_clean.parquet and sku_profiles.csv
scripts/ml_forward_forecast.py --snapshot live
```

`run_forward_forecast.py` keeps only the statsforecast half and moves to the archive with
the rest of that track, at which point no in-use code path touches `src/legacy/`.

**Why it is worth doing rather than leaving.** The stated goal is that the statsforecast
work is retained as a record and is not part of any live path. That is not true while the
weekly production run depends on a script belonging to it. It also removes the trap: today
someone who deletes the legacy track gets a forecast that keeps serving and stops moving,
with nothing erroring.

**Verification.** The extraction is behaviour-preserving if `sales_clean.parquet` and
`sku_profiles.csv` are byte-identical before and after for the same input week. Compare
checksums across the split rather than assuming, since the profiling has already been the
source of one silent population change (Section 4.32 of the design doc).

### What was actually done, 2026-08-13

**No script was written, because the right one already existed.**
`scripts/ml_prepare_data.py` has performed sync, ingest, clean, profile and then the
LightGBM forecast since it was written for the Action List's Run Forecast button. It
touches no statsforecast code at all. The cron was calling two other scripts and had simply
never been pointed at it.

So the fix was one line in `run_forecast_cron.sh`: replace `run_forward_forecast.py` plus
`ml_forward_forecast.py --snapshot live` with `ml_prepare_data.py --force`.

**A new `scripts/run_ingest.py` was written first and then deleted**, before it was
committed. It duplicated the first three steps of `ml_prepare_data.py` and, more
importantly, wrote straight into `data/processed/` rather than staging. That would have
reintroduced exactly the hazard BACKLOG 15 fixed: a run interrupted between the sales file
and the profile leaves segmentation describing one week and sales describing another, with
nothing on any screen saying so. Recorded because the mistake is an easy one, and the
lesson generalises: check whether the thing already exists before extracting it.

**The weekly run is now strictly safer than before**, which was not the goal but is the
larger effect. `ml_prepare_data.py` stages every artifact in a sibling directory and
commits with `os.replace` only after the forecast succeeds, so a crash or a dropped SSH
session leaves last week's files intact and being served. The two-script version had no
such protection, so until now the weekly run was the one path that did not benefit from the
BACKLOG 15 work.

**`scripts/run_forward_forecast.py` moved to `scripts/legacy/`** with a README explaining
why it is kept and why running it casually on the production machine is not harmless. The
`shipcore.fc_*` tables are no longer written by anything.

**Still to verify on the server**, since it cannot be checked from a machine without
database access: that the first cron run after this change produces a `sales_clean.parquet`
and `sku_profiles.csv` consistent with the previous week's. The staging behaviour means a
failure is safe, but a silent difference in the profile counts would not be.

---

## 29. Codebase cleanup: dead scripts and stale files

**Status: open, lowest priority, explicitly last.** Do this only if there is time after
everything else. Nothing depends on it and it carries a real risk of deleting something
whose only caller is a document.

`scripts/` holds roughly 96 entries, most of them one-off experiments from the model
development arc: `_debug_*`, `_sweep_*`, `experiment_*`, `plot_*`, and the numbered
`ml_NN_*` series. Fourteen dead files were already removed on 2026-08-12, so this is the
long tail rather than an untouched mess.

**What makes this less trivial than it looks.** The numbered `ml_NN_*` scripts are cited by
number throughout `ML_FORECAST_DESIGN.md` as the evidence for individual decisions, and the
rebaseline logs under `docs/rebaseline_2026-08-03-v2/` are named after them. A script that
looks abandoned may be the only reproducible route to a figure the design document asserts.
Deleting it does not break code; it breaks the audit trail, which is worse and quieter.

**Suggested approach, if it happens at all.**

1. Anything referenced by name in `docs/` stays, however scruffy.
2. Anything importable from `src/` stays.
3. For the remainder, check `git log` for the last touch and whether its output is committed
   under `outputs/reports/` or `docs/rebaseline_*`. Keep the ones whose output is cited.
4. Move rather than delete, into `scripts/archive/`, so the recovery is a `git mv` and not a
   revert.

**One known orphan, left deliberately.** `SkuForecastsService.getForecastBounds()` in the
Commerce repository lost its only caller when `/api/forecast/bounds` was deleted on
2026-08-13. It still has a passing test, so it is dead code that looks maintained, which is
the worst kind. Removing it means removing the service method, the repository method beneath
it and two tests, which was more than that day's change justified. It is a five-minute job
for whoever picks this item up.

Other candidates noted while working: `scratch_introspect_fc_products.js` and
`scratch_list_objects.js` in the Commerce repository root, `Archive.zip` in the same place,
and `docs/CUTOVER_TASK.md`, `DEPLOY_TASK.md`, `INVENTORY_EXPORT_TASK.md` and
`V1_AND_DASHBOARD_WIRING_TASK.md`, which are completed task briefs rather than reference
documents and could move to `docs/tasks_completed/`.
