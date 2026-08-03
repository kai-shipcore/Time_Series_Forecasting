# Backlog

Deferred work that has been decided or identified but not yet built. Each item records
what the change is, why it matters, and what blocks it. Completed work goes in
`WORKLOG.md`; this file is only for things still ahead of us.

---

## 1. Stockout-aware demotion in `src/profile.py`

**Status:** decided, blocked on data.

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
   `dashboard/data/inventory_snapshot.csv`, written by `scripts/export_inventory_snapshot.py`
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

## 2. `train_start` does two jobs, and promoted SKUs can never be backtested

**Status:** identified, not yet decided.

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

**Status:** wanted, needs its own basis.

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

**Status:** identified, lower priority, larger than a screen.

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

**Status:** identified, deferred. Nobody is using the export yet, which is the only reason
this is not urgent.

**The problem.** `exportCsv` in `action-list-content.tsx` builds its header from
`Object.keys(view[0])`, so the file contains every field on the row in whatever order the API
returned them: about forty columns including `forecast_over_recent`, `gap_closable_by_order`,
`n_windows`, `error_basis`, `demand_state` and `supply_gap_days`. The table above it shows ten,
chosen and ordered so a row reads as a sentence. Someone who filters the list and exports it
gets a different artefact from the one they were reading, with internal names as headings.

**Why it matters when it matters.** The export is the last step of the weekly cycle in
`dashboard/PLAN.md` §3.4: the purchaser works the list, then exports what they decided. A file
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

**Status:** BUILT 2026-07-31, pending verification against the live database. The assistant's
sandbox cannot reach Postgres, so the query below has been written and the display exercised
against simulated draft data, but no one has yet seen it run on real containers. What to check
is recorded at the end of this item.

**How this was reached, because the first answer was wrong.** This started as "let the purchaser
mark a SKU actioned", taken from `dashboard/PLAN.md` §3.4, where the walkthrough has the
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

**Status:** identified, deferred by decision (2026-07-31). Recorded so it is not rediscovered.

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

**Status:** identified 2026-07-31, both small, neither urgent.

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
