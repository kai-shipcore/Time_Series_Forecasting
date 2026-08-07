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

---

## 13. UI changes on the planning screens

**Status:** partly done 2026-08-05. Five items, listed smallest first. They are grouped because
they are all presentation changes on screens that already have their numbers right.

Progress on 2026-08-05: 13.4 done, 13.3 declined, 13.2 partly done. 13.1 and 13.5 unchanged. Each
is marked below rather than deleted, so the reasoning survives the decision.

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

**13.5 Explain the priorities on the screen.** The Action List shows a priority label per row and
a set of summary counts, and nothing on the page says what earns each label. The Streamlit
prototype carried a plain-language explanation of every priority and the port dropped it, which is
the same gap as the planning controls fixed on 2026-08-04. The explanation belongs where the
labels appear rather than in a legend elsewhere.

Sequenced after item 14, which changes what the labels are. Writing the explanation first means
writing it twice, and the current set is the harder one to explain because one of its members is
not the same kind of thing as the others.

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

**Status:** identified 2026-08-05, from a live incident. Small, and it blocks giving the Run
Forecast panel a working Stop button.

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

**Status: observed 2026-08-06, cause unknown.**

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

**Status: observed 2026-08-07, cause of the squatter not yet identified.**

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
