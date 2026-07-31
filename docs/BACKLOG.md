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

## 5. Forecast Validation outlier lists rank by an unweighted error

**Status:** identified, needs one decision.

**The problem.** The page's headline is pooled WAPE, which is demand-weighted, and the page says
so: errors are summed across SKUs before dividing, so heavier SKUs count more. The per-SKU
outlier lists directly beneath it then rank by *unweighted* per-SKU WAPE delta. The section under
the portfolio figure measures the opposite way from the figure.

**What that costs.** The 30 rows shown carry 1.85% of scored demand. Median volume on the "model
does worse" list is 33 units. The extremes are structural rather than real: maximum absolute delta
is 4.9 in the 10-to-50-unit band against 0.5 in the 200-plus band, because a small denominator lets
WAPE swing freely, so taking the top 15 by delta mechanically selects the smallest SKUs. The
section is headed "the list to read first: these are where trusting the model costs more than
trusting the sheet", and a planner following that instruction studies a 33-unit SKU.

**What is buried by it.** Restricting to 200 units and above surfaces a coherent pattern: 8 SKUs of
`CC-CN-03` and 4 of `CC-CP-03` in Dec-Feb, 7,082 units between them, where V1 runs 0.47 to 0.63 and
the model 0.00 to 0.16. That is the actual story of why the model wins, and the current list hides
it behind twenty-unit noise.

**The decision.** A stated minimum volume, adjustable on the page rather than applied silently.
The threshold is a judgement: 200 units gives 103 of 519 scored rows and a usable list, but the
number should be chosen rather than inherited from this note.

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

**Status:** open, needs a decision rather than work.

`Commerce_Integration/package.json` declares `engines.node` as `>=20.9 <24`. The development
machine runs v24.2.0, which npm reports as `EBADENGINE` on every install. Either the range is stale
and should be widened deliberately after testing, or the machine should move to Node 22 LTS, which
is inside it. Widening the range to match whatever happens to be installed is the same move as
relaxing a version pin to make an install succeed. Worth checking what the deployment server runs
while this is being decided: a major-version difference between there and a laptop is the usual
shape of "works locally, fails in production".
