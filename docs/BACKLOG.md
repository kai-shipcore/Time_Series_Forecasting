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
