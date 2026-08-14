# Future work

The full list of identified improvements, in plain language, grouped by what each needs
before it can start. Written for whoever continues the project.

The project summary (`Machine_Learning_Demand_Forecast_Proposal.md`) carries a short version
of this for a non-technical reader. This is the long version. Where an item has supporting
measurement, the source is named: `ML_FORECAST_DESIGN.md` for design decisions and the
version log, `BACKLOG.md` for work on the screens, `HANDOVER.md` for the findings.

---

## 1. Better input data

The forecast learns entirely from the sales record, so anything that makes that record
misleading limits every method at once, the legacy spreadsheet included. This group is the
largest opportunity in the list and none of it is model work.

### Stockouts

When a SKU is out of stock, recorded sales fall toward zero. The record reads that as demand
falling, when demand was present and could not be met. Any method trained on it learns to
forecast low for exactly the products that most need reordering.

Two separate fixes, both worth doing:

1. **Correct the history**, so a week the product was unavailable stops counting as a week of
   weak demand. This is a correction to the training target and it improves every method,
   the statistical prototype included.
2. **Give the model the stock position directly**, so it can distinguish a real decline from
   an empty shelf. Today it sees units sold and nothing else, so the two are identical to it.

Both are blocked on the same prerequisite: stockout dates per SKU are not recorded in usable
form, and assembling that history is expected to take considerable time. See
`ML_FORECAST_DESIGN.md` Section 5.3.

### Preorder timing

A preorder is booked when the order is placed, not when it ships, so demand lands in the
wrong week: an artificial spike at order time and a gap at fulfilment. It is most damaging on
newly launched SKUs, whose launch preorders can dominate a short history, and it corrupts
every derived quantity at once, since the levels, the ramp, the elevation input and the
seasonal round-trip are all built from the weekly series.

Partly actionable now. Preorders are flagged as `order_type` in
`fc_velocity_link_snapshot_forecast` and `src/ingest.py` does not filter on it, so two
treatments are testable immediately: exclude preorder rows from training, or down-weight
them. Attributing demand to the intended fulfilment week still needs a source recording that
date. See `ML_FORECAST_DESIGN.md` Sections 2.1 and 5.4 item 6.

### Price and promotions

The largest missing signal in the list. The forecast does not know whether a product was
promoted or what it was priced at. It sees units sold and infers everything else, so a
discount that moved 300 units is indistinguishable from ordinary demand for 300 units, and
the model carries that level forward as though it were the new normal.

This also explains part of the seasonal difficulty. The change in holiday behaviour after
December 2024 is a change in promotional activity, and it is currently absorbed into the
seasonal factors as though it were a property of the calendar. Every candidate in Section 2
below is derived from the SKU's own demand history; this would be the first genuinely
external input, and it addresses a cause rather than a symptom.

What is needed is a record of when promotions ran and at what price, delivered to the
forecasting system the way order history already is.

---

## 2. Model improvements, none of them blocked

All are testable on data already held. Each should be evaluated the way every accepted change
was: one hypothesis at a time, across the three development windows, against criteria
recorded before the run, and judged by the Section 1.5 decision rule. Candidates and their
hypotheses are in `ML_FORECAST_DESIGN.md` Section 5.2; the open questions are Section 5.5.

| Candidate | What it would tell the model | Notes |
|---|---|---|
| Volatility | How erratic the SKU is, so the forecast stays near the recent average for a jumpy SKU and can act on a smaller signal for a steady one | Section 5.2 |
| Demand level | How large the SKU is, so corrections shrink for small, noisy SKUs | Section 5.2 |
| Zero-recency | How recently the SKU had a zero week, since recent zeros signal dormancy or a supply gap | Section 5.2 |
| Product family | Which family the SKU belongs to, parsed from the SKU code, so patterns transfer within a family rather than across the whole catalogue | Section 5.2 |
| Recent-level ratios | Additional lags beyond the two already used | Section 5.2 |
| Bounded SKU age | Maturity, but encoded so it cannot extrapolate past the training range | Raw age was rejected, Section 4.28. Note the short/long split already carries much of this |

**Rejected, and not to be retried without new evidence:** learned seasonality from
calendar features (Section 4.9), the segment indicator (4.23), per-segment weighting (4.24),
hyperparameter tuning (4.26), raw SKU age (4.28), channel mix (v17, Section 6).

### Open questions worth settling

- **Seasonal adjustment for newer SKUs.** They receive none today, so no holiday uplift. The
  Q4 evidence weakly suggested one would help but rested on 14 SKUs. Revisit before Q4 2026,
  once a third holiday season and a corrected demand target are available. Section 5.5 item 6.
- **The holiday regime change.** Late-November promotions began after December 2024, so the
  training data spans two seasonal regimes and the older one is not representative. Any
  seasonal estimate pooled across both mixes them, and the Oct-Dec evaluation window sits
  entirely inside the older regime. Consider down-weighting or excluding pre-2025 holiday
  weeks once a third December confirms the newer pattern. Section 5.5 item 7.
- **Per-lead accuracy as a standard metric.** Error grows with distance, and separating weeks
  1 to 2 from weeks 8 to 10 would tell planners how much to trust the far end of the horizon.
  The lead-as-feature design makes it cheap to produce. Section 5.5 item 2.
- **Where the newer/established boundary sits.** It is at a year because `elev_long` compares
  a 4-week level against a 52-week rolling level and cannot be computed below that. Whether a
  year is also the best boundary, and whether two groups is the right number, has not been
  tested. Note the related convention question below.
- **The boundary is a week later than its name suggests.** `active_weeks` is a span, so a
  window of 50 rows counts as 49 weeks, in production and in `asof_history_length` alike.
  Changing it moves SKUs across the boundary in both places and re-baselines every recorded
  figure, so it is a measurement decision rather than a correction. Raised 2026-08-13.
- **Low-volume accuracy requires changing the success measure first.** Pooled WAPE is
  demand-weighted. A change that brought every low and mid-volume band to baseline parity
  would move it by about 0.006, against a 0.01 acceptance threshold and a single-window noise
  floor of 0.011 to 0.014. The criterion cannot detect the improvement, so anyone taking this
  on has to change and pre-register the criterion before starting. `HANDOVER.md` finding 3.
- **Intermittent SKUs as training examples**, without being forecast targets. Currently ruled
  out by reasoning rather than measurement. Section 5.5 item 3.

### Prediction intervals: deliberately deprioritised

Recorded here so the reasoning is not lost and the decision is not reopened as an oversight.
v11 emits a point forecast only, and the Forecast Validation chart says so rather than
leaving a reader to wonder where the band went.

The decision rests on three things. Most SKUs lack the history to support a reliable
interval, which is what the conformal bands on the retired statistical track already showed
by under-covering. The recommended order quantity already carries safety stock, which sizes
uncertainty from the same quantity an interval would. And under-ordering is the more
expensive direction to be wrong in, so the quantity that should be ordered sits at the high
end of any interval rather than at its centre.

Worth adding eventually. It would move the recommended quantity very little.
`_MIGRATE_PI_SQL` in `src/db.py` remains the pattern if it is taken up.

---

## 3. The screens

Detail and the design constraints are in `BACKLOG.md` under the item numbers given.

- **Order timing.** The Action List gives the quantity needed and the date stock is projected
  to run out. It does not say when to place the order. Supplier lead times and the container
  schedule would turn "order 400 units" into "order 400 units by this date", which is the
  decision the planner is actually making.
- **Draft containers, finishing the job.** Draft units show on the row and are deliberately
  not subtracted from the recommendation, since a draft is not a commitment. Two checks
  against live data remain: that drafted totals agree with the Container Planning screens,
  and how many rows carry drafts at all. If most do, the sub-line should become a sortable
  column. BACKLOG 9.
- **Money.** Every figure on both screens is in units, so the list cannot be triaged by spend.
  Needs a decided cost basis, unit against landed, and a source for it. A data question before
  a screen one. BACKLOG 10.
- **Ordered against recommended.** Nothing records whether a recommendation was acted on, so
  the system cannot show whether it is followed or where planners routinely overrule it.
  Blocked on purchase order outcomes, which the application does not hold.

---

## 4. Process

1. **Score the statistical prototype through the ML harness**, so the smooth/long comparison
   uses the same evaluation code as everything else and the three-way comparison rests on one
   scoring path rather than two. The most valuable item in this section, and worth more than
   another feature. Section 5.4 item 1.
2. Reduce training-row redundancy if needed: anchor thinning, per-tree row subsampling, and a
   reassessment of `min_child_samples` in light of effective sample size. Section 5.4 item 2.
3. Once the design freezes, reproduce results with `mlforecast` as an independent pipeline
   check. Section 5.4 item 3.
4. Recompute the bucket label as-of each evaluation cutoff. It is the last input in the
   harness still using future information. `HANDOVER.md` finding 6.
