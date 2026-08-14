# Future improvements

Everything identified and not done, grouped by what blocks it. Written for whoever continues
the project.

This document stands on its own. It does not assume you have read the others, and it names
the file and section to go to when you pick something up.

**Two things to know before proposing anything.** Most obvious ideas are already below,
several with a recorded reason they were rejected. And the project's rule is one hypothesis
at a time, with pass criteria written down **before** the experiment runs, judged by the
decision rule in `OVERVIEW.md` Section 5. A change is adopted only if it improves accuracy
consistently in sign across all three development windows and by a three-window mean of at
least 0.01.

---

## 0. Fix these first

Not improvements. Things that are currently wrong. Two of the three found on 2026-08-14 were
fixed the same day and are recorded here as done, so nobody re-opens them; one remains.

### 0.1 STILL OPEN: the accuracy report behind Forecast Validation is stale

`outputs/reports/ml_accuracy.csv` and `ml_accuracy_by_sku.csv` are dated 2026-07-30 and
predate the profiling fix of 2026-08-11, the threshold alignment of 2026-08-12 and the V1
as-of fix of 2026-08-13. Both files are tracked, so the stale copies are what the server
deploys. Sections 01 and 05 of Forecast Validation and the reliability tiers on the Action
List are all computed from them, which means they describe a population that no longer exists
and a V1 column with a known systematic error.

**This is the only remaining defect that misleads a reader about accuracy**, and the figures
visibly differ: v11 smooth/short in Oct-Dec reads 0.1783 from the CSV against the correct
0.2473.

Run `scripts/ml_accuracy_report.py`, which retrains on the three development windows and by
construction never touches the quarantined window. Check the grid against `OVERVIEW.md`
Section 6, then **commit the regenerated files**. An hour, most of it waiting.
`SCREENS.md` Section 3.7.

While doing it, fix the docstring: it says to refresh when the model version changes, and it
also needs refreshing when the **population** changes, which is what any profiling or
threshold change does. That omission is the reason this went unnoticed.

### 0.2 DONE 2026-08-14: the final test is served and rendered

BACKLOG item 30, which had been closed on a statement rather than on inspection while the
code still hardcoded `"evaluated": False`. The page had been telling readers the final test
had not been run since 2026-08-13.

`/planning/validation` now serves `outputs/reports/final_test.json` through
`_final_test_payload()`, passing the scores and provenance through unchanged so the page and
the file cannot disagree. `types.ts` carries `FinalTest` as a discriminated union, and
`final-test-section.tsx` renders both halves of the result at the same weight: the model beats
the spreadsheet significantly on both segments, and it ties the structural baseline. Verified
with `tsc --noEmit` and `eslint`, both clean.

Calibration figures are deliberately not rendered, because the runner records pooled WAPE and
the bootstrap only. If they are wanted on screen, `ml_41_final_test.py` should record them and
the payload will pass them through; the panel says where they currently live rather than
restating them. That is the only follow-up this item leaves.

### 0.3 DONE 2026-08-14: `final_test.json` is tracked

It is on the `.gitignore` allowlist beside the four other report files. It was the one
artifact in the repository that could not be regenerated: the test is single-use and
`ml_41_final_test.py` refuses to overwrite, so re-running is either refused or spends a window
that no longer exists.

---

## 1. Better input data

The forecast learns entirely from the sales record, so anything that makes that record
misleading limits every method at once, the spreadsheet included. **This group is the largest
opportunity in the list and none of it is model work.**

### 1.1 Stockouts

When a SKU is out of stock, recorded sales fall toward zero. The record reads that as demand
falling, when demand was present and could not be met. Any method trained on it learns to
forecast low for exactly the products that most need reordering.

Two separate fixes, both worth doing:

1. **Correct the history**, so a week the product was unavailable stops counting as a week of
   weak demand. This is a correction to the training target and it improves every method.
2. **Give the model the stock position directly**, so it can tell a real decline from an
   empty shelf. Today it sees units sold and nothing else, so the two are identical to it.

**Blocked** on the same prerequisite: stockout dates per SKU are not recorded in usable form,
and assembling that history is expected to take considerable time. Only a current inventory
snapshot exists, not history. `ML_FORECAST_DESIGN.md` Section 5.3.

### 1.2 Preorder timing

A preorder is booked when the order is placed, not when it ships, so demand lands in the
wrong week: an artificial spike at order time and a gap at fulfilment. It is most damaging on
newly launched SKUs, whose launch preorders can dominate a short history, and it corrupts
every derived quantity at once, since the levels, the ramp, the elevation input and the
seasonal round-trip are all built from the weekly series.

**Partly actionable now.** Preorders are flagged as `order_type` in
`fc_velocity_link_snapshot_forecast` and `src/ingest.py` does not filter on it, so two
treatments are testable immediately: exclude preorder rows from training, or down-weight
them. Attributing demand to the intended fulfilment week still needs a source recording that
date. Sections 2.1 and 5.4 item 6.

### 1.3 Price and promotions

**The largest missing signal in the list.** The forecast does not know whether a product was
promoted or what it was priced at. It sees units sold and infers everything else, so a
discount that moved 300 units is indistinguishable from ordinary demand for 300 units, and
the model carries that level forward as though it were the new normal.

This also explains part of the seasonal difficulty. The change in holiday behaviour after
December 2024 is a change in promotional activity, and it is currently absorbed into the
seasonal factors as though it were a property of the calendar.

Every candidate in Section 2 below is derived from the SKU's own demand history. **This would
be the first genuinely external input**, and it addresses a cause rather than a symptom.

What is needed is a record of when promotions ran and at what price, delivered the way order
history already is.

---

## 2. Model improvements, none of them blocked

All testable on data already held. Candidates and their hypotheses are in
`ML_FORECAST_DESIGN.md` Section 5.2.

| Candidate | What it would tell the model |
|---|---|
| Volatility | How erratic the SKU is, so the forecast stays near the recent average for a jumpy SKU and can act on a smaller signal for a steady one |
| Demand level | How large the SKU is, so corrections shrink for small, noisy SKUs |
| Zero-recency | How recently the SKU had a zero week, since recent zeros signal dormancy or a supply gap |
| Product family | Which family the SKU belongs to, parsed from the SKU code, so patterns transfer within a family rather than across the whole catalogue |
| Recent-level ratios | Additional lags beyond the two already used |
| Bounded SKU age | Maturity, encoded so it cannot extrapolate past the training range. Raw age was rejected in 4.28; note the short/long split already carries much of this |
| Shrunk seasonal multipliers | Empirically re-estimated monthly factors, shrunk toward the hand-set values |

### Rejected, and not to be retried without new evidence

Learned seasonality from calendar features (4.9), the segment indicator (4.23), per-segment
weighting (4.24), hyperparameter tuning (4.26, and again as v18), raw SKU age (4.28),
acceleration (v13), FBA channel share (v17), seasonal blends (v15 and v16).

Six perturbations of the long model each cost it 0.005 to 0.012.

---

## 3. Open questions worth settling

**Why V1 wins the entire Q4 window.** A trailing velocity method being more robust in the
peak season is not explained by anything in the record. This is the most interesting
unanswered question in the project and it sits underneath the two cells the model loses.

**Seasonal adjustment for newer SKUs.** They receive none today, so no holiday uplift. The Q4
evidence weakly suggested one would help but rested on 14 SKUs. Revisit before Q4 2026, once
a third holiday season and a corrected demand target are available. Section 5.5 item 6.

**The holiday regime change.** Late-November promotions began after December 2024, so the
training data spans two seasonal regimes and the older one is not representative. Any
seasonal estimate pooled across both mixes them, and the Oct-Dec evaluation window sits
entirely inside the older regime. Consider down-weighting or excluding pre-2025 holiday weeks
once a third December confirms the newer pattern. Section 5.5 item 7.

**Per-lead accuracy as a standard metric.** Error grows with distance, and separating weeks 1
to 2 from weeks 8 to 10 would tell planners how much to trust the far end of the horizon. The
lead-as-feature design makes it cheap to produce. Section 5.5 item 2.

**Weeks 11 to 13 are scored by nothing.** The evaluated horizon is 10 weeks while production
serves 13, so every recorded figure is a lower bound on the error of the horizon actually
used for ordering.

**Where the newer/established boundary sits, and what it means.** Three things, and they
should be settled together:

1. Whether the boundary should be a **row count rather than a span**. `active_weeks` is a
   span, so a window of 50 rows counts as 49 weeks and a SKU needs 51 rows to be labelled
   medium. The boundary named "50 weeks" sits a week later than it reads, in production and
   in evaluation alike, consistently rather than inconsistently.
2. Whether **50 is the right place** at all, which has never been measured for the LightGBM
   track. `scripts/experiment_training_length_threshold.py` tested 26 to 78 weeks, but it
   asked when the statistical prototype beats V1, which is a different question.
3. Whether **two groups is the right number**.

**Why this is not a quick fix.** 50 exists because `elev_long` is a 4-week mean over a 52-week
rolling mean with `min_periods=52` and cannot be computed below a year. The boundary exists
because the input requires it. Changing either the convention or the threshold moves SKUs
across the boundary in `asof_history_length` as well as in the profiler, which changes which
model serves them, which changes the scored population, which **re-baselines every recorded
figure including the final test's**. Current exposure is 15 smooth SKUs at 49 to 51 weeks.
The reason to do it is that the boundary should mean what it says, not that it is costing
accuracy.

**Low-volume accuracy requires changing the success measure first.** Pooled WAPE is
demand-weighted. A change that brought every low and mid-volume band to baseline parity would
move it by about 0.006, against a 0.01 acceptance threshold and a single-window noise floor
of 0.011 to 0.014. **The criterion cannot detect the improvement.** Anyone taking this on has
to change and pre-register the criterion before starting.

**Intermittent SKUs as training examples**, without being forecast targets. Currently ruled
out by reasoning rather than measurement. Section 5.5 item 3.

---

## 4. Prediction intervals: deliberately deprioritised

Recorded so the reasoning is not lost and the decision is not reopened as an oversight. v11
emits a point forecast only, and the Forecast Validation chart says so rather than leaving a
reader to wonder where the band went.

The decision rests on three things. Most SKUs lack the history to support a reliable
interval, which is what the conformal bands on the retired statistical track already showed
by under-covering. The recommended order quantity already carries safety stock, which sizes
uncertainty from the same quantity an interval would. And under-ordering is the more
expensive direction to be wrong in, so the quantity that should be ordered sits at the high
end of any interval rather than at its centre.

Worth adding eventually. It would move the recommended quantity very little. `_MIGRATE_PI_SQL`
in `src/db.py` remains the pattern if it is taken up.

---

## 5. Process

1. **Score the statistical prototype through the ML harness.** The prototype is the accuracy
   bar the whole project is judged against, and it has never been measured with the same code
   as everything it is compared to; its figures come from its own stored evaluation output.
   **The most valuable item in this document, and worth more than another feature.** Section
   5.4 item 1.
2. **Recompute the bucket label as-of each evaluation cutoff.** It is the last input in the
   harness still using future information.
3. Reduce training-row redundancy if needed: anchor thinning, per-tree row subsampling, and a
   reassessment of `min_child_samples` in light of effective sample size. Section 5.4 item 2.
4. Once the design freezes, reproduce results with `mlforecast` as an independent pipeline
   check. Section 5.4 item 3.

---

## 6. The screens

**Order timing.** The Action List gives the quantity needed and the date stock is projected to
run out. It does not say **when to place the order**. Supplier lead times and the container
schedule would turn "order 400 units" into "order 400 units by this date", which is the
decision the planner is actually making.

**Money.** Every figure on both screens is in units, so the list cannot be triaged by spend.
Needs a decided cost basis, unit against landed, and a source for it. A data question before
a screen one.

**Ordered against recommended.** Nothing records whether a recommendation was acted on, so the
system cannot show whether it is followed or where planners routinely overrule it. Blocked on
purchase order outcomes, which the application does not hold.

**Lot sizes and minimum order quantities.** The formula has no lot-size, MOQ or container
rounding anywhere. If the business needs it, that is new work rather than a missing config
value.

**Draft containers.** Draft units show on the row and are deliberately not subtracted from
the recommendation, since a draft is not a commitment. One check remains: how many rows carry
drafts at all. If most do, the sub-line should become a sortable column.

**The small frontend defects** in `SCREENS.md` Section 4, particularly the `error_basis`
values that render untranslated in the Korean locale.

---

## 7. Codebase cleanup

**Lowest priority, and it carries a real risk.** `scripts/` holds roughly 97 entries, most of
them one-off experiments from the model development arc. Fourteen dead files were already
removed on 2026-08-12, so this is the long tail rather than an untouched mess.

**What makes it less trivial than it looks.** The numbered `ml_NN_*` scripts are cited by
number throughout `ML_FORECAST_DESIGN.md` as the evidence for individual decisions, and the
rebaseline logs under `docs/rebaseline_2026-08-03-v2/` are named after them. A script that
looks abandoned may be the only reproducible route to a figure the design document asserts.
**Deleting it does not break code; it breaks the audit trail, which is worse and quieter.**

If it happens at all:

1. Anything referenced by name in `docs/` stays, however scruffy.
2. Anything importable from `src/` stays.
3. For the rest, check `git log` and whether its output is committed under `outputs/reports/`
   or `docs/rebaseline_*`. Keep the ones whose output is cited.
4. **Move rather than delete**, into `scripts/archive/`, so recovery is a `git mv` and not a
   revert.

Known candidates: `SkuForecastsService.getForecastBounds()` in the Commerce repository, which
lost its only caller and still has a passing test, so it is dead code that looks maintained;
`scratch_introspect_fc_products.js`, `scratch_list_objects.js` and `Archive.zip` in the
Commerce root; and the four completed task briefs in `docs/`, which could move to
`docs/tasks_completed/`.
