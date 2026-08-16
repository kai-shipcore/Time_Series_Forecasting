# Future improvements

Everything identified and not done, grouped by what blocks it.

Note: the project rule is one hypothesis at a time, with pass criteria written down before the experiment runs, judged by the adoption rule in `OVERVIEW.md` §5.

## 1. Better input data

The forecast learns entirely from the sales record. Anything that makes that record misleading limits every method at once, the spreadsheet included. None of this group is model work.

### 1.1 Stockouts

Out-of-stock weeks record sales near zero, which the training target reads as weak demand. Any method trained on it forecasts low for the products that most need reordering.

Two fixes:

1. **Correct the history**, so a week the product was unavailable stops counting as weak demand. Improves every method.
2. **Give the model the stock position directly**, to separate a real decline from an empty shelf.

**Blocked.** Stockout dates per SKU are not recorded in usable form and only a current inventory snapshot exists; assembling that history is expected to take considerable time. Design doc §5.3.

### 1.2 Preorder timing

A preorder is booked when the order is placed, not when it ships, so demand lands in the wrong week: a spike at order time, a gap at fulfilment. The levels, the ramp, the elevation input and the seasonal round-trip are built from the weekly series, so all are corrupted. Launch preorders can dominate a short history, so newly launched SKUs are worst affected.

**Partly actionable now.** Preorders are flagged as `order_type` in `fc_velocity_link_snapshot_forecast` and `src/ingest.py` does not filter on it. Two treatments are testable immediately:

- Exclude preorder rows from training
- Down-weight them

Attributing demand to the intended fulfilment week needs a source recording that date. Design doc §2.1 and §5.4 item 6.

### 1.3 Price and promotions

The forecast does not know whether a product was promoted or what it was priced at. A discount that moved 300 units is indistinguishable from ordinary demand for 300 units, and the model carries that level forward as the new normal.

The change in holiday behaviour after December 2024 is a change in promotional activity, currently absorbed into the seasonal factors as though it were a property of the calendar.

Every candidate in §2 derives from the SKU's own demand history; this would be the first external input. It requires a record of when promotions ran and at what price, delivered the way order history already is.

## 2. Model improvements, none blocked

All testable on data already held. Candidates and hypotheses are in the design doc §5.2.

| Candidate | What it would tell the model |
|---|---|
| Volatility | How erratic the SKU is, so the forecast stays near the recent average for a jumpy SKU and can act on a smaller signal for a steady one |
| Demand level | How large the SKU is, so corrections shrink for small, noisy SKUs |
| Zero-recency | How recently the SKU had a zero week, since recent zeros signal dormancy or a supply gap |
| Product family | Which family the SKU belongs to, parsed from the SKU code, so patterns transfer within a family |
| Recent-level ratios | Additional lags beyond the two already used |
| Bounded SKU age | Maturity, encoded so it cannot extrapolate past the training range. Raw age was rejected in §4.28, and the short/long split already carries much of this |
| Shrunk seasonal multipliers | Empirically re-estimated monthly factors, shrunk toward the hand-set values |

**Rejected, not to be retried without new evidence:** learned seasonality from calendar features (§4.9), the segment indicator (§4.23), per-segment weighting (§4.24), hyperparameter tuning (§4.26 and again as v18), raw SKU age (§4.28), acceleration (v13), FBA channel share (v17), seasonal blends (v15 and v16). Six perturbations of the long model each cost it 0.005 to 0.012.

## 3. Open questions

| Question | Detail |
|---|---|
| Why V1 wins the entire Q4 window | A trailing velocity method being more robust in the peak season is not explained by anything in the record. The largest open question in the project, and it sits underneath the two cells the model loses |
| Seasonal adjustment for newer SKUs | They receive none today, so no holiday uplift. The Q4 evidence weakly suggested one would help but rested on 14 SKUs. Revisit before Q4 2026, once a third holiday season and a corrected demand target are available. Design doc §5.5 item 6 |
| The holiday regime change | Late-November promotions began after December 2024, so training data spans two seasonal regimes and the older one is not representative. The Oct-Dec evaluation window sits entirely inside the older regime. Consider down-weighting or excluding pre-2025 holiday weeks once a third December confirms the newer pattern. Design doc §5.5 item 7 |
| Per-lead accuracy as a standard metric | Error grows with distance. Separating weeks 1 to 2 from weeks 8 to 10 would tell planners how much to trust the far end of the horizon. The lead-as-feature design makes it cheap to produce. Design doc §5.5 item 2 |
| Weeks 11 to 13 are scored by nothing | The evaluated horizon is 10 weeks while production serves 13, so every recorded figure is a lower bound on the error of the horizon actually used for ordering |
| Intermittent SKUs as training examples | Without being forecast targets. Ruled out by reasoning, never measured. Design doc §5.5 item 3 |

### The short/long boundary

Three questions to settle together.

| Question | Detail |
|---|---|
| Should the boundary be a row count instead of a span? | `active_weeks` is a span, so a window of 50 rows counts as 49 weeks and a SKU needs 51 rows to be labelled medium. The boundary named "50 weeks" sits a week later than it reads, in production and evaluation alike, consistently |
| Is 50 the right place at all? | Never measured for the LightGBM track. `scripts/experiment_training_length_threshold.py` tested 26 to 78 weeks, but it asked when the statistical prototype beats V1, a different question |
| Is two groups the right number? | Open |

**Rationale.** 50 exists because `elev_long` is a 4-week mean over a 52-week rolling mean with `min_periods=52` and cannot be computed below a year. Changing the convention or the threshold moves SKUs across the boundary in `asof_history_length` as well as in the profiler, which changes the scored population and re-baselines every recorded figure including the final test's. Current exposure is 15 smooth SKUs at 49 to 51 weeks.

The boundary is not costing accuracy.

### Low-volume accuracy

Requires changing the success measure first. Pooled WAPE is demand-weighted: bringing every low and mid-volume band to baseline parity would move it by about 0.006, against a 0.01 acceptance threshold and a single-window noise floor of 0.011 to 0.014. The criterion cannot detect the improvement; change and pre-register it before the work starts.

## 4. Prediction intervals, deliberately deprioritised

v11 emits a point forecast only, and the Forecast Validation chart states this. The decision rests on three things:

1. Most SKUs lack the history to support a reliable interval; the conformal bands on the retired statistical track under-covered.
2. The recommended order quantity already carries safety stock, sizing uncertainty from the same quantity an interval would.
3. Under-ordering is the more expensive direction to be wrong in, so the quantity to order sits at the high end of any interval, not at its centre.

Adding intervals would move the recommended quantity little. `_MIGRATE_PI_SQL` in `src/db.py` remains the pattern.

## 5. Process

| Item | Detail |
|---|---|
| Score the statistical prototype through the ML harness | The prototype is the accuracy bar the whole project is judged against, and it has never been measured with the same code as everything it is compared to. Design doc §5.4 item 1 |
| Recompute the bucket label as of each evaluation cutoff | The last input in the harness still using future information |
| Reduce training-row redundancy if needed | Anchor thinning, per-tree row subsampling, and a reassessment of `min_child_samples` in light of effective sample size. Design doc §5.4 item 2 |
| Reproduce results with `mlforecast` once the design freezes | An independent pipeline check. Design doc §5.4 item 3 |

## 6. The screens

| Item | Detail |
|---|---|
| Order timing | The Action List gives the quantity needed and the date stock is projected to run out, but not when to place the order. Supplier lead times and the container schedule would turn "order 400 units" into "order 400 units by this date", which is the decision the planner is actually making |
| Money | Every figure is in units, so the list cannot be triaged by spend. Needs a decided cost basis, unit against landed, and a source for it. A data question before a screen one |
| Ordered against recommended | Nothing records whether a recommendation was acted on, so the system cannot show whether it is followed or where planners routinely overrule it. Blocked on purchase order outcomes, which the application does not hold |
| Lot sizes and minimum order quantities | The formula has no lot-size, MOQ or container rounding. If the business needs it, that is new work, not a missing config value |
| Draft containers | Draft units show on the row and are deliberately not subtracted, since a draft is not a commitment. One check remains: how many rows carry drafts at all. If most do, the sub-line should become a sortable column |
| Small frontend defects | `SCREENS.md` §4, particularly the `error_basis` values that render untranslated in the Korean locale |

## 7. Codebase cleanup

Lowest priority. `scripts/` holds roughly 97 entries, most of them one-off experiments. Fourteen dead files were removed on 2026-08-12; this is the long tail.

Warning: the numbered `ml_NN_*` scripts are cited by number throughout `ML_FORECAST_DESIGN.md` as the evidence for individual decisions, and the rebaseline logs under `docs/rebaseline_2026-08-03-v2/` are named after them. Deleting one does not break code; it breaks the audit trail.

Procedure:

1. Keep anything referenced by name in `docs/`.
2. Keep anything importable from `src/`.
3. For the rest, check `git log` and whether the output is committed under `outputs/reports/` or `docs/rebaseline_*`. Keep those whose output is cited.
4. Move into `scripts/archive/`, so recovery is a `git mv`.

Known candidates:

- `SkuForecastsService.getForecastBounds()` in the Commerce repository, which lost its only caller and still has a passing test
- `scratch_introspect_fc_products.js`, `scratch_list_objects.js` and `Archive.zip` in the Commerce root
- The four completed task briefs in `docs/`, which could move to `docs/tasks_completed/`
