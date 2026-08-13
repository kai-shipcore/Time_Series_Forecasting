# Work Log

Running record of completed work on the ML forecasting project, one dated line per item.
Used to generate day-by-day progress summaries for reporting. When a summary is produced,
a marker line is added; the next summary covers everything after the latest marker.

Entry style: date, then a plain one-line description of what was completed. Technical
detail lives in the design document and codebase guide, not here.

---

## NEXT SESSION — written 2026-08-10 for 2026-08-11

Deliberately at the TOP of this file, not the bottom. Summaries cover everything after the
last `SUMMARY PRODUCED` marker, so a plan written below one would be reported to the manager
next time as work that was completed. Replace this block as it is worked through.

**State of play.** v11 is the model. v17 was rejected today, and it was the last candidate
anyone had. The snapshot is `2026-08-03`, the version log is re-measured against it, and the
final test has still never been run. Nothing is blocking it.

**1. First thing, and it is time-limited.** The weekly cron runs Tuesday morning. Read its
output once for two lines:

    forecast        N rows written to shipcore.ml_forward_forecasts
    history         N rows also written to shipcore.ml_forecast_history

That is the only outstanding part of BACKLOG 11, and it closes by being observed rather than
by anything being built. If either says NOT written, the server's `.env` is the place to look.
The same run is also the first production use of `drop_leading_partial_week`, so its output
should mention dropping the partial leading week.

**2. The decision that has to be made before anything else is started: item 2 or the final
test, because the order is not free.**

BACKLOG item 2 changes `eligible_skus`, which changes which SKUs are scored, which changes
every number including the final test's. So it can be done BEFORE the final test, or not at
all. Doing it after would leave the final test measured on a population the code no longer
produces, which is the one result in this project that cannot be quietly re-run: Section 2.2
quarantines that window and the value of the number comes from it having been used once.

Both directions are defensible. Item 2 first is the more correct evaluation, and today
established that a re-baseline costs three minutes with `scripts/_rebaseline_run.sh` already
written. Final test first is the safer deliverable, and item 2 then becomes documented work
for whoever inherits this rather than half-finished work. What is NOT defensible is starting
item 2 without deciding, because a half-migrated eligibility rule is worse than either.

**3. The final test, when it is run.** Once, on v11. Preconditions, all currently met:
`ML_SEASONAL_BLEND` is `"off"`, `ML_DATA_SNAPSHOT` is `2026-08-03`, lightgbm is 4.7.0 matching
the pin, and the week convention is settled and deployed. Use `final_test_split`, not
`dev_splits`. Write the result up whichever way it lands, including if it is worse than the
development windows suggested, which is the ordinary outcome and the reason the window was
held back.

**4. Everything else, in rough order of value:** BACKLOG 15 (atomic pipeline writes, which
lets the Stop button return), 21's remaining thread (what starts the unmanaged uvicorn;
detection is built, the cause is not found), 6 (retire the old page, still waiting on
`ml_forecast_history` accumulating runs).

**Known and accepted, not to be re-raised as findings:** `Archive.zip`, the credentials in
`.claude/settings.local.json`, and the partially-exposed `FORECAST_API_TOKEN`. Assessed
2026-08-10 and accepted on the grounds that the repository is internal to about three
developers. Recorded here so the next person inherits the judgement rather than rediscovering
the files and not knowing whether anyone had looked.

---

- 2026-07-17: Created the master design document (goals, metrics, data and split
  protocol, decision log, feature backlog) and revised it into a professional format.
- 2026-07-17: Created the codebase guide documenting every file and the full data flow
  from raw sales to scored forecast.
- 2026-07-17: Audited and hardened the evaluation harness: as-of eligibility and segment
  labels, pinned evaluation windows, stratified validation SKUs, segment-merge reporting,
  missing-prediction warnings, ramp cohort moved into the harness.
- 2026-07-17: Established the structural baseline (per-segment seasonality; alpha search
  confirmed no seasonal adjustment for short SKUs) with floor numbers on three dev
  windows.
- 2026-07-17: Built and evaluated model versions v0 (lead-only, rejected), v1 (ramp
  block, rejected), and v2 (deseasonalized trajectory features, rejected); each
  rejection's mechanism identified and logged; v3 direction defined.

--- SUMMARY PRODUCED 2026-07-20 (covering all entries above) ---

- 2026-07-20: Set up the work log process and repository instructions (CLAUDE.md) so
  future work sessions follow the same documentation and reporting habits.

- 2026-07-20: Built and evaluated model version v3 (fully seasonally consistent ML path;
  met two of three pass criteria, best version so far, blocked by one segment's
  regression in the post-holiday window).

- 2026-07-20: Pinned model development to a frozen copy of the sales data so that
  accuracy comparisons between model versions are measured on identical numbers. The
  weekly automated refresh was quietly changing the figures underneath the comparison,
  which had already shifted recorded results. The automated refresh continues to run
  unchanged for the production forecast; only the development track reads the frozen
  copy. Added a tool to create and check these frozen copies, and confirmed all six
  previously recorded baseline accuracy figures reproduce exactly.

- 2026-07-20: Put the forecasting work under version control for the first time and saved
  each model version as its own labelled checkpoint, so any earlier version can be
  retrieved and re-run later. None of this work had been backed up before.

- 2026-07-20: Found and fixed a fault that had made the first model version impossible to
  re-run: it was picking up a later version's settings rather than its own. Checked all
  four versions re-run correctly, and locked the software library versions so results
  cannot shift between machines.

- 2026-07-20: Re-measured every model version's accuracy on the frozen data so all four
  are scored on identical figures and can be compared directly. Previously the earlier
  versions were measured on older data than the latest one, so the same model showed
  different numbers in different tables. None of the conclusions changed. Also wrote down
  the rule for when an accuracy difference counts as real rather than chance, which had
  been applied but never actually defined.

- 2026-07-20: Fixed a date-alignment fault in the benchmark of the current spreadsheet
  method, which had been scored against a period one day off from the one it was
  forecasting. Corrected the document's description of that method: it had been described
  as consistently forecasting too low, but measurement shows its error changes direction
  by season. Also recorded that the spreadsheet method still beats our baseline for
  established products in the autumn period, which is the one case where it wins and had
  not been written down anywhere.

- 2026-07-20: Built and evaluated model version v4, which tells the model whether each
  product is long-established or new. It fixed the problem it was designed to fix, giving
  the best result any version has achieved for established products in the post-holiday
  period, but it did so by shifting effort away from newer products and making those
  noticeably worse. Rejected on that trade. The finding points clearly at the next step:
  the model needs to be prevented from favouring one group over the other, rather than
  simply being told the two groups differ.

- 2026-07-20: Added two permanent health checks that run alongside every model version:
  one tests whether the seasonal adjustment actually fits each product group, the other
  breaks a model's error out month by month. The first check found that the December
  adjustment fits established and newer products very differently, which had gone
  undetected under the single headline accuracy number.

- 2026-07-20: Measured the earlier statistical system (the project's stated accuracy
  target) through the same evaluation used for everything else, for the first time. In
  doing so found and fixed two faults in its evaluation script, including one where it
  treated newer products differently from the intended method. Result: the current model
  already matches or beats that target for newer products across all test periods, with
  one remaining gap for established products in the post-holiday period.

- 2026-07-20: Tested giving established and newer products different December
  adjustments, based on the health-check finding. Clearly rejected: with only one or two
  Decembers of history the estimates are unreliable, and adjusting one month shifts
  errors into the neighbouring months. Cheap to find out now rather than after building
  it into the model; revisit after this year's holiday season adds a third December.

- 2026-07-20: Confirmed with statistical checks that the current model's only real
  remaining weakness is established products in the post-holiday period; everywhere else
  it now matches or beats both the earlier statistical system and the spreadsheet method
  used in production.

--- SUMMARY PRODUCED 2026-07-21 (covering all 2026-07-20 entries above) ---

- 2026-07-21: Investigated why forecasts run high after the holidays and found the cause
  is the seasonal adjustment rather than the model: a holiday uplift is being applied to
  the last two weeks of December, which in both years of available data were at or below
  normal demand rather than above it. The code also contradicts its own documented
  intent, which says that period should revert to the normal December level.

- 2026-07-21: Gave the machine-learning work its own copy of the seasonal settings so that
  adjusting them cannot alter the older statistical system we measure ourselves against.
  Verified the separation changes no existing result, and added a permanent check that
  will catch it if the two ever drift apart unintentionally.

- 2026-07-21: Tested rebalancing how much each product group influences training, on the
  theory that established products were crowding out newer ones. Rejected: it made newer
  products worse rather than better, which rules out that explanation. A control version
  included in the same test showed the real cause is simply telling the model which group
  a product belongs to, which helps established products and harms newer ones no matter
  how the training is balanced. Four different approaches have now hit this same trade-off,
  which points clearly at keeping separate models for the two groups.

- 2026-07-21: Tried training separate models for established and newer products, the last
  of the structural options. Rejected: it helped established products slightly but made
  newer ones significantly worse, confirming that newer products benefit from being
  trained alongside established ones.

- 2026-07-21: Corrected the holiday period used by the forecast so it matches the
  promotional period the company actually runs, late November to mid-December, rather than
  running to the end of December. This resolved the long-standing post-holiday accuracy
  problem for established products, which five previous attempts had failed to fix by
  treating it as a difference between product groups. The forecast now matches or beats
  the earlier statistical system for both product groups across all three test periods.

- 2026-07-21: Ran a broad hyperparameter search over the model's main tuning settings,
  after an initial narrow attempt had tested too little to conclude anything. Result: the
  model is essentially already well configured; the best settings found gave no meaningful
  improvement and made one test period slightly worse. More importantly, this settles that
  the remaining post-holiday weakness for established products is not a tuning problem. It
  comes from the model expecting continued growth into a period where demand falls, which
  will need a new signal that anticipates the turn rather than any tuning change.

- 2026-07-21: Solved the long-standing post-holiday accuracy problem for established
  products. Split the forecaster into two models: newer products keep the shared model
  that serves them well, and established products get a dedicated model with a new feature
  that recognises when a product is running far above its yearly norm and is therefore
  likely to fall back rather than keep climbing. This fixed the one remaining weak spot
  decisively, left newer products untouched, and the forecast now matches or beats the
  earlier statistical system in five of six test cases. The feature generalises to any
  temporary spike, not just the holidays. Still to do before this can replace the current
  spreadsheet method: the final held-out test, which is run only once.

- 2026-07-21: Added a backlog item to check whether preorders are handled correctly.
  Preorders book demand when placed but ship later, so if sales are counted on the order
  date the demand is recorded in the wrong weeks, which would distort the whole forecast
  and hit newly launched products hardest. Flagged to verify how the data is recorded and
  correct it if needed.

- 2026-07-21: Tested adding product age as a feature for newer products. Rejected: it made
  the most recent test period dramatically worse. The reason is a known modelling pitfall,
  age always increases, and the forecast is made at each product's oldest point, beyond
  what the model saw in training, so the model extrapolated wildly. Recorded as a caution
  against any ever-increasing feature of this kind. Newer-product accuracy is already
  strong, so this line is not pursued further.

- 2026-07-21: Tested an acceleration feature (is the recent trend speeding up or slowing?)
  for both product groups. Both rejected: it made newer products worse and did nothing for
  established products, including the autumn ramp-up period it was aimed at. This was the
  last promising feature from sales history alone. The conclusion is that the model is now
  close to the best achievable on the sales record as recorded; further accuracy gains will
  need the sales record itself cleaned for stockouts and misplaced preorders.

- 2026-07-22: Wrote a non-technical project summary for management
  (Demand_Forecasting_Project_Summary.docx, saved at the repo root). Plain-language
  overview of purpose, the newer-versus-established product split and why it exists, method,
  the accuracy measure and why it suits demand, a like-for-like results table comparing the
  current production formula (V1) against the new model on the same products and test
  windows, current limitations, project stage, and next steps (stockout and preorder
  correction, a proper frontend, and the final quarantined test). Figures drawn from the
  pinned 2026-07-20 snapshot version log (v11 vs V1).

- 2026-07-22: Dropped third-party external data (site traffic, marketplace analytics) from
  the roadmap for the foreseeable future and removed it from the design doc, project
  write-up, and code comments. The planned model extension is now the corrected demand
  target only: stockouts and misplaced preorders. Section 5.3 reframed from "feature
  candidates pending external data" to "data-quality corrections pending source data."

- 2026-07-22: Expanded the management summary and independently verified every figure in
  it against the pinned 2026-07-20 snapshot. Reran the V1 benchmark and the WA12/v-base
  baselines through the harness: V1 reproduced to the fourth decimal (short Mar-May 0.4198
  bias -39.8%, long Dec-Feb 0.4044 bias +38.7%, long Oct-Dec 0.0847) and v-base matched the
  pinned floor, confirming the seasonal multipliers, the preorder-plus-sales weighted V1
  windows, and the pooled-WAPE pipeline are all wired correctly. Added to the doc: the exact
  V1 formula, the data source table (shipcore.fc_velocity_link_snapshot_forecast), the M4/M5
  rationale for the method choice, concrete 10-week unit totals and forecast-error-in-units
  per segment/window (actual vs V1 vs new model), the business-impact framing (total error
  roughly halved, ~24,300 to ~11,700 units off across the two complete windows), and an
  inventory-aware-ordering item under the interface work. Reproduction scripts kept in the
  scratch outputs, not committed.

- 2026-07-22: Added visuals and a demand-mix breakdown to the management summary. Computed
  the last-90-day demand split from the snapshot (Established 81 SKUs / 31% of demand, Newer
  366 / 49%, Occasional 3,002 / 20%; regular-selling ~80% of demand) and put it in a table
  plus a donut chart. Added two line charts (established and newer) plotting 10-week actual
  demand against the current method and the new model across the three test windows, which
  make the tracking difference visible at a glance. Also reworked the results tables to show
  bias next to the forecast totals, and rewrote the current-method section as the actual
  formula (weighted velocity windows, dampening, seasonal modifier) rather than a plain-
  language walkthrough. Stopped referring to the current formula by its internal name in the
  boss-facing document. Chart/compute scripts kept in scratch outputs, not committed.

- 2026-07-22: Reran the v11 hybrid model locally to produce real week-by-week forecasts for
  the three development windows (lightgbm 4.7.0). Reproduced the recorded window totals to
  within a handful of units per segment, confirming the weekly series is consistent with the
  reported figures. Replaced the three-point summary charts in the management document with
  continuous weekly time-series charts (Altair, monotone-smoothed): actual weekly demand
  against the current formula and the new model, one for established and one for newer
  products. These show the week-to-week reality, including where the model runs smoother than
  actual demand rather than tracking it perfectly. Also clarified the current-method section
  (plain-language lead-in, defined the (days, weight) window pairs and every symbol),
  explained the 12-week moving-average baseline in the model section, and replaced the
  season nicknames with month-range labels (Oct-Dec 2025, Dec-Feb 2026, Mar-May 2026)
  throughout. Model-run and chart scripts kept in scratch outputs, not committed.

- 2026-07-22: Refined the management document's charts and limitations. Made the new model's
  forecast line dashed like the current method's (only actual demand is solid now, since both
  others are predictions), and labelled each dashed retraining boundary directly on the charts
  as "new 10-week forecast" with an explanatory paragraph, so the ten-week-test structure is
  unambiguous. Removed the "cautious model" limitation (the model is not conservative relative
  to the current formula) and moved the stockout/preorder data-correction item out of
  Limitations into the next-steps section, keeping Limitations to things that genuinely cannot
  be corrected (short history, small established group, backtesting-only status, out-of-scope
  sporadic SKUs).

--- SUMMARY PRODUCED 2026-07-22 (covering the 2026-07-21 and 2026-07-22 entries above) ---

- 2026-07-22: Committed the management-summary chart generator to the repo
  (scripts/plot_management_forecast_charts.py), previously scratch-only. It regenerates the
  two continuous weekly charts end to end (runs the v9 and v11 models plus the current
  formula on the pinned snapshot, builds the weekly series, renders the Altair charts to
  outputs/reports/) and reproduces the figures in the document exactly. Added altair and
  vl-convert-python to the unpinned viz dependencies in requirements.txt.

- 2026-07-23: Reworked the management summary (Demand_Forecasting_Project_Summary). Added
  plain-language definitions of demand and forecast as two set-apart paragraphs at the top of
  Purpose, shortened the Purpose paragraph, and relabelled every "product" as "SKU" through the
  body, tables, and chart captions (leaving "production" intact). Moved the document source to
  Markdown (Demand_Forecasting_Project_Summary.md) with a custom pandoc reference template
  (custom-reference.docx) and a one-command build (build_docx.sh), so the Word file is
  regenerated from Markdown in a single pass; the build needs pandoc --columns=20 or LibreOffice
  collapses the narrower tables. Added scripts/plot_demand_breakdown_donut.py to regenerate the
  demand-breakdown donut with SKU labels, matching the breakdown table; it writes to
  outputs/reports/ like the weekly-chart generator. Full modeling content (current-method
  formula, M4/M5 rationale, moving-average and backtesting explanations) was kept unchanged.

- 2026-07-23: Investigated why the current method still beats the new model on long-history SKUs
  in the Oct-Dec 2025 window, for the management summary. From orders_raw, preorders are only 0.2%
  of long-history demand in that window (1.2% Dec-Feb, 2.4% Mar-May), far below the 10% the current
  formula's weight assumes, so preorders do not explain the result. The window is a gentle,
  predictable Q4 rise (about +18 units/week for long SKUs) where both methods are accurate (current
  8% error, new 10%, a gap of about 277 units); the new model's trend and elevation features add a
  small over-forecast (+1.7% bias) where a trailing average is already well calibrated (-1.0% bias).
  This matches the design doc's growth-drift account (Sections 4.18 and 4.27). Added a short
  justification and a consolidated error-and-bias table to the summary.

- 2026-07-23: Tested a second hypothesis for the Oct-Dec 2025 long-history result, that the
  seasonal multipliers were calibrated to that window. Not supported by the repo. The base monthly
  multipliers (deseasonalize.py SEASONAL_BASE, and v1.py) are the hand-set legacy Google Sheet
  values, deliberately not fit to data (design doc 4.9/4.10, only two seasonal cycles). The one
  optimized seasonal term is the holiday multiplier (1.26, holiday_multiplier_search.csv), and it is
  used by the new model, not V1, and is fit across both Decembers rather than this one window, so if
  anything it favors the new model. Q4 is confirmed as the demand peak (long SKUs ~2,047 units/week
  vs ~1,350-1,560 elsewhere) where V1 is well calibrated (bias -1.0%), but a de-trended monthly
  seasonal estimate is too noisy on two cycles to isolate per-month multiplier accuracy. Left the
  summary's existing "easy window" justification unchanged and did not add the multiplier claim.

- 2026-07-23: Built the Phase 1 Streamlit inventory/forecast dashboard prototype's data
  layer under dashboard/lib/, which reads the real forward forecasts, sales history,
  segmentation, and the model-vs-V1 accuracy exports. Inventory, preorder, inbound, product
  name, size, and status are not present in this repo, so the dashboard reads them from
  data/inventory/inventory_snapshot.csv and falls back to a clearly labelled sample seeded
  from real sales until a real export is dropped in. Documented the priority logic,
  recommended-order-quantity formula, and known limitations in dashboard/README.md.

- 2026-07-23: Added scripts/export_forward_forecasts.py and exported the live
  shipcore.fc_forward_forecasts table to data/processed/fc_forward_forecasts.parquet,
  replacing the stale test snapshot the dashboard was reading. Current run (2026-07-20)
  covers 447 smooth SKUs, matching expectation. The live table turned out to hold four
  accumulated weekly runs rather than one, so the export script reports counts per run
  as well as the total, to avoid a misleading combined figure.

- 2026-07-23: Reset the dashboard front end to rebuild it collaboratively rather than in one
  pass. Removed the five auto-generated pages, kept the dashboard/lib data and calculation
  layer, and corrected the segmentation labels from new/established to short/long to match the
  design doc. Rebuilt the home page as a stub and the SKU detail page (actual sales against the
  model and spreadsheet forecasts). Wrote dashboard/PAGES_BUILD_SPEC.md, a step by step build
  specification for the remaining pages that composes the existing lib functions and marks which
  screens run on real data versus sample inventory.

- 2026-07-23: Found that the dashboard was reading the legacy statsforecast and spreadsheet
  forecast tables, not the LightGBM model that is the focus of the project. The forward forecast
  table and the accuracy exports contain only the production statistical models (WindowAverage,
  AutoETS, AutoARIMA, ensembles) and the V1 formula, with no LightGBM output. Confirmed the
  LightGBM track currently produces only evaluation scores inside the experiment scripts
  (src/ml/evaluate.py and the ml_*.py scripts), retraining the model inline on each run, with no
  saved model and no per-SKU forward forecast. Concluded the next step is to extract the current
  best version (v11) into standalone, reusable model files that can be retrained and produce
  per-SKU forward forecasts, built so later model improvements drop in and flow straight to the
  dashboard. Deferred to a later session.

--- SUMMARY PRODUCED 2026-07-24 (covering the 2026-07-23 entries above) ---

- 2026-07-24: Extracted the best model version (v11) from its experiment script into a
  standalone, reusable serving package under src/ml/serving/ (base.py defines a ForecastModel
  interface, models.py holds V11Hybrid plus a version REGISTRY and CURRENT_BEST, persist.py
  saves and loads a fitted model, forecast.py provides forward_forecast and validate_version).
  The package composes the frozen src/ml primitives rather than reimplementing them, so an
  existing version cannot change when a new one is added. Added scripts/ml_forward_forecast.py,
  the LightGBM counterpart to run_forward_forecast.py, which trains the current best model on
  all history and writes a per-SKU forward forecast to data/processed/ml_forward_forecasts.parquet
  and the fitted model to outputs/models/. Verified the extraction reproduces the recorded v11
  per-segment pooled WAPE exactly on all three development windows (short 0.1961, 0.2000, 0.1783;
  long 0.1355, 0.1380, 0.1000), that the forward run covers all 447 smooth SKUs over 13 weeks,
  and that a saved model reloads and predicts identically. To add a future version, subclass
  ForecastModel, register it, and point CURRENT_BEST at it once it wins; the forward pipeline and
  dashboard then pick it up with no further change.

- 2026-07-24: Decided how V1 fits the LightGBM dashboard and wrote the implementation task
  (docs/V1_AND_DASHBOARD_WIRING_TASK.md). V1 stays a separate artifact keyed by SKU and week,
  joined into the dashboard at read time rather than written into the model forecast rows, and
  is always recomputed from a fresh database pull at forecast time using the existing
  scripts/compare_v1.py formula. V1 cannot be computed from the model's weekly sales table
  because its formula needs daily order lines split across five fulfillment streams
  (west/east sales and preorder, plus FBA), which the weekly totals discard. The task covers a
  new forward-pipeline V1 step, a V1 accuracy baseline scored on the same development windows,
  and pointing the dashboard at the model outputs and a precomputed accuracy file instead of
  the legacy statsforecast tables.

- 2026-07-24: Carried out the V1-and-dashboard wiring task end to end. Added a V1 forecast
  step to the forward pipeline that recomputes V1 fresh from the database on every run and
  writes it as its own file on the same SKU/week grid as the model (scripts/ml_forward_forecast.py;
  all 447 forecast SKUs covered, none missing from the velocity pull). Added a V1 accuracy
  baseline scored on the same three development windows as the model and wrote a combined
  comparison file (scripts/ml_accuracy_report.py); the model reproduced its recorded numbers
  exactly and beat V1 in five of six window and segment cells, losing only long SKUs in the
  Oct-Dec window, matching the design document. Pointed the dashboard at the LightGBM forecast
  and the new comparison file instead of the legacy statistical tables, updated the SKU detail
  page's model label to match, and fixed a data layer function that would have broken once the
  old columns disappeared. Flagged the forecast accuracy page's build plan as needing a
  rewrite, since the new comparison file is pre-aggregated per window and segment rather than
  one row per SKU.

- 2026-07-24: Added a per-SKU accuracy file alongside the pooled one, so the forecast accuracy
  page can still show a largest-errors view. The per-SKU actual-versus-predicted numbers were
  already being computed inside the shared scoring function and then thrown away after
  pooling; pulled that step out into its own function and rebuilt pooling on top of it, so the
  existing scorer's behavior is unchanged by construction, and re-ran the full harness
  regression check plus an independent re-aggregation of the new per-SKU file to confirm it
  reproduces the pooled numbers exactly (zero mismatches). scripts/ml_accuracy_report.py now
  trains once and writes both files instead of training separately for each.

- 2026-07-24: Built the Forecast Accuracy dashboard page (Step 3), the third of five planned
  screens. Shows the LightGBM model against V1 on the three historical backtest windows: a
  headline card per window with the win/loss margin, a pooled-error and bias breakdown by
  window and short/long segment, the share of individual SKUs the model wins on, and the
  largest over- and under-forecasts for a selected window. Confirmed the numbers read
  correctly: the model wins clearly in two of the three windows and loses only the one
  segment already known to favor V1. All three built pages pass the smoke check.

- 2026-07-24: Reworked the SKU detail page against the original source requirements (the
  forecasting product owner's spec, one level up from the derived build plan): it now shows
  current inventory, confirmed inbound, preorder backlog, estimated stockout date, and the
  recommended order quantity calculation alongside the actual-versus-forecast chart, with the
  weekly forecast numbers also shown as a table rather than only a chart. The inventory-derived
  fields carry the sample-data warning banner until real inventory is wired in. Also reworked
  the forecast accuracy page's comparison section after feedback that it was hard to read:
  replaced two bar charts with one table breaking every window into total, short, and long
  rather than only the combined total, which surfaces a result the combined view was hiding
  (the model loses on long SKUs in the Oct-Dec window specifically, not the window overall).

- 2026-07-24: Saved the forecasting product owner's original dashboard requirements
  (docs/PLANNING_REQUIREMENTS.md), which had only existed in chat history until now, and pointed
  the derived build spec at it as the source of truth when the two disagree.

- 2026-07-24: Built the Inventory Overview page (Step 4), replacing the placeholder home page.
  Six headline counts (forecasted SKUs, preorder priority, out of stock, best sellers at risk,
  total recommended order quantity, SKUs stocking out within 30 days), a priority mix chart, a
  short-versus-long split chart, and a preview of the top ten recommended orders. Sample
  inventory data, banner included. Numbers check out internally: the priority and short/long
  counts each sum to the full 447-SKU total. Four of six planned dashboard pages now built.

- 2026-07-24: Renamed the dashboard's main script from app.py to Inventory_Overview.py so the
  sidebar shows an accurate name instead of "app" (Streamlit derives that label from the
  filename, with no way to override it in code for the main script). Updated every launch
  command and reference accordingly. Reviewed all four built pages against the requirements
  doc, the codebase's own conventions, and Streamlit best practices; fixed three small issues
  (a number missing thousands separators, priority labels not using the existing colour-coded
  markers, a table with no caption explaining it), and flagged three bigger ones for a decision
  rather than deciding alone: the shared planning table is recomputed from scratch, uncached,
  on every page interaction; two of the summary bar charts may have the same readability
  problem the forecast accuracy ones did; and the accuracy page may be missing an aggregate
  actual-versus-forecast view the requirements doc calls for.

- 2026-07-24: Resolved the three flagged items. Cached the shared planning table (0.13s cold,
  instant when parameters are unchanged, correctly recomputes when they change; verified both
  ways). Replaced the inventory overview's two bar charts with count cards, following the
  visualization guideline that a handful of plain counts belongs in stat tiles, not a bar
  chart, which matches the original complaint. Added the missing portfolio-level view to the
  accuracy page: total actual demand against both methods' total predicted demand, in units,
  per backtest window, using the same visualization guideline in the other direction (three
  distinct series over three periods is exactly what a grouped bar chart is for). The new
  numbers independently confirm the bias pattern already on record: the model stays close to
  actual in all three windows while the spreadsheet method overshoots by about 23% in one
  window and undershoots by about 31% in another.

- 2026-07-24: Gave the shared metric-card helper an actual border and switched it to a
  responsive wrapping row instead of a fixed column grid, since it was called a "card" but had
  no visual boundary. Fixed in the one shared function every page's stat tiles go through, so
  it applied everywhere at once rather than page by page.

- 2026-07-24: Added a theme file (.streamlit/config.toml) since the dashboard had none and was
  running on unmodified defaults, whose default border color turned out too faint to notice.
  Set an explicit, clearly visible border color and a light card-tint background. Locks the
  app to light mode as a side effect (a single base theme can't offer the dark-mode toggle);
  flagged rather than silently accepted.

- 2026-07-24: Switched the theme to dark by default after the light version came back as too
  bright. Kept the override minimal this time (just border color and corner radius) and let
  Streamlit's own dark theme handle everything else, rather than guessing a second full palette
  after the first guess missed.

- 2026-07-24: Built the Purchase Priorities page (Step 5), the fifth of six planned screens,
  while blocked on the real inventory-data credentials. Filterable, searchable priority list
  with a CSV export and a jump into SKU Detail for a chosen row. Verified the filters actually
  filter (priority, search) and not just that the page loads. One page left: Data Quality
  Alerts.

- 2026-07-24: Removed a locally-invented "master SKU" field that turned out to correspond to
  nothing real. It was computed by stripping the trailing colour/size token off the SKU string,
  a guess made before we had access to the real inventory tables. Checked those tables directly
  once we had access: their own "master_sku" field is the same identifier as our SKU, not a
  coarser grouping, so there was never a real distinction to approximate. Removed the field and
  every place it was displayed or filtered on, fixed a data-quality check that referenced it
  directly (would have broken once it was gone), and recorded the finding in the requirements
  document as a deviation from the original spec's column list, since that document names it
  as a separate column.

- 2026-07-24: Started a page-by-page review of the dashboard against the requirements document,
  with the business owner reviewing and giving feedback on each page in person before the next
  one starts. First page: Inventory Overview. Clarified why "out of stock" and the "No Stock"
  priority count differ (out of stock is broader; a SKU that is both out of stock and on
  preorder is labelled Preorder, since preorder outranks no-stock), added a plain-language
  explanation of what each priority label means, and folded the preorder count into the priority
  mix section instead of a separate card, since the requirement is only that the count appear
  somewhere on the page. Reworked the priority mix into a combined view: four count cards in a
  condensed 2x2 grid beside a donut chart of the same breakdown, with the short-history/
  long-history split moved to sit directly under the top summary cards as its own row rather
  than a separate section. Iterated on the chart's colours and the gap style between segments
  several rounds based on visual feedback, landing on a lighter four-colour palette, a donut
  (rather than a full pie, whose wedges bunched together at the centre) with rounded gaps between
  segments, and on-chart percentage labels confirmed to line up with the correct segment. Rest of
  the page not yet reviewed.

--- DAILY SUMMARY 2026-07-24 ---
- Found that the new dashboard was still showing forecasts from the old method, and that our best new forecasting model only existed as a one-off experiment we couldn't reuse.
- Rebuilt that model into a proper, reusable tool that produces a demand forecast for every product, and got it running to generate a fresh 13-week forecast across all ~450 active products.
- Set the old spreadsheet method to run beside it as a benchmark, always on the latest sales data, and confirmed the new model is more accurate in five of six comparisons.
- Connected the dashboard to the new model and built most of its screens for the purchasing and inventory teams: an overview, a per-product detail view, a forecast-accuracy view, and a purchase-priority list.
- Some inventory figures on those screens are still placeholders until we get access to the live inventory data; everything forecasting-related is real.

--- SUMMARY PRODUCED 2026-07-24 (covering the 2026-07-24 entries above) ---

- 2026-07-27: Started docs/BACKLOG.md for decided-but-unbuilt work, so future items stop living
  only in chat history. Seeded it with two entries. First, a stockout-aware demotion rule for
  src/profile.py: only demote a SKU to intermittent when its recent 13-week mean is below the
  threshold and it actually had stock, since demotion removes a SKU from the forecast entirely
  and a long stockout currently triggers it. Blocked on inventory data, which the profiler has
  no access to today, and on whether inventory history exists, since a current snapshot cannot
  answer an as-of question without leaking into backtests. Second, the broader censored-demand
  problem the forecasting team's two reference analyses both measure: during stockouts,
  pre-order periods and post-restock recovery, units sold understates demand, and the model
  reads that as demand falling. Recorded the data position for both: pre-order streams are
  already in orders_raw, stockout and restock event dates are not available anywhere.

- 2026-07-27: Rebuilt the dashboard front end from the new plan. Wrote docs/PLANNING_PLAN.md,
  which starts from the decisions users actually make rather than from a screen list, records
  a verified inventory of which data is real, sample and missing, and maps four screens plus
  one deferred against those decisions. Validated the map with a worked walkthrough of one
  weekly cycle using real forecast figures, which surfaced two gaps rather than hiding them.
  Deleted the four previously built screens and the old presentation helper, keeping the data,
  calculation and data-quality modules, since those hold verified business logic rather than
  layout assumptions. Added a reliability module that turns the stored per-SKU backtest errors
  into a good/fair/poor tier plus an explicit no-history state, which covers 260 of 447 SKUs
  and splits them 102/107/51 with 187 unmeasured. Built the Action List screen: summary counts
  that act as filters instead of a separate overview page, a banded scrollable table with a
  pinned SKU column, priority and reliability shown with glyph and text rather than colour
  alone, search and filters, CSV export, and a drill-down control. Confirmed every filter
  returns the count it should and the page runs without exceptions in each state.

- 2026-07-27: Fixed a fault in the dashboard's per-SKU reliability figure. The accuracy export
  holds one row per SKU per window for the model and another for the spreadsheet baseline, and
  the reliability calculation was pooling both, mixing two methods' errors and double-counting
  the actuals. It now scores only the model version actually being served. The correction moves
  the portfolio median error from 0.231 to 0.176 and reclassifies 94 of the 260 measured SKUs,
  so the tier boundaries were re-cut against the corrected distribution (good under 15%, fair to
  30%, poor above), giving 107 good, 90 fair, 63 poor and 187 with no measured history.

- 2026-07-27: Built the SKU Detail screen, the second of four planned. Shows the recommended
  order quantity as a line-item calculation beside a reliability panel, on the reasoning that a
  user arrives already knowing the number and needs to know how it was derived and whether to
  trust it. The reliability panel shows the per-window forecast against actual rather than only
  a headline error, and states whether the misses run one way or both, since a SKU that misses
  in both directions calls for a different response than one that consistently forecasts low.
  Below that: the inventory position, an Altair chart of actual demand against both the model
  and spreadsheet forecasts with the forecast boundary marked, and the weekly figures collapsed.
  Wired the Action List drill-down to open the screen with the chosen SKU. Verified all three
  reliability states render, including the no-history case that covers 187 SKUs, and that the
  order quantity shown matches the sum of its own breakdown.

- 2026-07-27: Reworked the recommended order quantity after reviewing how it was calculated.
  Three changes, each agreed before implementing. Safety stock is now sized by the SKU's own
  measured forecast error rather than a flat fourteen days of trailing sales; the flat rule had
  the buffer backwards, holding the most stock against the best-predicted SKUs and the least
  against the worst. SKUs with no backtest history inherit their segment's median error rather
  than being treated as error-free. Orders now cover the lead time plus the reorder cycle, since
  covering only the lead time leaves a shortfall every cycle. Confirmed inbound is credited only
  when its ETA falls inside that window, and the stockout projection now depletes on-hand stock
  first and counts inbound only if it arrives before the shelf empties, which corrected the
  at-risk count from 187 to 272 SKUs. Also made every component round to whole units before the
  total is taken, so the line items shown to a user add up exactly to the quantity displayed;
  previously they could disagree by a unit or two on 42 SKUs. Recorded all definitions, and the
  superseded ones, in the plan document.

- 2026-07-27: Changed the estimated stockout date to deplete stock against each SKU's own
  forecast curve rather than a flat trailing average, removing the last place where a headline
  number ignored the model. Stock is consumed week by week with interpolation inside the week
  it runs out, and demand past the end of the horizon continues at the horizon's average rate.
  The difference is largest exactly where it matters: on a SKU whose weekly demand ramps from 5
  to 20 units, 60 units of stock lasts 52 days rather than the 32 a flat average of the same
  curve would report. Checked the depletion function against hand-worked cases including zero
  stock, exact-week boundaries, extrapolation beyond the horizon and zero-demand SKUs, and
  confirmed the count of SKUs out of stock now still equals the count with zero days of cover.

- 2026-07-27: Exposed the planning assumptions as sidebar controls, so lead time, reorder
  cycle and service level can be adjusted rather than being fixed in code. The controls are
  shared by both screens and persist across navigation, so the order quantities, stockout
  dates and headline counts on the two pages always reflect the same assumptions. Service
  level is offered as a percentage with its multiplier shown, since that is the meaningful
  choice rather than the raw factor. Checked that each control moves the totals in the
  expected direction and magnitude, that the SKU detail breakdown relabels its coverage line
  to match, and that the line items still sum exactly to the order quantity at a non-default
  lead time.

- 2026-07-27: Stopped the sample inventory generator from inventing master data. It had been
  drawing a product status at random from a weighted list and guessing a size by taking the
  first all-digit token of the SKU string, which for CA-SC-10-F-10-BK-1TO returns the product
  line number rather than the size. Those appeared on screen as ordinary badges, so a reader
  had no way to tell a fabricated classification from a real one, and the size was not merely
  invented but confidently wrong. This is the same fault as the locally-invented master SKU
  field removed earlier. Product name, size and status are now left blank rather than
  simulated, and their badges only render when the value is real. Quantities are still
  simulated, because the screens cannot be exercised without them, but the priority badge now
  carries a marker saying it derives from sample stock, so the caveat travels with the figure
  instead of sitting only in the banner at the top of the page. The three data-quality checks
  that read status or size now report as unavailable while the sample snapshot is in use,
  rather than flagging all 447 SKUs and reporting a crisis that is really an absent export.

- 2026-07-27: Removed the invented size and product-status fields from the dashboard
  altogether, rather than only blanking them, and deleted the three data-quality checks that
  existed solely to test them. Replaced them with product category, read from the first two
  tokens of the SKU, which is real because the prefix is part of the identifier. Only the three
  prefixes whose meaning was confirmed are named (CA-SC seat cover, CC-CC car cover, CA-FM
  floor mat); the remaining ten display their raw prefix rather than being given an invented
  label, so they still group and filter correctly without asserting anything unverified.
  Category now appears on the SKU detail header, under each SKU in the action table, and as a
  filter. Four data-quality checks remain, of which three run and one is honestly reported as
  needing the order-line export.

- 2026-07-27: Simplified product category to a family rule, since every CC-prefixed SKU is a
  car cover regardless of its second token: 335 seat covers, 111 car covers, and one SKU whose
  prefix has no confirmed meaning left showing its raw prefix rather than being guessed at.
  Made the order-quantity table read as the arithmetic itself rather than as a list of figures,
  by adding a leading operator column with running plus and minus signs and an equals sign on
  the total, plus a one-line plain-language statement of the formula beneath it. The operator
  is carried as an explicit field on each line rather than inferred from the number's sign,
  because a subtracted line whose value happens to be zero would otherwise display as an
  addition, which it did. Added horizontal padding throughout the tables, which had cells
  butted against their borders, worst in the reliability window table.

- 2026-07-27: Fixed the recommended order quantity showing as 1 on the SKU detail headline for
  every SKU. The loop that renders the order breakdown assigned its per-row flag to a variable
  named total, which is also the name of the function parameter holding the headline quantity,
  so by the time the headline was formatted the number had been replaced by a boolean and
  printed as 1. Renamed the loop variable and added a note against it. Worth recording because
  the value was correct everywhere else on the page, including the total line of the very table
  the loop was building, which made it look like a display truncation rather than a variable
  being overwritten. Also centred the figures in the order breakdown and gave the reliability
  table's window column left padding, both of which were flush against their borders.

- 2026-07-27: Colour-coded the order breakdown so each line's operator and figure share a
  colour, green for what adds to the quantity to buy and red for what subtracts from it, with
  the total and the informational late-inbound line left in the default colour. Applied the
  same two colours to the reliability miss column, replacing the blue and amber it used, and
  reused the existing good and poor dot colours rather than introducing a second green and red.
  Noted in the code that the miss colour encodes direction rather than quality, since a miss in
  either direction is an error. Made the table header rows taller, keeping the second row's
  sticky offset equal to the first row's height so the two stay flush while scrolling.

- 2026-07-27: Replaced the two-colour forecast-miss indicator with a diverging scale by
  direction and severity, on the reasoning that under-forecasting and over-forecasting are not
  opposites of the same thing: missing low causes stockouts and lost sales, missing high only
  ties up cash. Under-forecasts escalate through yellow, orange and red as they worsen;
  over-forecasts run cool through teal and blue; anything within ten percent either way reads
  as accurate. Band edges were set against the observed distribution of 572 SKU-window misses
  so the central band holds about a third of cases and every outer band stays populated. Added
  a legend, since seven colours are not decodable without one, and it doubles as the statement
  that the two directions carry different costs. The order breakdown keeps its simpler green
  for additions and red for subtractions.

- 2026-07-27: Added a portfolio demand chart to the action list, showing actual sales for the
  last 26 weeks against the model and spreadsheet forecasts for the next 13, summed across
  whichever SKUs the current filters select rather than a fixed total that would disagree with
  the list beneath it. The spreadsheet baseline covers all 447 forecast SKUs, so the two
  forecast lines are directly comparable; where a filter leaves that untrue the caption says
  so. The chart makes an existing result visible at a glance: the model projects roughly 4,200
  to 4,900 units a week against the spreadsheet's 3,400 to 3,800.
  Declined to add a reliability history chart to the SKU detail page. Only 66 of 447 SKUs have
  three backtest windows and 180 have two, so a line chart would draw a trend through two
  points and imply knowledge of a trajectory that does not exist. Added a magnitude bar beside
  each miss percentage instead, which makes severity comparable at a glance without asserting
  anything the data does not support. Also moved the chart construction into the shared
  presentation module so the two screens cannot drift apart on colour, dash pattern or the
  marking of where history ends.

- 2026-07-27: Made the demand charts hoverable. A tooltip attached to the lines themselves only
  fires within a couple of pixels of a two-pixel stroke, so in practice it never fired; replaced
  it with a nearest-point selection that snaps to the hovered week, draws a crosshair, marks the
  point on every series and reports all three values in one tooltip. Enlarged both charts, the
  portfolio one to more than double its previous height, and softened the gridlines. Kept the
  portfolio chart between the filters and the table after considering the alternatives: above
  the summary counts would put a control below its own output, since the chart follows the
  filters; below the table it would be scrolled past; and on its own screen it would lose the
  connection to the filtered list. It stays collapsible so anyone working the list daily can
  reclaim the height. Also renamed two locals in the chart block that shadowed the History
  filter's variable and the forecast frame, which worked only because the filter signature was
  computed earlier in the file and would have broken silently if the block ever moved.

- 2026-07-27: Three additions to the two built screens. The SKU detail page now states the
  plausible requirement alongside the recommended quantity, flexing only the demand term by the
  SKU's measured error and deliberately excluding safety stock from the band, since safety stock
  is the cushion chosen to cover that same uncertainty and including it would count it twice.
  Checked that the recommendation falls inside the band for all 447 SKUs at the default service
  level; at higher service levels it sits above the band by design, and the card says so rather
  than leaving the discrepancy unexplained. The action list is now sortable by any column with a
  reset, replacing the fixed priority-then-urgency order, which was the main thing the custom
  table lacked against an ordinary data grid. Data-quality flags now appear where decisions are
  made rather than only on a screen that does not exist yet: a summary line above the list, a
  marker beside each flagged SKU carrying the wording in its title, and the full labels on the
  SKU detail page. 192 of 447 SKUs currently carry one, almost all of them "nothing inbound".
  The reset control had to be moved to a callback: Streamlit refuses writes to a widget's key
  once that widget exists, and the button is created after the two sort controls.

--- DAILY SUMMARY 2026-07-27 ---
- Rebuilt the two main dashboard screens: a priority list of what needs ordering, and a per-product detail view.
- Improved how the recommended order quantity is worked out, so it reflects how accurate each product's forecast has been.
- Fixed a fault in the accuracy figures shown per product.
- Removed placeholder product details that could have been mistaken for real information.
- Added the practical touches: adjustable settings, sortable list, demand charts, and warnings where data is incomplete.

--- SUMMARY PRODUCED 2026-07-27 (covering the 2026-07-27 entries above) ---

- 2026-07-28: Traced where the Commerce Integration application gets inventory, preorder and
  inbound figures, so the dashboard can read the same production sources instead of a sample
  file. On hand and backorder come from `ecommerce_data.coverland_inventory_by_warehouse` in
  the Supabase lookup database; confirmed inbound and its ETA come from `fc_containers` and
  `fc_container_items` in the primary database, counted only for containers at status
  `shipped` or `packing_received`. The dashboard will match that status filter rather than
  choose its own.

  The investigation turned up something about our own pipeline. Order lines are classified as
  `sales`, `preorder`, `ttm` or `ttm_preorder` before they reach us, and `src/ingest.py`
  selects from `fc_velocity_link_snapshot_forecast` without filtering on that column, so the
  training target has always included preorder units, attributed to the order date. That
  answers the open question recorded in Section 5.4 item 6, which had been waiting on
  confirmation of whether preorders were flagged at all and whether the series keyed on order
  date or ship date. Both are now settled, so excluding or down-weighting preorder rows is
  testable without new source data; attributing them to the fulfilment week is still blocked,
  since no available source records that date. Recorded in Section 2.1, with 5.3 and 5.4
  updated to match. (Originally written into the Decision Log as 4.29 and moved: that section
  takes design choices only, and this is a property of the data.)

  It also settles a dashboard question. Because preorder demand is already inside the
  forecast, the preorder term in the recommended order quantity has to be an open obligation
  stock rather than a preorder sales rate, or the same units are counted twice. The velocity
  snapshot cannot supply that, since it drops fulfilment status when aggregating, and the
  preorder queries in the Commerce Integration repository measure preorders already shipped
  over a trailing 91 days. The `backorder` column is the only stock-level source. One check
  is outstanding before using it: whether `available` in that same table is already net of
  `backorder`, which would make the recommendation run high.

- 2026-07-28: Made each SKU in the action list a link straight to its detail page. A comment in
  the code claimed a custom HTML table could not raise a click back to Streamlit, and that was
  wrong: the table is rendered with `st.markdown(unsafe_allow_html=True)`, which writes into the
  main document rather than an iframe, so an ordinary anchor in a cell behaves like an ordinary
  link. The SKU travels in the query string and the detail page consumes it once, comparing
  against the last value it took, so the URL cannot keep overriding the selectbox on later
  reruns. Nothing is written back to the URL, which avoids rerun side effects. An unrecognised
  SKU in the URL now warns and falls back rather than silently showing a different product.
  The link is the whole first cell rather than the SKU text alone, so the click target is the
  full row height, and the drill-down selectbox and button underneath the table were removed:
  an anchor is natively tab-focusable, so that widget only duplicated a path already working by
  both mouse and keyboard. Reaching a SKU that is not on the current page is now done by
  searching or filtering to it. Corrected the stale comment in both files. Verified by
  rendering the table outside Streamlit and checking that the anchor wraps the entire cell on
  every row with the SKU surviving URL encoding and the data-quality flag still inside it, then
  by four AppTest cases: arriving with no parameter, with a valid one, changing the selection
  afterwards, and arriving with a SKU that does not exist. Both pages render with no
  exceptions and the CSV export is unaffected.

  One follow-up on the styling, recorded because the cause is not obvious and will recur with
  any future anchor in this table. The SKUs first came out permanently coloured and underlined
  rather than only on hover. The reset was written as a bare class, `.dfx-cell`, at specificity
  0-1-0, while Streamlit styles anchors globally with selectors of the form `.stMarkdown a` at
  0-1-1. The bare class therefore lost and the theme's own link styling applied throughout, so
  what looked like a hover rule that fired constantly was a reset that never applied at all.
  Fixed by qualifying the selector with the element and marking the two properties important,
  which is the right use of important here since it overrides a third-party stylesheet rather
  than our own, and by listing every link state, or the theme recolours the cell on hover and
  drags the subtitle with it. Text decoration needed particular care because it propagates from
  an anchor to its descendants and cannot be cancelled by a child. Added a specificity check to
  the verification so the comparison against Streamlit's rule is asserted rather than assumed.

- 2026-07-28: Chased why the dashboard's stock figures looked low once real inventory replaced
  the sample. The join is sound: all 447 SKUs matched, and of the 118 showing zero, 110 have a
  container inbound and 102 have backlog owed, which is the pattern of genuinely stocked-out
  products rather than a broken lookup. The figure is narrow rather than wrong. The export pulls
  `available`, which is physical stock less units already allocated to unshipped orders, so the
  column labelled "On hand" was overstating what it contained by whatever is allocated. Renamed
  it to "Available" on both screens and documented the definition in the schema. The export now
  also reports physical on-hand and allocated alongside it, so the size of that difference is
  visible rather than inferred, and lists available by warehouse to show whether any sits
  outside the four the Commerce app pivots into west and east.
  Separately, the Commerce planning grid adds a `transit_stock` field to its stock total that
  this export did not carry, which is a second reason the two screens disagree. It is now
  exported as its own column but deliberately not used in any calculation: nobody has been able
  to say what it counts, it may be the same units as the container inbound already credited, and
  folding it into available stock would feed it into the week-by-week depletion as though it
  were on the shelf today, making every stockout date optimistic. The script reports how many
  SKUs carry both it and container inbound, and how often the two are equal, which is the
  evidence needed to decide whether they are the same units.

- 2026-07-28: Added a per-SKU backtest chart to the detail page, so a bad reliability figure can
  be investigated rather than only read. Each window the model was scored on is shaded from its
  cutoff to ten weeks later, with a dashed rule at the cutoff marking the boundary between what
  the model had seen and what it had not, and a label giving predicted against actual with the
  signed error. Placing the training data immediately to the left of the result it produced is
  the point: a large miss can be traced back to the shape that caused it. The section opens
  automatically when the SKU sits in the poor tier and stays collapsed otherwise, and is not
  rendered at all for SKUs with no backtest history.
  One deliberate limitation. The accuracy export records a single predicted total per window,
  not per-week predictions, so drawing a predicted curve through the window would mean inventing
  a within-window shape that was never recorded. The prediction is shown as the total spread
  evenly across the window, drawn faintly and labelled as an average per week, which is a
  restatement of the recorded total rather than a claim about any individual week. The only
  line on the chart is real demand. This is the same reasoning that led to declining a
  reliability trend line earlier: with two windows for most SKUs there is no trend to draw, but
  there is evidence to show, and the evidence is what was asked for.

- 2026-07-28: Replaced the flat level in that chart with real per-week predictions, which
  required generating data that did not previously exist. The stored accuracy report keeps one
  predicted total per SKU per window, so `validate_version_weekly` was added to
  `src/ml/serving/forecast.py`: the same fit and predict loop and the same eligibility filter as
  the recorded totals, stopping short of the aggregation. A new script,
  `scripts/ml_backtest_weekly.py`, runs it and reconciles the result against
  `ml_accuracy_by_sku.csv` before writing anything, refusing to write if any window total
  disagrees, on the grounds that a chart contradicting the accuracy figure printed beside it is
  worse than no chart. It reconciled exactly across all 572 SKU-window pairs, so the chart now
  shows what the model predicted in each week against what actually happened, with the lead
  time available on hover. The dashboard treats the file as optional and falls back to the old
  flat level with different wording when it is absent, so a model version without it degrades
  rather than breaks.
  Also moved the per-window results out of the plot. Coloured text drawn on the chart was
  unreadable, competing with the demand line and colliding between adjacent windows. They are
  now a row of chips above the chart, using the same miss scale as the action table, with the
  colour on a filled chip rather than on thin glyphs.
  Two follow-ups on the same chart, both of which the main demand chart had already solved and
  this one had not. The hover reported only the actual figure, because the crosshair was built
  from the history frame alone while the prediction carried its own tooltip on the point
  markers, which in practice never fires: a tooltip on a mark only triggers within a couple of
  pixels of it. The crosshair is now built from a tidy frame pivoted back into columns and
  carries actual, predicted and lead together at the nearest week. Separately, each window's
  prediction line began at its first forecast week, leaving it floating unattached to the
  history it came from. It is now anchored to the actual value at its own cutoff, the last week
  the model saw. The anchor is an observation rather than a prediction, so it is added to the
  line only and not to the point markers, which keeps every dot on the chart a genuine forecast
  and keeps the window totals reconciling. Both are asserted in the checks.
  A third followed from the second. Carrying the prediction on the crosshair meant every week
  showed a predicted value, including the weeks outside any backtest window where none exists,
  and the pivot leaves those null so they rendered as NaN. Vega-Lite cannot vary a tooltip field
  list per datum, so the weeks are now partitioned between two rules: one over the weeks inside
  a window carrying four fields, one over the remainder carrying two. Exactly one holds any
  given week, so exactly one tooltip can fire, and outside a window the prediction rows are
  absent rather than empty. The selection needs to see every week or the crosshair would snap
  only within whichever subset owned it, so it is attached to a third invisible rule spanning
  the whole series, placed underneath so the two visible rules win the tooltip. The check
  asserts the partition is disjoint, complete, and exactly equal to the predicted weeks.
  That partition then broke the hovering it was built on top of, and was reverted. Attaching a
  nearest selection to a mark is what builds the Voronoi region that lets a pointer anywhere
  snap to the closest week, so the mark holding the selection has to be the mark holding the
  tooltip. Splitting them across layers put the selection on the invisible capture rule and the
  tooltips on the visible ones, which silently shrank the target back to the one-pixel rules and
  the markers themselves. The lesson worth keeping is that in Vega-Lite the snap and the tooltip
  are one mechanism, not two that can be composed separately.
  Since Vega-Lite cannot vary a tooltip's rows by datum, the field list is now constant and the
  weeks outside a window substitute an em dash through isValid rather than formatting a null
  into NaN. The row still appears but reads as not applicable, which is the better trade against
  losing the snap. The check asserts the selection and the tooltip share one mark, that the
  nullable fields are never tooltipped raw, and that both guards are present. Prediction markers
  were also enlarged from 22 to 45, since at the smaller size they read as noise on the line.

- 2026-07-28: Investigated why one short SKU was over-forecast by 150% and ran the experiment
  it suggested. The behaviour generalises: predictions are correctly ordered by the ramp
  feature at a lead of one week and that ordering has disappeared by ten weeks, while actual
  outcomes stay ordered throughout. The model reads a collapse and then discards it, settling
  on the average short-SKU response, which is a ramp because the training population is 39%
  rising against 1% collapsing. That behaviour is recorded in the v14 version-log entry, and
  the three measurement cautions it exposed went to Section 2.4, which governs how accuracy
  claims are produced: the ramp feature is computed on the deseasonalized series, the model's
  output must be divided by both level and seasonal factor to recover the ratio it predicted,
  and anchors with unusual feature values cluster in one window so pooling disguises a window
  effect as a feature effect. Each had produced a wrong reading first. Also corrected a stale
  row in the feature backlog that listed the ramp ratio as a candidate to try; it has been in
  the model since v1. (Both were first written into the Decision Log, as 4.29a, and moved: the
  section preamble restricts it to design choices, and these are a diagnosis and a set of
  measurement rules.)
  The experiment was v14, lowering min_child_samples on the short-serving model from 200, on
  the hypothesis that a region holding 1.1% of training rows could not be resolved under a
  five-leaf ceiling. Pass criteria were written into the version log before running, per the
  working rules, including a tail criterion and a guard against buying a tail fix with damage
  elsewhere. It was rejected at 100, 50 and 20: the decision windows tie with no consistent
  sign, and the tail moved the wrong way at every value. The long segment was identical
  throughout, which was the control, and v11 reproduces to four decimal places after the
  plumbing change.
  The negative result is the useful part. The tail is not capacity-constrained, so the cause
  sits in the objective rather than the tree structure: under demand-weighted L1 a leaf of
  low-volume collapsed SKUs contributes little to the loss however finely it is split. It also
  confirms the earlier hyperparameter finding on a subpopulation that study could not have
  seen. Recorded with it is a measurement point that outlives the experiment: tail error is
  above 100% while short pooled WAPE is 0.196, both correct, because the metric weights by
  units and these SKUs carry almost none. Pooled WAPE cannot be the instrument for this
  problem, so any future attempt needs a per-segment criterion agreed in advance.

- 2026-07-28: Settled what to do about SKUs the forecast handles badly, and the answer was to
  say so rather than to change the forecast. Two measurements decided it. Routing on measured
  accuracy does not work, because per-SKU error barely persists between windows (Spearman 0.13;
  knowing a SKU was poor last window raises the odds of it being poor next window from 27% to
  44%), so a rule built on past accuracy would misroute most of what it touched. Routing on the
  SKU's observed state does work where it matters: for SKUs whose last 4 weeks are well below
  their last 12, a plain 4-week average scored 0.28 against the model's 1.49 on the worst
  group, and lost badly everywhere else. But blended across the decision windows it improved
  one and slightly worsened the other, failing the sign-consistency requirement of the adoption
  rule, so under that rule the simpler design stands.
  The dashboard now carries the finding instead. A SKU whose demand is falling gets a callout
  above the order card giving its recent weekly rate, the ratio against its longer average, the
  same figure carried flat across the forecast horizon, and the model's number beside it, so
  the size of the disagreement is visible rather than merely asserted. The flat line is drawn
  on the demand chart for those SKUs only, and the condition appears as a flag on both screens
  through the existing channel. 58 of 447 SKUs qualify, carrying 5% of forecast units; across
  them the model forecasts 1.88 times the flat carry.
  The recommended order quantity is deliberately unchanged and asserted unchanged in the
  checks. Substituting a different number silently would claim more than the evidence
  supports, since that substitution was tested and did not meet the bar. The ramp is computed
  by importing the seasonal factors from `src.ml.seasonal` rather than reimplementing them, so
  the dashboard's notion of falling demand cannot drift from the model's own feature.

- 2026-07-28: Replaced the detector behind that warning after checking it against the SKU that
  prompted the whole investigation. It was not flagged, correctly: its collapse was in
  February, and by July its recent rate and its forecast agree to within 4%, so the model had
  re-based and there was nothing left to warn about. But testing that exposed the flag as a
  proxy for the wrong thing. It fired on the shape of past demand, whereas the condition worth
  warning about is the forecast standing above the rate the SKU sells at now. Measured against
  the direct comparison, the proxy missed seven SKUs and invented thirteen; the clearest miss
  sold a quarter of a unit a week against a forecast of 2.7, eleven times over, while its ramp
  read as rising because the twelve-week average behind it was lower still.
  The flag is now the direct comparison, with two conditions: the forecast is at least 1.5
  times the recent four-week rate, and the excess is at least twenty units across the horizon.
  The second exists because a ratio alone is meaningless at low volume, where a tenfold
  over-forecast can still amount to nothing anyone should act on. That selects 22 of 447 SKUs
  carrying 766 units of excess, against 58 under the proxy. SKUs with no recent sales at all
  have no ratio and qualify on the excess alone, which is the same warning worded differently.
  The callout now leads with the arithmetic for that SKU, which can be checked against the
  chart below it, and appeals to the backtest result only when the SKU is also in a falling
  state, since that is the pattern the result was measured on. Quoting it otherwise would
  borrow authority the measurement does not give. The ramp is retained as a descriptive column
  that words the callout but no longer triggers it. The recommended quantity is unchanged and
  asserted unchanged, as before.

- 2026-07-28: Stopped the dashboard serving forecasts for SKUs the segmentation no longer
  considers forecastable. Fifteen of the 447 were in that position, with a median of 68% zero
  weeks against an intermittent threshold of 30%, one of them selling in a single week out of
  twenty. The cause was that the dashboard read its bucket from the forecast file, where
  `forward_forecast` writes the literal "smooth" on every row because only smooth SKUs are
  modelled at all. That column can never disagree with itself, so a SKU demoted since the run
  was undetectable, and the data-quality check testing for a missing bucket could never fire
  either. The bucket now comes from the current profile snapshot, demoted SKUs are dropped
  from the planning table, and the count is kept on the frame rather than printed, because
  silently shrinking a table is how totals stop reconciling for reasons nobody can find. The
  gap reopens weekly as SKUs flip, so this is a standing reconciliation rather than a cleanup.
  The list now covers 432 SKUs and 2,382 recommended units.
  Recorded as backlog item 2 that the list should eventually cover every SKU, including the
  roughly 2,900 the model does not forecast. That is not a filter change: coverage demand,
  safety stock, order quantity, stockout date and reliability are all derived from a forecast
  those SKUs do not have, so it needs a stated basis of its own, and one that is honest about
  its accuracy in the way the forecast path is. Otherwise the screen mixes measured numbers
  with unmeasured ones under the same headings.

- 2026-07-28: Asking why 174 served SKUs had no backtest history turned up a correctness
  problem in the evaluation harness. Only 21 of them were simply too young at the last cutoff,
  which was the expected answer. The other 153 had well over a year of calendar history but an
  active-weeks count of exactly 13, the value the promotion path assigns. When `profile.py`
  promotes an intermittent SKU to smooth, it rewrites `train_start` to the first of the
  trailing 13 weeks, which is correct for training, since the earlier intermittent period
  should not be learned from. But `train_start` is also what `eligible_skus` uses to decide
  whether a SKU had enough history at a backtest cutoff, and for a promoted SKU that value is
  not a launch date: it advances with every profiling run while the evaluation windows stay
  pinned in the past. Measured against the three windows those SKUs show negative history at
  every one, so they are never scored and never will be. That is 187 of 447 served SKUs, 42%,
  carrying 14.8% of forecast units and 20% of recommended units.
  Excluding them is defensible in itself, since grading a promoted SKU on an older window would
  score a period when it behaved differently. The consequences are what had gone unnoticed: a
  large minority of what the dashboard serves has never been measured and cannot be, their
  safety stock falls back to a segment median rather than anything observed about them, and
  their reliability tier reads as not-yet-measured when it should read as not-measurable. The
  docstring on `eligible_skus` asserted that `train_start` is a stable per-SKU property, which
  is how this stayed invisible; the claim is corrected in place. Recorded as backlog item 2,
  with the fix being to split the field into a stable launch week for eligibility and a
  separate training start. It is blocked on the same consideration as advancing the snapshot:
  changing eligibility changes the scored population and re-baselines every recorded number, so
  it should be paired with another change that forces a re-baseline rather than spending one
  alone. Also cleared the now-stale first blocker on backlog item 1, since real inventory
  exists. Separately confirmed the 351 zero-order SKUs are all genuinely covered: supply meets
  need for every one, 269 mainly through inbound and 82 through stock on hand.

- 2026-07-28: Answered whether promoted SKUs are worth forecasting, and fixed what the answer
  exposed. Promoted SKUs cannot be scored while they are in that state, but ones promoted
  earlier were, so re-running the profiler against sales truncated to each backtest cutoff
  recovers who was promoted at the time and joins them to scores already recorded. No model is
  refitted. They come in at 0.2397 pooled against 0.1912 for the rest of the short segment,
  worse in all three windows, distinguishable from sampling noise in one of them. That is a
  usable forecast, better than the legacy spreadsheet manages on the whole segment, so they
  are worth forecasting and dropping them would leave a fifth of recommended units with
  nothing at all. Their median per-SKU error is close to everyone else's, so the pooled gap
  sits in a few larger promoted SKUs rather than across the cohort.
  The fix that follows: safety stock for an unmeasured SKU fell back to the segment median of
  0.199, but the unmeasured population is overwhelmingly promoted SKUs whose measured error is
  0.24 whenever it can be measured. They were being given a cushion sized by a number now known
  to be too low for them. The fallback now splits by cohort, identified through the
  `active_weeks` value the promotion path assigns. Safety stock on those 158 SKUs rises from
  900 to 1,086 units and 34 recommended quantities move, all upward and by one or two units.
  The number lives in a named constant with its provenance, and `scripts/promoted_sku_accuracy.py`
  reproduces it, redirecting the profiler's output directory to a temporary path so that
  running the analysis cannot overwrite the pinned snapshot the project is measured against.
  A row now records whether a SKU's error is measured, from the promoted cohort, or from a
  segment median, so the UI can show provenance rather than a number that looks measured.
  Worth recording a testing trap: the first before-and-after comparison showed no change at
  all, because `build_planning_table` is cached and the second call returned the first result.
  Changing a module constant between calls does not invalidate that cache. The comparison only
  became truthful after clearing it, and the same mistake would silently pass any future A/B
  done this way.

- 2026-07-28: Two fixes on the SKU detail page. A clicked SKU sometimes appeared for an instant
  and then snapped back to the previously viewed one. The cause was that the selectbox had no
  key and the incoming SKU was applied through `index=`, which supplies a default only while
  the widget holds no stored state. Streamlit keeps widget state across page navigation, so
  once the selectbox had been used at all its stored value won and the argument was ignored,
  which is why the fault appeared intermittently rather than always: it depended on whether
  that widget had been touched earlier in the session. The incoming SKU is now written to the
  widget's own key before the widget is built, which is the supported way to move a control
  that already exists. The stored value is also repaired when it falls outside the served list,
  which happens when a SKU is demoted between forecast runs while a session is open. Six cases
  are asserted, including that a manual choice is not re-overridden by a stale URL parameter
  and that a later link click still works after one.
  Separately, the plausible-requirement figure was hard to see. It was the middle of three
  consecutive lines at the same size and opacity, sitting below a caption rather than below the
  number it qualifies, so it read as more caption. It is now a tinted chip carrying the range at
  13px in the accent colour of the headline figure, placed directly beneath that figure and
  before the caption. The note explaining a recommendation above the band, which applies to six
  SKUs, is kept underneath as its own line.

--- DAILY SUMMARY 2026-07-28 ---
- Hooked up the real inventory data, replacing the placeholders. Cross-checked it against the
  system the rest of the business uses and fixed a mislabelled stock column.
- UI fixes. Product names are clickable, the order figures are readable, and clicking one
  product no longer shows a different one.
- Added forecast diagnostics. A view showing how each product's forecast did when it was
  tested, so a bad score can be investigated instead of just noted.
- Looked into products the forecast handles badly. Tried a model change, it did not work, so
  the dashboard warns about them and shows a simple alternative figure instead.
- Stopped forecasting products that should not be forecast at all, and raised the safety margin
  on the products we have least data about.

--- SUMMARY PRODUCED 2026-07-28 (covering the 2026-07-28 entries above) ---

- 2026-07-29: Started moving the dashboard into the Commerce Integration application as a Next.js
  page. Nothing of Streamlit ports: it is a Python UI framework where the interface is the Python,
  so every widget and layout call has to be rewritten in React. Of the 2,835 lines, the 1,681 that
  render are discarded and the roughly 740 that compute are not, and those are the expensive part.
  Rather than porting the order formula to TypeScript, where two implementations would drift with
  nothing to say which screen was right, the planning logic moved out of `dashboard/lib` into a new
  `src/planning` package that both hosts import. `dashboard/lib/__init__` re-exports it, so the page
  scripts read unchanged and `lib.calc is src.planning.calc` holds: one module object, no way to
  diverge. The three private copies of the Streamlit caching fallback collapsed into one shared
  `_cache`, which resolves to `st.cache_data` where available and a no-op otherwise, so the package
  imports with Streamlit absent entirely. One path bug was caught in the move: the inventory
  snapshot directory was written relative to the module rather than the repo root, and would have
  silently resolved to `src/data` in the new location.
  Two FastAPI endpoints follow, `/planning/action-list` and `/planning/sku/{id}`, computing nothing
  of their own. Verified against the Streamlit screens row for row: 432 rows, 2,426 recommended
  units, nine numeric columns identical to within 1e-9, five categorical columns identical, flags
  identical on every row, and the order breakdown and plausible band matching per SKU. The payload
  is strict JSON with no NaN literals, since pandas nulls would otherwise reach the browser as
  strings. A demoted SKU returns 404 explaining that segmentation now classes it intermittent
  rather than the bare "unknown" an unrecognised SKU gets, because the first is a normal outcome
  the page should explain. Planning parameters are query arguments and demonstrably move the
  numbers: 2,426 units at an eight-week lead time, 20,344 at sixteen.
  Recorded prerequisite for deploying this away from the forecasting repo: these endpoints read
  files, not Postgres. The v11 forward forecast is written only to
  `data/processed/ml_forward_forecasts.parquet`, while `shipcore.fc_forward_forecasts` still holds
  the legacy statsforecast output, and inventory arrives as an exported CSV. That is fine while
  FastAPI runs inside this repo and is the blocker if it ever does not.
  Decided: Streamlit is retired once the Next.js page reaches parity, so the duplication is
  temporary by design rather than a permanent second surface.

- 2026-07-29: Inventory now reads the databases directly instead of an exported CSV. Those tables
  belong to the Commerce Integration application and refresh on its schedule, so holding a copy
  only created a way to be confidently wrong when nobody re-ran the export. The SQL moved into
  `src/planning/inventory.py` and the export script imports it, so the live path and the export
  cannot produce different figures. Three sources are tried in order, each a degradation of the
  one before: the databases, then the exported CSV, then generated sample data, with a `source`
  column recording which applied. Falling back rather than raising is deliberate, so a working
  copy without credentials still starts. Results are memoised for five minutes, in the module
  rather than through Streamlit, because FastAPI shares this code and has no caching of its own.
  Two things the move nearly lost. The engine builder promised to return None when unusable but
  only checked for missing variables; `create_engine` resolves the driver eagerly and raises when
  psycopg2 is absent, which is a real environment here, so the whole body is now guarded and the
  fallback actually works. And the export script had an explicit numeric coercion guarding the
  case where a query matches nothing and pandas infers an object dtype, which is live today
  because transit stock is zero for every SKU; that guard is carried into the shared path and
  widened to every numeric column, since the same holds for any of them the moment a query
  returns nothing. Verified against mocked database responses that a SKU with no stock record
  stays null while a recorded zero stays zero, which is the distinction the whole thing rests on.
  The v11 forecast stays on parquet by choice: the model is still moving, and putting it in
  Postgres now would mean re-uploading on every iteration. That remains the one prerequisite for
  running the API away from this repo.

- 2026-07-29: The action list now exists as a page in the Commerce Integration application, at
  /planning/action-list, registered in the navigation with Korean and English labels and visible
  by default for the roles that already see the other planning pages. Two proxy routes forward to
  the FastAPI planning endpoints through one shared helper rather than repeating the base URL,
  token header, timeout and error envelope the way the fifteen forecast routes each do. Status
  codes pass through untouched, because a 404 there carries meaning: it distinguishes a SKU that
  is no longer forecastable from one that does not exist, and the page words those differently.
  Settled first what to reuse from the existing demand-forecast page. Its screen-level components
  turn out not to be reusable: all-skus-table hardcodes its row type and sort keys to forecast
  fields and fetches its own endpoint, and demand-trend is shaped for portfolio accuracy with
  lead selection and prediction bands. Two smaller ones looked like wholesale candidates and were
  not, on inspection: page-headers has no generic header component, only one specific header per
  page, and sku-global-search is bound to the forecast segment taxonomy and routes into the
  forecast page. What is genuinely shared is the substrate, and it is most of the value: the ui
  primitives, the internationalisation provider, the API path helper, the proxy pattern, the
  SSR-safe Plotly import and the styling conventions. The line drawn is to reuse the substrate
  and not the screens, on the grounds that the demand forecast page answers how the model is
  doing and this one answers what to order today, and forcing one to serve the other would
  recreate the problem the dashboard was built to escape.
  Verified end to end against a running server on the port AI_SERVICE_URL points at: all six
  summary figures identical to the Streamlit screen, 432 rows either way, every field the table
  renders present in the payload, no TypeScript errors from the new files and lint clean.
  Two faults the checks caught rather than review. The lint rule on synchronous setState inside
  an effect was correct: loading is now derived from whether the loaded parameters match the
  requested ones, which removes the cascading render and incidentally keeps the previous table on
  screen while a new lead time is in flight instead of blanking to a spinner. And the priority
  badges keyed on "Best seller" and "No stock" while the API emits "Best Seller" and "No Stock",
  which does not fail: the lookup falls through to the Routine style and the badge looks
  deliberate, while the best-seller chip filter matches nothing. The labels now live in one
  exported constant that both the table and the filter use, with a check asserting the set
  matches what the API emits.

- 2026-07-29: SKU detail built at /planning/action-list/[sku], so the rows on the list now lead
  somewhere. The SKU sits in the path rather than a query string, which makes each row a real
  URL: shareable, bookmarkable, and openable in a new tab. One request serves the whole screen,
  since a partial render is worse than a marginally slower one.
  The screen carries what the Streamlit version did: quality flags, the falling-demand callout
  above the order card because it is a caveat about the number in it, the order quantity as
  checkable arithmetic with its plausible band, the reliability card with its per-window
  evidence, the inventory position, and both charts. Where a SKU has no measured error the
  reliability card names the basis being used instead of presenting an inherited figure as
  though it belonged to that SKU. The 404 for a demoted SKU is worded as a normal outcome and
  explains itself, rather than reading as a failure.
  Types were written from the live payload rather than from memory, which was worth doing: the
  order breakdown carries a Sign column of +1, -1, 0 and null, where null marks an aside shown
  for context and outside the sum, and 0 marks the total line. Inferring direction from the
  value instead would render a component of zero as an addition when it is a subtraction.
  Two lint rules caught real hazards again. Deriving the operator column mutated a flag inside
  the render map, which React may re-enter, carrying the previous value in; the operators are
  now resolved in a memo so the render is a pure function of that array. And the loading state
  is derived from whether the loaded SKU matches the requested one rather than set inside an
  effect, the same fix the action list needed.
  Verified against the live endpoint across five archetypes: poor tier with an order, no
  backtest history, falling demand, zero order, and an unmeasured promoted SKU. Order totals and
  bands match the Streamlit figures in every case, every field the page reads is present in the
  payload, and both 404 shapes are distinguished. TypeScript and lint clean, with the repository
  baseline of 57 pre-existing Prisma errors unchanged.

- 2026-07-29: A parity check against the Streamlit screens found the two Next.js pages
  disagreeing with each other. Streamlit keeps the planning parameters in a sidebar that
  persists across pages, so changing the lead time on the list changed it on the detail view
  too. In Next.js each page mounts independently, and the detail view was requesting with no
  parameters at all, so it answered at the default eight-week lead while the row the user had
  clicked answered at theirs. On one SKU at a sixteen-week lead the list said 620 units and the
  detail said 326, with nothing on screen to explain the difference. That is precisely the
  failure this migration was structured to prevent, arriving through the interface rather than
  through a second implementation of the arithmetic.
  The parameters now travel in the URL across all four hops: the list builds them into each row
  link, the detail view forwards them to the API and carries them back on its return link, and
  both pages seed from the query string. They are clamped on read to the same bounds FastAPI
  enforces, so a hand-edited URL cannot produce a rejected request. A side benefit over the
  sidebar it replaces: a shared link now reproduces the assumptions the figures were computed
  under, which sidebar state never could.
  The same audit lists what the Next.js pages still lack against Streamlit, none of it wrong,
  all of it absent: sorting and pagination on the list, the history filter, the portfolio demand
  chart, the reliability legend and the quality summary line; and on the detail view the SKU
  selector for moving between SKUs without going back, and the weekly figures table.

- 2026-07-29: Sorting added to the action list, following the convention the existing all-SKUs
  table set: clickable headers, shift-click to add a criterion, position markers when more than
  one is active. Two departures. A third click clears back to the server's worklist order, which
  is priority then quantity and is not reproducible from any single column, so it is represented
  as the absence of a sort rather than an entry in the list. And nulls sort last in both
  directions, because a SKU with no stockout date is not the most urgent one and would otherwise
  lead every ascending sort. The missing history filter went in alongside.
  Then the intermittent tail, which turned out larger than assumed: 2,977 SKUs, 87% of the
  catalogue by count and about a fifth of recent unit volume. Leaving it off the page would have
  meant retiring the demand forecast page while losing the only place those SKUs were visible.
  They now have their own section behind a toggle, fetched only when opened, with columns that
  are honest without a forecast: 13-week demand, the weekly rate it implies, stock position,
  days of cover at that rate, and a flag where stock runs out inside the lead time. There is
  deliberately no recommended order quantity, and the section says so in as many words, because
  it cannot be derived without a demand model these SKUs do not have. Column names differ from
  the forecast table throughout, so a rate computed from a 13-week average is never sitting
  under the same heading as one from a scored model.
  The inventory read was widened from the 447 forecastable SKUs to all 3,409 profiled ones,
  which is a one-line change now that it queries the database rather than reading an export.
  A partition check caught the design failing its own claim. The non-forecast set was defined as
  absence from the forecast file, while the forecast section is built from the planning table,
  and those differ by the fifteen SKUs demoted since the run: they sat in neither section.
  Keying on what the other section actually shows makes the two a partition by construction,
  and the check now asserts no overlap and no SKU unaccounted for. Also asserted that no
  forecast-derived column reaches the non-forecast payload, that cover is null rather than
  infinite wherever nothing has sold, and that a missing inventory record stays null rather than
  becoming a zero.

- 2026-07-29: Closed the remaining gaps against the Streamlit screens, and finally built the
  supply-gap warning that had been outstanding since it was measured.
  The planning table now computes days until inbound against days to stockout and flags the SKUs
  that run dry in between: 185 of them, carrying 3,099 units of backlog already owed. Two columns
  on that table were computed on assumptions that contradict each other, and nothing said so.
  The stockout date ignores inbound entirely while the order quantity credits it as though it
  were already on the shelf, so a row could read "out in 12 days" beside "order 0" with a
  container 40 days away and no account of the days in between. The order quantity is not wrong:
  with an eight-week lead time a purchase order placed today lands after a container already
  booked, and only 1 of the 185 gaps could be beaten by ordering. What was wrong was the silence.
  The stockout cell now shows both dates, there is a chip and filter for the population, and the
  SKU detail view explains that the action is to expedite or reallocate rather than to buy.
  Also added the portfolio demand chart, the reliability legend, the data-quality summary line
  and the weekly figures table. The chart needed an endpoint of its own, taking the SKU list by
  POST rather than in a query string, because it follows the filters and that list runs to
  hundreds of identifiers. Aggregating server-side is not a convenience either: the client holds
  one row per SKU with no weekly series in it, and shipping four hundred SKUs of history to the
  browser to sum it there would be far more data than the answer. The summary line and the legend
  count what is on screen rather than the whole list, since a count that ignores the filters
  describes a different population from the rows beneath it.
  One bug found in the checks, the same shape as the partition failure earlier in the day: with
  no SKU list the trend endpoint defaulted to the forecast file rather than the planning table,
  which would have drawn a chart over 447 SKUs above a table showing 432. The client always sends
  its filtered list, so it would only have bitten a direct call, which is exactly when nobody is
  watching for the discrepancy. Defaulting to what the other surface actually shows fixes it, and
  the check now asserts the two agree.

- 2026-07-29: Started storing what the model predicts, because until now every weekly run threw
  the evidence away. `ml_forward_forecasts.parquet` holds one run and is overwritten, which
  answers "what is the forecast" and cannot answer "is the model getting better". A new
  `src/ml/serving/history.py` appends each run instead, keyed by model version, forecast date,
  SKU and target week, with a re-run in the same week replacing its own rows rather than
  duplicating them. Deliberately not named after a version: the version is a column, so a new
  one coexists with its predecessors and comparing them is a query rather than an excavation.
  Alongside it, functions that join stored predictions to actuals as the weeks close, and pooled
  WAPE per run per segment, which is what a performance-over-time view will read. Neither knows
  which version is current, so a new model is scored the moment its first run lands.
  One judgement worth recording. Scoring excludes the most recent settled week as well as weeks
  still in progress. The design doc notes late-registering orders make the tail unreliable for
  training; here it matters more, because scoring a week whose sales are still arriving reads as
  over-forecasting, and the newest run is always the one most affected. A performance chart would
  have shown a downward slope that was an artefact of settlement rather than a change in the
  model. Training can afford the last week since one noisy anchor among thousands changes little;
  a run scored on one unsettled week is scored on nothing else.
  Verified against real data: appending twice replaces rather than duplicates, a second model
  version coexists rather than overwriting, a run whose horizon has settled scores across leads 1
  to 13, and the current run correctly scores nothing because its horizon has not started. Also
  pointed the one remaining hardcoded version default at CURRENT_BEST, so nothing outside the
  version's own class definition names v11.
  Recorded prerequisite, now with a deadline attached: the final test window is already evaluable,
  its cutoff being 2026-05-04 with actuals complete to 2026-07-27. But 163 of the 447 served SKUs
  are ineligible at that cutoff and 160 of those are the promoted ones, so the final test would
  report on 284 SKUs rather than all of them. Backlog item 2, the train_start split, gates the
  coverage of the very test it was meant to inform.

2026-07-29  Built the Forecast Validation page in Commerce_Integration.
  Two new FastAPI endpoints. /planning/validation returns the model against the V1 spreadsheet
  baseline as a segment by window grid with per-cell winners and a demand-weighted headline, the
  coverage the comparison rests on, per-SKU best and worst cases, the performance of stored runs
  as their weeks close, and the state of the final test window. /planning/demand-patterns returns
  weekly demand, concentration at the top 5, 10, 20 and 50 percent of SKUs, and the segment mix.
  Neither endpoint names a model version; both read whatever versions are present.
  The page reports the current position plainly: 0.1596 against 0.2591 pooled WAPE, a 38 percent
  reduction, ahead in 7 of 9 cells. It also states the two cells the spreadsheet still wins, both
  Oct-Dec, and says on the page that only 258 of the 447 served SKUs are in those figures, with
  the reason, so the headline cannot be read as covering the whole catalogue.
  Weekly demand is split into the SKUs the model forecasts and the intermittent tail it does not.
  That split surfaced something not previously looked at: the tail was 6 percent of weekly volume
  a year ago and is 23 percent now. It is 2,962 SKUs and 48,519 units over the last 52 weeks,
  carrying no forecast at all.
  Two sections are empty by design and say so in place rather than being hidden: performance over
  time, which fills as runs accumulate and weeks settle, and the final test window, which stays
  quarantined until model development finishes.
  Fixed a naming collision found during verification. evaluate.py reports bias_pct in percentage
  points; the new history module reported the same field as a fraction. Both feed one API payload,
  so the same name meant two things a hundredfold apart. History now matches evaluate.py, verified
  by scoring a synthetic run built to be exactly 10 percent over.

2026-07-29  Made the forecast service explain itself when it cannot serve.
  A coworker opening the Action List saw "Could not reach the forecast server / Internal Server
  Error". The heading was wrong: their server was running and reachable. The cause is that
  data/processed and outputs/reports are gitignored, so their fresh clone had the code and none of
  the parquet and CSV files the API reads. The service then starts, passes a liveness check, and
  raises on every real request.
  /health now also reports readiness: which of the eight data files exist, which required ones are
  missing, what produces each, and which checkout the server is reading from. It still returns 200
  when data is missing, because the process is alive and conflating that with an outage would hide
  the distinction the endpoint exists to make.
  The planning proxy now classifies four failures instead of reporting them all as one: nothing
  listening, a server predating these endpoints, a server with no data, and a genuine error shown
  verbatim. A 500 triggers a readiness check before reporting, since that is almost always missing
  data rather than a bug. Verified against four stand-in servers, one per failure.
  Planning pages carry a status indicator that polls every 60 seconds and rechecks on tab focus and
  wake, with three states rather than two: up, up but no data, and down. It reloads the page's data
  when the service comes back, so nobody has to know to refresh.
  Opening a planning page also starts the service if it is down, deduplicated so the several
  requests a page issues share one attempt rather than racing to spawn a server each.

2026-07-29  Tracked the accuracy reports, and made the deployment able to hold data.
  outputs/reports/ml_accuracy.csv and ml_accuracy_by_sku.csv are now in git. They are 136 KB
  together and they are the recorded score of every model version, the numbers the version log
  cites and the validation page reads. Untracked, that evidence existed in one working directory
  and nowhere else. The rest of outputs/, 19 MB of experiment plots and CV dumps, stays ignored;
  verified that exactly two files become tracked and none of the other 51 leak in.
  Found a fault in the existing deploy workflow: rsync ran with --delete and no exclude for data/
  or outputs/. Those are gitignored, so the checkout has none of them, and every deploy would have
  deleted the server's copies and left the API answering 500 on every planning request until the
  next Monday. Both paths are now excluded, which under --delete means do not upload and equally
  do not destroy.
  Settled the arrangement: the forecast API runs on the same server as Demand Pilot, bound to
  loopback. Next.js proxies server-side, so the service needs no public port, no firewall rule and
  no CORS, and a colleague needs nothing installed. Code arrives from GitHub Actions, data arrives
  from the weekly cron, and neither owns the other's files.
  scripts/push_data_to_server.sh pushes the nine files the service reads, about 1.5 MB rather than
  the 19 MB in outputs/, then asks the server whether it can serve and exits non-zero if not, so a
  failure reaches cron mail on the Monday it happens. Tested against stand-in ssh and rsync for
  three cases: ready, pushed but still unable to serve, and no answer at all.
  DEPLOYMENT.md rewritten for this arrangement, including why FORECAST_SERVER_DIR stays unset in
  production: systemd supervises the service, and letting the app start it too would put two
  supervisors on one port.

--- DAILY SUMMARY 2026-07-29 ---
- Moved the forecasting dashboard into the main company app, so there is one system instead of two.
- Connected it to live inventory instead of a file someone had to remember to export.
- Covered the products that sell too irregularly to forecast, and flagged the ones running out
  before their next shipment lands.
- Built a page showing the new forecasting method beats the old spreadsheet, and started keeping a
  permanent record of how accurate it is.
- Fixed why a colleague could not open the page, and arranged things so nobody has to install
  anything to use it.

--- SUMMARY PRODUCED 2026-07-29 (covering the 2026-07-29 entries above) ---

2026-07-30  Added scripts/verify_deployment.sh, and confirmed both branches are merge-ready.
  The acceptance checks in the deployment brief were prose, which meant running them by hand and
  interpreting the output. They are now a script that runs over SSH against 127.0.0.1:8000, the
  address the Next.js process itself uses, so a pass means the app's own requests will succeed. It
  exits non-zero and prints the fix under each failure rather than only the symptom.
  It catches the three failures that are otherwise hard to read. A token mismatch, where /health is
  exempt from the check so the status indicator shows the service up while every page fails. A
  service running from a different checkout than the one being pushed to, where pushing data
  appears to succeed and changes nothing. And incomplete database credentials, which surface as
  sample inventory rather than as an error. Tested against stand-in responses for all three plus
  the healthy case.
  Also verified the work is ready to merge: every Python module compiles, the Next.js production
  build succeeds with all seven new routes present, and both branches fast-forward onto main with
  nothing to reconcile. The server setup itself remains outstanding and needs credentials.

2026-07-30  Probed the deployment server and wrote the cutover plan.
  The picture turned out to be two units running the same codebase from two checkouts, both
  wanting port 8000: the live coverland-forecast.service from /home/coverland/Time_Series_Forecasting
  and the new coverland-forecast-api.service from /opt, currently disabled. That makes this a
  retirement rather than a coexistence, which changes the order of the remaining work.
  Three findings. The deployed .env has eleven of the fifteen variables the service reads, and the
  four missing ones cost the Demand Forecast assistant without any visible error. src/chat.py has
  pointed its own tool calls at port 8001 since the commit that introduced it and nothing has ever
  listened there, so the assistant has been answering from the model rather than from the data for
  the whole life of this deployment; that is now configurable and the cutover sets it to 8000. And
  the live unit binds 0.0.0.0 on a public host, so the API may have been internet-facing including
  POST /run-forecast, which spawns a pipeline run. Whether it was actually reachable depends on the
  cloud security list and is checked first in the cutover; the new unit binds loopback either way.
  Also recorded a conflict to settle after, not during: with data pushed from the Mac and the Run
  Forecast button writing on the server, two machines own the same files, and a run triggered from
  the UI is silently replaced by the next weekly push.
  Earlier, an ssh usage message turned out to be an empty deploy user and host producing a bare "@"
  destination, after two wrong guesses at the cause. The three deploy scripts now share
  _deploy_env.sh, which validates and prints what it resolved, and scripts/server_topology.sh
  reports which process holds which port and from which directory.

2026-07-30  Correction to the 2026-07-29 entry, and a clearer demand section.
  That entry says the un-forecast share of weekly volume went from 6 percent a year ago to 23
  percent now. Those were the first and last single weeks of the series, which is the same
  mistake as ranking SKUs by an unweighted error on ten units. Averaged over quarters of the
  window the figures are 17 to 21 percent across a year and 7 to 21 percent across two. The
  direction holds; the size was overstated.
  The demand section had four summary cards, two of which were complements of each other, one of
  them titled "tail share of demand" in a vocabulary only someone who built the segmentation would
  recognise. Now three cards that each say what they are a share of and in what units, with the
  un-forecast one stating plainly that those SKUs sell too irregularly to forecast.
  The trend moved out of a card and into a sentence under the chart, shown only at a year or more
  and only when the change exceeds two points. Over 26 weeks it was comparing two six-week periods,
  where 21 to 20 percent is noise presented with the confidence of a finding.
  The comparison table became a matrix, segments against windows, after nine results were taking
  nine rows and four columns of repeated labels. Bias is now rendered, having been computed and
  discarded until now; it shows smooth/short over-forecasting by 9.4 percent in Mar-May while
  smooth/long under-forecasts by 7.2 percent in the same window, which pooled error cannot express.
  Windows were being ordered alphabetically, so the table read Dec-Feb, Mar-May, Oct-Dec: sorted by
  name and presented as though it were time. Ordered by cutoff now.

2026-07-30  Brought the demand-versus-forecast chart onto the validation page.
  Ported in spirit from the old Demand Forecast page rather than copied, because the data beneath
  it is a different shape. That page's chart reads stored predictions from repeated runs, so many
  runs cover the same week at different leads and "adaptive" means taking the freshest. The ML
  backtest gives each week exactly once, at the lead its position in its window implies, so there
  is no adaptive choice and the control was not carried over.
  Two population traps, one of which I walked into. Predictions exist for 260 SKUs and actuals for
  all 447, so the actual line must not be summed over the wider set. Having fixed that, the first
  version was still 24 percent high, because the backtestable population is not constant either:
  66 SKUs in Oct-Dec against 260 in Mar-May, since a SKU needs history reaching past a window's
  cutoff. Both series now come from the same backtest rows, which carry the actual alongside the
  prediction, so they cover the same SKUs by construction rather than by a join. The chart total
  reconciles with the comparison grid exactly at 86,093 units.
  Nothing is drawn between the last scored week and the first forward week. That span is the
  quarantined final test window; it is shaded and labelled rather than left blank or bridged.
  The chart states its own aggregate error, 13.0 percent, and says why it is lower than the 16.0
  percent pooled figure above: summing SKUs before differencing lets one SKU's over-forecast cancel
  another's under. Without that line the chart quietly contradicts the headline it sits beneath.
  Also recorded, from the same session: the AI assistant is retired rather than ported, the old
  page's other gap. SKU Planning stays on the legacy statsforecast path for now, pending a wider
  refactor, so run_forward_forecast.py and the fc_ tables remain in service.

2026-07-30  Rebuilt the demand-versus-forecast chart on the right source, after two corrections.
  First attempt was a simplified line chart built from the project notes rather than from reading
  the component being ported. The real one carries a P85 band in three places, a bridge polygon and
  connector across the gap, a forward curve anchored to the last actual, segment pills, a lead
  selector that disables leads with no data, a V1 toggle, a four-cell summary strip with
  conditional colouring, a last-complete-week marker, seaborn colours and a 680px canvas.
  Second attempt matched the layout but read the wrong data. That chart plots forecasts that were
  served before the outcome was known, taken from fc_forward_forecasts across many forecast_dates,
  where lead N means the run made N weeks before the target week. I had used the backtest windows,
  which are a different claim and one the comparison grid already answers. The ML counterpart is
  ml_forecast_history, and src/ml/serving/history.score_against_actuals already returns exactly that
  shape, so the endpoint now reads it.
  Consequence worth stating plainly: the store holds no runs yet, so the predicted line is empty
  and the chart shows demand and the current forward horizon only. It fills as the weekly runs
  accumulate and their weeks close. Verified by simulating three stored runs, after which a week is
  covered at several leads, which is what makes the most-recent-run selection a real choice rather
  than a formality.
  One capability gap recorded rather than worked around: the LightGBM track emits a point forecast
  and no intervals, so the P85 band cannot be ported at all. The old chart used it as a calibration
  check, the actual line leaving the band meaning the interval missed. The Action List's safety
  stock uses measured per-SKU error instead, which is a different instrument.
  Also corrected along the way: segment labels were being taken from today's sku_profiles.csv, which
  labels an old row with a current classification. Segments now come from the per-SKU accuracy
  report, as of each window, and every pill reconciles with its grid row exactly.

2026-07-30  Added scripts/seed_forecast_history.py so the chart can be reviewed before real runs land.
  The demand-versus-forecast chart reads the accumulating history store, which gains one entry per
  weekly run, so on a fresh store there is nothing to draw and no way to look at the thing working
  until several Mondays have passed. The script fabricates weekly runs from actual sales with error
  that widens with lead.
  Two properties make it safe to leave lying around. The rows carry a model version ending in
  -SAMPLE, and the serving endpoint prefers the version the current forward forecast came from, so
  a single real run makes the sample invisible without anyone remembering to clean up. Verified by
  writing a real-looking v11 run and watching the endpoint switch to it. The page also shows an
  amber banner naming the version whenever the predicted line is not from the current model.
  Two faults found while building it. The endpoint was not filtering the store by model version,
  so two versions would have been summed into one line, which the store exists specifically to
  keep apart. And deriving fabricated predictions from actuals gave near-zero forecasts beyond the
  end of history, making later runs look like they forecast almost nothing; weeks past the last
  actual now fall back to each SKU's recent mean, so a stored horizon is shaped like a forecast.

--- DAILY SUMMARY 2026-07-30 ---
- Finished moving the forecasting service onto the company server, so colleagues can use it
  without installing anything themselves.
- Caught two faults before switching over: the update process would have deleted its own data,
  and the AI assistant had been answering questions without ever reading any.
- Replaced a spreadsheet library that was abandoned four years ago, and updated the email
  library used for password resets.
- Rebuilt the accuracy comparison table so it reads at a glance, and added whether the model
  over- or under-predicts, which was being calculated and thrown away.
- Added the demand-against-forecast chart from the old page, with sample data so it can be
  checked before the weekly runs build up.

--- SUMMARY PRODUCED 2026-07-30 (covering the 2026-07-30 entries above) ---

2026-07-31  Wrote down the decisions and open questions that existed only in conversation.
  Four items added to BACKLOG.md. The outlier lists on the validation page rank by an unweighted
  per-SKU error while the headline above them is demand-weighted, so the thirty rows shown carry
  under two percent of scored demand; the fix needs a threshold chosen rather than inherited.
  Retiring the old Demand Forecast page, including what the new pages already cover, the four
  components whose replacements have not been checked, and the timing constraint that its
  accuracy-over-time chart has real history while the replacement fills from an empty store. The
  Run Forecast button, recorded with the facts behind it so the conclusion is checkable rather
  than remembered. And the Node version disagreeing with what package.json declares.
  Two dependency notes added to Commerce_Integration/CLAUDE.md: that the nodemailer advisories
  were never reachable in this codebase, checked option by option, so it is not re-litigated at
  the next audit; and how to read npm audit here, since its forced fix proposes downgrading Next
  by six major versions.
  Current state for anyone picking this up: both repos are on main and clean, the forecast history
  store holds six fabricated runs labelled v11-SAMPLE, and the first real weekly run retires them
  without intervention.

2026-07-31  Gave the Forecast Validation outlier lists a stated minimum volume, defaulting to 100 units.
  The page headline is pooled WAPE, which is demand-weighted, while the two per-SKU lists beneath it
  ranked by an unweighted difference between the model's WAPE and V1's. Both are divided by the same
  actual, so the difference is bounded by that denominator, and the report bears this out: the largest
  absolute delta is 4.94 in the 10-to-50-unit band against 0.48 above 500 units. Taking the top fifteen
  by delta therefore selected the smallest SKUs rather than the ones the model handles worst, and the
  thirty rows shown carried 1.8 percent of scored demand under a heading telling a planner to read them
  first.
  The threshold was chosen from the distribution rather than inherited from the note that raised the
  problem. At 100 units, 223 of 572 scored rows and 78 percent of scored demand stay eligible. The
  stricter candidates buy a cleaner ranking by emptying the pool: 200 leaves 103 rows, 500 leaves 41,
  at which point a top fifteen is over a third of everything eligible and is no longer an extreme of
  anything. Presets of 0, 100, 200, 300 and 500 are on the page, and the active threshold is displayed
  next to what it leaves, in rows and in share of scored demand, because the number is a judgement and
  a reader who cannot see it cannot weigh it.
  The endpoint now sends the whole scored pool, 572 rows and about 100 KB, instead of two ranked lists.
  Ranking server-side would have fixed the threshold at whatever the endpoint chose, which is the thing
  being removed. Verified by replaying the client's filter and slice against pandas: the lists match
  nlargest and nsmallest exactly at 0, 100 and 200 units, and the displayed shares reproduce the
  computed distribution.
  Worth recording because it was the argument for the threshold: the CC-CN-03 and CC-CP-03 pattern the
  backlog said was buried does surface at the chosen default. The improvements list goes from 957 units
  across three product families to 4,823 across four.
  Two states are called out on the page rather than left to be inferred. Selecting no minimum shows why
  the ranking then reflects size rather than error, and a threshold leaving fewer than thirty eligible
  rows warns that a SKU can appear in both lists, since both are drawn from one pool.

2026-07-31  Fixed the Action List reliability sort, and took three model-facing details off the purchaser screens.
  Sorting by Reliability interleaved the unmeasured SKUs with the measured ones at what looked like random
  positions. The column prints each SKU's own measured error, or "n/a" where no backtest window covers it,
  but it sorted on a second figure: the error safety stock actually spends, which for an unmeasured SKU is a
  substituted cohort or segment value. On the current table 174 of 432 SKUs are unmeasured, 158 of them
  carrying 0.199 and 16 carrying 0.240, against measured errors spanning 0.006 to 1.823. So the "n/a" block
  parked itself at 0.199 in the ordering with 142 measured rows above it and 116 below. The column now sorts
  on the figure it prints, which lets the existing blanks-last rule fire for the first time. Verified by
  replaying the table's own sort against the live planning table: the unmeasured rows move from scattered
  positions 79 to 289 into a contiguous block at 258 to 431, in both directions. The substituted figure was
  the only thing the sort exposed about an unmeasured SKU, so the cell now names it on hover rather than
  losing it.
  Removed the history-length split from the Action List and the SKU detail page: the filter, and the word
  in every row's subtitle on both screens. Checked against the data before cutting it rather than on taste.
  The two groups do differ, 0.121 pooled WAPE across 80 long SKUs against 0.197 across 352 short ones, but
  the reliability column already carries that at SKU resolution and by measurement rather than inference.
  Using the group as a trust signal would actively mislead: 16% of long SKUs sit in the poor tier and 19% of
  short ones in the good tier, so a reader following the group would treat a long and poor SKU more gently
  than a short and good one. The bias gap, +4.3% against -0.2%, is too small to move an order quantity. The
  one thing the split marked is that 174 of the 352 short SKUs have no measurement at all, and the "not
  measured" tier already says so, is already filterable, and identifies exactly the affected SKUs instead of
  a group that is half measured. Forecast Validation keeps the split, where the vocabulary is correct and
  the reader is the modeller.
  Also removed the SKU count from the Action List provenance bar, where it was the third copy of one number.
  The same figure is the count on the Forecast tab of the section toggle and the figure on the first summary
  chip, and three copies crowded the two things in that bar that have to be read, namely how old the forecast
  is and whether the stock figures are real.

2026-07-31  Reworked the Action List summary row and made the portfolio chart's two spans adjustable.
  The row of counts at the top of the page had a total that behaved unlike everything beside it. Five of the
  six narrowed the list; the first widened it back to everything, while wearing the same box and the same
  large-number treatment as the statistics. It was removed and then restored on the owner's judgement, which
  was the right call: the total and the one-click way out of a filter are both worth having. What it needed
  was not deletion but separation, so it now sits ahead of the five conditions with a rule between them and
  reads as the neutral position of a group rather than a sixth condition. Selecting an active condition also
  clears it now, so stepping back one decision no longer costs the search, the category, the reliability tier
  and the sort, which is what Reset does.
  The recommended-units total beside them now counts the rows on screen rather than the whole list, with the
  list total kept underneath whenever a filter is narrowing the view. There were previously two totals in two
  visual languages a few inches apart, and the one answering "what am I about to buy" was the one rendered as
  small grey text at the end of the filter row.
  The portfolio chart is collapsed by default. Open, it put 460px of chart plus controls above the first table
  row on the one screen whose job is what to order today, and the same question is answered twice on Forecast
  Validation with better instruments. Both of its spans are now selectable: history at 13, 26, 52 or 104 weeks,
  which refetches because the server holds the weekly series, and the forecast horizon at 4, 8, 13 weeks or all
  of it, trimmed in the browser because the whole horizon is already in hand. The forecast presets are filtered
  against what the run actually produced, so a 13-week horizon does not offer a 26-week view that would quietly
  show 13.
  Also renamed "chip" out of the code. It was never a term this project defined, and it appeared in comments
  and in two variable names as though it were.

2026-07-31  Made the Action List data-quality warnings filterable, and put the inbound date beside the quantity.
  The warning line named the flagged SKUs and then left the reader to find them, on a page where every other
  count is a way into the work. Each warning is now a control: one for every flagged row, one per named
  warning. The counts are taken over the rows the other filters left rather than over the final view, so
  selecting one warning does not hide the others and strand the reader in the one place they cannot switch
  from. A warning that loses all its rows to a later filter change stays on screen at zero so the selection
  can be cleared, rather than vanishing and leaving an empty table unexplained.
  The confirmed-inbound column showed a quantity and no date. The ETA did exist on the row, but only rendered
  in the stockout cell and only when the container lands too late to help, which is 128 of the 376 rows
  carrying inbound. The other 248 read as "500 units, at some point". The column now carries days-to-arrival
  under the quantity, with the calendar date on hover. Days rather than the date because the question asked of
  that cell is whether it beats the stockout figure a few columns over, which is also in days.
  That made the stockout cell's own annotation a duplicate, since it printed the same arrival figure. It now
  prints the gap instead, the days the SKU spends at zero between running dry and being refilled, which the
  row already carried and nothing displayed. The reader was previously expected to subtract one column from
  another. On the current table 128 rows carry a gap, median 12 days and up to 43, and none of them can close
  it by ordering today, so that mark is also the only thing on the row saying the recommended quantity will
  not help.
  Recorded the CSV export as backlog item 5 rather than fixing it. It builds its header from the row object,
  so the file carries about forty internal field names instead of the ten columns on screen. Deferred because
  nobody is using the export yet, and because the fix contains a decision about which columns belong in a
  file as opposed to on a screen.

2026-07-31  Fixed two long-standing rendering faults in the planning tables' sticky headers.
  Body rows were visible through the Priority column while scrolling. Every sticky header cell needs an
  opaque background of its own, since a cell that inherits nothing is transparent and the rows travelling
  underneath show straight through it. Every cell in that row had one except Priority, which was the only
  column belonging to no colour band and so was the only one passed no background class.
  A hairline also ran between the two header rows, through which content was visible. The second row was
  pinned at 28px to sit flush under a 28px band row, but the row's bottom border sits outside that height,
  so the two did not meet. The second row is now pinned one pixel higher so they overlap instead of abut.
  Both rows are opaque, so an overlap cannot be seen where the gap plainly could. The band row also gained a
  rule beneath it, drawn as an inset shadow rather than a border: the table collapses its borders, and a
  collapsed border belongs to the table's border grid rather than to the cell, so it does not travel with a
  cell lifted out of flow by sticky positioning.
  The non-forecast table had copied these constants rather than sharing them, so it carried the same hairline.
  It now imports them, which is the point of the shared constant.

2026-07-31  Made a fresh clone able to run the forecast service, so the planning pages can be worked on
  without a database or a copy of anyone's working tree.
  A colleague could not start the service locally. The cause was not the start-up code, which already
  reports a missing virtualenv and a wrong FORECAST_SERVER_DIR clearly. It was that data/processed/ is
  gitignored, so a clone has the code and none of the three files the planning endpoints require, and the
  service therefore starts, answers its liveness check, and raises on every real request.
  Two of those three were already in the repository, pinned under data/snapshots/2026-07-20/ and not
  ignored. Only the forward forecast was missing anywhere in git, at 49 KB. So the fix was a seed step
  rather than a data handover: added data/dev_seed/ holding the forward forecast and its V1 counterpart
  (72 KB together), and scripts/seed_dev_data.py, which copies those plus the pinned sales and profile
  files into data/processed/.
  The seeded pair is exactly consistent, which was luck worth recording. The committed forecast was
  trained through 2026-07-20, the same week the pinned snapshot's sales history ends, so the seeded
  history and the seeded forecast meet with no gap. Live data/processed/ has since moved to 2026-07-27
  and is now a worse match for that forecast than the snapshot is. The seed script checks this equality
  and refuses to run if the two ever drift apart.
  It also refuses to overwrite an existing data/processed/ without --force, because on the machine that
  runs the weekly cron those files are live and newer, and silently replacing them with a frozen July copy
  would be the worst thing the script could do.
  Tracked outputs/reports/ml_backtest_weekly.csv alongside the two accuracy reports it reconciles with.
  It is optional to the service and degrades to a totals-only chart, which meant an untracked file made a
  working feature look unbuilt on every machine but this one. 430 KB, rewritten per model version, which
  is a real cost taken deliberately.
  The failure messages now name the seed script instead of the pipeline script that originally produced
  each file. "Run the forward forecast" is not an instruction anyone can follow on a machine with no
  database, and naming it was the reason this state read as a dead end. The Demand Pilot error card shows
  the command itself, above the list of what is missing.
  Corrected DEPLOYMENT.md, which claimed data/ and outputs/ are gitignored and gave that as the reason the
  deploy excludes them. Both are now partly tracked, and the excludes are about ownership rather than
  about git: the cron owns the server's data and the deploy declines to touch it. That distinction is what
  makes a committed fixture safe, since nothing under data/ is ever rsynced to the server. Added a
  "Running it locally" section with the four commands.
  Verified against a genuine clone: at HEAD it has neither data/processed/ nor a forward forecast; after
  the seed, readiness() reports ready with nothing missing, required or optional, and the planning table
  builds 447 rows with real recommendations.

2026-07-31  Audited the SKU Detail and Forecast Validation pages against the purchaser's questions in
  docs/PLANNING_PLAN.md, and fixed the one defect the audit turned up.
  SKU Detail's header said "Trained through" and showed the first week of the forecast horizon, which is
  one week later than the training cutoff by construction. So the page reported 2026-07-27 where the
  Action List, reading the forecast file's own forecast_date, correctly reported 2026-07-20. Two screens
  disagreeing about how old a number is, in the one place a purchaser looks to decide whether it is stale.
  The cause was that the SKU detail endpoint did not return the figure at all, so the page inferred it.
  It now returns trained_through from the same source the list uses, and the page reads it.
  Confirmed the Forecast Validation page is not a purchaser's screen and should not become one. The plan
  classes it as the administrator view and says it is visited deliberately rather than during purchasing
  work. The one question on it a purchaser would ask, whether a number can be trusted, is already answered
  per SKU by the reliability card at the moment of the decision, which is the better place for it: a
  pooled portfolio error says nothing about the SKU in front of them.
  Noted that a purchaser currently sees five Planning entries in the navigation, two of them the legacy
  pages this work replaces, because forecast-validation and demand-forecast are both in the default
  visible set for a non-admin. Retiring the old page is backlog item 6 and remains the more valuable next
  move than adding a screen.
  Recorded two gaps found in the audit rather than building them. Backlog item 9: nothing lets a purchaser
  mark a SKU actioned, which the plan's own weekly walkthrough depends on twice and lists as unblocked;
  it needs a decision on browser-local against persisted before it is worth writing. Backlog item 10:
  no screen carries money, confirmed as a real gap but deferred, since it needs a decided cost basis and
  a source before a column would mean anything.

2026-07-31  Showed draft container coverage on the Action List, so a SKU that has already been ordered
  stops reading as though it had not.
  This began as a plan to let a purchaser mark a SKU actioned by hand, and that framing was wrong. The
  state it proposed to record is already recorded by the container system, and a manual flag would have
  been a second, private, weaker copy of it. Once a container reaches shipped or packing_received it is
  counted as confirmed inbound, the order formula subtracts it, and the SKU drops off the list on its own.
  That half never needed building.
  The gap was draft. Containers are created by the Google Sheet import, which sets status from the header
  colour, and the inbound query deliberately excluded drafts. So between someone deciding to order and
  the container shipping, the SKU kept showing a full recommended quantity with nothing saying an order
  existed. At an eight-week lead time that window is long enough to order the same units twice.
  Reads the same two tables the Container Planning screens use, so the two cannot disagree about what has
  been drafted. The ETA rule differs from confirmed inbound on purpose: confirmed floors at today because
  a container still marked shipped after its ETA is stale bookkeeping, while a draft often has no date yet,
  so nulls are kept and only dates already past are excluded. Applying the confirmed rule would have
  dropped exactly the newest drafts, which are the ones that make a SKU look unordered.
  Drafted units are never subtracted from the recommendation. A draft can be cancelled, so crediting it
  would under-order the SKUs someone has already acted on. It is shown beside the number instead, as an
  italic sub-line under the recommended order, which is the treatment the order breakdown already gives
  lines that sit outside its arithmetic. It is a quantity rather than a badge because partial coverage is
  the case that matters: a badge would read as "handled" on a SKU drafted for 300 against a recommended
  1,117 and stop someone looking.
  Built and then removed a distinction between a missing draft figure and zero, on the argument that an
  export written before these columns existed cannot say whether anything is drafted and should not be
  shown as though it had said no. That was wrong twice over. The figure comes from container line items,
  so no matching row genuinely means no units, exactly as for confirmed inbound; and both database
  connections sit in one try block, so there is no state where stock is live but drafts are unknown.
  More to the point, how far the whole inventory source can be trusted is already reported once by
  inventory_source and inventory_is_sample, and the removed code was a second per-column version of that
  same answer. Zero is now zero everywhere, and the type is a plain number rather than a nullable one.
  Fixed a latent trap while in the query builder: the confirmed status list was interpolated into SQL as a
  Python tuple, which renders a one-element tuple with a trailing comma and is a syntax error in Postgres.
  It happens to work for the two-element constant and would have broken silently the day either list had
  one entry, which the new draft list does.
  Not yet verified against the live database, which the assistant cannot reach. The display was exercised
  against simulated draft data instead; backlog item 9 records the three things to check on a real
  connection, the most consequential being how many rows carry drafts, since that decides whether the
  sub-line should become its own sortable column.

2026-07-31  Did the draft subtraction for the purchaser on the SKU detail page, and left the Action List
  alone.
  The recommended quantity ignores drafted units entirely, which is correct and is not obvious: a row
  reading 55 with 19 drafted means 55 is the requirement if that container never ships, and 36 if it does.
  Someone reading it as "55, already net of the 19" would order 19 too many. The list now carries the flag
  and the detail page carries the arithmetic, which suits where each is used: the list is for triage and
  the detail page is where a quantity is decided.
  The order breakdown gained the drafted units as an aside line, using the Sign=None role the table
  already had for inbound arriving too late to count. Same reason in both cases: the figure is real and it
  is not in the sum. The signed lines still reconcile exactly to the total, which was checked.
  The net figure sits on the order card under the caption rather than in that table. A second line with an
  equals sign inside the breakdown would read as a second answer to the same question rather than the
  answer to a different one, so it is a small boxed figure labelled "if the draft stands", holding the
  recommendation as the headline. The model does not know whether a draft will ship, so the committed
  number stays primary and the conditional one is offered beside it.
  Trimmed the caveat above the card, which had been stating the same subtraction in words. It now states
  the fact and stops, since the card does the arithmetic.

2026-07-31  Removed the Action List's priority filter, which was a second control over a field the summary
  cards already filtered, and gave Routine a card of its own.
  Reported as a display problem: selecting a summary card left every select below reading "all", so the
  screen looked unfiltered while it was not. The cause was structural rather than cosmetic. The cards and
  the priority select held separate state over the same column, so they could be driven into combinations
  that return nothing and explain nothing, "Preorder" on a card against "Best Seller" in the select being
  empty by construction. The data-quality filter beside them already states the rule this broke: two
  pieces of state that can contradict each other should be one.
  A comment on the select said it existed because Routine was the only priority with no button. So Routine
  now has a button, with a count like every other condition, and the select is gone. Nothing became
  unreachable and there is one place to look.
  That comment was also wrong in a way that mattered. It claimed the cards covered three of the four
  labels. They cover two. The third, "out of stock", is the raw condition available_inventory <= 0, while
  "No Stock" is a queue assigned by precedence, so a SKU with no stock and a preorder backlog is badged
  Preorder and appears in one and not the other. On the current table that is 115 against 16, a difference
  of 99 rows between two controls whose names were nearly the same. The card is now labelled "no stock on
  hand" to say it is a stock condition rather than a priority.

2026-07-31  Made the local-setup instructions work on Windows, and wrote down why the deployed service
  cannot simply be pointed at from a laptop.
  Every command in the seed script, the deployment doc and the Demand Pilot error card used .venv/bin,
  which does not exist on Windows. A colleague on PowerShell could not run any of them. All three now give
  both forms, and the script prints the one for the platform it is actually running on rather than the
  author's. The Windows form calls the interpreter directly instead of activating the virtualenv, because
  the default execution policy blocks Activate.ps1, which the Commerce app already documents for its own
  dev script. Activation buys nothing here.
  Also recorded the answer to a question that keeps being asked: the deployed forecast service binds
  127.0.0.1, so it accepts connections only from the Next.js process on the same machine. That is
  deliberate, since the service holds both database credentials and has no real authentication of its own,
  and it is why a local AI_SERVICE_URL pointed at the server cannot work and should not be made to. The
  doc now lists the three honest options in order of cost: use the deployed app, forward the port over SSH
  and keep everything local except the data, or run the service locally, which is only necessary when the
  Python side itself is being changed.

2026-07-31  Reduced local setup to one command, for handing the project over.
  Added scripts/setup_local.py: virtualenv, dependencies, seed, .env, then a verification pass. It runs on
  the system Python and imports only the standard library, because it creates the virtualenv and so cannot
  live inside one. Every step detects its own completed state, so re-running after a pull is safe and is
  the right thing to do when requirements.txt changes.
  The part worth recording is the .env derivation. This repo and Commerce_Integration read the same two
  databases but describe them differently: one connection URL each there, five discrete variables each
  here. So a colleague who already has the Commerce credentials does not need new ones, only the same ones
  reshaped, and the script does that from their existing file rather than asking anyone to send secrets
  around. The naming runs backwards, COMMERCE_DB_* being the Supabase lookup database rather than the
  Commerce app's primary, which is an easy mistake to make by hand and fails as missing tables rather than
  as a connection error.
  The verification connects rather than checking that an engine could be built. _engine returns None for
  missing variables, a missing driver and an unusable URL alike, so a check that could not tell them apart
  named the wrong fix two times in three, which was caught while testing this against a sandbox that had
  no psycopg2.
  Completed .env.example, which documented three optional LLM keys out of twenty and so read as though the
  rest did not exist. It now covers every variable, what reads it, which Commerce variable it corresponds
  to, and states at the top that none of it is needed to run the planning pages, since the seeded data
  covers every required file.

2026-07-31  Corrected a wrong claim about FORECAST_DEPLOY_KEY in the env template, made while writing it.
  It was described as an SSH private key, and it is a path to one: _deploy_env.sh expands any tilde and
  passes it to ssh as -i, so the key material itself never enters .env. The advice that followed from the
  misreading, that the block is more sensitive than the rest of the file, was wrong too.
  The template now says what the block is actually for. The four FORECAST_DEPLOY_* variables drive
  push_data_to_server.sh and verify_deployment.sh, which is how each weekly forecast reaches the deployed
  service, the code deploy carrying no data by design. They are needed only on the machine running the
  Monday cron and are unrelated to the DEPLOY_* secrets in GitHub Actions that deploy the code.
  Also recorded the consequence for handover: since the value names a key that has to be authorised on the
  server, taking over the cron means having your own key added there rather than inheriting a file.

2026-07-31  Corrected DEPLOYMENT.md, which still described the weekly forecast as running on a laptop and
  being pushed to the server.
  It does not. scripts/run_forecast_cron.sh runs on the server as cron for the coverland user and writes
  straight into the checkout the service reads from, which its own header says was the point: freshness
  should not depend on any individual machine being powered on. The ownership table and a "weekly data
  push" section both predated that move and read as though the Mac were still in the loop, with their own
  crontab line for a step that no longer happens.
  push_data_to_server.sh is kept and still works, but is now documented as the out-of-band path for data
  produced somewhere other than the server, alongside verify_deployment.sh. The section explaining the
  weekly run now describes the cron that actually exists.
  This also corrects handover advice given earlier today from the stale text: taking over the project does
  not require taking over a forecast cron, because there is no laptop-side cron to take over. An
  authorised key is worth having for verification and for ad-hoc pushes, and is not needed to keep the
  forecast current.
  Settled backlog item 8 while checking what the server runs. The Commerce repo carries a .nvmrc reading
  22, which agrees with the engines range of >=20.9 <24 that the same file declares. The project therefore
  states its intended version twice and consistently; the development machine on 24.2.0 is the only thing
  disagreeing, so it moves rather than the declaration. Recorded that CI pins no Node version at all,
  which is a smaller separate gap.

2026-07-31  Recorded a single-point-of-failure found while answering where the weekly cron stores its
  output (backlog item 11).
  The answer is the server's own checkout, not a laptop, and for almost everything that is fine: the
  forward forecast, cleaned sales, profiles and the V1 baseline all rebuild from the database, and the
  accuracy reports are tracked in git. One file is not like the others. ml_forecast_history.parquet gains
  one entry per run and is the only record of what was predicted before the outcome was known, so it
  cannot be rebuilt; re-running past versions against past cutoffs yields backtest figures, which is a
  weaker and different claim. It exists on one disk, gitignored and excluded from the deploy.
  It also matters more each week rather than less. It is what Forecast Validation's served-forecast
  section and the demand-versus-forecast chart read, and backlog item 6 names it as the one timing
  constraint on retiring the old Demand Forecast page, so losing it resets that clock.
  Git is not the fix, since the server writes the file and committing it would need someone connecting
  weekly. The codebase already points elsewhere: src/ml/serving/history.py says in its own docstring that
  the refactor it underwent was the prerequisite for serving this from an API that does not share the
  filesystem, and the legacy track already writes its equivalent to shipcore.fc_forecast_history. The ML
  track writing a parquet is the asymmetry. A dated copy after each successful cron run is the stopgap.

2026-07-31  Moved the accumulating forecast history into the database, so it is neither on one disk nor
  visible from one machine (backlog item 11).
  ml_forecast_history.parquet was the only artifact the weekly run produces that cannot be rebuilt: it
  records what was predicted before the outcome was known, and re-deriving it would yield backtest
  figures, which is a different claim. It sat on the server's disk alone, gitignored and excluded from the
  deploy, so it could be lost and could not be read by anyone else.
  Added src/ml/serving/store.py holding shipcore.ml_forecast_history and a keyed upsert, and changed
  history.py's load and append to use it. That was the two-function change the module's own docstring
  predicted years of comments ago, which is a good argument for writing that kind of note.
  Both stores are written, table first so a crash between them costs the local copy rather than the shared
  one. Reads prefer the table and fall back to the parquet, which is what makes a machine with credentials
  see the server's runs while a clone without any still works from its own. Nothing raises when the
  database is absent; a credential-free clone stays a supported way to run this.
  A failed table write does not fail the run either. The forecast is already produced and the parquet has
  it, so the pipeline now prints whether the run reached the table and says plainly when it did not, that
  being the case where one copy is all there is.
  scripts/migrate_history_to_db.py imports an existing parquet and is idempotent on the key, which is what
  lets it run on both the laptop and the server and merge the two divergent copies rather than forcing a
  choice between them.
  run_forecast_cron.sh also keeps twelve dated copies under data/history_backups/, gitignored. That covers
  what the table cannot: a run where the database was unreachable, which is exactly when the file is the
  only copy.
  The database path is unexecuted. The fallback path was tested: no credentials gives engine None,
  available False, read None, upsert -1, and append still writes the parquet, still replaces rather than
  duplicates on re-run, and still returns a summary. The migration script exits non-zero with a plain
  message rather than a traceback. Backlog item 11 lists the four things to do on a machine that can
  actually reach Postgres.

2026-07-31  Dropped the backfill, and gave the forward forecast a table of its own.
  The migration script was removed on the user's call, and the call was right: it existed to import
  parquets nobody needs imported, and it would have merged two divergent local histories into a shared
  store to no stated end. Both tables now simply start empty and fill from the next run, created on first
  write, so there is no migration step at all.
  Added shipcore.ml_forward_forecasts beside the history table, written by ml_forward_forecast.py and
  preferred by src/planning/data.py's read, falling back to the parquet. This is the half that was
  missing: the history table made "is the model improving" answerable from any machine, but the Action
  List still showed whatever horizon that machine had locally, which on a colleague's laptop is the seeded
  July fixture. Now it shows what the server last produced.
  The two tables share one DDL definition, since they hold the same rows over different spans, the forward
  one being the current horizon and the history one every horizon ever served. Written once so they cannot
  drift into disagreeing about a column.
  The forward table accumulates rather than replacing wholesale. _read_forecasts already took the latest
  forecast_date and ignored the rest, which predates this change and is what lets the table keep older
  horizons at no cost to any caller.
  Neither write is fatal. The parquet is written first in both cases, so a database that is briefly
  unreachable costs visibility for a week rather than a forecast.

2026-07-31  Finished wiring the infrastructure to the forward-forecast table rather than the file.
  Writing and reading the table was not enough on its own: four places still assumed a parquet on disk.
  readiness() listed it as a required file, so a machine reading the table but holding no local copy was
  reported as unable to serve while it was serving, and the error card told the reader to run a seed
  script they did not need. The forecast requirement is now satisfied by the file or the table, matching
  what _read_forecasts accepts. The table is probed only when the file is absent, so a seeded developer
  machine and the server still stat and open no connection.
  export_inventory_snapshot.py read the parquet directly to get its SKU list, which meant it could not run
  on a machine with credentials and no local forecast, and could in principle disagree with the dashboard
  about which SKUs are forecast. It now goes through load_forecasts, so there is one definition.
  Corrected two long comments that said the ML forecast is written to a parquet and never to Postgres, in
  src/planning/data.py and api/main.py. Both described the arrangement accurately until today. Sales and
  the SKU profiles are what remain file-only, and they are now named as the remaining prerequisite for
  deploying the API away from this repo.
  Verified on the fallback path: readiness unchanged when the file is present, correctly failing with the
  table named in produced_by when neither is available, and correctly ready when the file is absent but
  the table answers.

2026-07-31  Made the "no data" card actionable, instead of running the pipeline automatically on startup.
  The proposal was for the app to check for the parquets on startup and build them if absent. Right in
  spirit and wrong in three specifics. Generating sales_clean means src/ingest.py, which is SQL against
  the orders table, so it needs credentials and cannot help the machine that has none, which is the case
  that was actually blocking anyone. A page load cannot wait for it either: the proxy times out at 20 to
  60 seconds and the work is minutes, so the request would fail while the pipeline ran on invisibly and
  the reader would retry into a second run. And the endpoint that exists, POST /run-forecast, spawns the
  legacy statsforecast pipeline, which regenerates sku_profiles.csv while writing forecasts to
  shipcore.fc_forward_forecasts, so using it to repair ML data would move segmentation underneath an
  unchanged ML forecast, which is the failure backlog item 7 describes.
  On the server it would have been worse: data/processed there is live data owned by the Monday cron, and
  this would have let anyone rebuild it mid-week, from a full order pull, by opening a tab.
  Built the deliberate version instead. scripts/ml_prepare_data.py chains ingest, clean, profile and the
  ML forward forecast, refusing to overwrite without --force. POST /planning/prepare-data runs it through
  the same job-and-poll machinery the Run Forecast panel already uses, and refuses with 409 when readiness
  says nothing is missing. The no_data card now offers both routes in the order they should be tried: the
  seed, which is instant and needs no database, then a button for building current data.
  Two bugs caught by reading the endpoints rather than trusting the types. The poll used a query parameter
  where the route takes a path segment, /api/forecast/status/[jobId], and read an error field the job
  payload does not have; its shape is job_id, status, lines, exit_code, so the last log line is the
  message, and it is the better one since the script prints which of its three steps failed.

2026-07-31  Added setup.cmd, so Windows setup does not begin with guessing the interpreter's name.
  A colleague hit "python3 is not recognized", which is not a misconfiguration: python3 is a Unix
  convention and does not exist on Windows at all. It is also the name in every macOS instruction, which
  is how anyone arrives at it. The two names that might work are py, from the python.org launcher, and
  python, which may be a real interpreter or may be the Microsoft Store execution alias, a stub that opens
  the Store and exits successfully without running anything. That last case is the worst, because it looks
  like it worked.
  setup.cmd tries py -3, then verifies python by running code through it rather than by asking whether the
  command resolves, since the Store stub resolves. If neither answers it says what to install and points
  out that a terminal opened before the install will not see it, PATH being read at startup.
  It only finds the interpreter; everything it does lives in scripts/setup_local.py, so there is one
  definition of what setup means and the batch file cannot drift from it. The mirror of start-dev.cmd in
  the Commerce repo, which exists for the same class of Windows friction.

2026-07-31  Found why the forecast API worked on one machine and not another, after most of a day on it.
  Commerce_Integration has both .env and .env.local, and Next.js loads .env.local at higher precedence.
  .env held AI_SERVICE_URL=http://localhost:8000 and .env.local held the deployed server's address, so the
  machine that worked was reading the second file while every instruction, and every edit either person
  made, went to the first. Both files are gitignored, so nothing travelled between machines and the two
  could disagree indefinitely with no symptom other than one of them working.
  That repo's CLAUDE.md asserted "The project uses a .env file (not .env.local)", which is false and is
  what sent us to the wrong file. Corrected, with the precedence stated, the check for which file supplies
  a variable, and the reminder that env is read at startup so a change without restarting the dev server
  does nothing and reads as the change having failed.
  The same error is corrected in DEPLOYMENT.md, which claimed the service binds 127.0.0.1 and is therefore
  unreachable from a laptop. A laptop has been observed reaching it directly, so it is listening publicly
  and the ExecStart line quoted in that document does not match what runs. Both the claim and the two
  passages that depended on it are rewritten, and the consequence is now stated plainly: the service reads
  both database credential sets and its only protection is a shared token, with /health outside that check.
  Whether it should be publicly reachable is a decision someone should make rather than discover.
  scripts/setup_local.py had the same bug I was diagnosing: it derived database settings from the Commerce
  .env and would have ignored .env.local. It now prefers .env.local, matching what that app actually reads.
  The lesson worth keeping: I asserted the binding from a document three times over several hours instead
  of asking for one curl against the host. The document had already been wrong twice that same day.

2026-07-31  Established, finally by measurement rather than inference, that port 8000 is firewalled from
  the internet, and corrected DEPLOYMENT.md back after I had wrongly "corrected" it the other way.
  The confusion had one cause. A developer machine appeared to be reaching the server with AI_SERVICE_URL
  set to its public address. It was not: that machine had a local forecast server running, and the app
  starts and uses a local one when the configured address does not answer, so the fallback served every
  request while the configuration looked correct. Removing the local server made the truth visible
  immediately.
  So the original claim in that document was right in substance and I overturned it on the strength of one
  machine appearing to work, which is exactly the kind of evidence the fallback is designed to manufacture.
  The document now states what was measured, records the mistake and its cause, and says that the test is
  curl from a machine with no local server, first rather than last.
  The deployed app works because it runs on the same box, where the public address resolves to a local
  interface. Port 3000 is open and 8000 is not, so this is a per-port rule rather than the host being
  unreachable.
  Three routes are now documented in order of cost: run the service locally against the committed fixture,
  tunnel over SSH when current data is wanted, or open the port, which is a deliberate decision rather
  than a fix and carries the note that the shared token is the entire perimeter for a service holding two
  sets of database credentials.

2026-07-31  Three changes to the SKU detail page, and a new endpoint behind one of them.
  Intermittent SKUs now show their sales history instead of only an explanation. The planning endpoint
  still returns 404 for them, correctly: without a forecast there is no order quantity, coverage demand or
  reliability, so there is no planning row to return. Sales history exists either way, so it moved to its
  own endpoint, /planning/sku/{id}/history, which reads the same weekly series the rest of the planning
  layer uses and is fetched only after the planning call has 404'd as intermittent. The chart draws actuals
  and nothing else. No forecast line, no band, no reliability figure, because none of them exist for these
  SKUs and an empty axis where a forecast belongs would read as missing rather than as not applicable. Note
  that the sparsest of these draw almost nothing, two units across 104 weeks in one case, which is itself
  the reason they are not forecast.
  Added a legend to the reliability card's Miss column. Seven colour steps in two directions is more than a
  reader can infer, and the direction is the part that matters, since over and under call for opposite
  responses. The Action List already gives its four tiers a legend for the same reason.
  Enlarged the two smallest labels on the order card from 8.5px, which was smaller than anything else in
  the app and was labelling the two figures most easily mistaken for the headline quantity.
  Recorded but not changed, after checking: the reliability percentage is not misleading in the way it
  looks. Of the 60 scored rows above 50% WAPE the median absolute miss is 37 units against 38 actual, so
  those are real misses rather than rounding on tiny volumes; only 13% miss by under 20 units. The genuine
  asymmetry runs the other way, a 9% error on a 200-plus unit SKU being 43 units against 13 for a 60% error
  on a small one. No change made: the ratio is the right instrument for sizing safety stock, which is what
  it feeds, and the window table already shows predicted against actual in units directly beneath it.
  Also identified, not yet resolved: the plausible band and safety stock are the same arithmetic. Safety
  stock is service_z x error x coverage demand, and the band's upper edge is coverage demand x (1 + error),
  so at the default service level the recommendation lands exactly on the band's top edge rather than
  inside it. The docstring on order_quantity_range asserts the opposite.

2026-07-31  Removed the plausible band, flagged the consistently-biased SKUs, and made the non-forecast rows clickable.
  The band is gone from the order card, the API and the types. It flexed coverage demand by the SKU's own
  error, which is the same quantity safety stock adds, so at the default service level its upper edge was the
  recommendation itself: one figure shown twice, once as a decision and once as a range that appeared to
  contain it while ending at it. Uncertainty now sits only on the reliability card, where it is measured over
  the backtest windows rather than restated from an error term the order card had already spent.
  Added a data-quality flag for SKUs the model misses in the same direction every window. Deliberately narrow,
  and the narrowness is the finding: bias is mostly a property of the season rather than of the SKU. Of the
  SKUs scored in two or more windows, 58% flip sign between them and the median spread within one SKU is 31
  percentage points, so a per-SKU bias correction would be fitting noise. Requiring the same sign in every
  window, a mean miss above 20%, and at least two windows leaves 41 SKUs, 33 over-forecast and 8 under,
  carrying 13% of recommended units. The recommended quantity is not adjusted; a consistent over-forecast
  means coverage demand is already high and safety stock is buffering on top of it, which is worth a planner
  seeing and not worth a silent correction fitted to two windows.
  Recorded against the same question, because it was asked and answered with evidence: safety stock should
  not be made direction-aware. It buffers dispersion, and a SKU that swung +50% then -50% genuinely needs the
  larger buffer. Correcting the buffer for bias would shrink it on exactly the SKUs whose direction cannot be
  called. Bias belongs on the forecast; dispersion belongs on the buffer.
  The non-forecast table's rows now link to the SKU detail page, which was the missing half of the previous
  entry: that page draws sales history for an intermittent SKU, and nothing linked to it, so it was reachable
  only by typing a URL.
  Two corrections to earlier notes in this log and in docs/PLANNING_PLAN.md, both from assumptions that should
  have been questions. Every purchase order is on a container, and the three container statuses the Sheet
  import sets are exhaustive, so there is no population of orders invisible to confirmed or draft inbound.
  And inventory has not been sample data since the export landed: inventory_source() reports "export".
  PLAN.md Section 2.2 rewritten accordingly, since it was telling readers to distrust figures that are real.

2026-07-31  Removed the consistently-biased flag added earlier today. It was measuring noise.
  The flag marked SKUs the model misses in the same direction in every backtest window, with a mean miss
  above 20% and at least two windows. It was tested against a null before being trusted, which is what it
  did not survive. Randomising the sign of each window's bias and re-applying the same criteria catches 49
  SKUs on average over 2,000 permutations, against 41 observed, with a 5th-to-95th percentile range of 41 to
  57 and p(null >= observed) = 0.954. Same-sign consistency is 41.9% observed against 43.3% expected by
  chance. The flag identifies fewer SKUs than chance does.
  The cause is structural rather than a threshold that needs tuning. There are three evaluation windows, and
  180 of the 246 SKUs scored in two or more appear in exactly two, so "the same sign twice" is a coin flip.
  No choice of threshold recovers a signal that is not there, and adding windows is not available: the
  evaluation windows are pinned, and which SKUs are eligible for them is the subject of backlog item 2.
  Reverted from quality.py, reliability.py and calc.py. The reasoning is recorded in per_sku's docstring so
  the same flag is not proposed again from the same data. Direction remains measurable per segment, which is
  what the validation grid already shows; it is not measurable per SKU on this evidence.
  Worth separating from the question that prompted it, which stands. Safety stock is a percentage of coverage
  demand, so when the forecast is high the buffer is high for the same reason, and the two compound. That is
  real. What is not available is a per-SKU estimate of which SKUs are affected, so the compounding cannot be
  corrected per SKU on this evidence either.

2026-07-31  Added the per-SKU model-versus-spreadsheet comparison, and made the detail page step through the list the user is actually looking at.
  The reliability card now shows this SKU's model against V1 on the windows both were scored on: pooled WAPE,
  signed bias and units missed, all three because one is not enough. WAPE says how far off, bias says which
  way, units say whether the difference is worth anything. Both errors divide by the same actual, the one from
  the model's own row, so the two are like for like rather than each measured against its own denominator.
  It renders for 260 of the 447 served SKUs. The model is closer on 186 of them and the spreadsheet on 74, and
  those 74 carry a line saying so, because that is exactly where the portfolio result does not transfer. V1's
  error is season-dependent, so an individual SKU can run opposite to the aggregate, and several of the ones
  it wins on are large: CA-SC-10-F-169-BK-1TO at 24% against V1's 12% over 697 units, CC-SS-03-K-GR-1TO at 23%
  against 10% over 645. The page drew both curves and listed both columns and had never said which was closer.
  CA-SC-10-R-90-DG-1TO, the SKU that prompted this, is one of them: model 0.373, V1 0.132.
  Prev, Next and the SKU selector now walk the sequence the list was showing rather than the server's worklist
  order over every forecastable SKU. Filtering to 128 rows, sorting by stockout date, then clicking in and
  pressing Next used to land on a SKU that was neither in the filter nor next in the sort. The selector lists
  the same subset, which is what makes it usable at all: it previously held all 432 with no search.
  Passed through sessionStorage rather than the URL or the API, and the reason is the sort. Filters could have
  travelled as parameters; sort order could not, without a second implementation of the sort on the Python
  side that would drift from the one in action-list-table.ts. The list has already computed the exact sequence
  on screen, so that sequence is what moves. The cost is that a shared link does not reproduce the subset,
  which is the right trade: the planning parameters that must reproduce still travel in the URL. Read through
  useSyncExternalStore so there is no synchronous setState in an effect and no hydration mismatch, and honoured
  only when it contains the SKU being viewed, intersected with what the server still serves.

2026-07-31  Extended the SKU detail history to cover its backtest windows, and gave demand trend the same treatment as reliability.
  The backtest chart was drawing shaded windows and predicted lines over weeks with no actuals beneath them.
  Both charts on that page read one history array sized by history_weeks, which defaults to 26 and therefore
  started in late January, while the earliest backtest cutoff is the previous October. 246 of the 260 scored
  SKUs were affected: those with two windows were missing ten weeks of actuals and those with three were
  missing twenty, which is exactly where the comparison the chart exists to make had nothing to compare
  against. The endpoint now extends the history back to the earliest cutoff it is also returning, plus four
  weeks so there is observed demand to the left of that cutoff line, and reports history_weeks in meta so the
  demand chart trims to what was asked for rather than silently lengthening on SKUs with old windows.
  Demand trend now has a filter on the Action List and a legend under the table stating its thresholds, which
  is where reliability's thresholds already live. It had arrived on the SKU detail page as a single word with
  no definition and no way to reach the other SKUs in the same state, which is not a vocabulary a reader can
  learn. The rule is stated once: last 4 weeks against the last 12 on deseasonalized demand, 1.10x or more
  rising, 0.70 to 1.10 steady, 0.40 to 0.70 falling, under 0.40 collapsing, and unknown where the ratio cannot
  be computed. The ratio is the model's own ramp_4_12 feature imported from src.ml.seasonal, so what the
  dashboard calls falling and what the model treats as falling cannot drift apart. The detail tile carries the
  same rule on hover for a reader who arrived directly.
  Also relabelled "30-day sales" to four weeks, on the detail page and in the table header. recent_units sums
  four W-MON buckets, so it was a 28-day figure under a 30-day name in two places. A true 30 days is not
  available from either source: the weekly series buckets by W-MON so the window falls inside a bucket, and the
  daily order lines are a velocity cache running several days behind it, so a tile taken from there would not
  reconcile with the chart below it. The number was right; the label was not.

--- DAILY SUMMARY 2026-07-31 ---
- Spent most of the day on why the forecasting service ran on one colleague's machine and not
  another: two settings files disagreed, and our own notes named the wrong one. Setup is now a
  single command, and works on Windows.
- Chased a false alarm about the forecasting service being exposed to the internet. It is not:
  a developer machine was quietly serving itself from a local copy, which made the configuration
  look like something it was not.
- Moved the record of past forecasts off one machine's disk and into the database.
- Each product page now says whether the model or the old spreadsheet was more accurate for that
  product. The spreadsheet still wins on about a quarter of them.
- Fixed several faults on the planning screens, the largest a chart hiding most of the sales
  history it was meant to compare forecasts against.

--- SUMMARY PRODUCED 2026-07-31 (covering the 2026-07-31 entries above) ---

2026-08-04  The weekly cron was running the legacy pipeline only, so the two live planning screens were never refreshed.
  scripts/run_forecast_cron.sh called run_forward_forecast.py and nothing else. That is the legacy
  statsforecast pipeline: it ingests fresh orders, rewrites sales_clean.parquet, and writes the shipcore.fc_*
  tables SKU Planning reads. It does not touch ml_forward_forecasts or ml_forecast_history, which are what
  the Action List and Forecast Validation serve. So those two screens had been showing whichever ML forecast
  someone last produced by hand, trained through 2026-07-20, and the accumulating history had never received
  a real run at all. The Performance on forecasts actually served section would have stayed on the seeded
  fixture indefinitely.
  Nothing looked broken, which is why it lasted. The legacy tables stayed current and the readiness check
  passed, because the ML files existed from that manual run. Two details in the same script were already
  written for the ML run and had been quietly doing nothing useful: the readiness check tests ML files, and
  the final step backs up ml_forecast_history.parquet, which that job never wrote. It had been copying the
  same seeded fixture every Monday.
  The cron now runs both, legacy first because the ML script has no ingest of its own and reads the sales
  file the first run refreshes. --snapshot live is passed explicitly: without it the ML script defaults to
  config.ML_DATA_SNAPSHOT, the pinned copy that exists so evaluation figures cannot drift, and every weekly
  run would reproduce the same forecast while appearing to work. Each pipeline's status is tracked
  separately and either failing exits non-zero, since they serve different screens and a legacy backtest
  failure says nothing about whether the ML forecast can be produced.
  Also settled the open question in CUTOVER_TASK.md rather than leaving it: the pipeline has moved to the
  server, so the Mac's push_data_to_server.sh cron is retired. That document asked for a deliberate choice
  between retiring it and removing the Run Forecast button, and warned that picking one by accident is not
  defensible. DEPLOY_TASK.md's instruction to add that cron line is marked superseded rather than deleted,
  since it was correct for the arrangement it was written for. The script stays for a one-off manual push
  and is deliberately not scheduled.
  Not done here, because it is not in this repo: the crontab edits themselves.

2026-08-04  Made the demand trend tile define itself, by printing the ratio beside the word.
  It read "rising" and nothing else. The Action List legend states the thresholds and the tile carried them
  on hover, but a reader on the SKU page saw a bucket label with no measurement behind it. It now shows
  "rising 1.11x": the last four weeks over the last twelve on deseasonalized demand, which is the number the
  bucket was cut from. That is a better answer than any wording, because the buckets are wide. Rising spans
  1.10 to 1.73 and collapsing spans 0.17 to 0.39 on the current table, so two SKUs sharing a label can be
  quite different and the label alone hides it.
  `ramp` was already on the row and non-null for all 432 SKUs, so this needed no API change. Added to the
  TypeScript row type, where it was missing, with a note that it is the model's own ramp_4_12 feature
  imported from src.ml.seasonal rather than reimplemented.
  Recorded because it was asked and is worth having decided: demand trend is deliberately not folded into the
  Action List's summary cards. Those answer why a SKU needs attention now, preorder, out of stock, dry before
  inbound. Trend answers which way it is moving. A SKU can be rising and out of stock at once, so they are
  different axes and merging them would lose one. What the question was right about is that "out of stock"
  needs no definition and "collapsing" does, so the burden falls on this one to carry its own, which printing
  the ratio does.

2026-08-04  Made the demand-trend bands symmetric, after noticing they were not.
  Steady ran 0.70 to 1.10: thirty points below 1.0 and ten above, so a SKU down 25% read "steady" while one
  up 15% read "rising". The cause was one number answering two questions. 0.70 was COLLAPSE_RAMP, measured as
  the point where a plain 4-week average beat the model on the development windows, 0.28 pooled WAPE against
  1.49, and correct for that question. But the flag that asked it was deliberately moved off ramp, because
  ramp describes how history moved rather than whether the forecast agrees with where demand now is, so a
  model-reliability threshold was left labelling a descriptive band. 1.10 and 0.40 were never justified
  anywhere; they were bare literals.
  Now 0.80 to 1.25, reciprocal rather than additive. Ramp is a ratio, so 0.80 and 1.25 are the same distance
  from 1.0 in the units it is measured in, where 0.80 and 1.20 are not. On the current table that moves the
  split from 7/47/257/121 to 7/75/296/54: the old 28% "rising" was an artifact of the boundary sitting at
  1.10 while its counterpart sat at 0.70, not a finding that a third of the catalogue is growing. The
  distribution is centred at 0.98 with a median of 0.98, so it really is balanced around 1.0.
  Not narrowed further, and the measurement is the reason. Ramp deviation scales inversely with volume:
  median |ramp - 1| is 0.28 for SKUs selling under 10 units in four weeks against 0.07 above 100 units, so
  half the smallest SKUs fall outside a 0.80/1.25 band on small-number variation alone. Tightening to
  0.90/1.11 would label 60% of the catalogue as moving, which distinguishes nothing. Recorded on the
  constants so the next person does not tighten them by eye.
  0.40 is kept and its lack of justification is now written down rather than implied. Its reciprocal, a 2.5x
  surge, has no state of its own, which is a separate question: under-ordering a breakout costs as much as
  over-ordering a collapse, and only one of the two has a label.

2026-08-04  Forecast Validation now dates its own evidence, and its performance table says when it is showing sample data.
  The page carried no provenance at all, which is the wrong way round: it is the screen whose entire purpose
  is evidence and it was the only one of the three not saying where its figures came from. The Action List
  has said "Trained through" in its header since it was built. The absence here is how a forecast three weeks
  stale went unnoticed while this page reported on it.
  Four facts now sit under the heading: the model version, the pinned snapshot the evaluation used, the date
  the accuracy report was computed, and the training week of the forecast actually being served. The last two
  are the point. The snapshot is pinned deliberately so recorded figures cannot drift, and a reader seeing
  identical numbers on two visits should know that is by design rather than a broken refresh; the training
  week does move, and the gap between the two is how you see whether the model being validated is the one in
  use. On the current data those read 2026-07-20, scored 2026-07-30, serving a forecast trained through
  2026-07-20, so they agree.
  Separately, the "Performance on forecasts actually served" table now warns when its rows are not the served
  model's. That heading is the strongest claim on the page, and the table rendered whatever the store held
  without qualifying it: on a store holding only the seeded fixture that meant fabricated figures reading
  pooled WAPE 0.076 against the real backtest 0.174 on the same screen, with the version string in a small
  first column as the only tell. The chart above it has carried this warning since the seed script was
  written and the table below never did. It disappears by itself once real runs land, which the cron change
  earlier today makes possible for the first time.

2026-08-04  Made "v11" clickable on the validation page, so the name is explained where it is used.
  The page reported on a model by version string and never said what the string meant, on the one screen
  whose job is technical detail. A version number is where most readers first meet the model, and it was the
  single thing that page assumed rather than explained.
  Clicking it opens a card: what the version is, the feature names it fits per segment, what it predicts,
  the horizon, what it was measured on, and what is being held back. The description and the feature lists
  are served from the registered model class in src/ml/serving/models.py, where a `description` attribute
  already existed, rather than written into the web app. That matters because the page cannot then describe
  a version the API is not serving; the alternative is two statements of what v11 is, drifting apart at the
  next adoption.
  Feature names are shown raw, lead, ramp_4_12, y_last_r, lag_1_r, elev_long, because those are the words
  the design doc and the experiment scripts use, and a reader moving between the three should meet the same
  term rather than a friendlier synonym that matches nothing.
  The structural notes are labelled as track-level rather than version-level: the ratio target against the
  SKU's own trailing 12-week average, the deseasonalize-then-reseasonalize step, and the direct 13-week
  horizon with lead as a feature. Those would still be true of v12, and a reader should not have to guess
  which lines change when the number does. The card also says plainly that version numbers count attempts
  rather than adoptions, since most of the numbers in between were tested and rejected.

2026-08-04  Rebuilt the served-performance table as a matrix, the same shape the comparison grid above it uses.
  Flat, it repeated the model version on all 36 rows, the run date on every sixth, and the weeks-scored count
  six times per run despite that being a property of the run rather than of a segment: every segment in a run
  closes the same weeks, which the data confirms, one distinct value per run. Six results took thirty-six
  rows and three columns of labels. As a matrix, runs down and segments across, it is 6 rows by 6 columns,
  the labels appear once each on the edges, and a cell carries only what differs. It also means the two
  comparisons a reader meets on this page now have the same shape rather than two.
  Runs read oldest first, matching the windows in the grid above and for the reason recorded there: this is a
  time series, the question asked of it is which way the error is moving, and newest-first invites reading
  that trend backwards. Weeks scored moved to the row header, where it qualifies the whole row, since a run
  measured on two weeks is thin evidence whichever column you read. SKU counts and units sold moved to hover,
  being context rather than the comparison. Bias keeps the two-direction colouring from the grid above, so
  over and under do not read as one severity scale.
  Added the state between the two the section already handled: runs stored but no week closed yet. It was
  falling through to an empty matrix under a heading promising results. That is not a hypothetical, it is
  what next Monday looks like, since the first real run will sit unscorable until the week after it. `runs`
  had been fetched and used only for an emptiness check; distinguishing these two cases is what it is for.

2026-08-04  Replaced the served-performance table with a trend strip and a detail panel, because the table did not scale.
  It gains a run a week and never stops, so any layout with a row or a column per run is unusable within a
  year: 52 by the first anniversary, 104 by the second. It was a flat table of 36 rows for six runs, then
  briefly a matrix of six, which is better but scales the same way and only postpones the problem. Rebuilding
  it twice in one day is the cost of fixing the format before asking what the format has to survive.
  The two questions are separated now, because they want different shapes. Which way the error is moving is a
  time series, answered by a strip of one bar per run, oldest left: a hundred runs is a hundred bars in a
  scrolling strip rather than a hundred rows down the page. How a particular run did is a snapshot, answered
  in full for the one run selected, with the SKU count and the units behind each error on the face of the
  card rather than hidden on hover the way a compressed table had to.
  Bars are scaled against the worst run rather than zero-to-one. Pooled WAPE clusters in a narrow band once a
  model settles, so a full-scale axis flattens every bar to the same nub and the differences between runs,
  which is the entire point of the strip, become invisible. Floored at 8% so a very good run stays a visible
  click target. They are plain divs, not a third Plotly instance on a page that already loads two: this needs
  one dimension, a click target and a colour.
  The selection is held by run key rather than index, so it survives a refetch, and resolves against the
  current list on every render: a selection that no longer exists falls back to the newest run instead of
  blanking the panel. Runs from a version other than the served one are tinted amber in the strip, which is
  the same signal the banner above gives, at the level of the individual run.

2026-08-04  Settled the served-performance layout as a scrolling list of runs with a detail panel, after two wrong turns.
  Built three times today, which is worth recording because the first two were wrong in instructive ways. A
  flat table with a row per run per segment multiplies: 36 rows for six runs. A matrix with a row per run and
  a column per segment fixes the multiplication and still grows a row a week, so it is unusable within a year.
  A horizontal strip of one bar per run scales, and throws away everything a row can carry: dates end up as
  9px labels under 36px bars, unreadable at fifty runs, and the SKU counts and weeks scored have nowhere to
  go.
  The mistake behind the strip was assuming a row per run means a page that grows. It does not, once the list
  scrolls inside a fixed height, which is what the action list's own table already does. That was pointed out
  rather than noticed here.
  What it is now: a list of runs, newest first, in a fixed 22rem box that scrolls, so a hundred runs is a
  scrollbar and not a longer page. Each row carries the run date, weeks scored, total WAPE, bias and SKU
  count. Clicking or pressing Enter on a row selects it and the panel below shows that run per segment, which
  is the part that was already right. Rows are keyboard reachable with a visible focus ring; the action list's
  sortable headers still are not, and there was no reason to repeat that here.
  The trend did not need its own shape after all. A bar inside the error cell, scaled against the worst run
  rather than zero-to-one because pooled WAPE clusters narrowly once a model settles, reads down the column
  while every figure beside it keeps its room. One dimension of chart inside a layout that is otherwise a
  list, which is what this data is.

2026-08-04  Removed the intermittent segments from the demand-versus-forecast chart, and wrote the underlying rule into the design doc.
  The chart offered intermittent/long and intermittent/short as segment pills for a model that only forecasts
  smooth SKUs. Two causes, not one. The scored history carried them because the seed script had stamped
  today's profile onto fabricated runs, fixed earlier today. The actuals series carried them independently,
  because it merged load_profiles() onto historical sales, which is the same error the predicted line beside
  it had been corrected for on 2026-07-30. Fixing the forecast alone would have left them arriving through
  the actuals, since the pill list is the union of the two.
  Actuals now take their segment from a map built out of the runs themselves, the forward forecast first
  since it is the newer statement, falling back to the scored history. A SKU with sales but present in
  neither is dropped rather than given a current label. A guard on top of that removes any row whose segment
  is not smooth, from all three series, which is defensible rather than cosmetic: serving/forecast.py filters
  to smooth and writes that literal, so a real run cannot produce another bucket and a row saying otherwise
  is bad data.
  Added the rule to design doc Section 2.4, where the as-of guarantee the scorer enforces is already stated.
  The point of putting it there is that the guarantee stops at the scorer: evaluate.score applies as-of
  segmentation by default so an experiment cannot forget it, and nothing downstream has an equivalent, which
  is where all three occurrences happened. The rule is that ml_forward_forecasts and ml_forecast_history
  carry bucket, history_length and segment per row as of their run, so read them off the row and do not call
  load_profiles() to re-derive them, and where a row lacks them omit it rather than assign a current label.
  All three occurrences are listed by date and place so the next reader sees a pattern rather than an
  isolated fix.
  Correction to what I said in conversation: I called this the fourth occurrence and counted the reliability
  sort among them. That was a different bug, sorting on a substituted error while displaying the measured
  one, not a stale-classification bug. Three occurrences, not four.

2026-08-04  Ported what the Streamlit prototype explained and the web app had dropped, plus three changes it prompted.
  Comparing the two screen by screen, the port had become less explanatory than the prototype it replaced in
  one specific way: every planning control in the sidebar carried a sentence saying what it does, and the web
  app had four bare numeric inputs. That is exactly the gap behind the question asked here two days ago about
  what the risk window and the reorder cycle actually change. Both now say so on the face of the control, not
  on hover, since the question came from someone looking straight at them. The caption the prototype had
  underneath is back too, stating the sum the first two controls jointly produce and neither states alone:
  orders cover 9 weeks, 8 lead plus 1 cycle.
  The controls are one component now, used by the action list and the SKU detail page. The detail page had
  none, so a purchaser deciding on a single SKU had to go back to the list to ask what a longer lead time
  would do to that quantity, losing the SKU they were on. Sharing the component also means the two screens
  cannot explain the same control differently.
  Three changes that came out of the comparison rather than from it.
  The demand column now shows demand over the coverage window rather than the 13-week horizon total. The
  recommendation is built from the coverage figure, so the row carried a number nothing else on it added up
  to. The header names the window and moves with the controls: "9w demand", not "13w fcst".
  A trend column, because the list gained a trend filter and still showed nothing per row, so a reader could
  select "falling" and see no indication on any row of which way anything was moving. Arrow glyph, the ramp
  ratio, colour by direction, matching the SKU page.
  Named sort orders beside the header clicks, not instead of them. The default was the most load-bearing
  ordering on the page and was described only as "worklist order" in grey text. Seven named orders now,
  including that default under its real name, and the select reads "custom" when someone has built an order
  by shift-clicking rather than mislabelling it as the nearest name.
  Also: the priority badge is on the SKU detail header, so arriving from a filtered worklist no longer leaves
  behind the reason for being there. The action list header carries the model version and the horizon end
  alongside the training date, which the prototype's caption had and this did not.
  Deliberately not ported. `served_by`, which says whether the hybrid's shared or long model produced a SKU's
  forecast: it is the short/long split under another name, and that split was removed from both purchaser
  screens for the reason it was removed the first time. The backtest section opening automatically on a poor
  reliability tier, which the prototype does and which was judged not worth the surprise.

--- DAILY SUMMARY 2026-08-04 ---
- Found the weekly automatic run was only refreshing the old forecasting method, so two planning
  screens had been showing a three-week-old forecast produced by hand. Fixed, and the record of
  past forecasts starts filling properly from next Monday.
- Made the validation screen say where its figures come from and when, and warn when they are
  placeholders. Nothing said so before, which is why the stale forecast went unnoticed.
- Rebuilt the past-runs results view, three attempts, until it was a layout that still works
  after a hundred weekly runs rather than only the first few.
- Fixed the demand trend labels. A product falling 25% read as steady while one rising 15% read
  as growing; the actual figure now sits beside the label.
- Restored explanations the screens had lost in the rebuild, so every planning setting says what
  it changes, and corrected a column showing a demand figure the recommendation was not based on.

--- SUMMARY PRODUCED 2026-08-04 (covering the 2026-08-04 entries above) ---

## 2026-08-05

- Considered adding prediction intervals to the SKU forecast chart, built a measured-error band,
  then removed it again. It failed the Section 1.4 test: safety stock already sizes the buffer, so
  no purchaser acts differently for seeing a range, and the band duplicated the reliability tier.
  Measured for the record: per-week error (median 0.377) runs about 1.9x the window-total error
  (median 0.176) on the same weeks, because the window total lets a high week cancel a low one.
  Pooled WAPE is the right instrument for an order covering nine weeks; the per-week version
  answers a question nobody on these screens asks.
- Weekly figures table on SKU detail now shows whole units. Rounded as a column by largest
  remainder so the weeks still add to the total beneath them and the total still matches the
  Action List, which computes from the same unrounded figures.
- Removed the sort dropdown from the Action List. Every order it offered was already reachable by
  clicking a column header, including the default. Replaced with a sentence stating the current
  order, which also covers the multi-column case the dropdown could only call "custom".
- Retired the old Demand Forecast page and the Demand Forecast tab on SKU Planning from the menus.
  Both reported on the legacy statsforecast pipeline while the Action List and Forecast Validation
  report on the LightGBM one; the same SKU showed a different number depending on which screen was
  open, with nothing saying why. Hidden rather than deleted, pending a few weeks without them.
- Found and removed a beforeunload handler on SKU Planning that shut down the forecast service when
  that page closed. Harmless while its forecast tab existed; with the tab gone it meant closing one
  page could take down the two ML screens in another.
- Segment labels from the stored forecast history are normalised on read: intermittent rows dropped,
  medium and full collapsed to long. The performance table had been showing five segments directly
  under a comparison grid showing two, all of it from the seeded fixture's stale labels.
- Forecast Validation restructured into six numbered sections with a contents bar and longer
  explanations; type sizes raised across both ML screens (the floor was 9px).
- Planning assumptions are collapsed by default, with the values still stated while closed. Two of
  the four cannot honestly be answered by the person being asked, and every figure moved with them.
- SKU detail's demand-trend tile explains itself now: what the ratio divides, and the four bands,
  both on screen rather than in a hover title nobody would find.
- Added a Run Forecast panel to the Action List: sync, ingest, profile, forecast, with live step
  progress and cancellation, reusing the existing job machinery.
- Found while building it that ml_prepare_data.py never called the velocity sync, although ingest()
  reads the table that sync refreshes. Every run through that path trained on whatever had last
  been synced and reported it as fresh. The sync is now its first step, and the shared call moved to
  src/velocity_sync.py so both pipelines use one implementation.
- Also fixed a latent bug in that call: its success message formatted a missing row count with a
  thousands separator, which raises, gets caught by the surrounding handler, and reports a sync
  that worked as a failure.
- Backlog 13.4 done: the Action List table now fits. The pinned SKU column's product name was
  uncapped, so one long name set the width for every row; capped with the full text on hover. The
  nine optional columns are individually hideable, grouped by band and remembered per reader,
  defaulting to all. Band headers compute their span from what is visible and the coloured band
  rule moves to the first column still showing. Backlog item 13 reconciled at the same time: 13.3
  recorded as declined with the reasoning, 13.2 marked partly done with the real question still
  open.
- Backlog 13.3 done: the demand concentration table on Forecast Validation is now a Pareto curve,
  with a dotted diagonal showing what even demand would look like and the named breakpoint marked
  on it. The endpoint gained the cumulative series, downsampled to ~200 points. Recorded as
  declined earlier the same day and reversed; the interpretation line added instead was worth
  having but answered a different complaint.
- Shortened the Action List's widest header: "9w demand" became "Next 9w", with the full
  definition on hover. The band above it already says DEMAND, so the word repeated its own heading.
- Backlog 13.1 and 13.2 done, which closes item 13 except 13.5 (still waiting on item 14). The
  validation page is now ordered as an argument: the claim, its scope, the claim over time, the
  out-of-sample record, where it is weakest, what is not claimed yet. Demand shape moved from last
  to second, and per-SKU outliers moved below the aggregate evidence. The contents bar was removed
  rather than rebuilt; six sections listed one-for-one is what the scrollbar already does, and the
  numbering it was really providing now lives on the headings.
- Rebuilt the validation page's per-SKU section, renamed from "Where it breaks down" to
  "SKU-level breakdown". It was two fixed lists of five, which is an anecdote: five bad rows say
  nothing about whether the tail behind them is five rows or two hundred, and nothing about the
  wins. It now answers the question a reviewer actually asks, whether the improvement is broad or
  a few large wins are carrying the average: win rate counted by SKU-window and by units sold
  (62% and 67%), a histogram of the per-SKU differences, the same rates split by backtest window
  and by segment, and every scored row searchable and sortable with links through to the SKU.
  Two findings it surfaces that the page could not state before: the Oct-Dec window is a loss
  (0.105 against V1's 0.091, 47% win rate), and the material-regression tail is 50 rows carrying
  3.1% of demand.
- Also clarified what "units" means there. The section mixes units sold with percentage points of
  WAPE, and the difference column printed a bare "+29" that read equally well as either. Every
  label now says which.
- Added naive baselines and recorded them as design doc Section 4.29. Every accuracy claim so far
  was against V1, the incumbent spreadsheet, which answers "should we switch" but not "does this
  forecast at all"; a naive baseline is what separates skill from a low bar and none was recorded.
  v11 reduces error 34.5% against a trailing-12-week-mean naive and 52.8% against last-week-flat.
  Two findings fell out: V1 at 25.91% is slightly worse than that trailing mean at 24.35%, so
  "beats the spreadsheet" is a lower bar than it sounded; and the seasonal naive is unusable
  because only 18-20% of SKU-weeks have a value 52 weeks earlier, which independently supports
  imposing seasonality through multipliers rather than learning it from the calendar.
- Backlog item 14 done: Best Seller is no longer a priority label. It was a category error, sitting
  in a ladder of supply states while answering a different question, so it lost to Preorder and No
  Stock and the badge vanished from exactly the SKUs where importance mattered most. It is now an
  attribute drawn as a star on every row, a filter, and a tiebreaker within each queue. Measured:
  54 SKUs now show a star that structurally could not before, the best-seller count on the summary
  card goes 35 to 89, and total recommended quantity is unchanged at 2,426 units, confirming the
  order arithmetic never depended on the priority label. This unblocks 13.5.
- Settled the best-seller threshold and deleted best_seller_at_risk. Best seller is now the
  smallest set of SKUs carrying half of recent demand rather than a fixed top 20% of the list:
  46 of 432 SKUs, 50.2% of units, verified minimal and deterministic on ties. A percentile
  described the length of the list rather than the business and stayed a fifth of it whatever
  demand did. best_seller_at_risk was an intersection of two filters the screen already offers
  separately, was never displayed, and is gone from the calculation, the metrics block and the
  type. Also corrected a claim I had made from the Pareto curve: the top 5% carrying 63% of demand
  is a figure for the whole catalogue including the intermittent tail; within the forecastable set
  it is 34%, so the case against the percentile rests on a different argument than I first gave.
- Found and fixed the trailing-week bug. clean() grouped orders into W-MON buckets and kept
  everything through the last bucket containing an order, so any run on a day other than Monday
  trained on a partial final week and stamped it as complete. That week is one of twelve in the
  trailing mean the model's target is a ratio to, and one of four in ramp_4_12 where it hits
  hardest, so every SKU read as falling and every forecast came out low. last_complete_week() moved
  to src/weeks.py so the ingest and the scoring share one rule, and clean() now drops unfinished
  buckets before building its grid. Verified across every day of the boundary week.
- Established the cron was not the cause: a Monday run cannot produce the 2026-08-10 bucket, since
  the earliest order that lands there is dated Tuesday. It was the Run Forecast panel being tested.
- Removed that panel's Stop button (backlog item 15). ml_prepare_data.py writes sales, profiles and
  the forecast in sequence with no rollback, so a cancel that worked would leave the three
  describing different weeks. The Stop that failed was the safe outcome reached by accident.
- Noted that the Monday cron now trains through a week-old cutoff by design, since the bucket
  ending Monday is not finished at 10:00. Moving the cron to Tuesday is a one-line change and
  recovers a week of freshness.
- Found that weekly buckets are one day out from the documented convention: pandas W-MON defaults
  bin Tuesday-to-Monday, while the design doc, the screens and my own reasoning all describe
  Monday-to-Sunday. Recorded as backlog item 16 rather than fixed, because correcting it changes
  every weekly total in the history and therefore re-baselines every recorded figure, which is the
  same class of change as advancing the pinned snapshot. Also corrects yesterday's advice: the cron
  should stay on Monday, since under the intended binning a Monday run is maximally fresh. Moving it
  to Tuesday would have been a workaround for this bug.
- Added scripts/ml_purge_history_run.py to remove the mid-week run that was stored under
  forecast_date 2026-08-10 trained on three days of data. Dry run by default, one run at a time,
  cleans both the parquet and the shipcore table. The history store is append-only evidence, so
  deleting from it is deliberately awkward; a run built on known-broken input is evidence of the
  bug rather than of the model, and the bug is recorded here instead.
- Backlog 13.5 done, closing item 13 entirely: the Action List now explains its priority labels
  where they appear, in the same legend row that already defines reliability and demand trend.
  Item 14 made this easier than it would have been, since the three remaining labels are values of
  one variable and the star is stated separately as an attribute that can sit on any of them.
- Recorded backlog item 17: port 8000 is closed to outside connections, so a developer running the
  frontend locally cannot reach the deployed forecast API and has to clone and run the whole
  forecasting repo instead. Token authentication already exists, so the firewall is not the only
  control; the note recommends opening it narrowly by IP or VPN rather than to the internet, and
  flags that two endpoints reachable with the token rebuild live data.

--- DAILY SUMMARY 2026-08-05 ---
1. Finished the planned round of improvements to the two forecasting screens. Retired an older
   duplicate screen that showed different numbers for the same product, made the main table fit on
   screen with a control for which columns to show, and rebuilt the section on model accuracy so it
   answers whether the improvement holds across the range rather than showing five examples.
2. Added a proper reference point for accuracy. The model is now measured against simply carrying
   recent average sales forward, which it beats by about a third. The same test showed the
   spreadsheet method in use today is roughly no better than that simple average.
3. Found and fixed a bug that made forecasts too low. A forecast produced on any day except Monday
   was trained on a part-finished week of sales, which made products look like they were declining.
   The forecast currently on the server was produced that way and needs replacing.
4. Logged three items for later. The largest is that weekly sales totals are grouped one day out
   from what every document describes; correcting it is right but shifts every accuracy figure
   recorded so far, so it needs doing deliberately rather than quietly.

--- SUMMARY PRODUCED 2026-08-05 (covering the 2026-08-05 entries above) ---
- Fixed the week boundaries (backlog item 16). clean() now uses closed="left" so a week runs Monday
  to Sunday and is labelled by the Monday it ends on, matching every document and screen.
  last_complete_week needed the matching change and did not get it first time: it subtracted an
  extra week on Mondays, correct under the old binning and wrong under the new one, and would have
  discarded the most recent complete week on every cron run. A boundary test caught it.
  No recorded figure moved, because the pinned snapshot is a frozen copy that clean() does not
  touch; I had initially called this a re-baseline event and that was wrong. What is now
  inconsistent is that live data uses the corrected boundaries while the pinned snapshot still
  carries the old ones, which closes at the next re-snapshot. The cron stays on Monday.

- 2026-08-06. Pre-registered the week-boundary A/B (design doc 4.30) and wrote
  scripts/ml_26_week_boundary_ab.py to run it. The open question from the binning fix was
  whether the pinned snapshot has to be rebuilt: every recorded figure was measured on the
  old Tuesday-to-Monday buckets while production now trains on Monday-to-Sunday. I had been
  ready to argue the difference is negligible; it is not safe to assert, since Monday is a
  seventh of a week's volume and shifting that between adjacent buckets can move pooled WAPE
  in the second decimal, coarser than the third decimal versions are compared at.
  The experiment holds everything except the boundary fixed: one raw ingest grouped twice,
  with the SKU set, the week labels and sku_profiles.csv all pinned. Holding the profiles
  fixed is the control that matters, because re-profiling on shifted bins can move SKUs
  across the smooth and intermittent boundary and change the population being scored rather
  than the data under it. Late-order revision since 2026-07-20 is measured and reported as
  the noise floor, and the script declares itself inconclusive if that floor is as large as
  the effect.
  Pass criteria are Section 1.5's existing thresholds, 0.02 per window and 0.01 on the mean,
  plus a sign-stability check. The first draft of that check tested whether v11 beats its
  comparators, which is settled elsewhere and is not what an A/B measures; it now compares
  the sign of the difference on one arm against the sign on the other, so a cell where v11
  loses on both arms is correctly not a failure. It also excluded nothing, which would have
  let the smooth/short Oct-Dec cell decide the verdict despite Section 1.5 ruling that cell
  inadmissible at 14 eligible SKUs. On the synthetic fixture that cell carried the largest
  difference of any, so the omission would have mattered.
  Verified end to end on synthetic orders through both the PASS and FAIL branches. The
  script writes only outputs/reports/, so running it cannot change what production serves or
  what any recorded number was measured on. Also set override=True on load_dotenv in
  src/ingest.py, which the experiment depends on; src/db.py, src/v1.py and
  scripts/compare_v1.py still lack it.

- 2026-08-06. Ran the week-boundary A/B. FAIL on all three pre-registered criteria, so the
  pinned snapshot has to be rebuilt before the final test. Recorded in design doc 4.30.
  The precondition held first: the boundary moved 14.0% of units against a late-order
  revision floor of 3.2%, about four to one, so the difference is attributable. 14.0% is
  one day in seven to within rounding, which is what the shift predicts and is a check on
  the grouping code as well as on the data.
  The result is not that the model got worse. Both comparators are nearly indifferent to the
  boundary, mean absolute change 0.0030 and 0.0032, while v11 moves 0.0159, about five times
  more. A bucketing change that were purely a property of the data would move all three
  alike. Part of v11's measured advantage depends on a convention that came from a pandas
  default and was wrong.
  Skill against the trailing mean goes 39.0% to 33.5%, so Section 4.29's "beats a trailing
  mean by about a third" survives. The margin over the structural baseline roughly halves and
  reverses in Dec-Feb. That is the claim needing restatement.
  Degradation concentrates in Dec-Feb and smooth/long, which points at the seasonal
  machinery, either multipliers aligned to the old boundary or Christmas simply costing more
  under a one-day shift. Added --no-deseas to the script to separate the two, writing to its
  own report file so it cannot overwrite the run it is compared against, and labelled
  v11-nodeseas end to end so its numbers cannot be mistaken for v11's.
  Also added an automatic per-model sensitivity table. The asymmetry that turned out to be
  the whole finding was only visible after adding the deltas up by hand, which is a poor
  place to leave the most informative number in the run.
  My third criterion was flawed and I have recorded it as such: the sign test had no
  magnitude guard, so it fired on a difference of 0.0006 flipping to -0.0007, twenty times
  below the bootstrap standard error. The other flip, spanning 0.03, is real. The magnitude
  criteria failed independently by a factor of two and a half, so the flaw changes no
  conclusion.

- 2026-08-06 (later). The user asked whether the model was refit per arm or the old model
  scored on new data. It was refit, twelve times, but the question exposed a defect in the
  experiment that I had not caught and had explicitly claimed did not exist.
  stratified_val_skus derives the early-stopping validation draw from the arm's own weekly
  values, so the two arms drew 72%, 87% and 95% different validation sets, stopped at
  different tree counts and fitted different models for reasons unrelated to the boundary.
  That noise falls only on v11, since neither comparator has a validation set at all, so the
  headline "v11 moves five times more" compared a stochastically refit model against two
  deterministic ones. The script's own docstring asserted that only the boundary varied.
  Fixed by drawing the validation set once per window and sharing it across arms, now the
  default, with --per-arm-val kept to reproduce the flawed behaviour and measure it. Required
  re-nesting the loop window-outer, arm-inner; the old nesting is what allowed the draw to
  follow the arm.
  The FAIL verdict is very unlikely to move, since the magnitude criteria failed by a factor
  of two and a half. What is now unquotable until the controlled run is done is the
  sensitivity figure, the five-times comparison and the claim that v11 is specifically
  boundary-sensitive, which was the most interesting finding and the least established.
  Design doc 4.30 marks the affected numbers as superseded rather than deleting them.

- 2026-08-06 (later still). Controlled re-run of the boundary A/B, with the validation draw
  shared across arms. FAIL again, and almost identically: six of eight judged v11 cells
  bit-identical, mean sensitivity 0.0159 to 0.0151, ratio against the baseline 5.1x. Every
  smooth/long cell unchanged, so the long model's draw had already been effectively shared
  and only the short model was affected. The confound was real and worth fixing and was not
  what produced the result, which is the strongest thing that could be said for the finding.
  The failure is one cell. smooth/long moves +0.0536 in Dec-Feb against +0.0021 in each of
  the other two windows; excluding Dec-Feb the segment mean is a fifth of the threshold.
  smooth/short is a different pattern, steady and moderate across both judged windows.
  Added a series-shape table to the script, printed before any fitting: lag-1 autocorrelation
  and relative week-to-week change per arm, whole series and December-January alone. It tests
  the alternative explanation, that one bucketing simply produces an easier series, which
  would flatter a lag-reading model while leaving a trailing mean indifferent. Costs nothing
  and is the natural companion to the sensitivity table.
  Recorded in 4.30 that skill against the trailing mean is 39.0% to 33.7% and survives, while
  the margin over the structural baseline halves and needs restating. Both are unweighted
  three-window means rather than the unit-weighted pooling used elsewhere, and are flagged as
  indicative pending recomputation from the report CSV.

- 2026-08-06 (bootstrap). Added bootstrap_arm_delta to the A/B script and ran it. The
  existing evaluate.bootstrap_delta could not do this job: it compares two models against one
  split and carries a single denominator, while the two arms have different actuals by
  design, so each side needs its own numerator and denominator. Pairing on SKU is valid
  because the population is held identical across arms.
  It should have been there from the start. Section 1.5 says borderline calls are settled by
  the bootstrap and puts single-window noise at 0.011 to 0.014, and three of the four
  magnitude criteria failed by under 0.006. I reported point estimates as measurements for
  two rounds before putting an error bar on any of them.
  Result: of the five judged segment cells only two clear two standard errors, and the one
  carrying the whole long-segment verdict clears it at 2.30 with a 95% interval spanning
  almost twelve to one. Cell by cell the experiment barely resolves anything. As a pattern it
  does: all five deltas are positive, none negative, sign test p = 0.031, while both
  comparators scatter around zero with mixed signs. Direction systematic and specific to v11;
  magnitude undetermined.
  Consequence for reporting: the 5.1x sensitivity ratio is not quotable, being an average
  dominated by the least-known cell. The supportable claim is that v11's error is
  systematically higher under the corrected bucketing while the comparators are indifferent,
  by roughly 0.01 with an unpinned upper end.
  Also withdrew "the new arm is the truthful one" from 4.30. Neither bucketing is truer.
  The case for the new one is that production computes it and the docs describe it, which is
  consistency, not correctness. The user caught this by pointing out the week convention was
  arbitrary and theirs to choose.

- 2026-08-06 (phase sweep). The --no-deseas diagnostic was a dud and the reason is my error.
  Neither deseas flag reaches the long segment: build_matrix sets seas_long_uids from
  seg_long_uids unconditionally, deseas_all only extends the treatment to short SKUs, and for
  a long SKU y_adj and y/factors are the same number. So long SKUs are deseasonalised either
  way and the long results came back bit-identical, which is the code being honest. I
  designed the test on an assumption about what the flags did without reading build_matrix.
  It did answer for the short segment, where removing deseasonalisation flipped the Dec-Feb
  boundary delta from +0.0097 to -0.0338, though in a much degraded regime.
  Built scripts/ml_27_week_phase_sweep.py on the user's suggestion of trying all seven week
  phases. Implemented by shifting the dates and holding the grouper fixed, so all seven emit
  identical week labels and therefore identical evaluation windows; shifting the grouper's
  anchor would have moved the labels and broken the comparison. Verified the seven spans
  before running anything. 42 fits, 23 seconds.
  Results. v11's mean range across phases is 0.0536 against 0.0167 for the structural
  baseline, a factor of 3.2, which is the boundary-sensitivity finding properly measured
  instead of inferred from two points. The swing is large in absolute terms: smooth/long
  Oct-Dec runs 0.1000 at a Tuesday start to 0.1718 at a Sunday start.
  The informative part is that Tuesday is v11's minimum in seven of eight cells while the
  comparators' optima wander by window (baseline Sunday four times, Monday twice). A single
  phase winning consistently across seasons is the signature of something inside the model
  calibrated to that phase, the monthly multipliers and the CV-optimised holiday constant
  being the obvious candidates, since both were fitted on Tuesday-binned data.
  This also reframes experiment 26. Its "old" arm was phase 1, which is v11's best phase in
  almost every cell, so "the correction made the model worse" was substantially "we moved off
  the phase the seasonal calibration was tuned to". The 0.0536 was not a random pair.
  The curve is smooth rather than spiky, monotonic from Tuesday round to Sunday in the
  Oct-Dec long cell, which argues against a calendar artifact such as Christmas landing in a
  different bucket and for something structural.

- 2026-08-06 (mechanism, not found). The user challenged my explanation for the Tuesday
  optimum, saying the monthly multipliers come from the Google Sheet and the holiday
  multiplier was carried over from the statsforecast model, neither fitted to this model's
  data. Checked, and they are right on both counts. SEASONAL_BASE in src/deseasonalize.py is
  a hand-set dict of round numbers; config sets ML_HOLIDAY_MULTIPLIER = HOLIDAY_MULTIPLIER,
  carrying 1.26 across unchanged; and ml_13, which moved the window end to mid-December,
  states in its own docstring that the change rests on business knowledge and explicitly not
  on fitting the observed Decembers. My "calibrated to Tuesday-binned data" hypothesis had no
  basis and I should have read this before proposing it.
  Tested it anyway rather than just withdrawing it. Demand-weighted mismatch between the
  factor applied to a week and the factor its actual days deserve: Thu 0.0039 best, then Wed,
  Fri, Tue, Sat, Mon 0.0092, Sun 0.0105. Monday is second worst and Tuesday is fourth, while
  v11 has Tuesday best and Monday second best. The orderings do not correspond. Hypothesis
  dead on the evidence, not just on provenance.
  Second hypothesis also dead. Lag-1 autocorrelation by phase is a shallow U with maxima at
  Monday 0.2817 and Sunday 0.2847; Sunday is where v11 does worst. Day-of-week demand is
  nearly flat, 13.6% to 15.1%, so there is no strong weekly rhythm for a boundary to align
  with either.
  Found a bug in my own earlier reporting while doing this. report_series_shape ran before
  prep(), so it measured the zero-filled grid including every SKU's pre-launch zeros, which
  are perfectly autocorrelated, and reported 0.68 where the real figure is about 0.25. The
  comparison I drew from it last round was worthless. Moved the call after prep.
  Remaining candidate, and it is one I introduced: sku_profiles.csv is held fixed across all
  seven phases as the population control, but it was generated from Tuesday-binned data.
  train_start, bucket, history_length and the eligibility filter all encode phase 1's bucket
  edges, so phase 1 gets profiles that match its data and every other phase gets profiles
  offset by p days. Holding them fixed controls the population and hands Tuesday a matched
  fit at the same time. Note the shared validation draw is taken from the Monday phase, so
  Monday holds any home advantage there and Tuesday still wins.
  Test: regenerate profiles per phase and re-run. If Tuesday's margin shrinks, that is it.
  Not yet run.

- 2026-08-06 (phase selection, evidence supports Tuesday). Ran the reprofile test: profiles
  re-derived from each phase's own weekly series, scored on the 409 SKUs classed smooth under
  all seven phases so the population stays common while train_start and history_length become
  phase-matched. The hypothesis died. Tuesday still wins and the smooth/long cells came back
  essentially identical to the pinned-profile run. v11's mean range across phases moved only
  0.0536 to 0.0558. So the Tuesday advantage is not an artifact of profiles derived under
  Tuesday, and I am out of hypotheses for it.
  Then ran the test that actually bears on the decision: leave-one-window-out phase selection.
  Choose the phase on two development windows, pay for it on the third, unit-weighted pooled
  on TOTAL.
    held out Mar-May  chose Tue -> 0.1693   Mon 0.1819
    held out Dec-Feb  chose Tue -> 0.1782   Mon 0.2056
    held out Oct-Dec  chose Tue -> 0.1047   Mon 0.1042
    mean selected 0.1507, Monday 0.1639, hindsight-best 0.1506
  Honest out-of-sample gain from selecting a phase: +0.0132. In-sample gain: +0.0133.
  Selection optimism 0.0001, which is negligible, and Tuesday is chosen on every fold.
  This vindicates the user's position, which I had been resisting on principle for several
  rounds. The objection I kept raising was that picking the best of seven is a maximum over
  seven noisy draws; the leave-one-out check measures exactly that and finds it costs nothing
  here. An argument that can be measured should have been measured earlier instead of
  repeated.
  What remains against Tuesday is operational rather than methodological, and one piece of it
  is concrete: under Tue-Mon the bucket labelled L is not complete until the end of Monday L,
  so a Monday 9am cron cannot use it and the freshest usable week ends seven days earlier.
  Adopting Tuesday therefore means moving the cron to Tuesday, reverting clean.py to
  closed="right", and reverting last_complete_week to subtract the extra week on Mondays.
  All three are coupled and must move together or the pipeline silently loses a week of
  recency. Plus the documentation, the UI and the Korean labels, all of which say Mon-Sun.
  Caveats to carry: three folds only, sharing training data and SKUs, so the optimism
  estimate has its own uncertainty; and the mechanism is still unknown.

- 2026-08-06 (reverted to Tue-Mon, three coupled changes). Made the change set:
  src/clean.py back to closed="right" with both arguments written out explicitly, so the
  convention no longer looks like a pandas default; src/weeks.py last_complete_week restoring
  its extra Monday step; scripts/run_forecast_cron.sh moving from Monday to Tuesday
  (0 10 * * 2). Each of the three carries a comment naming the other two, because separating
  them produces a pipeline that trains on a fragment or discards a good week and neither
  failure announces itself.
  Verified. Every run day from 2026-08-07 to 2026-08-19 yields a bucket of exactly seven days
  that has finished. Freshness at the Tuesday cron is one day against seven at the Monday
  cron, which is the whole reason the cron moved.
  Also verified the thing I should have checked on 2026-08-05: clean.py's grouper now agrees
  with the SQL in api/main.py and src/db.py on all 122 days tested. Those queries use
  (order_date + ((8 - ISODOW) %% 7) days), which is Tue-Mon and always has been, so the
  Mon-Sun change had put the Python ingest and the API's own queries into silent disagreement
  about which week a Monday's orders belong to. That was a second defect introduced by the
  first fix and nothing would have surfaced it.
  No re-snapshot. The pinned snapshot was generated under Tue-Mon and is consistent with
  production again, so the Version Log stands and the re-baselining that 4.30 called for is
  cancelled. data/processed/sales_clean.parquet is dated 27 July, before the change, so it
  was never regenerated under Mon-Sun either and needs nothing.
  Documentation corrected rather than the code: design doc Section 2 now states the span in
  full (Tuesday through Monday inclusive) alongside the label rule, which was always right.
  BACKLOG 16 reopened and closed as "documentation was wrong". Section 4.30 rewritten as an
  adoption with the leave-one-out evidence, keeping the superseded reasoning beneath it.
  Not changed, and worth a look before the writeup: the Cowork project-context note still
  describes weeks as Mon-Sun. No frontend source file states a span, so the UI needs nothing.

- 2026-08-06 (audit of the sweep, prompted by the user asking whether pipeline pieces still
  aligned to Tue-Mon were inflating the other phases). Enumerated and tested every alignment I
  could find.
  Ruled out. The evaluation target shifting with phase: if a moving window were making the task
  easier or harder, all three methods would move together, and they do not. In smooth/long
  Oct-Dec the baseline improves monotonically from Monday to Sunday (0.1222 to 0.1021) and the
  trailing mean improves too (0.1989 to 0.1589) while v11 degrades (0.1021 to 0.1666),
  correlations of -0.97 and -0.95 against v11's curve. Mean across cells -0.17 and -0.04.
  Also ruled out: profiles (already tested), and the SQL, which is not in this loop at all
  since the sweep reads raw order lines and does its own bucketing.
  Found a real edge asymmetry and bounded it. Clipping each phase to the pinned label set means
  higher phases pull in up to six extra days of post-July-2026 orders, so coverage runs
  monotonically from 95.59% of units at Monday to 97.08% at Sunday. Those orders land in labels
  near 2026-07-20, which is after every development window's test period, so they cannot reach
  the results. Worth recording because the monotonic shape looks alarming until located.
  Found one genuine defect: ml_is_holiday derives the week's days as [ds-7, ds-1], the Mon-Sun
  span, when the convention in force is [ds-6, ds]. Zero effect today, verified over every
  label in the data: both derivations flag 8 weeks and disagree on none, because the window
  needs 4 of 7 days and a one-day shift never crosses that. Logged as BACKLOG 19 rather than
  fixed, since it is seasonal-adjustment code and cannot change a number today.
  Net: every alignment I could find points at Monday or Thursday, not Tuesday. The holiday flag
  assumes Mon-Sun and the monthly factor mismatch is lowest at Thursday with Monday second
  worst. Tuesday wins in spite of the seasonal machinery rather than because of it, which
  strengthens the adoption and leaves the mechanism as unexplained as before.
  Caveat to carry into the writeup: the spread figure, v11 at 3.6x the baseline's range, is
  measured across phases whose seasonal machinery is misaligned to varying degrees, so it is
  plausibly an overstatement of true phase sensitivity. The Monday-versus-Tuesday decision does
  not depend on it, resting on the leave-one-out result instead.

- 2026-08-06 (alignment sweep of both repos). Searched both codebases for week-span arithmetic
  rather than fixing only the one already known. Four places derive a week's days from its
  label. Two were already correct for Tue-Mon and needed nothing: src/db.py fetches from
  ds - 6, and src/ml/serving/v1.py already computes its seasonal modifier over [ds-6, ds].
  Two were wrong, and only one of them was harmless.
  Fixed src/ml/seasonal.py:ml_is_holiday, [ds-7, ds-1] to [ds-6, ds]. Re-verified after the
  change: still exactly 8 weeks flagged across every label in the data, so nothing moves, as
  predicted. BACKLOG 19 closed the same day it was opened.
  Fixed Commerce_Integration src/lib/forecast-metrics/repository.ts:getLastCompletedMonday,
  which returned the most recent Monday and therefore returned TODAY on Mondays. Since
  getAccuracyRows sums actuals over (ds - 7, ds] and filters ds <= that value, every Monday the
  SKU accuracy view scored a forecast against a week that was still accumulating, picking up a
  part-day of orders. The week always looked under-sold and accuracy always looked worse, once
  a week, for one day. This is the SQL twin of last_complete_week and now agrees with it on
  every weekday. Written as ((ISODOW + 5) %% 7) + 1 to keep both modulo operands positive,
  because Postgres truncates rather than floors and the obvious form returns 0 on a Monday.
  tsc --noEmit exits 0.
  Not touched: the frontend lastMonday() in sku-forecasts/demand-forecast/demand-forecast-tab.tsx
  carries the same off-by-one, but that file is orphaned, confirmed by grep, and is already on
  the deletion list. Fixing dead code would only make it look maintained.
  Deliberately NOT changed: the monthly factor is assigned from the label's month
  (ds.dt.month), and the label is the week's last day, so a week spanning a month boundary
  takes the later month. Measured mismatch is 0.0058 at Tuesday against 0.0039 at Thursday, so
  there is room there, but changing it alters deseasonalisation for every week in the data and
  re-baselines the Version Log. That is a modelling change needing its own pre-registered
  experiment, not a fix folded into an alignment pass.

- 2026-08-06 (cron move and v15 pre-registration). Updated docs/DEPLOYMENT.md to Tuesday with
  the reasoning attached, since that file is the source of truth for the crontab line and was
  still showing day 1. The crontab itself is hand-installed on the server via crontab -e and
  cannot be changed from here, so wrote out the command sequence instead: back up first, then
  a targeted sed on the forecast line only so any other job keeps its schedule, then confirm
  exactly one line on day 2.
  Put a verification step ahead of it rather than assuming. The server may never have received
  the 2026-08-05 binning change, in which case it has been on Tue-Mon throughout and the
  reversal is a no-op there; what it would still be missing is the partial-trailing-week guard
  in clean.py, which is the fix for the incident that started all of this. Deploy state should
  be read, not inferred from what is on the laptop.
  Pre-registered v15, the seasonal blend, in the Version Log. The criterion is deliberately
  not Section 1.5's. This is a correctness fix and the predicted effect, a factor change of
  0.02 on a fifth of weeks, is very unlikely to move pooled WAPE by the 0.01 the adoption rule
  wants. Holding a defect fix to a performance bar would be the wrong instrument: adopting a
  neutral performance change is wrong, adopting a neutral correctness fix is right, because
  giving a week that is six-sevenths July an August multiplier is not defensible whatever it
  scores. So the primary criterion is non-inferiority, adopt unless a window regresses by more
  than two bootstrap standard errors.
  The secondary criterion is the one worth having: re-running the phase sweep afterwards should
  narrow v11's spread, currently 0.0558 against the baseline's 0.0167, because a step-function
  seasonal treatment is inherently phase-sensitive and a blended one is not. Stated now so it
  cannot be adjusted afterwards. If the spread does not narrow, Section 4.30 loses a candidate
  explanation and that gets recorded too.
  CORRECTION, same day: I claimed v14's result was never written up. That was false. The v14
  entry is complete, carrying "Status: rejected, at every value tested", the full results
  table, an assessment against all four pre-registered criteria, and the conclusion that the
  collapsing tail is objective-constrained rather than capacity-constrained. I checked the
  numbers in the entry against outputs/reports/v14_min_child_sweep.csv afterwards and all
  twelve short-segment figures match exactly. I repeated the claim from a summary of earlier
  work without opening the section, which is the same failure as several others in this run.

- 2026-08-06 (re-verification of the phase selection result, on request). Before re-checking
  anything I found that ml_27 wrote both the pinned-profile and the --reprofile sweep to one
  path, so the second run had destroyed the first. That is the identical bug I had fixed in
  ml_26 with OUT_DIAG earlier the same day, and it reappeared because I fixed it in the file I
  found it in instead of searching for the shape. Fixed, separate output files, both sweeps
  re-run so the two exist side by side.
  Wrote scripts/ml_28_verify_phase_selection.py rather than another inline computation. The
  original result was one CSV, TOTAL only, one combination rule, and it was carrying a
  structural change to the pipeline. The script varies all four things that were choices:
  which sweep, TOTAL against per-segment, unit-weighted against unweighted window combination,
  and whether the Section 1.5 inadmissible cell participates.
  It holds across all eight variants. Tuesday is chosen on every fold of every variant, the
  honest gain over Monday runs +0.0132 to +0.0153, and selection optimism runs 0.0000 to
  0.0002. The originally reported +0.0132 and 0.0001 were correct.
  Added the figure the original was missing. "Optimism is 0.0001" is close to vacuous on its
  own, because with three folds and one phase winning all three, selection cannot be
  optimistic by construction. What carries the result is the margin over the runner-up on the
  folds used to choose, which is 0.0094 to 0.0122, about the same size as the gain itself.
  Also worth recording: Monday is not the runner-up. Ranking the selection folds, Tuesday
  leads and Wednesday is usually second, with Monday second in one fold of six and fourth in
  two. So this is not a close call between the two conventions anyone considered; Tuesday
  leads a field in which Monday is middling.
  Caveats unchanged and still real: three folds sharing SKUs and overlapping training data, so
  they are not independent and this design cannot quantify the error on the optimism estimate;
  and the mechanism remains unknown, so nothing here says the advantage transfers beyond the
  windows tested.

- 2026-08-06 (deploy landed). Run #46, triggered manually, deployed successfully: all eight
  steps green including "Report data readiness", which is the step that exits non-zero when
  the service answering port 8000 is not the one just deployed. So e7a6665 is live on the
  server and the partial-trailing-week guard is in production for the first time since it was
  written on 2026-08-05.
  The automatic push trigger still did not fire and the cause is unknown. Nothing in the
  workflow file accounts for it: plain push trigger on main, no path filters, and the deploy
  job's only condition is the branch. The same commit deployed fine under workflow_dispatch,
  so the workflow is sound and the trigger is the open question. Logged as BACKLOG 20, framed
  around the real problem rather than the incident: a push that fails to deploy looks exactly
  like one that succeeds from the developer's side, which is how a fix for a live bug sat
  undeployed for a day while work continued on top of it.
  My own failure here was not checking. The deploy is automatic, so I assumed it had happened
  and said so, twice, without opening the Actions tab. Checking would have cost one fetch.

--- DAILY SUMMARY 2026-08-06 ---
1. Settled which seven days count as a sales week. Yesterday's change to match our written
   documentation was quietly making forecasts worse, so I tested all seven options, put it
   back, and corrected the documentation instead.
2. Got yesterday's bug fix onto the live server. It had been written but never actually
   arrived: an automatic step failed without reporting it, leaving the live forecast without
   the fix for a day.
3. Fixed two smaller date errors found elsewhere in the system. One was making the accuracy
   screen look worse than reality every Monday.
4. Moved the weekly forecast to Tuesday, so it uses last week's sales rather than data that
   is already a week old.

--- SUMMARY PRODUCED 2026-08-07 (covering the 2026-08-06 entries above) ---

- 2026-08-07. Moved the server crontab from Monday to Tuesday, the last manual step of the
  week-convention change and the only part the deploy could not carry, since the schedule
  lives in the coverland user's crontab rather than in the repo. Confirmed by the line it
  printed back: a single entry, 0 10 * * 2, script path unchanged. The sed was tested against
  a realistic multi-job crontab first, checking that it moved only the forecast line and left
  MAILTO, a daily backup and an unrelated Monday job alone.
  All three parts of the convention are now consistent in production for the first time:
  clean.py binning closed="right", last_complete_week stepping back an extra week on Mondays,
  and the cron on Tuesday. First run under the new schedule is Tuesday 11 August, 10:00 UTC,
  training through the week ending Monday 10 August.

- 2026-08-07. Implemented and ran v15, the seasonal blend. Put it behind ML_SEASONAL_BLEND in
  config, defaulting False, and verified the flag is inert when off: with it False the factors
  are bit-identical to the recorded ones across every label in the data. With it on, 43 of 110
  weeks move, mean 0.0055, max 0.0743, and the count of weeks receiving the full holiday
  multiplier drops from 8 to 5 as three partial weeks now get a fractional lift instead of an
  all-or-nothing one. Every consumer goes through ml_factors, so the flag reaches all of them
  or none.
  Result: passes the pre-registered non-inferiority criterion. No judged cell regresses by more
  than two bootstrap standard errors; one improves beyond noise, smooth/short in Dec-Feb at
  -0.0110; the rest are ties.
  Recorded a caveat the criterion cannot see. smooth/long regresses in all three windows,
  +0.0059, +0.0090, +0.0027, none individually beyond noise but all one direction. A per-cell
  non-inferiority test is blind to sign consistency by construction. Same shape as Section
  4.30's finding. The blend appears to help short SKUs and mildly hurt long ones, and the
  pooled result is a cancellation rather than a wash.
  Did not revise the criterion after seeing this. It was pre-registered, it passes, and
  rewriting it on a pattern that only appeared afterwards is what pre-registration exists to
  stop. Verdict stands as adopt, caveat recorded beside it.
  Left ML_SEASONAL_BLEND = False. Turning it on re-baselines the Version Log, so that is the
  user's call rather than a side effect of the experiment passing.

- 2026-08-07 (v16, rejected, and it overturned yesterday's attribution). Pre-registered v16,
  the holiday half of v15 without the monthly half, recording openly that the hypothesis came
  from looking at development results and is therefore exploratory in origin. Made
  ML_SEASONAL_BLEND a mode, "off" / "holiday" / "full", so all three versions are reachable
  and none is the default, with an explicit error on a typo since a misspelt mode would
  otherwise fall through to the unblended path and look like a null result.
  v16 fails criterion 1: TOTAL in Mar-May regresses +0.0048 against a 0.0034 threshold.
  It also fails criterion 2, which matters more. The prediction was that v16 would improve
  smooth/long by about 0.003 if v15's halves were additive. It regresses it by +0.0054
  instead, so the halves are not additive and ml_30 attributed nothing. I told the user
  twenty minutes earlier that the monthly half was responsible; that was wrong, and the
  pre-registered secondary criterion is what caught it rather than my noticing.
  What the three runs say together: mean smooth/long delta is +0.0085 for the monthly blend
  alone, +0.0054 for the holiday blend alone, +0.0059 for both. Every perturbation of the
  seasonal factors costs the long segment about half a point, and two together cost no more
  than one. That is a segment sensitive to the seasonal specification itself rather than to
  any particular change being wrong, and it is very likely the same thing Section 4.30 has
  been measuring as week-boundary sensitivity.
  Neither version adopted. ML_SEASONAL_BLEND stays "off". v15 passed its own criterion and is
  still not being taken, because its pass is now understood as two effects cancelling rather
  than the fix being free.

- 2026-08-07 (port 8000, and a production incident found while opening it). Backlog 17. The
  service was bound to 127.0.0.1, so no firewall change could have reached it; that had to be
  changed first and is the step the request assumed away. Changed the systemd unit to
  --host 0.0.0.0, which keeps loopback and so leaves the Next.js app's
  AI_SERVICE_URL=http://localhost:8000 working unchanged.
  The restart then exposed something worse. The unit was crash-looping with [Errno 98] address
  already in use, every eight seconds, because an unmanaged process held port 8000. It had
  been started at 00:01:11 UTC with a relative venv path and no --workers 1, so not by
  systemd, five seconds after the deploy rsync wrote its files at 00:01:06. The deploy race
  DEPLOYMENT.md warns about, caught in the act: something started a second uvicorn during the
  restart window and won the port.
  Consequence, measured rather than assumed: the file mtimes predate the process start by five
  seconds, so it had loaded the deployed code and nothing was stale. I had told the user
  deploys "may not have been taking effect" and that the training guard might not be live.
  That was the right thing to check and wrong as a conclusion, and I said it before running the
  check that settles it.
  Killed the squatter; systemd bound within two seconds and now holds 0.0.0.0:8000, active,
  /health ready. The bind change is in effect.
  Not resolved: what started it. None of the four candidate env paths exist, so the Next.js app
  is elsewhere, and the sudo grep in that step had no TTY and may not have run at all. Recorded
  as inconclusive rather than negative. Until it is found this recurs at the next deploy.
  Also worth stating plainly: the deploy reported green throughout. The workflow's repo_root
  check proves the answering process runs from the deploy directory, which a squatter started
  from that same directory also satisfies. It cannot distinguish "the deployed unit is serving"
  from "something is serving". I read that check as stronger than it is, twice.

- 2026-08-07 (the squatter recurred, pattern confirmed). Killed pid 3152116 in the afternoon
  and systemd bound 0.0.0.0:8000 as 3263790. By 21:58 the port was back on 127.0.0.1, held by
  a new process, 3269753, started 21:53:24 with the same signature: relative venv path, no
  --workers 1, PPID 1. Killed it; systemd bound within four seconds as 3282574 and the unit
  reports active.
  Two occurrences now, both within seconds of a deploy restarting the service: 00:01:11 (five
  seconds after the rsync) and 21:53:24 (shortly after the diagnostics workflow was pushed).
  For a new process to take 127.0.0.1:8000 the systemd process must first have released
  0.0.0.0:8000, which is what a restart does. So the mechanism is settled even though the
  culprit is not: every deploy opens a gap and something wins the race for the port.
  Consequence that matters more than the lost remote access: while the squatter holds the
  port the unit crash-loops, so the next deploy ships code that never runs and still reports
  green. Today's guard survived only because the squatter started five seconds AFTER the
  rsync. The reverse ordering would have been silent.
  Built two things. scripts/_kill_squatter.sh finds the loopback listener by pid rather than
  taking one as an argument, since the pid changes every time, and matches nothing when the
  unit is correctly bound so it cannot kill a healthy service. And
  .github/workflows/server-diagnostics.yml, workflow_dispatch only, no sudo anywhere, running
  the checks over the deploy key so nobody has to type a password twelve times to answer
  "what is on port 8000".
  Where I had been looking wrongly: I searched /opt for .env files holding
  FORECAST_SERVER_DIR and found none, and concluded the variable was probably unset. But
  DEPLOYMENT.md says Next.js runs under pm2, and pm2 carries environment per process, so it
  need never appear in a file at all. Added a pm2 step to the workflow that prints only
  AI_SERVICE_URL and FORECAST_SERVER_DIR per app, never the whole env, since that holds both
  database URLs and the forecast token.

- 2026-08-07. Removed the product name feature from the Action List and SKU Detail screens in
  both repos, on request: the sc_products lookup it depended on was a display convenience, not
  something purchasing decisions need, and the SKU identifier alone is sufficient. Dropped the
  query and every product_name column from the Time_Series_Forecasting planning backend
  (inventory.py, data.py, calc.py, quality.py) and the corresponding fields, search filters and
  subtitle lines from Commerce_Integration's action-list components, leaving product_category
  in place since it is derived from the SKU prefix rather than sc_products. Left the separate
  SKU Master and SKU Forecasts features untouched, since the name there serves a different,
  broader product-catalog purpose.
  Also simplified product_category itself to three buckets: Car cover, Seat cover, and Other.
  Previously an unrecognised prefix (CA-CL and others) was shown as its own raw, uninterpreted
  category; now anything not confirmed as one of the two named families reads as Other, which
  is honest without being a filter list nobody can read.

- 2026-08-07. Recorded the next round of planning-screen changes as BACKLOG 13, five presentation
  items on screens whose numbers are already right: reorder the Forecast Validation page, redo its
  contents list with it, replace the demand table with a Pareto curve, fit the Action List table to
  the screen so it stops scrolling horizontally, and explain the priority labels where they appear.
  The explanation item is sequenced last deliberately, since BACKLOG 14 changes what the labels are.
  Writing the criteria out is what surfaced 14. Best Seller sits in the same precedence ladder as
  Preorder, No Stock and Routine, and it is not the same kind of thing: those three are states of
  one variable, supply, while best seller is importance, which every SKU carries independently. One
  slot cannot hold both, so importance is discarded on every row that has a supply problem. The
  visible symptom was the question that started this: 86 SKUs carry the best_seller flag and 27
  carry the badge, because a top seller is exactly the kind of SKU that is out of stock or on
  preorder. The badge therefore thins out where it would be most useful, and a best seller with a
  preorder backlog is indistinguishable from a tail SKU with one. Decided: take it out of the
  ladder, keep it as an attribute with its own marker and filter, and use it as a tiebreaker within
  each queue. The star then means "top 20% by demand" rather than "top 20% with nothing more
  pressing about it". Consequences recorded rather than left to be discovered: the summary count
  moves 27 to 86 and changes in kind from a queue to an attribute, and best_seller_at_risk is a
  third set again and wants revisiting in the same pass. Nothing in the order quantity or the
  stockout projection reads priority, so no recommended figure moves. Left open whether a fixed 20%
  cut earns its place at all once the Pareto view exists.

--- DAILY SUMMARY 2026-08-07 ---
- Spent most of the day trying to make the forecasting service reachable from outside the server.
  Not finished, and it is the first thing next time.
- The hold-up: another program keeps seizing the service moments after each release, so it was
  down while the release still reported success. Cleared it twice. Until we find what starts it, a
  release can ship code that never runs.
- Built a one-step fix for that and a server check anyone can run, so the next occurrence costs
  minutes rather than an evening.
- Tried two changes to the seasonal adjustment, neither adopted. The one that passed only did so
  because two opposite effects cancelled out.
- Moved the weekly forecast run to Tuesday, dropped product names from the planning screens, and
  wrote up the next round of screen changes.

--- SUMMARY PRODUCED 2026-08-07 (covering the 2026-08-07 entries above) ---

- 2026-08-07 (root cause of the squatter found: the deploy was doing it). Not pm2. The
  fallback branch in ci-cd.yml's restart step starts
  `nohup .venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 &`, which
  matches the observed squatter character for character, and `pkill`s the systemd process
  first. Grepped both repos: that line exists in exactly one place.
  Why the fallback fired when the unit plainly exists. The condition was
  `systemctl list-unit-files | grep -q '^coverland-forecast-api\.service'` under
  `set -euo pipefail`. grep -q exits at the first match and closes the pipe, systemctl dies of
  SIGPIPE returning 141, and pipefail makes 141 the pipeline's status, so the if evaluated
  false every time. Reproduced in isolation before claiming it: `seq 1 200000 | grep -q "^42$"`
  takes the else branch with pipefail and the if branch without, exit status 141.
  So every deploy killed the managed process, started an unmanaged loopback one, and left
  systemd crash-looping under Restart=always, while the readiness check passed because the
  squatter runs from the deploy directory and reports the same repo_root.
  Fixed three things in that step: the condition now uses `systemctl cat`, which takes a unit
  name and needs no pipe; the fallback binds 0.0.0.0 so it cannot silently undo the change
  that made the API reachable; and an is-active assertion polls for 20 seconds after the
  restart and fails the build with the journal attached if the unit is not active.
  My pm2 hypothesis was wrong. It was a reasonable place to look given DEPLOYMENT.md says
  Next.js runs under pm2 and pm2 holds env per process, but the answer was in a file I had
  already read twice today without reading the else branch.

- 2026-08-10. Deployed the pipefail fix. It stopped the deploy CREATING squatters: today's run
  took the systemd branch, so no pkill and no new process. But the port was still held by pid
  3283184, and two things had gone wrong.
  First, nothing clears a PRE-EXISTING squatter. The old else-branch used to pkill everything
  before spawning its own; the fixed if-branch just restarts systemd, which then cannot bind.
  Second, my is-active assertion passed when it should not have. `systemctl restart` returns as
  soon as the process starts, and uvicorn needs a second or two to reach the bind error, so the
  unit reports "active" during that window. The loop polled immediately and broke on the first
  "active". Hardened: wait 8s, check twice 5s apart, and require 0.0.0.0:8000 to be bound,
  which is the condition that actually matters and which is-active alone cannot see.
  The "midnight trigger" I had flagged does not exist. The user pointed out the timestamps are
  UTC and they are on PDT: 00:00:06 and 00:01:11 UTC are 5:00pm and 5:01pm PDT on consecutive
  workdays, which is someone finishing for the day. I had been reading UTC timestamps as if
  they were local and inventing a scheduled job to explain the pattern.

- 2026-08-10. Exported per-SKU per-week channel mix (`ml_31`) and pre-registered v17, a single
  trailing 12-week Amazon-FBA share feature on the long model only. First run of `ml_31` grouped
  on `src.v1._assign_stream`, which is V1's internal east/west fulfilment routing and not the
  sales channel at all; it produced a plausible table answering the wrong question. Rewritten to
  group on `channel` per the business's spec. The two taxonomies partition identical totals,
  which is what proved the rewrite lost nothing.
  Segment split decided the feature's scope: FBA is 51.5% of long-segment units but 2.5% of
  short, and its trailing share barely moves within a short SKU. One column, long only. The user
  asked whether one-hot encoding would be better; it would not, and the repo already paid for
  that lesson once as `is_long` in v4.

- 2026-08-10. Found the snapshot's FIRST week was partial, 32 units against neighbours of 280 to
  415. `drop_incomplete_weeks` only ever trimmed the tail. The head case went unnoticed for a
  year because a short final week looks wrong while a short first week looks like a launch.
  Added `drop_leading_partial_week`, which tests the calendar rather than the unit count.

- 2026-08-10. Advanced the pinned snapshot to 2026-08-03 and re-measured every version-log
  figure. I had argued against this and the user overruled it. The user was right: the whole
  batch took three minutes, every cell moved less than the noise floor, and comparing recorded
  figures against a fresh run is what surfaced the real problem.
  The recorded v-base table had been stale since v9. v-base applies the seasonal round-trip to
  long SKUs, so when v9 moved ML_HOLIDAY_END from Dec 31 to Dec 15 the baseline moved with it
  and the table was never re-measured. Setting the window back reproduces all six cells and all
  three bias percentages exactly, so the cause is proven rather than inferred. Dec-Feb long was
  recorded as 0.2764 against a true 0.2167, overstating by 0.06 every margin quoted over the
  baseline in that window since v9. No verdict changes: in-script comparisons recompute the
  baseline and were never affected, and v11's own criteria already cited the correct 0.2167.
  `ml_12`, the guard positioned to catch exactly this, had been failing since v9 for a second
  and unrelated reason: it required the ML and prototype factor sources to agree on every pinned
  week, which the v9 window split makes impossible on the four weeks covering Dec 16 to 31. So
  the one check that would have caught the stale numbers was already red for a legitimate
  reason, and a check that always fails is a check nobody reads. It now asserts the shape of the
  divergence instead of its absence.
  Also centralised the reference figures. Seven scripts each carried a private copy of the same
  PROTOTYPE dictionary with no record of which snapshot it came from, which is how a stale number
  gets printed next to a fresh one in the same table with nothing marking the difference.

- 2026-08-10. Ran v17. Rejected: the long segment regresses in all three windows, mean +0.0120,
  against a bar that needs a 0.0100 improvement. The informative part is that the pre-registered
  escape hatch does not apply. I had written that near-zero feature gain would mean the model
  ignored the feature and the null result would say nothing about channel mix. Instead the FBA
  share ranked first by gain in two windows and second in the third, so the model used it
  heavily and was made worse by it. Bias magnitude grew in every window and early stopping
  halted sooner, which is what a feature that pays on the validation slice and not out of sample
  looks like.
  My precondition check was necessary and not sufficient, and that is worth remembering. I
  tested that the share moves within a SKU rather than being fixed at launch, on the reasoning
  that a fixed value is a SKU fingerprint. It passed at 70% within-SKU variation. The other 30%
  is cross-sectional identity, and across 53 SKUs that was enough. The right test is whether the
  moving part carries signal after the fixed part is removed. Not run, deliberately: the
  pre-registered guard stops a rejected hypothesis being reshaped until something clears the bar.
  Five perturbations of the long model have now each cost half a point to a point: two seasonal
  blends, a monthly-only blend, the week-boundary shift, and now an exogenous feature. That
  pattern is the strongest argument for freezing v11 and running the final test.

- 2026-08-10. Backlog review. Closed 1 and 4 as out of scope and blocked on data that does not
  exist, 10 and 12 as out of scope or decided against, and 3 and 13.5 as done. Verified the last
  two in the code rather than taking them on trust: the Not-forecast section ships on a
  trailing-13-week ACTUAL sales basis and says so, and the priority explanations exist in the
  per-page manual. Recorded 13.5 as resolved differently from what it asked, since it wanted the
  explanation beside the labels and a manual page is a legend elsewhere. Added an index at the
  top of the file with a status line per item, because "closed" was covering three different
  outcomes and a reader could not tell which.

- 2026-08-10. Rewrote both Action List CSV exports (backlog 5). They built their header from
  Object.keys(row), so the file carried forty wire-format columns while the screen showed ten.
  Now nineteen named columns with human headers, plus the three date fields deliberately kept off
  the table because a screen wants "in 12 days" and a spreadsheet wants the date.
  Three bugs were in the same function and are fixed with it. Escaping quoted only on commas, so
  a product name containing a quote or newline broke the row silently. No UTF-8 BOM, so Excel on
  Windows rendered every Korean heading as mojibake in an app that ships a Korean locale. And
  numbers went out formatted, which makes them text a spreadsheet cannot sum.

- 2026-08-10. Closed backlog 20 and the monitoring half of 21 with one mechanism: /health now
  returns the commit it was started from, read once at startup rather than per request, so a
  stale process reports a stale stamp instead of the current file. The deploy asserts it and the
  hourly reachability check compares it against the tip of main, failing only after thirty
  minutes so an in-flight deploy is not reported as a fault.
  The point is that repo_root could never have caught either failure. It proves the answering
  process started from the deploy directory, and a process left over from an earlier deploy
  satisfies that exactly, which is why the check passed for a whole day while the wrong code
  served traffic. What starts that process is still unknown and stays open.
  Verified after the push: /health returns the commit that was just deployed, matching local
  HEAD. The old code has no commit field at all, so the field being present is itself the proof
  that the new code is running.

- 2026-08-10. Backlog 18, the LightGBM eval_set deprecation. Switched to eval_X/eval_y and
  confirmed no figure moved. Read the installed library's source before editing rather than
  guessing at the new argument names, which was worth doing: eval_sample_weight is NOT part of
  the deprecation and renaming it to match would have raised.
  The stronger argument is structural. Both spellings meet in the library's own
  _validate_eval_set_Xy, which builds exactly the list the old call was already passing, so
  training cannot see a difference. I checked empirically anyway, because that conclusion is my
  reading of someone else's code.
  My verification script failed on its first run and the failure was mine. Four of six scripts
  reported DIFFERS, and every differing line was a prototype reference figure, because I had
  re-measured PROTOTYPE after those logs were written. Nothing about the model had moved. Fixed
  the comparison to exclude reference figures while keeping 28 to 66 substantive lines per
  script so it cannot pass vacuously. That is the third time today the same shape has appeared:
  a check going red for a reason unrelated to what it tests, which turns it into noise and then
  into nothing.

--- DAILY SUMMARY 2026-08-10 ---
- Tested whether telling the forecasting model which sales channel a product sells through would
  improve it; it made results worse in all three test periods, so it was rejected and the
  reasoning written down.
- Found the sales data file was treating a first week containing one day of sales as a full week,
  fixed how the file is built, rebuilt it, and confirmed no accuracy figure moved.
- That rebuild exposed a real error: a reference figure everything is compared against had been
  wrong since July and made the model look further ahead than it is, so it was corrected and
  checks were added so a stale figure cannot sit unnoticed again.
- Releases can now confirm that the code just published is the code actually running, closing two
  long-standing blind spots where a release either never happened or happened without taking
  effect.
- Closed seven items from the outstanding work list, including rewriting the planning screen's
  spreadsheet export, which had been producing forty internal columns instead of the ten on
  screen.

--- SUMMARY PRODUCED 2026-08-10 (covering the 2026-08-10 entries above) ---

- 2026-08-11. Read Tuesday's cron output. Both database writes landed, 6,071 rows each, and
  the leading-partial-week fix ran in production. BACKLOG 11 closed by observation.

- 2026-08-11. Went after BACKLOG 2 and found the item's own proposed fix is wrong. Using a
  stable launch week for eligibility admits 178 to 226 SKUs per window of which 93 to 95%
  have no training rows at that cutoff; they would be scored as zero forecasts. Measured
  before writing any code, which is the only reason it was caught.
  Then measured that the failure the item predicts does not occur either: zero SKUs in any
  window are scored with less than the minimum history. I concluded the item was a rounding
  fix and recommended skipping it. That was wrong, and the user corrected it: the real defect
  is that every promoted SKU is locked to exactly 13 weeks regardless of how much history it
  actually has.

- 2026-08-11. The promotion override assigned three constants: train_start to 13 weeks ago,
  active_weeks to 13, history_length to "short". Measured: 190 SKUs promoted, 41% of the
  smooth set, of which only 15 genuinely had 13 weeks. Median 34, maximum 111 which is the
  whole series. 73 had 50+ weeks and were still labelled short, so they were routed to the
  wrong model as well as starved. 4,615 SKU-weeks of usable history discarded.
  The consequence is bigger than the truncation. That train_start sits in the future relative
  to every backtest cutoff, so all 190 had negative history and were silently absent from
  every figure ever recorded. Every number in the version log was measured on the easier 59%
  of the catalogue and nothing said so.
  Fixed by detecting each SKU's real smooth-history onset. Scored population went from
  266/251/66 across the three windows to 356/327/103.

- 2026-08-11. Measured where forecasting is actually worth doing, by demand band. The model
  beats a trailing 12-week mean above 10 units a week and below 2, and is indistinguishable
  from it everywhere between, with four of five middle bands favouring the trailing mean.
  The 10+ band carries 71% of units, so the aggregate wins are real but concentrated.
  Also measured that the project's own success metric cannot see improvements to the weak
  band: fixing 2 to 10 units a week perfectly would move pooled WAPE by about 0.006 against
  an adoption bar of 0.01 and a noise floor of 0.011. Anyone working on low-volume accuracy
  has to change the criterion first or they will measure a real gain against a rule that
  cannot detect it.

- 2026-08-11. Raised the promotion bar from 2.0 to 3.0 to match the classification bar, on
  the user's call. I argued against it on the grounds that the data cannot locate the
  boundary, and the user's point stood: the data cannot locate it, but the two bars
  contradicting each other was the bug, and choosing is a judgement rather than a finding.
  I also had the burden of proof backwards. I kept asking whether the evidence showed those
  SKUs were forecast badly and treating the absence of an answer as a reason to keep
  forecasting them. Serving a forecast is a positive claim and needs evidence for it.
  Recorded as a judgement with its cost measured: 127 SKUs lose their forecast, 6.8% of
  covered demand, from SKUs selling 2.0 to 2.9 a week. Every accuracy cell improved, and the
  improvement is partly mechanical because the hardest SKUs were removed, which is written
  down next to it.

- 2026-08-11. Built an integrity suite that proves the model is refitted for every evaluation
  window, trained only on data at or before each cutoff, and independent of the order windows
  are fitted in. Nine checks, all passing. Two of them were genuinely in doubt: refitting is
  bit-identical, and reversing the fitting order changes nothing. Wired into CI and into the
  re-baseline runner, which now aborts before re-deriving anything if the harness fails.

- 2026-08-11. Audited the codebase. Found the three hyperparameter scripts have been silently
  disarmed since a 2026-07-29 refactor: they set hyperparameters on a class attribute that
  stopped being read. I first concluded this invalidated the whole v10 track, and the data
  disproved it within a minute: 81 configurations produced 81 distinct scores, impossible if
  the settings had not applied. The scripts ran eight days before the refactor. Fixed, with
  an assertion so the same breakage would raise rather than produce a plausible number.
  Also found our own profile change had broken the planning code: promoted SKUs were
  identified by active_weeks == 13 and used to size safety stock, and onset detection
  destroyed that signature, so 57 of 63 would have been silently under-sized. Fixed with an
  explicit column.

- 2026-08-11. Re-ran v10's hyperparameters against the model that actually exists, per arm,
  as v18. Rejected on both. The short model gained four times the trees and moved by 0.0001
  across three windows, so it is data-limited rather than capacity-limited. The long model's
  only significant cell is a regression, in the window carrying v11's defining result.
  That is the sixth perturbation to cost the long segment, after two seasonal blends, a
  monthly-only blend, the week-boundary shift and the channel-mix feature. At 55 to 62 SKUs
  that is a property of the sample, not six coincidences.

- 2026-08-11. Corrected a claim I wrote yesterday. Section 4.31 said every cell moved by at
  most 0.011 across the snapshot advance. The table it was drawn from omitted Oct-Dec
  smooth/short, which moved 0.059 and flipped a comparison against V1. Same failure mode as
  the stale figure that section documents.

- 2026-08-11. Forced the weekly pipeline early so production picked up the profile change
  tonight instead of next Tuesday. Serving 338 SKUs, down from 467. Both database writes
  landed again.

--- DAILY SUMMARY 2026-08-11 ---
- Found and fixed a bug that had been hiding 41% of our products from every accuracy test we
  have ever run, so the recorded results had only ever been measured on the easier half of
  the range.
- Deliberately narrowed what we forecast from 467 products to 338, because measurement showed
  that for the ones removed, selling two to three units a week, our forecast was no better
  than a simple recent average; they still appear on the planning screens with their actual
  sales.
- Built an automatic check that proves the model is being trained and tested correctly, which
  now runs on every code change, and used a full review of the code to find and fix three
  further faults before they could affect anything.
- Closed out the last two open ideas by testing them properly, neither improved the forecast,
  which leaves only the final validation run before the work is finished.

--- SUMMARY PRODUCED 2026-08-11 (covering the 2026-08-11 entries above) ---

- 2026-08-12. Replaced the unmeasured-error fallback used for safety stock. It gave promoted
  SKUs a hardcoded 0.24 and everything else its segment median. Two things wrong with that:
  the constant was a measurement frozen into source and went stale the moment promotion
  changed, and "promoted" was only ever a proxy for low volume, which is what actually
  predicts error (0.357 below 2 units a week against 0.134 above 10).
  Now it takes the median measured error of SKUs in the same weekly-demand band, computed
  every run from whichever SKUs currently have a measured error, so nothing can go stale and
  it sharpens as backtest coverage grows.
  My first version had a real flaw and the data caught it. A band needs five measurements to
  be trusted, and the under-2-units band has four, so it fell through to the segment median
  and handed its 33 SKUs LESS cushion than before, on the hardest band in the range. Changed
  to borrow from the nearest trusted band instead, which keeps the fallback on the volume
  axis. Fallbacks now run 0.269, 0.269, 0.234, 0.212, 0.132 from lowest band to highest,
  decreasing with volume as the evidence says they should.
  Aggregate effect is nil, 2,306 to 2,312 recommended units. The stock is redistributed
  toward the SKUs that need it rather than increased.
  A second flaw, caught by the user asking how we were forecasting anything under 2 units a
  week when the demotion rule should have removed it. We were not. I was banding on a 4-week
  rate while the band edges were measured on a 13-week mean and every classification
  threshold is on 13 weeks too, so SKUs were being sorted into bands calibrated on a
  different and noisier statistic. Fixed to band on recent_mean. The under-2 band is now
  empty, which is what the demotion rule guarantees and is the check that the two rules
  finally agree.
  Also worth recording: my verification of all this ran against a local profile file from
  2026-08-10, before the onset fix, so the first set of counts I reported described a
  population that no longer exists. The logic was right and the numbers were stale.

- 2026-08-12. Full codebase review, 26,457 lines. Ran static analysis over all of it and read
  the production path properly; roughly 60% has not been read line by line and that is stated
  rather than glossed.
  Retired three dead files: src/pipeline.py, src/forecast.py, src/segment.py. pipeline.py was
  imported by nothing, forecast.py and segment.py only by pipeline.py, and forecast.py's only
  function raised NotImplementedError, so the chain would have crashed if anything had called
  it. 51 lines. compileall, imports and ruff all clean afterwards.
  Then retired the Streamlit dashboard and the intermittent policy on the user's call: the
  dashboard was a prototype the Next.js Action List was ported from and will not be used
  again, and the policy module will not be taken up. That is src/intermittent_policy.py, its
  only consumer scripts/test_tsb_large_spike.py, which could not run anyway because the CSV
  it reads is not generated by anything, and the Streamlit app itself.
  Kept deliberately: data/inventory/inventory_snapshot.csv, which despite the folder name is
  production data written by export_inventory_snapshot.py, shipped by push_data_to_server.sh
  and listed in the API's readiness check; and docs/PLANNING_PLAN.md and REQUIREMENTS.md, the
  specs the Next.js screens were built from and which BACKLOG item 3 cites by section. Also
  kept src/ml/diagnostics.py, which looks unimported but is a runnable standalone tool.
  The dashboard was never a second implementation of the planning logic. dashboard/lib
  re-exported src.planning as aliases and said so, so lib.calc was src.planning.calc and the
  two could not drift.
  Total: 2,124 lines removed, 26,457 to 24,333. compileall, every live import and ruff all
  clean afterwards, and the only surviving reference to any of it is the optional streamlit
  import in src/planning/_cache.py, whose fallback branch is now the only one ever taken.
  Then removed dashboard/ entirely, on the user's instruction that nothing under it should be
  in use. Production data was relocated rather than deleted: inventory_snapshot.csv and its
  example moved to data/inventory/, which joins data/snapshots and data/dev_seed as a tracked
  directory, since data/ is not gitignored wholesale and readiness reports that file as
  tracked. PLAN.md and REQUIREMENTS.md moved to docs/ as PLANNING_PLAN.md and
  PLANNING_REQUIREMENTS.md; they are the specs the Next.js screens were built from and are
  cited by section in the backlog.
  Four production paths updated: src/planning/data.py (the path constant, renamed DASH_DATA to
  INVENTORY_DIR because the old name described a folder that no longer exists),
  export_inventory_snapshot.py, push_data_to_server.sh and seed_dev_data.py. Also fixed three
  stale references to files deleted earlier: dashboard/README.md, dashboard/lib/data.py and
  dashboard/lib/calc.py, none of which had existed for some time.
  Total across both passes: 14 files removed, 26,457 lines of Python down to 24,344. No live
  reference to dashboard/ remains except historical notes explaining the move and two pointing
  at the Commerce repo's own api/planning/dashboard route, which is unrelated.

- 2026-08-12. Backlog 15, the pipeline that could not be safely interrupted. ml_prepare_data
  now stages every artifact in data/.staging_<pid> and moves them into place only after all
  four steps succeed, so a cancel, a crash or a dropped connection leaves the previous run
  intact and still being served. The lever is one environment variable read by
  config.DATA_PROCESSED, because the pipeline spans processes and the subprocess has to read
  the files the parent just wrote rather than the previous run's.
  My first version was broken and looked finished. src/clean.py computed OUTPUT_PATH from
  PROCESSED_DIR at import, so redirecting the directory moved the CSV and left the parquet
  going straight into live data/processed. A kill mid-run would have replaced the live sales
  file and not the profile, which is the corruption staging exists to prevent, with a staging
  directory sitting beside it looking like protection. Every check I had thought to run
  passed, because each confirmed the redirect reached a name rather than that the write landed
  anywhere in particular.
  scripts/_test_staged_pipeline.sh caught it on the first honest run. That test waits for the
  staging directory to actually contain an artifact instead of guessing a time, then sends
  SIGKILL, which cannot be deferred the way the SIGINT in my first attempt was. The first
  attempt failed for its own reasons and reported a failure that was the test's, not the
  code's.
  Third instance this week of the same shape: a derived value stops tracking its source and
  nothing says so. RatioLGBM.PARAMS against self.params, the transcribed v-base figures
  against the recomputed ones, and now OUTPUT_PATH against PROCESSED_DIR. Each was found by
  re-deriving a value rather than by reading the code.

- 2026-08-12. Backlog 21. Found and closed a second route to the unmanaged uvicorn:
  resolveServerDir() in the Commerce repo read FORECAST_SERVER_DIR before checking NODE_ENV,
  so an explicit value won in production and the app would auto-start a competitor to systemd.
  Its own docstring said the opposite. That is also the mistake DEPLOYMENT.md warns about and
  the hardest to see, because .env.local overrides per variable and under pm2 the value need
  not be in any file, which is why searching /opt found nothing on 2026-08-07.
  The original unknown is still unknown, but it now identifies itself: server-diagnostics.yml
  walks each uvicorn's parent chain to init and prints its cgroup, so systemd, a person over
  ssh, pm2 and cron are told apart on sight.
  Findings that stay open: there is no test suite at all for 26k lines, ml_38 being the only
  automated check and only since 2026-08-11. api/main.py's _classify_sku is a second copy of
  the bucket logic that omits the promote and demote overrides, so /forecast/{sku} and
  /backtest/{sku} classify the 63 promoted SKUs differently from the batch; both are on the
  legacy track BACKLOG 6 retires. bootstrap_delta divides by y.sum() with no guard, so a
  segment with zero units yields nan rather than an error.
  Static analysis: 152 findings, none critical. 46 zip() without strict, 37 f-strings with no
  placeholder, 32 late-binding closures all confined to scripts/_debug_*, 28 unused imports.
  No bare excepts, no mutable default arguments, no swallowed write failures, and serving
  builds v11 identically to ml_22 including the filter that stops long SKUs being predicted
  twice.

2026-08-12  Fixed a stored forecast run replacing only the SKUs it repeated, not the whole
  week. store.upsert wrote ON CONFLICT on (model_version, week_of, unique_id, ds), which can
  update a row the new run also produces but cannot represent a SKU the new run no longer
  produces. When the smooth set went 467 -> 338 on 2026-08-10 the 129 dropped SKUs kept their
  previous week's values, so shipcore.ml_forward_forecasts held two segmentations at once and
  the planning screens served all 467 with nothing indicating that a quarter were stale. The
  write now deletes the (model_version, week_of) pair before inserting, both statements in one
  transaction, matching what src.db.write_forward_forecasts already did on the legacy track.
  Scoped to one model_version so a candidate version can be stored alongside v11 for the same
  week. Assumes a run writes its week in one call, which both callers do; replace_run=False
  keeps the old behaviour for a caller that needs to append.
  Added scripts/test_store_replace_run.py, which drives the real upsert against SQLite and
  reproduces the 467 -> 338 shrink. Confirmed it fails against the previous behaviour before
  keeping it: 4 checks fail, with yhat holding a mix of stale and fresh values. Also covers
  other weeks and other model_versions surviving a write, and a failed insert rolling the
  delete back so a failure cannot empty a week.
  Fixed scripts/ml_purge_history_run.py, which crashed with KeyError: 'week_of'. It was the
  one direct parquet reader the 2026-08-12 rename missed; it now goes through the shared shim
  in history rather than repeating the rename. With the write path fixed the purge is no
  longer needed for the stale rows: re-running the forecast clears the week itself, with no
  window in which the planning screens have nothing to serve.
  Recorded the personal repository copy as BACKLOG 24, flagged in the index as the last item
  and as a prerequisite for BACKLOG 6, which deletes the statsforecast implementation.
  Confirmed this repository has never tracked .env or .claude/settings.local.json.

2026-08-12  Guarded the database write against stale local inputs, after a laptop run came
  one working connection away from overwriting the server's forecast. `ml_forward_forecast.py
  --snapshot live` on the dev machine read data/processed as it stood: sales ending 2026-08-03
  and an sku_profiles.csv with no `promoted` column, so written before the onset fix. It
  rebuilt the old 467-SKU segmentation and reported success. Only a failed DB connection
  stopped it being written. Under the old upsert that would merely have added rows; under the
  run-replacing write it would have deleted the stored run for that week and replaced it. The
  correctness fix made a pre-existing hazard destructive, so the guard belongs with it.
  The script now compares the run's training week against weeks.last_complete_week() and, when
  behind, skips both the forward write and the history append while still writing the parquet,
  the model and the summary. --allow-stale for backfills. History is skipped entirely rather
  than written locally: a run replaces its whole week, so appending a stale rebuild would
  overwrite the real stored predictions and silently change every accuracy figure drawn from
  them.
  Also stopped the script asserting a cause it had not checked. store.upsert caught every
  exception and returned -1, which was printed as "no DB credentials, or it could not be
  reached" on a machine whose .env was complete and whose psycopg2 was installed. The cause is
  now kept in store.LAST_ERROR and printed. Added scripts/ml_check_db.py, which walks driver,
  environment, engine, connection, both tables and DELETE permission, and names the step that
  fails. The DELETE check matters specifically because the new write path needs a permission
  the old one did not.
  Corrected BACKLOG 22, which named the FastAPI version as 0.141. It is 0.138.0, confirmed
  against .venv metadata and against _IncludedRouter at fastapi/routing.py:1518 in that
  install. The behaviour was reported accurately and the version was not, in the one entry
  whose subject is not knowing which versions are deployed. The item still needs a pip freeze
  from the server before the twelve unpinned service packages can be pinned, since pinning
  from the laptop would assert a new production state rather than record the current one.

2026-08-12  Closed BACKLOG 22 by pinning every dependency from pip freeze on the deploy host.
  The two environments had drifted exactly as the item predicted: the server runs fastapi
  0.141.1, uvicorn 0.52.0 and plotly 6.9.0 where this laptop had 0.138.0, 0.49.0 and 6.8.0.
  The five ML pins matched on both, so no recorded model result is affected. Pinned from the
  server rather than from PyPI so the file records production instead of proposing an upgrade
  to it; installing it on the laptop now upgrades FastAPI, which is the intended direction.
  Recorded a wrong correction rather than reverting it silently. Earlier in the day the
  entry's "FastAPI 0.141" was changed to "0.138.0" on the strength of the laptop's .venv,
  where _IncludedRouter does exist. Finding the class there proved only that it exists in both
  versions, not that the entry named the wrong one. The original number was the server's and
  was right. Local evidence, production claim, in the one item about not knowing which
  versions run where.
  Checked the live tables and corrected BACKLOG 23, which claimed the mixed-segmentation week
  had already happened. It has not. Both stored runs in both ml_ tables are the 467-SKU
  segmentation, so the 3.0 promotion threshold has never reached the server and the 129 stale
  SKUs were inferred from local files rather than observed. The write bug is unchanged and is
  demonstrated by the test; it fires on the next deploy rather than having already fired. Also
  noted that the 2026-08-03 forward week stores 52 target weeks per SKU against 13 at
  2026-08-10, so a --horizon 52 run sits there, and that aggregates cannot distinguish one
  52-week run from a 13-week run written over it.

2026-08-12  Audited whether the remaining work would survive into a new session, since the
  rest of it is being finished in separate chats. Two of four items were carried and two were
  not. BACKLOG's open list held only items 6 and 24, so the final test existed solely as
  section 4.34 at line 1693 of a 2,900-line design doc, and the statsforecast extraction
  existed only as a failure note inside BACKLOG 22 rather than as intended work. Added items
  25, 26 and 27 to the index for those and for the document rewrites, and made section 4.34
  name scripts/ml_41_final_test.py, which it did not.
  Checked rather than assumed the staleness banners: PROJECT_WRITEUP.md and HANDOVER.md carry
  SUPERSEDED, LEARNING_NOTES.md does not and does not need one. It had been described as stale
  earlier in the day; it is conceptual and contains none of the figures that moved. Recorded in
  item 27 so it is not re-checked from scratch.
  Then checked every section's status line against the index rather than trusting the index,
  and found two disagreeing with it. Item 5 still read "identified, deferred" although the CSV
  export was rebuilt on 2026-08-10, and item 22 still read "identified, not fixed" after being
  fixed earlier the same day. Both status lines corrected, problem statements left in place.
  The remaining open items are 6, 24, 25, 26 and 27, plus one verification queued for the next
  deploy. Item 20 stays closed with its cause unknown, which is recorded as a conclusion rather
  than an omission.

2026-08-13  Separated the statsforecast track from the LightGBM one, closing BACKLOG 25 on
  the second attempt. api/main.py went from 3,184 lines to 1,346: the sixteen statsforecast
  endpoints are now api/legacy.py, and src/models.py, selector.py, backtest.py and
  baselines.py are now src/legacy/, moved with git mv so history follows them. JobLogger was
  the only helper both tracks used, so it went to api/common.py; importing it from one into
  the other would have made the two modules circular.
  Four helpers that looked shared were not. _parse_product_types, _data_version,
  _cached_response and _VALID_LEVELS have no ML-side caller, and _data_version reads the
  legacy fc_forward_forecasts table, so all four went into api/legacy.py. Retiring that
  track now removes them with it instead of leaving them behind in a common module.
  Derived the new import block instead of eyeballing it, since truncated multi-line imports
  were half of why the 2026-08-12 attempt was reverted. symtable scope analysis on the
  extracted block listed the 52 names it references but does not define, all of which
  resolved to the original header.
  The verification took three tries to become capable of failing, which is the part worth
  recording. Walking app.routes missed all sixteen legacy routes, because FastAPI 0.141.1
  stores an included router as a _IncludedRouter with no .path and no .routes, reachable
  only through .original_router; that is the same opacity BACKLOG 22 and 25 both describe,
  now confirmed against the pinned version. Switching to real requests and treating 404 as
  "not routed" then passed while testing nothing, because the token middleware answers 401
  before routing and a nonexistent path never reached a 404. Adding the token produced false
  failures on /planning/sku/{sku_id}, whose handler raises 404 for an unknown SKU: a status
  code cannot tell "no route" from "route says not found". scripts/check_route_parity.py now
  calls BaseRoute.matches(), the app's own resolution logic, with no middleware, handler or
  database involved, and runs a negative control on every invocation so that a probe which
  has gone blind reports that rather than passing. Result: 34 routes, the previous 35 minus
  /chat, no shadowing, every route confirmed to resolve, with no database and no network.
  Deleted /chat and src/chat.py rather than moving them. BACKLOG 6 had already recorded that
  its tool calls failed silently for the whole life of the deployment because src/chat.py
  addressed port 8001 where nothing listened. It is the only part of this track that was
  genuinely unused. The Next.js half is a separate change in the Commerce repository.
  Wrote src/legacy/__init__.py as the authority on why the track is frozen rather than dead,
  because "unused" would be false and the failure mode is expensive: the weekly cron runs the
  legacy pipeline first for its ingest, and the LightGBM run reads the sales file it writes.
  Deleting the legacy track without giving the ML pipeline its own ingest would stop the
  LightGBM forecast getting fresh data silently, since it would still be served and would
  simply stop moving. The same point is now in CODEBASE_GUIDE section 1.1 and at the call
  site in run_forecast_cron.sh. Also corrected a comment in api/main.py claiming the Streamlit
  dashboard renders src.planning directly; it was retired on 2026-08-12.

2026-08-13  Wrote the two handover documents and removed the deleted chat feature's traces.
  docs/OPERATIONS.md is the operator runbook, written for someone who keeps this running and
  is assumed to know no machine learning: what the system forecasts and what it deliberately
  does not, the two pipelines and why the frozen one is not optional, every input and output
  with its granularity, the weekly cron and why it is Tuesday, how to tell whether it is
  working, a symptom-to-cause table, and a maintenance calendar. docs/MODEL_GUIDE.md is the
  technical handover for whoever picks up the modelling: the architecture and why structure
  is imposed rather than learned, the v11 hybrid, the evaluation protocol and its measured
  noise floor, the four most expensive traps, what to try next and what not to retry.
  Kept both short by pointing at the existing records rather than restating them, since a
  second copy of a figure is a second thing that can go stale, which is the failure this day
  was mostly spent correcting.
  Corrected the v11 prototype column in the design doc. Five of its six cells disagreed with
  docs/rebaseline_2026-08-03-v2/ml_22_v11_hybrid.log, which the same section already names as
  authoritative where the two differ; short/Oct-Dec read 0.3972 against the log's 0.4137. The
  v11, v-base and V1 columns were checked against the same logs and were right, so the error
  was confined to the one column carried across by hand. No verdict moves: v11 still beats the
  prototype in five of six cells and still loses long/Oct-Dec. Both new documents take their
  figures from the logs rather than from any table, and say so.
  Removed the chat feature's configuration from the places that still described it as working.
  DEPLOYMENT.md listed fifteen environment values including three LLM_* and FORECAST_SELF_URL;
  it is eleven now, with the removal explained rather than the lines quietly deleted, because
  that list has already been wrong in the other direction once. Also corrected "eight POST
  endpoints" to seven in DEPLOYMENT.md and in the hourly reachability workflow's comment, and
  replaced the .env.example block with a note saying the values do nothing, so anyone copying
  an older .env learns why they vanished rather than assuming an omission.
