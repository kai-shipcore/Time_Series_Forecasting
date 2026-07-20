# Demand Forecasting: LightGBM Model Design Document

**Status:** living document, updated with every design iteration.
**Scope:** the machine-learning (LightGBM) forecasting track in `src/ml/` and `scripts/ml_*.py`.
**Audience:** written to be readable without prior knowledge of the forecasting codebase.
**Companion document:** `CODEBASE_GUIDE.md` explains the files and code that implement this
design. This document covers decisions and rationale; the companion covers implementation.

---

## 1. Goal & Metric

### 1.1 Background and context

The company forecasts weekly unit demand for roughly 3,300 active SKUs (vehicle seat covers
and accessories) to drive inventory planning. Two forecasting systems exist today:

- **V1 (current production method):** the spreadsheet formula the company operates on
  today. It blends recent sales velocity over 7, 15, 30, 60, and 90-day windows and scales
  the result by hand-set monthly seasonal multipliers. Benchmarking in this project shows
  that it underforecasts chronically (a bias of −33 to −39% across smooth SKUs, and −43 to
  −45% on the short-history segment defined below) because trailing velocity averages
  cannot anticipate growth.
- **The statistical prototype (phase 1 of this project):** per-SKU models from the
  `statsforecast` library (AutoARIMA, AutoETS, moving averages), selected by
  cross-validation for each SKU. Seasonality is handled by dividing the hand-set monthly
  multipliers out of the data before fitting and multiplying them back into the forecasts
  afterward. The prototype is not yet in production, but it beats V1 wherever both were
  measured. On full-history SKUs it achieved pooled WAPE 0.159 versus V1's 0.259, with
  bias +0.2% versus −6.7%, across 6 cross-validation windows
  (`outputs/reports/v1_comparison.csv`). On short-history SKUs it achieved pooled WAPE
  0.222 versus 0.314, with bias +14.0% versus −20.4%, in the April to June 2026 backtest
  stored in the `fc_forward_forecasts_test` database table (339 SKUs, 13 forecast weeks,
  recomputed against current actuals; consistent with the 0.218 versus 0.291 recorded in
  `config.py` when the backtest was originally run). The prototype is the accuracy bar the
  ML model must clear. Note that the prototype's short-segment bias in that backtest was
  +14%: it corrects V1's chronic underforecasting but overshot in that window, which is
  useful context for the bias goals below.

**SKU segmentation.** Not every SKU can be meaningfully forecast. Each SKU is classified at
training time by its demand pattern:

- **Intermittent** (approximately 3,000 SKUs): demand is sporadic, with 30% or more of
  recent weeks showing zero sales. No forecast is produced for these SKUs; they are out of
  scope for this project.
- **Smooth** (approximately 450 SKUs): demand is regular enough to model. Although they are
  only about 13% of SKUs, they carry about 83% of total unit demand, which is why this
  segment is where forecast accuracy matters commercially.

A note on how counts are quoted in this document. Catalog composition is a moving
quantity: the weekly refresh reclassifies SKUs as they launch, go dormant, and cross the
segmentation thresholds, and the smooth population moved by fifteen SKUs in a single
refresh during this project. Catalog-level figures are therefore given as approximations
and should not be cited as exact. Evaluation figures are different. Once the ML inputs are
pinned to a data snapshot (Section 2.1), the eligible population of each evaluation window
is a fixed and reproducible property of that snapshot, so per-window counts are stated
exactly. Counts drawn from stored artifacts of past runs are likewise exact, because those
artifacts do not change.

Smooth SKUs are further split by history depth, because history depth determines what a
model can learn from them:

- **smooth/short** (approximately 360 SKUs, the large majority): fewer than 50 weeks of
  active sales history. This is too little data for per-SKU model selection, so the
  statistical prototype forecasts these SKUs with a 12-week moving average ("WA12"). WA12
  is accurate on stable SKUs but structurally lags growth.
- **smooth/long** (approximately 80 SKUs): 50 or more weeks of history. This category
  merges the prototype's internal "medium" group (50 to 104 weeks) and "full" group (104 or
  more weeks). They are treated jointly in this project because the prototype handles them
  nearly identically, and because the medium group on its own is fewer than thirty SKUs,
  too small to evaluate reliably. The prototype forecasts these SKUs with its
  cross-validation selected statistical models.

### 1.2 Goal

Build one global LightGBM model that forecasts weekly demand for all smooth SKUs and beats
the strongest existing method in each segment: WA12 on smooth/short, and the
cross-validation selected statistical models on smooth/long. V1, the current production
method, is reported alongside as the business-as-usual reference. "Global" means a single
model trained across all SKUs jointly, so that patterns learned on one SKU (for example,
how demand ramps up after launch) transfer to others. This is the main structural advantage
a machine-learning model has over the current per-SKU approach, given that no individual
SKU has much history.

The longer-term motivation is extensibility. A feature-based model can absorb external
signals such as Google Analytics (GA4) site traffic, stockout records, and marketplace
analytics, which per-SKU statistical models structurally cannot use. The near-term
deliverable is a baseline model built from sales history alone; external data is added
only after this baseline has demonstrated measurable value.

**Architecture stance.** A central design decision is how much of the forecasting problem
the model should learn from data, and how much should be imposed as fixed structure. The
two largest public forecasting competitions provide the relevant evidence, and they point
in different directions depending on data depth.

In the M5 competition (2020), participants forecast daily unit sales for approximately
30,000 Walmart products using more than five years of history. The winning solutions were
LightGBM models that received calendar information (day of week, month, holiday and
promotional events) as ordinary input features and learned every seasonal effect from the
data. This worked because five years of history contain five to six observations of each
annual event, which is enough for the model to estimate the effects reliably.

In the M4 competition (2018), participants forecast 100,000 series that were typically
short and had no accompanying covariates. The winning method was a hybrid: an
exponential-smoothing component estimated each series' level and seasonality, and a neural
network learned only the residual dynamics, from data that had first been normalized and
deseasonalized. Notably, entries that attempted to learn all structure directly from the
raw series performed poorly.

Our dataset resembles M5 in problem type (retail unit demand for related SKUs, with
external covariates planned) but resembles M4 in history depth, since two years of data
contain only two observations of each seasonal event. We therefore follow the M4
architecture: per-SKU scale and seasonality are imposed structurally, the latter using the
existing hand-set monthly multipliers, and LightGBM is limited to learning the residual
dynamics that the data can support, such as growth ramps and lifecycle effects. An early
experiment in this project confirmed the necessity of this restriction. When LightGBM was
given calendar-derived features and allowed to learn seasonality itself, it reproduced the
behavior of the specific months it had seen rather than a general seasonal pattern, and
overforecast the post-holiday trough by +123% (see the Decision Log, Section 4.9).

LightGBM is retained as the machine-learning component for two reasons. First, the planned
external data sources (GA4 traffic, stockout records, marketplace analytics) enter the
model as tabular features, and gradient-boosted trees are the strongest established method
for learning from tabular features. Second, the cost of failure is low: the statistical
prototype remains the proposed replacement for V1 unless the machine-learning model
demonstrably outperforms it.

### 1.3 Primary metric: pooled WAPE per segment

All models (LightGBM, WA12, V1, and the statistical models) are scored identically, on
identical data splits, through one scoring function (`src/ml/evaluate.py`).

For a given 10-week evaluation window, each SKU's forecast total and actual total are
summed over the window. Then, for each segment:

```
pooled WAPE = Σ_SKUs | forecast_total − actual_total |  ÷  Σ_SKUs actual_total
```

Lower is better; 0.20 means the segment's forecasts were off by 20% of its demand.
"Pooled" means SKUs are weighted by their demand volume: an error on a 200-unit-per-week
SKU counts one hundred times as much as an error on a 2-unit-per-week SKU. This mirrors
the commercial cost of forecast error, and it matches the accuracy convention used
throughout the statistical prototype's evaluation record, so results are directly
comparable to it.

### 1.4 Secondary metrics

- **Bias**, defined as (Σ forecasts ÷ Σ actuals) − 1, per segment. Bias measures
  systematic over-forecasting or under-forecasting. For context: V1's chronic
  underforecasting is a known business problem (−33 to −39% on smooth SKUs overall in this
  project's benchmarks, and −43 to −45% on the short-history segment). The statistical
  prototype corrects the chronic underforecast but with imperfect calibration: +0.2% on
  full-history SKUs over 6 CV windows (`v1_comparison.csv`), −5.0% pooled across smooth
  SKUs on its stored held-out test run (`test_evaluation.csv`, an earlier data snapshot),
  and +14.0% on short-history SKUs in the stored April to June 2026 backtest
  (`fc_forward_forecasts_test`). The goal is fixing underforecasting without overshooting
  into overforecasting.
- **Ramp-cohort WAPE and bias**: the same metrics restricted to SKUs whose recent 4-week
  average demand is at least 1.2 times their 12-week average at forecast time, meaning
  SKUs that were visibly growing when the forecast was made. This is the growth-tracking
  diagnostic. Trailing-average methods (V1, WA12) structurally underforecast this cohort,
  and improving it is a primary motivation for the ML model. Implemented in the shared
  harness (`src/ml/evaluate.py:ramp_cohort` and `cohort_score`) with the standard
  eligibility filter applied.

### 1.5 Decision rule for design changes

Evaluation uses three development windows spanning different seasons (defined in Section 2). A
design change (a feature, hyperparameter, or structural choice) is adopted only if it
improves pooled WAPE with a consistent sign across all three development windows and shows
a three-window mean improvement of at least 0.01. A difference on a single window of less
than 0.02 is treated as inconclusive on its own. An SKU-level bootstrap measurement (1,000
resamples of the SKU population; `src/ml/evaluate.py:bootstrap_delta`) put the sampling
noise of a single-window paired WAPE difference at roughly ±0.011 to ±0.014 standard
deviation, so single-window differences of that size are indistinguishable from chance.
For borderline calls, the same bootstrap decides: a change is adopted only if its mean
improvement exceeds twice the bootstrap standard error of the paired difference. When a
change fails these criteria, the simpler design wins by default.

**The "significant" column** in the Section 6 version tables reports whether a single
window's difference is distinguishable from SKU sampling noise, by the same
two-standard-error threshold applied to that one window: `|delta| > 2 x se`, where both
come from `bootstrap_delta`. It is a noise test, not an adoption test; adoption requires
the sign consistency and three-window mean above. The rule is implemented as
`src/ml/evaluate.py:is_significant` so that the tables cannot drift from it. It was
previously undefined, which meant the labels could not be reproduced or checked. The
alternative reading, that the 95% interval excludes zero, agrees with this rule on all 24
version, window and segment cells measured on the pinned snapshot, so the choice does not
currently change any label. The smooth/long segment's
former "medium" subgroup, fewer than thirty SKUs, is reported for completeness but never
decides an adoption on its own. For the smooth/short segment, the Oct-Dec 2025 window is
excluded from the decision process entirely: only 14 short SKUs were eligible at that
cutoff, too few to produce a meaningful result, so short-segment decisions rest on the
Mar-May window and the Dec-Feb window plus the bootstrap (see Section 4.16). Every adoption or rejection is recorded
in the Decision Log (Section 4) with its evidence.

### 1.6 Success bar

For each segment, the LightGBM model qualifies as the proposed method when it achieves
pooled WAPE less than or equal to the statistical prototype's on all three development
windows and the held-out final test window, with absolute bias no worse than the
prototype's. The decision is made per segment: for example, the ML model may become the
proposal for smooth/short while the statistical models remain the proposal for
smooth/long. Both tracks already clear V1, the current production method, by a wide
margin; V1 is reported as the business-as-usual reference in all evaluations.

---

## 2. Data & Splits Protocol

### 2.1 Data sources and preparation

The model trains on the same processed artifacts as the statistical prototype, produced by
the existing ingestion pipeline:

- **`data/processed/sales_clean.parquet`**: weekly unit sales per SKU. The source is the
  `shipcore.fc_velocity_link_snapshot_forecast` database table (complete order history,
  all sales channels), aggregated to calendar weeks. Weeks follow the codebase's `W-MON`
  convention: each week is labeled by the Monday on which it ends, so a label of
  2026-07-13 refers to the week ending Monday, July 13. The data forms a complete grid:
  every SKU has a row for every week, with zero filled in for weeks without sales.
- **`data/processed/sku_profiles.csv`**: each SKU's segment classification (bucket,
  history length, training start date), produced by the profiling stage.

**The ML track reads pinned copies of both files, not the live ones.** The weekly cron
rewrites `data/processed/` in place, revising recent actuals as late orders register and
regenerating the profile snapshot. The ML harness therefore reads from
`data/snapshots/<date>/`, selected by `ML_DATA_SNAPSHOT` in `config.py` and currently set
to 2026-07-20. The production pipeline continues to read `data/processed/` and continues
to follow the weekly refresh; the two paths share no file. Advancing the snapshot is a
deliberate change that requires re-baselining recorded results, in the same way as
advancing the window anchor. See the Decision Log, Section 4.21.

Two preparation rules are applied on load, identical to the statistical prototype's
backtest (`src/backtest.py`), so that both model families always see the same data:

1. **Ramp-up trimming.** For SKUs that launched partway through the data window, weeks
   before the SKU's `train_start` date are dropped. Pre-launch zeros are not demand
   history and would distort any feature computed from them.
2. **Trailing-week trimming.** The most recent `TRIM_TRAILING_WEEKS` weeks (currently 0)
   can be dropped because late-registering orders make the tail unreliable. The setting
   is honored wherever it is set in `config.py`.

The database is the system of record but is not directly reachable from every working
environment, so database tables are accessed through export scripts
(`scripts/export_forecast_history.py`) that snapshot them to `data/processed/*.parquet`.
One lesson is recorded here for future readers: actual table schemas differ from older
internal documentation (for example, `fc_forecast_history` stores one row per SKU per run
with a 13-week total, not per-week rows), so the export script discovers schemas at
runtime rather than assuming them.

### 2.2 Evaluation windows

Models are evaluated by rolling-origin backtesting: pick a cutoff week, train on
everything up to it, forecast the following 10 weeks, and score against what actually
happened. Four such windows are defined on the current data snapshot (sales through the
week ending 2026-07-13):

| Window | Training cutoff | Test period | Role |
|---|---|---|---|
| Final test | 2026-05-04 | 2026-05-11 to 2026-07-13 | **Quarantined.** Used once, as the final go/no-go gate. |
| Mar-May 2026 (dev) | 2026-02-23 | 2026-03-02 to 2026-05-04 | Development (spring) |
| Dec-Feb (dev) | 2025-12-15 | 2025-12-22 to 2026-02-23 | Development (post-holiday trough) |
| Oct-Dec 2025 (dev) | 2025-10-06 | 2025-10-13 to 2025-12-15 | Development (Q4 peak) |

All feature, hyperparameter, and structural decisions are judged on the three development
windows only. The final test window is reserved for the finished model, because any window
that informs design decisions gradually stops being a fair test: iterate against it enough
times and the design becomes fitted to that window without ever training on it. The three
development windows intentionally span three distinct seasonal regimes, so a design change
must work across seasons to be adopted.

Two protocol notes:

- **Windows are pinned to a fixed anchor date.** The source data refreshes weekly, so the
  windows are anchored to a fixed date (`ML_FINAL_TEST_CUTOFF` in `config.py`, currently
  2026-05-04, the last training week of the final-test split) rather than to the latest
  week in the data. Every window is derived by stepping back from this anchor, so a weekly
  refresh cannot move them. New weeks that arrive after the final-test window are ignored
  until the anchor is advanced deliberately, which requires re-baselining recorded
  results. See the Decision Log, Section 4.14.
- **Data content is pinned to a snapshot.** The anchor fixes which weeks each window
  covers. It does not fix the values inside those weeks, which the weekly refresh revises.
  The ML inputs are therefore read from a dated snapshot rather than from the live
  processed files (Section 2.1 and the Decision Log, Section 4.21). The anchor and the
  snapshot are two separate pins and both are required for a result to reproduce.
- **Known limitation: overlapping training sets.** With roughly two years of total
  history, the four training sets overlap heavily. The windows therefore test robustness
  across target seasons; they are not four independent samples, and results should be read
  accordingly.
- **Known limitation: thin short segment in older windows.** Short-history SKUs are recent
  by definition, so few of them had enough history to be eligible (Section 4.15) at older
  cutoffs. On the pinned snapshot the short segment has 206 eligible SKUs in the Mar-May
  window and 194 in the Dec-Feb window, but only 14 in the October 2025 window. For this
  reason, the Oct-Dec window is excluded from short-segment decisions (Sections 1.5 and
  4.16); its short results are still printed for reference. Evaluating the short segment across seasons is inherently
  limited by how little history recent SKUs have.

### 2.3 Internal validation (model fitting only)

Separately from the evaluation windows, LightGBM requires a validation slice inside each
training set to decide when to stop adding trees (early stopping). This slice is a random
15% of SKUs, including all of their rows across all seasons, held out from tree fitting.
Whole SKUs are held out rather than random rows because adjacent rows of the same SKU are
near-duplicates: their rolling-window features share 11 of 12 weeks. Splitting by row
would place near-identical rows on both sides of the boundary and make validation error
misleadingly low.

This design replaced an earlier approach that failed in practice. Using the last
10 calendar weeks of training as the validation slice made the slice seasonally
unrepresentative, and training halted after one or two trees whenever the slice fell on an
atypical season (see the Decision Log, Section 4.8). The SKU-based slice has a known blind spot
in the opposite direction: it cannot detect a model memorizing calendar-specific patterns,
because both sides of the split contain the same calendar weeks. That failure mode is
instead prevented structurally, by not exposing calendar-identifying features to the model
(Section 3).

The validation SKUs are stratified by segment (short versus long) and by within-segment
demand-volume tercile, so the slice represents the whole portfolio. Given the small size of
the smooth population, a purely random draw can under-sample high-volume SKUs, which
dominate the demand-weighted stopping signal. The selection is `src/ml/dataset.py:stratified_val_skus`. On the first
development window it reproduced the population's segment mix almost exactly (28.6% long
versus 28.9% in the population), where a random draw of the same size undershot at 23.8%.

### 2.4 Single source of truth for measurement

All accuracy claims in this project are produced by one code path: `src/ml/dataset.py`
(loading and splits), then the model under test, then `src/ml/evaluate.py` (scoring). The
scorer reports per-segment tables with SKU counts, enforces that predictions fall inside
the test window (a guard against accidental use of future data), and computes pooled WAPE
and bias exactly as defined in Sections 1.3 and 1.4.

Scoring applies two as-of correctness rules automatically (Section 4.15): only SKUs with
at least 13 weeks of history at the cutoff are scored, and each SKU's segment is its
history length as of the cutoff rather than today. Both are on by default in
`evaluate.score`, so an experiment cannot forget them.

Rules of practice:

- Raw per-segment results are reported before any summary or interpretation.
- A change is judged by the decision rule in Section 1.5.
- Legacy evaluation artifacts (`v1_comparison.csv`, `test_evaluation.csv`,
  `fc_forward_forecasts_test`, and figures quoted in `config.py`) are historical
  references from the prototype era. They are cited as context, but no new claim rests on
  them.
- Backlog: run the statistical prototype's models through this same harness so the
  smooth/long comparison in Section 1.6 uses the same evaluation code as every other
  result in this project.

## 3. Current Model Specification

**Version: restart baseline (step 0). No machine learning yet.**

The current "model" is the structural baseline that every future feature must beat. It
contains no learned parameters. For each SKU, the forecast for a target week is:

- **smooth/long:** the trailing 12-week average of factor-adjusted (deseasonalized)
  history, multiplied by the target week's seasonal factor. This is WA12 with the
  production seasonal round-trip.
- **smooth/short:** the trailing 12-week average of raw history, with no seasonal
  adjustment. This is plain WA12.

The segment split of the seasonal treatment is the outcome of measurement, not
assumption; see Sections 4.10 and 4.17. Segments are as of the forecast cutoff
(Section 4.15).

Baseline floor on the development windows (pooled WAPE), measured on the pinned
2026-07-20 snapshot:

| Window | smooth/short | smooth/long |
|---|---|---|
| Mar-May 2026 (dev) | 0.2097 | 0.1321 |
| Dec-Feb (post-holiday) | 0.1788 | 0.2764 |
| Oct-Dec (Q4) | 0.4861 (reference only) | 0.1209 |

Bias at the same points: short −7.1%, +1.7%, −48.6%; long −5.6%, +24.4%, −8.3%.

The first learned candidate, LightGBM with a lead-only feature (a global growth-drift
correction), was tested and rejected (Section 4.18); the structural baseline above
remains the champion. Feature development continues one hypothesis at a time
(Section 5.1), judged against these floor numbers by the Section 1.5 decision rule, with
SKU-level ramp features as the next candidate. This section will be rewritten when a
learned model first beats the baseline. The version-by-version chronology, including
what each version contained and how it performed against its predecessor and the
baseline, is maintained in Section 6.

## 4. Decision Log

This section records every meaningful design choice: what was decided, what the
alternatives were, why the decision went the way it did, and what evidence supports it.
Rejected ideas are recorded along with the reason for the rejection. This prevents the
team from re-testing ideas that have already failed, and it documents the conditions under
which a rejected idea could become viable again (for example, once more historical data is
available).

Entries are numbered in the order they were made. All entries below date from the
project's design phase in July 2026.

### 4.1 Evaluation metric: pooled WAPE rather than RMSE

All accuracy evaluation uses pooled WAPE as defined in Section 1.3, with bias as a secondary
check. The main alternative considered was RMSE (root mean squared error).

The business cost of a forecast error grows roughly in proportion to the number of units
the forecast is wrong by, because overstock and missed sales both scale with units. WAPE
prices errors the same way. RMSE squares errors, which makes one large miss count as much
as dozens of small ones; a model tuned to RMSE protects itself against rare large misses
at the expense of everyday accuracy, which is the wrong trade for inventory planning. WAPE
is also the convention used throughout the prototype's records, which keeps every number
in this project comparable with the past. A practical side benefit is that with an
absolute-error training objective, the model optimizes the same quantity it is graded on
(see Section 4.6).

### 4.2 The measurement harness was built and validated before any model

The first artifact built in this project was the shared loader, splitter, and scorer
(`src/ml/dataset.py`, `src/ml/evaluate.py`), validated by scoring the known baselines
through it before any machine-learning model existed. If the evaluation code itself is
flawed, every result produced with it is unreliable. The most common failure mode in
projects of this kind is a subtle mismatch between the new model's evaluation and the
incumbent's, which makes the comparison meaningless.

As validation evidence, the harness reproduced the known baseline numbers (WA12 pooled
WAPE of approximately 0.23 on smooth/short, matching the recorded 0.218 to 0.222 range
from the stored April to June backtest), and V1's known weaknesses (chronic underforecasting
of approximately −20% or worse) appeared exactly where expected.

### 4.3 A quarantined final test window with three development windows

The most recent 10-week window is reserved for a single final evaluation, and all
iteration is judged on three earlier windows spanning three seasons (Section 2.2). The
alternative, evaluating on the most recent window throughout, was used briefly for
harness validation before any tuning occurred, and was then retired. Any window that
repeatedly informs design decisions gradually becomes fitted, even though the model never
trains on it. Reserving the newest window preserves one unbiased evaluation for the final
decision.

### 4.4 Medium-history and full-history SKUs are reported as one segment

Evaluation reporting merges the medium-history and full-history SKUs into a single
smooth/long segment. The prototype treats the two groups nearly identically, both are
defined by having enough history for per-SKU statistics, and the medium group, fewer than
thirty SKUs, is too small to measure reliably: its WAPE swung by ±0.3 between windows for
the same unchanged baseline, which is pure sampling noise. The underlying labels are kept, so the merge
exists only at the reporting layer and can be undone at any time.

### 4.5 Target definition: ratio to the SKU's trailing 12-week average

The model predicts the ratio of the target week's sales to the SKU's trailing 12-week
average at forecast time, rather than raw units. Predictions are converted back to units
by multiplying by that same average. The alternative considered was predicting raw units
with a count-friendly objective such as Tweedie.

The ratio form has three properties that motivated the choice. First, it puts every SKU on
the same scale, so one global model can learn jointly from a SKU selling 3 units per week
and a SKU selling 200 units per week, which is the core requirement for a global model to
work. Second,
it bounds extrapolation: predictions can exceed a SKU's own history only up to ratio
values actually observed across the whole catalog, which prevents runaway trend
projections. Third, it provides a safe floor: a model that learns nothing predicts a ratio
of about 1.0, which reproduces the WA12 baseline exactly, so the worst case is a tie with
the incumbent rather than a regression. Raw-units modeling remains a reasonable future
ablation.

### 4.6 Training loss: absolute error on the ratio, weighted by SKU demand level

LightGBM trains with L1 (absolute-error) loss on the ratio, with each row weighted by the
SKU's trailing 12-week average. By algebra, the absolute unit error equals the SKU's level
multiplied by the absolute ratio error, so this weighted loss equals the pooled-WAPE
numerator: training optimizes the deployment metric rather than a proxy.

A second, deliberate property is that L1 targets the median outcome rather than the mean,
which makes the model resistant to chasing rare spike weeks. This conservatism is
something the project explicitly wants, given past problems with over-extrapolating
models. The known cost is a possible systematic underforecast on skewed data, which is why
bias is always reported alongside WAPE.

### 4.7 Multi-horizon strategy: direct forecasting with lead as a feature

All features are computed at the forecast anchor week, and the model receives the lead
(how many weeks ahead the prediction is, from 1 to 10) as an input, so one model serves
all horizons. Two alternatives were considered. Recursive forecasting, in which the model
predicts one week, feeds the prediction back in as input, and repeats, was rejected
because errors compound step by step; this is a known failure mode for tree-based
forecasters. Training ten separate models, one per lead, is the standard "direct" method
and was deferred rather than rejected: it multiplies the moving parts by ten, and the
lead-as-feature design lets long horizons share what short horizons learn.

### 4.8 Internal validation redesigned to hold out whole SKUs rather than recent weeks

The early-stopping validation slice is a random 15% of SKUs with all of their history,
rather than the last 10 calendar weeks of training. With the earlier time-based slice,
training stalled at one or two trees whenever the slice fell on a seasonally unusual
period (the post-holiday trough or the holiday season): trees that helped the average week
looked harmful on the unrepresentative slice, so early stopping fired immediately. With
the SKU-based slice, training runs to normal lengths (hundreds to roughly 1,600 trees).

A limitation was documented at adoption: an SKU-based slice contains the same calendar
weeks as the training rows, so it cannot detect a model memorizing calendar-specific
patterns. That risk is handled structurally instead (see Sections 4.9 and 4.10). The
validation SKUs are now selected by a stratified draw rather than a purely random one; see
Section 4.13.

### 4.9 Rejected: letting the model learn seasonality from calendar-derived features

No feature that identifies the calendar position of a week (month number, seasonal-factor
value, or any other date-derived code) is given to the model. Two experiments support the
rejection. With a month feature, the model overforecast the post-holiday trough by +123%
(pooled WAPE 1.20 versus the baseline's 0.33 on that window). Removing the month feature
but keeping the seasonal-factor values as features changed essentially nothing, with the
bias again near +123%. The reason is that the factor values function as month
identifiers: a tree uses a feature only to sort rows into groups, never as a multiplier,
so any date-derived value gives the model a way to isolate specific historical periods.

The root cause is data depth. The data contains only two observed cycles of each seasonal
event, so a group defined by "holiday-window weeks" has effectively two members, and the
model memorizes what those two specific periods did rather than learning a transferable
pattern. This rejection is contextual rather than permanent: the same technique won the M5
competition with five years of data, and it should be revisited once roughly four or more
seasonal cycles are on file.

### 4.10 Seasonality is imposed structurally (deseasonalize, model, reseasonalize)

The hand-set monthly multipliers, the same ones the prototype uses, are divided out of the
target before training and multiplied back into predictions afterward. The model itself
never sees seasonal information. Trees cannot apply a multiplier; they can only form
groups, and with two observed cycles per season, groups become memorized history (Section 4.9).
Performing the multiplication outside the model enforces the seasonal prior exactly and
reduces the model's job to learning residual dynamics. This is the architecture that won
the M4 competition for short-history series (Section 1.2), and it mirrors the prototype's own
deseasonalize-fit-reseasonalize pattern, so both tracks treat seasonality the same way.

Status: adopted as the restart baseline design; implementation is the next build step.

### 4.11 Hand-rolled pipeline rather than the mlforecast framework, for now

The training-matrix, feature, and evaluation code is written in this repository rather
than using Nixtla's `mlforecast` wrapper. The custom target (Section 4.5), custom weighted loss
(Section 4.6), and structural seasonality (Section 4.10) would all require custom extensions inside the
framework anyway, and during the design phase, full visibility into every step is worth
more than the framework's convenience. `mlforecast` uses the same underlying LightGBM
model, so the design transfers if the project later migrates for production hardening. A
planned check: once the design freezes, an `mlforecast` run with equivalent settings
should approximately reproduce our numbers. Two independent implementations agreeing is
strong evidence against pipeline bugs.

### 4.12 The noise floor for design decisions was calibrated by bootstrap

The adoption rule in Section 1.5 states that single-window differences under 0.02 are
inconclusive, that adoption requires a consistent sign across all three windows with a
mean improvement of at least 0.01, and that borderline calls are decided by bootstrap. An
SKU-level bootstrap with 1,000 resamples measured the sampling noise of a single-window
paired WAPE difference at roughly ±0.011 to ±0.014 standard deviation for models whose
predictions differ substantially. The originally proposed ±0.01 floor was below the noise level and would
have allowed chance results to drive decisions. The measurement utility is
`src/ml/evaluate.py:bootstrap_delta`.

### 4.13 Stratified selection of the internal-validation SKUs

The early-stopping validation SKUs (Section 2.3) are selected by a stratified draw rather
than a uniform random one. The strata are segment (short versus long) crossed with
within-segment demand-volume tercile, giving six cells. A proportional 15% is drawn from
each cell, with a floor of two SKUs per cell. This guarantees that high-volume SKUs, which
carry most of the demand-weighted early-stopping signal, are represented, and that the
slice's segment mix matches the portfolio. On the first development window, the stratified
draw reproduced the population's long-segment share to within 0.3 percentage points (28.6%
versus 28.9%), where a same-size random draw undershot at 23.8%. The selection function is
`src/ml/dataset.py:stratified_val_skus`, and the model accepts the resulting SKU set
through the `val_uids` argument of `LGBMForecaster.fit`.

### 4.14 Evaluation windows pinned to a fixed anchor date

The source data (`sales_clean.parquet`) is regenerated weekly by the forecast pipeline.
The evaluation windows were originally derived from the latest week in the data, which
meant every weekly refresh would shift all four windows forward by a week. That shift
would have three consequences: results recorded in this document would no longer reproduce,
because they were measured on windows that no longer exist; a feature tested one week could
not be compared to a feature tested the next, because they would be scored on different
data; and the three development windows would drift off the seasons they were chosen to
represent.

To prevent this, the windows are anchored to a fixed date, `ML_FINAL_TEST_CUTOFF` in
`config.py` (currently 2026-05-04). The split builder steps back from this anchor to
construct every window, so the windows stay fixed regardless of how much new data arrives.
Weeks after the final-test window are simply not used. When the project deliberately
incorporates newer data, the anchor is advanced as an explicit, recorded change, and the
affected results are re-measured. This behavior was verified by extending the data by one
simulated week and confirming all four windows stayed identical. The implementation is the
`anchor` argument of `src/ml/dataset.py:make_splits`, used by `dev_splits` and
`final_test_split`.

### 4.15 As-of evaluation: cutoff eligibility and as-of history length

The scorer originally used a single present-day snapshot (`sku_profiles.csv`) to decide
which SKUs to score and which segment each belonged to, and applied that snapshot to every
historical window. This distorted backtests in two ways, both measured directly.

First, SKUs were scored in windows where they had little or no history. At the October
2025 development cutoff, 201 of 264 scored SKUs had fewer than 13 weeks of history, and 12
had none at all because they had not launched yet. Scoring a not-yet-launched SKU as a
forecast of roughly zero against its later real demand inflates the error. The fix is an
eligibility rule that mirrors the statistical prototype's backtest: a SKU is scored in a
window only if it had at least `MIN_SIM_HISTORY_WEEKS` (13) weeks of history at the cutoff,
measured as (cutoff minus train_start). After the fix, the WA12 short-segment WAPE on the
February 2026 window moved from 0.243 to 0.208, in line with the prototype's recorded
0.218. The rule is `src/ml/dataset.py:eligible_skus`, applied by default inside
`evaluate.score` and `evaluate.bootstrap_delta`.

Second, the segment label was anachronistic. Because a SKU only accumulates history over
time, its present-day history length is always greater than or equal to its history length
at any past cutoff, so today's labels systematically push SKUs that were young at the time
into the longer segments. At the three development cutoffs, 24, 26, and 14 SKUs labeled
"long" today were actually short at the cutoff, and none moved the other way. The fix
computes history length as of the cutoff from (cutoff minus train_start), using the same
50 and 104-week thresholds as the profiler. The function is
`src/ml/dataset.py:asof_history_length`. A consequence worth noting: with only two years of
data, no SKU reaches 104 weeks of history at any evaluation cutoff, so the "full" segment
does not appear in any backtest window. It exists only in the present-day snapshot, which
further supports merging medium and full for reporting (Section 4.4).

Remaining limitation: the bucket label (smooth versus intermittent) is still taken from the
snapshot, not recomputed as of the cutoff, because that requires the full profiling
statistics over the as-of history window. The statistical prototype has the same
limitation. Recomputing the bucket as of each cutoff is a possible future improvement.

### 4.16 Bucket drift measured and deferred; Oct-Dec window excluded for the short segment

Two follow-ups to the as-of work in Section 4.15 were examined and decided together.

First, the size of the remaining bucket limitation was measured rather than assumed. The
profiler's classification rules were recomputed using only the data available at each
development cutoff, and the resulting as-of smooth set was compared with the present-day
smooth set actually being scored. The direction that corrupts scores, SKUs scored in a
window despite not being classifiable as smooth at the time, affected 1, 6, and 6 SKUs
across the three windows (0%, 2%, and 9% of the scored population). The larger direction,
49 to 52 SKUs that were smooth at the time but are excluded today because they have since
gone intermittent, narrows coverage but does not corrupt comparisons, because every model
is scored on the identical population. There is also a substantive argument for scoring
the present-day smooth set: the adoption decision concerns the SKUs the company will
forecast going forward. On this evidence, recomputing the bucket as of each cutoff was
deferred. (The measurement approximated the profiler by applying its thresholds to as-of
data without re-detecting ramp-up start dates; that approximation is adequate for sizing
the effect.)

Second, differing SKU counts across windows were assessed for their effect on
comparability. Model-versus-model comparisons are made within a window on the identical
SKU set, so counts do not affect who wins a window, and absolute WAPE levels were never
comparable across windows in any case because seasons and populations differ. The one real
problem is the October 2025 window's short segment, which has only 14 eligible SKUs. A
result from 14 SKUs is noise (the medium subgroup was already retired for this reason), and
under the consistency rule it could have vetoed changes supported by the 206-SKU and
194-SKU windows. The decision: the Oct-Dec window is excluded from short-segment decisions
entirely. Short-segment adoption evidence comes from the Mar-May window and the Dec-Feb
window plus the bootstrap; the Oct-Dec window's short results are still printed for
reference. The Oct-Dec window remains fully in force for the long segment, where all 52 of
its SKUs are eligible.

### 4.17 Seasonality is applied per segment: full for long SKUs, none for short SKUs

The structural deseasonalization adopted in Section 4.10 was tested as the restart
baseline (`scripts/ml_03_baseline_deseas.py`): a 12-week average computed on
factor-adjusted history, forecast as level times the target week's factor, compared with
the same average on raw history. The result split cleanly by segment.

For long-history SKUs, deseasonalization won in all three development windows, with
bootstrap confirmation in each: WAPE improved by 0.007 (spring), 0.312 (the post-holiday
window, where raw WA12 carries the Q4 peak into January and over-forecasts by +58%), and
0.077 (the Q4 window). It is adopted for the long segment.

For short-history SKUs, deseasonalization failed the decision rule: a trivial improvement
in spring (0.2097 to 0.200) against a large loss in the post-holiday window (0.1788 to
0.287, bias moving from +1.7% to −22.5%). The mechanism: the hand-set factors encode
mature seasonal behavior, and young SKUs express a muted version of it because their
growth partially offsets the January trough. An age-damping compromise was then tested
(`scripts/ml_04_alpha_search.py`): short SKUs received factor^alpha, with alpha searched
from 0 to 1 on the two short-decision windows. The result was monotone, with no interior
optimum: every increase in alpha traded a negligible spring gain for a larger
post-holiday loss, and the search selected alpha = 0, meaning no seasonal adjustment for
short SKUs at all. That is the adopted design.

Caveat recorded at adoption: the Q4 reference window (14 short SKUs, excluded from
decisions) suggested deseasonalization does help short SKUs during the holiday season
itself. With no seasonal adjustment, short SKUs receive no holiday uplift. Both variants
underforecast that window severely regardless (bias −42% versus −48%), because young
SKUs' growth into Q4 dominates. This is an open question to revisit before Q4 2026 (see
Section 5.5).

### 4.18 Rejected: LightGBM v0 with a global growth-drift correction (lead-only feature)

The first learned model of the restart used a single feature, the forecast lead, so the
only thing it could learn was the average ratio at each horizon. It learned a real
pattern: ratios rise with lead (roughly 1.05 at one week ahead to 1.22 at ten weeks ahead
in the December-cutoff model), which is the portfolio's average growth expressed as a
number, and the direct cause of the chronic underforecasting that motivates this project.

Applied unconditionally, however, the correction failed the decision rule. It won the
spring window (short segment: 0.1957 versus the baseline's 0.2097, bootstrap-significant)
and the Q4 reference window, both periods where demand kept growing. It lost the
post-holiday window in both segments (long: 0.4163 versus 0.2764; short: 0.2160 versus
0.1788, both significant), because demand contracts there and the model pushed its growth
correction anyway, on top of the already over-forecasting January reseasonalization. By
design the model cannot see the calendar (Section 4.9), so it cannot learn an exception
for the trough.

The rejection points at the next hypothesis rather than away from the approach: the
growth correction must be conditioned on each SKU's own recent trajectory (the ramp
features of Section 5.2) instead of applied to every SKU equally. A model with ramp
features should apply growth to SKUs that are actually growing and hold the ratio near
1.0 for the rest, which addresses the trough failure while keeping the spring and Q4
gains. The post-holiday window is the sharp test of that hypothesis. Experiment:
`scripts/ml_05_lgbm_v0.py`.

### 4.19 Rejected: v1 ramp features on segment-native series; v2 direction set

v1 added the ramp feature block (trailing 4-week over 12-week ratio and two recent-week
ratios) computed on the same series as each segment's level: deseasonalized for long
SKUs, raw for short SKUs. The learned behavior was sensible and readable (predicted
ratios rising monotonically with ramp state, mild mean-reversion for dipping SKUs, ramp
as the top feature in every window).

The outcome split exactly along data cleanliness. For long SKUs, whose trajectories are
measured net of seasonality, v1 met the pre-registered criterion: it repaired v0's
post-holiday failure (0.3208 versus v0's 0.4163, significant) and improved the other
windows directionally, though no window individually beat the baseline significantly and
the three-window mean fell short of the adoption bar. For short SKUs, whose trajectories
are raw, the December cutoff presented holiday-inflated ramps; the model applied its
correctly learned growth response to a seasonal artifact and over-forecast January by
+54% (WAPE 0.5555 versus the baseline's 0.1788). The ramp cohort improved substantially in
spring and Q4 and was destroyed in the post-holiday window by the same mechanism.

Verdict: rejected under the decision rule (inconsistent signs for short; insufficient
margin for long). The failure isolates a specific input problem rather than a hypothesis
problem: trajectory features must be measured net of seasonality for every SKU, even
though forecast scaling for short SKUs remains unadjusted (Section 4.17). That is v2:
factor-adjust the feature inputs for all SKUs, leave levels, targets, and output scaling
unchanged. Experiment: `scripts/ml_06_lgbm_v1.py`.

### 4.20 v2 rejected (target contamination); v3 validated the hypothesis but is blocked
### by a long-segment regression

v2 implemented the Section 4.19 direction: trajectory features computed on the
factor-adjusted series for all SKUs, everything else unchanged. It reduced the v1
short-segment Dec-Feb failure by about a third, measured as excess over the baseline
(0.5555 to 0.4230 against a baseline of 0.1788, so 0.3767 of excess down to 0.2442), but
remained far worse than the baseline, with a +40% residual over-forecast. The residual ruled out
the "factors over-clean young SKUs" explanation, which predicts under-forecasting, and
exposed the deeper issue: for short SKUs the training target itself was still seasonal.
The model, unable to see the calendar, attributed seasonal target movements to
co-occurring feature states, inflating its learned growth response. A secondary finding:
Dec-Feb long predictions degraded even though long SKUs' own inputs were identical,
demonstrating that changing one segment's training rows changes the shared trees that
serve the other segment.

v3 followed: the entire ML path made seasonally consistent for every SKU (levels,
targets, output scaling), while the structural baseline kept its Section 4.17 form. This
conditionally reopened Section 4.17 at model level, on the argument that the baseline
experiment rejected full deseasonalization without its necessary companion: a growth
model to offset the January over-cut for young, growing SKUs. The argument held. The
Dec-Feb short catastrophe disappeared (0.1943 versus baseline 0.1788, a statistical tie),
the Mar-May short win persisted (0.1863 versus 0.2097, significant), Oct-Dec short
(reference) improved dramatically (0.1826 versus 0.4861), and bias reached the best
calibration of any version (within ±6% everywhere except the known Dec-Feb long
segment). The predicted under-forecast signature for muted young-SKU seasonality did not
appear, so short-specific damped factors remain unneeded for now.

v3 is nevertheless not adopted: Dec-Feb long regressed significantly against the
baseline (0.3145 versus 0.2764), violating the third pre-registered criterion. Across
v1/v2/v3 the Dec-Feb long segment has oscillated (0.3208, 0.3348, 0.3145) while long SKUs'
own features never changed, implicating the shared global trees rather than any feature.
Candidate directions, undecided at time of writing: a segment indicator feature, separate
models per segment, per-segment sample weighting, or a volatility feature that damps
corrections on erratic SKUs. Experiments: `scripts/ml_07_lgbm_v2.py`,
`scripts/ml_08_lgbm_v3.py`.

### 4.21 ML inputs pinned to a dated data snapshot

Section 4.14 pinned the evaluation windows to a fixed anchor date so that the weekly
refresh could not shift which weeks a window covers. That fixed the window boundaries but
not the data inside them. The cron rewrites `data/processed/sales_clean.parquet` and
`sku_profiles.csv` in place every Monday, revising recent actuals as late orders register
and regenerating the segment labels. Two consequences were observed directly rather than
predicted. The v3 evaluation had to carry a note that its baseline figures differed from
earlier entries in the third decimal because of the 2026-07-20 refresh, and the same
refresh moved the smooth SKU count from 432 to 447 and the Dec-Feb window's eligible short
population from 195 to 194. Under the Section 1.5 decision rule, where a mean
improvement of 0.01 decides adoption and single-window differences under 0.02 are
inconclusive, third-decimal drift in the comparison baseline is not acceptable: a model
version evaluated one week and a model version evaluated the next were not being measured
against the same numbers.

The ML harness now reads its two inputs from `data/snapshots/<date>/`, selected by
`ML_DATA_SNAPSHOT` in `config.py` and currently set to 2026-07-20. The snapshot is a
physical copy, so the cron and the ML track share no file and cannot desync. Snapshot
files are written read-only, so an accidental overwrite fails rather than silently
corrupting the pinned data, and each snapshot carries a `manifest.json` recording file
checksums, row and SKU counts, the week range, and the segment mix, which makes it
verifiable that the pinned data has not changed.

The alternative considered was pinning by convention, meaning keeping a copy for reference
and remembering to restore it before an ML run. It was rejected because it fails silently:
forgetting the step produces plausible numbers rather than an error. The scope was
deliberately limited to the ML development track. `load_weekly` is called only by
`scripts/ml_*.py`, so the production pipeline, the statistical prototype's backtest, and
the FastAPI app were left reading `data/processed/` and continue to follow the weekly
refresh, which is the intended behavior for production.

Verified at adoption: the three development windows resolve to the same cutoffs as before
(2026-02-23, 2025-12-15, 2025-10-06), and the structural baseline reproduces all six
recorded v-base figures from the Section 6 v3 table to the third decimal (short 0.2097,
0.1788, 0.4861; long 0.1321, 0.2764, 0.1209). Refresh immunity was confirmed by pointing
`DATA_PROCESSED` at perturbed data and observing that the pinned loader returned unchanged
values while the unpinned loader moved. Implementation: `src/ml/dataset.py:data_dir`, the
`snapshot` argument of `load_weekly`, and `scripts/ml_snapshot_data.py` for creating and
verifying snapshots.

A consequence to plan for: the pinned snapshot ages. Sales after the week ending
2026-07-20 are invisible to the ML track until the snapshot is advanced, which is correct
during a comparison campaign but means the snapshot must be advanced deliberately, with
affected results re-measured, before any claim about current demand is made.

### 4.22 Model versions preserved as tagged commits; the default-drift incident

Every model version is now a commit tagged `model/v-base` through `model/v3`, carrying
that version's per-segment results in the commit message. Before this, none of the ML
track was under version control at all: the harness, the model code, the experiment
scripts and this document were all untracked, so the only copy of every result in the
version log was the working tree on one machine.

The decision to preserve versions individually rather than as one checkpoint was made
because of what the exercise uncovered. Verifying that each version could still be
reproduced showed that v0 could not. `RatioLGBM` took its feature list from a default
argument, that default moved from `FEATURES_V0` to `FEATURES_V1` when the ramp block was
added in v1, and `scripts/ml_05_lgbm_v0.py` had never been updated to name its own feature
set. The script had silently become a v1 run, and by the time it was tested it failed
outright, because the lead-only probe at the end of the script passes one feature to a
model trained on four. A rejected version that cannot be re-run is a rejection that cannot
be revisited, which matters here because Section 4.18 explicitly nominates v0's growth
correction for reuse once it can be conditioned properly. The remedy is structural:
`features` is now a required argument with no default, so no experiment can inherit a
later version's configuration, and `ml_08` states both seasonal flags explicitly for the
same reason.

Two limits are recorded honestly. First, the commits were created after the fact, so their
dates are the date of the reconstruction rather than of the original work; the intermediate
states of `model.py` were rebuilt by removing the flags each later version added, and each
reconstructed stage was verified by running that version's script and confirming it
reproduces the figures in the Section 6 tables. Second, the recorded figures for v0, v1 and
v2 were measured before the 2026-07-20 refresh, on data that was never snapshotted and
therefore cannot be recovered. Re-running those versions on the pinned snapshot reproduces
their qualitative findings and their adoption decisions, but not their exact numbers; the
largest divergence is the v1 Dec-Feb long segment. Only v3 was recorded on data that still
exists. This is the concrete cost of having pinned the data one refresh later than the work
began, and it is the reason the snapshot exists.

Dependency versions are pinned for the same reason. `requirements.txt` previously carried
no version constraints and `lightgbm` was absent from the project virtual environment
entirely, which leaves the solver free to move underneath results that are compared at the
third decimal.

### 4.23 Rejected: v4 segment indicator. The model trades short SKUs for long ones

v4 added one feature to v3, a binary `is_long` taken as of the cutoff. The hypothesis was
that the shared global trees could not separate the two populations, so splits driven by
short SKUs also applied to long ones, and that an explicit indicator would repair the
Dec-Feb long regression blocking v3 (Section 4.20).

The mechanism worked exactly as predicted for long SKUs. Dec-Feb long improved from 0.3145
to 0.2367, significantly better than v3 and no longer a regression against the baseline;
it is the best figure any version has posted in that cell. The indicator was genuinely
used, carrying 11.4% of total gain in the spring window, and the Dec-Feb tree count rose
from 37 to 250.

It failed on the other side. The Mar-May short win was lost (0.1863 to 0.2311) and Dec-Feb
short became a significant regression against the baseline (0.1943 to 0.2474), with
short-segment bias roughly tripling in both windows. One of three pre-registered criteria
was met, so v4 is rejected.

The reason appears to be capacity allocation rather than a fault in the feature. Long SKUs
are a minority of SKUs but carry 72% to 98% of the training weight across the three
windows, because they have more anchor rows and rows are weighted by demand level. Before
v4 the model could not act on that imbalance, because it had no way to tell the segments
apart. The indicator gave it one, and under a demand-weighted loss the profitable move is
to specialise on the segment carrying the weight. The model did exactly what it was asked
to do.

This reframes the problem. The Section 4.20 diagnosis, that segment interaction is the
obstacle, is supported: separating the segments demonstrably fixes the long side. What is
now clear is that an indicator alone lets the model choose which segment to serve, and the
loss function makes that choice for it. The remaining candidates from Section 4.20 address
this directly: per-segment sample weighting, which removes the imbalance the model is
exploiting, or separate models per segment, which removes the choice entirely. Separate
models are the cleaner test of the hypothesis, since they cannot trade one segment against
the other, at the cost of abandoning the cross-segment transfer that motivates a global
model (Section 1.2). Experiment: `scripts/ml_09_lgbm_v4.py`.

---

## 5. Feature Backlog & Open Questions

### 5.1 How features are added

Features are added one hypothesis at a time. Each candidate below states the hypothesis it
encodes, meaning the reason to believe it should help. A candidate is implemented, run on
the three development windows, and adopted or rejected by the Section 1.5 decision rule; either
way, the outcome is recorded in the Decision Log. Related columns that express a single
hypothesis (for example, two ways of measuring the same ramp) are tested together as a
block. Rejected features are recorded with their context and may be retested after the
model changes around them.

### 5.2 Feature candidates available now (sales history)

| Candidate | Hypothesis |
|---|---|
| Ramp ratio (4-week average ÷ 12-week average) | Recent acceleration persists into the near future; directly targets WA12's inability to see growth. |
| Recent-level ratios (last week ÷ 12-week average, and similar lags) | The most recent weeks carry the most information about next week, especially at short leads. |
| SKU age (weeks since first sale) | Young SKUs are systematically in ramp-up; their dynamics differ from mature SKUs. |
| Demand level (log of 12-week average) | Larger SKUs have steadier demand; corrections should shrink for small, noisy SKUs. |
| Volatility (rolling standard deviation ÷ mean) | For erratic SKUs the model should stay close to the baseline; for steady SKUs it can act on smaller signals. |
| Zero-recency (weeks since last zero week, recent zero count) | Recent zeros signal dormancy or supply gaps; expected demand should discount accordingly. |
| Product-type attributes (parsed from SKU codes) | Different product families (seat covers versus car covers) have different demand dynamics; lets patterns transfer within families. |
| Channel mix (share of FBA versus web sales per SKU) | Channel composition affects volatility and growth behavior; also preparation for channel-specific external data. |
| Empirically re-estimated seasonal multipliers, shrunk toward the hand-set values | The hand-set multipliers are priors, not measurements; pooling all SKUs may support better estimates for well-observed months. Applies inside the structural mechanism (Section 4.10), not as model features. |

### 5.3 Feature candidates pending external data

| Candidate | Hypothesis | Prerequisite |
|---|---|---|
| GA4 traffic signals (lagged product views, sessions, add-to-carts) | Site traffic leads sales by days to weeks, providing a leading indicator the sales history cannot contain. Only lagged values are usable, because future traffic is unknown at forecast time. | GA4-to-BigQuery export being set up; a SKU-to-GA4-item mapping must be built. |
| Stockout correction (per-week in-stock fraction) | Recorded sales understate true demand during stockouts. The first use is cleaning the training target, not adding a feature. Likely improves the statistical prototype too. | Stockout dates per SKU (promised, not yet available). |
| Marketplace analytics (Amazon, eBay, Walmart) | Equivalent leading indicators for the roughly 27% of demand GA4 cannot see, FBA especially. | Data access being investigated. |

### 5.4 Process backlog

1. Run the statistical prototype's models through the harness so the smooth/long
   comparison uses the same evaluation code as everything else (Section 2.4).
2. Reduce training-row redundancy if needed: anchor thinning (every second week) and/or
   per-tree row subsampling; also reassess `min_child_samples` in light of effective
   sample size.
3. Once the design freezes, reproduce results with `mlforecast` as an independent pipeline
   check (Section 4.11).
4. Keep `orders_raw.parquet` and the database exports fresh (auto-refresh added to
   `compare_v1.load_raw`; export scripts exist for the forecast tables).
5. Correct the internal project documentation where it describes `fc_forecast_history`
   with a schema the real table does not have.

Completed items are recorded in the Decision Log: stratified internal validation
(Section 4.13) and pinned evaluation windows (Section 4.14).

### 5.5 Open questions

1. Success-bar strictness: is equal WAPE with better bias and extensibility sufficient for
   the ML model to become the proposed method for a segment, or must it be strictly
   better? (Section 1.6 currently says less than or equal.)
2. Should per-lead accuracy (for example, weeks 1 to 2 versus weeks 8 to 10) become a
   standard secondary metric now or later? The lead-as-feature design makes it easy to
   produce.
3. Should intermittent SKUs ever enter training as extra examples (not as forecast
   targets)? The current stance is no, because their near-zero levels make the ratio
   target unstable, but this was decided by reasoning rather than measurement.
4. The winsorization level for the ratio target (currently the 99.5th percentile) should
   be revisited once the restart baseline exists.
5. Normal tree counts after the validation redesign appear to range from roughly 100 to
   1,600 depending on the window; whether that variation itself carries information (for
   example, about seasonal regime stability) is unexplored.
6. Short-SKU seasonality in Q4: the adopted design applies no seasonal adjustment to
   short SKUs (Section 4.17), which means no holiday uplift for them. The Q4 reference
   window weakly suggested an uplift would help, but it could not be measured reliably on
   14 SKUs. Revisit before Q4 2026, ideally once GA4 traffic signals are available as a
   leading indicator of the holiday ramp. The production system faces the same question.

---

## 6. Model Version Log

Each model version is recorded here when its evaluation completes. Full evidence and
reasoning live in the referenced Decision Log entries; this section is the compact
chronology. WAPE numbers are pooled per segment. Short-segment decisions use the Mar-May window and
the Dec-Feb window only; Oct-Dec short is reference (Section 4.16). "BEST" marks the version currently
serving as the reference to beat.

**All figures in this section were re-measured on the pinned 2026-07-20 snapshot**, so
every version is scored on identical data and the tables are directly comparable to each
other. Previously they were not: v0, v1 and v2 had been recorded before the 2026-07-20
refresh and v3 after it, which left the same model carrying different numbers in different
tables. A model appearing in more than one table below now shows the same value in each,
and the v-base row is identical everywhere. The original pre-refresh figures are preserved
in the git history at the tagged version commits (Section 4.22); they are not reproducible,
because the data they were measured on was never snapshotted. The verdicts are unchanged:
re-measurement moved individual numbers but did not reverse any adoption decision.

| Version | Change from previous | Status | Details |
|---|---|---|---|
| v-base | structural baseline, no learned parameters | **BEST** | Sections 3, 4.17 |
| v0 | + LightGBM, lead feature only | rejected | Section 4.18 |
| v1 | + ramp feature block | rejected | Section 4.19 |
| v2 | trajectory features computed on deseasonalized data for all SKUs | rejected | Section 4.20 |
| v3 | fully deseasonalized ML path for all SKUs (features, targets, output) | closest yet, not adopted | Section 4.20 |
| v4 | + `is_long` segment indicator | rejected | Section 4.23 |

### v-base (July 2026)

- **Contents:** trailing 12-week level per SKU; seasonal factor round-trip for long
  SKUs, none for short (Sections 4.10, 4.17). No learned parameters.
- **Status:** BEST. This is the floor every later version must beat.

| Pooled WAPE | Mar-May | Dec-Feb | Oct-Dec |
|---|---|---|---|
| smooth/short | 0.2097 | 0.1788 | (0.4861) |
| smooth/long | 0.1321 | 0.2764 | 0.1209 |

Bias: short −7.1%, +1.7%, −48.6%; long −5.6%, +24.4%, −8.3%.

### v0 (July 2026)

- **Added:** LightGBM with a single feature, the forecast lead. Learned a global
  growth-drift correction (ratio about 1.05 at lead 1 rising to 1.22 at lead 10).
- **Status:** rejected (Section 4.18). Inconsistent signs across windows.
- **Lesson:** growth correction must be conditioned on each SKU's own trajectory, not
  applied to every SKU equally.

| Pooled WAPE | v0 | v-base | v0 vs base | significant |
|---|---|---|---|---|
| short, Mar-May | **0.1957** | 0.2097 | −0.0140 | yes |
| short, Dec-Feb | 0.2160 | **0.1788** | +0.0372 | yes |
| long, Mar-May | **0.1275** | 0.1321 | −0.0046 | no |
| long, Dec-Feb | 0.4163 | **0.2764** | +0.1399 | yes |
| long, Oct-Dec | **0.1117** | 0.1209 | −0.0092 | no |
| short, Oct-Dec (ref) | **0.4165** | 0.4861 | −0.0696 | (reference) |

Bias: short −2.7%, +15.2%, −41.7%; long −1.0%, +40.0%, +4.4%.

### v1 (July 2026)

- **Added:** the ramp feature block (`ramp_4_12`, `y_last_r`, `lag_1_r`), computed on the
  same series as the level: deseasonalized for long SKUs, raw for short SKUs.
- **Pass criterion, stated before running:** hold the baseline in the post-holiday window
  (Dec-Feb) while keeping v0's Mar-May gain.
- **Status:** rejected (Section 4.19). Met the criterion for long SKUs, failed it
  catastrophically for short SKUs.
- **Lesson:** the ramp hypothesis works when the trajectory is measured net of
  seasonality (long segment: fixed v0's Dec-Feb failure) and fails when it is not (short
  segment: read the raw Q4 bump as growth and pushed it into January, +53% bias).

| Pooled WAPE | v1 | v0 | v-base | v1 vs base | significant |
|---|---|---|---|---|---|
| short, Mar-May | **0.1742** | 0.1957 | 0.2097 | −0.0355 | yes |
| short, Dec-Feb | 0.5555 | 0.2160 | **0.1788** | +0.3767 | yes |
| long, Mar-May | 0.1346 | **0.1275** | 0.1321 | +0.0025 | no |
| long, Dec-Feb | 0.3208 | 0.4163 | **0.2764** | +0.0444 | yes |
| long, Oct-Dec | **0.1030** | 0.1117 | 0.1209 | −0.0179 | no |
| short, Oct-Dec (ref) | **0.2320** | 0.4165 | 0.4861 | −0.2541 | (reference) |

Bias: short −2.0%, +54.2%, −14.0%; long −0.0%, +30.0%, +1.3%.

Ramp cohort: large improvements in the Mar-May window (0.249 versus baseline 0.324) and the Oct-Dec window (0.196
versus 0.353); destroyed in the Dec-Feb window (0.540 versus 0.161) by the same mechanism as above.

### v2 (July 2026)

- **Changed:** trajectory features computed on the factor-adjusted series for ALL SKUs;
  levels, targets, and forecast scale unchanged (still raw for short SKUs).
- **Pass criterion, stated before running:** remove the v1 short-segment Dec-Feb failure
  (hold the baseline there) while keeping v1's short-segment Mar-May gain.
- **Status:** rejected (Section 4.20). Cut the Dec-Feb short failure by a third but
  nowhere near holding the baseline.
- **Lesson:** cleaning the features was only half the contamination; for short SKUs the
  training TARGET was still seasonal, so the model learned inflated growth responses
  from seasonally lifted labels it could not explain.

| Pooled WAPE | v2 | v1 | v-base | v2 vs base | significant |
|---|---|---|---|---|---|
| short, Mar-May | 0.1854 | **0.1742** | 0.2097 | −0.0243 | yes |
| short, Dec-Feb | 0.4230 | 0.5555 | **0.1788** | +0.2442 | yes |
| long, Mar-May | 0.1377 | 0.1346 | **0.1321** | +0.0056 | no |
| long, Dec-Feb | 0.3348 | 0.3208 | **0.2764** | +0.0584 | yes |
| long, Oct-Dec | **0.1014** | 0.1030 | 0.1209 | −0.0195 | no |
| short, Oct-Dec (ref) | **0.2287** | 0.2320 | 0.4861 | −0.2574 | (reference) |

Bias: short +0.8%, +40.1%, −14.0%; long −0.6%, +31.5%, +0.8%.

### v3 (July 2026)

- **Changed:** the ENTIRE ML path made seasonally consistent for every SKU: levels,
  training targets, and output scaling all factor-adjusted (short SKUs included). The
  structural baseline itself is unchanged. Conditionally reopens Section 4.17 at model
  level: the hypothesis is that full deseasonalization becomes viable for short SKUs
  once the learned growth response can offset the January over-cut.
- **Pass criteria, stated before running:** hold the baseline on Dec-Feb short; keep the
  Mar-May short win; no long-segment regression.
- **Status:** two of three criteria met; not adopted (the Dec-Feb long regression blocks
  it), but the closest version yet and the core hypothesis validated. See Section 4.20.
- **Lesson:** the Dec-Feb long segment has been unstable under every learned version
  while its own features never changed; the shared global trees are the suspect, which
  points at segment interaction handling rather than new features as the next problem.

| Pooled WAPE | v3 | v2 | v-base | v3 vs base | significant |
|---|---|---|---|---|---|
| short, Mar-May | 0.1863 | **0.1854** | 0.2097 | −0.0234 | yes |
| short, Dec-Feb | 0.1943 | 0.4230 | **0.1788** | +0.0155 | no |
| long, Mar-May | 0.1345 | 0.1377 | **0.1321** | +0.0024 | no |
| long, Dec-Feb | 0.3145 | 0.3348 | **0.2764** | +0.0381 | yes |
| long, Oct-Dec | **0.1011** | 0.1014 | 0.1209 | −0.0198 | no |
| short, Oct-Dec (ref) | **0.1826** | 0.2287 | 0.4861 | −0.3035 | (reference) |

Bias: short +6.3%, +5.3%, −3.7%; long +0.5%, +29.2%, +1.5%.

Note: this entry's figures used to differ from the earlier entries in the third decimal,
because it was measured after the 2026-07-20 weekly refresh and they were measured before
it. That discrepancy is what prompted pinning the data (Section 4.21). Every table in this
section has since been re-measured on the pinned snapshot, so the baseline row is now
identical across all of them.

### v4 (July 2026)

- **Change from v3:** one feature added, `is_long`, a binary segment indicator (1 for
  medium/full history, 0 for short) taken as of the forecast cutoff via
  `asof_history_length`, never from the present-day snapshot. Everything else is v3
  unchanged, including `deseas_all=True`. One hypothesis at a time.
- **Hypothesis:** across v1, v2 and v3 the Dec-Feb long segment oscillated (0.3208,
  0.3348, 0.3145) while long SKUs' own features never changed, which points at the shared
  global trees rather than any feature (Section 4.20). Without a segment feature the trees
  cannot separate the two populations, so splits driven by the numerous short SKUs also
  apply to long ones. An explicit indicator gives the model the option to condition on
  segment, which should repair the Dec-Feb long regression that blocks v3.

**Pass criteria, stated before running:**

1. Dec-Feb long holds the baseline: no significant regression against v-base (0.2764).
   This is the single criterion v3 failed and the reason v4 exists.
2. The Mar-May short win survives: still significantly better than v-base (0.2097).
3. No new significant regression against v-base in any other decision cell relative to v3.

**Disconfirming evidence, also pre-registered:** if `is_long` carries near-zero gain in
feature importance, the hypothesis is wrong regardless of what the WAPE numbers do, and
any movement should be read as noise rather than as the indicator working.

**A caution recorded before seeing results.** Long SKUs are a minority of SKUs but a
majority of the training signal, because they have more history and therefore more anchor
rows, and because rows are weighted by demand level. Measured on the pinned snapshot:

| window | long share of SKUs | of training rows | of training weight |
|---|---|---|---|
| Mar-May | 20.2% | 53.6% | 72.5% |
| Dec-Feb | 20.2% | 67.8% | 83.8% |
| Oct-Dec | 22.0% | 91.7% | 98.1% |

If the Dec-Feb long problem is caused by short SKUs dominating the fit, that framing is
questionable on these numbers, since long SKUs already carry 83.8% of the weight in that
window. An indicator may therefore be the wrong instrument, and per-segment sample
weighting or separate models (the other candidates in Section 4.20) may be needed instead.
This is stated in advance so that a null result is read as informative rather than
disappointing.

**Status: rejected (Section 4.23). One of three criteria met.**

| Pooled WAPE | v4 | v3 | v-base | V1 | v4 vs base | significant | v4 vs v3 | significant |
|---|---|---|---|---|---|---|---|---|
| short, Mar-May | 0.2311 | **0.1863** | 0.2097 | 0.4198 | +0.0214 | no | +0.0448 | yes |
| short, Dec-Feb | 0.2474 | 0.1943 | **0.1788** | 0.3017 | +0.0686 | yes | +0.0531 | yes |
| long, Mar-May | 0.1376 | 0.1345 | **0.1321** | 0.3195 | +0.0055 | no | +0.0031 | no |
| long, Dec-Feb | **0.2367** | 0.3145 | 0.2764 | 0.4044 | −0.0397 | no | −0.0778 | yes |
| long, Oct-Dec | **0.1004** | 0.1011 | 0.1209 | 0.0847 | −0.0205 | no | −0.0007 | no |
| short, Oct-Dec (ref) | **0.1799** | 0.1826 | 0.4861 | 0.7893 | −0.3062 | (reference) | −0.0027 | (reference) |

Bias: short +17.6%, +16.9%, −3.2%; long −2.7%, +20.7%, +1.0%.
Criterion 1 met. Dec-Feb long, the cell that blocked v3, improved from 0.3145 to 0.2367,
a significant gain over v3 and now below the baseline's 0.2764 rather than significantly
above it. It is also the best Dec-Feb long figure of any version, and its bias improved
from +29.2% to +20.7% against a baseline of +24.4%.

Criteria 2 and 3 failed, in the same direction. The Mar-May short win did not survive
(0.1863 to 0.2311, a significant loss against v3 and no longer a win against the
baseline), and Dec-Feb short became a new significant regression against the baseline
(0.1943 to 0.2474). Short-segment bias roughly tripled in both windows, to +17.6% and
+16.9%.

The indicator was used, so the disconfirming check does not apply: `is_long` carried
11.37% of total gain in Mar-May and 3.54% in Dec-Feb, though only 0.45% in Q4. Tree counts
also moved sharply in Dec-Feb, from 37 to 250, meaning the model found substantially more
structure to fit once it could condition on segment.

Ramp cohort: improved in Mar-May (0.2509 against v3's 0.2704) and Q4 (0.1664 against
0.1691), and degraded badly in Dec-Feb (0.2353 against 0.1847, bias +16.6% against +5.6%),
which is the same short-segment damage seen in the headline numbers.
