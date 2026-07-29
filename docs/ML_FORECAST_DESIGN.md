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
  the result by hand-set monthly seasonal multipliers. Run through this project's harness
  on the pinned snapshot (`scripts/ml_02_v1_benchmark.py`), its pooled WAPE and bias are:

  | V1 | Mar-May | Dec-Feb | Oct-Dec |
  |---|---|---|---|
  | smooth/short | 0.4198 (−39.8%) | 0.3017 (+5.0%) | 0.7893 (+6.1%) |
  | smooth/long | 0.3195 (−30.5%) | 0.4044 (+38.7%) | 0.0847 (−1.0%) |

  The structural baseline of Section 3 beats V1 in five of these six cells, usually by a
  wide margin. The exception is long-history SKUs in the Q4 window, where V1 is the more
  accurate forecast (0.0847 against the baseline's 0.1209, with bias of −1.0% against
  −8.3%). That is the only place any incumbent currently beats the baseline, and it is
  worth understanding before changing how long SKUs are treated.

  A caution on characterising V1's error. Earlier revisions of this document described V1
  as underforecasting chronically by −33 to −39%, and attributed it to trailing velocity
  averages being unable to anticipate growth. The table above shows the direction is not
  constant: V1 underforecasts the spring window heavily, over-forecasts the post-holiday
  window by +38.7% on long SKUs, and is close to unbiased in Q4. The growth explanation
  accounts for the spring result and not the others, so V1 is better described as having
  large, season-dependent error than as a uniform underforecast.
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
  +14%: it corrects V1's underforecast on that segment but overshot in that window, which
  is useful context for the bias goals below.

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

The longer-term motivation is extensibility. A feature-based model can absorb a corrected
demand signal that per-SKU statistical models cannot easily use, in particular a sales
record cleaned for stockouts and preorders (Section 5.3). The near-term deliverable is a
baseline model built from the sales history as recorded; the correction work is layered on
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

Our dataset resembles M5 in problem type (retail unit demand for related SKUs) but
resembles M4 in history depth, since two years of data
contain only two observations of each seasonal event. We therefore follow the M4
architecture: per-SKU scale and seasonality are imposed structurally, the latter using the
existing hand-set monthly multipliers, and LightGBM is limited to learning the residual
dynamics that the data can support, such as growth ramps and lifecycle effects. An early
experiment in this project confirmed the necessity of this restriction. When LightGBM was
given calendar-derived features and allowed to learn seasonality itself, it reproduced the
behavior of the specific months it had seen rather than a general seasonal pattern, and
overforecast the post-holiday trough by +123% (see the Decision Log, Section 4.9).

LightGBM is retained as the machine-learning component for two reasons. First, additional
inputs such as a stockout- and preorder-corrected demand target, together with further
sales-derived features, enter the model as tabular inputs, and gradient-boosted trees are
the strongest established method for learning from tabular features. Second, the cost of
failure is low: the statistical
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
  systematic over-forecasting or under-forecasting. For context: V1's calibration is poor
  and season-dependent rather than uniformly negative, ranging from −39.8% on short SKUs
  in spring to +38.7% on long SKUs after the holidays (Section 1.1). The statistical
  prototype corrects the spring underforecast but with imperfect calibration: +0.2% on
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
smooth/long. Both tracks clear V1, the current production method, in five of the six
segment-window cells, usually by a wide margin; the exception is long-history SKUs in the
Q4 window, where V1 is the more accurate forecast (Section 1.1). V1 is reported as the
business-as-usual reference in all evaluations.

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

  **What the target contains.** The source table classifies each order line by `order_type`,
  with four values (`sales`, `preorder`, `ttm`, `ttm_preorder`), derived upstream from
  boolean `is_preorder` and `is_ttm` columns rather than from parsing tags. `src/ingest.py`
  selects `order_date`, `link_master_sku` and `link_qty` with no restriction on that column,
  so `y` is the sum of all four types, attributed to the week the order was placed rather
  than the week it shipped. `src/v1.py` treats preorder the same way by construction, its
  final window being 30 days of `order_type = 'preorder'` at weight 0.10 added to the sales
  windows. This is stated here because it is a property of the training target that is easy
  to assume otherwise; it also answers the prerequisite recorded against the preorder
  correction in Section 5.3 and Section 5.4 item 6, both of which were waiting on whether
  preorders were flagged at all and whether the series keyed on order or ship date.
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

Three cautions apply to any analysis that inspects the model's features or its predicted
ratios directly, rather than going through the scorer. Each produced a wrong reading before
being caught, during the v14 diagnosis:

1. The trajectory features are computed on the deseasonalized series (`y_feat`), so an
   analysis of `ramp_4_12` or the recent-level ratios must deseasonalize too. Bucketing on
   raw sales gives a different and misleading picture.
2. The model's output is `ratio x level x factor`, so recovering the ratio it actually
   predicted means dividing by both the level and the target week's seasonal factor, not by
   a raw trailing mean.
3. Anchors with unusual feature values are often concentrated in one window, so pooling
   windows can make a window effect look like a feature effect. Read such comparisons within
   a window, or state the confound.

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

Scope for new entries: design choices only, meaning a question that had alternatives and
was settled by evidence or argument. Bugs, corrections, and infrastructure changes do not
belong here even when they matter. A bug fix goes in `WORKLOG.md` and in a comment where
the code was wrong; a change to how the project is built or stored goes in
`CODEBASE_GUIDE.md`; a correction to a claim goes in the section making the claim. Entries
should stay near 200 words; if one needs more, the material probably belongs in the
section it concerns.

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
from the stored April to June backtest), and V1's large errors appeared where expected.
Note that the V1 benchmark carried a W-MON alignment fault at that time, corrected later;
the corrected figures are in Section 1.1.

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
number, and the direct cause of the underforecasting that motivates this project.

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

The window anchor (Section 4.14) fixes which weeks each evaluation window covers. It does
not fix the values inside them: the weekly cron rewrites `data/processed/` in place,
revising recent actuals as late orders register and regenerating the segment labels. Under
the Section 1.5 rule, where a mean improvement of 0.01 decides adoption, a comparison
baseline that drifts in the third decimal between runs is not usable. The v3 evaluation
had already had to carry a note to that effect.

The ML harness therefore reads its two inputs from `data/snapshots/<date>/`, selected by
`ML_DATA_SNAPSHOT` in `config.py`. The snapshot is a physical copy written read-only with
a checksum manifest, so the cron and the ML track share no file. The alternative, keeping
a copy and remembering to restore it before a run, was rejected because forgetting the
step produces plausible numbers rather than an error.

Scope is the development track only. `load_weekly` is called solely by `scripts/ml_*.py`,
so the production pipeline, the prototype's backtest, and the FastAPI app keep reading
`data/processed/` and keep following the weekly refresh, which is correct for production.

The snapshot ages by design: sales after its date are invisible to the ML track until it
is advanced deliberately, which requires re-baselining recorded results. Verified at
adoption: the three dev windows resolve to unchanged cutoffs, the structural baseline
reproduces all six recorded v-base figures, and pointing `DATA_PROCESSED` at perturbed
data left the pinned loader unmoved.

### 4.22 Moved: model-version preservation and the default-drift incident

Recorded here in July 2026, then relocated: the tagged-commit scheme is documented in
`CODEBASE_GUIDE.md` (file inventory) and the default-drift fix in `WORKLOG.md` and a
comment in `scripts/ml_05_lgbm_v0.py`, per the scope rule above. The number is retained
because the commit that introduced the entry cites it as 4.22.

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

### 4.24 Rejected: v7 per-segment weighting. The segment indicator itself is the problem

Section 4.23 diagnosed v4's failure as capacity allocation: long SKUs carry 72% to 98% of
the training weight, so under a demand-weighted loss the model specialises on them once
`is_long` lets it tell the segments apart. v7 tested the implied remedy, partially
equalising the two segments' weight (`BALANCE = 0.5`, fixed in advance) on top of v4.

The remedy failed, and in a way that disconfirms the diagnosis. Upweighting short rows made
short forecasts worse, not better: Mar-May short went 0.2311 under v4 to 0.2554 under v7,
against v3's 0.1863. If the mechanism were the loss declining to spend capacity on short
SKUs, paying it more to do so should have helped.

The reference arm settles it. `v7ref` applies the same weighting WITHOUT `is_long`, and it
is the best short model in two of three windows while being the worst long model. Sorting
every version by whether it carries the indicator separates the results cleanly: v3 and
v7ref protect short and cannot fix long; v4 and v7 fix long and damage short. Weighting
moves nothing across that line.

The corrected reading is that the indicator is not a neutral capability the loss then
misuses. Giving the trees an explicit segment split lets them fit long-specific structure,
and that structure actively misleads short SKUs, which are the numerous but individually
light population. Rebalancing does not undo it because the split is in the tree geometry,
not in the weights. Section 4.23's capacity-allocation account is superseded by this.

What this closes: of the four candidates named in Section 4.20 for the Dec-Feb long
problem, three are now spent. The segment indicator is v4, per-segment weighting is v7, and
per-segment seasonal factors are v5, all rejected. Every one of them repaired long at
short's expense, across four independent attempts. The remaining candidate, separate models
per segment, is the only one that structurally cannot make that trade, and it is now
supported by those four failures rather than by argument. Its known cost is abandoning the
cross-segment transfer that motivates a global model (Section 1.2), which the short segment
may depend on more than the long one. Experiment: `scripts/ml_14_lgbm_v7.py`.

### 4.25 Adopted: the holiday window ends mid-December (v9)

`ML_HOLIDAY_END` moves from (12, 31) to (12, 15). The uplift now covers the promotional
period the company actually runs, late November to mid-December, and the weeks after it
fall back to the ordinary December factor that `SEASONAL_HOLIDAY[12] = 1.00` has always
defined and the code never applied.

The basis is business knowledge, stated plainly because it is not a measurement. The
company runs promotions from late November to mid-December and expects to continue. That
also explains why the two observed Decembers disagree so sharply: December 2024 predates
the practice, so it is a different regime rather than a conflicting sample. `config.py`'s
own comment has described the intended behaviour since the beginning.

Three earlier findings had to be settled first. The membership test now uses the days a
week covers rather than its W-MON label, because a fixed date boundary tested on labels
drifts year to year and covered Nov 18 to Dec 08 in 2024 against Nov 17 to Dec 14 in 2025.
The ML factors were separated from the prototype's (`src/ml/seasonal.py`) so that changing
one cannot move the accuracy bar or V1. And v6 established that the change fails at
baseline level, which is why it was retested inside a model.

Evidence: all three pre-registered criteria met, Dec-Feb long improving from 0.3145 to
0.2528 with no significant regression anywhere, and v9 at or below the prototype in all six
cells. Full table in Section 6.

This closes the Dec-Feb long problem that v4, v5, v6, v7 and v8 each failed to fix. The
reason those failed is now clear: all five treated it as a segment-interaction problem,
and Section 4.24 and the v8 result together showed the segments genuinely want opposite
things from a shared model. It was never a segment problem. It was a mis-specified seasonal
window, and every segment-differentiated remedy was compensating for it in the wrong place.

Two limitations stand. The setting is a judgement about the promotional calendar and
becomes wrong if that calendar changes, which is why it is a config constant reviewed
annually rather than a fitted parameter. And the long segment is about 30 effective units
after accounting for 23 correlated variants, so its intervals overstate confidence.

### 4.26 Rejected: hyperparameter tuning. The model is not misconfigured

A random search of 81 configurations over eight LightGBM parameters, scored on the
validation slice with no test-window contact, produced a validation-loss spread of 1.24%
end to end and a best-versus-current gap of 0.138%. Only `min_child_samples` showed a real
effect, and the current 200 is a mild misconfiguration rather than a serious one: values
from 5 to 100 tie, 200 is marginally worse, 500 and 1000 are clearly worse. Carried to the
development windows (v10), the search winner produced a significant regression on Mar-May
short and ties everywhere else, failing the adoption rule.

Two things are settled by this. First, an earlier suspicion that `min_child_samples=200`
was badly wrong, raised when v8's short model trained a single tree on 2,854 rows, does not
hold for the shared model: at 34,000-plus training rows the setting barely matters. It
could still matter for genuinely small matrices, which keeps the v8 separate-models
question open rather than closed. Second, and more useful, the two Dec-Feb losses to the
structural baseline are invariant to hyperparameters: 81 configurations move them by under
0.001. That relocates the problem definitively. It is not capacity, regularisation, or
early stopping; it is the growth drift of Section 4.18, and it lives in the features or the
target, not the fit. The next candidates are therefore a turning-point feature and the
corrected demand target of Section 5.3, not further tuning.

### 4.27 Adopted: hybrid model and the elevation feature (v11)

The forecasting model becomes two models. Short-history SKUs are predicted by the shared
global model (v9 unchanged); long-history SKUs by a dedicated model trained on long SKUs
only, whose feature set drops the 4wk/12wk ramp and adds `elev_long`, the 4wk demand level
against the trailing annual level.

This resolves the Dec-Feb long deficit that has run through the entire project. The cause,
established across v4, v7, v8 and v10, was never a segment interaction to be tuned away and
never a regularisation problem: it was the growth drift of Section 4.18, the model
extrapolating a ramp into a window where mature demand contracts, with no feature able to
see the turn. The elevation feature is that missing signal. It is calendar-blind and
per-SKU, learned from every elevated-then-reverted episode across all long SKUs and weeks,
so unlike the v9 holiday window it does not rest on two Decembers alone. Splitting the
models is what lets long use it: a shared model leaks any long-targeted feature into short
through the shared trees (Section 4.24), while a long-only model has no such path, and short
keeps the cross-segment transfer it depends on (v8).

Evidence: all three pre-registered criteria met. Dec-Feb long improved from 0.2528 to
0.1380, significant against v-base (0.2167), with bias falling from +22.2% to +5.0%. Short
is identical to v9. No long regression elsewhere. Full table in Section 6.

Design cost and limitations, recorded honestly. Two models are more to maintain, which is
accepted. The long model rests on about 54 SKUs, 23 of them correlated variants, so its
intervals overstate confidence and any future marginal change to it should be distrusted on
this data. The elevation feature cannot separate a temporary spike from a genuine new
plateau, so a long SKU that truly breaks out will be under-forecast; this is acceptable for
a mature segment that rarely ramps, and the retained recent-level features still track
gradual growth. As with every recent gain, the standing status is BEST on the development
windows only; the quarantined final test (Section 2.2) has never been run and is what v11
must clear before it can be proposed to replace V1.

### 4.28 Rejected: SKU age feature. Monotonic trend features extrapolate badly

`sku_age` (weeks since first sale) was added to the shared model to let it modulate ramp
expectation by maturity. It failed hard: Mar-May short tripled to 0.6146 with +59% bias, a
significant regression, while the other windows were mild and insignificant.

The cause is not the hypothesis but the encoding. Age is monotonic and unbounded, and every
forecast is made at a SKU's maximum age, past the ages its own training anchors cover. A
tree cannot extrapolate a monotonic feature beyond its training support, so the learned
age-to-ratio slope produced runaway predictions at the latest cutoff, where ages are
highest. This is a general caution for any trend-like feature (age, cumulative volume, a raw
time index) in this tree model, recorded so it is not rediscovered.

The `sku_age` backlog candidate (Section 5.2) is marked accordingly: retest only with a
bounded encoding, and only if short-SKU error is shown to have a systematic age component,
which it does not currently appear to.

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
| ~~Ramp ratio (4-week average ÷ 12-week average)~~ | **Not a candidate: already in the model.** `ramp_4_12` has been in `FEATURES_V1` since v1, which is the feature set the shared model serves short SKUs with. This row was stale and is retained struck through so the error is not repeated. Its behaviour is analysed in the v14 version-log entry: the feature works at short lead and its influence decays with the horizon. |
| Recent-level ratios (last week ÷ 12-week average, and similar lags) | The most recent weeks carry the most information about next week, especially at short leads. |
| SKU age (weeks since first sale) | Young SKUs are systematically in ramp-up; their dynamics differ from mature SKUs. Raw age rejected (Section 4.28): a monotonic feature extrapolates badly at the prediction boundary. Retest only with a bounded encoding. |
| Demand level (log of 12-week average) | Larger SKUs have steadier demand; corrections should shrink for small, noisy SKUs. |
| Volatility (rolling standard deviation ÷ mean) | For erratic SKUs the model should stay close to the baseline; for steady SKUs it can act on smaller signals. |
| Zero-recency (weeks since last zero week, recent zero count) | Recent zeros signal dormancy or supply gaps; expected demand should discount accordingly. |
| Product-type attributes (parsed from SKU codes) | Different product families (seat covers versus car covers) have different demand dynamics; lets patterns transfer within families. |
| Channel mix (share of FBA versus web sales per SKU) | Channel composition affects volatility and growth behavior. |
| Empirically re-estimated seasonal multipliers, shrunk toward the hand-set values | The hand-set multipliers are priors, not measurements; pooling all SKUs may support better estimates for well-observed months. Applies inside the structural mechanism (Section 4.10), not as model features. |

### 5.3 Data-quality corrections pending source data

These are corrections to the recorded demand series itself, not new model features. Both fix
weeks where the recorded units do not reflect true demand, so both improve the training
target for every method, the statistical prototype included, and together they are the
primary planned extension to the model. The scope of this project is the sales record and
these corrections to it; no third-party feeds are in scope.

| Correction | Rationale | Prerequisite |
|---|---|---|
| Stockout correction (per-week in-stock fraction) | Recorded sales understate true demand during stockouts, so any week with a stockout trains the model on an artificially low number. The first use is cleaning the training target, not adding a feature. | Stockout dates per SKU (promised, not yet available). |
| Preorder correction (attribute demand to the fulfilment week) | A preorder books demand when the order is placed but ships weeks or months later. If the weekly series is keyed on order date, the demand lands in the wrong week: an artificial spike at order time and a gap at fulfilment. This is most damaging for newly launched SKUs, whose launch preorders can dominate their short history. | Partly resolved (Section 2.1). Preorders are flagged as `order_type` in the source table and the series is keyed on order date, so excluding or down-weighting preorder rows can be tested now. Attributing demand to the fulfilment week still needs a source recording the intended fulfilment date. |

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
6. Correct for preorders in the weekly series. The recording question is now answered
   (Section 2.1): preorders are flagged as `order_type` in
   `fc_velocity_link_snapshot_forecast`, `src/ingest.py` does not filter on it, and demand
   is keyed on order date. Preorder demand therefore does land in the week the order was
   placed, giving an artificial spike at order time and a gap at fulfilment. This corrupts
   every derived quantity at once, since levels, ramp, elevation, and the seasonal
   round-trip are all built from the weekly series, and it is most damaging for newly
   launched SKUs, whose launch preorders can dominate their short history. Because the flag
   is available per row, two treatments are testable immediately: exclude preorder rows from
   training, or down-weight them. Attributing them to the intended fulfilment week remains
   blocked on a source that records that date. This is a data-integrity correction, not a
   model feature.

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
   14 SKUs. Revisit before Q4 2026, once a third holiday season and the corrected demand
   target are available. The production system faces the same question.
7. Regime change in the holiday period. The company began running late-November to
   mid-December promotions after December 2024, so the training data spans two different
   seasonal regimes and the older one is not representative of the future. This affects
   more than the holiday window: any seasonal estimate pooled across both years mixes
   them, and the Oct-Dec evaluation window sits entirely inside the older regime. Consider
   whether pre-2025 holiday weeks should be down-weighted or excluded once a third
   December is available to confirm the newer pattern.

---

## 6. Model Version Log

Each model version is recorded here when its evaluation completes. Full evidence and
reasoning live in the referenced Decision Log entries; this section is the compact
chronology. WAPE numbers are pooled per segment. Short-segment decisions use the Mar-May window and
the Dec-Feb window only; Oct-Dec short is reference (Section 4.16). "BEST" marks the version currently
serving as the reference to beat.

**v-base is the floor, not the bar.** Every comparison in this section is against v-base,
which is a 12-week moving average with a seasonal round-trip. That makes it the right
internal control for feature work, because a model that learns nothing predicts a ratio of
1.0 and reproduces it exactly (Section 4.5), so any movement is attributable to the
feature under test. It is NOT the standard the project has to meet. The success bar is the
statistical prototype (Sections 1.2 and 1.6), which for smooth/long uses cross-validation
selected ARIMA, ETS and ensembles rather than a moving average, and is a materially harder
target. The prototype has still not been run through this harness, so no version below has
been compared against the thing that decides adoption. Beating v-base is necessary and not
sufficient. Closing that gap is backlog item 1 (Section 5.4).

**All figures in this section were re-measured on the pinned 2026-07-20 snapshot**, so
every version is scored on identical data and the tables are directly comparable to each
other. Previously they were not: v0, v1 and v2 had been recorded before the 2026-07-20
refresh and v3 after it, which left the same model carrying different numbers in different
tables. A model appearing in more than one table below now shows the same value in each,
and the v-base row is identical everywhere. The original pre-refresh figures are preserved
in the git history at the `model/v*` tagged commits; they are not reproducible,
because the data they were measured on was never snapshotted. The verdicts are unchanged:
re-measurement moved individual numbers but did not reverse any adoption decision.

| Version | Change from previous | Status | Details |
|---|---|---|---|
| v-base | structural baseline, no learned parameters | superseded by v9 | Sections 3, 4.17 |
| v0 | + LightGBM, lead feature only | rejected | Section 4.18 |
| v1 | + ramp feature block | rejected | Section 4.19 |
| v2 | trajectory features computed on deseasonalized data for all SKUs | rejected | Section 4.20 |
| v3 | fully deseasonalized ML path for all SKUs (features, targets, output) | closest yet, not adopted | Section 4.20 |
| v4 | + `is_long` segment indicator | rejected | Section 4.23 |
| v5 (stage 1) | per-segment December/holiday multiplier, tested at baseline level | rejected | Section 6 |
| v6 (stage 1) | holiday window ends mid-December, tested at baseline level | two of three criteria met; not adopted, superseded by a design flaw | Section 6 |
| v7 | + per-segment sample weighting on top of v4 | rejected | Section 4.24 |
| v8 | separate models per segment | rejected | Section 6 |
| v9 | holiday window ends mid-December, inside v3 | superseded by v11 | Section 4.25 |
| v10 | hyperparameters tuned on the internal validation slice | rejected; current settings retained | Section 6 |
| v11 | hybrid: shared short model + dedicated long model with an elevation feature | **BEST**, all criteria met; final test pending | Section 4.27 |
| v12 | + SKU age feature in the shared (short) model | rejected | Section 4.28 |
| v13 | acceleration feature: two independent tests (short model, and long model) | both rejected | Section 6 |

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

| Pooled WAPE | v0 | v-base | V1 | v0 vs base | significant |
|---|---|---|---|---|---|
| short, Mar-May | **0.1957** | 0.2097 | 0.4198 | −0.0140 | yes |
| short, Dec-Feb | 0.2160 | **0.1788** | 0.3017 | +0.0372 | yes |
| long, Mar-May | **0.1275** | 0.1321 | 0.3195 | −0.0046 | no |
| long, Dec-Feb | 0.4163 | **0.2764** | 0.4044 | +0.1399 | yes |
| long, Oct-Dec | **0.1117** | 0.1209 | 0.0847 | −0.0092 | no |
| short, Oct-Dec (ref) | **0.4165** | 0.4861 | 0.7893 | −0.0696 | (reference) |

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

| Pooled WAPE | v1 | v0 | v-base | V1 | v1 vs base | significant |
|---|---|---|---|---|---|---|
| short, Mar-May | **0.1742** | 0.1957 | 0.2097 | 0.4198 | −0.0355 | yes |
| short, Dec-Feb | 0.5555 | 0.2160 | **0.1788** | 0.3017 | +0.3767 | yes |
| long, Mar-May | 0.1346 | **0.1275** | 0.1321 | 0.3195 | +0.0025 | no |
| long, Dec-Feb | 0.3208 | 0.4163 | **0.2764** | 0.4044 | +0.0444 | yes |
| long, Oct-Dec | **0.1030** | 0.1117 | 0.1209 | 0.0847 | −0.0179 | no |
| short, Oct-Dec (ref) | **0.2320** | 0.4165 | 0.4861 | 0.7893 | −0.2541 | (reference) |

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

| Pooled WAPE | v2 | v1 | v-base | V1 | v2 vs base | significant |
|---|---|---|---|---|---|---|
| short, Mar-May | 0.1854 | **0.1742** | 0.2097 | 0.4198 | −0.0243 | yes |
| short, Dec-Feb | 0.4230 | 0.5555 | **0.1788** | 0.3017 | +0.2442 | yes |
| long, Mar-May | 0.1377 | 0.1346 | **0.1321** | 0.3195 | +0.0056 | no |
| long, Dec-Feb | 0.3348 | 0.3208 | **0.2764** | 0.4044 | +0.0584 | yes |
| long, Oct-Dec | **0.1014** | 0.1030 | 0.1209 | 0.0847 | −0.0195 | no |
| short, Oct-Dec (ref) | **0.2287** | 0.2320 | 0.4861 | 0.7893 | −0.2574 | (reference) |

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

| Pooled WAPE | v3 | v2 | v-base | V1 | v3 vs base | significant |
|---|---|---|---|---|---|---|
| short, Mar-May | 0.1863 | **0.1854** | 0.2097 | 0.4198 | −0.0234 | yes |
| short, Dec-Feb | 0.1943 | 0.4230 | **0.1788** | 0.3017 | +0.0155 | no |
| long, Mar-May | 0.1345 | 0.1377 | **0.1321** | 0.3195 | +0.0024 | no |
| long, Dec-Feb | 0.3145 | 0.3348 | **0.2764** | 0.4044 | +0.0381 | yes |
| long, Oct-Dec | **0.1011** | 0.1014 | 0.1209 | 0.0847 | −0.0198 | no |
| short, Oct-Dec (ref) | **0.1826** | 0.2287 | 0.4861 | 0.7893 | −0.3035 | (reference) |

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

### v5, stage 1 (July 2026) — rejected

- **Change:** the holiday-window multiplier (currently 1.26 for every smooth SKU) becomes
  per-segment. The `seasonal_fit` diagnostic measured December residuals of 0.733 for long
  and 1.494 for short after correction: one shared factor over-corrects mature SKUs and
  under-corrects young ones simultaneously. No other factor changes; one hypothesis.
- **Stage 1 tests at baseline level** (deseasonalized WA12 with split factors versus the
  same with the shared factor), so no model confounds the attribution. Only if it
  validates does the split factor enter the ML path as v5 proper.
- **Estimation, leak-free:** the empirical factor is re-estimated per evaluation window
  from that window's training data only, as 1.26 times the demand-weighted holiday-window
  residual, then shrunk halfway to the hand-set value: new = 1.26 + 0.5 x (empirical −
  1.26). The shrinkage weight 0.5 is fixed in advance, not searched. At the older cutoffs
  the estimate rests on one to two observed Decembers, which is precisely why w is not 1.
- **Pass criteria, stated before running:**
  1. Long segment: improves Dec-Feb and Oct-Dec versus the shared-factor baseline, with
     no significant Mar-May regression. (Long is where 1.26 over-corrects; Dec-Feb long
     carries a +45.9% December bias today.)
  2. Short segment: improves Dec-Feb versus the shared-factor deseasonalized WA12
     (0.2863), with no significant Mar-May regression. Note the honest expectation: only
     about 1.5 of Dec-Feb's ten test weeks are December, so the December fix alone is
     unlikely to close the full gap to raw WA12 (0.1788); narrowing it is the test.
  3. If either segment's sign is inconsistent across its decision windows, the change is
     rejected per Section 1.5 and the January/February misfit becomes the next suspect.

**Status: rejected. Failed every pre-registered criterion.**

| Pooled WAPE | shared 1.26 | split factor | significant |
|---|---|---|---|
| long, Mar-May | 0.1321 | **0.1299** | no |
| long, Dec-Feb | **0.2764** | 0.2820 | yes |
| long, Oct-Dec | **0.1209** | 0.1823 | yes |
| short, Mar-May | **0.2014** | 0.2098 | yes |
| short, Dec-Feb | **0.2863** | 0.2977 | yes |
| short, Oct-Dec (ref) | 0.4251 | 0.4251 | (estimator fell back to 1.26) |

Two mechanisms, both instructive. First, year-to-year variance: at the Oct-Dec cutoff the
leak-free window contains only December 2024, which for long SKUs implied a factor of
1.006; November-December 2025 then spiked anyway. One observed December is not an
estimate, and w=0.5 shrinkage cannot rescue n=1. Second, coupling through the level:
lowering the long December factor raises the deseasonalized level, which raises January
and February forecasts, 8.5 of Dec-Feb's ten test weeks, which were already
over-forecast. A single month cannot be corrected in isolation. The `seasonal_fit`
diagnostic's December finding stands as a description of the misfit, but the correction
requires joint re-estimation of all months with trend, realistically after Q4 2026 adds a
third December. The shared factors defended themselves; v3's short-segment qualification
(measured under them) is unaffected. Experiment: `scripts/ml_11_dec_factor_split.py`.

### The bar, measured (July 2026)

`scripts/ml_10_prototype_benchmark.py` runs the statistical prototype (CV model selection
for long SKUs, deseasonalized WA12 for short, reusing the prototype pipeline code, with
as-of history labels and per-window selection) through the harness. First measurement of
the Section 1.6 bar. Prototype pooled WAPE: short 0.2014 / 0.2863 / 0.4251, long 0.1411 /
0.2737 / 0.0911 across Mar-May / Dec-Feb / Oct-Dec.

Against it, with bootstrap: v3 is better or tied everywhere except one cell, long
Dec-Feb (+0.0408, significant). v3 therefore meets the Section 1.6 dev-window criteria
for smooth/short (lower or equal WAPE on all three windows, bias +6.3/+5.3/−3.7 versus
the prototype's −7.2/−22.4/−42.5), pending the one-shot final test and the open
strictness question (Section 5.5.1). Long does not qualify. Against V1, v3's only
non-win, long Oct-Dec, is a statistical tie (+0.0164, CI spans zero, 1.8 standard
errors, leaning V1).

### v6, stage 1 (July 2026) — not adopted

- **Change:** `ML_HOLIDAY_END` moves from (12, 31) to (12, 15), so the holiday uplift
  covers the weeks spanning Nov 17 to Dec 14 and stops there. Weeks covering Dec 15 to 28
  fall back to `SEASONAL_HOLIDAY[12] = 1.00`, a value the code currently defines and never
  applies. Note the W-MON boundary: membership is tested on the label, so (12, 15) keeps
  the week covering Dec 8-14 while (12, 14) would drop it. Nothing else changes.
- **Tested at baseline level first**, as with v5: deseasonalized WA12 under both windows,
  no model involved, so the attribution is unambiguous.

**Basis for the change, stated plainly.** This rests on business knowledge, not on fitting
the data. The company runs promotions from late November to mid-December, and expects to
continue doing so. December 2024 predates that practice, which is why the two observed
Decembers disagree so sharply: 2024 shows a collapse through December and a January
rebound, 2025 a Black Friday peak then a flat December. They are different regimes, not
conflicting samples of one. `config.py`'s own comment already describes the intended
behaviour ("Dec 1-14 captures the pre-Christmas peak; Dec 15-31 reverts to the normal
December factor"), which the code has never implemented.

The supporting measurement is weak by construction and should not be leaned on: the weeks
covering Dec 16-29 sat at or below the typical level in both years (0.49 and 0.39 in 2024,
1.00 and 0.95 in 2025), so the uplift is unsupported in both regimes even though the
regimes differ in every other respect.

**Pass criteria, stated before running:**

1. Dec-Feb long improves against the current window. This is the segment-window the change
   targets, where December currently carries +45.9% bias.
2. No significant regression in Mar-May or Oct-Dec, in either segment. The change touches
   only weeks inside the old window, so Mar-May should be untouched; Oct-Dec contains
   affected weeks and is the real risk.
3. Dec-Feb short does not regress significantly.

**Known limitation, recorded in advance.** The Oct-Dec evaluation window and the training
data both contain December 2024, the pre-promotional regime. If the regime change is real,
that data is misleading for the future regardless of what this experiment shows, and the
Oct-Dec result in particular should be read as evidence about 2024's behaviour rather than
about the change's merit. This is a broader problem than the holiday window and is
recorded as an open question (Section 5.5).

**Status: two of three criteria met, not adopted.**

| Pooled WAPE | end 12-31 | end 12-15 | significant |
|---|---|---|---|
| long, Mar-May | 0.1321 | **0.1302** | no |
| long, Dec-Feb | 0.2764 | **0.2167** | yes |
| long, Oct-Dec | 0.1209 | 0.1209 | no weeks affected |
| short, Mar-May | 0.2014 | **0.1927** | yes |
| short, Dec-Feb | **0.2863** | 0.3181 | yes |
| short, Oct-Dec | 0.4251 | 0.4251 | no weeks affected |

Criterion 1 met, and by the largest margin any change has produced in that cell: Dec-Feb
long fell from 0.2764 to 0.2167, with bias improving from +24.4% to +16.9%. Mar-May
improved in both segments. Criterion 3 failed: Dec-Feb short regressed significantly.

The reason is the same segment divergence that sank v4. In that window long over-forecasts
at +24.4% while short under-forecasts at −22.4%, so removing holiday uplift helps one and
hurts the other. Criterion 2 passed but tested nothing: the Oct-Dec test period ends at the
label 2025-12-15, which is inside the window under both settings, so zero weeks changed.

Not adopted for a second and independent reason found after the run. Window membership is
tested on the W-MON label rather than the days a week covers, so a fixed (month, day)
boundary drifts year to year: (11,20)-(12,15) covers Nov 18 to Dec 08 in 2024 but Nov 17
to Dec 14 in 2025. The 2024 window silently loses the second week of December. A window
pinned to real promotional dates cannot be defined this way, so the tested configuration is
not the one that would be deployed. Any future attempt should decide membership on covered
days. Experiment: `scripts/ml_13_holiday_window.py`.

### v7 (July 2026) — rejected

- **Change:** per-segment sample weighting. Training rows are currently weighted by the
  SKU's demand level alone, which makes the loss equal the pooled-WAPE numerator
  (Section 4.6) but also means long SKUs carry 72% to 98% of the total training weight
  across the three windows despite being about 20% of SKUs. This rescales weights so the
  two segments contribute more equally, partially: each segment's weight is multiplied by
  `(0.5 / its current share) ** BALANCE` with `BALANCE = 0.5` fixed in advance, not
  searched. `BALANCE = 0` is exactly today's behaviour, `1.0` would fully equalise.
- **Tested on top of v4, not v3.** v4 established that the model *can* separate segments
  when given `is_long`, and that it then specialises on the segment carrying the weight
  (Section 4.23). Weighting without the indicator would change which rows dominate the
  shared trees but would not let the model treat the segments differently, so it does not
  test the diagnosed mechanism. v3 + weighting is run alongside as a reference to keep the
  attribution clear.

**Pass criteria, stated before running:**

1. Dec-Feb long improves significantly against v3 (0.3145). This is the only cell where
   any version still loses to the prototype, and it is what the change targets.
2. The short segment retains v3's qualification: short pooled WAPE at or below the
   prototype's on all three windows (0.2014, 0.2863, 0.4251). This is the result v4
   destroyed and the main risk of reintroducing the indicator.
3. No significant regression against v3 in any other decision cell.

**Recorded in advance:** this deliberately breaks the Section 4.6 property that the
training loss equals the deployment metric. That property was defined against pooled WAPE
across all SKUs, whereas adoption is decided per segment (Section 1.6), so per-segment
weighting arguably aligns the loss with the actual decision rule. If v7 is adopted,
Section 4.6 needs amending to say so.

**Status: rejected (Section 4.24). One of three criteria met, and the hypothesis is
disconfirmed rather than merely unsupported.**

| Pooled WAPE | v7 | v4 | v3 | v7ref | prototype | v7 vs v3 | significant |
|---|---|---|---|---|---|---|---|
| short, Mar-May | 0.2554 | 0.2311 | **0.1863** | 0.1898 | 0.2014 | +0.0691 | yes |
| short, Dec-Feb | 0.2363 | 0.2474 | **0.1943** | 0.2029 | 0.2863 | +0.0421 | yes |
| long, Mar-May | 0.1424 | 0.1376 | **0.1345** | 0.1389 | 0.1411 | +0.0079 | no |
| long, Dec-Feb | 0.2507 | **0.2367** | 0.3145 | 0.3305 | 0.2737 | −0.0638 | yes |
| long, Oct-Dec | 0.1024 | **0.1004** | 0.1011 | 0.1025 | 0.0911 | +0.0014 | no |
| short, Oct-Dec (ref) | 0.2092 | 0.1799 | 0.1826 | **0.1761** | 0.4251 | +0.0266 | no |

Bias: short +20.7%, +16.2%, −10.0%; long −2.8%, +22.2%, +1.5%.

Criterion 1 met: Dec-Feb long improved significantly against v3. Criteria 2 and 3 failed:
short regressed significantly in both decision windows, and Mar-May short (0.2554) is now
worse than the prototype (0.2014), so v3's short-segment qualification is lost.

`v7ref` (weighting WITHOUT the segment indicator) is the informative arm and was included
for exactly this reason. It is the best short model in two of three windows (0.1898,
0.2029, 0.1761) and the worst long model (0.3305 in Dec-Feb). So the damage to short
tracks the indicator, not the weighting: every version carrying `is_long` (v4, v7) harms
short, every version without it (v3, v7ref) protects short and cannot repair long.
Rebalancing the weights did not change that, and v7 landed close to v4 on long while being
worse than v4 on Mar-May short.

Tree counts rose sharply under rebalancing, from 41 to 117 in Mar-May and 37 to 155 in
Dec-Feb, so the model fit considerably more structure and got worse, consistent with
overfitting the upweighted rows. Experiment: `scripts/ml_14_lgbm_v7.py`.

### v8 (July 2026) — rejected

- **Change:** two independent LightGBM models, one fitted on long SKUs and one on short,
  each on the v3 configuration (`FEATURES_V1`, `deseas_all=True`). `is_long` is omitted
  because it is constant within each model. Predictions are concatenated and scored
  exactly as before.
- **Hypothesis:** the last candidate from Section 4.20, and the only one that structurally
  cannot trade one segment against the other. Four attempts to fix Dec-Feb long inside a
  shared model (v4 indicator, v5 seasonal factors, v6 window, v7 weighting) each repaired
  long at short's expense, and Section 4.24 established that the damage tracks the segment
  indicator rather than the loss weighting. Separate models remove the shared tree
  structure entirely.
- **Known cost:** this abandons the cross-segment transfer that motivates a global model
  (Section 1.2). Short SKUs have the least history and may depend on it most, so short is
  where this is expected to hurt if it hurts anywhere.

**Pass criteria, stated before running:**

1. Dec-Feb long improves significantly against v3 (0.3145).
2. Short retains v3's qualification: pooled WAPE at or below the prototype's on all three
   windows (0.2014, 0.2863, 0.4251), with no significant regression against v3.
3. No significant regression against v3 in any other decision cell.

**Two limitations recorded in advance, so a null result is read correctly.** The long model
has only 6 to 7 validation SKUs for early stopping, because 15% of roughly 54 long SKUs is
inherently thin; its stopping point will be noisy. The short model has 2,854 training rows
in the Oct-Dec window against `min_child_samples=200`, roughly fourteen leaves' worth. If
short underperforms specifically where its matrix is small, that is evidence about
regularisation rather than about separate models, and should trigger a hyperparameter
retest rather than a rejection.

**Status: rejected. One of three criteria met.**

| Pooled WAPE | v8 | v3 | prototype | v8 vs v3 | significant |
|---|---|---|---|---|---|
| short, Mar-May | 0.2703 | **0.1863** | 0.2014 | +0.0840 | yes |
| short, Dec-Feb | 0.2423 | **0.1943** | 0.2863 | +0.0481 | yes |
| long, Mar-May | **0.1340** | 0.1345 | 0.1411 | −0.0004 | no |
| long, Dec-Feb | **0.2832** | 0.3145 | 0.2737 | −0.0312 | yes |
| long, Oct-Dec | 0.1013 | **0.1011** | 0.0911 | +0.0002 | no |
| short, Oct-Dec (ref) | 0.4183 | **0.1826** | 0.4251 | +0.2357 | (reference) |

Criterion 1 met; 2 and 3 failed, with short regressing significantly in all three windows
and Mar-May short falling below the prototype, losing v3's qualification.

The pre-registered inconclusive clause applies to Oct-Dec: the short model trained ONE tree
on 2,854 rows, so early stopping fired immediately and it is effectively predicting a
constant. Its 0.4183 is near the structural baseline's 0.4861 and says nothing about
separate models, only that `min_child_samples=200` is wrong at that matrix size.

The clause does not cover Mar-May, where the short model had 37,130 rows and 200 trees and
still returned 0.2703 against v3's 0.1863. That is direct evidence for the cost anticipated
in Section 1.2: cross-segment transfer does real work for short SKUs, and removing it hurts
them even with ample data.

Taken with v4, v5, v6 and v7, the pattern is now consistent rather than merely repeated:
shared structure helps short and hurts long, isolating the segments helps long and hurts
short, and v3 sits at the best available compromise. Note also that long is nearly
indifferent to all of it, moving 0.1345 to 0.1340 in Mar-May and 0.1011 to 0.1013 in
Oct-Dec. Only Dec-Feb long responds, which suggests that cell is not primarily a segment
problem. Experiment: `scripts/ml_15_lgbm_v8.py`.

### v9 (July 2026) — BEST

- **Change:** `ML_HOLIDAY_END` moves from (12, 31) to (12, 15), inside v3. Two supporting
  corrections land with it: holiday membership is now decided on the days a week covers
  rather than its label, and the ML factors are separate from the prototype's, so neither
  the prototype nor V1 moves.
- **Why this is not v6 again.** v6 tested the same window change at baseline level and was
  not adopted. v3 succeeded precisely where the baseline failed once before (Section 4.20:
  full deseasonalization became viable for short SKUs only when a learned growth response
  could offset it), so the window change deserves the same treatment. This is the first
  test of it inside a model.
- **Basis:** business knowledge, not fitting. Promotions run late November to mid-December
  and are expected to continue; December 2024 predates that practice. The measured support
  is deliberately weak and secondary: weeks covering Dec 16-29 sat at or below the typical
  level in both regimes.

**Pass criteria, stated before running:**

1. Dec-Feb long improves significantly against v3 (0.3145).
2. Short retains v3's qualification: at or below the prototype on all three windows
   (0.2014, 0.2863, 0.4251), with no significant regression against v3. At baseline level
   this is where v6 failed, so it is the criterion under test.
3. No significant regression against v3 elsewhere.

**Recorded in advance:** if this fails the same way every segment-differentiated attempt
has failed, the Dec-Feb long cell should be accepted as a limitation of two years of data
rather than pursued further, and effort moved to hyperparameters and the corrected demand
target (stockouts and preorders).

**Status: all three criteria met. First version to do so. Final test not yet run.**

| Pooled WAPE | v9 | v3 | v-base | prototype | v9 vs v3 | significant |
|---|---|---|---|---|---|---|
| short, Mar-May | **0.1961** | 0.1863 | 0.2097 | 0.2014 | +0.0098 | no |
| short, Dec-Feb | 0.2000 | 0.1943 | **0.1788** | 0.2863 | +0.0057 | no |
| long, Mar-May | 0.1394 | 0.1345 | **0.1302** | 0.1411 | +0.0049 | no |
| long, Dec-Feb | 0.2528 | 0.3145 | **0.2167** | 0.2737 | −0.0616 | yes |
| long, Oct-Dec | 0.1023 | **0.1011** | 0.1209 | 0.0911 | +0.0012 | no |
| short, Oct-Dec (ref) | **0.1783** | 0.1826 | 0.4861 | 0.4251 | −0.0043 | no |

Bias: short +9.4%, −1.5%, −4.3%; long +2.9%, +22.2%, +1.3%.

Criterion 1 met: Dec-Feb long fell from 0.3145 to 0.2528 and is below the prototype's
0.2737 for the first time. That cell has blocked every version since v1.

Criterion 2 met, and it is the one that matters. v6 tested the identical window change at
baseline level and failed here, with Dec-Feb short regressing significantly. Inside v3 it
does not: short stays under the prototype in all three windows with no significant
regression. This is the second time the pattern from Section 4.20 has held, where a
seasonal change that a fixed baseline cannot absorb becomes viable once a learned growth
response can offset it.

Criterion 3 met: the three small increases are all statistical ties.

**v9 is at or below the prototype in all six segment-window cells**, which is the Section
1.6 dev-window condition for both segments rather than short alone. Bias improves where it
mattered: Dec-Feb long from +29.2% to +22.2%, Dec-Feb short from +5.3% to −1.5%.

Two cautions carried forward. The change rests on business knowledge about the promotional
calendar, not on measurement; the supporting data is two contradictory Decembers, and if
the promotional pattern changes the setting becomes wrong. It is a config constant so that
it can be revised. And the long segment is roughly 30 effective units once the 23
correlated `CC-CN-03`/`CC-CP-03` variants are accounted for, so the Dec-Feb long interval
is narrower than the true uncertainty. Experiment: `scripts/ml_16_lgbm_v9.py`.

### v10 (July 2026) — rejected

`RatioLGBM.PARAMS` has never been tuned. `min_child_samples=200` and `num_leaves=31` were
set once and never revisited, and `colsample_bytree=1.0` still carries the comment "v0 has
one feature; sampling is meaningless", true with one feature and stale with four. Two
results make this pressing rather than cosmetic: v8's short model trained a single tree on
2,854 rows, so `min_child_samples=200` is demonstrably wrong at small matrix sizes, and v9
loses significantly to a 12-week moving average in both Dec-Feb cells, which is consistent
with a badly regularised model.

**Protocol, and why it differs from every other version here.** Tuning is a
multiple-comparison problem, unlike the single structural changes the Section 1.5 rule was
written for. Searching a grid on the three development windows would fit those windows and
quietly spend them. So:

1. Candidates are scored ONLY on the internal validation slice (Section 2.3), the 15% of
   SKUs already held out from tree fitting. The selection metric is the weighted L1 that
   early stopping already computes on that slice, which by Section 4.6 equals the
   pooled-WAPE numerator, averaged across the three training sets. No test-window data is
   touched at any point in the search.
2. The grid is fixed in advance at 12 configurations: `min_child_samples` in
   {20, 50, 100, 200} crossed with `num_leaves` in {15, 31, 63}. `learning_rate` stays at
   0.05 and `n_estimators` stays a cap that early stopping resolves. `colsample_bytree` is
   tested only on the winner, as two further fits.
3. The single winning configuration is then evaluated ONCE against v9 on the development
   windows. That one comparison is the only contact the search has with them.

**Pass criteria, stated before running:** adopt only if the winner improves the
three-window mean pooled WAPE by at least 0.01 with a consistent sign, per Section 1.5,
and produces no significant regression in any cell. A tuned model that merely ties is not
adopted, because the current settings are already recorded and a tie is not evidence.

**Recorded in advance:** the most interesting outcome is not a better mean. It is whether
the two Dec-Feb losses to v-base survive tuning. If they do, the growth-drift mechanism of
Section 4.18 is the cause rather than regularisation, and the fix belongs in the features
or the target rather than the hyperparameters.

**Status: rejected. Current hyperparameters retained.**

The search ran in two stages as pre-registered. Stage 1 scored 81 configurations on the
validation slice only: the current settings plus 80 random draws over eight parameters
(learning_rate, num_leaves, min_child_samples, colsample_bytree, subsample, reg_alpha,
reg_lambda, patience). The surface was nearly flat: the full spread of validation L1 was
1.24%, and the best config beat the current settings by 0.138%. The only parameter with a
clear effect was `min_child_samples` (Spearman +0.70 with loss); values from 5 to 100
formed a plateau and the current 200 sat just past it, with 500 and 1000 clearly worse.
`patience` correlated +0.00, which retired the theory that early stopping was the binding
constraint.

Stage 2 evaluated the single winner (learning_rate 0.010, num_leaves 127,
min_child_samples 10, colsample 0.96, subsample 0.72, reg_alpha 1.0, reg_lambda 0.01,
patience 30), refit at the full tree cap, once against v9:

| Pooled WAPE | v9 | v10 | v-base | v10 vs v9 | significant |
|---|---|---|---|---|---|
| short, Mar-May | **0.1961** | 0.2024 | 0.2097 | +0.0063 | yes |
| short, Dec-Feb | 0.2000 | 0.1986 | **0.1788** | −0.0014 | no |
| long, Mar-May | 0.1394 | 0.1427 | **0.1302** | +0.0033 | no |
| long, Dec-Feb | 0.2528 | 0.2521 | **0.2167** | −0.0008 | no |
| long, Oct-Dec | 0.1023 | 0.1009 | 0.1209 | −0.0014 | no |
| short, Oct-Dec (ref) | **0.1783** | 0.1818 | 0.4861 | +0.0036 | no |

Rejected on all three criteria: no 0.01 mean improvement, a significant regression in
Mar-May short, and five of six cells statistical ties.

The pre-registered question is answered decisively. Both Dec-Feb losses to v-base survive
tuning intact (long 0.2521 versus 0.2167, short 0.1986 versus 0.1788): the best
hyperparameters found across 81 configurations move them by under 0.001. The Dec-Feb gap
is therefore not a regularisation problem. It is the growth-drift mechanism of Section 4.18,
the model learning ratios that rise with lead, which is correct on average and wrong
exactly when demand contracts after the holidays. Closing it needs a feature that
anticipates the turn or the corrected demand target of Section 5.3, not tuning. The current
hyperparameters, `min_child_samples=200` included, are retained. Experiment:
`scripts/ml_18_tune_wide.py`, verified by `scripts/ml_19_tune_verify.py`.

### v11 (July 2026) — BEST

**Architecture.** Two models instead of one.
- SHORT SKUs are predicted by the shared v9 model (`FEATURES_V1`, trained on all smooth
  SKUs). Their predictions are identical to v9 by construction, so the short-segment
  qualification cannot break. Short SKUs benefit from cross-segment transfer (v8), so they
  keep the shared model.
- LONG SKUs are predicted by a model trained on long SKUs ONLY, with feature set
  `FEATURES_V11_LONG = [lead, y_last_r, lag_1_r, elev_long]`. It drops `ramp_4_12`, the
  4wk/12wk growth feature that drives the Section 4.18 over-forecast, and adds `elev_long`,
  the 4wk/52wk elevation against the annual norm. The recent-level ratios `y_last_r` and
  `lag_1_r` remain, so gradual growth is still tracked; only the sharp-ramp extrapolation
  is removed and replaced with mean-reversion pressure.

**Why this specific design.** It is the synthesis the whole v4-v10 arc points to. Section
4.24 established that long-targeted changes in a shared model damage short via shared trees;
v8 established that a long-only model forecasts long well while a short-only model loses the
transfer short needs; the v11 exploration established that elevation is a real long-SKU
signal but leaks into short inside a shared model. Splitting the models removes the shared
trees, so long gets elevation and short is untouched.

**Improvement over the exploration.** The long model's early-stopping validation SKUs are
re-stratified WITHIN the long segment (about 9 SKUs) rather than taken as the long slice of
the whole-portfolio draw (6-7 SKUs, and not representative of long volume tiers). This is a
cleaner early-stopping signal and makes the formal run distinct from the exploration.

**Pass criteria, stated before running:**
1. Dec-Feb long improves significantly against v-base (0.2167). This is the defining test:
   the cell that has blocked every version, and where a moving average still beats v9.
2. Short is identical to v9 in all three windows (delta exactly 0, by construction; verified).
3. No significant long regression against v9 in Mar-May or Oct-Dec.

**Recorded limitations.** The long model rests on about 54 SKUs, 23 of them correlated
`CC-CN-03`/`CC-CP-03` variants, so its effective sample is small and its intervals overstate
confidence. The elevation feature cannot distinguish a temporary spike from a genuine new
plateau, so a long SKU that truly breaks out to a higher level will be under-forecast; this
is acceptable because long SKUs are mature and rarely ramp, and `y_last_r`/`lag_1_r` still
carry gradual-growth signal. The Dec-Feb win, like the v9 window, ultimately rests on two
observed Decembers, though the elevation feature is trained on every elevated-then-reverted
episode across all long SKUs and weeks, not on December alone.

**Status: all three criteria met. New BEST. Final test not yet run.**

| Pooled WAPE | v11 | v9 | v-base | prototype | v11 vs v-base (long) |
|---|---|---|---|---|---|
| short, Mar-May | 0.1961 | 0.1961 | 0.2097 | 0.2014 | (short = v9) |
| short, Dec-Feb | 0.2000 | 0.2000 | 0.1788 | 0.2863 | (short = v9) |
| short, Oct-Dec (ref) | 0.1783 | 0.1783 | 0.4861 | 0.4251 | (short = v9) |
| long, Mar-May | 0.1355 | 0.1394 | 0.1302 | 0.1411 | +0.0053, tie |
| long, Dec-Feb | **0.1380** | 0.2528 | 0.2167 | 0.2737 | −0.0787, significant |
| long, Oct-Dec | 0.1000 | 0.1023 | 0.1209 | 0.0911 | −0.0209, tie |

Long bias: −7.2%, +5.0%, +1.7% across the three windows; the Dec-Feb figure
falls from v9's +22.2% to +5.0%.

The defining result: Dec-Feb long, which every version since v1 has lost and where a moving
average still beat v9, is now 0.1380, well below both v-base (0.2167) and the prototype
(0.2737). Short is identical to v9 by construction, so its qualification is intact, and the
long model does not regress in the other two windows. v11 beats the prototype in five of
six cells, losing only Oct-Dec long, the cell V1 also wins.

The within-long re-stratification improved the result over the exploration (Dec-Feb long
0.1380 versus the exploration's 0.1557 for the same feature set), confirming that the
long-only model's early stopping benefits from validation SKUs chosen to represent the long
volume tiers rather than sliced from a whole-portfolio draw. Experiment:
`scripts/ml_22_v11_hybrid.py`.

### v12 (July 2026) — rejected

- **Change:** add `sku_age` (weeks since first sale, already computed but never used as a
  feature) to the shared model, which serves short SKUs in the v11 hybrid. The dedicated
  long model is unchanged, so only short predictions can move. Feature set
  `FEATURES_SHORT_AGE = FEATURES_V1 + [sku_age]`.
- **Hypothesis:** short SKUs are young and mostly ramping; a young SKU ramps steeper than
  one approaching maturity. Age lets the model modulate its ramp expectation by maturity.
  The pre-check found age carries growth signal not captured by the existing ramp feature
  (Spearman +0.22 to +0.27 with the ratio target in Dec-Feb and Oct-Dec, and low
  correlation with `ramp_4_12`), though it is weak in Mar-May (+0.04).
- **Elevation not tested for short, and why:** short SKUs genuinely ramp, so being above
  their own baseline is often healthy growth rather than a spike to revert, and the
  existing ramp already captures short-SKU reversion better than a longer-baseline
  elevation would (−0.56 versus −0.52). Short SKUs also have too little history for a stable
  long baseline (Dec-Feb median about 16 weeks).

**Pass criteria, stated before running:**
1. Short improves against v11 (= v9) with a consistent sign and no significant regression
   in any window.
2. Long is unchanged (the long model is not touched; verified identical).

**Recorded expectation:** the signal is moderate, short is already good (beats the
prototype in all three windows), and its remaining error is more dispersion than systematic
bias, so a null or marginal result is the likely outcome and would itself be informative.

**Status: rejected. Criterion 1 failed catastrophically.**

| Pooled WAPE | v12 | v11 | v-base | v12 vs v11 (short) |
|---|---|---|---|---|
| short, Mar-May | 0.6146 | 0.1961 | 0.2097 | +0.4185, significant |
| short, Dec-Feb | 0.2146 | 0.2000 | 0.1788 | +0.0146, no |
| short, Oct-Dec (ref) | 0.2002 | 0.1783 | 0.4861 | +0.0219, no |
| long (all) | = v11 | | | +0.0000 |

Criterion 2 met (long identical). Criterion 1 failed, and not marginally: Mar-May short
tripled, with bias jumping to +59.4%.

The cause is a boundary-extrapolation pathology of a monotonic feature. `sku_age` grows
without bound, and every forecast is made at the SKU's oldest age, older than nearly all of
its own training anchors, which stop about ten weeks before the cutoff because they need a
realized target inside the training window. The model learned a positive age-to-ratio slope
from the SKU cross-section (the +0.22 to +0.27 pre-check correlation) and then applied it at
ages beyond where it had per-SKU signal, extrapolating to large over-forecasts. Mar-May is
the latest cutoff, so ages are highest and the extrapolation most extreme; the earlier
windows were milder and not significant.

The lesson generalises: raw monotonic trend features are unsafe in a tree model evaluated at
the edge of the age distribution. A bounded encoding (a young/mature indicator, or capped
age) would not extrapolate, but short SKUs are already strong and their residual error is
dispersion rather than a systematic age effect, so this is not pursued further. Experiment:
`scripts/ml_23_v12_age.py`.

### v13 (July 2026) — both rejected

Two independent single-feature tests of `accel` (4wk demand versus the 4wk before it, the
scale-free second-order trajectory term), each isolating one half of the v11 hybrid.

- **v13-short:** add `accel` to the shared model. Only short predictions can move; long is
  the unchanged dedicated model. Hypothesis: a decelerating short SKU is near the top of
  its ramp, which the first-order ramp feature cannot see.
- **v13-long:** add `accel` to the dedicated long model (`FEATURES_V11_LONG + accel`). Only
  long predictions can move. Hypothesis and main target: Oct-Dec is the ramp-UP into Q4,
  and acceleration is the mirror of what elevation did for the post-holiday decline, so its
  best shot is Oct-Dec long, the one cell where v11 still loses to V1 (0.1000 vs 0.0847).
  A shared-model version of this was marginal in the v11 exploration; the retry is in the
  dedicated long model, which is the changed condition.

**Pass criteria, stated before running (each test judged on its own segment):**
1. The feature improves its segment with a consistent sign across windows and no
   significant regression in any window.
2. The untouched segment is identical (verified).

**Recorded expectation:** acceleration was marginal in the earlier shared-model test, so a
null result is likely for both; the long/Oct-Dec cell is the one place a real gain is
plausible.

**Status: both tests rejected.**

| Pooled WAPE | v11 | v13-short | v13-long |
|---|---|---|---|
| short, Mar-May | 0.1961 | 0.2160 (+0.0200, sig) | 0.1961 (—) |
| short, Dec-Feb | 0.2000 | 0.2268 (+0.0269, sig) | 0.2000 (—) |
| short, Oct-Dec (ref) | 0.1783 | 0.1818 (+0.0035) | 0.1783 (—) |
| long, Mar-May | 0.1355 | 0.1355 (—) | 0.1370 (+0.0016) |
| long, Dec-Feb | 0.1380 | 0.1380 (—) | 0.1440 (+0.0059) |
| long, Oct-Dec | 0.1000 | 0.1000 (—) | 0.1035 (+0.0035) |

v13-short significantly regressed short in two windows. Acceleration is noisy for short
SKUs (feature variance 1.4 versus long's 0.65), redundant with the ramp they already carry,
and the added noise hurt a segment whose residual error is dispersion rather than a missing
signal. v13-long tied everywhere and, notably, its pre-registered target, Oct-Dec long, got
slightly worse rather than better (0.1000 to 0.1035): the ramp-up mirror hypothesis did not
hold, because the long model already handles the Q4 ramp through elevation and recent
levels. Experiment: `scripts/ml_24_v13_accel.py`.

This closes the productive sales-history features. Of everything tried, only elevation (in
the long model) improved on v9, and only for the decline it targets. Age, acceleration, and
per-segment weighting all failed; the model is at or near the information ceiling of the
sales series, and further error reduction needs the external leading indicators of Section
5.3 or the target-cleaning of preorders and stockouts.

### v14 (July 2026) — min_child_samples for the collapsing tail

`min_child_samples` lowered from 200 in the shared model, which serves short SKUs. Values
tested: 100, 50, 20, against the v11 baseline of 200. No feature changes and no change to
the dedicated long model, so only short predictions can move.

**The observation this comes from.** Short SKUs whose demand has already collapsed before
the cutoff are forecast to recover. Grouping short SKU-anchors by deseasonalized
`ramp_4_12` (4-week over 12-week mean) and comparing the model's predicted ratio with the
realised ratio, on the pinned snapshot:

| ramp at cutoff | n | lead 1 | lead 4 | lead 7 | lead 10 |
|---|---|---|---|---|---|
| collapsed < 0.4 | 12 | 0.88 / 0.49 | 1.02 / 0.52 | 1.30 / 0.36 | 1.36 / 0.52 |
| falling 0.4-0.7 | 12 | 0.90 / 0.93 | 1.03 / 0.69 | 1.31 / 0.62 | 1.33 / 0.73 |
| flat 0.7-1.1 | 125 | 1.04 / 1.17 | 1.06 / 1.14 | 1.10 / 0.95 | 1.11 / 0.88 |
| rising > 1.1 | 194 | 1.31 / 1.54 | 1.31 / 1.25 | 1.31 / 1.10 | 1.36 / 1.49 |

(predicted ratio / actual ratio.) At lead 1 the predictions are ordered correctly by ramp.
By lead 10 that ordering is gone, all four buckets sit between 1.11 and 1.36, while the
realised ratios remain ordered. The model reads a collapse at short lead and then discards
it, reverting to the average short-SKU response, which is a ramp because the population is
39% rising against 1% collapsing.

**The hypothesis.** This is a resolution limit, not a missing signal. At the Mar-May cutoff
the collapsed region holds 1,060 of 94,540 training rows, 1.1%. With
`min_child_samples=200` at most five leaves can describe it, against `num_leaves=31`
competing across the whole feature space. Lowering the minimum lets the trees carve out the
region; at 50 it would support about 21 leaves.

**Why Section 4.26 does not already settle this.** That search scored aggregate validation
loss, which a 1.1% subpopulation cannot move: even a complete fix there would be invisible
against a 1.24% end-to-end spread. It also recorded that values from 5 to 100 tie globally
with 200 marginally worse, so lowering the setting is not expected to cost anything in the
bulk. 4.26's conclusion stands for the aggregate and is silent on the tail.

**Pass criteria, stated before running:**
1. **Primary, Section 1.5 as usual.** Adoption requires improvement in short pooled WAPE
   with a consistent sign across the decision windows (Mar-May and Dec-Feb; Oct-Dec is
   excluded for short per Section 4.16) and a mean improvement of at least 0.01. Long must
   be identical, which is verified rather than assumed.
2. **Tail criterion, secondary and not sufficient alone.** Pooled WAPE over the collapsed
   and falling anchors (deseasonalized ramp < 0.7) must improve by at least 0.05, and the
   predicted-ratio ordering by ramp bucket must survive to lead 10 rather than collapsing
   into a single band.
3. **Guard.** Any significant regression in short on either decision window rejects the
   value outright, whatever the tail does. A tail fix bought by damaging the other 93% is
   not an improvement.
4. If several values pass, the mildest change from 200 wins, since 4.26 found 5 to 100
   indistinguishable in the bulk and there is no reason to move further than the evidence
   requires.

**Recorded expectation.** A null result on the primary criterion is likely: the tail is 1.1%
of rows, so even a large tail improvement may move short pooled WAPE by less than the 0.01
adoption threshold, and the metric weights by demand, which these low-volume collapsed SKUs
lack. The honest hope is criterion 2 passing while criterion 1 ties, which under Section 1.5
is a rejection for adoption. That outcome would still be worth recording, because it would
locate the problem as a metric-visibility issue rather than a model-capacity one, and
because the dashboard's per-SKU reliability tiers surface exactly the SKUs that pooled WAPE
is insensitive to.

**Status: rejected, at every value tested.**

| Pooled WAPE | v11 (200) | mcs=100 | mcs=50 | mcs=20 |
|---|---|---|---|---|
| short, Mar-May | 0.1961 | 0.2007 (+0.0046) | 0.1981 (+0.0020) | 0.1967 (+0.0006) |
| short, Dec-Feb | 0.2000 | 0.1979 (−0.0021) | 0.2009 (+0.0009) | 0.1992 (−0.0008) |
| short, Oct-Dec (ref) | 0.1783 | 0.1857 (+0.0074) | 0.1900 (+0.0117) | 0.1828 (+0.0045) |
| long, all three windows | unchanged | identical | identical | identical |
| tail, short ramp < 0.7 | 1.0854 | 1.1153 (+0.0299) | 1.1085 (+0.0231) | 1.0874 (+0.0020) |

Criterion 1 fails: no consistent sign across the decision windows, with 100 improving
Dec-Feb while regressing Mar-May and 50 doing the reverse, and every difference an order of
magnitude below the 0.01 adoption threshold. These are ties. Criterion 2 fails in the
opposite direction to the hypothesis: the tail was required to improve by 0.05 and instead
degraded at all three values. Long is identical everywhere, so the control holds and the
movement is genuinely confined to short. Experiment: `scripts/ml_25_v14_min_child.py`.

**What this settles.** The collapsing tail is not capacity-constrained. The pre-registered
reasoning was that 1.1% of training rows and a five-leaf ceiling prevented the trees from
resolving the region; quadrupling that headroom did not help and mildly hurt, so the ceiling
was never the binding constraint. The model has the ramp feature, has room to split on it,
and reverts anyway. That relocates the cause to the objective rather than the tree
structure: under `regression_l1` with `sample_weight = level` (Section 4.6), a leaf covering
collapsed SKUs holds mostly low-volume anchors whose contribution to the loss is small
however finely the region is partitioned. Finer partitioning of a lightly weighted region
changes little, and costs some variance, which is what the small consistent regressions look
like.

This is an independent confirmation of Section 4.26 on a subpopulation that section's
aggregate scoring could not have detected, so it strengthens that conclusion rather than
qualifying it. Hyperparameters are now settled for both the bulk and the tail.

**The measurement point, which outlives the experiment.** Tail pooled WAPE is 1.0854, above
100% error, while short pooled WAPE is 0.196. Both are correct: the metric weights by units
and these SKUs carry almost none. Pooled WAPE is the right metric for the inventory decision
it serves and cannot be the instrument for this problem, because a subpopulation that is
1.1% of rows and a smaller share of units cannot move it by the 0.01 the adoption rule
requires. Any future attempt at the collapsing tail needs a stated per-segment or per-SKU
criterion agreed in advance, as criterion 2 was here. The dashboard's per-SKU reliability
tiers already surface these SKUs, which is the appropriate place for them to be visible.

**Remaining candidates, none tested.** Down-weighting or excluding preorder rows from
training (Section 2.1, now unblocked); the stockout correction of Section 5.3, since a
collapse that is really a stockout is a censored observation rather than a demand signal;
and a demand-weighting change, which is the mechanism this experiment implicates but is a
change to the metric's own definition of importance and should not be made casually.
