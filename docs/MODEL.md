# The model: what it does and how it works

**Audience:** whoever picks up the modelling. Assumes pandas, cross-validation and
gradient-boosted trees, and does not re-teach them.

**Where this sits.** `OVERVIEW.md` has the problem, the measured results and the evaluation
protocol; read it first. This document is the architecture and the reasoning. The
exhaustive record is `ML_FORECAST_DESIGN.md`, Section 4 for every decision with its evidence
and Section 6 for the version log. Where this document is less specific than that one, that
one wins.

**The rule that governs the project.** One hypothesis at a time, evaluated on the three
development windows through `src/ml/evaluate.py`, judged by the decision rule in
`OVERVIEW.md` Section 5, and recorded in the decision log whether adopted or rejected. Pass
criteria are written down **before** the experiment runs. If you keep nothing else from this
project, keep that: most of the value in the record is the list of things that did not work.

---

## 1. The problem in modelling terms

Forecast weekly unit demand per SKU, out to thirteen weeks, for the 340 SKUs that sell
regularly. Roughly two years of history. No covariates worth speaking of beyond the sales
record itself.

Two facts shape everything else.

**1. Two years means two observations of each annual seasonal event.** That is not enough to
learn seasonality from data.

**2. Most SKUs are young.** 247 of 340 forecastable SKUs have under 50 weeks of history, so
per-SKU model fitting has very little to work with.

Fact 1 rules out the M5-competition approach, in which LightGBM is handed calendar features
and learns seasonality itself. That was tried here and failed exactly as the literature
predicts: it reproduced the specific months it had seen rather than a general pattern, and
over-forecast the post-holiday trough by +123% (design doc Section 4.9).

Fact 2 is the argument for a **global** model. A single model trained across all SKUs
jointly lets a pattern learned on one SKU, such as how demand behaves after launch, transfer
to another. That cross-SKU transfer is the main structural advantage this approach has over
the per-SKU statistical prototype, and it is measurable: a short-only model loses it and gets
worse (v8).

The design therefore follows **M4** rather than M5. Scale and seasonality are imposed
structurally; LightGBM is confined to learning the residual dynamics the data can support.

---

## 2. Architecture

### 2.1 The target is a ratio, not a level

For each SKU and target week, the model predicts

```
y_target / (trailing 12-week mean of that SKU's history at the cutoff)
```

This is the single most important design choice to understand. It has a property worth
appreciating: **a model that learns nothing predicts a ratio of 1.0, which reproduces the
structural baseline exactly.** Every measured movement is therefore attributable to the
feature under test, which is what makes the version log interpretable at all.

The trailing 12-week mean is called the **anchor level** and appears in the code as `denom`
(`src/ml/model.py:264`). Every ratio feature is divided by it.

### 2.2 Seasonality is a round-trip, applied per segment

Hand-set monthly multipliers are divided out of the history before fitting, and the
forecasts are multiplied back afterwards. The model never sees the calendar.

**This is applied to long SKUs only.** Short SKUs receive no seasonal adjustment. That split
is a measured result rather than an assumption (design doc Sections 4.10 and 4.17), and it
is implemented by `adjusted_series(weekly, long_uids)` at `src/ml/model.py:176`.

The multipliers live in `src/ml/seasonal.py`. A separate holiday window overrides the
monthly factor between `ML_HOLIDAY_START = (11, 20)` and `ML_HOLIDAY_END = (12, 15)` at
`ML_HOLIDAY_MULTIPLIER = 1.26` (`config.py:141-143`). Note the ML end date of 15 December
differs from the prototype's 31 December; v9 moved it because the promotions run late
November to mid December. `ML_SEASONAL_BLEND` is `"off"` for v11; the `"holiday"` and
`"full"` blends were tried as v16 and v15 and rejected.

### 2.3 Multi-horizon by direct forecasting

One model covers all thirteen horizons, with the horizon itself as an input feature
(`lead`). There is no recursion, so an error at week 1 does not compound into week 13.

### 2.4 The current model, v11: a hybrid

Two models, because a single shared model could not serve both segments.

| Segment | Trained on | Features |
|---|---|---|
| **smooth/short** | **all** smooth SKUs | `lead`, `ramp_4_12`, `y_last_r`, `lag_1_r` |
| **smooth/long** | **long SKUs only** | `lead`, `y_last_r`, `lag_1_r`, `elev_long` |

Defined at `src/ml/model.py:40` (`FEATURES_V1`) and `:53` (`FEATURES_V11_LONG`).

The long model **drops** `ramp_4_12`, the growth feature that drives a persistent
over-forecast, and **adds** `elev_long`, which measures how elevated the SKU is against its
own annual norm. The recent-level ratios stay in both, so gradual growth is still tracked;
only sharp-ramp extrapolation is removed and replaced with mean-reversion pressure.

### 2.5 The features, defined

All are computed in `build_matrix` (`src/ml/model.py:191-291`). `denom` is the anchor level
from 2.1; `roll4`, `roll12` and `roll52` are trailing means over that many weeks.

| Feature | Expression | What it tells the model |
|---|---|---|
| `lead` | 1 to 13 | How far ahead this row is forecasting. Makes one model serve all horizons. |
| `ramp_4_12` | `roll4 / denom` | Last month against last quarter. Detects a SKU that is ramping. **Short model only.** |
| `y_last_r` | `y_feat / denom` | The anchor week against the level. Recent position. |
| `lag_1_r` | `shift(1) / denom` | The week before the anchor against the level. Together with `y_last_r`, short-run direction. |
| `elev_long` | `roll4 / roll52` | Four weeks against the SKU's own annual level. High means running unsustainably hot, so revert. **Long model only.** |

`elev_long` needs 52 rows with `min_periods=52` and cannot be computed below a year of
history, which is neutralised to 1.0 when absent. **This is why the short/long boundary sits
at 50 weeks in the first place.** The boundary exists because the feature requires it, which
is not recorded in the design doc and has been misread once already.

### 2.6 Why split the models rather than add a segment indicator

Because that was tried, twice. A segment indicator (v4) and per-segment sample weighting
(v7) both failed for the same reason: in a shared model the trees are shared, so a change
made for long SKUs damages short SKUs through them. Splitting removes the shared trees.
Short SKUs keep the shared model because they need the cross-segment transfer; long SKUs do
not.

### 2.7 Hyperparameters

Near-default, at `src/ml/model.py:350-353`:

```python
n_estimators=3000,          # cap; early stopping picks the real count
learning_rate=0.05,
num_leaves=31,
min_child_samples=200,
```

Early stopping runs against an internal validation slice. Tuning was tried twice, as v10 and
again as v18 on the hybrid architecture, and rejected both times. **The model is not
misconfigured; that is not where the remaining error is.**

---

## 3. Segmentation: how a SKU gets its labels

Computed weekly by `src/profile.py`, written to `data/processed/sku_profiles.csv`. This is
what decides whether a SKU is forecast at all and which model serves it.

### 3.1 Bucket: smooth or intermittent

Two independent filters, both in `classify()` (`src/profile.py:216`). A SKU is
**intermittent** if either applies:

| Test | Constant | Value |
|---|---|---|
| 30% or more of weeks are zero | `ZERO_PCT_INTERMITTENT` | 0.30 |
| Mean weekly demand below the cutoff | `MEAN_INTERMITTENT_CUTOFF` | 3.0 |

Otherwise it is **smooth**.

### 3.2 The dynamic overrides, and the hysteresis

The bucket is then adjusted on the trailing `RECENT_WEEKS = 13` window.

**Promote**, intermittent to smooth (`src/profile.py:238-245`): recent zero percentage below
`RECENT_ZERO_PCT_UPGRADE = 0.20` **and** recent mean at or above
`RECENT_MEAN_UPGRADE = 3.0`.

**Demote**, smooth to intermittent: recent mean below `RECENT_MEAN_DOWNGRADE = 2.0`.

**The two thresholds differ on purpose, and this is not a bug.** Equal bars would make a SKU
hovering at the threshold flip between smooth and intermittent week after week, and each
flip removes or restores its forecast. The gap between 3.0 and 2.0 is hysteresis.

### 3.3 History length

`_history_length(active_weeks)` at `src/profile.py:76`:

| Label | Threshold | Meaning |
|---|---|---|
| `short` | under 50 weeks | Too little history for the long model's inputs |
| `medium` | 50 to 104 | Enough for `elev_long` |
| `full` | 104 or more | Two or more seasonal cycles |

`medium` and `full` are merged into "long" everywhere the model is concerned.

**A known off-by-one, deliberately left.** `active_weeks` is a span,
`(data_end - train_start).days / 7`, so a window of 50 rows counts as 49 weeks. A SKU
therefore needs **51** rows to be labelled medium, and the boundary named "50 weeks" sits a
week later than it reads. Evaluation and production agree with each other because
`dataset.asof_history_length` computes the same span, so they are consistently one week late
rather than inconsistent. Changing it moves SKUs across the boundary in both places, which
re-baselines every recorded figure including the final test's. It is a measurement decision
with a re-baseline attached rather than a correction. See `FUTURE_IMPROVEMENTS.md`.

### 3.4 Promoted SKUs and `_smooth_onset`

When a SKU is promoted, its `train_start` must be set to where its smooth behaviour actually
begins. `_smooth_onset` (`src/profile.py:134`) walks backwards from the recent window and
finds the longest span that still satisfies the promotion test.

**Read the docstring before touching this.** The original code assigned `train_start` to the
start of the 13-week window unconditionally, which pinned every promoted SKU's history to a
date that moves forward every profiling run and sits in the future relative to every
backtest cutoff. All 190 promoted SKUs, 41% of the then-smooth set, were silently dropped
from every evaluation figure ever recorded. Only 15 of them genuinely had 13 weeks; the
median had 34 and the maximum 111. Fixed 2026-08-11.

**This is why no figure recorded before 2026-08-11 is comparable to one recorded after.**
When you meet an old number, check which snapshot it was measured on.

---

## 4. Running an experiment

```bash
.venv/bin/pip install -r requirements.txt     # exact pins; results compare at the 3rd decimal
.venv/bin/python scripts/ml_XX_your_experiment.py
```

`lightgbm` is not currently installed in `.venv`. Install requirements before any `ml_*`
script.

Experiments live in `scripts/ml_*.py`, import from `src/ml/`, and add nothing to it, so the
library stays reusable. Each runs its candidate against the three development windows and
prints a per-segment table.

The workflow:

1. Write the hypothesis and the pass criteria into the version log **before running**.
2. Run on the three development windows only. Never the final test window.
3. Show the raw per-segment output before summarising it.
4. Record the outcome in the decision log, **including rejection**.

Model versions are tagged commits (`model/v-base` onward), each holding that version's tree
with its results in the commit message. Check one out to re-run it.

**The final test window is quarantined.** `scripts/ml_41_final_test.py` is the only script
that touches it. It refuses to run against a dirty tree, refuses to overwrite an existing
result, and checks its preconditions first, because a run against a stale snapshot or a
mismatched LightGBM does not produce a weaker result, it produces one that cannot be
interpreted at all, and the window is spent either way.

**It has now been run.** Section 4.35 of the design doc records the result and
`OVERVIEW.md` Section 6 summarises it. Do not run it again.

---

## 5. What was rejected, and must not be retried without new evidence

Each was tested and the reasoning is recorded at the section given.

| Rejected | Where | Why in one line |
|---|---|---|
| Calendar features for learned seasonality | 4.9 | Reproduced the months it had seen; +123% on the post-holiday trough |
| Segment indicator (v4) | 4.23 | Shared trees carry the damage between segments |
| Per-segment sample weighting (v7) | 4.24 | Same reason as above |
| Hyperparameter tuning (v10, v18) | 4.26 | No gain twice, on two architectures |
| Raw SKU age (v12) | 4.28 | Monotonic features extrapolate badly past the training range |
| Acceleration (v13) | Section 6 | No consistent sign |
| FBA channel share (v17) | Section 6 | No consistent sign |
| Seasonal blends (v15 full, v16 holiday) | Section 6 | Neither beat `ML_SEASONAL_BLEND="off"` |

Six perturbations of the long model each cost it 0.005 to 0.012. That list is a better
account of the work than the changes that landed.

---

## 6. Traps, from the ones actually hit

**The evaluation was silently excluding 41% of the forecastable catalogue.** Section 3.4
above. Every figure recorded before 2026-08-11 describes a different population.

**A recorded baseline figure was stale from v9 onward** and nobody noticed, because nothing
re-derived it. **If a number is quoted in prose rather than computed, assume it can rot.**
This project has now hit that failure three separate times: the baseline figure, five of six
mis-transcribed prototype cells in the design doc's own v11 table, and the V1 column
described in `OVERVIEW.md` Section 6.

**The evaluation selects its population using future information.** Eligibility is decided
partly by data after the cutoff. Documented, not fixed. It inflates results by an amount
that is hard to quantify.

**Promoted SKUs cannot be validated by backtest, only forward.** A SKU promoted into the
smooth set has, by construction, a history that would not have qualified it at an earlier
cutoff. Backtesting it answers a question about a SKU that did not exist in that form.

**The V1 comparison had an as-of off-by-one until 2026-08-13.** `v1_predictions` passed
`cutoff - 1 day`, which was correct under a Monday-to-Sunday week convention and wrong from
the moment that was reverted to Tuesday-to-Monday on 2026-08-06. Nobody updated it with the
revert. It mattered in the direction that flatters the model.

---

## 7. The other forecaster in this repository

`src/legacy/`, `api/legacy/` and `scripts/legacy/` hold the statsforecast prototype: per-SKU
AutoARIMA, AutoETS and moving averages selected by cross-validation. Retired 2026-08-13.
Nothing runs it and its router is not mounted.

For modelling purposes it matters for one reason: **it is the accuracy bar.** The success
criterion was never beating a moving average, it was beating this, per segment. It still
wins one cell, smooth/long in Oct-Dec, at 0.0918 against v11's 0.1040, and why is not
understood.

It has also **never been run through the shared harness.** Its figures come from its own
stored evaluation output, so the bar the project is judged against has been measured with
different code from everything it is compared to. That is the first item in
`FUTURE_IMPROVEMENTS.md` and the most valuable single thing left to close.
