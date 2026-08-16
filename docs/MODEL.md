# Model

Architecture and reasoning for the live model. Assumes pandas, cross-validation and gradient-boosted trees; `MODEL_PRIMER.md` covers the same ground without them. `OVERVIEW.md` has the problem, results and evaluation protocol.

`ML_FORECAST_DESIGN.md` is the exhaustive record: §4 for every decision with its evidence, §6 for the version log. It is authoritative wherever this document is less specific.

**Governing rule.** One hypothesis at a time, evaluated on the three development windows through `src/ml/evaluate.py`, judged by the adoption rule in `OVERVIEW.md` §5, recorded in the decision log whether adopted or rejected. Pass criteria are written down first.

## 1. Problem in modelling terms

Forecast weekly unit demand per SKU out to 13 weeks, for the 340 SKUs that sell regularly. Roughly two years of history, no useful covariates beyond the sales record.

| Fact | Consequence |
|---|---|
| Two years means two observations of each annual seasonal event, not enough to learn seasonality from data | Rules out the M5-competition approach, where LightGBM is handed calendar features and learns seasonality itself. Tried here: it reproduced the specific months it had seen and over-forecast the post-holiday trough by +123% (design doc §4.9) |
| Most SKUs are young. 247 of 340 forecastable SKUs have under 50 weeks of history, so per-SKU fitting has little to work with | Argues for a **global** model. One model trained across all SKUs jointly lets a pattern learned on one SKU transfer to another. That cross-SKU transfer is the main structural advantage over the per-SKU prototype, and it is measurable: a short-only model loses it and performs worse (v8) |

The design follows M4, not M5. Scale and seasonality are imposed structurally; LightGBM is confined to the residual dynamics the data supports.

## 2. Architecture

### 2.1 Target is a ratio, not a level

```
y_target / (trailing 12-week mean of that SKU's history at the cutoff)
```

The denominator is the **anchor level**, `denom` in `src/ml/model.py:264`. Every ratio feature is divided by it.

**Rationale.** A model that learns nothing predicts 1.0, reproducing the structural baseline, so every measured movement is attributable to the feature under test.

### 2.2 Seasonality as a round trip

| Element | Detail |
|---|---|
| Round trip | Hand-set monthly multipliers are divided out of history before fitting, then multiplied back into forecasts. The model never sees the calendar |
| Scope | **Long SKUs only.** Short SKUs receive no seasonal adjustment. Measured result (design doc §4.10, §4.17), implemented by `adjusted_series(weekly, long_uids)` at `src/ml/model.py:176` |
| Multipliers | `src/ml/seasonal.py` |
| Holiday override | Overrides the monthly factor between `ML_HOLIDAY_START = (11, 20)` and `ML_HOLIDAY_END = (12, 15)` at `ML_HOLIDAY_MULTIPLIER = 1.26` (`config.py:141-143`). The ML end date of 15 December differs from the prototype's 31 December; v9 moved it because promotions run late November to mid December |
| Blend | `ML_SEASONAL_BLEND` is `"off"` for v11. The `"holiday"` and `"full"` blends were tried as v16 and v15 and rejected |

### 2.3 Multi-horizon by direct forecasting

One model covers all 13 horizons with the horizon as an input feature (`lead`). No recursion, so a week 1 error does not compound into week 13.

### 2.4 Current model, v11

Two models; a shared model could not serve both segments.

| Segment | Trained on | Features |
|---|---|---|
| smooth/short | All smooth SKUs | `lead`, `ramp_4_12`, `y_last_r`, `lag_1_r` |
| smooth/long | Long SKUs only | `lead`, `y_last_r`, `lag_1_r`, `elev_long` |

Defined at `src/ml/model.py:40` (`FEATURES_V1`) and `:53` (`FEATURES_V11_LONG`).

The long model drops `ramp_4_12`, the growth feature driving a persistent over-forecast, and adds `elev_long`. Recent-level ratios stay in both, so gradual growth is tracked.

### 2.5 Feature definitions

Computed in `build_matrix` (`src/ml/model.py:191-291`). `roll4`, `roll12`, `roll52` are trailing means over that many weeks.

| Feature | Expression | Signal |
|---|---|---|
| `lead` | 1 to 13 | How far ahead this row forecasts. Makes one model serve all horizons |
| `ramp_4_12` | `roll4 / denom` | Last month against last quarter. Detects a ramping SKU. Short model only |
| `y_last_r` | `y_feat / denom` | Anchor week against the level. Recent position |
| `lag_1_r` | `shift(1) / denom` | Week before the anchor against the level. With `y_last_r`, short-run direction |
| `elev_long` | `roll4 / roll52` | Four weeks against the SKU's annual level. High means running hot, so revert. Long model only |

`elev_long` requires 52 rows (`min_periods=52`), cannot be computed below a year of history, and is neutralised to 1.0 when absent.

Note: the short/long boundary sits at 50 weeks because `elev_long` requires it.

### 2.6 Why split the models

A segment indicator (v4) and per-segment sample weighting (v7) both failed for one reason: shared trees carry a change made for long SKUs into short SKUs. Splitting removes the shared trees. Short SKUs keep the shared model for cross-segment transfer; long SKUs do not need it.

### 2.7 Hyperparameters

Near-default, at `src/ml/model.py:350-353`:

```python
n_estimators=3000,          # cap; early stopping picks the real count
learning_rate=0.05,
num_leaves=31,
min_child_samples=200,
```

Early stopping runs against an internal validation slice. Tuning was tried as v10 and again as v18 on the hybrid architecture, rejected both times.

## 3. Segmentation

Computed weekly by `src/profile.py`, written to `data/processed/sku_profiles.csv`. Decides whether a SKU is forecast and which model serves it.

### 3.1 Bucket

Two independent filters in `classify()` (`src/profile.py:216`). A SKU is **intermittent** if either applies, otherwise smooth.

| Test | Constant | Value |
|---|---|---|
| 30% or more of weeks are zero | `ZERO_PCT_INTERMITTENT` | 0.30 |
| Mean weekly demand below the cutoff | `MEAN_INTERMITTENT_CUTOFF` | 3.0 |

### 3.2 Dynamic overrides

Adjusted on the trailing `RECENT_WEEKS = 13` window.

| Override | Condition | Location |
|---|---|---|
| Promote (intermittent to smooth) | Recent zero percentage below `RECENT_ZERO_PCT_UPGRADE = 0.20` **and** recent mean at or above `RECENT_MEAN_UPGRADE = 3.0` | `src/profile.py:238-245` |
| Demote (smooth to intermittent) | Recent mean below `RECENT_MEAN_DOWNGRADE = 2.0` | |

**Rationale.** Equal bars would make a SKU near the threshold flip weekly, and each flip removes or restores its forecast. The gap between 3.0 and 2.0 is hysteresis.

### 3.3 History length

`_history_length(active_weeks)` at `src/profile.py:76`:

| Label | Threshold | Meaning |
|---|---|---|
| `short` | Under 50 weeks | Too little history for the long model's inputs |
| `medium` | 50 to 104 | Enough for `elev_long` |
| `full` | 104 or more | Two or more seasonal cycles |

`medium` and `full` merge into "long" everywhere the model is concerned.

Warning: known off-by-one, left in place. `active_weeks` is a span, `(data_end - train_start).days / 7`, so a window of 50 rows counts as 49 weeks and a SKU needs **51** rows to be labelled medium. `dataset.asof_history_length` computes the same span, so evaluation and production are consistently one week late. Changing it moves SKUs across the boundary in both and re-baselines every recorded figure including the final test's. See `FUTURE_IMPROVEMENTS.md` §3.

### 3.4 Promoted SKUs and `_smooth_onset`

A promoted SKU's `train_start` must be set to where smooth behaviour begins. `_smooth_onset` (`src/profile.py:134`) walks backwards from the recent window to the longest span still satisfying the promotion test. Its docstring precedes any change.

Warning: no figure recorded before 2026-08-11 is comparable to one recorded after. A defect fixed on that date had excluded 190 promoted SKUs, 41% of the then-smooth set, from every evaluation figure. Older figures carry the snapshot they were measured on.

## 4. Running an experiment

```bash
.venv/bin/pip install -r requirements.txt     # exact pins; results compare at the 3rd decimal
.venv/bin/python scripts/ml_XX_your_experiment.py
```

`lightgbm` is not currently installed in `.venv`. Install requirements before any `ml_*` script.

Experiments live in `scripts/ml_*.py` and import from `src/ml/` without adding to it. Each runs its candidate against the three development windows and prints a per-segment table.

1. Write the hypothesis and pass criteria into the version log before running.
2. Run on the three development windows only, never the final test window.
3. Show the raw per-segment output before summarising.
4. Record the outcome in the decision log, including rejection.

Model versions are tagged commits (`model/v-base` onward), each holding that version's tree, with its results in the commit message.

**The final test window is quarantined.** `scripts/ml_41_final_test.py` is the only script that touches it. It refuses a dirty tree, refuses to overwrite an existing result, and checks preconditions first.

It has been run. Design doc §4.35 records the result, `OVERVIEW.md` §6 summarises it. Do not run it again.

## 5. Rejected approaches

Each was tested. Do not retry without new evidence.

| Rejected | Reference | Reason |
|---|---|---|
| Calendar features for learned seasonality | §4.9 | Reproduced the months it had seen; +123% on the post-holiday trough |
| Segment indicator (v4) | §4.23 | Shared trees carry damage between segments |
| Per-segment sample weighting (v7) | §4.24 | Same as above |
| Hyperparameter tuning (v10, v18) | §4.26 | No gain twice, on two architectures |
| Raw SKU age (v12) | §4.28 | Monotonic features extrapolate badly past the training range |
| Acceleration (v13) | §6 | No consistent sign |
| FBA channel share (v17) | §6 | No consistent sign |
| Seasonal blends (v15 full, v16 holiday) | §6 | Neither beat `ML_SEASONAL_BLEND="off"` |

Six perturbations of the long model each cost it 0.005 to 0.012.

## 6. Measurement limits

`OVERVIEW.md` §7 has the full list. Three bear on modelling work:

- **Quoted figures rot.** A number stated in prose can go stale without anything failing; this has happened three times. Computed values are preferred
- **Promoted SKUs cannot be validated by backtest, only forward.** A promoted SKU has, by construction, a history that would not have qualified it at an earlier cutoff
- **The V1 comparison carried an as-of off-by-one until 2026-08-13**, erring in the direction that flatters the model

## 7. The retired forecaster

`src/legacy/`, `api/legacy/` and `scripts/legacy/` hold the statsforecast prototype: per-SKU AutoARIMA, AutoETS and moving averages selected by cross-validation. Retired 2026-08-13, router not mounted.

It is the accuracy bar; the success criterion was beating it per segment. It still wins one cell, smooth/long in Oct-Dec, at 0.0918 against v11's 0.1040. Why is not understood.

It has never been run through the shared harness; its figures come from its own stored evaluation output, so the bar was measured with different code. First item in `FUTURE_IMPROVEMENTS.md`.
