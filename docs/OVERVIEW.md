# Overview

Entry point: what was built, how the pieces connect, how accuracy is measured, and the results.

| Document | Covers |
|---|---|
| `MODEL.md` | Model architecture, features, how to change it |
| `MODEL_PRIMER.md` | The same ideas without ML background assumed |
| `DATA_AND_PIPELINE.md` | Data sources and the weekly job |
| `SCREENS.md` | The two UI screens and their endpoints |
| `DEPLOYMENT.md` | Server setup, local development, runbook |
| `FUTURE_IMPROVEMENTS.md` | Open work, grouped by what blocks it |

Three reference files hold the full record, authoritative where this document is less specific.

- `ML_FORECAST_DESIGN.md`: every experiment with its evidence, and the decision log including rejections
- `WORKLOG.md`: dated account of work performed
- `BACKLOG.md`: itemised known work

## 1. Problem

3,525 active SKUs of vehicle seat covers and accessories. Purchasing needs a weekly unit demand estimate per SKU over the coming quarter.

**V1**, the incumbent, is a spreadsheet formula in `src/v1.py`: sales velocity over 7, 15, 30, 60 and 90 day windows times hand-set monthly seasonal factors. Two structural limits:

1. A trailing average cannot anticipate change; it under-forecasts ramps and over-forecasts fades
2. Its error is large and sign-flips by season, so no correction fixes it

Output is a 13-week-ahead weekly unit forecast per SKU, published to Demand Pilot, where the Action List converts it into recommended order quantities.

## 2. System components

Two repositories, two processes, one database.

| Repository | Stack | Role |
|---|---|---|
| `Time_Series_Forecasting` | Python, FastAPI | Model, weekly pipeline, and the API serving forecasts and planning calculations. Service on port 8000, also run by the weekly cron. Reads and writes PostgreSQL directly |
| `Commerce_Integration` | TypeScript, Next.js 16 | Hosts the Action List and Forecast Validation screens. Planning figures are computed in Python and proxied through thin handlers under `/api/planning/*` |

**Rationale.** The separation holds the order formula to one implementation.

```
shipcore.fc_velocity_link_snapshot_forecast   (order lines, source of truth)
        │  weekly pipeline: scripts/ml_prepare_data.py
        ▼
data/processed/sales_clean.parquet            (one row per SKU per week)
data/processed/sku_profiles.csv               (segment labels per SKU)
        │  LightGBM v11
        ▼
shipcore.ml_forward_forecasts                 (13 weeks ahead, per SKU)
shipcore.ml_forecast_history                  (every forecast served)
        │  src/planning/calc.py + live inventory
        ▼
FastAPI  /planning/action-list, /planning/validation
        │  Next.js proxy, src/lib/planning-api.ts
        ▼
Action List and Forecast Validation screens
```

## 3. Forecast scope

SKUs are classified at training time by demand pattern. Counts from pinned snapshot `2026-08-03-v2`.

| Class | Meaning | SKUs |
|---|---|---|
| smooth | Regular enough to model | 340 |
| intermittent | Zero sales in most weeks | 3,185 |
| | **Total** | **3,525** |

Smooth SKUs are 9.6% of the catalogue and carry 78.6% of all recorded units, 75.8% over the trailing year.

**Rationale.** Expected units per week is not meaningful when most weeks are zero. Intermittent SKUs receive no forecast; the Action List displays them on a different basis (`SCREENS.md` §2.9).

Smooth SKUs split again by available history.

| Segment | Threshold | SKUs |
|---|---|---|
| smooth/short | Under 50 weeks of active sales | 247 |
| smooth/long | 50 weeks or more | 93 |

`long` merges `medium` (50 to 104 weeks, 40 SKUs) and `full` (104 or more, 53 SKUs). Labels are recomputed weekly, so a SKU enters or leaves the forecast with no code change.

Warning: the smooth population shifted by 15 SKUs in one weekly refresh. Catalogue counts are approximate and dated; evaluation counts are exact, a property of the snapshot.

## 4. What was built

### Live model: LightGBM v11

Gradient-boosted trees trained across all SKUs jointly, predicting a **demand multiplier** on each SKU's recent level. Seasonality is divided out before fitting and multiplied back after. Short SKUs use a model trained on all smooth SKUs, long SKUs one trained on long SKUs only.

**Rationale.** A model that learns nothing predicts 1.0, reproducing a plain 12-week moving average. That average is the floor, the **structural baseline**; accuracy above it is attributable to the model. Architecture: `MODEL.md`.

### Retired prototype: statsforecast

Phase one was per-SKU AutoARIMA, AutoETS and moving averages selected by cross-validation. Retired 2026-08-13. Code remains under `src/legacy/`, `api/legacy/` and `scripts/legacy/`; the router is not mounted.

It remains the accuracy bar; the success criterion was beating it per segment.

Removed at the same time:

| Removed | Was at |
|---|---|
| Demand Forecast page | `/planning/demand-forecast` |
| SKU Planning's Demand Forecast tab | `/planning/sku-forecasts/[sku]?tab=forecast` |
| Fourteen Next.js proxy routes | `/api/forecast/*` |

`?tab=forecast` still resolves and lands on Sales Analysis. The per-SKU view of the served model is `/planning/action-list/[sku]`.

## 5. Accuracy measurement

### Metric

**Pooled WAPE per segment.** Sum absolute error across SKUs, divide by summed actuals. Lower is better. 0.20 means forecasts were off by 20% of the units the segment sold.

- Pooling weights SKUs by volume, matching what an error costs
- Bias, the signed version, is reported alongside; equal WAPE can be wrong in opposite directions
- Scoring is on the 10-week total per SKU, since stock is ordered for a horizon

### Windows

Four rolling-origin windows, each scored on data after its training cutoff. The three development windows span three seasonal regimes.

| Window | Cutoff | Test period | Role |
|---|---|---|---|
| Oct-Dec 2025 | 2025-10-06 | 2025-10-13 to 2025-12-15 | Development, Q4 peak |
| Dec-Feb | 2025-12-15 | 2025-12-22 to 2026-02-23 | Development, post-holiday trough |
| Mar-May 2026 | 2026-02-23 | 2026-03-02 to 2026-05-04 | Development, spring |
| Final test | 2026-05-04 | 2026-05-11 to 2026-07-13 | Quarantined, used once |

### Adoption rule

A change is adopted only if it:

1. Improves accuracy in the same direction across all three development windows, and
2. Gains at least 0.01 WAPE on the three-window mean

Single-window differences below 0.02 are inconclusive. Bootstrap resampling puts single-window noise at ±0.011 to ±0.014.

Pass criteria enter the version log before the experiment runs. Adoptions and rejections are both recorded, v0 through v18.

### Two pins

- `ML_FINAL_TEST_CUTOFF` fixes which weeks each window covers
- `ML_DATA_SNAPSHOT` fixes the values inside those weeks, since the weekly refresh revises recent actuals as late orders register

Warning: advancing either re-baselines every recorded number. Advancing one does not imply the other followed.

## 6. Results

### Development windows

Pooled WAPE on snapshot `2026-08-03-v2`. Lower is better.

| Segment and window | v11 (current) | prototype | V1 (spreadsheet) |
|---|---|---|---|
| short, Mar-May | **0.1926** | 0.2028 | 0.3351 |
| short, Dec-Feb | **0.1994** | 0.2904 | 0.2240 |
| short, Oct-Dec *(reference only)* | **0.2473** | 0.4137 | 0.2210 |
| long, Mar-May | **0.1350** | 0.1437 | 0.2776 |
| long, Dec-Feb | **0.1389** | 0.2690 | 0.3928 |
| long, Oct-Dec | 0.1040 | **0.0918** | 0.0851 |

Sources differ by column:

- v11: `docs/rebaseline_2026-08-03-v2/ml_22_v11_hybrid.log`
- V1: re-measured 2026-08-13 after an as-of off-by-one was fixed in `scripts/ml_02_v1_benchmark.py`, recorded in `ML_FORECAST_DESIGN.md` §1.1. `ml_02_v1_benchmark.log` predates the fix and is superseded

The Oct-Dec short cell is reference only: 14 short SKUs were eligible at that cutoff, and it is excluded from short-segment decisions.

v11 beats the spreadsheet in four of six cells and loses both Oct-Dec cells by 0.02 to 0.04.

Note: why V1 is more robust in Q4 is not understood, and is the largest open question.

### Final test

Run once on 2026-08-13 at 12:02 PDT, commit `4a19ca1`, snapshot `2026-08-03-v2`. Cutoff 2026-05-04, test weeks 2026-05-11 to 2026-07-13, 303 SKUs.

| Segment | v11 | V1 | Structural baseline |
|---|---|---|---|
| smooth/short | 0.2061 | 0.3772 | 0.2013 |
| smooth/long | 0.1324 | 0.1872 | 0.1282 |
| **TOTAL** | **0.1784** | **0.3059** | **0.1739** |

**Primary criterion passed.** v11 beats V1 on both segments and on TOTAL.

| Comparison | Delta | SE | 95% CI | Verdict |
|---|---|---|---|---|
| short, v11 vs V1 | −0.1711 | 0.0215 | [−0.2135, −0.1269] | Significant |
| long, v11 vs V1 | −0.0548 | 0.0174 | [−0.0879, −0.0198] | Significant |
| short, v11 vs baseline | +0.0048 | 0.0108 | [−0.0169, +0.0255] | Indistinguishable |
| long, v11 vs baseline | +0.0042 | 0.0141 | [−0.0240, +0.0305] | Indistinguishable |

On TOTAL, v11's error is 42% smaller than the spreadsheet's. V1 came in 28% low against v11's 0.0%, and 12.9% high on short SKUs post-holiday. Both deltas against the structural baseline are under one standard error.

### Where the model adds value

Across four windows and eight segment cells, v11 is never significantly worse than the baseline and is better in two, both seasonal turning points.

| | Structural baseline | v11 |
|---|---|---|
| Q4 ramp-up, short SKUs (WAPE) | 0.4605 | **0.2473** |
| Q4 ramp-up, short SKUs (bias) | −46.0% | **+7.5%** |
| Post-holiday, long SKUs (WAPE) | 0.2171 | **0.1389** |
| Post-holiday, long SKUs (bias) | +16.7% | **+4.3%** |

The final test window, May to July, is a flat stretch, where the two behave identically.

**Rationale.** A trailing average cannot turn a corner it has not yet seen.

Unproven: whether v11 helps in ordinary quarters; on current evidence it neither helps nor hurts.

## 7. Stated limitations

Full list in `ML_FORECAST_DESIGN.md` §2.2 and `MODEL.md` §6.

- **Two years of data.** Every seasonal result rests on one or two observations of each season
- **The four windows are not independent samples.** Training sets overlap heavily; they test robustness across seasons
- **The long model rests on about 54 SKUs**, 23 of them correlated `CC-CN-03`/`CC-CP-03` variants, so its confidence intervals overstate confidence
- **Evaluated horizon is 10 weeks; production serves 13.** Weeks 11 to 13 are scored by no window, so every figure is a lower bound
- **Eligibility uses future information.** Scored SKUs are chosen partly from data after the cutoff, inflating results by an amount hard to quantify. Documented, not fixed
- **The prototype has never been run through the shared harness.** Its stored evaluation output is used, so the bar was measured with different code from everything compared to it. First item in `FUTURE_IMPROVEMENTS.md`
- **The quarantined window was evaluated once during development**, on 2026-08-11 by `scripts/ml_34_asof_bucket_audit.py`; nothing was tuned on it
- **No prediction intervals.** v11 emits a point forecast only. Reasoning in `FUTURE_IMPROVEMENTS.md` §4
