# Overview: what this project is and how well it works

**Audience:** anyone picking this up. Read this first. It explains what was built, how the
pieces fit together, how the forecast is evaluated, and what the measured results are.

**The other documents.** This one is the entry point and the other three follow from it.

| Question | Document |
|---|---|
| How does the model work, and how do I change it? | `MODEL.md` |
| Where does the data come from, and what runs weekly? | `DATA_AND_PIPELINE.md` |
| How do the two screens work, and how do I maintain them? | `SCREENS.md` |
| What is worth doing next? | `FUTURE_IMPROVEMENTS.md` (standalone, read when you want it) |

**The record, kept but not curated.** `ML_FORECAST_DESIGN.md` is the full decision log and
version log, with every experiment and its evidence including the rejections. `WORKLOG.md`
is the dated account of what happened. `BACKLOG.md` is the item-by-item list of known work.
These are references to search rather than documents to read start to finish, and they are
the authority whenever this document is less specific than they are.

---

## 1. The problem

The company sells vehicle seat covers and accessories, 3,525 active SKUs on the pinned data
snapshot. Somebody has to decide how many of each to order and when. That decision needs an
answer to "how many of these will we sell per week over the next quarter".

The answer in production before this project is a spreadsheet formula, referred to
throughout as **V1**. It blends sales velocity over 7, 15, 30, 60 and 90 day windows and
multiplies by hand-set monthly seasonal factors. It has two structural problems. A trailing
average cannot anticipate change, so it under-forecasts products that are ramping and
over-forecasts ones that are fading. And its error is large and changes sign by season, so
no single correction fixes it.

This project produces a weekly unit demand forecast per SKU, thirteen weeks ahead, and
publishes it to the Demand Pilot web application, where the Action List turns it into
recommended order quantities.

---

## 2. What the system is made of

Two repositories, two processes, one database.

### `Time_Series_Forecasting` (Python, FastAPI)

The forecasting system. It holds the model, the weekly pipeline, and the API that serves
forecasts and planning calculations. It runs as a long-lived service on port 8000 and is
also what the weekly cron invokes. It reads and writes PostgreSQL directly.

### `Commerce_Integration` (TypeScript, Next.js 16)

The web application, called Demand Pilot. It owns the screens. It computes nothing about
forecasting: every planning figure on screen is calculated in Python and proxied through
thin Next.js route handlers under `/api/planning/*`. That separation is deliberate and worth
preserving, because it means there is exactly one implementation of the order formula.

### How a number reaches the screen

```
shipcore.fc_velocity_link_snapshot_forecast   (order lines, the source of truth)
        │
        │  weekly pipeline: scripts/ml_prepare_data.py
        ▼
data/processed/sales_clean.parquet            (one row per SKU per week)
data/processed/sku_profiles.csv               (segment labels per SKU)
        │
        │  LightGBM v11
        ▼
shipcore.ml_forward_forecasts                 (13 weeks ahead, per SKU)
shipcore.ml_forecast_history                  (every forecast ever served)
        │
        │  src/planning/calc.py  +  live inventory
        ▼
FastAPI  /planning/action-list, /planning/validation
        │
        │  Next.js proxy, src/lib/planning-api.ts
        ▼
Action List and Forecast Validation screens
```

`DATA_AND_PIPELINE.md` covers every step of this. `SCREENS.md` covers the last two.

---

## 3. What gets forecast, and what does not

Not every SKU can be usefully forecast. Each is classified at training time by its demand
pattern. Counts below are from the pinned snapshot `2026-08-03-v2` and were recomputed for
this document rather than copied from an older one.

| Class | Meaning | SKUs |
|---|---|---|
| **smooth** | Demand regular enough to model | 340 |
| **intermittent** | Zero sales in most weeks | 3,185 |
| | **Total** | **3,525** |

Smooth SKUs are 9.6% of the catalogue and carry **78.6%** of all recorded units, 75.8% over
the trailing year. That concentration is the reason forecasting only the smooth SKUs is a
reasonable scope rather than a gap. No forecast is produced for intermittent SKUs at all;
"expected units next week" is not a quantity that means anything when most weeks are zero.
The Action List still shows them, on a different basis, described in `SCREENS.md`.

Smooth SKUs are split again by how much history they have, because that changes what is
learnable:

| Segment | Threshold | SKUs |
|---|---|---|
| **smooth/short** | Under 50 weeks of active sales | 247 |
| **smooth/long** | 50 weeks or more | 93 |

`long` merges two internal labels, `medium` (50 to 104 weeks, 40 SKUs) and `full` (104 or
more, 53 SKUs). They are treated together because the medium group alone is too small to
evaluate reliably.

These labels are recomputed every week, so a SKU moves between them over time. That is
intended, and it is why a SKU can appear in or vanish from the forecast without anyone
changing anything.

**A caution on quoting these counts.** Catalogue composition moves. The smooth population
shifted by fifteen SKUs in a single weekly refresh during this project, and figures written
before 2026-08-11 describe a materially different population, because a profiling defect
was excluding 41% of forecastable SKUs from every evaluation until then. Treat catalogue
counts as approximate and dated. Evaluation counts are different and are stated exactly,
because they are a fixed property of a pinned snapshot.

---

## 4. What was built

### The live model: LightGBM v11

A single gradient-boosted tree model trained across all SKUs jointly, which predicts a
**demand multiplier** on each SKU's recent level rather than predicting units directly.
Seasonality is divided out before fitting and multiplied back afterwards. It is a hybrid:
short-history SKUs are served by a model trained on all smooth SKUs, long-history SKUs by a
dedicated model trained only on long SKUs.

The property worth understanding is that a model which learns nothing predicts a multiplier
of 1.0, which reproduces a plain twelve-week moving average exactly. That moving average is
the floor, called the **structural baseline** throughout. Every point of accuracy above it
is attributable to the model, which is what makes the version log interpretable.

`MODEL.md` has the architecture, the features, and the reasoning.

### The retired prototype: statsforecast

Phase one of the project was a different system: per-SKU AutoARIMA, AutoETS and moving
averages, selected by cross-validation. It was **retired on 2026-08-13**. Its code is still
in the tree under `src/legacy/`, `api/legacy/` and `scripts/legacy/`, kept as a record. Its
API router is not mounted, nothing schedules it, and nothing calls it.

It still matters for one reason: **it is the accuracy bar.** The success criterion was never
beating a moving average, it was beating this. It still wins one cell.

Retired at the same time, and worth knowing because references survive:

| Removed | Was at |
|---|---|
| The Demand Forecast page | `/planning/demand-forecast` |
| SKU Planning's Demand Forecast tab | `/planning/sku-forecasts/[sku]?tab=forecast` |
| Fourteen Next.js proxy routes | `/api/forecast/*` |

`?tab=forecast` still resolves rather than returning 404; it lands on Sales Analysis. The
per-SKU view of the served model is now at `/planning/action-list/[sku]`.

---

## 5. How accuracy is measured

### The metric

**Pooled WAPE per segment.** Sum the absolute error across SKUs, divide by summed actuals.
Lower is better. A value of 0.20 means the segment's forecasts were off by 20% of the units
it actually sold.

"Pooled" means SKUs are weighted by volume, so a big SKU counts more than a small one, which
matches what an error actually costs. Bias, the signed version, is reported alongside,
because two forecasters with the same WAPE can be wrong in opposite directions and only one
of them causes stockouts.

Accuracy is scored on the **ten-week total per SKU**, not week by week, because stock is
ordered for a horizon rather than for each week separately.

### The windows

Four rolling-origin windows. A model is always scored on data after its training cutoff,
never on what it trained on.

| Window | Cutoff | Test period | Role |
|---|---|---|---|
| Oct-Dec 2025 | 2025-10-06 | 2025-10-13 to 2025-12-15 | Development, Q4 peak |
| Dec-Feb | 2025-12-15 | 2025-12-22 to 2026-02-23 | Development, post-holiday trough |
| Mar-May 2026 | 2026-02-23 | 2026-03-02 to 2026-05-04 | Development, spring |
| **Final test** | **2026-05-04** | **2026-05-11 to 2026-07-13** | **Quarantined, used once** |

Three development windows spanning three seasonal regimes, so a change has to work across
seasons to be adopted. One window quarantined and never looked at during development.

### The decision rule

A change is adopted only if it improves accuracy **consistently in sign across all three
development windows** and by a **three-window mean of at least 0.01**. A single-window
difference under 0.02 is inconclusive: bootstrap resampling of the SKU population puts the
noise of a single-window paired difference at roughly ±0.011 to ±0.014. That floor was
measured rather than assumed, which is why the threshold is a number instead of a feeling.

Pass criteria are written into the version log **before** the experiment runs. Both
adoptions and rejections are recorded with their reasoning. Versions v0 through v18 are
logged this way and most were rejected, which is the evidence that the process is honest.

### Two pins, both required

`ML_FINAL_TEST_CUTOFF` fixes which weeks each window covers. `ML_DATA_SNAPSHOT` fixes the
values inside those weeks, because the weekly refresh revises recent actuals as late orders
register. **Advancing either re-baselines every recorded number in the project.** Do not
advance one and assume the other followed.

---

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

**Provenance, stated exactly, because these two columns have different sources and the
existing documents got this wrong.**

- The **v11 column** comes from `docs/rebaseline_2026-08-03-v2/ml_22_v11_hybrid.log`, the
  raw output of the run.
- The **V1 column** was **re-measured on 2026-08-13** after an as-of off-by-one was fixed in
  `scripts/ml_02_v1_benchmark.py`, and is recorded in `ML_FORECAST_DESIGN.md` Section 1.1.
  The `ml_02_v1_benchmark.log` file in that same rebaseline directory **predates the fix and
  is superseded.** Documents which cite it for V1, and say "the logs win", are wrong on that
  point. The fix improved V1 by 0.005 to 0.011 in two windows and cost it up to 0.017 in the
  third, so the older recorded margins over V1 were systematically overstated.

The Oct-Dec short cell is marked reference only because just 14 short SKUs were eligible at
that cutoff, too few to conclude from. It is excluded from short-segment decisions.

**Read this honestly.** v11 beats the spreadsheet in **four of six** cells, often by a wide
margin, and loses **both** Oct-Dec cells to it by 0.02 to 0.04. Any document claiming five of
six is out of date. **Why V1 is more robust in the Q4 window is not understood**, and it is
the most interesting unanswered question in the project.

### The final test

Run once, on 2026-08-13 at 12:02 PDT, commit `4a19ca1`, snapshot `2026-08-03-v2`.
Quarantined window: cutoff 2026-05-04, test weeks 2026-05-11 to 2026-07-13, **303 SKUs**.

| segment | v11 | V1 (spreadsheet) | structural baseline |
|---|---|---|---|
| smooth/short | 0.2061 | 0.3772 | 0.2013 |
| smooth/long | 0.1324 | 0.1872 | 0.1282 |
| **TOTAL** | **0.1784** | **0.3059** | **0.1739** |

**The primary criterion passed.** v11 beats V1 on both segments and on TOTAL, all
significant.

| comparison | delta | se | 95% CI | verdict |
|---|---|---|---|---|
| short, v11 vs V1 | −0.1711 | 0.0215 | [−0.2135, −0.1269] | v11 ahead, significant |
| long, v11 vs V1 | −0.0548 | 0.0174 | [−0.0879, −0.0198] | v11 ahead, significant |

On TOTAL, v11's error is **42% smaller** than the spreadsheet's. On calibration, V1 came in
28% low over this window against v11's 0.0%, which for ordering is the difference between
absorbing a shortfall by hand and not. Note that V1's bias is season-dependent rather than
always low; it ran 12.9% high on short SKUs in the post-holiday window.

**The second half of the result, which matters as much.** On this same window v11 **ties the
structural baseline**:

| comparison | delta | se | 95% CI | verdict |
|---|---|---|---|---|
| short, v11 vs baseline | +0.0048 | 0.0108 | [−0.0169, +0.0255] | indistinguishable |
| long, v11 vs baseline | +0.0042 | 0.0141 | [−0.0240, +0.0305] | indistinguishable |

Both deltas are smaller than one standard error. Reporting only the V1 win would be the
failure mode the Forecast Validation page refuses in its own code guide: a comparison that
only reports its wins is not evidence.

### Where the model actually earns its keep

The tie is the expected result, not a disappointment, and the fuller picture is the more
useful claim. Across all four windows and eight segment cells, **v11 is never significantly
worse than the baseline and is significantly better in two.** Both are seasonal turning
points:

| | structural baseline | v11 |
|---|---|---|
| Q4 ramp-up, short SKUs (WAPE) | 0.4605 | **0.2473** |
| Q4 ramp-up, short SKUs (bias) | −46.0% | **+7.5%** |
| Post-holiday, long SKUs (WAPE) | 0.2171 | **0.1389** |
| Post-holiday, long SKUs (bias) | +16.7% | **+4.3%** |

A trailing average cannot turn a corner it has not yet seen. That is the gap the model
closes, and it is concentrated rather than spread evenly across the year. The final test
window, May to July, is a flat stretch, and in flat stretches the two are the same
forecaster.

**What stays unproven is the wide middle:** whether v11 helps in ordinary quarters. On this
evidence it does not, and it does not hurt either.

---

## 7. What is honest to say about it, and what is not

Stated plainly, because naming these is a credibility marker rather than a weakness. The
full list is in `ML_FORECAST_DESIGN.md` Section 2.2 and `MODEL.md` Section 6.

- **Two years of data.** Every seasonal result rests on one or two observations of each
  season. The approach is sound; the exact numbers carry that uncertainty.
- **The four windows are not independent samples.** With two years of history their training
  sets overlap heavily. They test robustness across target seasons, not four separate draws.
- **The long model rests on about 54 SKUs**, 23 of them correlated `CC-CN-03`/`CC-CP-03`
  variants. Its effective sample is smaller than the count suggests and its confidence
  intervals overstate confidence.
- **The evaluated horizon is 10 weeks; production serves 13.** Weeks 11 to 13 of every
  served forecast are scored by no window at all, so every recorded figure is a lower bound
  on the error of the horizon actually used for ordering.
- **Eligibility uses future information.** The evaluation decides which SKUs to score partly
  from data after the cutoff. This is documented rather than fixed, and it inflates results
  by an amount that is hard to quantify. Read it before quoting a figure to someone who will
  act on it.
- **The prototype has never been run through the shared harness.** It is compared using its
  own stored evaluation output. The bar the project is judged against has been measured with
  different code from everything it is compared to. This is the single most valuable open
  item and it is first in `FUTURE_IMPROVEMENTS.md`.
- **The quarantined window was not pristine.** It was evaluated once during development, on
  2026-08-11, by `scripts/ml_34_asof_bucket_audit.py`. Nothing was tuned on it and the
  profiling has changed since, but a reader is entitled to know.
- **No prediction intervals.** v11 emits a point forecast only. The reasoning is deliberate
  and recorded in `FUTURE_IMPROVEMENTS.md`; it is a decision, not an oversight.

---

## 8. Known defects at handover

Three were found on 2026-08-14. Two are fixed; one remains and needs a command run against
the database.

**Fixed: the Forecast Validation page said the final test had not been run.** The API
hardcoded `"evaluated": False` and the page rendered "Not evaluated yet, deliberately", which
had been false since 2026-08-13. `/planning/validation` now serves
`outputs/reports/final_test.json` and the page renders both halves of the result, the win
against the spreadsheet and the tie against the structural baseline. `SCREENS.md` Section 3.6.

**Fixed: `outputs/reports/final_test.json` was not in version control.** It is now on the
`.gitignore` allowlist alongside the other four report files. It was the one artifact here
that cannot be regenerated, because the test is single-use and the runner refuses to
overwrite, so an untracked copy on one machine was one `rm` from taking the commit, the input
md5 and the bootstrap intervals with it.

**Still open: the accuracy report behind the same page is stale.**
`outputs/reports/ml_accuracy.csv` is dated 2026-07-30, so it predates the profiling fix, the
threshold alignment and the V1 as-of fix. It is tracked, so the stale version is committed
and deployed. The figures in Section 6 above are correct; the ones rendered in sections 01
and 05 of Forecast Validation, and the reliability tiers on the Action List, are not. Fixing
it is one run of `scripts/ml_accuracy_report.py` followed by a commit. `SCREENS.md`
Section 3.7.
