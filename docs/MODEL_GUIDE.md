# Model guide: technical handover

**Audience:** whoever picks up the modelling. Assumes pandas, cross-validation and
gradient-boosted trees. It does not re-teach those.

**What this is.** An orientation document. It tells you what the model is, why it has the
shape it has, how to run an experiment, what has already been ruled out, and what is worth
trying next. It is deliberately short, because the exhaustive record already exists and
duplicating it would create a second source that can disagree with the first.

| For | Read |
|---|---|
| Every decision with its evidence, and every rejection | `ML_FORECAST_DESIGN.md` sections 4 and 6 |
| What each file does and how data flows | `CODEBASE_GUIDE.md` |
| Running the thing in production | `OPERATIONS.md` |
| Caveats and traps, written by the person who hit them | `HANDOVER.md` |
| Work identified but not done | `BACKLOG.md` |

**The one rule that governs this project.** One hypothesis at a time, evaluated on the three
development windows through `src/ml/evaluate.py`, judged by the decision rule in design doc
section 1.5, and recorded in the decision log whether it was adopted or rejected. Pass
criteria are written down **before** the experiment runs. If you keep nothing else, keep
this: most of the value in the record is the list of things that did not work.

---

## 1. The problem in modelling terms

Forecast weekly unit demand per SKU, out to thirteen weeks, for roughly 450 SKUs that sell
regularly. Roughly two years of history. No covariates worth speaking of beyond the sales
record itself.

Two facts shape everything else:

1. **Two years means two observations of each annual seasonal event.** That is not enough to
   learn seasonality from data.
2. **Most SKUs are young.** The large majority of forecastable SKUs have under 50 weeks of
   history, so per-SKU model fitting has very little to work with.

Fact 1 rules out the M5-competition approach, where LightGBM is handed calendar features and
learns seasonality itself. That was tried here and failed exactly as the literature predicts:
it reproduced the specific months it had seen rather than a general pattern, and
over-forecast the post-holiday trough by +123% (design doc section 4.9).

Fact 2 is the argument for a **global** model. A single model trained across all SKUs jointly
lets a pattern learned on one SKU, such as how demand behaves after launch, transfer to
another. That cross-SKU transfer is the main structural advantage this approach has over the
per-SKU statistical prototype, and it is measurable: a short-only model loses it and gets
worse (v8).

---

## 2. Architecture: structure imposed, residual learned

The design follows M4 rather than M5. Scale and seasonality are **imposed structurally**;
LightGBM is confined to learning the residual dynamics the data can actually support.

**The target is a ratio, not a level.** For each SKU and target week, the model predicts

```
y_target / (trailing 12-week average of that SKU's history at the cutoff)
```

This is the single most important design choice to understand, and it has a property worth
appreciating: a model that learns nothing predicts a ratio of 1.0, which reproduces the
structural baseline exactly. Every measured movement is therefore attributable to the
feature under test, which is what makes the version log interpretable.

**Seasonality is a round-trip, applied per segment.** For long SKUs, history is divided by
hand-set monthly multipliers before fitting and forecasts are multiplied back afterwards.
For short SKUs, no seasonal adjustment is applied. That split is a measured result, not an
assumption (design doc sections 4.10 and 4.17).

**Multi-horizon is handled by direct forecasting with `lead` as a feature**, rather than
recursion. One model covers all thirteen horizons and the horizon is an input.

### The current model, v11: a hybrid

Two models, because a single shared model could not serve both segments.

| Segment | Model | Features |
|---|---|---|
| smooth/short | shared model, trained on **all** smooth SKUs | `lead`, `ramp_4_12`, `y_last_r`, `lag_1_r` |
| smooth/long | dedicated model, trained on **long SKUs only** | `lead`, `y_last_r`, `lag_1_r`, `elev_long` |

The long model drops `ramp_4_12`, the 4-week/12-week growth feature that drives a persistent
over-forecast, and adds `elev_long`, the 4-week/52-week elevation against the SKU's own
annual norm. The recent-level ratios stay, so gradual growth is still tracked; only sharp-ramp
extrapolation is removed and replaced with mean-reversion pressure.

**Why split the models rather than add a segment indicator?** Because that was tried, twice.
A segment indicator (v4) and per-segment sample weighting (v7) both failed for the same
reason: in a shared model the trees are shared, so a change made for long SKUs damages short
SKUs through them. Splitting the models removes the shared trees. Short SKUs keep the shared
model because they need the cross-segment transfer; long SKUs do not.

Hyperparameters are near-default (`learning_rate=0.05`, `num_leaves=31`,
`min_child_samples=200`, early stopping against an internal validation slice). Tuning was
tried twice, as v10 and again as v18 on the hybrid architecture, and rejected both times.
The model is not misconfigured; that is not where the remaining error is.

---

## 3. Evaluation, and why it is arranged this way

**Metric: pooled WAPE per segment.** Sum absolute error across SKUs, divide by summed
actuals. SKUs are weighted by volume, which matches the commercial cost of error. Bias and a
ramp-cohort breakdown are reported alongside as secondary metrics.

**Four rolling-origin windows**, three for development and one quarantined:

| Window | Cutoff | Test period | Role |
|---|---|---|---|
| Final test | 2026-05-04 | 2026-05-11 to 2026-07-13 | Quarantined, single use |
| Mar-May 2026 | 2026-02-23 | 2026-03-02 to 2026-05-04 | Development, spring |
| Dec-Feb | 2025-12-15 | 2025-12-22 to 2026-02-23 | Development, post-holiday trough |
| Oct-Dec 2025 | 2025-10-06 | 2025-10-13 to 2025-12-15 | Development, Q4 peak |

Three windows spanning three seasonal regimes, so a change has to work across seasons to be
adopted. **Adoption requires a consistent sign across all three windows and a three-window
mean improvement of at least 0.01.** A single-window difference under 0.02 is inconclusive:
bootstrap resampling of the SKU population puts the sampling noise of a single-window paired
difference at roughly ±0.011 to ±0.014, so smaller differences are indistinguishable from
chance. The noise floor was measured rather than assumed, which is why the threshold is a
number and not a feeling.

**Two independent pins are required for any result to reproduce.** `ML_FINAL_TEST_CUTOFF`
fixes which weeks each window covers. `ML_DATA_SNAPSHOT` fixes the values inside those
weeks, because the weekly refresh revises recent actuals as late orders register. Advancing
either re-baselines every recorded number. Do not advance one and assume the other followed.

**Known limitations of the evaluation, stated because they bound what the numbers mean:**

- With two years of history the four training sets overlap heavily. These are not four
  independent samples; they test robustness across target seasons.
- Short-history SKUs are recent by definition, so few were eligible at older cutoffs. The
  Oct-Dec window has only 14 eligible short SKUs and is excluded from short-segment
  decisions entirely.
- The long model rests on about 54 SKUs, 23 of them correlated `CC-CN-03`/`CC-CP-03`
  variants. Its effective sample is smaller than the count suggests and its intervals
  overstate confidence.

---

## 4. Where it stands

Pooled WAPE on snapshot `2026-08-03-v2`. Lower is better.

| Segment and window | v11 | prototype | V1 (spreadsheet) |
|---|---|---|---|
| short, Mar-May | 0.1926 | 0.2028 | 0.3351 |
| short, Dec-Feb | 0.1994 | 0.2904 | 0.2240 |
| short, Oct-Dec (reference) | 0.2473 | 0.4137 | 0.2210 |
| long, Mar-May | 0.1350 | 0.1437 | 0.2776 |
| long, Dec-Feb | 0.1389 | 0.2690 | 0.3928 |
| long, Oct-Dec | 0.1040 | 0.0918 | 0.0851 |

**Source of these figures.** `docs/rebaseline_2026-08-03-v2/ml_22_v11_hybrid.log` and
`ml_02_v1_benchmark.log`, which are the raw output of the runs, rather than a table
transcribed from them. The design doc's own v11 table had five of six prototype cells
transcribed wrong until 2026-08-13; if a number here ever disagrees with a table
elsewhere, the logs win.

**Four of six against the legacy spreadsheet**, losing both Oct-Dec cells by 0.02 to 0.04.
The headline result is Dec-Feb long: 0.1389 against a moving-average baseline's 0.2171 and
the prototype's 0.2685. That cell had blocked every version since v1 and is where a plain
moving average still beat the model as recently as v9.

**Two open questions in that table, both worth someone's attention:**

1. **Why V1 wins the entire Q4 window is not understood.** A trailing velocity method being
   more robust in the peak season is not explained by anything in the record. This is the
   most interesting unanswered question in the project.
2. **Mar-May long is level, not ahead.** The margin against the baseline narrowed when a
   profiling defect was fixed (see below), and this cell is now a tie.

**The prototype has still not been run through this harness.** It is compared using its own
stored evaluation output, not the shared scorer. That is a real gap: the success bar is the
prototype, and it has never been measured with the same code as everything else. It is item
1 in the design doc's process backlog and it is the first thing worth closing.

---

## 5. Traps, from the ones that were actually hit

Full list in `HANDOVER.md`. The four most expensive:

**The evaluation was silently excluding 41% of the forecastable catalogue.** A promotion
override in the profiling meant a large share of smooth SKUs never appeared in any
evaluation window. Every figure recorded before 2026-08-11 describes a different population
from every figure after it. When you read an old number, check which snapshot it was
measured on.

**A recorded baseline figure was stale from v9 onward** and nobody noticed, because nothing
re-derived it. If a number is quoted in prose rather than computed, assume it can rot.

**The evaluation selects its population using future information.** Eligibility is decided
partly by data after the cutoff. This is documented, not fixed, and it inflates results in a
way that is hard to quantify. Read `HANDOVER.md` finding 6 before quoting any figure to
someone who will act on it.

**Promoted SKUs cannot be validated by backtest, only forward.** A SKU promoted into the
smooth set has, by construction, a history that would not have qualified it at an earlier
cutoff. Backtesting it answers a question about a SKU that did not exist in that form.

---

## 6. What to try next

In the order the evidence supports.

**1. Run the prototype through the shared harness.** Closes the gap in section 4. Until this
is done, "the model beats the prototype" rests on comparing two different scoring codebases.

**2. The data-quality corrections, which are the largest available gain.** Both fix weeks
where recorded units do not reflect true demand, so both improve the training target for
every method including the prototype:

- **Stockout correction.** Recorded sales understate demand during a stockout, so every such
  week trains the model on an artificially low number. Needs per-SKU stockout dates, which
  were promised and have not arrived.
- **Preorder correction.** A preorder books demand when the order is placed but ships weeks
  later, so the series has an artificial spike at order time and a gap at fulfilment. This is
  most damaging for newly launched SKUs, whose launch preorders can dominate their short
  history. Preorders **are** flagged as `order_type` in the source table, so excluding or
  down-weighting them can be tested today. Attributing demand to the fulfilment week still
  needs a source recording intended fulfilment dates.

**3. Feature candidates not yet tried**, from design doc section 5.2: recent-level ratios at
other lags, volatility, zero-recency, product-type attributes parsed from SKU codes, and
empirically re-estimated seasonal multipliers shrunk toward the hand-set values.

**Do not retry these without new evidence.** Each was tested and rejected, with the reasoning
recorded: calendar features for seasonality (4.9), segment indicator (4.23), per-segment
weighting (4.24), hyperparameter tuning (4.26, and again as v18), raw SKU age (4.28,
monotonic features extrapolate badly at the prediction boundary), acceleration (v13), FBA
channel share (v17). Six perturbations of the long model each cost it 0.005 to 0.012.

---

## 7. Running an experiment

```bash
.venv/bin/pip install -r requirements.txt     # exact pins; results compare at the 3rd decimal
.venv/bin/python scripts/ml_XX_your_experiment.py
```

Experiments live in `scripts/ml_*.py`, import from `src/ml/`, and add nothing to it, so the
library stays reusable. Each one runs its candidate against the three development windows
and prints a per-segment table.

The workflow:

1. Write the hypothesis down, and the pass criteria, **in the version log, before running.**
2. Run on the three development windows only. Never the final test window.
3. Show the raw per-segment output before summarising it.
4. Record the outcome in the decision log, including rejection.

Model versions are tagged commits (`model/v-base` onward), each holding that version's tree
with its results in the commit message. Check one out to re-run it.

**The final test window is quarantined.** `scripts/ml_41_final_test.py` is the only script
that touches it. It refuses to run against a dirty tree, refuses to overwrite an existing
result, and checks its preconditions first, because a run against a stale snapshot or a
mismatched LightGBM does not produce a weaker result, it produces one that cannot be
interpreted at all, and the window is spent either way.

---

## 8. The other forecaster in this repository

`src/legacy/`, `api/legacy/` and `scripts/legacy/` hold the statsforecast prototype:
per-SKU AutoARIMA, AutoETS and moving averages selected by cross-validation. It was
retired on 2026-08-13. Nothing runs it and its API router is not mounted.

For modelling purposes it matters for one reason: **it is the accuracy bar.** The success
criterion was never beating a moving average, it was beating this, per segment. It still
wins one cell, smooth/long in Oct-Dec, at 0.0918 against 0.1040, and why is not understood.

It has also never been run through the shared harness, which is the gap named in section 4.
Its figures come from its own stored evaluation output. If you close one thing from this
document, close that: the bar the project is judged against has been measured with
different code from everything it is compared to.
