# Learning Notes

A plain-language study guide to what this project is, how the model works, and why
every major choice was made. Written to be understood, not to impress. If you can explain
each part here in your own words without notes, you understand the project.

The formal record lives in `ML_FORECAST_DESIGN.md` (decisions and evidence) and
`CODEBASE_GUIDE.md` (where the code lives). This document is the teaching version.

---

## 0. The whole project in one paragraph

The company forecasts weekly demand for vehicle accessories so it can order the right amount
of stock. The old method is a spreadsheet formula (called V1) that blends recent sales and
multiplies by hand-set seasonal factors; it systematically forecasts wrong because trailing
averages cannot anticipate change. We built a machine-learning model that forecasts a
demand *multiplier* on top of each product's recent level, learns patterns from the whole
catalog at once, and can be extended with new signals later. It matches or beats the
spreadsheet in five of six seasonal tests.
<!-- SUPERSEDED 2026-08-12: four of six, and two exceptions rather than one. A profiling
     defect found on 2026-08-11 had hidden 41% of the catalogue from every evaluation;
     correcting it, and matching the two demand thresholds, moved both the population and
     the figures. V1 now wins both autumn cells. Current numbers live in
     ML_FORECAST_DESIGN.md Section 6. Rewritten here after the final test. -->
The one hard problem, established products being
over-forecast after the holidays, was solved by fixing a seasonal-calendar bug and by
adding a feature that recognises when a product is running unsustainably high. The model is
now close to the best achievable on sales data alone; further gains need new data sources.

---

## Part A. How the model actually works

### A1. The core trick: predict a multiplier, not demand

We do **not** predict demand directly. For each product we first compute its recent typical
level, a 12-week average of sales. The model's only job is to predict a **multiplier**: for
N weeks ahead, will demand be 1.0x that level, 1.1x, 0.8x? The forecast is:

    forecast = recent level  x  predicted multiplier  x  seasonal factor

Why do it this way? Because it makes the model's contribution measurable. A model that
learns *nothing* predicts a multiplier of exactly 1.0 and reproduces a plain moving average.
That moving average is our floor ("v-base"). So every point of accuracy the model adds is
clearly attributable to the model, not to the level or the season. This is the single most
important design idea to understand.

### A2. What the model is: trees, then boosting

The model is **LightGBM**, which is a large collection of **decision trees**.

A single decision tree is a flowchart of yes/no questions: "is recent growth above 1.2? yes
-> is the forecast more than 5 weeks out? yes -> predict multiplier 1.15." One tree is weak
and blocky.

**Gradient boosting** builds hundreds of trees in sequence. Each new tree looks at the
errors the previous trees left behind and tries to correct them. Add all the trees together
and you get a flexible, accurate predictor. "Boosting" = many weak models stacked;
"gradient" = each one is aimed at fixing the leftover error of the ones before it.

The model stops adding trees when a held-out slice of products stops improving ("early
stopping"), so it does not just keep fitting noise.

### A3. What a feature actually is

A **feature** is one column of numbers the trees are *allowed to ask questions about*.
That is the whole definition. If you do not give the model a column, it literally cannot
split on it; the information is invisible to it. "Adding a feature" means "giving the trees
a new question they can ask." The model then learns from the data whether that question
helps tell high-multiplier situations apart from low ones.

### A4. Our features, each in one line

All are scale-free (ratios), so a big product and a small product are described on the same
scale, which is what lets one model serve the whole catalog.

- **lead** — how many weeks ahead we are forecasting. Lets the model learn that demand
  tends to drift as the horizon lengthens.
- **ramp_4_12** (4-week average / 12-week average) — is the product accelerating recently?
  High means "growing lately."
- **elev_long** (4-week average / 52-week average) — is the product far above its *yearly*
  normal? High means "this is unusually high and probably not sustainable, expect a drop."
  This is the opposite instinct to ramp, which is exactly why both are useful.
- **y_last_r, lag_1_r** — the last one or two weeks relative to the 12-week level.
  Fine-grained recent momentum.

`ramp` says "it went up, expect more." `elev_long` says "it went up a lot versus the whole
year, expect it to come back down." A good forecaster needs both instincts, and the trees
learn when to trust each.

---

## Part B. How we measure, and how we avoid fooling ourselves

This is the part that makes the work trustworthy. It matters more than any single model.

### B1. The accuracy metric: pooled WAPE

**WAPE** = total absolute error / total actual demand. We sum each product's forecast over
the 10-week horizon, compare to its actual 10-week total, and pool across products so
bigger-demand products count more. Lower is better; 0.20 means 20% error.

Two deliberate choices: we score **10-week totals** because that is how the forecast is
used (you order stock for the horizon, not for each week separately), and we **weight by
demand** because a 20% miss on a big seller costs more than on a tiny one.

### B2. The as-of rule: never use the future

At any forecast point ("cutoff"), the model may only use information that existed then.
This sounds obvious but is easy to break: a product that is "established" *today* may have
been "new" a year ago, so we recompute each product's category *as of the cutoff*, not from
today's labels. Using today's information to score the past is a subtle form of cheating
that inflates results; the harness guards against it automatically.

### B3. The split protocol: three dev seasons + one quarantined test

We evaluate on three past 10-week windows in different seasons (spring, post-holiday, Q4
ramp), so a result never hinges on one lucky period. One further window, the most recent, is
**quarantined**: it has never been looked at during development. It exists to be used
**once**, at the very end, as the final go/no-go test. The reason: if you keep checking
against the same test while tweaking, you eventually fit it by accident. The quarantined
window is the exam the model cannot study for.

### B4. The bootstrap: is a difference real or luck?

When model A beats model B by a little, is that real or just which products happened to be
in the segment? The **bootstrap** re-draws the product list at random thousands of times and
measures how much the accuracy gap wobbles. If the gap is much bigger than the wobble, it is
real; if not, it is noise. We only believe an improvement that passes this.

### B5. Pre-registration: decide the pass criteria before running

Before each experiment we write down what would count as success. This stops the very human
habit of running something, seeing a nice-looking number, and inventing a reason it matters.
Both wins and rejections get recorded with their reasons in the Decision Log.

---

## Part C. The main design choices, and why

- **Deseasonalize before modelling.** We divide demand by seasonal factors so the model
  learns on a flattened series, then multiply the season back in at the end. This stops the
  model from having to relearn "December is big" for every product.
- **Segment by history length.** Products split into "short" (young, under ~50 weeks) and
  "long" (established). They behave differently: young products genuinely ramp, established
  ones are steady and mean-reverting. Treating them the same forces a compromise that serves
  neither well.
- **The hybrid: two models.** Short products use one shared model that learns across the
  whole catalog (they have little history, so they benefit from patterns borrowed from
  others). Long products use a dedicated model with the elevation feature. This came from
  discovering that any change aimed at long products, inside a single shared model, quietly
  damaged short products, because all the products share the same trees. Splitting the models
  removes that interference.
- **Separate seasonal settings for the ML track.** The ML model is measured against an older
  statistical system that shares the same seasonal code. We gave the ML track its own copy so
  that improving the model cannot accidentally move the yardstick we measure it against.

---

## Part D. The version story (v0 to v11) as one narrative

Read as a single arc, this is the most impressive part to explain, because it shows the
reasoning, not just the result.

- **v0-v3** built the model up one idea at a time: a growth-drift correction, then
  trajectory features, then making the whole path seasonally consistent. v3 was strong
  everywhere except one stubborn cell: established products after the holidays, which it
  over-forecast.
- **v4, v5, v7, v8** all tried to fix that cell by treating it as a "the two product groups
  need different handling" problem: a segment flag, per-segment seasonal factors, per-segment
  training weights, fully separate models. **Every one improved established products but
  damaged newer ones.** The repeated failure was the clue: the segments genuinely want
  opposite things from a shared model.
- **v6/v9** found the real bug. The holiday uplift was being applied to late December, when
  demand had actually already fallen, and the calendar window even drifted year to year.
  Fixing it to match the real promotional period (late Nov to mid-Dec) helped a lot.
- **v10** tried hyperparameter tuning and confirmed the remaining gap was *not* a tuning
  problem, which ruled out a whole avenue.
- **v11** solved it. The post-holiday over-forecast was the model extrapolating a ramp into a
  decline, and it had no feature that could see the turn coming. The **elevation** feature is
  that missing signal. Putting it in a dedicated long-product model (so it cannot leak into
  short products) finally beat the moving-average floor in the cell that had blocked every
  version, while leaving short products untouched.
- **v12 (age) and v13 (acceleration)** were then rejected, confirming we had reached the
  ceiling of what sales history can provide.

The one-sentence version: *it was never a "which group" problem and never a tuning problem;
it was a mis-specified seasonal calendar plus a missing turning-point signal.*

---

## Part E. The lessons that generalise beyond this project

1. **Trees interpolate well and extrapolate terribly.** A tree learns rules inside the range
   of values it saw. Feed it a feature that only ever increases (product age, a date, a
   running total) and at prediction time the value sits past everything in training; the tree
   applies its last rule and runs off a cliff. This is exactly why the age feature blew up.
2. **More features is not better.** Every feature is another way for the model to fit noise.
   Acceleration failed for young products because the ramp feature already carried the signal
   and the extra column just added noise to an already-good segment.
3. **Chase the mechanism, not the metric.** The breakthroughs came from asking *why* a cell
   was wrong (a ramp extrapolated into a decline) rather than trying random model tweaks.
   Diagnostics that open up a single number ("where exactly is the error?") were worth more
   than any tuning.
4. **A model that ties a good simple baseline has learned nothing useful there.** By design,
   predicting 1.0 reproduces the moving average, so a tie means the model added no value in
   that cell; only a real, significant gain counts.
5. **Guard the final test with your life.** The single most common way to fool yourself is to
   keep peeking at the same test data while tweaking. One quarantined window, used once, is
   the defence.

---

## Part F. How the diagnostic correlation numbers are computed

When checking whether a candidate feature is worth building (for example "does elevation
predict the coming decline?"), the quick screen is a **Spearman rank correlation**:

1. For each product, take the feature value at the cutoff (e.g. elevation = 4-week / 52-week)
   and what actually happened next (future demand / current level).
2. Rank the products on each of the two quantities.
3. Correlate the ranks.

We rank first (Spearman) rather than correlate raw values (Pearson) because demand has
extreme outliers; a couple of huge products would otherwise dominate. Ranking makes the
biggest product just "rank 1," not "1000x everything." A value near -0.5 or +0.5 is a strong,
reliable relationship; near 0 is none. Negative means "high feature goes with lower future
demand," which for elevation is the signal we want ("running high now -> falls later"). This
is only a screen to decide whether to build the feature; the real test is always a full
experiment scored by pooled WAPE with a bootstrap.

---

## Glossary

- **WAPE** — weighted absolute percentage error; our accuracy metric, lower is better.
- **Pooled** — summed across products before dividing, so big products count more.
- **Cutoff** — the "today" of a backtest; the model may use nothing after it.
- **Segment** — short (young) vs long (established) products.
- **v-base / baseline** — the moving-average floor; predicting multiplier 1.0 reproduces it.
- **Prototype** — the older statistical system, the accuracy bar the ML model must clear.
- **V1** — the spreadsheet formula currently used in production.
- **Deseasonalize** — divide out seasonal factors before modelling, multiply back after.
- **Feature** — a column the trees can ask questions about.
- **Elevation** — 4-week level vs the yearly level; the feature that solved the holiday cell.
- **Bootstrap** — resampling test to tell a real accuracy gap from luck.
- **Pre-registration** — writing down the success criteria before running an experiment.
- **Quarantined test** — the held-out window, used once, never during development.

---

## Self-test: can you explain each of these without notes?

1. Why we predict a multiplier instead of demand, and what "predicting 1.0" reproduces.
2. What a decision tree is, and what "boosting" adds.
3. What a feature is, and what `ramp_4_12` vs `elev_long` each tell the model.
4. Why pooled WAPE on 10-week totals matches how the forecast is used.
5. The as-of rule, and one concrete way you could break it by accident.
6. Why there are three dev windows and one quarantined test you have never looked at.
7. What the bootstrap protects against.
8. The whole v4-to-v11 story in three sentences.
9. Why the age feature blew up (extrapolation) and why acceleration was rejected (noise).
10. Why this is near the ceiling on sales data, and what the next lever is.
