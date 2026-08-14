# Model primer: how the forecast works, in plain language

**Audience:** anyone who needs to understand the model without a machine-learning background.
If you can explain each section here in your own words, you understand the project well
enough to maintain it, present it, or decide what to do with it.

**This is not the technical reference.** `MODEL.md` has the architecture, the features, and
the code pointers. `OVERVIEW.md` has the measured results and the evaluation protocol. This
document explains the ideas underneath both, without assuming you already know what a
decision tree is.

---

## 1. What the model predicts

We do **not** predict demand directly. For each product we first compute its recent typical
level, a 12-week average of sales. The model's only job is to predict a **multiplier**: for
N weeks ahead, will demand be 1.0x that level, 1.1x, 0.8x? The forecast is:

```
forecast = recent level  ×  predicted multiplier  ×  seasonal factor
```

Why do it this way? Because it makes the model's contribution measurable. A model that
learns *nothing* predicts a multiplier of exactly 1.0 and reproduces a plain moving average.
That moving average is the floor (called the **structural baseline**). Every point of
accuracy the model adds is clearly attributable to the model, not to the level or the
season. This is the single most important design idea to understand.

---

## 2. What the model is: trees, then boosting

The model is **LightGBM**, which is a large collection of **decision trees**.

A single decision tree is a flowchart of yes/no questions: "is recent growth above 1.2?
yes → is the forecast more than 5 weeks out? yes → predict multiplier 1.15." One tree is
weak and blocky.

**Gradient boosting** builds hundreds of trees in sequence. Each new tree looks at the
errors the previous trees left behind and tries to correct them. Add all the trees together
and you get a flexible, accurate predictor. "Boosting" means many weak models stacked;
"gradient" means each one is aimed at the leftover error of the ones before it.

The model stops adding trees when a held-out slice of products stops improving (**early
stopping**), so it does not keep fitting noise.

---

## 3. What a feature is

A **feature** is one column of numbers the trees are allowed to ask questions about. That is
the whole definition. If you do not give the model a column, it literally cannot split on it;
the information is invisible. "Adding a feature" means giving the trees a new question they
can ask. The model then learns from the data whether that question helps.

### The features in this model

All are scale-free ratios, so a big product and a small product are described on the same
scale, which is what lets one model serve the whole catalog.

| Feature | What it tells the model |
|---|---|
| **lead** | How many weeks ahead we are forecasting. Lets the model learn that demand drifts as the horizon lengthens. |
| **ramp_4_12** | 4-week average / 12-week average. Is the product accelerating? High means growing lately. **Short-history products only.** |
| **elev_long** | 4-week average / 52-week average. Is the product far above its yearly normal? High means running unsustainably hot, expect a drop. **Long-history products only.** |
| **y_last_r, lag_1_r** | The last one or two weeks relative to the 12-week level. Fine-grained recent momentum. |

`ramp` says "it went up, expect more." `elev_long` says "it went up a lot versus the whole
year, expect it to come back down." A good forecaster needs both instincts, and the trees
learn when to trust each.

---

## 4. How accuracy is measured

### Pooled WAPE

**WAPE** = total absolute error / total actual demand. We sum each product's forecast over
the 10-week horizon, compare to its actual 10-week total, and pool across products so bigger
products count more. Lower is better; 0.20 means 20% error.

Two deliberate choices: we score **10-week totals** because that is how the forecast is used
(stock is ordered for a horizon, not for each week separately), and we **weight by demand**
because a 20% miss on a big seller costs more than on a tiny one.

### The as-of rule

At any forecast point (the "cutoff"), the model may only use information that existed then.
A product that is "established" today may have been "new" a year ago, so we recompute each
product's category as of the cutoff. Using today's information to score the past is a subtle
form of cheating that inflates results.

### Three development windows plus one quarantined test

We evaluate on three past 10-week windows in different seasons (spring, post-holiday, Q4
ramp), so a result never hinges on one lucky period. One further window is **quarantined**:
it has never been looked at during development. It exists to be used **once**, at the very
end, as the final go/no-go test. The reason: if you keep checking against the same test
while tweaking, you eventually fit it by accident.

### The bootstrap

When model A beats model B by a little, is that real or just which products happened to be
in the segment? The **bootstrap** re-draws the product list at random thousands of times and
measures how much the accuracy gap wobbles. If the gap is much bigger than the wobble, it is
real. We only believe an improvement that passes this.

### Pre-registration

Before each experiment we write down what would count as success. This stops the habit of
running something, seeing a nice-looking number, and inventing a reason it matters. Both wins
and rejections are recorded. Most versions were rejected, which is the evidence that the
process is honest.

---

## 5. The key design choices

**Deseasonalize before modelling.** We divide demand by seasonal factors so the model learns
on a flattened series, then multiply the season back in at the end. This stops the model from
having to relearn "December is big" for every product. (With only two years of data, it
cannot learn that reliably anyway.)

**Segment by history length.** Products split into "short" (under ~50 weeks of sales) and
"long" (50 weeks or more). They behave differently: young products genuinely ramp,
established ones are steady and mean-reverting.

**The hybrid: two models.** Short products use one shared model that learns across the whole
catalog (they have little history, so they benefit from patterns borrowed from others). Long
products use a dedicated model with the elevation feature. This came from discovering that
any change aimed at long products, inside a single shared model, quietly damaged short
products, because all the products share the same trees. Splitting the models removes that
interference.

---

## 6. The version story (v0 to v11)

Read as a single arc, this shows the reasoning, not just the result.

**v0-v3** built the model up one idea at a time: a growth-drift correction, then trajectory
features, then making the whole path seasonally consistent. v3 was strong everywhere except
one stubborn cell: established products after the holidays, which it over-forecast.

**v4, v5, v7, v8** all tried to fix that cell by treating it as a segment-handling problem:
a segment flag, per-segment seasonal factors, per-segment training weights, fully separate
models. **Every one improved established products but damaged newer ones.** The repeated
failure was the clue: the segments genuinely want opposite things from a shared model.

**v6/v9** found the real bug. The holiday uplift was being applied to late December, when
demand had actually already fallen. Fixing the calendar window to match the real promotional
period (late Nov to mid-Dec) helped a lot.

**v10** tried hyperparameter tuning and confirmed the remaining gap was *not* a tuning
problem, ruling out a whole avenue.

**v11** solved it. The post-holiday over-forecast was the model extrapolating a ramp into a
decline, and it had no feature that could see the turn coming. The **elevation** feature is
that missing signal. Putting it in a dedicated long-product model (so it cannot leak into
short products) finally beat the moving-average floor in the cell that had blocked every
version, while leaving short products untouched.

**v12-v18** were all rejected, confirming the ceiling of what sales history can provide.

The one-sentence version: *it was never a "which group" problem and never a tuning problem;
it was a mis-specified seasonal calendar plus a missing turning-point signal.*

---

## 7. Lessons that generalise

1. **Trees interpolate well and extrapolate terribly.** A tree learns rules inside the range
   of values it saw. Feed it a feature that only ever increases (product age, a date, a
   running total) and at prediction time the value sits past everything in training; the tree
   applies its last rule and runs off a cliff.
2. **More features is not better.** Every feature is another way for the model to fit noise.
   A feature that duplicates a signal already carried by another just adds noise to an
   already-good segment.
3. **Chase the mechanism, not the metric.** The breakthroughs came from asking *why* a cell
   was wrong rather than trying random model tweaks. Diagnostics that decompose a single
   number ("where exactly is the error?") were worth more than any tuning.
4. **A model that ties a good simple baseline has learned nothing useful there.** By design,
   predicting 1.0 reproduces the moving average, so a tie means the model added no value in
   that cell; only a real, significant gain counts.
5. **Guard the final test with your life.** The most common way to fool yourself is to keep
   peeking at the same test data while tweaking. One quarantined window, used once, is the
   defence.

---

## Glossary

| Term | Meaning |
|---|---|
| **WAPE** | Weighted absolute percentage error; the accuracy metric, lower is better |
| **Pooled** | Summed across products before dividing, so big products count more |
| **Cutoff** | The "today" of a backtest; the model may use nothing after it |
| **Segment** | Short (young) vs long (established) products |
| **Structural baseline** | The moving-average floor; predicting multiplier 1.0 reproduces it |
| **Prototype** | The older statistical system (statsforecast), the accuracy bar |
| **V1** | The spreadsheet formula used before this project |
| **Deseasonalize** | Divide out seasonal factors before modelling, multiply back after |
| **Feature** | A column the trees can ask questions about |
| **Elevation** | 4-week level vs the yearly level; the feature that solved the holiday cell |
| **Bootstrap** | Resampling test to tell a real accuracy gap from luck |
| **Pre-registration** | Writing down the success criteria before running an experiment |
| **Quarantined test** | The held-out window, used once, never during development |
