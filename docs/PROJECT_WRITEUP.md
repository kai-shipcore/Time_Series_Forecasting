# Demand Forecasting: Project Write-Up

A narrative account of the project: the problem, the solution, how it was validated, and
what it achieved. Written to be explained out loud to a non-technical audience or an
interviewer. It is the story; the depth lives in two companion documents, referenced
throughout so nothing is duplicated:

- **`ML_FORECAST_DESIGN.md`** — every design decision with its evidence, the metric
  definitions, and the full version log.
- **`LEARNING_NOTES.md`** — plain-language explanations of how the model works, the
  features, and the general lessons.

Cross-references are written as "→ ML_FORECAST_DESIGN.md, Section X" so they stay valid
regardless of how the file is viewed.

> Status: living document. Sections marked TODO are planned work (final validation test,
> forward forecast, UI) not yet completed.

---

## 1. Business Problem

The company orders stock against a weekly demand forecast. The forecast in production is a
spreadsheet formula that blends recent sales and applies hand-set seasonal factors. It is
systematically wrong because trailing averages cannot anticipate change, and being wrong
costs money in two directions: too low means stockouts and lost sales, too high means cash
tied up in overstock. This project replaced that method with a machine-learning model that
matches or beats it in five of six seasonal tests, and packaged it so the company can
actually use and maintain it.

---

## 2. Goal and Scope

Forecast weekly unit demand, ten weeks ahead, for the products steady enough to model. Of
roughly 3,300 active products, only about 450 sell regularly enough to forecast (the
"smooth" products); they are about 13% of the catalog but carry about 83% of total demand,
which is why they are the ones worth getting right. The other ~3,000 sell too sporadically
to forecast and are out of scope.

The smooth products are split again by how much history they have, because that changes what
is learnable:

- **New products** (~360, under ~50 weeks of sales): too little history to model
  individually; they are mostly still ramping up after launch.
- **Established products** (~80, 50+ weeks): enough history for richer per-product models;
  their demand is steadier and tends to revert to a normal level.

These two groups behave differently and, as the project found, need different handling.

Accuracy is judged the way the business actually uses the forecast: the ten-week total per
product (you order stock for the horizon, not for each week separately), with bigger
products weighted more heavily.

→ Depth: goal, metric, and segmentation in ML_FORECAST_DESIGN.md, Section 1.

## 3. Current Method

The production method ("V1") is a spreadsheet formula: it blends sales velocity over several
recent windows (7, 15, 30, 60, 90 days) and multiplies by monthly seasonal factors set by
hand. Two structural problems:

- **Trailing averages lag reality.** They are a rear-view mirror: they cannot see change
  coming, so they under-forecast products that are ramping and over-forecast ones that are
  fading.
- **Its error is large and swings by season**, so it cannot be corrected with a single
  adjustment. Measured through this project's evaluation, V1 under-forecasts the spring
  window by 30–40%, over-forecasts established products after the holidays by nearly 40%,
  and is close to accurate in the autumn ramp. (An earlier belief that V1 "always forecasts
  low" turned out to be a spring-only artifact; the measurement corrected it.)

Note there was also a **second, better method built earlier in this project** (the
"statistical prototype": per-product statistical models chosen automatically). It beats V1
but was never put into production, so V1 remains what the company actually runs. The
prototype becomes important later as the accuracy bar the new model has to clear.

→ Depth: measured V1 and prototype accuracy per season in ML_FORECAST_DESIGN.md,
Section 1.1.

## 4. Proposed Solution

A single machine-learning model (a gradient-boosted tree model, LightGBM) trained across all
products at once, that predicts a demand **multiplier** on top of each product's recent
level rather than predicting demand directly.

Three reasons for this design:

- **One model, shared learning.** Training across the whole catalog lets patterns learned on
  one product (for example, how demand ramps after launch) transfer to others. That matters
  because no single product has much history; the current per-product spreadsheet cannot
  share information this way. This is the main structural advantage of the approach.
- **Attribution.** A model that learns nothing predicts a multiplier of 1.0 and reproduces a
  plain moving average. That moving average is the floor. So every point of accuracy the
  model adds is clearly the model's doing, which makes honest comparison possible.
- **Extensibility.** A feature-based model can absorb a corrected demand signal later (for
  example, sales cleaned for stockouts and misplaced preorders) that a fixed formula cannot
  use. The architecture was chosen so those improvements can drop in when the data is ready.

**Why this specific architecture (the evidence-based part).** The design borrows from the
two largest public forecasting competitions, which point in different directions depending
on how much history you have. In the M5 competition (Walmart, 5+ years of daily data), the
winners were tree models that were simply handed the calendar and learned all seasonality
from the data, because five years give five or six examples of each holiday. In the M4
competition (100,000 mostly short series with no extra data), the winners did the opposite:
they *imposed* each series' level and seasonality as fixed structure and let the model learn
only what was left, because short series do not contain enough repeats of each season to
learn it reliably. Our data looks like M5 in problem type but like M4 in depth (two years =
two examples of each season). So we followed M4: impose the seasonal pattern as fixed
structure, and let the model learn only the demand multiplier on top. This is why the model
predicts a multiplier rather than raw demand, and why seasonality is handled separately.

→ Depth: the ratio-target design and its rationale in ML_FORECAST_DESIGN.md, Sections 1.2,
4.5, and 4.6; plain-language version in LEARNING_NOTES.md, Part A.

## 5. How the Model Works

In one breath: compute each product's recent typical level, predict a multiplier on it with
a pile of decision trees, multiply the season back in. The trees ask yes/no questions about
a handful of features (recent growth, position versus the yearly norm, forecast horizon) and
combine into a flexible predictor; each new tree corrects the errors of the ones before it.

→ Depth: how trees and boosting work, and what each feature does, in LEARNING_NOTES.md,
Parts A and F. Keep this brief when presenting; go deep only if asked.

## 6. Validation Approach

This is the part that makes the results trustworthy, and it deserves emphasis. If someone
asks "how do you know this isn't just tuned to look good," this is the answer.

- **Graded on an exam it could not study for.** The model is always scored on data from
  after its training cutoff, never on what it trained on. One recent window is *quarantined*
  and has never been looked at during development; it exists to be used exactly once, as the
  final go/no-go check.
- **Three seasons, not one.** Every result is measured across three past windows (spring,
  post-holiday, autumn ramp) so a conclusion never rests on one lucky period.
- **Real difference, not luck.** Every claimed improvement is checked with a resampling test
  (the bootstrap): it re-draws the product list at random a thousand times to see whether
  the accuracy gap holds up or was just which products happened to be included. A gap
  smaller than that random wobble does not count.
- **The bar is set high, and decided in advance.** A change is only adopted if it improves
  accuracy consistently across all three seasons by a set margin, not just on average. And
  the overall target is not merely to beat the spreadsheet, it is to match or beat the
  stronger statistical prototype, per product group, including on the final quarantined test.
  Every experiment's pass criteria were written down before running it, and both wins and
  rejections were recorded with their reasons. Roughly fourteen model versions are logged
  this way, most of them rejected, which is itself the evidence that the process is honest.

→ Depth: decision rule in ML_FORECAST_DESIGN.md, Section 1.5; success bar in Section 1.6;
splits and quarantine in Section 2; plain-language version in LEARNING_NOTES.md, Part B.

## 7. Forecasting Inputs

- **Predict a multiplier on the recent level** (Proposed Solution above).
- **Remove seasonality before modelling, add it back after**, so the model does not relearn
  the calendar for every product.
- **Two models, not one.** New products use a shared model that borrows patterns from the
  whole catalog (they have little history of their own); established products use a dedicated
  model with a feature that spots when demand is running unsustainably high. This split came
  from discovering that any change aimed at established products, inside a single shared
  model, quietly hurt the new ones.
- **Features:** forecast horizon, recent growth, recent levels, and position versus the
  yearly norm ("elevation"), the feature that fixed the hardest problem.

→ Depth: deseasonalization (Section 4.10, 4.17), the hybrid (Section 4.27), the feature list
and each feature's job in LEARNING_NOTES.md, Part A4.

## 8. Results

Measured on the three past seasons as ten-week-total accuracy (WAPE, lower is better). The
final model is "v11". The numbers to compare against are the production spreadsheet (V1) and
the statistical prototype, which is the bar the model had to clear.

| Accuracy (WAPE) | v11 (new model) | Prototype (the bar) | V1 (production) |
|---|---|---|---|
| New products, spring | 0.196 | 0.201 | 0.420 |
| New products, post-holiday | 0.200 | 0.286 | 0.302 |
| New products, autumn | 0.178 | 0.425 | 0.789 |
| Established, spring | 0.136 | 0.141 | 0.320 |
| Established, post-holiday | 0.138 | 0.274 | 0.404 |
| Established, autumn | 0.100 | 0.091 | 0.085 |

Reading it:

- **The new model matches or beats the production spreadsheet in five of six cells**, most
  by a wide margin (for example, new products in autumn: 0.18 versus 0.79).
- **It also matches or beats the stronger statistical prototype in five of six cells** — so
  it clears the harder bar, not just the easy one.
- **The one exception is established products in the autumn ramp**, where the spreadsheet's
  hand-tuned autumn factors still edge ahead (0.085 vs 0.100). This is the single remaining
  weak spot.
- **The problem that blocked every earlier version — established products over-forecast after
  the holidays — was solved.** Error in that cell fell from 0.25 to 0.14, and its
  over-forecast bias dropped from +22% to +5%. This was the hardest part of the project and
  the main technical achievement.

A note on what these numbers mean: 0.14 means the ten-week forecast is off by about 14% of
actual demand on average, weighted toward bigger products. Lower is better.

→ Depth: full six-cell table for every version, with significance tests, in
ML_FORECAST_DESIGN.md, Section 6 (v11 entry). Visuals: TODO (forecast-vs-actual and
accuracy-comparison charts for the deck / UI).

## 9. Forward Forecast

TODO. The forward forecast (predicting the coming weeks, not backtesting) needs the v11
hybrid wired into the production forecast pipeline, which currently runs the older method.
Once wired, this section shows the model's forward demand curve per product with its
confidence band.

## 10. Limitations

Stated plainly, because naming them is a credibility marker, not a weakness:

- **Two years of data.** Every seasonal result rests on one or two observations of each
  season. The approach is sound; the exact numbers carry that uncertainty and should firm up
  with a third year.
- **The established-product model is small** (~54 products, many of them close variants), so
  its accuracy estimates are less certain than the headline suggests.
- **One remaining weak cell:** established products in the autumn ramp, where the incumbent
  still wins narrowly.
- **The final validation test has not been run yet** (Section 5's quarantined window). The
  model is best on development data; it must clear that one-shot test before it can be
  proposed as a replacement.
- **Data-quality risks not yet ruled out:** whether preorders and stockouts are recorded in
  the correct weeks. Misplaced demand would distort every forecast (→ ML_FORECAST_DESIGN.md,
  Section 5.4).

## 11. Next Steps

- **Run the final validation test** on v11, then wire it into the forecast pipeline so the
  company can use it.
- **Correct the sales record for stockouts and preorders, and feed the corrected demand into
  the model.** Stockout weeks understate true demand, and preorders can record demand in the
  wrong week; both distort the training data. Fixing them improves every method at once and
  is the primary remaining lever on accuracy, most valuable where the model is weakest (new
  products). This is why the feature-based model was chosen in the first place.
- **Build a proper frontend** so the forecasts can be viewed and used day to day, rather than
  read out of scripts and tables.

→ Depth: data-quality corrections in ML_FORECAST_DESIGN.md, Section 5.3; process backlog in
Section 5.4.

---

## Appendix: what I would want to be able to explain on the spot

(Personal study checklist; mirrors the self-test in LEARNING_NOTES.md, Part "Self-test".)

1. Why predict a multiplier, and what "predicting 1.0" reproduces.
2. Pooled WAPE on ten-week totals, and why that matches how the forecast is used.
3. The quarantined-test discipline and why it is never touched during development.
4. The whole v4-to-v11 story: the segments want opposite things from a shared model, which
   is why the hybrid was the answer, and elevation was the missing signal.
5. Why the model is near the ceiling on the sales record as recorded, and why cleaning it
   (stockouts, preorders) is the next lever.
