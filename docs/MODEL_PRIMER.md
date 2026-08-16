# Model primer

Conceptual introduction to the forecast model for readers without a machine-learning background.

For the technical reference see `MODEL.md`. For measured results and the evaluation protocol see `OVERVIEW.md`.

## 1. Prediction target

The model does not predict demand. It predicts a **multiplier** on each product's recent level.

```
forecast = recent level  ×  predicted multiplier  ×  seasonal factor
```

The recent level is a 12-week average of sales. The multiplier answers: for N weeks ahead, will demand be 1.0x that level, 1.1x, 0.8x?

**Rationale.** A model that learns nothing predicts 1.0 and reproduces a plain moving average. That moving average is the floor, called the **structural baseline**. Accuracy above the floor is attributable to the model rather than to the level or the season. This is the most important design idea.

## 2. Algorithm

LightGBM, a gradient-boosted collection of decision trees.

| Concept | Definition |
|---|---|
| Decision tree | A flowchart of yes/no questions. "Is recent growth above 1.2? yes → is the forecast more than 5 weeks out? yes → predict 1.15." A single tree is weak |
| Gradient boosting | Hundreds of trees built in sequence, each correcting the errors left by the previous ones. "Boosting" means weak models stacked; "gradient" means each targets the leftover error |
| Early stopping | Tree-building halts when a held-out slice of products stops improving, which prevents fitting noise |

## 3. Features

A feature is one column the trees may split on. A column the model is not given is invisible to it.

All features are scale-free ratios, so large and small products are described on the same scale. This is what allows one model to serve the whole catalogue.

| Feature | Definition | Signal | Used by |
|---|---|---|---|
| `lead` | 1 to 13 | Weeks ahead. Lets demand drift with horizon length | Both models |
| `ramp_4_12` | 4-week mean / 12-week mean | Product is accelerating | Short only |
| `y_last_r` | Anchor week / 12-week mean | Recent position | Both models |
| `lag_1_r` | Week before anchor / 12-week mean | Short-run direction, with `y_last_r` | Both models |
| `elev_long` | 4-week mean / 52-week mean | Product is far above its yearly normal, so expect reversion | Long only |

Four features per model, five distinct. `ramp_4_12` reads "it went up, expect more"; `elev_long` reads "it went up against the whole year, expect reversion". Additional lag features are a candidate in `FUTURE_IMPROVEMENTS.md` §2, not currently used.

## 4. Accuracy measurement

Protocol and results: `OVERVIEW.md` §5 and §6.

| Concept | Definition |
|---|---|
| Pooled WAPE | Total absolute error / total actual demand. Lower is better; 0.20 means 20% error |
| 10-week totals | The scoring unit, because stock is ordered for a horizon, not for each week separately |
| Demand weighting | Larger products count more, because a 20% miss on a large seller costs more |
| As-of rule | At any cutoff the model may use only information that existed then. Each product's category is recomputed as of the cutoff |
| Development windows | Three past 10-week windows in different seasons, so no result hinges on one period |
| Quarantined window | A fourth window, unexamined during development, used once as the final test |
| Bootstrap | The product list is re-drawn at random thousands of times to measure how much an accuracy gap moves. A gap much larger than that movement is real |
| Pre-registration | Success criteria are recorded before each experiment. Wins and rejections are both logged |

## 5. Design decisions

**Deseasonalise before modelling.** Demand is divided by seasonal factors, the model fits the flattened series, and the season is multiplied back at the end. This removes the need to relearn "December is big" per product, which two years of data cannot support.

**Segment by history length.** Short is under about 50 weeks of sales, long is 50 weeks or more. Young products ramp; established products are steady and mean-reverting.

**Two models, not one.** Short products use a model trained across the whole catalogue, benefiting from borrowed patterns. Long products use a dedicated model carrying `elev_long`.

Warning: any change aimed at long products inside a single shared model damaged short products, because all products share the same trees. Do not re-merge the models.

## 6. Version history

| Versions | Outcome |
|---|---|
| v0 to v3 | Built up incrementally: growth-drift correction, trajectory features, seasonal consistency. Strong except on established products after the holidays, which were over-forecast |
| v4, v5, v7, v8 | Treated that cell as a segment-handling problem: segment flag, per-segment seasonal factors, per-segment training weights, separate models. Each improved established products and damaged newer ones |
| v6, v9 | Located the defect. The holiday uplift was applied to late December, after demand had fallen. Correcting the window to late November through mid December gave a substantial gain |
| v10 | Hyperparameter tuning produced no gain, ruling out configuration as the cause |
| v11 | The post-holiday over-forecast was ramp extrapolation into a decline, with no feature able to detect the turn. `elev_long` supplies that signal. Placed in a dedicated long model so it cannot affect short products |
| v12 to v18 | All rejected, confirming the ceiling of what sales history alone provides |

Summary: a mis-specified seasonal calendar plus a missing turning-point signal, not a segment-assignment or tuning problem.

## Glossary

| Term | Meaning |
|---|---|
| WAPE | Weighted absolute percentage error; the accuracy metric, lower is better |
| Pooled | Summed across products before dividing, so large products count more |
| Cutoff | The "today" of a backtest; the model may use nothing after it |
| Segment | Short (young) vs long (established) products |
| Structural baseline | The moving-average floor; predicting multiplier 1.0 reproduces it |
| Prototype | The retired statistical system (statsforecast), the accuracy bar |
| V1 | The spreadsheet formula used before this project |
| Deseasonalise | Divide out seasonal factors before modelling, multiply back after |
| Feature | A column the trees can ask questions about |
| Elevation | 4-week level against the yearly level; the feature that solved the holiday cell |
| Bootstrap | Resampling test distinguishing a real accuracy gap from chance |
| Pre-registration | Writing down success criteria before running an experiment |
| Quarantined test | The held-out window, used once, never during development |
