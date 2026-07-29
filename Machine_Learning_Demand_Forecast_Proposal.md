# New Proposed Machine Learning Demand Forecast Model

Prepared by Yuchan  |  July 2026

------------------------------------------------------------------------

# Summary

## Purpose

The company forecasts weekly demand for every SKU to plan how much stock to order. Today that runs on an in-house spreadsheet formula, called the **current method** here. This project replaces it with a more accurate model for the SKUs where accuracy matters most, learning from the company's own sales history.

Two terms, used throughout:

- **Demand**: units of a SKU that customers actually ordered in a given week.
- **Forecast**: a prediction of that weekly demand for a week that has not happened yet, in the same units.

## Results

**The new model cuts forecast error roughly in half.** Tested across two recent periods (Dec 2025 to May 2026, about 20 weeks, ~66,800 units of demand):

- Current method: **~24,300 units** off.
- New model: **~11,700 units** off, about half as much.
- That is **~12,600 fewer units** of forecast error.

**What that is worth.** Those **~12,600 fewer units** of forecast error are each either stock ordered that does not sell (cash tied up) or demand that goes unmet (a lost sale). At an example $200 average selling price, that is a potential **$2.5M** in reduced inventory exposure over these 20 weeks. It is an upper bound, since the real value depends on the company's own margins and holding costs, but the direction is clear: less cash tied up in overstock and fewer sales lost to stockouts.

**Where the gains come from:**

- Long-history SKUs, Dec-Feb 2026: error falls from **40%** to **14%**, correcting the current method's post-holiday over-forecast.
- Short-history SKUs, Mar-May 2026: error falls from **42%** to **20%**, correcting its spring under-forecast.
- Long-history SKUs, Oct-Dec 2025: the current method is nominally ahead, **8%** vs **10%** error.

That 2-point gap is within measurement noise rather than a systematic weakness, and it is the only window where the current method comes out ahead. Everywhere else the new model is far more accurate.

**Forecast error and bias, by group and period**

| SKU group and period | Current off by | New off by | Current bias | New bias |
| --- | --- | --- | --- | --- |
| Long history, Mar-May 2026 | 4,169 (32%) | **1,768 (14%)** | -30.5% | -7.2% |
| Long history, Dec-Feb 2026 | 4,590 (40%) | **1,566 (14%)** | +38.7% | +5.0% |
| Long history, Oct-Dec 2025 | **1,536 (8%)** | 1,813 (10%) | -1.0% | +1.7% |
| Short history, Mar-May 2026 | 9,857 (42%) | **4,605 (20%)** | -39.8% | +9.4% |
| Short history, Dec-Feb 2026 | 5,704 (30%) | **3,782 (20%)** | +5.0% | -1.5% |

*"Off by" is total error, what has to be stocked for; bias is the net over- or under-forecast.*

*Why "within noise": a difference this small is within the normal variation between one set of SKUs and another. Re-tested on resampled SKUs, it does not hold up.*

Week by week (actual demand solid, both forecasts dashed): the new model (blue) tracks demand closely but smoothly, while the current method (red) is a near-flat rate that drifts. Each dashed vertical line is a fresh 10-week forecast.

**Long-history SKUs: weekly demand vs the two forecasts (Oct 2025 to May 2026)**

![](outputs/reports/management_chart_established.png)

**Short-history SKUs: weekly demand vs the two forecasts (Dec 2025 to May 2026)**

![](outputs/reports/management_chart_newer.png)

*The short-history chart starts in December 2025; too few short-history SKUs existed before then to compare.*

## Which SKUs the forecast covers

The company sells about 3,400 SKUs, but only ~450 sell regularly enough to forecast. Those are a small slice of the catalogue yet about **80% of all units sold**, so that is where accuracy matters most. The project forecasts them and leaves the rest out.

Each SKU is sorted on two questions:

- **How regular are its sales?**
    - **Smooth**: sells in most weeks (under about 30% of weeks with no sales). Forecastable.
    - **Intermittent**: sporadic, 30% or more of weeks with no sales. Not forecastable.
- **How much history does it have?**
    - **Short history**: under 50 weeks of sales, roughly under a year.
    - **Long history**: 50 weeks or more.

That gives **three demand groups**:

- **Long history (smooth)**, about 80 SKUs: its own model, tuned for settled demand that returns to a normal level.
- **Short history (smooth)**, about 370 SKUs: a separate but similar model, tuned for newer SKUs still finding their level.
- **Intermittent**, about 3,000 SKUs: too sporadic to predict, so no forecast, only sales history.

**Demand breakdown by group (last 90 days)**

| SKU group | SKUs | Demand, last 90 days | Share | Avg units / SKU |
| --- | --- | --- | --- | --- |
| Long history (smooth) | 81 | 21,745 | 31% | 268 |
| Short history (smooth) | 366 | 34,816 | 49% | 95 |
| Intermittent | 3,002 | 14,452 | 20% | 5 |
| **All SKUs** | **3,449** | **71,013** | **100%** | **21** |

The two smooth groups are about one in eight of the catalogue but roughly 80% of demand, and short-history SKUs carry the largest single share, so forecasting them well matters as much as the long-history SKUs.

![](outputs/reports/demand_breakdown_donut.png)

## How performance is measured

**The metric: WAPE, error as a share of demand.** The measure is WAPE (weighted absolute percentage error): the total forecast miss across all SKUs in a group, divided by the group's total actual sales. A result of 20% means the forecasts were off by 20% of demand. Lower is better.

Example. Say a group has two SKUs in one period:

- A high-volume SKU sells **200 units**; the forecast said 180, so it is **off by 20**.
- A low-volume SKU sells **20 units**; the forecast said 30, so it is **off by 10**.

Total miss is 30 units on 220 sold, so WAPE is **about 14%**. A plain average of the two percentage misses would read 30% (10% and 50% split evenly), but that lets a 10-unit miss on a tiny SKU count as much as a miss on a SKU that actually matters for stocking. Weighting by demand keeps the number tied to the units and dollars that matter. Alongside it we track **bias**: whether the forecast runs net high (overstock risk) or net low (lost-sales risk).

**How the accuracy is proven: testing on the past.** Every figure here comes from backtesting, which grades the model the way it would be used rather than on data it has already seen:

- **Train only on the past.** The model sees sales up to a chosen past date and nothing after, so it cannot see the sales it is being tested against.
- **Forecast forward.** From that date it predicts the next 10 weeks.
- **Compare to reality.** Those predictions are scored against what actually sold.
- **Across seasons.** Repeated at three points in the year (Oct-Dec, Dec-Feb, Mar-May), so the result does not depend on one favorable window.
- **Fair, side by side.** The current method runs through the identical test, on the same SKUs and periods, so the comparison is like-for-like.
- **A held-back final test.** A separate, more recent period was quarantined and never used during development. It gives an independent check of accuracy before the model can replace the current method.

## Limitations

Stated openly so expectations stay realistic.

- **Results are from backtesting; the final test is still to come.** The held-back period has not been run yet. It must pass before the model can replace the current method.
- **Only about two years of history.** That is one or two examples of each season, so the figures should become more reliable with another year of data. The holiday period is least certain, since the promotion pattern changed after 2024.
- **The long-history group is small.** About 80 SKUs, several close variants, so its figures rest on a small sample and may move.
- **Short-history SKUs are inherently harder.** Short histories mean less to learn from, so their forecasts are less certain.
- **Intermittent SKUs are out of scope.** The ~3,000 sporadic SKUs are not forecast at all.

## Where the project stands

The model is built and performs well in backtesting, but it is not in production yet, so the current spreadsheet method still runs day to day. The forecasting method itself is largely in place. What remains is the final validation test, the data corrections, and the user interface, all listed below.

## What comes next

- **Clean the sales data and feed it in.** Two fixes matter most. Stockouts push a week's recorded sales toward zero, which looks like falling demand when it was just unmet, and trains the model to under-forecast. Preorders are recorded when placed but ship later, so the demand lands in the wrong week, worst for new SKUs whose launch preorders dominate their short history. Both fixes help the current method too and are expected to be the largest remaining accuracy gain.
- **Build the interface with inventory-aware ordering.** The tool will also read current stock. When a SKU is low, out of stock, or has preorders waiting, it can raise the recommended order above the plain forecast, so orders cover both demand and backlog. This is part of the interface, not the model itself.
- **Run the final validation test.** Score the model once on the held-back period before proposing it to replace the current method.

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# Technical Detail

*The sections below are optional background. Part 1 covers what is needed to act on the results.*

## Why this approach was chosen

The current spreadsheet formula has two limits: it cannot learn from its own errors, and it cannot share information between SKUs, so each SKU is forecast on its own.

The new approach is a **LightGBM** model trained jointly across SKUs rather than one model per SKU. In practice it is split into two closely related versions by history length, covered in "How the new model works."

- **What LightGBM is.** A gradient-boosted decision tree model: it combines many small decision trees, each trained to correct the errors of the ones before it. It is the standard high-performing method for tabular data, meaning data laid out as columns of features describing each case (here, each SKU and week).
- **Why train across SKUs.** One model over all SKUs lets a pattern learned on one SKU carry to others, for example how demand ramps in the weeks after a launch. A SKU with little history borrows from SKUs with more history that behave like it. Per-SKU models cannot share this way.

The specific design follows the two largest public forecasting competitions. Our data matches each on a different point, so we take the model family from one and the seasonality handling from the other.

**M5 (2020): the model family.** M5 asked entrants to forecast daily unit sales for about 30,000 Walmart retail products.

- **How it matches us.** Same problem type: a catalogue of related retail SKUs whose demand patterns partly transfer between products. Our vehicle-accessory catalogue is the same shape of problem, only smaller.
- **What won.** Global LightGBM models, the same family used here. That is the direct evidence for the model choice.

**M4 (2018): the seasonality handling.** M4 covered about 100,000 series, most with short histories.

- **How it matches us.** Short history. We have about two years, so only one or two examples of each season.
- **What won.** Methods that removed seasonality up front as a fixed step and had the model learn only the leftover pattern, because short histories cannot teach seasonality reliably.
- **So we do the same.** The hand-set monthly multipliers are divided out of demand before training (deseasonalizing), the model learns the residual, and the multipliers are applied back afterward.

We differ from M5 in one place, and it is the same point where we match M4. M5 winners had more than five years of data, five or six examples of each annual event, enough for the model to learn seasonality directly. We do not. When we tested giving the model a month feature, it memorized the two specific years it had seen and over-forecast the post-holiday demand dip by 123%. So seasonality is imposed as fixed structure rather than learned.

The result is an M5-style model trained across SKUs, applied to M4-style deseasonalized demand. The same feature-based design also lets the model take in corrected data later (see What comes next), which the fixed formula cannot use.

## How the new model works

The model does not predict units directly. It predicts a **ratio**: how far a week's demand will run above or below the SKU's recent typical level. The forecast is that ratio times the level, times the season.

- **The recent level.** Each SKU's baseline is its **12-week average**, the trailing average of the last 12 weeks of demand. For long-history SKUs it is computed on deseasonalized demand.
- **The ratio.** The model predicts a multiplier on that level. A ratio of **1.0 just repeats the 12-week average**, so with no useful signal the forecast falls back to a safe baseline and only moves off it when the features give a reason. This keeps it from reacting to every noisy week.
- **The season.** For long-history SKUs the target week's monthly multiplier is applied at the end, completing the deseasonalize-then-reseasonalize round-trip from the previous section.

**What the model looks at.** Each SKU-week is described by a few features, all drawn from sales history:

- **Lead:** how many weeks ahead the target week is.
- **Recent-level ratios:** the last week, and the week before, each against the 12-week average, since the most recent weeks carry the most signal for the near term.
- **Ramp** (short-history model): the 4-week average against the 12-week average, which flags a SKU that is accelerating.
- **Elevation** (long-history model): the recent 4-week level against the SKU's own trailing annual level, which flags a SKU running above its normal yearly baseline.

**Two models, split by history.** Short-history and long-history SKUs use separate but similar models, because the same signal means different things for each:

- **Short history** keeps the ramp feature. A new SKU is often genuinely ramping after launch, so recent acceleration tends to continue.
- **Long history** drops the ramp feature and adds elevation. A mature SKU running above its usual level is more often a temporary spike than lasting growth, so the model treats "above its own annual baseline" as a reason to expect a return to normal rather than extend the spike. It keeps the recent-level ratios, so genuine gradual growth is still tracked.

The model is trained to minimize absolute error on the ratio, weighted by each SKU's demand level. That is the same demand weighting as the WAPE accuracy metric, so training and scoring optimize the same thing.

## Where the data comes from

Both methods read the same source: the company's full order history in the database (the `shipcore.fc_velocity_link_snapshot_forecast` table), covering every sales channel (West, East, and Amazon FBA). Because both are built from identical data, the comparison in Results is like-for-like, not an artifact of different inputs. The series is keyed on order date, with preorders recorded as a separate stream, which is the record the data-quality corrections in What comes next act on.

## The current method

The current method estimates a daily demand rate for each SKU from its recent sales pace, weighting the most recent weeks most and including preorders, then projects it forward across the forecast horizon with a seasonal adjustment. The daily rate is the sum of three streams: the main sales channel (West), the secondary channel (East), and Amazon (FBA):

> rate = West + East + FBA

Each of West and East is a weighted blend of recent sales pace, where every look-back window contributes units-per-day over its own length:

> West, East = Σ w · ( units(last d days) / d )

The sum runs over these **(d, w)** pairs, where d is the number of days in the look-back window and w is that window's weight in the final daily rate: (90, 0.10), (60, 0.15), (30, 0.30), (15, 0.20), (7, 0.15) on normal sales, plus (30, 0.10) on preorders. The weights add up to 1, so normal orders and preorders enter together as one weighted average, and the nearer windows carry the most weight. Each of West and East is then smoothed (damped) against its own value one week earlier, where S is the current blend and R is the blend one week back:

> damp(S, R) = 0.1 R + 0.9 S if |S - R| / R \< 0.5, else 0.2 R + 0.8 S

so a small week-to-week change is mostly kept and a large jump is held back. Amazon (FBA) is a plain 30-day average, units(last 30) / 30, with no smoothing. The horizon forecast then multiplies the daily rate by the horizon length H and a seasonal modifier M:

> forecast = rate · H · M (H = 70 days)

where H is the number of days being forecast (70, i.e. ten weeks) and M is the average of the monthly seasonal factors over that horizon, weighted by how many of the days fall in each month. The monthly factors are m = { 0.75, 0.80, 0.90, 0.95, 1.00, 1.00, 1.00, 1.00, 1.00, 1.10, 1.25, 1.30 } for January through December, so demand is marked down in winter and up toward the end of the year. As part of this project the formula was re-run on the company's data and reproduced its reported accuracy to the fourth decimal, so the figures in Results are the method's own numbers.
