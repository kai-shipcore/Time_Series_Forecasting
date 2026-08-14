# Demand Forecasting: Project Summary

Prepared by Yuchan  |  August 2026

------------------------------------------------------------------------

# Summary

## Purpose and scope

The company forecasts weekly demand per SKU to decide how much stock to order. At the start of this project that was an in-house spreadsheet formula, referred to throughout as the **current method**.

The objective was to find out whether a model trained on the company's own sales history could forecast more accurately, and if so, to put that forecast in front of the planning team as a working tool.

**A note on the comparison.** The current method was reproduced in code from the spreadsheet as it stood in mid-July 2026, and every figure here compares against that version. If the spreadsheet has changed since, the comparison is against the earlier form. The exact formula, its constants, and the points where it is most likely to have diverged are in the Technical Detail section.

The work covered five areas, all complete and in production:

1. A way to measure forecast accuracy fairly, so that any two methods can be compared on equal terms.
2. The forecast itself. An initial statistical model was built first, then replaced by the machine learning model now in use.
3. The data pipeline and the service that produce and publish the forecast.
4. Deployment on the company server, including an automated weekly update that runs without manual intervention.
5. Two screens in Demand Pilot: one showing what to order, and one showing how accurate the forecast has been.

Two terms are used throughout:

- **Demand**: units of a SKU that customers ordered in a given week.
- **Forecast**: a prediction of that weekly demand for a week that has not yet occurred, in the same units.

## What the model produces

- A predicted number of units sold for each of the next thirteen weeks, for every SKU it covers.
- Republished weekly, automatically.
- Standard orders and preorders both count as demand, in the week the order was placed, so one forecast covers both.

## Results

The model was validated once, against a ten-week period set aside at the start of the project and not examined until development finished. Because it was never used while building the model, it measures accuracy independently rather than judging the work by its own standard.

**Over that period the new model reduced forecast error by 42 percent compared with the current method.**

| Measure, ten weeks | Current method | New model |
| --- | --- | --- |
| Units of forecast error | 12,470 | **7,274** |
| Error as a share of demand | 30.6% | **17.8%** |
| Net over or under forecast | 28.0% under | **0.0%** |

Measured across 40,762 units of actual demand, May to July 2026.

- **Financial significance.** Each unit of error is either stock ordered that does not sell or demand that cannot be met. The new model removed 5,196 units of it. At an assumed $200 average selling price that is roughly $1.0 million in reduced inventory exposure, an upper bound rather than a realised saving, since the actual value depends on margins and holding costs.
- **Forecast bias.** The current method ran 28 percent below actual demand overall and 35 percent below on newer SKUs. A forecast that is consistently low is a standing stockout risk somebody corrects by hand every ordering cycle. The new model was within 0.1 percent in aggregate.

## Where the improvement comes from

The largest gains come where demand changes direction, particularly entering and leaving the fourth-quarter peak. The current method estimates a recent sales rate and projects it forward, so it responds to a change only after the change has happened.

Accuracy by period and SKU group, measured as error as a share of demand:

| Period | SKU group | Current method | New model |
| --- | --- | --- | --- |
| Dec 2025 to Feb 2026 | Established | 39.3% | **13.9%** |
| Dec 2025 to Feb 2026 | Newer | 22.4% | **19.9%** |
| Mar to May 2026 | Established | 27.8% | **13.5%** |
| Mar to May 2026 | Newer | 33.5% | **19.3%** |
| Oct to Dec 2025 | Established | **8.5%** | 10.4% |
| Oct to Dec 2025 | Newer | **22.1%** | 24.7% |

The current method wins both October to December cells. That is the only period of the three where it does, and both differences fall within normal variation between samples.

In the charts below, actual demand is the solid black line and both forecasts are dashed. Each vertical marker is a new ten-week forecast being issued.

**Established SKUs: weekly demand against both forecasts, October 2025 to May 2026**

![](outputs/reports/management_chart_established.png)

After Christmas, demand falls from about 2,400 units per week to about 1,000. The current method instead rises to about 2,300 and holds there, projecting December's rate forward, then over-corrects and sits roughly 400 units per week low through the spring. The new model follows the decline.

**Newer SKUs: weekly demand against both forecasts, December 2025 to May 2026**

![](outputs/reports/management_chart_newer.png)

The same effect in the opposite direction. From March, demand rises toward 2,900 units per week while the current method stays near 1,600. The new model follows it. The chart starts in December because too few newer SKUs existed before then for a fair comparison.

## What was delivered

**1. A method for measuring forecast accuracy.**

- The existing spreadsheet formula reproduced in code, matching its reported accuracy to four decimal places.
- Both methods run through the same scoring process, on the same SKUs and periods, so no result depends on a difference in data or calculation.
- It reports whether a gap between two methods is larger than normal sample variation, so a small difference is not mistaken for an improvement.
- Built first, before any model existed, so later results could not be judged by the work that produced them.

**2. An initial statistical model.**

- Standard statistical techniques applied per SKU, with the most suitable one selected automatically for each.
- Ran weekly and published a thirteen-week forecast through its own screens.
- More accurate than the current method, and set the standard the machine learning work had to exceed.
- Retired in August 2026 once that standard was beaten, and its screens removed. The code is kept because every accuracy figure recorded during the project is measured against it.

**3. The machine learning model now in use.**

- Trained across all covered SKUs together rather than one model per SKU, so a pattern seen in one product informs forecasts for others. Most SKUs have under a year of history and say little alone.
- Eighteen versions built and tested. Fourteen rejected for not improving accuracy, each with the measurements behind the rejection.
- Two related models in production, one for newer SKUs and one for established SKUs. Growth in a newly launched SKU usually continues; the same pattern in an established SKU is more often temporary.

**4. The data pipeline.**

- One process runs the whole sequence: refresh the order data, rebuild each SKU's weekly sales history, reclassify every SKU, produce the thirteen-week forecast, publish it.
- Every file is staged and moved into place only after the whole run succeeds, so a failure leaves last week's forecast intact and in use rather than a half-updated set.
- Published to the database and to a file. Every forecast ever issued is kept in an accumulating record, backed up weekly, and it is the only part of the system that cannot be rebuilt: it holds what was predicted before the outcome was known, which is what makes after-the-fact accuracy reporting possible.

**5. The forecast service.**

- Runs on the company server and supplies Demand Pilot with the forecast, the detail behind each recommendation, the accuracy tables and demand history. The team sees it in the application they already use, and nobody needs a copy of the data or the modelling software.
- Access requires a token.
- A status check reports whether the service has data and which version of the code is running, separating three problems that look identical from outside: a server that is down, one running an old version, and one running correctly with nothing to serve.

**6. Weekly automation and deployment.**

- The forecast regenerates every Tuesday morning, unattended. Tuesday rather than Monday because a sales week closes Monday night, so a Monday run would always be a week behind.
- Code and data have separate owners and cannot overwrite each other. Code deploys automatically on publish; data is produced on the server itself, so no other machine needs to be running.
- Three checks run unprompted: a failed weekly run notifies, an hourly check confirms the service is reachable and still refusing untokened requests, and the screens report when they cannot reach it.

**7. The Action List.** The screen showing what to order. It combines the forecast with stock on hand, outstanding preorders and confirmed incoming shipments, and ranks SKUs by which require attention. A forecast alone is not an ordering instruction: 400 predicted units means something different depending on whether 50 or 5,000 units are already in the warehouse.

- A recommended order quantity per SKU, shown on the detail page as the arithmetic behind it, so it can be checked by hand.
- An estimated stockout date, run against that SKU's own forecast curve rather than a flat average, and accounting for whether incoming stock lands before the shelf empties.
- A reliability rating per SKU with the evidence behind it. Where a SKU has never been measurable, the screen says so rather than showing a figure that belongs to something else.
- Filters, sorting, column selection, search and export, so the list can be worked through as a queue.
- Adjustable planning parameters, such as weeks of cover. Applied by the service rather than the browser, so the order formula exists once rather than twice.
- A demand chart that follows the current filters, so the chart and the table always describe the same products.
- A separate view of the SKUs the model does not forecast, sized from recent sales and labelled as such.
- A button to run the forecast on demand, for when the weekly run is not soon enough.

**8. The Forecast Validation screen.** The screen showing whether the forecast can be relied upon. Five parts:

- The model against the current method across every SKU group and test period, including the cells where the current method still wins. A comparison that reports only its wins is not evidence.
- Demand patterns, showing what demand looks like independent of any forecast, and how much of it the model covers.
- Actual weekly demand against what the forecast said those weeks would be, continuing past the last completed week into the current forecast.
- How each weekly forecast performed once its weeks completed. A stronger claim than the comparison above, because those were published before the outcome was known. Fills in as weeks pass.
- A SKU-level breakdown of whether the improvement is broad or carried by a few products, by SKU count and by units, traceable to the individual product.

**9. Documentation and handover.** Enough for someone else to take the work over:

- An operating guide for running the system day to day.
- A deployment reference covering the server setup and what to do when the service stops responding.
- A guide to the code, and a technical guide to the model.
- A full record of design decisions, including the fourteen rejected versions and the evidence behind each.
- A handover note of findings and caveats a successor would otherwise rediscover, and a list of identified but incomplete work.

## Which SKUs are forecast

The company sells approximately 3,500 SKUs. Around 340 sell regularly enough for a weekly forecast to be meaningful. That count is not fixed: SKUs are reclassified each week as sales patterns change.

A small share of the catalogue by SKU count and the large majority of units sold, which is why accuracy was concentrated there.

Each SKU is assessed on two questions:

- **How regular are its sales?** A SKU selling in most weeks can be forecast. One with no sales in a substantial share of recent weeks is classified as irregular and gets no forecast, because any weekly figure would be unsupported. Its sales history is still shown.
- **How much history does it have?** Under a year is treated as newer and forecast using a different model from established SKUs.

## How accuracy was measured

All figures come from testing against past periods, using only information that would have been available at the time.

- The model is trained on sales up to a chosen past date and given nothing after it.
- From that date it forecasts the following ten weeks, which are compared against what actually sold.
- The process is repeated at three points across the year, so results do not depend on one favourable period.
- The current method is tested identically, on the same SKUs and periods.
- One ten-week period was set aside before development began and used only once, at the end.

The criteria for that final test were recorded in advance: what would count as success, what would count as failure, and what outcome was expected. The project also committed in advance to recommending the current method be kept if the model did not beat it. Fixing the criteria beforehand stops results being reinterpreted afterwards.

Accuracy is total forecast error divided by total actual demand, so 20 percent means forecasts were off by 20 percent of units sold. It weights each SKU by volume, so an error on a high-volume product counts more. Bias sits alongside it, recording whether forecasts run consistently high, meaning overstock risk, or low, meaning lost sales.

## Limitations

**Not in scope.** Three exclusions, each a decision rather than a gap:

- **The roughly 3,200 irregular-selling SKUs are not forecast.** Most of the catalogue by SKU count, a minority of units sold. They have no sales in most weeks, so a weekly figure would be an estimate with nothing behind it. They stay visible on the Action List, sized from recent sales and labelled as such. Forecasting them needs a different method answering a different question, closer to whether a SKU sells at all this month than to how many.
- **Revenue and margin.** Everything forecast, recommended and measured is in units. No figure in the system is in currency.
- **Order timing and supplier logistics.** The Action List gives the quantity needed and when stock is projected to run out, not when to order, how to combine orders across suppliers, or how to fill a container.

**Limits of the forecast itself.**

- **The improvement is not uniform.** Gains concentrate where demand changes direction. In stable periods the model performs much like simpler approaches.
- **A brand new SKU cannot be forecast at all.** The model needs history before it can produce a number, so a product gets nothing until it has sold for some months. A real gap, since a launch is when a wrong order costs most, but a figure drawn from no history would be a guess presented as a forecast.
- **Coverage is effectively car covers and seat covers.** Only three or four categories sell regularly enough to forecast. The rest are too new or too small and fall into the irregular group.
- **A SKU with an unusual pattern is served less well.** One model trained across all covered SKUs is what lets a product with little history borrow from products with more. The cost is that a SKU behaving unlike the rest is pulled toward the general pattern.
- **Accuracy is concentrated on the highest-volume SKUs.** The model fits what most of the demand looks like, and the accuracy measure weights each SKU by volume, so both favour the same products. Deliberate, because that is where the money is, but it means an individual low-volume SKU may be forecast better by something simpler. Raising the bar for which SKUs are forecast removed much of this; the judgement is applied to the group rather than product by product, so some remain.
- **The seasonal factors are set by hand, not learned.** Learning them would be preferable and is not realistic on two years of history that differ this much from each other: the model would learn one specific past rather than a repeating pattern. The holiday period is least certain, the promotional pattern having changed after 2024.
- **The established SKU group is small**, several of its products close variants, so its figures rest on a modest sample.
- **Newer SKUs are inherently harder**, having less history to learn from.
- **Forecasts run thirteen weeks; accuracy is measured over ten.** Error grows with distance, so the last three weeks are less certain than the measured figures suggest.
- **Recorded demand understates true demand during stockouts**, which affects every method including the current one.

## Further work

Four things, in order of expected value. A full list with the reasoning behind each is kept in the project repository.

1. **Correct the sales data for stockouts.** When a SKU is out of stock, recorded sales fall toward zero, and the record reads that as demand falling. Every forecasting method learns from that record, so all of them predict low for exactly the products that most need reordering. Blocked on stockout dates per SKU, which are not recorded in usable form today.
2. **Give the forecast price and promotion data.** It currently sees units sold and nothing else, so a discount that moved 300 units looks like ordinary demand for 300 units. This is also part of the seasonal problem above: the change in holiday behaviour after 2024 is a promotional change being treated as a calendar one. The largest missing input.
3. **Add order timing to the Action List.** Supplier lead times and the container schedule would turn "order 400 units" into "order 400 units by this date", which is the decision being made.
4. **Put money on the screens.** Everything is in units, so the list cannot be triaged by spend. Needs a decided cost basis and a source for it before it means anything.

Beyond these, a set of model improvements is identified and testable, and a smaller set is blocked on more sales history accumulating.

## Current status

- In production and running without manual intervention.
- The forecast updates weekly on the company server.
- The Action List and Forecast Validation screens are in use by the planning team.
- The screens built on the initial statistical model have been retired, so there is one forecast in the system rather than two that could disagree.
- Everything needed to continue the work is documented, as described in item 9 above.

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# Technical Detail

*The following sections provide additional background and are not required to act on the results above.*

## Why this approach was selected

The current spreadsheet formula has two structural limitations: it cannot learn from its own past errors, and it cannot use information from one SKU when forecasting another, so each product is forecast in isolation.

The replacement is a machine learning model of a type commonly used for this class of problem, trained across all covered SKUs together. A pattern seen in one product, such as how demand behaves after a launch, can then inform forecasts for similar products, so a SKU with limited history draws on comparable SKUs with more. Per-SKU models cannot do this, and it is the main structural advantage available given how recent much of the catalogue is.

The design follows the two largest public forecasting competitions, which are the standard reference points for this kind of work. The company's data resembles each on a different dimension:

- The first covered daily unit sales for approximately 30,000 retail products, the same class of problem faced here. Models of the type used in this project produced the winning entries, which is the basis for the approach selected.
- The second covered approximately 100,000 series, most with short histories, matching the company's roughly two years of data. The successful approaches there removed seasonal variation as a fixed preliminary step and had the model learn only the remaining pattern, on the basis that short histories cannot reliably teach seasonal behaviour.

Both findings are applied here. Seasonality uses the existing monthly factors rather than being learned. When the model was allowed to learn it directly, it reproduced the specific two years it had seen and over-forecast the post-holiday period by 123 percent.

## How the model works

The model does not predict unit volumes directly. It predicts how far a week's demand will run above or below that SKU's recent typical level, and the forecast is that proportion applied to the recent level, adjusted for season.

- **Recent level.** Each SKU's reference point is its average weekly demand over the previous twelve weeks.
- **The predicted proportion.** A value of 1.0 repeats that average unchanged, so with no clear signal the forecast stays at the recent level and moves away from it only where the data supports doing so.
- **Seasonal adjustment.** For established SKUs, the relevant monthly seasonal factor completes the calculation.

The inputs are few, and all derived from the sales history itself:

- How many weeks ahead the forecast week is.
- The most recent week and the week before it, each relative to the twelve-week average.
- For newer SKUs, the four-week average relative to the twelve-week average, which indicates acceleration.
- For established SKUs, the recent four-week level relative to that SKU's own level over the previous year, which indicates whether it is running above its normal range.

Newer and established SKUs use separate models because the same signal means different things for each. A newly launched SKU showing acceleration is usually still growing; an established SKU running above its normal level is more often experiencing a temporary increase, so the model expects a return toward normal. Both keep the recent-level inputs, so genuine gradual growth is still reflected.

The boundary between the two sits at a year of sales history because the established model's comparison against the previous year cannot be calculated with less than that. It is a requirement of the input rather than a figure chosen for convenience.

This was not the initial approach. A single combined model was tested first and rejected, because adjustments that improved forecasts for established SKUs degraded them for newer ones.

## How development decisions were made

- Changes were made one at a time and evaluated across three test periods covering different seasons.
- The criteria for acceptance were recorded before each test.
- A change was accepted only if it improved accuracy consistently across all three periods by more than a measured threshold for normal variation. That threshold was established by repeatedly resampling the SKU population, which showed that differences below approximately one percentage point cannot be distinguished from chance and were therefore treated as inconclusive.
- Eighteen versions were evaluated and fourteen rejected, each with the measurements supporting the rejection, so that future work does not repeat approaches already shown not to help.

## Data sources

Both forecasting methods read the same order history, so the comparison reflects the methods themselves rather than differences in input.

| Used for | Table | Contents |
| --- | --- | --- |
| Demand, both methods | `shipcore.fc_velocity_link_snapshot_forecast` | Complete order-line history, all sales channels, no date limit. Order date, SKU, quantity, order type and channel |
| Stock on hand | `ecommerce_data.coverland_inventory_by_warehouse` | On hand, allocated, available and backorder, per warehouse. Read live |
| Incoming shipments | `shipcore.fc_container_items` with `shipcore.fc_containers` | Confirmed inbound quantities and arrival dates, and draft containers. Read live |
| Forecast published to | `shipcore.ml_forward_forecasts` | The current thirteen-week forecast, one row per SKU per week |
| Forecast history | `shipcore.ml_forecast_history` | Every forecast ever issued, accumulating |

Two points about the demand data:

- It must be the uncapped table. The similarly named `shipcore.fc_velocity_link_snapshot` holds only 120 days, which is shorter than the look-back windows the current method uses.
- Demand is recorded against the date the order was placed, with preorders flagged separately by order type.

Weeks run Tuesday to Monday and are labelled by the Monday on which they end. The weekly update runs on Tuesday morning so that it includes the week that closed the previous day.

## The current method, for reference

**This is the formula as it stood in mid-July 2026**, which is when it was reproduced in code. Every figure attributed to the current method in this document comes from this version. If the spreadsheet has been changed since, the comparison is against the earlier form and would need re-running to reflect the current one.

The current method estimates a daily demand rate for each SKU from its recent sales pace, weighting recent weeks most heavily and including preorders, then projects that rate forward with a seasonal adjustment. The daily rate is the sum of three channels: the main sales channel (West), the secondary channel (East), and Amazon (FBA):

> rate = West + East + FBA

West and East are each a weighted blend of recent sales pace, where each look-back window contributes units per day over its own length:

> West, East = Σ w · ( units(last d days) / d )

The sum runs over these (d, w) pairs, where d is the number of days in the look-back window and w is that window's weight: (90, 0.10), (60, 0.15), (30, 0.30), (15, 0.20), (7, 0.15) for standard sales, plus (30, 0.10) for preorders. The weights total 1, so standard orders and preorders combine into a single weighted average, with nearer windows carrying the greatest weight. West and East are then each smoothed against their own value one week earlier, where S is the current blend and R is the blend one week prior:

> damp(S, R) = 0.1 R + 0.9 S if |S - R| / R \< 0.5, else 0.2 R + 0.8 S

so a small week-to-week change is largely retained and a large movement is moderated. Amazon (FBA) uses a plain 30-day average with no smoothing. The forecast multiplies the daily rate by the forecast length H and a seasonal modifier M:

> forecast = rate · H · M (H = 70 days)

where M is the average of the monthly seasonal factors across the forecast period, weighted by how many days fall in each month. The monthly factors are m = { 0.75, 0.80, 0.90, 0.95, 1.00, 1.00, 1.00, 1.00, 1.00, 1.10, 1.25, 1.30 } for January through December.

The reproduction matched the spreadsheet's reported accuracy to four decimal places, so the figures attributed to the current method in this document are its own rather than an approximation of it.

**Where the reproduction is most likely to differ from the spreadsheet**, if it does, in rough order of how much each would move the answer:

- The as-of date. It uses the day before the forecast date. Using the forecast date itself changes a worked example by 4.4 percent.
- The preorder weighting. Preorders are 20 percent of demand overall and over half of some SKUs, but enter the blend at a fixed weight of 0.10. If the spreadsheet weights them differently, the divergence is largest on exactly the preorder-heavy SKUs.
- The damping thresholds and whether Amazon is damped. Here it is not, and it bypasses the blend entirely.
- The order in which a line is classified. The channel check runs before the order type check, so an Amazon preorder counts as Amazon rather than as a preorder.
