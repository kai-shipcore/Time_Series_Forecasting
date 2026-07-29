# Demand Forecasting Frontend: Plan

Working plan for the forecasting and inventory dashboard. Replaces the earlier build spec,
which was derived from the requirements document without first establishing what decisions
the screens exist to support.

Status: outline. Sections 1 and 2 are settled. Section 3 onward is to be developed.

Scope: a Streamlit prototype (Phase 1 of the roadmap), intended for internal testing with
the forecasting and purchasing teams before any approved functionality moves into the
production web application.

---

## 1. Foundations

### 1.1 Purpose

Turn the demand forecast into decisions that purchasing, inventory and logistics can act on,
and make the basis for each decision inspectable enough to be trusted.

### 1.2 Users and their decisions

Every screen exists to serve at least one of these. A screen that serves none should not be
built.

| User | The decision they are making |
|---|---|
| Purchasing | Which SKUs need an order this week, and how many units of each |
| Purchasing | When to place an order, given container lead times and confirmed inbound |
| Inventory and logistics | Which SKUs are about to run out, and which are already out |
| Inventory and logistics | Whether an existing inbound shipment covers the gap or not |
| Forecasting owner | Whether the model is performing, and where it is not |
| Forecasting owner | Whether a specific number can be trusted before acting on it |
| Business owner | Whether the underlying data is sound enough to approve the process |

### 1.3 Questions that bring someone to the dashboard

Stated as the user would phrase them, to be mapped to screens in Section 3.

1. What needs my attention right now?
2. How much should I order for this SKU, and why that number?
3. When will this SKU run out?
4. Is this forecast reliable for this particular SKU?
5. Why is this number what it is?
6. Is the new model actually better than the spreadsheet we used before?
7. Which numbers on this screen should I not trust, and why?
8. What did we decide to order last time, and what happened?

### 1.4 Design principles

**Every number maps to an action.** If we cannot name what a user does differently because
of a figure, it does not belong on an operational screen. Diagnostic figures belong in the
validation area.

**Caveats travel with the number.** A warning about data quality or suppressed demand is
useless on a screen the purchaser never opens. Cross-cutting concerns appear as badges,
columns and annotations next to the affected figure, not as separate tabs. This applies
specifically to the three gaps identified in Section 3.2.

**Sample data is always labelled.** Any figure not derived from real data carries a visible
marker, at the point of display. This convention already exists and is retained.

**Record the reasoning.** Each screen carries a short design note explaining why it is laid
out as it is, following the convention in the reference screens from the forecasting team. A
prototype exists to be reviewed, and a reviewer cannot evaluate a layout whose reasoning is
undocumented.

**Verify before displaying.** Every displayed calculation is checked against the source
forecast output and raw data before the screen is considered done.

---

## 2. Data reality

Confirmed by inspection on 2026-07-27. This section exists because the three largest delays
in this project so far were all discovered at build time rather than at planning time.

### 2.1 Real and available

| Data | Source | Notes |
|---|---|---|
| Forward forecast, per SKU per week | `data/processed/ml_forward_forecasts.parquet` | LightGBM v11. 447 SKUs, 366 short and 81 long, 13 weeks forward |
| Spreadsheet (V1) forecast | `data/processed/v1_forward_forecasts.parquet` | Recomputed from a fresh database pull on every forecast run |
| Weekly sales history | `data/processed/sales_clean.parquet` | Weekly totals per SKU, W-MON convention |
| Segmentation | `data/processed/sku_profiles.csv` | Bucket and history length per SKU |
| Model accuracy, pooled and per SKU | `outputs/reports/ml_accuracy*.csv` | Model against V1, three backtest windows, by segment |
| Order lines, daily, by stream | `data/processed/orders_raw.parquet` | Includes `west_preorder` and `east_preorder`, about 19,400 rows |

### 2.2 Sample until an export lands

Available inventory, preorder backlog, confirmed inbound, inbound ETA, product name, size
and product status. These live in the Commerce and Supabase databases. The dashboard reads
`dashboard/data/inventory_snapshot.csv` when present and falls back to a labelled sample.

Everything downstream of these fields is therefore also provisional: recommended order
quantity, estimated stockout date, the Preorder and No Stock priorities, best sellers at
risk, and the inventory-dependent data-quality checks.

### 2.3 Not available anywhere

**Stockout and restock event dates per SKU per channel.** Nothing in this repository records
when a SKU went out of stock or was restocked. Per-channel marketplace detail is collapsed
into the west, east and fba streams. This blocks the restock recovery analysis entirely and
limits how precisely supply-distorted demand can be detected. See `docs/BACKLOG.md` item 2.

**Inventory history.** Only a current snapshot is expected. If confirmed, any inventory-aware
logic inside the profiler is restricted to forward runs, because a current snapshot cannot
answer an as-of question in a backtest without leaking. See `docs/BACKLOG.md` item 1.

**Purchase order outcomes.** No record of what was previously ordered, so the operational
workflow gap in Section 3.2 has no history to draw on yet.

### 2.4 Consequence for sequencing

Screens resting only on Section 2.1 can be built and trusted now. Screens resting on 2.2 can
be built now and become trustworthy when the export lands. Anything resting on 2.3 cannot be
built yet and should not be planned in detail until the data question is settled.

---

## 3. Information architecture

### 3.1 Screen map

Four screens, plus one deferred. Derived by grouping the questions in Section 1.3 by the user
and the moment at which they are asked, rather than by inheriting the previous screen list.

| # | Screen | Questions answered | Primary user | Data tier |
|---|---|---|---|---|
| S1 | Action List | 1, 3 | Purchasing, logistics | 1 and 2 |
| S2 | SKU Detail | 2, 4, 5 | Purchasing | 1 and 2 |
| S3 | Forecast Validation | 6 | Forecasting owner, business owner | 1 |
| S4 | Data Quality | 7 | Forecasting owner, business owner | 1 and 2 |
| S5 | Order Workflow | 8 | Purchasing | 3, blocked |

**S1. Action List.** The operational home. A priority-sorted list of SKUs needing attention,
with search, filtering and export, and a row-level jump into S2.

*Departs from the source requirements document, deliberately.* That document lists an
Inventory Overview and a Purchase Priority List as separate screens. This plan merges them.
Four of its six summary metrics are counts of subsets of the priority list itself
(preorder-priority, out of stock, best sellers at risk, stocking out within 30 days). On a
separate screen they are figures a user reads and then navigates away from. As filter
controls above the list they become the entry point to the work, and one navigation step
disappears from the most frequent task in the application. All six figures still appear.
`REQUIREMENTS.md` is treated as a useful starting point rather than a specification to
satisfy clause by clause.

**S2. SKU Detail.** Everything about one SKU: actual sales against the model and spreadsheet
forecasts, the weekly figures as a table, current inventory position, and the recommended
order quantity shown as a line-item calculation rather than a single number. This screen
carries the burden of question 5, so the reasoning behind a figure belongs here.

**S3. Forecast Validation.** The administrator view. Model against spreadsheet across the
backtest windows and segments, portfolio actual against predicted, and the largest over and
under forecasts. Entirely tier 1, so it is fully buildable and trustworthy now.

**S4. Data Quality.** The list of exceptions for the person fixing them. Per Section 1.4 the
individual flags also appear inline on S1 and S2 next to the affected SKU; this screen exists
so that someone working through the backlog of issues can see them in one place, which is a
different task from being warned mid-decision.

**S5. Order Workflow.** Deferred. Nothing records what was previously ordered, so there is no
history to build against. See Section 3.2.

### 3.2 The three cross-cutting gaps

Per Section 1.4 these attach to the figures they qualify rather than becoming destinations.

**Forecast depth.** Splits into three parts with different readiness.

- *Per-SKU reliability, buildable now.* `outputs/reports/ml_accuracy_by_sku.csv` holds each
  SKU's actual against predicted for each backtest window. Aggregated per SKU this gives a
  historical error rate that discriminates usefully: across the 260 SKUs with history the
  median is 0.23, the best tenth are at or below 0.13, and the worst tenth are above 0.48.
  Surfaces as a reliability column on S1 and a panel on S2.
  Coverage limit: only 260 of 447 forecast SKUs have any backtest history, because the
  remaining 187 were too new to be eligible in the older windows. "No history yet" must be a
  displayed state rather than a blank, and it is itself informative, since a SKU with no track
  record warrants more caution.
- *Prediction intervals, needs model work.* The v11 model emits a point forecast only. There
  is no upper or lower bound in `ml_forward_forecasts.parquet`. Adding conformal intervals to
  the serving layer is tractable, since the legacy statistical pipeline already did this, but
  it is model work and belongs in the ML track rather than this plan.
- *Scenario adjustment.* Letting a planner override an assumption and see the effect. Depends
  on which assumptions are worth exposing, to be decided in Section 4 with the screen specs.

**Supply-distorted demand.** Partly reachable. Pre-order periods are detectable from the
`west_preorder` and `east_preorder` streams already in `orders_raw.parquet`, so affected SKUs
can be flagged on S1 and explained on S2. Stockout and restock detection is blocked with no
event data (Section 2.3). The measurement work belongs to `docs/BACKLOG.md` item 2; what
belongs here is the display: a badge on the affected SKU and a plain explanation on S2 that
recent sales for this SKU understate demand and the forecast is therefore likely low.

**Operational workflow.** Largely blocked, with one exception. Tracking recommended against
ordered needs purchase order outcomes that no available source holds. What can be done now is
lightweight: export of the filtered list, and a local record of which SKUs have been actioned
this cycle so a purchaser working through the list does not lose their place. Both are useful
without any new data source. The full workflow stays as S5 until the data question is settled.

### 3.3 Navigation and flow

One path dominates and should be frictionless: land on S1, filter to a subset, open a SKU in
S2, decide, return to S1 with filters intact. S3 and S4 are visited deliberately rather than
in the course of purchasing work, so they may sit lower in the navigation.

### 3.4 Worked example

A walkthrough of one weekly cycle, using real figures from the 2026-07-20 forecast run. Its
purpose is to test the screen map: any step that cannot be completed reveals a missing
screen, and any screen that never appears is not earning its place. Inventory figures are
from the sample snapshot and are marked where they matter.

**Monday, shortly after the 9am forecast run. A purchaser opens the application.**

*S1, Action List.* The header reads 447 forecasted SKUs, 74 on preorder priority, 95 out of
stock, 40 best sellers at risk of stockout, 187 stocking out within 30 days, and 23,519 units
of total recommended order quantity. The list below is sorted by priority: 74 Preorder, then
85 No Stock, then 63 Best Seller, then 225 Routine. This answers "what needs my attention",
and the 30-day column answers "when will it run out". The purchaser clicks the "stocking out
within 30 days" figure and the list narrows to those 187.

*S2, SKU Detail, first SKU.* Top of the filtered list is `CA-SC-10-F-10-BK-1TO`, a
short-history SKU with 590 units of preorder backlog. The screen shows its sales history
against the model and spreadsheet forecasts, then the order calculation as line items:

```
Preorder demand              590
Lead-time demand           1,283
Safety stock                 330
Available inventory       -1,086
Confirmed inbound              0
--------------------------------
Recommended order quantity 1,117
```

Estimated stockout is 11 September, 46 days out. The reliability panel shows this SKU's
forecast has been off by about 13% across two backtest windows, which is in the best decile
of the portfolio. That answers "how much should I order and why that number", and "is this
forecast reliable". The purchaser accepts 1,117 units and marks the SKU as actioned.

*S2, second SKU.* Further down sits `CC-CS-03-J-GR-1TO`, a long-history best seller with 2
units on hand, forecast to stock out tomorrow. Recommended order is 125 units. But the
reliability panel shows this SKU's forecast has historically been off by 48%, among the worst
in the portfolio. This is the case the whole plan turns on. Without that panel the purchaser
orders 125 units against a figure that has been wrong by roughly half in both directions
before. With it, they know to widen the order or check the SKU manually. A point forecast
with no reliability attached invites exactly the wrong kind of confidence.

*Back to S1.* Filters are still applied and the actioned SKUs are marked, so the purchaser
resumes where they left off. They export the filtered list when finished.

**Later, the forecasting owner.**

*S3, Forecast Validation.* Answers "is the new model actually better". Model against
spreadsheet across three backtest windows and both segments, which currently shows the model
ahead in five of six cells.

*S4, Data Quality.* Answers "which numbers should nobody trust". The exceptions list, worked
through as a backlog rather than read mid-decision.

**What the walkthrough exposes.** Every step completes with the four screens, and each screen
appears. Two gaps are visible rather than hidden. First, 187 of 447 SKUs have no backtest
history, so the reliability panel that drove the second decision is unavailable for those and
must show an explicit "no history yet" state. Second, "mark as actioned" has no storage
behind it yet, which is the small piece of the operational workflow gap that is buildable now
(Section 3.2).

---

## 4. Screen specifications

To be developed, one screen at a time, after Section 3 is agreed. Each specification covers
the decision served, contents, data dependencies, calculations used, sample against real
fields, and the design note.

## 5. Calculations and definitions

Stated once here, implemented once in `dashboard/lib/calc.py`, and referenced everywhere
else. Revised 2026-07-27 after a review of the order quantity; the previous definitions are
recorded at the end of this section so the change is traceable.

### 5.1 Coverage window

```
coverage_weeks = lead_time_weeks + review_period_weeks     (default 8 + 1 = 9)
```

An order must cover demand until the next order can arrive, not merely until this one does.
Covering only the lead time leaves a shortfall of one review period in every cycle.

### 5.2 Coverage demand

The model's forecast summed over the first `coverage_weeks` forecast weeks. Where the horizon
is shorter than the window, remaining weeks are padded with the horizon's weekly average.

### 5.3 Forecast error used for a SKU

The SKU's own pooled error from the backtest windows for the served model version
(`lib/reliability.py`). A SKU with no backtest history inherits the median error of its
segment rather than being treated as error-free, because unmeasured is not the same as
accurate.

### 5.4 Safety stock

```
safety_stock = service_z x forecast_error x coverage_demand      (default z = 1.0)
```

Sized by how wrong this SKU's forecast has actually been. The previous flat rule allocated
the buffer backwards, holding the most stock against the best-predicted SKUs.

### 5.5 Confirmed inbound

Only inbound with an ETA inside the coverage window is credited against the order. Inbound
arriving later, or with no ETA at all, is reported separately and not subtracted, since it
cannot cover the demand this order is for.

### 5.6 Recommended order quantity

```
ROQ = preorder_backlog + coverage_demand + safety_stock
      - available_inventory - inbound_arriving_in_window        (floored at 0)
```

Every component is rounded to whole units before the total is taken, so the line items
displayed to a user add up exactly to the quantity shown.

### 5.7 Estimated stockout date

Stock is depleted against the SKU's own forecast curve, week by week, interpolating within
the week it runs out. Past the end of the horizon, demand continues at the horizon's average
weekly rate. Inbound extends cover only if it arrives before the shelf empties; arriving
later makes it a refill after a stockout rather than a prevention of one, and those SKUs are
flagged (`stockout_before_inbound`).

Using the forecast rather than a trailing average matters most where the two disagree. A SKU
whose weekly demand ramps from 5 to 20 units holds out 52 days on 60 units of stock, where a
flat average of the same curve would claim 32. The trailing average cannot see a seasonal
peak coming; the forecast can, and every other figure on these screens already relies on it.

Note that the "average per day" figure shown on SKU Detail is still the trailing 30-day rate.
That is a factual statement about recent sales rather than a projection, and is labelled as
such; it is not used to compute the stockout date.

### 5.8 Priority

Highest applicable, in order: Preorder (open backlog), No Stock (available at or below zero),
Best Seller (top share of recent units, default 20%), otherwise Routine.

### 5.9 Superseded definitions

Before 2026-07-27 the order quantity used a flat `safety_days x trailing average daily sales`
buffer, covered the lead time only, and subtracted all confirmed inbound regardless of ETA.
The stockout date treated all inbound as available immediately.

## 6. Build sequence

To be developed, following the tiering in Section 2.4.

## 7. Open questions and known limitations

To be developed, with items that block work recorded in `docs/BACKLOG.md` rather than here.
