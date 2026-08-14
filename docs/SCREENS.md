# The screens: Action List and Forecast Validation

**Audience:** the engineer maintaining these two pages. It says what each part displays,
which endpoint serves it, where every figure is computed, and what is currently wrong.

**Where this sits.** `OVERVIEW.md` explains what the forecast is for, `MODEL.md` how it is
produced, `DATA_AND_PIPELINE.md` how it reaches the database. This document covers the last
hop, from database to pixel.

**The one architectural rule.** The Next.js app computes **nothing** about forecasting or
planning. Every figure on both screens is calculated in Python, in
`Time_Series_Forecasting/src/planning/`, and proxied through thin route handlers. Preserve
that. It means there is exactly one implementation of the order formula, and it is testable
without a browser.

---

## 1. How a request travels

```
browser
  └─ fetch(apiPath("/api/planning/…"))          src/lib/api-path.ts:12
       └─ Next route handler                     src/app/api/planning/**/route.ts
            └─ proxyPlanning()                   src/lib/planning-api.ts:19
                 │  AI_SERVICE_URL, default http://localhost:8000
                 │  header x-forecast-token: FORECAST_API_TOKEN
                 └─ FastAPI                      api/main.py
                      └─ src/planning/calc.py    ← every number is computed here
```

**Route handlers are one-liners on purpose.** They do no validation: FastAPI's
`Query(ge=…, le=…)` is the single source of bounds, and the client clamps to the same bounds
in `types.ts:171-185`. Adding validation in the middle would create a third place for the
bounds to disagree.

### The proxy's error taxonomy

`planning-api.ts:63-159`. Worth knowing because the page renders each differently.

| Condition | Result | Note |
|---|---|---|
| Connection failure | `ensureForecastServer()`, then one retry | A **local** server is auto-started; a **remote** one gets a 750 ms pause. Then `{kind:"unreachable"}` 503 |
| Upstream 404, not a `/planning/sku/` path | `{kind:"outdated"}` | "the forecast server is running an older revision" |
| Upstream 500 or above | Calls `forecastHealth()`; if `ready === false`, `{kind:"no_data"}` 503 naming the missing files | |
| Anything else | Status passed through verbatim | Deliberate, so a 404 from the SKU endpoint keeps its meaning |

The auto-start behaviour is the cause of the diagnostic trap in `DATA_AND_PIPELINE.md`
Section 8: a misconfigured `AI_SERVICE_URL` looks like it works, because something local
answers.

### Endpoints

| Next route | Upstream | Timeout |
|---|---|---|
| `GET /api/planning/action-list` | `/planning/action-list` | 20 s |
| `GET /api/planning/sku/[sku]` | `/planning/sku/{id}` | 20 s |
| `GET /api/planning/sku/[sku]/history` | `/planning/sku/{id}/history` | 20 s |
| `GET /api/planning/not-forecast` | `/planning/not-forecast` | **40 s**, roughly 7× the SKUs |
| `GET /api/planning/validation` | `/planning/validation` | **60 s**, scoring stored runs loads full history on a cold server |
| `GET /api/planning/demand-patterns` | `/planning/demand-patterns` | 30 s |
| `GET /api/planning/demand-vs-forecast` | `/planning/demand-vs-forecast` | 30 s |
| `POST /api/planning/demand-trend` | `/planning/demand-trend` | 30 s, **does not use the shared helper**; the helper is GET-only |
| `POST /api/planning/run-forecast` | `/planning/run-forecast` | 30 s |
| `GET /api/forecast/status/[jobId]` | `/forecast-status/{job_id}` | 5 s, also not via the helper |

`/api/forecast/status` is the **only** survivor of the fourteen `/api/forecast/*` proxy
routes deleted on 2026-08-13. It is generic job machinery that knows nothing about which
pipeline produced the job, which is why it survived.

---

## 2. The Action List

**Purpose.** Turn the forecast into a worklist: which SKUs need an order, how many units, and
in what order to deal with them.

**Routes.** `src/app/planning/action-list/page.tsx` for the list,
`src/app/planning/action-list/[sku]/page.tsx` for the detail. The SKU sits in the path rather
than the query string so a row is shareable and middle-clickable.

### 2.1 Component tree

```
ActionListPage (server)
├── ActionListPageHeader
└── ActionListContent                    action-list-content.tsx:76   ← the hub, 948 lines
    ├── provenance bar                   trained_through / model / horizon_end /
    │                                    demoted_since_forecast / SAMPLE-inventory warning
    │   ├── ModelCard                    model version popover
    │   └── ForecastServerStatus         up/down badge; onRecovered triggers a refetch
    ├── RunForecast                      run-forecast.tsx:67, collapsed by default
    ├── section toggle                   "Forecast" vs "Not forecast"
    ├── NotForecastSection               only when section === "not-forecast"
    │   └── NotForecastTable             13-week-rate table, no order quantity
    ├── summary filter buttons           7 clickable counts; doubles as the only priority filter
    ├── PlanningControls                 lead / review / service-z / risk window
    ├── search, category/tier/trend selects, Reset
    ├── ColumnPicker                     which optional columns are visible
    ├── Export CSV                       csv-export.ts
    ├── PortfolioChart                   collapsed; actual + forecast summed over filtered SKUs
    ├── data-quality warning line        counts of r.flags, click to filter
    ├── ActionListTable                  action-list-table.tsx
    │   └── ColumnHeaderMenu             sort + per-column checkbox filter
    └── PlanningError                    error card with retry
```

`rememberSkuSequence()` (`sku-sequence.ts:37`) writes the on-screen filtered and sorted order
to `sessionStorage`, so the detail page's Prev/Next walks the same sequence the user is
looking at rather than a canonical one.

**`DEFAULT_SORT = []`** (`action-list-table.tsx:254`). No client sort, deliberately, because
the server's worklist order is not reproducible from any single column.

### 2.2 The recommendation formula

**This is the most important code on either screen.** `src/planning/calc.py:410-427`.

```python
# Every component is rounded to whole units BEFORE the total is taken, so the
# figures shown in the breakdown add up exactly to the recommended quantity.
df["safety_stock"] = (z * df["error_used"] * df["coverage_demand"]).round()
df["coverage_demand"] = df["coverage_demand"].round()
for col in ("preorder_backlog", "available_inventory",
            "confirmed_inbound", "inbound_in_window", "inbound_excluded"):
    df[col] = df[col].round()

df["recommended_order_qty"] = (
    df["preorder_backlog"]
    + df["coverage_demand"]
    + df["safety_stock"]
    - df["available_inventory"]
    - df["inbound_in_window"]
).clip(lower=0).astype(int)
```

In words:

```
coverage_weeks = lead_time_weeks + review_period_weeks
safety_stock   = round(service_z × error_used × coverage_demand_raw)

recommended_order_qty = max(0, int(
      round(preorder_backlog)
    + round(coverage_demand)
    + safety_stock
    − round(available_inventory)
    − round(inbound_in_window)
))
```

Note that `safety_stock` is computed on the **unrounded** coverage demand, one line before
coverage demand is itself rounded.

**Each input, precisely.**

| Input | Source | Note |
|---|---|---|
| `coverage_demand` | `_coverage_demand`, `calc.py:97-110` | Sum of `yhat` over the first `coverage_weeks` forecast weeks. If the horizon is shorter than the window, the shortfall is padded at the horizon's mean weekly rate |
| `available_inventory` | `SUM(available)` = on_hand − allocated | **Not** net of backorder |
| `preorder_backlog` | `SUM(backorder)`, same table | **Added** to the requirement |
| `inbound_in_window` | `calc.py:398-408` | Confirmed statuses only, and **ETA must fall on or before `coverage_end`**. Inbound with **no ETA is never credited** |
| `safety_stock` | `calc.py:415` | `z × error_used × coverage_demand`. The SKU's **own measured WAPE** times the demand it must cover, not a flat days-of-cover rule |
| `lead_time_weeks` | Parameter | Enters only through `coverage_weeks` and `coverage_end` |
| **drafts** | | **Never subtracted.** See below |

**Drafts are shown and not subtracted, deliberately** (`calc.py:238-240`, `types.ts:22-34`):
a draft is not a commitment, so crediting it would under-order exactly the SKUs someone has
already acted on.

**No lot size, MOQ or container rounding exists anywhere.** If the business needs it, that is
new work, not a missing config value.

**Defaults**, `calc.py:63-83`, mirrored in `types.ts:142-147`:

```python
DEFAULT_PARAMS = {
    "lead_time_weeks": 8,          # supplier + transit
    "review_period_weeks": 1,      # how often orders are placed
    "service_z": 1.0,              # ≈84% service level; 1.65 ≈ 95%
    "best_seller_demand_share": 0.50,
    "stockout_horizon_days": 30,
}
```

`best_seller_demand_share` is not exposed in the UI.

### 2.3 The breakdown, and why rounding order matters

`calc.py:612-660` builds the arithmetic shown on the detail page. Each line carries a `Sign`:
`+1` add, `−1` subtract, `0` total, `None` informational aside. It prefers the **stored**
`recommended_order_qty` over recomputing, so the breakdown can never disagree with the list.

That is also why every component is rounded before summing. Rounding only the total lets the
displayed lines disagree with it by a unit or two, which destroys the point of showing the
arithmetic at all.

### 2.4 Days of cover and the projected stockout date

`_days_until_consumed`, `calc.py:124-146`. Walks the SKU's **own forecast curve** week by
week, interpolating linearly inside the week the stock runs out, and continuing at the
horizon's mean rate past the end. Zero-forecast weeks are skipped, not treated as depletion.

Applied at `calc.py:435-457`. The rule to know:

```python
in_time = inb > 0 and np.isfinite(eta) and eta <= t_on_hand
```

**Inbound extends cover only if it lands before the shelf empties.** Otherwise it is a refill
after a stockout and `days_to_stockout` ignores it entirely.

That leaves two apparently contradicting assumptions on the same row: `days_to_stockout`
ignores late inbound, while `recommended_order_qty` credits any inbound inside the coverage
window. **The supply gap reconciles them** (`calc.py:528-544`): when inbound arrives after
the projected stockout, `supply_gap_days` is the difference, and `gap_closable_by_order` says
whether ordering now would actually help.

### 2.5 The priority ladder

`calc.py:502-511`. Three states, first match wins, lowest number first.

| Rank | Label | Condition |
|---|---|---|
| 1 | **Preorder** | `preorder_backlog > 0` |
| 2 | **No Stock** | `available_inventory <= 0` |
| 99 | **Routine** | everything else |

All three are values of one variable, what the stock situation is, which is what makes a
ladder the right shape.

**"Best Seller" used to be rank 3 and was removed** (`calc.py:485-501`). It answers a
different question, importance rather than supply state, and it always lost to Preorder and
No Stock, which are exactly the queues top sellers are most often in. It is now an
independent boolean computed at `calc.py:472-483` as the smallest set of SKUs carrying 50% of
recent units.

**Worklist order**, `calc.py:555-559`: priority, then best seller, then order quantity, with
`mergesort` for stability because most Routine rows tie at zero.

**The client mirror is case-sensitive.** `PRIORITY` in `action-list-table.tsx:169-184` is a
dictionary keyed by those exact strings. `"No stock"` would silently fall through to the
Routine style rather than erroring.

### 2.6 Reliability tiers

`src/planning/reliability.py:34-53`. Per-SKU WAPE from
`outputs/reports/ml_accuracy_by_sku.csv`.

| Tier | Bound | Glyph |
|---|---|---|
| good | ≤ 0.15 | ●●● |
| fair | ≤ 0.30 | ●●○ |
| poor | above | ●○○ |
| none | unmeasured | ○○○ |

That CSV was re-run on 2026-08-14. See Section 3.7 for when it needs refreshing.

### 2.7 `error_basis` and the demand-band fallback

`calc.py:299-355`. Safety stock needs a per-SKU error, and most SKUs do not have a measured
one. The fallback chain, in order:

**own measured WAPE → demand-band median → segment median → overall median → 0.0**

`error_basis` reports which was used: `"measured"`, `"demand band"`, `"segment median"`,
`"overall median"`.

The bands are weekly units, left-closed, `calc.py:35`:

```python
ERROR_BAND_EDGES = [0.0, 2.0, 4.0, 6.0, 10.0, float("inf")]
MIN_BAND_MEASURED = 5
```

They are the bands the error curve was measured on, not round numbers: pooled WAPE is flat
within them and steps between them.

**Two details that are load-bearing.**

Banding is on **`recent_mean`**, the profiler's trailing **13-week** weekly mean, not on
`recent_units / 4`. The band edges were measured on a 13-week mean and every threshold in
`src/profile.py` uses the same window. Banding on 4 weeks put 11 SKUs below 2 units a week
where, by the rule that governs demotion, none are.

A thin band, under 5 measured SKUs, borrows from the **nearest trusted band along the volume
axis**, ties going to the lower-volume side. The first version fell through to the segment
median instead, and the [0,2) band, hardest to forecast at 0.357 and thinnest at 4 SKUs,
inherited a median dominated by high-volume SKUs. A guard meant to be cautious gave its 33
members **less** cushion, by 0.038.

`PROMOTED_ERROR_FALLBACK = 0.24`, a hardcoded constant, was removed 2026-08-12. The tombstone
is `calc.py:42-61`.

**Two frontend staleness bugs here.** `types.ts:89` still declares
`"measured" | "promoted cohort" | "segment median" | string` and
`reliability-card.tsx:71-75` still maps `"promoted cohort"`. Neither knows `"demand band"` or
`"overall median"`, so those fall through `basisLabel[errorBasis] ?? errorBasis` and render
the raw English string. Not a crash, but the **Korean locale shows untranslated text**.

### 2.8 The `forecast_runs_high` caveat

`calc.py:376-389`. Flags a SKU whose forecast is at least 1.5× its recent 4-week average
**and** at least 20 units above it. Both conditions required.

**Surfaced, never acted on.** The recommended quantity still uses the model. The callout is
placed **above** the order card on the detail page on purpose, so a reader sees the caveat
before the number.

### 2.9 The Not-forecast section

`calc.py:663-795`. Intermittent SKUs are about 87% of the catalogue and a fifth of recent
volume, so leaving them off the screen entirely was not honest.

**Membership is defined by absence from the planning table, not from the forecast file**
(`calc.py:711-718`). Keying on the forecast file let fifteen SKUs demoted since the run fall
through both sections. Keying on the planning table makes the two sections a partition by
construction.

**A different basis, with different column names on purpose.** No `recommended_order_qty`, no
coverage demand, no safety stock, no WAPE, no stockout date. None can exist without a
forecast, and absence is the honest answer. Instead: a trailing 13-week rate, days of cover
from it, and a boolean `reorder_signal` when cover is below the lead time. The table headings
are "13w demand", "per week", "cover", which appear nowhere else, so a reader cannot mistake
one for a scored forecast figure.

Two smaller decisions worth keeping: `days_of_cover` is `NaN` rather than infinity when
nothing has sold, because dividing by zero would read on screen as "never runs out" when the
question does not apply. And inventory fields are left **blank rather than zeroed** when
absent, because "no record" is not "record showing none".

It is fetched **only when the section is opened**, because it covers roughly seven times as
many SKUs.

### 2.10 The Run Forecast button

`run-forecast.tsx:67`, collapsed by default, directly under the provenance bar.

`POST /api/planning/run-forecast?horizon=N` → FastAPI `api/main.py:185-240` → a background
thread running `scripts/ml_prepare_data.py --force --horizon N` in its own process group.
`create_job("forecast")` shares a job type with `/planning/prepare-data`, so a concurrent
request gets **409**; they write the same files.

Status polls `GET /api/forecast/status/{jobId}` every **2 seconds**, keyed on `jobId` and
`status` so it stops itself.

**The progress bar is an undeclared contract.** `run-forecast.tsx:61`:

```ts
const m = /Step (\d)\/4/.exec(line);
if (m) seen = Math.max(seen, Number(m[1]));
```

It regexes the script's own stdout. Renaming those prefixes in `ml_prepare_data.py` silently
breaks the bar. `Math.max` rather than a count, so a line mentioning an earlier step cannot
walk progress backwards.

On success only, the list refetches so `trained_through` moves. **Deliberately not on
failure**, because refetching after a failure would show unchanged numbers as though the
failure had been applied.

**There is no cancel button**, though `/cancel-forecast/{job_id}` still exists. Stopping this
pipeline partway is worse than letting it finish.

### 2.11 The detail page

`/planning/action-list/[sku]`. Order card rendered **as arithmetic** rather than as a number,
reliability card with the per-window backtest table, stat row, and two charts.

For a SKU the model does not forecast, `/planning/sku/{id}` returns 404 with **three distinct
detail messages** (`api/main.py:502-520`): in the forecast run but demoted, profiled but never
forecast, or unknown SKU. The page then falls back to `/planning/sku/[sku]/history` and shows
sales history only.

`/planning/sku/{id}` also returns the full SKU list and the row's position in it, about 10 KB,
to avoid a second request for Prev/Next.

**There is no plausible band any more.** `api/main.py:631-635` records why: it flexed coverage
demand by the same error that safety stock adds, so its upper edge *was* the recommendation.

---

## 3. Forecast Validation

**Purpose.** Let someone decide whether to trust the forecast. It is six sections of evidence
read top to bottom, and the order is the argument.

**Route.** `src/app/planning/forecast-validation/page.tsx`.

**The components live in `src/components/planning/validation/`, not `forecast-validation/`.**
That mismatch is the first thing to know.

### 3.1 The sections, in render order

Numbering comes from `VALIDATION_SECTIONS` (`section-heading.tsx:34-41`); render order is
`validation-content.tsx:214-297`. They agree, and **they must**: a heading numbered 03 sitting
fourth is worse than no numbering at all. Adding a section means adding it to that array in
its place.

| # | id | Title | Claim |
|---|---|---|---|
| 01 | `comparison` | Model versus spreadsheet | The model beats V1 on pooled WAPE across the same windows, **with every losing cell shown** |
| 02 | `demand` | How demand is shaped | The scope of that claim: which SKUs the model speaks for and how much volume they carry. Nothing here is a forecast |
| 03 | `trajectory` | Demand vs forecast | The same evidence drawn over time: when the error happened and in which direction |
| 04 | `over-time` | Performance on forecasts actually served | The out-of-sample record: forecasts issued before the outcome was known |
| 05 | `outliers` | SKU-level breakdown | Whether the pooled improvement is broad or carried by a few large wins |
| 06 | `final-test` | Final test window | What is deliberately not claimed yet |

**Two conditional renders that will confuse you.** Section 03 is skipped entirely when the
trend fetch fails or returns no predictions, with no error card, deliberately. Section 04 is
suppressed when the history store is empty, to avoid two panels saying the same thing.

**The design standard this page holds itself to**, from `comparison-section.tsx:1-14`:

> The headline is one number, and one number is exactly what invites the wrong conclusion, so
> the grid underneath is not optional detail. It shows every segment and every window
> including the cells the spreadsheet still wins, because **a comparison that only reports its
> wins is not evidence.**

Model versions are read from the payload rather than named in the components, so a new model
appears without any frontend file changing.

### 3.2 Data provenance per section

| Section | Source | Computed | Clock |
|---|---|---|---|
| 01, 05 | `outputs/reports/ml_accuracy.csv`, `ml_accuracy_by_sku.csv` | **Stored**, by `scripts/ml_accuracy_report.py` | **Pinned** |
| 02 | `load_sales()` | Live, on request | **Live** |
| 03, 04 | `src/ml/serving/history.py` | Live, from the accumulating store | **Live** |
| 06 | `outputs/reports/final_test.json` | Stored | **Pinned** |

**The clock column is the thing to understand about this page.** Live sections read
`data/processed`, which the Tuesday cron rewrites, and are supposed to move every week.
Pinned sections read the snapshot named by `ML_DATA_SNAPSHOT` and are supposed not to,
because their value is being comparable across model versions. Both kinds sat on the page
with nothing distinguishing them until 2026-08-14, so a reader had to assume one or the
other and either assumption was wrong for half the page.

`basis` on the response says which, per section, and `demand-patterns` carries its own
`basis` so section 02 self-dates without reference to the other payload.

`meta.accuracy_computed` is the `run_at` recorded in `ml_accuracy_meta.json`. It was the
**file mtime** of `ml_accuracy.csv` until 2026-08-14, which describes the filesystem rather
than the measurement: `git checkout`, `cp -r` and every deploy rewrite it. When the manifest
is absent the mtime is still used, and `basis.accuracy.computed_at_is_mtime` says so, which
the page renders as a caveat rather than presenting an inferred date as a recorded one.

### The drift check

`src/planning/provenance.py`. The pinned sections are only trustworthy while the snapshot
they were measured on still resembles what is being served, and nothing could express that
before.

Two conditions, reported separately because they have different detection:

| Flag | Condition | Caught by a name comparison? |
|---|---|---|
| `snapshot_stale` | The report's snapshot is not `ML_DATA_SNAPSHOT` | Yes |
| `population_stale` | The forecastable cohort has moved more than 5% from what was scored | **No** |

The second is the one that matters, because a snapshot can be re-cut in place with a
re-profiled population under the same name. That is exactly what happened on 2026-08-11.

**The drift measure excludes the intermittent tail, and that is load-bearing.** Intermittent
SKUs are 87% of the catalogue and are never forecast or scored. Measured over the whole
catalogue the 2026-08-11 re-profile reads **3.8%** and sits under any tolerance loose enough
to be usable; over the forecastable cohort the same event reads **42.1%**, against the 41%
recorded in Section 3.7. A check that dilutes its own signal below its own threshold is
worse than no check, because it reports all clear.

Movement is **not** halved. Halving is right within a closed set and this set is not closed:
a SKU demoted out of smooth leaves the cohort rather than swapping within it.

Reported, never repaired. `/health` carries it, the cron prints it as a warning rather than
a failure, and the page shows a banner. Nothing re-runs the report automatically, for the
reason in Section 3.7.

`comparison.grid` carries **model versions as dynamic keys** per `(segment, window)` cell,
which is why `ValidationCell` has an index signature. `windows` is sorted chronologically by
cutoff rather than by name, because alphabetical order reads Dec-Feb, Mar-May, Oct-Dec and
invites a backwards trend reading.

`outliers.rows` sends the **whole** scored pool, roughly 572 rows; `top_n` is how many the
page displays, not how many are transferred.

### 3.3 The charts

**Plotly**, via `react-plotly.js`, loaded client-only with `dynamic(..., { ssr: false })`. No
Recharts on this page.

| Chart | Plots |
|---|---|
| Weekly demand, stacked area | Forecast SKUs against the intermittent tail |
| Pareto concentration curve | Cumulative demand against cumulative SKUs, with an even-demand reference line |
| Demand vs forecast trajectory | Actual weekly units, stored-run predictions at a selectable lead, forward horizon, V1 |

Two things that look like charts and are **not** Plotly: the per-SKU delta histogram in
`outliers-section.tsx` is CSS-sized divs clipped at ±1 with labelled overflow buckets, and the
per-run error trend in `over-time-section.tsx` is a bar drawn inside a table cell.

Two absences the charts explain rather than hide: **no prediction interval**, because v11
emits a point forecast only (`has_intervals: false`); and **V1 on the forward horizon only**,
because the history store keeps the model's own predictions while V1 is recomputed per run.

### 3.4 Internationalisation

**Not an i18n library.** A `pick(ko, en)` function from a React context,
`src/lib/i18n/i18n-provider.tsx:67-72`. Strings are inline at the call site, Korean first.

A keyed dictionary `t(key)` backed by `messages.ts` also exists, but **no file under
`validation/` uses it.**

```tsx
"use client";
import { useI18n } from "@/lib/i18n/i18n-provider";

export function MySection() {
  const { pick } = useI18n();
  return <h2>{pick("예측 검증", "Forecast Validation")}</h2>;
}
```

Interpolation duplicates the template inside **both** arguments rather than wrapping `pick`,
because Korean reorders the operands:

```tsx
{pick(
  `${headline.cells_total}개 구간 중 ${headline.cells_won}개에서 우세`,
  `ahead in ${headline.cells_won} of ${headline.cells_total} cells`,
)}
```

`pick` also supplies `title=` tooltips and Plotly trace `name:` fields, so translated text
reaches chart legends.

**Checklist for a new section:** `"use client"`, `useI18n()`, wrap every literal in `pick`,
pass translated `title` and `description` into `SectionHeading`, and add the section to
`VALIDATION_SECTIONS` **in its render position** with its `[ko, en]` label pair.

Locale resolves as: default `"en"`, then `localStorage["demandpilot-locale"]`, then
`GET /api/user/preferences` under `app.locale`.

### 3.5 The final test data

`outputs/reports/final_test.json`, written by `scripts/ml_41_final_test.py`, which refuses to
overwrite it.

```json
{
  "run_at": "2026-08-13T12:02:18-07:00",
  "commit": "4a19ca1d177bcb596d31af16b5fa818f1d458ecf",
  "snapshot": "2026-08-03-v2",
  "v1_orders_raw": { "path": "…/orders_raw.parquet",
                     "md5": "fd90514306d8601e126700262ab02c8c" },
  "cutoff": "2026-05-04",
  "test_weeks": ["2026-05-11", … , "2026-07-13"],
  "scores": {
    "v11":      { "smooth/long": 0.1324, "smooth/short": 0.2061, "TOTAL": 0.1784 },
    "V1":       { "smooth/long": 0.1872, "smooth/short": 0.3772, "TOTAL": 0.3059 },
    "baseline": { "smooth/long": 0.1282, "smooth/short": 0.2013, "TOTAL": 0.1739 }
  },
  "v11_vs_v1":       { "short": {"delta": -0.1711, "se": 0.0215, …},
                       "long":  {"delta": -0.0548, "se": 0.0174, …} },
  "v11_vs_baseline": { "short": {"delta":  0.0048, "se": 0.0108, …},
                       "long":  {"delta":  0.0042, "se": 0.0141, …} }
}
```

**This file is tracked, as of 2026-08-14.** `.gitignore:38` excludes `outputs/reports/*` and
the lines below it re-include specific files by name. `final_test.json` was not among them
until now, so the only copy lived on the machine that ran it. That mattered more than for the
other reports: the test is single-use and `ml_41_final_test.py` refuses to overwrite, so
re-running is not a recovery path, and losing the file would have taken the commit, the input
md5 and the bootstrap intervals with it. `BACKLOG.md` says "the file is committed"; that was
wrong when written and is true now.

---

### 3.6 The final test panel

**Fixed 2026-08-14. BACKLOG item 30, which had been closed without the code being changed.**

Until then `api/main.py` hardcoded `"evaluated": False` with a comment reading "Pinned,
quarantined, and not yet run", and `validation-content.tsx` rendered an `EmptySection` on
**both** branches. The page therefore told readers "Not evaluated yet, deliberately", which
had been false since 2026-08-13. It was a false statement in the last section of the one
screen whose purpose is to let someone decide whether to trust the forecast, and it was the
section carrying the strongest evidence the project has.

**How it works now.**

`_final_test_payload()` in `api/main.py` reads `outputs/reports/final_test.json` and passes
`scores` and the provenance fields through **unchanged**, so a figure on the page and a figure
in that file cannot disagree without the file itself being wrong. Two things are derived
rather than passed through, both so the web app never knows a model version by name:

- **`methods`** names which key in `scores` is the model, which is the spreadsheet and which
  is the structural baseline. Roles are resolved by elimination: `V1` and `baseline` are fixed
  names the runner writes, and whatever else is present is the model under test.
- **`comparisons`** flattens the `<model>_vs_<other>` blocks into a list, each entry carrying
  what it compares. In the file those are keys containing a version name, which a TypeScript
  interface cannot describe without hardcoding it.

A missing or unreadable file returns `evaluated: false`, matching the endpoint's posture
elsewhere: a section with no evidence renders as "nothing here yet" rather than taking the
page down. The unevaluated copy now says the checkout is missing the file, which is the
honest reading on a fresh clone, rather than saying the test is pending.

`types.ts` carries `FinalTest` as a **discriminated union** rather than optional fields, so
the unevaluated case cannot be rendered with half a result. `cutoff` is in both arms because
the section description and `ModelCard` read it either way.

`final-test-section.tsx` renders it. **The constraint the component exists to satisfy** is
that the result has two halves and both must carry the same weight: the model beats the
spreadsheet by a wide margin, and it ties the structural baseline on this same window. The
two verdict panels are the same size, side by side, and neither is a footnote to the other.
Rendering only the first is the failure mode `comparison-section.tsx` refuses in its own code
guide, and it applies harder here because the section description tells the reader to weight
this above everything above it.

Significance is computed in the component from whether the 95% interval excludes zero, rather
than passed in as a flag. That distinction is the entire content of the second panel: a delta
whose interval straddles zero is a reading, not a result.

The headline cards deliberately mirror the three in `comparison-section.tsx`, so a reader who
has scrolled past that grid recognises the shape and the only thing that feels different is
which window it is.

**What is deliberately not rendered.** Calibration. The runner records pooled WAPE and the
bootstrap only, so bias for this window is not in the result file; it is in
`ML_FORECAST_DESIGN.md` Section 4.35. The payload carries `has_bias: false` and the panel says
where the figures are rather than restating numbers it cannot read from the measurement. If
they are wanted on screen, `ml_41_final_test.py` should record them and the payload passes
them through like everything else.

**Kept unchanged:** the section description above the panel, which explains why the window was
quarantined and why that makes its figure worth more than the ones above it. That is still
true after the test has run, and it is what stops the result being read as one more backtest.

**One thing that went stale with this change and was not fixed.**
`section-heading.tsx:12-15` describes the page order as ending on "what is deliberately not
claimed yet". The last section is now the strongest claim on the page rather than the absence
of one. That same comment also lists a section order that already disagreed with the
`VALIDATION_SECTIONS` array 20 lines below it; the array is the one to trust.

### 3.7 The accuracy report refresh cycle

**Fixed 2026-08-14.** Re-run against snapshot `2026-08-03-v2`, committed, and pushed to the
server. The figures now match `OVERVIEW.md` Section 6.

**When to re-run it.** `scripts/ml_accuracy_report.py` needs refreshing when the snapshot is
re-cut or the profiler changes. It does **not** belong on the weekly cron, because it reads
the pinned snapshot: a weekly run would retrain three windows to rewrite identical bytes.
The trigger is a population change, which the cron cannot predict. Hence detection rather
than scheduling: the cron reports the condition via the drift check, and a human runs the
script.

**Why it is not automated.** The output is compared against the design doc at the third
decimal and quoted in the management proposal. A pass that regenerates it without anyone
deciding it should be regenerated moves published figures silently.

---

## 4. Known defects, collected

**Still open.**

| # | Where | What |
|---|---|---|
| 2 | `types.ts:89`, `reliability-card.tsx:71-75` | `error_basis` values out of date; Korean locale shows untranslated strings. Section 2.7 |
| 4 | `run-forecast.tsx:61` | Undeclared `Step N/4` stdout contract with `ml_prepare_data.py`. Section 2.10 |
| 5 | `SkuForecastsService.getForecastBounds()` | Orphaned when `/api/forecast/bounds` was deleted. Still has a passing test, so it is dead code that looks maintained |

None is a crash. None misleads a reader about accuracy.

**Fixed 2026-08-14.**

| Where | What |
|---|---|
| `api/main.py`, `validation-content.tsx`, `types.ts`, new `final-test-section.tsx` | The final test is served from its result file and rendered, both halves. Section 3.6 |
| `.gitignore` | `final_test.json` is tracked. Section 3.5 |
| `api/main.py` coverage block, new `src/planning/provenance.py`, new `section-basis.tsx` | Live and pinned figures are labelled, `coverage` carries a date on each side, and a superseded accuracy report is detected instead of being invisible. Section 3.2 |
| `ml_accuracy_report.py`, `.gitignore` | `ml_accuracy_meta.json` records what the report was measured on. `meta.accuracy_computed` is no longer a file mtime |
| `run_forecast_cron.sh` | The weekly run reports whether the accuracy report still matches the served population |
| `forecast-validation/page.tsx`, `section-heading.tsx` | Code guides left stale by the final-test change above. Was defect 3 |
| `outputs/reports/ml_accuracy*.csv` | Re-run against snapshot `2026-08-03-v2`. Was defect 1. Section 3.7 |
| `column-picker.tsx` | The Columns panel was `z-30`, which tied the sticky column-name row and lost to the sticky corner cell, so it opened underneath the headers it overlaps. Now `z-50` |
