# Screens: Action List and Forecast Validation

Maintainer reference: what each part displays, which endpoint serves it, where each figure is computed.

**Architectural rule.** Every figure on both screens is computed in Python, in `Time_Series_Forecasting/src/planning/`, and proxied through thin route handlers.

**Rationale.** One implementation of the order formula, testable without a browser. Preserve it.

## 1. Request path

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

Route handlers hold no validation. FastAPI's `Query(ge=…, le=…)` is the source of bounds; `types.ts:171-185` clamps to the same.

### Proxy error taxonomy

`planning-api.ts:63-159`.

| Condition | Result |
|---|---|
| Connection failure | `ensureForecastServer()`, then one retry. A local server is auto-started; a remote one gets a 750 ms pause. Then `{kind:"unreachable"}` 503 |
| Upstream 404, not a `/planning/sku/` path | `{kind:"outdated"}`, "the forecast server is running an older revision" |
| Upstream 500 or above | Calls `forecastHealth()`; if `ready === false`, `{kind:"no_data"}` 503 naming the missing files |
| Anything else | Status passed through verbatim, so a 404 from the SKU endpoint keeps its meaning |

Auto-start causes the diagnostic trap in `DATA_AND_PIPELINE.md` §9.

### Endpoints

| Next route | Upstream | Timeout |
|---|---|---|
| `GET /api/planning/action-list` | `/planning/action-list` | 20 s |
| `GET /api/planning/sku/[sku]` | `/planning/sku/{id}` | 20 s |
| `GET /api/planning/sku/[sku]/history` | `/planning/sku/{id}/history` | 20 s |
| `GET /api/planning/not-forecast` | `/planning/not-forecast` | 40 s, roughly 7× the SKUs |
| `GET /api/planning/validation` | `/planning/validation` | 60 s, scoring stored runs loads full history on a cold server |
| `GET /api/planning/demand-patterns` | `/planning/demand-patterns` | 30 s |
| `GET /api/planning/demand-vs-forecast` | `/planning/demand-vs-forecast` | 30 s |
| `POST /api/planning/demand-trend` | `/planning/demand-trend` | 30 s, not via the shared helper, which is GET-only |
| `POST /api/planning/run-forecast` | `/planning/run-forecast` | 30 s |
| `GET /api/forecast/status/[jobId]` | `/forecast-status/{job_id}` | 5 s, also not via the helper |

`/api/forecast/status` is the only survivor of the fourteen `/api/forecast/*` proxy routes deleted on 2026-08-13.

## 2. Action List

**Purpose.** Turn the forecast into a worklist: which SKUs to order, how many units, in what order.

**Routes.** `src/app/planning/action-list/page.tsx` (list), `.../[sku]/page.tsx` (detail). The SKU is in the path, so rows are shareable.

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
    ├── summary filter buttons           7 clickable counts; the only priority filter
    ├── PlanningControls                 lead / review / service-z / risk window
    ├── search, category/tier/trend selects, Reset
    ├── ColumnPicker                     which optional columns are visible
    ├── Export CSV                       csv-export.ts
    ├── PortfolioChart                   collapsed; actual + forecast over filtered SKUs
    ├── data-quality warning line        counts of r.flags, click to filter
    ├── ActionListTable                  action-list-table.tsx
    │   └── ColumnHeaderMenu             sort + per-column checkbox filter
    └── PlanningError                    error card with retry
```

`rememberSkuSequence()` (`sku-sequence.ts:37`) writes the filtered, sorted order to `sessionStorage` for detail-page Prev/Next.

`DEFAULT_SORT = []` (`action-list-table.tsx:254`) disables client sort.

**Rationale.** The server's worklist order is not reproducible from any column.

### 2.2 Recommendation formula

`src/planning/calc.py:410-427`.

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

Components round to whole units **before** the total, so the detail-page breakdown sums exactly to the quantity. `safety_stock` uses unrounded coverage demand, rounded one line later.

| Input | Source | Note |
|---|---|---|
| `coverage_demand` | `_coverage_demand`, `calc.py:97-110` | Sum of `yhat` over the first `coverage_weeks` forecast weeks. If the horizon is shorter, the shortfall is padded at the horizon's mean weekly rate |
| `available_inventory` | `SUM(available)` = on_hand − allocated | **Not** net of backorder |
| `preorder_backlog` | `SUM(backorder)`, same table | **Added** to the requirement |
| `inbound_in_window` | `calc.py:398-408` | Confirmed statuses only, ETA on or before `coverage_end`. Inbound with no ETA is never credited |
| `safety_stock` | `calc.py:415` | `z × error_used × coverage_demand`. The SKU's own measured WAPE times the demand it must cover, not a flat days-of-cover rule |
| `lead_time_weeks` | Parameter | Enters only through `coverage_weeks` and `coverage_end` |
| Drafts | `calc.py:238-240`, `types.ts:22-34` | **Never subtracted.** A draft is not a commitment, so crediting it would under-order exactly the SKUs someone has already acted on |

No lot size, MOQ or container rounding exists anywhere.

Defaults, `calc.py:63-83`, mirrored in `types.ts:142-147`:

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

### 2.3 The breakdown

`calc.py:612-660` builds the detail-page arithmetic. Each line carries a `Sign`: `+1` add, `−1` subtract, `0` total, `None` informational. Stored `recommended_order_qty` is preferred over recomputing.

### 2.4 Days of cover and projected stockout date

`_days_until_consumed`, `calc.py:124-146`. Walks the forecast curve weekly: linear interpolation inside the week stock runs out, the horizon's mean rate past the end. Zero-forecast weeks are skipped.

Applied at `calc.py:435-457`:

```python
in_time = inb > 0 and np.isfinite(eta) and eta <= t_on_hand
```

Inbound extends cover only if it lands before the shelf empties. `recommended_order_qty` credits any inbound in the coverage window; `calc.py:528-544` sets `supply_gap_days` to the difference and `gap_closable_by_order` to whether ordering now helps.

### 2.5 Priority ladder

`calc.py:502-511`. Three states, first match wins, lowest number first.

| Rank | Label | Condition |
|---|---|---|
| 1 | Preorder | `preorder_backlog > 0` |
| 2 | No Stock | `available_inventory <= 0` |
| 99 | Routine | Everything else |

Rank 3 "Best Seller" was removed at `calc.py:485-501` and is now an independent boolean at `calc.py:472-483`, the smallest set of SKUs carrying 50% of recent units.

Worklist order, `calc.py:555-559`: priority, best seller, order quantity; `mergesort` for stability.

Warning: the client mirror is case-sensitive. `PRIORITY` in `action-list-table.tsx:169-184` is keyed by those exact strings; `"No stock"` falls through silently to the Routine style.

### 2.6 Reliability tiers

`src/planning/reliability.py:34-53`. Per-SKU WAPE from `outputs/reports/ml_accuracy_by_sku.csv`, re-run 2026-08-14. Refresh cycle in §3.7.

| Tier | Bound | Glyph |
|---|---|---|
| good | ≤ 0.15 | ●●● |
| fair | ≤ 0.30 | ●●○ |
| poor | Above | ●○○ |
| none | Unmeasured | ○○○ |

### 2.7 `error_basis` and the demand-band fallback

`calc.py:299-355`. Most SKUs lack a measured error. `error_basis` names the step that supplied the one used.

| Order | Source | `error_basis` |
|---|---|---|
| 1 | Own measured WAPE | `"measured"` |
| 2 | Demand-band median | `"demand band"` |
| 3 | Segment median | `"segment median"` |
| 4 | Overall median | `"overall median"` |
| 5 | 0.0 | none |

Bands are weekly units, left-closed, `calc.py:35`:

```python
ERROR_BAND_EDGES = [0.0, 2.0, 4.0, 6.0, 10.0, float("inf")]
MIN_BAND_MEASURED = 5
```

**Rationale.** The edges are where the error curve was measured. Pooled WAPE is flat within a band and steps between bands.

Warning: banding is on `recent_mean`, the profiler's trailing 13-week weekly mean, not `recent_units / 4`; every threshold in `src/profile.py` uses that window. Banding on 4 weeks put 11 SKUs below 2 units a week where the demotion rule puts none.

A band under 5 measured SKUs borrows from the nearest trusted band along the volume axis, ties to the lower-volume side.

**Rationale.** The segment median cut the [0,2) band's 33 members' cushion by 0.038. That band is hardest to forecast at 0.357 and thinnest at 4 SKUs.

`PROMOTED_ERROR_FALLBACK = 0.24` was removed 2026-08-12. Tombstone at `calc.py:42-61`.

**Open defect.** `types.ts:89` declares `"measured" | "promoted cohort" | "segment median" | string`; `reliability-card.tsx:71-75` maps `"promoted cohort"`. `"demand band"` and `"overall median"` fall through `basisLabel[errorBasis] ?? errorBasis` as untranslated English in the Korean locale.

### 2.8 The `forecast_runs_high` caveat

`calc.py:376-389`. Flags a SKU whose forecast is at least 1.5× its recent 4-week average **and** at least 20 units above it. Both are required.

Surfaced, never acted on: the quantity still uses the model. The callout sits above the order card.

### 2.9 Not-forecast section

`calc.py:663-795`. Intermittent SKUs, about 87% of the catalogue and a fifth of recent volume, are shown. Membership is absence from the planning table, not from the forecast file (`calc.py:711-718`).

**Rationale.** Keying on the forecast file let fifteen SKUs demoted since the run fall through both sections. The planning table makes the sections a partition.

No `recommended_order_qty`, coverage demand, safety stock, WAPE or stockout date exists without a forecast. The section carries a trailing 13-week rate, days of cover from it, and `reorder_signal` when cover is below the lead time. The headings "13w demand", "per week", "cover" are used nowhere else.

| Case | Convention |
|---|---|
| Nothing has sold | `days_of_cover` is `NaN`, not infinity; zero-division would read as "never runs out" |
| Inventory record absent | Fields are blank, not zeroed: "no record" is not "record showing none" |

Fetched only on opening the section: roughly seven times as many SKUs.

### 2.10 Run Forecast button

`run-forecast.tsx:67`, collapsed by default, directly under the provenance bar.

`POST /api/planning/run-forecast?horizon=N` → FastAPI `api/main.py:185-240` → a background thread running `scripts/ml_prepare_data.py --force --horizon N` in its own process group.

| Behaviour | Detail |
|---|---|
| Job type | `create_job("forecast")` shares a type with `/planning/prepare-data`, so a concurrent request gets **409**. They write the same files |
| Polling | `GET /api/forecast/status/{jobId}` every 2 seconds, keyed on `jobId` and `status`, so it stops itself |
| Refetch | On success only, so `trained_through` moves. On failure the list would show unchanged numbers as though the failure had applied |
| Cancel | No button, though `/cancel-forecast/{job_id}` still exists |

Warning: the progress bar is an undeclared contract. `run-forecast.tsx:61`:

```ts
const m = /Step (\d)\/4/.exec(line);
if (m) seen = Math.max(seen, Number(m[1]));
```

It regexes the script's own stdout; renaming those prefixes in `ml_prepare_data.py` silently breaks the bar. `Math.max`, not a count, so an earlier step cannot walk progress backwards.

### 2.11 Detail page

`/planning/action-list/[sku]`. Order card as arithmetic; reliability card with the per-window backtest table; stat row; two charts.

For an unforecast SKU, `/planning/sku/{id}` returns 404 with three distinct detail messages (`api/main.py:502-520`): in the forecast run but demoted; profiled but never forecast; unknown SKU. The page falls back to `/planning/sku/[sku]/history`, sales history only.

`/planning/sku/{id}` also returns the full SKU list and the row's position, about 10 KB, sparing a second request for Prev/Next.

No plausible band exists. `api/main.py:631-635` records why: it flexed coverage demand by the same error safety stock adds, so its upper edge was the recommendation.

## 3. Forecast Validation

**Purpose.** Decide whether to trust the forecast. Six sections of evidence read top to bottom; the order is the argument.

**Route.** `src/app/planning/forecast-validation/page.tsx`. Components live in `src/components/planning/validation/`, not `forecast-validation/`.

### 3.1 Sections in render order

Numbering comes from `VALIDATION_SECTIONS` (`section-heading.tsx:34-41`), render order from `validation-content.tsx:214-297`. A new section joins that array in its render position.

| # | id | Title | Claim |
|---|---|---|---|
| 01 | `comparison` | Model versus spreadsheet | The model beats V1 on pooled WAPE across the same windows, with every losing cell shown |
| 02 | `demand` | How demand is shaped | Scope: which SKUs the model speaks for and how much volume they carry. Nothing here is a forecast |
| 03 | `trajectory` | Demand vs forecast | The same evidence over time: when the error happened and in which direction |
| 04 | `over-time` | Performance on forecasts actually served | Forecasts issued before the outcome was known |
| 05 | `outliers` | SKU-level breakdown | Whether the pooled improvement is broad or carried by a few large wins |
| 06 | `final-test` | Final test window | The quarantined result |

| Conditional render | Condition |
|---|---|
| 03 skipped, no error card | The trend fetch fails or returns no predictions |
| 04 suppressed | The history store is empty, which would give two panels saying the same |

**Design standard**, `comparison-section.tsx:1-14`: the grid shows every segment and every window, including the cells the spreadsheet wins.

Model versions are read from the payload, so a new model needs no frontend change.

### 3.2 Data provenance per section

| Section | Source | Computed | Clock |
|---|---|---|---|
| 01, 05 | `ml_accuracy.csv`, `ml_accuracy_by_sku.csv` | Stored, by `scripts/ml_accuracy_report.py` | Pinned |
| 02 | `load_sales()` | Live, on request | Live |
| 03, 04 | `src/ml/serving/history.py` | Live, from the accumulating store | Live |
| 06 | `outputs/reports/final_test.json` | Stored | Pinned |

| Clock | Reads | Expected behaviour |
|---|---|---|
| Live | `data/processed`, rewritten by the Tuesday cron | Moves every week |
| Pinned | The snapshot named by `ML_DATA_SNAPSHOT` | Does not move; its value is comparability across model versions |

`basis` on the response says which, per section; `demand-patterns` carries its own, so section 02 self-dates. `meta.accuracy_computed` is the `run_at` in `ml_accuracy_meta.json`. Absent the manifest, the file mtime is used and `basis.accuracy.computed_at_is_mtime` says so, rendered as a caveat.

### 3.3 The drift check

`src/planning/provenance.py`. Pinned sections hold only while the snapshot they were measured on resembles what is served.

| Flag | Condition | Caught by name comparison? |
|---|---|---|
| `snapshot_stale` | The report's snapshot is not `ML_DATA_SNAPSHOT` | Yes |
| `population_stale` | The forecastable cohort has moved more than 5% from what was scored | No |

`population_stale` covers a snapshot re-cut in place with a re-profiled population under the same name, as on 2026-08-11.

Warning: the drift measure excludes the intermittent tail, 87% of the catalogue, never forecast or scored. The 2026-08-11 re-profile reads 3.8% over the whole catalogue and 42.1% over the forecastable cohort.

Movement is not halved: a SKU demoted out of smooth leaves the cohort instead of swapping within it.

Drift is reported, never repaired: `/health` carries it, the cron prints a warning, the page shows a banner. See §3.7.

| Payload field | Detail |
|---|---|
| `comparison.grid` | Model versions as dynamic keys per `(segment, window)` cell, hence the index signature on `ValidationCell` |
| `windows` | Sorted chronologically by cutoff. Alphabetical order reads Dec-Feb, Mar-May, Oct-Dec and invites a backwards trend reading |
| `outliers.rows` | The whole scored pool, roughly 572 rows. `top_n` is how many the page displays, not how many transfer |

### 3.4 Charts

| Chart | Plots |
|---|---|
| Weekly demand, stacked area | Forecast SKUs against the intermittent tail |
| Pareto concentration curve | Cumulative demand against cumulative SKUs, with an even-demand reference line |
| Demand vs forecast trajectory | Actual weekly units, stored-run predictions at a selectable lead, forward horizon, V1 |

Plotly is used for the three above. Two further elements look like charts and are not:

| Element | Implementation |
|---|---|
| Per-SKU delta histogram, `outliers-section.tsx` | CSS-sized divs clipped at ±1 with labelled overflow buckets |
| Per-run error trend, `over-time-section.tsx` | A bar drawn inside a table cell |

Absences:

| Absence | Cause |
|---|---|
| No prediction interval | v11 emits a point forecast only (`has_intervals: false`) |
| V1 on the forward horizon only | The history store keeps the model's predictions; V1 is recomputed per run |

### 3.5 Internationalisation

Strings on both pages use `pick(ko, en)`, Korean first. No file under `validation/` uses the keyed `t(key)` dictionary.

Checklist for a new validation section:

1. `"use client"`
2. `useI18n()`
3. Wrap every literal in `pick`
4. Pass translated `title` and `description` into `SectionHeading`
5. Add the section to `VALIDATION_SECTIONS` in its render position with its `[ko, en]` label pair

Note: interpolation duplicates the template inside both arguments, since Korean reorders the operands. `pick` also supplies `title=` tooltips and Plotly trace `name:` fields, so translated text reaches chart legends.

### 3.6 Final test data and panel

`outputs/reports/final_test.json`, written by `scripts/ml_41_final_test.py`, which refuses to overwrite.

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

Tracked as of 2026-08-14. `.gitignore:38` excludes `outputs/reports/*`, with the lines below re-including specific files by name. The test is single-use: re-running is not a recovery path.

`_final_test_payload()` in `api/main.py` passes `scores` and the provenance fields through unchanged. Two fields are derived, so the web app never knows a model version by name.

| Derived field | Content |
|---|---|
| `methods` | Which key in `scores` is the model, the spreadsheet, the structural baseline. Roles resolve by elimination: `V1` and `baseline` are fixed names the runner writes; anything else is the model under test |
| `comparisons` | The `<model>_vs_<other>` blocks flattened into a list, each entry carrying what it compares. In the file those are keys containing a version name, undescribable in TypeScript without hardcoding it |

A missing or unreadable file returns `evaluated: false` and the section renders as "nothing here yet". `types.ts` carries `FinalTest` as a discriminated union, so the unevaluated case cannot render half a result. `cutoff` is in both arms, read by the section description and `ModelCard`.

`final-test-section.tsx` renders two verdict panels of equal size, side by side. Significance is computed in the component from whether the 95% interval excludes zero, not passed as a flag.

Calibration is not rendered. The runner records pooled WAPE and the bootstrap only; bias for this window is in `ML_FORECAST_DESIGN.md` §4.35. The payload carries `has_bias: false` and the panel says where the figures are. Rendering them requires `ml_41_final_test.py` to record them.

**Known stale comment.** `section-heading.tsx:12-15` describes the page as ending on "what is deliberately not claimed yet", no longer true, and lists a section order disagreeing with the `VALIDATION_SECTIONS` array 20 lines below. The array is authoritative.

### 3.7 Accuracy report refresh cycle

`scripts/ml_accuracy_report.py` needs refreshing when the snapshot is re-cut or the profiler changes. Last run 2026-08-14 against snapshot `2026-08-03-v2`.

**Rationale.** Not on the weekly cron: it reads the pinned snapshot, so a weekly run would retrain three windows to rewrite identical bytes. The trigger is a population change, which the cron cannot predict and the drift check reports.

**Rationale.** Not automated at all: the output is compared against the design doc at the third decimal and quoted in the management proposal. Regenerating it without a decision moves published figures silently.

## 4. Open defects

None is a crash. None misleads a reader about accuracy.

| # | Where | What |
|---|---|---|
| 2 | `types.ts:89`, `reliability-card.tsx:71-75` | `error_basis` values out of date; Korean locale shows untranslated strings. §2.7 |
| 4 | `run-forecast.tsx:61` | Undeclared `Step N/4` stdout contract with `ml_prepare_data.py`. §2.10 |
| 5 | `SkuForecastsService.getForecastBounds()` | Orphaned when `/api/forecast/bounds` was deleted. Still has a passing test, so it is dead code that looks maintained |
