# The V1 spreadsheet formula, as this project reimplements it

Written to be compared line by line against the spreadsheet it was derived from.
Every constant and every edge rule is stated, because a difference in any of them
changes the comparison this project's headline result rests on.

Source of truth in this repo: `scripts/compare_v1.py`. The serving path that
writes `v1_forward_forecasts` is `src/ml/serving/v1.py` and calls the same
functions.

Verified 2026-08-04 to reproduce the stored figure exactly for
`CA-SC-10-R-90-DG-1TO`: daily rate 1.385333, weekly 9.697333, matching
`v1_forward_forecasts.parquet` to six decimal places.

---

## 1. Input data

One table: `shipcore.fc_velocity_link_snapshot_forecast`, which is the unbounded
order history. Columns used: `order_date`, `link_master_sku`, `link_qty`,
`order_type`, `channel`.

Note this is **not** the 120-day-capped `fc_velocity_link_snapshot`. Using the
capped one truncates the windows below.

### Streams

Every order line is assigned to exactly one of five streams. The check on
channel happens **first**, so an Amazon FBA line never lands in west or east
regardless of its `order_type`.

| condition | stream |
|---|---|
| `channel = 'Amazon FBA'` | `fba` |
| `order_type = 'sales'` | `west_sales` |
| `order_type = 'preorder'` | `west_preorder` |
| `order_type = 'ttm'` | `east_sales` |
| `order_type = 'ttm_preorder'` | `east_preorder` |
| anything else | **excluded entirely** |

Quantities are `link_qty`, summed per SKU per stream per calendar day.

---

## 2. Constants

### Blend windows

Six windows, identical for the west and east streams. Weights sum to 1.00.

| lookback (days) | weight | stream |
|---|---|---|
| 90 | 0.10 | sales |
| 60 | 0.15 | sales |
| 30 | 0.30 | sales |
| 15 | 0.20 | sales |
| 7 | 0.15 | sales |
| 30 | 0.10 | **preorder** |

Preorder enters at 0.10 only, through a single 30-day window. Sales enters at
0.90 across five windows.

### Monthly seasonal multipliers

| Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.75 | 0.80 | 0.90 | 0.95 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.10 | 1.25 | 1.30 |

### Horizon

`HORIZON_DAYS = 70` for the native 70-day total. The weekly serving path uses 7
days per week instead; see section 6.

---

## 3. Window sum

For a SKU, a stream, an end date and a window length in days:

```
window_sum(stream, end, days) = max(0, cumsum[end] − cumsum[end − days])
```

where `cumsum[d]` is total units in that stream on or before day `d`, on a daily
grid with zero-filled gaps, and lookups use the last known value at or before the
date.

The interval is **(end − days, end]**: the end date is included, the start date
is not. `days` is calendar days, not trading days.

A SKU with no rows in a stream returns 0.

---

## 4. Blend rate, per stream group

For `prefix` in {`west`, `east`}, at an as-of date:

```
blend(prefix, as_of) = max(0, Σ over the six windows of
                              weight × window_sum(stream, as_of, days) / days)
```

where `stream` is `{prefix}_sales` for the five sales windows and
`{prefix}_preorder` for the sixth.

Each window contributes a **per-day rate** — units divided by the window length —
not a unit total. So a 90-day window holding 132 units contributes
`0.10 × 132/90 = 0.1467` units/day.

---

## 5. Damping against last week

Each of west and east is damped toward its value one week earlier. Let

- `S` = `blend(prefix, cutoff)` — this week
- `R` = `blend(prefix, cutoff − 7 days)` — a week ago

```
if R == 0:                       result = S
else:
    change = |S − R| / R
    if change < 0.5:             result = 0.10 × R + 0.90 × S
    else:                        result = 0.20 × R + 0.80 × S
```

A larger week-on-week move is damped **more**, not less: 20% weight on last week
instead of 10%.

FBA is **not** damped and does not use the blend at all:

```
fba_rate = window_sum('fba', cutoff, 30) / 30
```

---

## 6. Daily rate and forecast

```
daily(cutoff) = dampen(west) + dampen(east) + fba_rate
```

### Native 70-day total (`compare_v1.py`, used for backtesting)

```
v1_forecast(cutoff) = daily(cutoff) × 70 × seasonal(cutoff+1 … cutoff+70)
```

### Weekly forward forecast (`src/ml/serving/v1.py`, what the dashboard shows)

The as-of date is **one day before the forecast date**:

```
asof = forecast_date − 1 day
daily = daily(asof)                        # computed once per SKU

for each forecast week ds:                 # ds is the W-MON label
    v1_yhat(ds) = daily × 7 × seasonal(ds − 6 days … ds)
```

The daily rate is computed **once** and reused for every week of the horizon.
Only the seasonal multiplier varies by week. There is no trend term: absent a
month boundary, every week of the horizon carries the same number.

---

## 7. Proportional seasonal modifier

For a date range, the day-weighted average of the monthly multipliers:

```
seasonal(start, end):
    total_days = (end − start).days + 1
    weighted = 0
    for each calendar month the range touches:
        chunk_days = number of days of the range falling in that month
        weighted += SEASONAL[month] × chunk_days
    return weighted / total_days
```

A week wholly inside one month gets that month's multiplier exactly. A week
straddling September and October gets a blend proportional to the day split,
which is why consecutive forecast weeks can differ slightly at a month boundary.

---

## 8. Worked example

`CA-SC-10-R-90-DG-1TO`, as-of **2026-07-19** (forecast_date 2026-07-20 minus one
day). East stream is empty for this SKU.

| window | units | per day | weight | contribution |
|---|---|---|---|---|
| west_sales 90d | 132 | 1.4667 | 0.10 | 0.14667 |
| west_sales 60d | 99 | 1.6500 | 0.15 | 0.24750 |
| west_sales 30d | 33 | 1.1000 | 0.30 | 0.33000 |
| west_sales 15d | 10 | 0.6667 | 0.20 | 0.13333 |
| west_sales 7d | 0 | 0.0000 | 0.15 | 0.00000 |
| west_preorder 30d | 81 | 2.7000 | 0.10 | 0.27000 |

```
S (west, 2026-07-19) = 1.12750
R (west, 2026-07-12) = 1.70583
change = |1.12750 − 1.70583| / 1.70583 = 0.339      → under 0.5
west  = 0.10 × 1.70583 + 0.90 × 1.12750 = 1.18533

fba   = 6 units / 30 days                = 0.20000
east  = 0

daily = 1.18533 + 0 + 0.20000            = 1.38533

week of 2026-08-03 (wholly in August, multiplier 1.00):
    1.38533 × 7 × 1.00 = 9.69733
```

Stored value in `v1_forward_forecasts.parquet` for that week: **9.697333**.

---

## 9. Things most likely to differ from the sheet

Worth checking these first, in rough order of how much they move the answer.

1. **The as-of date.** One day before the forecast date. Using the forecast date
   itself gives 9.267 rather than 9.697 for the example above, a 4.4% difference
   from a one-day shift.
2. **Preorder weighting.** Preorder is 20.1% of portfolio demand and over half of
   some SKUs, but enters the blend at a fixed 0.10. If the sheet weights it
   differently, or excludes it, the divergence is largest exactly on
   preorder-heavy SKUs.
3. **The damping thresholds.** 0.5 change, and 0.10/0.90 against 0.20/0.80.
4. **Whether FBA is damped.** Here it is not, and it bypasses the blend entirely.
5. **The channel check running before the order-type check**, so FBA preorders
   count as `fba` rather than as preorder.
6. **Window boundaries.** `(end − days, end]`, end inclusive, start exclusive.
7. **Order date versus ship date.** Everything here is attributed to the date the
   order was placed. A preorder booked in June for August fulfilment counts in
   June.
8. **The source table.** The unbounded `_forecast` table, not the 120-day-capped
   one.
