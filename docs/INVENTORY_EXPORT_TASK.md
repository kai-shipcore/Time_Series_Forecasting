# Task: real inventory export for the dashboard

The dashboard currently runs on a generated sample inventory snapshot. This task replaces it
with a real export pulled from the same tables the Commerce Integration application uses in
production.

Background on what the forecast's training target already contains, which is why the preorder
column needs care, is in the design doc, Section 2.1. Read it before
starting. The short version is that on-hand and backorder come from the warehouse inventory
table, while confirmed inbound and its ETA come from the container tables, and those two sets
may live in different databases.

**Work in two stages. Do not write the export script until stage 1 is reported back.** Several
of the assumptions below are unverified, and the point of stage 1 is to find out which ones are
wrong before code is built on them.

---

## Stage 1: establish routing and column semantics

### 1a. Which database is reachable

`src/ingest.py` and `src/db.py` already connect to the primary database using the `DB_*`
variables in `.env`. That connection reaches the `shipcore` schema. What is not established is
whether the same connection reaches `ecommerce_data`, which is a separate Supabase database in
the Commerce Integration configuration (`SUPABASE_LOOKUP_DATABASE_URL`).

Three candidate sources exist for the same inventory figures, in decreasing order of rawness:

| Source | Schema | Notes |
|---|---|---|
| `ecommerce_data.coverland_inventory_by_warehouse` | Supabase lookup | The origin table. One row per master SKU per warehouse. Columns `on_hand`, `allocated`, `available`, `backorder`, `created_at`. |
| `shipcore.sc_inventory_snapshot` | primary | Truncated and reloaded from the origin table by `src/sql/Data_sync_sc_inventory_snapshot.sql` in the Commerce repo. Columns `master_sku`, `warehouse_code`, `on_hand_qty`, `available_qty`, `backorder_qty`, `snapshot_at`. |
| `shipcore.fc_stats` | primary | A cache written by the Commerce app's stats refresh endpoint. Inventory is already pivoted into `west_stock`, `east_stock`, `west_available_stock`, `east_available_stock`, `back`, `transit_stock`. Note `back` is stored **negated**. |

Report which of the three are reachable from this repo's connection and which are actually
populated, including row counts and the age of the newest row. Prefer the rawest reachable
source. If only `fc_stats` is available, say so, because it is a cache with a refresh cadence
this repo does not control, and that limitation needs recording.

### 1b. Column semantics

This is the check that gates the recommended order quantity. Neither repository defines the
arithmetic between these columns, because they arrive pre-computed from ShipHero.

```sql
SELECT
  COUNT(*)                                                        AS rows,
  COUNT(*) FILTER (WHERE available = on_hand - allocated)         AS avail_eq_onhand_minus_alloc,
  COUNT(*) FILTER (WHERE available = on_hand - allocated - backorder) AS avail_also_net_of_backorder,
  COUNT(*) FILTER (WHERE backorder > 0)                           AS rows_with_backorder,
  COUNT(*) FILTER (WHERE available < 0)                           AS negative_available,
  COUNT(*) FILTER (WHERE on_hand < 0)                             AS negative_on_hand
FROM ecommerce_data.coverland_inventory_by_warehouse;
```

Adjust the table and column names if stage 1a routes to one of the other two sources.

What matters is the third column. **If `available` is already net of `backorder`, then the
dashboard formula double-counts**, because it subtracts available stock and separately adds the
backlog. Report the counts as raw numbers rather than a conclusion, and include a dozen sample
rows where `backorder > 0` so the relationship can be read directly.

### 1c. Coverage against the forecast

The forecast covers 447 SKUs. Report how many of them appear in the chosen inventory source,
and list up to twenty that do not. This matters because the dashboard currently fills unmatched
SKUs with zero, which makes a SKU missing from inventory indistinguishable from one genuinely
at zero stock. There is a comment in `dashboard/lib/quality.py` recording this.

---

## Stage 2: the export script

Only after stage 1 is reported. Write `scripts/export_inventory_snapshot.py` following the
pattern of the existing `scripts/export_forecast_history.py`.

### Output contract

Write `dashboard/data/inventory_snapshot.csv` with exactly these columns, which is the schema
`dashboard/lib/data.py::inventory_columns()` already expects. Do not add columns and do not
rename any.

| Column | Meaning | Source |
|---|---|---|
| `unique_id` | master SKU, joins to the forecast | `master_sku` |
| `product_name` | display name | best available; leave blank rather than synthesising one |
| `available_inventory` | units on hand and sellable | `SUM(available)` across warehouses |
| `preorder_backlog` | units owed and not yet delivered | `SUM(backorder)` across warehouses, sign corrected to positive |
| `confirmed_inbound` | units on the water, confirmed | `SUM(qty)` from container items, filtered as below |
| `inbound_eta` | arrival date of the earliest confirmed container | `MIN(eta_date)`, ISO format `YYYY-MM-DD`, blank when nothing is inbound |

### Inbound filter

Match the production filter exactly rather than inventing one. From
`src/app/api/planning/dashboard/route.ts` in the Commerce repo:

```sql
FROM shipcore.fc_container_items ci
JOIN shipcore.fc_containers c ON c.id = ci.container_id
WHERE c.status IN ('shipped', 'packing_received')
```

Draft containers are deliberately excluded. `shipped` means the final list was sent and
`packing_received` means the packing list arrived. If either status value has changed since
this was written, report that rather than guessing a replacement.

### Requirements

1. **Do not fill missing SKUs with zero.** A SKU absent from the inventory source must be
   written as an empty cell, not `0`, so the dashboard can tell "no inventory record" from
   "zero units in stock". The loader already coerces missing columns to `NaN`.
2. **Do not invent any field.** If `product_name` is not available for a SKU, leave it blank.
   Do not derive a name, a size, a status or a category from the SKU string. Category is
   already derived correctly in `dashboard/lib/data.py::product_category()` and does not belong
   in this export.
3. Use `load_dotenv(override=True)`, per the note in `CLAUDE.md` about stale `DB_*` shell
   exports.
4. Run under `.venv/bin/python`.
5. Print a summary on completion: rows written, SKUs matched against the forecast, SKUs
   missing, total units in each of the three quantity columns, and the snapshot timestamp of
   the source table.
6. The script must be safe to re-run and should overwrite the CSV in place.

### Verification

After writing the file, confirm the dashboard picks it up:

```
.venv/bin/python -c "from dashboard.lib import data as D; \
inv = D.load_inventory(); print(inv['is_sample'].iloc[0], len(inv)); print(inv.head())"
```

`is_sample` must come back `False`. Then load the planning table and report the recommended
order quantity total before and after the swap, since the sample and the real data will not
agree and the size of the difference is worth seeing.

---

## Out of scope

Do not change `dashboard/lib/calc.py`, the recommended order quantity formula, or anything in
`src/ml/`. If stage 1b shows the formula double-counts, report it and stop. That is a decision
to be made with the design doc open, not a fix to apply inside this task.
