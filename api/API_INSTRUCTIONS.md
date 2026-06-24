# Coverland Forecast API — Integration Guide

## Overview

This is a FastAPI service that returns weekly demand forecasts for individual SKUs.
Forecasts are pre-computed and stored in a database — the API does no model fitting at
request time, so responses are fast.

---

## Starting the server

From the project root (`Time_Series_Forecasting/`):

```bash
.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Interactive docs (test endpoints in browser): `http://localhost:8000/docs`

---

## Endpoints

### `GET /forecast/{sku_id}`

Returns a Plotly chart and metadata for a single SKU.

**Example:**
```
GET http://localhost:8000/forecast/CC-CN-03-P-GR-1TO
```

**Response (200):**
```json
{
  "chart": "<Plotly JSON string>",
  "meta": {
    "sku_id":        "CC-CN-03-P-GR-1TO",
    "bucket":        "smooth",
    "model":         "AutoARIMA",
    "confidence":    "standard",
    "forecast_date": "2026-06-23",
    "has_pi":        true,
    "forward_weeks": 13
  }
}
```

**Response (404):** SKU not found or no forecast has been run yet.

**`chart` field:** A JSON string produced by Plotly. Parse with `JSON.parse(response.chart)` to get
`{ data, layout }` which can be passed directly to any Plotly renderer.

**`meta` fields:**
| Field | Description |
|---|---|
| `bucket` | `"smooth"` or `"low_volume"` — demand pattern classification |
| `model` | Name of the statistical model selected for this SKU |
| `confidence` | `"standard"` or `"low"` — low means the model's CV error was high, treat forecast with caution |
| `forecast_date` | Date the forecast job last ran |
| `has_pi` | Whether prediction interval bands are included (`true` for smooth full-history SKUs only) |
| `forward_weeks` | Number of forecasted weeks (currently 13) |

---

### `GET /health`

Returns `{ "status": "ok" }`. Use for uptime checks.

---

## React integration

Install the Plotly wrapper:
```bash
npm install react-plotly.js plotly.js
```

Component:
```jsx
import { useEffect, useState } from "react";
import Plot from "react-plotly.js";

export default function ForecastChart({ skuId }) {
  const [fig, setFig]     = useState(null);
  const [meta, setMeta]   = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!skuId) return;
    setFig(null);
    setError(null);

    fetch(`http://localhost:8000/forecast/${encodeURIComponent(skuId)}`)
      .then((res) => {
        if (!res.ok) throw new Error(`SKU not found: ${skuId}`);
        return res.json();
      })
      .then(({ chart, meta }) => {
        setFig(JSON.parse(chart));
        setMeta(meta);
      })
      .catch((err) => setError(err.message));
  }, [skuId]);

  if (error) return <p>Error: {error}</p>;
  if (!fig)  return <p>Loading forecast...</p>;

  return (
    <div>
      <Plot
        data={fig.data}
        layout={fig.layout}
        style={{ width: "100%", height: "400px" }}
        useResizeHandler
      />
      {meta && (
        <p style={{ fontSize: 12, color: "#666" }}>
          Model: {meta.model} · {meta.bucket} · {meta.confidence} confidence ·
          Run: {meta.forecast_date}
        </p>
      )}
    </div>
  );
}
```

Usage:
```jsx
<ForecastChart skuId="CC-CN-03-P-GR-1TO" />
```

---

## What the chart contains

The returned Plotly figure has three traces:

1. **Actual demand** (blue solid line) — last 26 weeks of real sales data
2. **Forecast** (orange dashed line) — 13-week forward prediction
3. **P70 interval band** (orange shaded region) — prediction interval shown only for
   smooth/full-history SKUs. Means the model expects ~70% of outcomes to land inside
   this range. Only present when `meta.has_pi === true`.

A vertical dotted line marks the boundary between historical actuals and the forecast.

---

## Important notes

- **Intermittent SKUs are excluded.** The API returns 404 for SKUs classified as
  intermittent (high zero-week rate, very low mean demand). These are managed via a
  separate reorder-point policy, not time-series forecasting.
- **Forecasts go stale.** The forecast job (`scripts/run_forward_forecast.py`) must be
  re-run whenever new sales data arrives to keep results current. `meta.forecast_date`
  tells you when it last ran.
- **Low-confidence SKUs** (`meta.confidence === "low"`) include short-history SKUs
  (< 1 year of data) and SKUs where the model's cross-validation error was high. Display
  a warning to the user for these.
- **CORS** is open (`allow_origins=["*"]`). Restrict this to your frontend's origin in
  production.
