import sys
import copy
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from statsforecast import StatsForecast
from statsforecast.utils import ConformalIntervals

from config import FREQUENCY, USE_SEASONAL_ADJUSTMENT, OUTPUTS_REPORTS
from src.db import read_latest_forecast, read_actuals, read_segments, get_engine
from src.models import get_models
from src.baselines import get_baselines
from src.deseasonalize import deseasonalize, reseasonalize

_CONFORMAL_LEVEL = 70
_MAX_N_WINDOWS = 5

app = FastAPI(title="Coverland Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

@app.get("/forecast/{sku_id}")
async def get_forecast(
    sku_id: str,
    weeks: int = Query(default=26, ge=0, description="Actuals history window in weeks; 0 = all available"),
    cutoff: str = Query(default=None, description="Last actuals date to include (YYYY-MM-DD). Defaults to last completed Monday."),
    start: str = Query(default=None, description="Start date for actuals (YYYY-MM-DD). Overrides weeks when provided."),
    model: str = Query(default="Auto", description="Model override. 'Auto' uses the pre-computed stored forecast."),
    horizon: int = Query(default=0, ge=0, le=52, description="Forecast horizon in weeks; 0 = use stored horizon."),
):
    forecast = read_latest_forecast(sku_id)
    if forecast.empty:
        raise HTTPException(status_code=404, detail=f"No forecast found for SKU '{sku_id}'")

    actuals = read_actuals(sku_id, n_weeks=weeks if weeks > 0 else None, start_date=start or None)
    if actuals.empty:
        raise HTTPException(status_code=404, detail=f"No sales history found for SKU '{sku_id}'")

    if cutoff:
        cutoff_ts = pd.Timestamp(cutoff).normalize()
    else:
        # Default: last completed Monday (start of current week)
        today = pd.Timestamp.today().normalize()
        days_since_monday = today.dayofweek  # Mon=0
        cutoff_ts = today - pd.Timedelta(days=days_since_monday)

    actuals = actuals[actuals["ds"] <= cutoff_ts]

    # Trim forecast to start strictly after the last actual week
    last_actual_ds = actuals["ds"].max()
    forecast = forecast[forecast["ds"] > last_actual_ds]

    # ── Stored metadata ───────────────────────────────────────────────────────
    meta_bucket      = str(forecast["bucket"].iloc[0])
    meta_hist_len    = str(forecast["history_length"].iloc[0]) if "history_length" in forecast.columns else "full"
    stored_model     = str(forecast["selected_model"].iloc[0])
    confidence       = str(forecast["confidence"].iloc[0])
    forecast_date    = str(forecast["forecast_date"].iloc[0])
    forward_weeks    = len(forecast)

    # ── Model override / horizon extension: re-run when needed ───────────────
    effective_horizon = horizon if horizon > 0 else forward_weeks
    model_for_run = model if model != "Auto" else stored_model
    needs_rerun = (effective_horizon > forward_weeks) or (model != "Auto" and model != stored_model)

    if needs_rerun:
        train = read_actuals(sku_id, n_weeks=None)
        if train.empty:
            raise HTTPException(status_code=404, detail=f"No training data for SKU '{sku_id}'")
        train.insert(0, "unique_id", sku_id)
        train = train[train["ds"] <= cutoff_ts].copy()

        # Apply ramp-up trimming — same as the weekly batch job
        _, _, detected_train_start = _classify_sku(train)
        train = train[train["ds"] >= detected_train_start].reset_index(drop=True)

        use_deseas = USE_SEASONAL_ADJUSTMENT and meta_bucket == "smooth" and meta_hist_len != "short"
        fit_data = deseasonalize(train[["unique_id", "ds", "y"]]) if use_deseas else train[["unique_id", "ds", "y"]]

        model_min = 20 if meta_bucket == "smooth" else 8
        n_windows = max(0, min(_MAX_N_WINDOWS, (len(train) - model_min) // effective_horizon))

        try:
            candidates = get_models(meta_bucket, meta_hist_len)
        except ValueError:
            candidates = get_models("low_volume", "full")
        candidate_names = {type(m).__name__ for m in candidates}
        try:
            baselines = [b for b in get_baselines(meta_bucket, meta_hist_len) if type(b).__name__ not in candidate_names]
        except ValueError:
            baselines = []

        sf = StatsForecast(models=copy.deepcopy(candidates + baselines), freq=FREQUENCY, n_jobs=-1)
        if n_windows >= 2:
            pi = ConformalIntervals(h=effective_horizon, n_windows=n_windows)
            fcast = sf.forecast(df=fit_data, h=effective_horizon, level=[_CONFORMAL_LEVEL], prediction_intervals=pi)
        else:
            sf.fit(fit_data)
            fcast = sf.predict(h=effective_horizon)

        fcast["ds"] = pd.to_datetime(fcast["ds"])
        if use_deseas:
            fcast = reseasonalize(fcast)
        if "ds" not in fcast.columns:
            fcast = fcast.reset_index()

        yhat_s, lo_s, hi_s, resolved_model = _pick_cols(fcast, model_for_run)
        has_pi_override = lo_s is not None

        forecast = pd.DataFrame({
            "ds":             fcast["ds"],
            "yhat":           yhat_s.clip(lower=0).round(),
            "yhat_lo":        lo_s.clip(lower=0).round() if has_pi_override else pd.Series([None] * len(fcast)),
            "yhat_hi":        hi_s.clip(lower=0).round() if has_pi_override else pd.Series([None] * len(fcast)),
            "bucket":         meta_bucket,
            "selected_model": resolved_model,
            "confidence":     confidence,
        })
        stored_model = resolved_model
        forecast_date = str(pd.Timestamp.today().date())

    bucket     = meta_bucket
    model_used = stored_model
    has_pi        = forecast["yhat_lo"].notna().any()

    fig = go.Figure()

    # ── Historical actuals ────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=actuals["ds"],
        y=actuals["y"],
        mode="lines+markers",
        name="Actual demand",
        line=dict(color="#4C72B0", width=2),
        marker=dict(size=5),
        hovertemplate="Actual demand: %{y:.0f}<extra></extra>",
    ))

    # ── PI band ───────────────────────────────────────────────────────────
    if has_pi:
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast["ds"], forecast["ds"].iloc[::-1]]),
            y=pd.concat([forecast["yhat_hi"], forecast["yhat_lo"].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(221, 132, 82, 0.18)",
            line=dict(color="rgba(0,0,0,0)"),
            name="P70 interval",
            showlegend=True,
            hoverinfo="skip",
        ))
        # Invisible trace so unified hover shows the actual [lo, hi] interval
        fig.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_hi"],
            mode="none",
            name="P70 interval",
            showlegend=False,
            customdata=list(zip(
                forecast["yhat_lo"].round().astype(int),
                forecast["yhat_hi"].round().astype(int),
            )),
            hovertemplate="P70 interval: [%{customdata[0]}, %{customdata[1]}]<extra></extra>",
        ))

    # ── Point forecast ────────────────────────────────────────────────────
    # Prepend the last actual point so the line connects visually without
    # including it as a real forecast — suppress its hover with None sentinel
    last_actual_y = float(actuals.loc[actuals["ds"] == last_actual_ds, "y"].iloc[0])
    forecast_x = pd.concat([pd.Series([last_actual_ds]), forecast["ds"]], ignore_index=True)
    forecast_y = pd.concat([pd.Series([last_actual_y]), forecast["yhat"]], ignore_index=True)

    fig.add_trace(go.Scatter(
        x=forecast_x,
        y=forecast_y,
        mode="lines+markers",
        name="Forecast",
        line=dict(color="#DD8452", width=2, dash="dash"),
        marker=dict(size=5),
        hovertemplate="Forecast: %{y:.0f}<extra></extra>",
    ))

    # ── Cutoff line ───────────────────────────────────────────────────────
    cutoff = actuals["ds"].max()
    fig.add_vline(
        x=cutoff.timestamp() * 1000,
        line_width=1,
        line_dash="dot",
        line_color="#AAAAAA",
    )

    fig.update_layout(
        xaxis_title="Week",
        yaxis_title="Units",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=40, b=50),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#F0F0F0"),
        yaxis=dict(showgrid=True, gridcolor="#F0F0F0", rangemode="tozero"),
    )

    return JSONResponse({
        "chart": fig.to_json(),
        "meta": {
            "sku_id":         sku_id,
            "bucket":         bucket,
            "history_length": meta_hist_len,
            "model":          model_used,
            "confidence":     confidence,
            "forecast_date":  forecast_date,
            "has_pi":         bool(has_pi),
            "forward_weeks":  len(forecast),
        },
        "forecastDates":  forecast["ds"].dt.strftime("%Y-%m-%d").tolist(),
        "forecastValues": forecast["yhat"].round().clip(lower=0).astype(int).tolist(),
        "forecastUpper":  forecast["yhat_hi"].round().clip(lower=0).astype(int).tolist() if has_pi else None,
    })


def _classify_sku(train: pd.DataFrame) -> tuple[str, str, pd.Timestamp]:
    """Return (bucket, history_length, train_start) for a single-SKU training DataFrame.
    Inlines src/profile.py logic to avoid writing sku_profiles.csv."""
    from src.profile import (
        _detect_ramp_up, _history_length,
        ZERO_PCT_INTERMITTENT, CV_THRESHOLD, MEAN_INTERMITTENT_CUTOFF,
    )
    grp = train.sort_values("ds").reset_index(drop=True)
    data_end = grp["ds"].max()
    _, _, train_start = _detect_ramp_up(grp)
    active_weeks = max(1, int(round((data_end - train_start).days / 7)))
    active = grp[grp["ds"] >= train_start]
    y = active["y"].values if not active.empty else grp["y"].values
    mean = float(y.mean()) if len(y) else 0.0
    std = float(y.std()) if len(y) > 1 else 0.0
    cv = std / mean if mean > 0 else np.inf
    zero_pct = float((y == 0).mean()) if len(y) else 1.0

    if zero_pct >= ZERO_PCT_INTERMITTENT or mean < MEAN_INTERMITTENT_CUTOFF:
        bucket = "intermittent"
    elif cv >= CV_THRESHOLD:
        bucket = "low_volume"
    else:
        bucket = "smooth"

    hist_len = _history_length(active_weeks)
    if bucket == "intermittent" and hist_len == "short":
        hist_len = "full"  # no short model set for intermittent
    return bucket, hist_len, train_start


def _pick_cols(
    fcast: pd.DataFrame, model_name: str
) -> tuple[pd.Series, pd.Series | None, pd.Series | None, str]:
    """Extract (yhat, yhat_lo, yhat_hi, actual_model_name) for the given model."""
    lo_suf = f"-lo-{_CONFORMAL_LEVEL}"
    hi_suf = f"-hi-{_CONFORMAL_LEVEL}"
    non_data = {"unique_id", "ds", "cutoff", "y"}
    data_cols = [
        c for c in fcast.columns
        if c not in non_data and not c.endswith(lo_suf) and not c.endswith(hi_suf)
    ]

    if model_name.startswith("Ensemble:"):
        parts = model_name.replace("Ensemble:", "").split("+")
        yhat_cols = [c for c in parts if c in fcast.columns] or (data_cols[:1])
        lo_cols = [f"{p}{lo_suf}" for p in parts if f"{p}{lo_suf}" in fcast.columns]
        hi_cols = [f"{p}{hi_suf}" for p in parts if f"{p}{hi_suf}" in fcast.columns]
        return (
            fcast[yhat_cols].mean(axis=1),
            fcast[lo_cols].mean(axis=1) if lo_cols else None,
            fcast[hi_cols].mean(axis=1) if hi_cols else None,
            model_name,
        )

    actual = model_name if model_name in fcast.columns else (data_cols[0] if data_cols else None)
    if actual is None:
        return pd.Series([0.0] * len(fcast)), None, None, model_name
    lo_col, hi_col = f"{actual}{lo_suf}", f"{actual}{hi_suf}"
    return (
        fcast[actual],
        fcast[lo_col] if lo_col in fcast.columns else None,
        fcast[hi_col] if hi_col in fcast.columns else None,
        actual,
    )


@app.get("/backtest/{sku_id}")
async def run_backtest(
    sku_id: str,
    cutoff: str = Query(..., description="Last training week (YYYY-MM-DD, a Monday)"),
    horizon: int = Query(default=13, ge=1, le=52),
    history_weeks: int = Query(default=0, ge=0, description="Training weeks before cutoff; 0 = all"),
    train_start: str | None = Query(default=None, description="Explicit training start date (YYYY-MM-DD). Overrides history_weeks."),
    model: str = Query(default="Auto"),
):
    # ── Load all actuals ──────────────────────────────────────────────────
    all_actuals = read_actuals(sku_id, n_weeks=None)
    if all_actuals.empty:
        raise HTTPException(404, f"No sales history for SKU '{sku_id}'")
    all_actuals.insert(0, "unique_id", sku_id)

    cutoff_ts = pd.Timestamp(cutoff).normalize()
    train = all_actuals[all_actuals["ds"] <= cutoff_ts].copy()
    eval_df = all_actuals[
        (all_actuals["ds"] > cutoff_ts)
        & (all_actuals["ds"] <= cutoff_ts + pd.Timedelta(weeks=horizon))
    ].copy()

    if train.empty:
        raise HTTPException(400, "No training data before cutoff")

    if train_start:
        train = train[train["ds"] >= pd.Timestamp(train_start)].reset_index(drop=True)
    elif history_weeks > 0:
        train = train.tail(history_weeks).reset_index(drop=True)

    # ── Profile + ramp-up trimming ────────────────────────────────────────
    bucket, hist_len, detected_train_start = _classify_sku(train)
    # Apply ramp-up trimming unless caller already specified an explicit start
    if not train_start and history_weeks == 0:
        train = train[train["ds"] >= detected_train_start].reset_index(drop=True)

    # ── Build model list ──────────────────────────────────────────────────
    try:
        candidates = get_models(bucket, hist_len)
    except ValueError:
        candidates = get_models("low_volume", "full")
    candidate_names = {type(m).__name__ for m in candidates}
    try:
        baselines = [b for b in get_baselines(bucket, hist_len) if type(b).__name__ not in candidate_names]
    except ValueError:
        baselines = []
    all_model_list = candidates + baselines

    # ── Resolve model name ────────────────────────────────────────────────
    if model == "Auto":
        sel_path = OUTPUTS_REPORTS / "selection.csv"
        resolved = type(all_model_list[0]).__name__
        if sel_path.exists():
            sel = pd.read_csv(sel_path)
            row = sel[sel["unique_id"] == sku_id]
            if len(row):
                resolved = str(row["model"].iloc[0])
    else:
        resolved = model

    # ── Deseasonalize ─────────────────────────────────────────────────────
    use_deseas = USE_SEASONAL_ADJUSTMENT and bucket == "smooth" and hist_len != "short"
    fit_data = deseasonalize(train[["unique_id", "ds", "y"]]) if use_deseas else train[["unique_id", "ds", "y"]]

    # ── Fit + predict ─────────────────────────────────────────────────────
    model_min = 20 if bucket == "smooth" else 8
    n_windows = max(0, min(_MAX_N_WINDOWS, (len(train) - model_min) // horizon))

    sf = StatsForecast(models=copy.deepcopy(all_model_list), freq=FREQUENCY, n_jobs=-1)
    if n_windows >= 1:
        pi = ConformalIntervals(h=horizon, n_windows=n_windows)
        fcast = sf.forecast(df=fit_data, h=horizon, level=[_CONFORMAL_LEVEL], prediction_intervals=pi)
    else:
        sf.fit(fit_data)
        fcast = sf.predict(h=horizon)

    fcast["ds"] = pd.to_datetime(fcast["ds"])
    if use_deseas:
        fcast = reseasonalize(fcast)
    if "ds" not in fcast.columns:
        fcast = fcast.reset_index()

    # ── Pick columns ──────────────────────────────────────────────────────
    yhat_s, lo_s, hi_s, model_used = _pick_cols(fcast, resolved)

    eval_lookup = eval_df.set_index("ds")["y"].to_dict() if not eval_df.empty else {}
    today_ts = pd.Timestamp.today().normalize()

    predictions = []
    lo_vals = lo_s.values if lo_s is not None else [None] * len(fcast)
    hi_vals = hi_s.values if hi_s is not None else [None] * len(fcast)
    for ds_val, yhat_v, lo_v, hi_v in zip(fcast["ds"].values, yhat_s.values, lo_vals, hi_vals):
        ds_ts = pd.Timestamp(ds_val)
        actual = int(eval_lookup.get(ds_ts, 0)) if ds_ts <= today_ts else None
        predictions.append({
            "ds": str(ds_ts.date()),
            "yhat": max(0, round(float(yhat_v))) if pd.notna(yhat_v) else 0,
            "yhat_lo": max(0, round(float(lo_v))) if lo_v is not None and pd.notna(lo_v) else None,
            "yhat_hi": max(0, round(float(hi_v))) if hi_v is not None and pd.notna(hi_v) else None,
            "actual": actual,
        })

    actuals_context = [
        {"ds": str(pd.Timestamp(r["ds"]).date()), "y": int(r["y"])}
        for _, r in train.iterrows()
    ]

    # ── Metrics ───────────────────────────────────────────────────────────
    completed = [p for p in predictions if p["actual"] is not None]
    pi_weeks = [p for p in completed if p["yhat_lo"] is not None]

    total_actual   = sum(p["actual"] for p in completed)
    total_yhat     = sum(p["yhat"]   for p in completed)
    total_abs_err  = sum(abs(p["yhat"] - p["actual"]) for p in completed)

    # Per-week MAE
    mae = round(total_abs_err / len(completed)) if completed else None

    # Horizon WAPE: |sum(yhat) - sum(actual)| / sum(actual)
    # Measures total demand accuracy over the full horizon — errors that cancel
    # across weeks don't count against the model, matching our model selection metric.
    horizon_wape = round(abs(total_yhat - total_actual) / total_actual * 100) if total_actual > 0 else None

    # Horizon bias: positive = over-forecast, negative = under-forecast
    horizon_bias = round((total_yhat - total_actual) / total_actual * 100) if total_actual > 0 else None

    # MASE = per-week MAE / in-sample naive MAE
    train_y = train.sort_values("ds")["y"].values
    mae_naive = float(np.mean(np.abs(np.diff(train_y)))) if len(train_y) > 1 else None
    mase = (
        round(total_abs_err / len(completed) / mae_naive, 2)
        if (completed and mae_naive and mae_naive > 0) else None
    )

    # P70 coverage (per-week — what fraction of individual weeks fell inside the band)
    coverage = (
        round(sum(1 for p in pi_weeks if p["yhat_lo"] <= p["actual"] <= p["yhat_hi"]) / len(pi_weeks) * 100)
        if pi_weeks else None
    )

    return JSONResponse({
        "predictions": predictions,
        "actuals_context": actuals_context,
        "horizon_wape": horizon_wape,
        "horizon_bias": horizon_bias,
        "mae": mae,
        "mase": mase,
        "coverage": coverage,
        "model_used": model_used,
        "bucket": bucket,
        "history_length": hist_len,
        "train_start": str(detected_train_start.date()),
        "training_weeks": len(train),
        "completed_weeks": len(completed),
    })


@app.get("/segmentation")
async def get_segmentation():
    """Aggregate the latest forward forecasts into segment-level metrics.

    Segments:
      smooth / full or medium history  → StatsForecast
      smooth / short history           → V1
      intermittent                     → Restock Policy
      low_volume                       → Not Forecasted
    """
    engine = get_engine()
    sql_segments = """
        WITH latest_dates AS (
            SELECT unique_id, MAX(forecast_date) AS latest_date
            FROM shipcore.fc_forward_forecasts
            GROUP BY unique_id
        ),
        weekly_ranked AS (
            SELECT f.unique_id, f.bucket, f.history_length, f.yhat,
                   ROW_NUMBER() OVER (PARTITION BY f.unique_id ORDER BY f.ds) AS week_num
            FROM shipcore.fc_forward_forecasts f
            INNER JOIN latest_dates ld
                ON f.unique_id = ld.unique_id AND f.forecast_date = ld.latest_date
        ),
        sku_10w AS (
            SELECT unique_id, bucket, history_length, SUM(yhat) AS yhat_10w
            FROM weekly_ranked
            WHERE week_num <= 10
            GROUP BY unique_id, bucket, history_length
        )
        SELECT bucket, history_length, COUNT(*) AS sku_count, SUM(yhat_10w) AS volume_10w
        FROM sku_10w
        GROUP BY bucket, history_length
        ORDER BY bucket, history_length
    """
    sql_total = "SELECT COUNT(DISTINCT link_master_sku) AS total FROM shipcore.fc_velocity_link_snapshot"
    with engine.connect() as conn:
        rows = conn.execute(text(sql_segments)).fetchall()
        total_skus = int(conn.execute(text(sql_total)).scalar() or 0)

    # Map DB rows to canonical segment ids
    def _segment_id(bucket: str, history_length: str) -> str:
        if bucket == "smooth":
            return "smooth_short" if history_length == "short" else "smooth_full"
        return bucket  # "intermittent" | "low_volume"

    SEGMENT_META = {
        "smooth_full":  {"name": "Smooth / Full History",  "method": "StatsForecast",  "forecasted": True},
        "smooth_short": {"name": "Smooth / Short History", "method": "V1",             "forecasted": True},
        "intermittent": {"name": "Intermittent",           "method": "Restock Policy", "forecasted": False},
        "low_volume":   {"name": "Low Volume",             "method": "Not Forecasted", "forecasted": False},
    }

    # Accumulate per segment
    agg: dict[str, dict] = {sid: {"sku_count": 0, "volume_10w": 0.0} for sid in SEGMENT_META}
    for bucket, history_length, sku_count, volume_10w in rows:
        sid = _segment_id(str(bucket), str(history_length))
        if sid in agg:
            agg[sid]["sku_count"] += int(sku_count)
            agg[sid]["volume_10w"] += float(volume_10w or 0)

    total_volume = sum(v["volume_10w"] for v in agg.values())
    forecasted_skus   = sum(v["sku_count"]  for sid, v in agg.items() if SEGMENT_META[sid]["forecasted"])
    forecasted_volume = sum(v["volume_10w"] for sid, v in agg.items() if SEGMENT_META[sid]["forecasted"])

    def _pct(n: float, d: float) -> float:
        return round(n / d * 100, 1) if d else 0.0

    segments = []
    for sid, meta in SEGMENT_META.items():
        v = agg[sid]
        segments.append({
            "id":         sid,
            "name":       meta["name"],
            "method":     meta["method"],
            "forecasted": meta["forecasted"],
            "sku_count":  v["sku_count"],
            "volume_10w": round(v["volume_10w"]),
            "volume_pct": _pct(v["volume_10w"], total_volume),
        })

    return {
        "total_skus":          total_skus,
        "forecasted_skus":     forecasted_skus,
        "forecast_sku_pct":    _pct(forecasted_skus, total_skus),
        "total_volume_10w":    round(total_volume),
        "covered_volume_10w":  round(forecasted_volume),
        "covered_volume_pct":  _pct(forecasted_volume, total_volume),
        "segments":            segments,
    }


@app.get("/segments")
async def get_segments(
    weeks: int = Query(default=10, ge=1, le=52, description="Number of completed weeks to sum demand over"),
):
    return JSONResponse(read_segments(weeks))


@app.get("/health")
async def health():
    return {"status": "ok"}
