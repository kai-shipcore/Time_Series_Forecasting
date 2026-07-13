import sys
import os
import copy
import signal
import time
import threading
import subprocess
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import iterate_in_threadpool
from sqlalchemy import text
from statsforecast import StatsForecast
from statsforecast.utils import ConformalIntervals

from config import (
    FREQUENCY, USE_SEASONAL_ADJUSTMENT, OUTPUTS_REPORTS, TEST_WEEKS, CONFORMAL_LEVELS, FORECAST_HORIZON,
    SHORT_HISTORY_WEEKS, MIN_SIM_HISTORY_WEEKS, MAX_CONFORMAL_WINDOWS, MIN_CONFORMAL_WINDOWS,
)

HORIZON_WEEKS = round(FORECAST_HORIZON / 7)
from src.db import (
    read_latest_forecast, read_actuals, read_segments, get_engine, get_global_start, _product_type_where,
    create_job, append_job_lines, touch_job, set_job_pgid, finish_job, get_job,
    job_cancel_requested, request_job_cancel, recover_orphaned_jobs, cleanup_old_jobs, ensure_jobs_table,
    ensure_indexes,
)
from pydantic import BaseModel
from src.profile import _detect_ramp_up
from src.models import get_models
from src.baselines import get_baselines
from src.deseasonalize import deseasonalize, reseasonalize
from src.v1 import load_raw_for_v1, build_index as build_v1_index, forecast_total as v1_forecast_total
from src.chat import stream_chat


def _parse_product_types(product_type: str) -> list[str] | None:
    """Parse a comma-separated product_type param into a list, or None for 'All'."""
    if product_type == "All":
        return None
    pts = [p.strip() for p in product_type.split(",") if p.strip()]
    return pts if pts else None


class JobLogger:
    """Buffers log lines and flushes to fc_jobs in batches, so per-line
    subprocess output doesn't become one DB write per line."""
    def __init__(self, job_id: str, flush_every: int = 20, flush_secs: float = 1.0):
        self.job_id = job_id
        self.buf: list[str] = []
        self.flush_every = flush_every
        self.flush_secs = flush_secs
        self._last_flush = time.monotonic()
        self._lock = threading.Lock()

    def append(self, line: str) -> None:
        with self._lock:
            self.buf.append(line)
            if (len(self.buf) >= self.flush_every
                    or time.monotonic() - self._last_flush >= self.flush_secs):
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if self.buf:
            append_job_lines(self.job_id, self.buf)
            self.buf = []
        self._last_flush = time.monotonic()


app = FastAPI(title="Coverland Forecast API")

FORECAST_API_TOKEN = os.getenv("FORECAST_API_TOKEN")

@app.middleware("http")
async def _token_auth(request, call_next):
    if FORECAST_API_TOKEN and request.url.path != "/health":
        if request.headers.get("x-forecast-token") != FORECAST_API_TOKEN:
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
    return await call_next(request)


_resp_cache: dict[str, tuple[float, str, dict]] = {}  # key -> (expires, version, payload)

def _data_version() -> str:
    """Changes when a new run is stored or a new week completes."""
    engine = get_engine()
    with engine.connect() as conn:
        max_fd = conn.execute(text(
            "SELECT MAX(forecast_date) FROM shipcore.fc_forward_forecasts")).scalar()
    today = pd.Timestamp.today().normalize()
    last_complete = today - pd.Timedelta(days=today.dayofweek or 7)
    return f"{max_fd}|{last_complete.date()}"


def _cached_response(key: str, ttl: float, compute) -> dict:
    version = _data_version()
    hit = _resp_cache.get(key)
    now = time.monotonic()
    if hit and hit[0] > now and hit[1] == version:
        return hit[2]
    payload = compute()
    _resp_cache[key] = (now + ttl, version, payload)
    return payload


@app.on_event("startup")
def _startup_jobs():
    try:
        ensure_jobs_table()
        n = recover_orphaned_jobs()
        if n:
            print(f"Recovered {n} orphaned job(s) from a previous run")
        cleanup_old_jobs(days=14)
        ensure_indexes()
    except Exception as exc:
        # Don't block server start if the DB is briefly unavailable
        print(f"Startup DB check failed: {exc}")

_VALID_LEVELS = {40, 60, 70, 80, 90}

@app.get("/forecast/{sku_id}")
def get_forecast(
    sku_id: str,
    weeks: int = Query(default=26, ge=0, description="Actuals history window in weeks; 0 = all available"),
    cutoff: str = Query(default=None, description="Last actuals date to include (YYYY-MM-DD). Defaults to last completed Monday."),
    start: str = Query(default=None, description="Start date for actuals (YYYY-MM-DD). Overrides weeks when provided."),
    model: str = Query(default="Auto", description="Model override. 'Auto' uses the pre-computed stored forecast."),
    horizon: int = Query(default=0, ge=0, le=52, description="Forecast horizon in weeks; 0 = use stored horizon."),
    level: int = Query(default=70, description="Conformal interval level (40, 60, 70, 80, 90). Display labels: P70→P85, P80→P90, etc."),
):
    if level not in _VALID_LEVELS:
        level = 70
    lo_col = f"yhat_lo_{level}"
    hi_col = f"yhat_hi_{level}"
    display_label = f"P{(100 + level) // 2}"

    forecast = read_latest_forecast(sku_id)
    if forecast.empty:
        raise HTTPException(status_code=404, detail=f"No forecast found for SKU '{sku_id}'")

    pad = get_global_start() if weeks == 0 and not start else None
    actuals = read_actuals(sku_id, n_weeks=weeks if weeks > 0 else None, start_date=start or None, pad_from=pad)
    if actuals.empty:
        raise HTTPException(status_code=404, detail=f"No sales history found for SKU '{sku_id}'")

    if cutoff:
        cutoff_ts = pd.Timestamp(cutoff).normalize()
    else:
        # NOTE: all "today" logic is server-local time; the deploy host must stay pinned to one timezone.
        today = pd.Timestamp.today().normalize()
        days_back = today.dayofweek or 7  # Mon=0 → go back 7; otherwise go back dayofweek
        cutoff_ts = today - pd.Timedelta(days=days_back)

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
        n_windows = max(0, min(MAX_CONFORMAL_WINDOWS, (len(train) - model_min) // effective_horizon))

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
        if n_windows >= MIN_CONFORMAL_WINDOWS:
            pi = ConformalIntervals(h=effective_horizon, n_windows=n_windows)
            fcast = sf.forecast(df=fit_data, h=effective_horizon, level=CONFORMAL_LEVELS, prediction_intervals=pi)
        else:
            sf.fit(fit_data)
            fcast = sf.predict(h=effective_horizon)

        fcast["ds"] = pd.to_datetime(fcast["ds"])
        if use_deseas:
            fcast = reseasonalize(fcast)
        if "ds" not in fcast.columns:
            fcast = fcast.reset_index()

        yhat_s, lo_s, hi_s, resolved_model = _pick_cols(fcast, model_for_run, level=level)
        has_pi_override = lo_s is not None

        forecast = pd.DataFrame({
            "ds":    fcast["ds"],
            "yhat":  yhat_s.clip(lower=0).round(),
            lo_col:  lo_s.clip(lower=0).round() if has_pi_override else pd.Series([None] * len(fcast)),
            hi_col:  hi_s.clip(lower=0).round() if has_pi_override else pd.Series([None] * len(fcast)),
            "bucket":         meta_bucket,
            "selected_model": resolved_model,
            "confidence":     confidence,
        })
        stored_model = resolved_model
        forecast_date = str(pd.Timestamp.today().date())

    bucket     = meta_bucket
    model_used = stored_model
    has_pi        = forecast[lo_col].notna().any()

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
            y=pd.concat([forecast[hi_col], forecast[lo_col].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(221, 132, 82, 0.18)",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{display_label} interval",
            showlegend=True,
            hoverinfo="skip",
        ))
        # Invisible trace so unified hover shows the actual [lo, hi] interval
        fig.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast[hi_col],
            mode="none",
            name=f"{display_label} interval",
            showlegend=False,
            customdata=list(zip(
                forecast[lo_col].round().astype(int),
                forecast[hi_col].round().astype(int),
            )),
            hovertemplate=f"{display_label} interval: [%{{customdata[0]}}, %{{customdata[1]}}]<extra></extra>",
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
        "forecastUpper":  forecast[hi_col].round().clip(lower=0).astype(int).tolist() if has_pi else None,
        "level":          level,
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
    fcast: pd.DataFrame, model_name: str, level: int = 70
) -> tuple[pd.Series, pd.Series | None, pd.Series | None, str]:
    """Extract (yhat, yhat_lo, yhat_hi, actual_model_name) for the given model at the given level."""
    lo_suf = f"-lo-{level}"
    hi_suf = f"-hi-{level}"
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
def run_backtest(
    sku_id: str,
    cutoff: str = Query(..., description="Last training week (YYYY-MM-DD, a Monday)"),
    horizon: int = Query(default=13, ge=1, le=52),
    history_weeks: int = Query(default=0, ge=0, description="Training weeks before cutoff; 0 = all"),
    train_start: str | None = Query(default=None, description="Explicit training start date (YYYY-MM-DD). Overrides history_weeks."),
    model: str = Query(default="Auto"),
    level: int = Query(default=70, description="Conformal interval level (40, 60, 70, 80, 90)."),
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
    n_windows = max(0, min(MAX_CONFORMAL_WINDOWS, (len(train) - model_min) // horizon))

    sf = StatsForecast(models=copy.deepcopy(all_model_list), freq=FREQUENCY, n_jobs=-1)
    if n_windows >= MIN_CONFORMAL_WINDOWS:
        pi = ConformalIntervals(h=horizon, n_windows=n_windows)
        fcast = sf.forecast(df=fit_data, h=horizon, level=CONFORMAL_LEVELS, prediction_intervals=pi)
    else:
        sf.fit(fit_data)
        fcast = sf.predict(h=horizon)

    fcast["ds"] = pd.to_datetime(fcast["ds"])
    if use_deseas:
        fcast = reseasonalize(fcast)
    if "ds" not in fcast.columns:
        fcast = fcast.reset_index()

    # ── Pick columns ──────────────────────────────────────────────────────
    if level not in _VALID_LEVELS:
        level = 70
    yhat_s, lo_s, hi_s, model_used = _pick_cols(fcast, resolved, level=level)

    eval_lookup = eval_df.set_index("ds")["y"].to_dict() if not eval_df.empty else {}
    today_ts = pd.Timestamp.today().normalize()

    predictions = []
    lo_vals = lo_s.values if lo_s is not None else [None] * len(fcast)
    hi_vals = hi_s.values if hi_s is not None else [None] * len(fcast)
    for ds_val, yhat_v, lo_v, hi_v in zip(fcast["ds"].values, yhat_s.values, lo_vals, hi_vals):
        ds_ts = pd.Timestamp(ds_val)
        actual = int(eval_lookup.get(ds_ts, 0)) if ds_ts <= today_ts else None
        predictions.append({
            "ds":      str(ds_ts.date()),
            "yhat":    max(0, round(float(yhat_v))) if pd.notna(yhat_v) else 0,
            "yhat_lo": max(0, round(float(lo_v))) if lo_v is not None and pd.notna(lo_v) else None,
            "yhat_hi": max(0, round(float(hi_v))) if hi_v is not None and pd.notna(hi_v) else None,
            "actual":  actual,
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

    # Coverage: fraction of individual weeks where actual fell inside the selected level band
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
        "level": level,
        "model_used": model_used,
        "bucket": bucket,
        "history_length": hist_len,
        "train_start": str(detected_train_start.date()),
        "training_weeks": len(train),
        "completed_weeks": len(completed),
    })


# ── Segment simulation ───────────────────────────────────────────────────────

def _run_segment_simulation(
    segment: str,
    cutoff_ts: pd.Timestamp,
    horizon: int,
    model_param: str,
    pts: list[str] | None,
    log_fn=None,
    cancel_check=None,
) -> dict:
    """Fit fresh models for all SKUs in a smooth segment and return per-SKU backtest results."""

    def log(msg: str):
        if log_fn:
            log_fn(msg)

    def cancelled() -> bool:
        return cancel_check is not None and cancel_check()

    log("Sim-Step 0: Loading profiles…")
    profiles_path = ROOT / "data" / "processed" / "sku_profiles.csv"
    if not profiles_path.exists():
        return {"error": "sku_profiles.csv not found — run the forecast pipeline first."}

    prof = pd.read_csv(profiles_path, usecols=["unique_id", "train_start", "bucket"])
    prof["train_start"] = pd.to_datetime(prof["train_start"])
    # Restrict to smooth bucket — intermittent/low_volume SKUs are excluded even if
    # they happen to have high active_weeks at the cutoff date.
    prof = prof[prof["bucket"] == "smooth"].copy()
    prof["aw"] = ((cutoff_ts - prof["train_start"]).dt.days // 7).clip(lower=0).astype(int)

    log("Sim-Step 1: Filtering SKUs…")
    if segment == "smooth_full":
        eligible     = prof[prof["aw"] >= SHORT_HISTORY_WEEKS].copy()
        hist_len_key = "full"
    else:  # smooth_short
        eligible     = prof[(prof["aw"] >= MIN_SIM_HISTORY_WEEKS) & (prof["aw"] < SHORT_HISTORY_WEEKS)].copy()
        hist_len_key = "short"

    uid_to_ts = dict(zip(eligible["unique_id"], eligible["train_start"]))
    uid_to_aw = dict(zip(eligible["unique_id"], eligible["aw"]))
    uid_list  = list(uid_to_ts.keys())
    log(f"  → {len(uid_list)} SKUs eligible")

    horizon_start = cutoff_ts + pd.Timedelta(weeks=1)
    horizon_end   = cutoff_ts + pd.Timedelta(weeks=horizon)

    def _empty():
        return {
            "segment": segment, "weeks": horizon, "mode": "simulation",
            "period_start": str(horizon_start.date()),
            "period_end":   str(horizon_end.date()),
            "skus": [],
        }

    if not uid_list:
        return _empty()

    if cancelled():
        return {"cancelled": True}

    log("Sim-Step 2: Loading demand data…")
    engine    = get_engine()
    pt_clause = _product_type_where("link_master_sku", pts) if pts else "TRUE"
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT link_master_sku AS unique_id,
                   DATE_TRUNC('week', order_date::timestamp)::date AS ds,
                   SUM(link_qty) AS y
            FROM shipcore.fc_velocity_link_snapshot_forecast
            WHERE link_master_sku IN :uids
              AND {pt_clause}
            GROUP BY link_master_sku, DATE_TRUNC('week', order_date::timestamp)::date
        """), {"uids": tuple(uid_list)}).fetchall()

    if not rows:
        return _empty()

    all_demand = pd.DataFrame(rows, columns=["unique_id", "ds", "y"])
    all_demand["ds"] = pd.to_datetime(all_demand["ds"])

    # Restrict uid_list to SKUs that survived the product-type filter
    uid_list = [u for u in uid_list if u in all_demand["unique_id"].values]

    if cancelled():
        return {"cancelled": True}

    log("Sim-Step 3: Zero-filling training grids…")
    global_grid = pd.DataFrame({"ds": pd.date_range(all_demand["ds"].min(), cutoff_ts, freq="W-MON")})
    train_frames: dict[str, pd.DataFrame] = {}
    eval_lookups: dict[str, dict[pd.Timestamp, int]] = {}
    demand_by_uid = {uid: g for uid, g in all_demand.groupby("unique_id")}

    for uid in uid_list:
        grp = demand_by_uid.get(uid)
        if grp is None:
            continue
        ts          = uid_to_ts[uid]
        train_raw   = grp[grp["ds"] <= cutoff_ts][["ds", "y"]].rename(columns={"y": "y_act"})
        merged      = global_grid.merge(train_raw, on="ds", how="left")
        merged["y"] = merged["y_act"].fillna(0.0)
        train_df    = merged[merged["ds"] >= ts][["ds", "y"]].sort_values("ds").reset_index(drop=True)
        train_df.insert(0, "unique_id", uid)

        if len(train_df) < 8:
            continue

        train_frames[uid] = train_df
        eval_grp          = grp[(grp["ds"] > cutoff_ts) & (grp["ds"] <= horizon_end)]
        eval_lookups[uid] = dict(zip(eval_grp["ds"], eval_grp["y"].astype(int)))

    valid_uids = list(train_frames.keys())
    if not valid_uids:
        return _empty()

    bucket     = "smooth"
    # short included since Jul 2026 — keep in sync with run_forward_forecast.py
    use_deseas = USE_SEASONAL_ADJUSTMENT
    model_min  = 20

    try:
        candidates = get_models(bucket, hist_len_key)
    except ValueError:
        candidates = get_models("low_volume", "full")
    candidate_names = {type(m).__name__ for m in candidates}
    try:
        baselines = [b for b in get_baselines(bucket, hist_len_key) if type(b).__name__ not in candidate_names]
    except ValueError:
        baselines = []
    all_models_list = candidates + baselines

    combined  = pd.concat([train_frames[uid] for uid in valid_uids]).reset_index(drop=True)
    fit_data  = deseasonalize(combined[["unique_id", "ds", "y"]]) if use_deseas else combined[["unique_id", "ds", "y"]]
    min_train = min(len(train_frames[uid]) for uid in valid_uids)
    # Group-level window count at horizon length — used only for conformal PIs below.
    n_windows = max(0, min(MAX_CONFORMAL_WINDOWS, (min_train - model_min) // horizon))

    if cancelled():
        return {"cancelled": True}

    # ── Model selection ───────────────────────────────────────────────────────
    # CV windows are TEST_WEEKS (10) long to match the pipeline, and the window
    # count is computed PER SKU so one short series no longer caps the whole
    # group. SKUs are batched into cohorts by window count (StatsForecast needs
    # a uniform n_windows per call). A single window can't referee the full
    # model menu, so n=1 cohorts choose between the two robust defaults only.
    uid_to_model: dict[str, str] = {}

    if model_param == "Auto":
        uid_n = {uid: max(0, min(MAX_CONFORMAL_WINDOWS, (len(train_frames[uid]) - model_min) // TEST_WEEKS))
                 for uid in valid_uids}
        cohorts: dict[int, list[str]] = {}
        for uid, n in uid_n.items():
            if n >= 1:
                cohorts.setdefault(n, []).append(uid)
        n_cv = sum(len(v) for v in cohorts.values())
        log(f"Sim-Step 4: CV model selection ({TEST_WEEKS}-week windows, per-SKU count; "
            f"{n_cv}/{len(valid_uids)} SKUs eligible)…")
        binary_menu = [m for m in all_models_list if type(m).__name__ in ("AutoETS", "WindowAverage")]
        non_meta = {"unique_id", "ds", "cutoff", "y", "bucket", "history_length"}
        for n, uids in sorted(cohorts.items(), reverse=True):
            if cancelled():
                return {"cancelled": True}
            menu = binary_menu if (n == 1 and len(binary_menu) >= 2) else all_models_list
            cohort_fit = fit_data[fit_data["unique_id"].isin(uids)]
            log(f"  → cohort n_windows={n}: {len(uids)} SKUs, {len(menu)} candidates")
            try:
                sf_cv = StatsForecast(models=copy.deepcopy(menu), freq=FREQUENCY, n_jobs=-1)
                cv    = sf_cv.cross_validation(df=cohort_fit, h=TEST_WEEKS, n_windows=n, step_size=TEST_WEEKS)
                if use_deseas:
                    cv = reseasonalize(cv)
                model_cols = [c for c in cv.columns if c not in non_meta]
                for uid, uid_cv in cv.groupby("unique_id"):
                    best_wape, best = float("inf"), None
                    for col in model_cols:
                        if uid_cv[col].isna().all():
                            continue
                        window_wapes = [
                            abs(w[col].sum() - w["y"].sum()) / max(w["y"].sum(), 1e-6)
                            for _, w in uid_cv.groupby("cutoff")
                        ]
                        if window_wapes:
                            avg = sum(window_wapes) / len(window_wapes)
                            if avg < best_wape:
                                best_wape, best = avg, col
                    if best:
                        uid_to_model[str(uid)] = best
            except Exception as exc:
                log(f"  → cohort n_windows={n} CV failed ({exc}); those SKUs use defaults")
        log(f"  → selected models for {len(uid_to_model)} SKUs")
    else:
        fixed = model_param
        log(f"Sim-Step 4: Fitting models ({fixed}, {len(valid_uids)} SKUs)…")
        uid_to_model = {uid: fixed for uid in valid_uids}

    # Fill any gaps (n=0 SKUs, failed cohorts) with selection.csv or a safe default
    if model_param == "Auto":
        default_model = "WindowAverage" if hist_len_key == "short" else "AutoETS"
        fittable = {type(m).__name__ for m in all_models_list}
        sel_map: dict[str, str] = {}
        sel_path = OUTPUTS_REPORTS / "selection.csv"
        if sel_path.exists():
            sel_df = pd.read_csv(sel_path)
            for _, row in sel_df.iterrows():
                m = str(row["model"])
                if m.startswith("Ensemble:"):
                    m = m.replace("Ensemble:", "").split("+")[0]
                sel_map[str(row["unique_id"])] = m
        for uid in valid_uids:
            if uid not in uid_to_model:
                m = sel_map.get(uid, default_model)
                if m not in fittable:   # e.g. "V1" label — not a fittable column here
                    m = default_model
                uid_to_model[uid] = m

    if cancelled():
        return {"cancelled": True}

    # ── Forecast ──────────────────────────────────────────────────────────────
    log(f"  → Forecasting {len(valid_uids)} SKUs…")
    sf = StatsForecast(models=copy.deepcopy(all_models_list), freq=FREQUENCY, n_jobs=-1)
    try:
        if n_windows >= MIN_CONFORMAL_WINDOWS:
            pi    = ConformalIntervals(h=horizon, n_windows=n_windows)
            fcast = sf.forecast(df=fit_data, h=horizon, level=CONFORMAL_LEVELS, prediction_intervals=pi)
        else:
            sf.fit(fit_data)
            fcast = sf.predict(h=horizon)
    except Exception:
        try:
            sf2   = StatsForecast(models=copy.deepcopy(all_models_list), freq=FREQUENCY, n_jobs=-1)
            sf2.fit(fit_data)
            fcast = sf2.predict(h=horizon)
        except Exception as exc:
            return {"error": f"Forecast failed: {exc}"}

    fcast["ds"] = pd.to_datetime(fcast["ds"])
    if use_deseas:
        fcast = reseasonalize(fcast)
    if "ds" not in fcast.columns:
        fcast = fcast.reset_index()

    today_ts    = pd.Timestamp.today().normalize()
    sku_results: list[dict] = []
    fcast_by_uid = {uid: g for uid, g in fcast.groupby("unique_id")} if "unique_id" in fcast.columns else None

    for uid in valid_uids:
        uid_fcast = (fcast_by_uid.get(uid) if fcast_by_uid is not None else None) or fcast
        if uid_fcast.empty:
            continue

        model_for_uid                   = uid_to_model.get(uid, "AutoETS")
        yhat_s, lo_s, hi_s, model_used = _pick_cols(uid_fcast, model_for_uid)
        has_pi        = lo_s is not None
        eval_lookup   = eval_lookups[uid]
        yhat_total    = 0
        yhat_lo_total = 0
        yhat_hi_total = 0
        demand_total  = 0
        completed_weeks = 0

        lo_vals = lo_s.values if lo_s is not None else [None] * len(uid_fcast)
        hi_vals = hi_s.values if hi_s is not None else [None] * len(uid_fcast)

        for ds_val, yhat_v, lo_v, hi_v in zip(uid_fcast["ds"].values, yhat_s.values, lo_vals, hi_vals):
            ds_ts = pd.Timestamp(ds_val)
            # Only count weeks where actual demand is available (same window for both sides)
            if ds_ts > today_ts:
                continue
            completed_weeks += 1
            yhat        = max(0, round(float(yhat_v))) if pd.notna(yhat_v) else 0
            yhat_total += yhat
            if has_pi and lo_v is not None and pd.notna(lo_v):
                yhat_lo_total += max(0, round(float(lo_v)))
            if has_pi and hi_v is not None and pd.notna(hi_v):
                yhat_hi_total += max(0, round(float(hi_v)))
            demand_total += eval_lookup.get(ds_ts, 0)

        aw = uid_to_aw.get(uid)
        sku_results.append({
            "unique_id":           uid,
            "bucket":              bucket,
            "history_length":      hist_len_key,
            "selected_model":      model_used,
            "confidence":          "standard",
            "yhat_total":          yhat_total,
            "yhat_lo_total":       yhat_lo_total if has_pi else None,
            "yhat_hi_total":       yhat_hi_total if has_pi else None,
            "demand_total":        demand_total,
            "active_weeks":        int(aw) if aw is not None else None,
            "weeks_to_graduation": max(0, SHORT_HISTORY_WEEKS - int(aw)) if aw is not None else None,
            "_completed_weeks":    completed_weeks,
        })

    # Compute V1 baseline for all segments (short no longer routes to V1 —
    # its default is WindowAverage(12), so V1 is a genuine comparison there too)
    for r in sku_results:
        r["v1_yhat_total"] = None
    if sku_results:
        try:
            v1_raw   = load_raw_for_v1()
            v1_index = build_v1_index(v1_raw)
            horizon_days = horizon * 7
            for r in sku_results:
                full_total = v1_forecast_total(v1_index, r["unique_id"], cutoff_ts, horizon_days)
                cw = r["_completed_weeks"]
                r["v1_yhat_total"] = round(full_total * cw / horizon) if horizon > 0 and cw > 0 else None
        except Exception as exc:
            log(f"  → V1 baseline failed: {exc}")

    for r in sku_results:
        r.pop("_completed_weeks", None)

    log("Sim-Step 5: Complete.")
    return {
        "segment":      segment,
        "weeks":        horizon,
        "mode":         "simulation",
        "period_start": str(horizon_start.date()),
        "period_end":   str(horizon_end.date()),
        "skus":         sku_results,
    }


@app.post("/segment-simulate-job/{segment}")
def start_segment_simulation(
    segment: str,
    cutoff: str = Query(..., description="Cutoff date (YYYY-MM-DD, a Monday)"),
    horizon: int = Query(default=13, ge=1, le=52),
    model: str = Query(default="Auto"),
    product_type: str = Query(default="All"),
):
    """Start a simulation job and return a job_id to poll.
    Rejects with 409 if a simulation job is already running."""
    if segment not in ("smooth_full", "smooth_short"):
        raise HTTPException(400, "Simulation is only supported for smooth_full and smooth_short.")

    pts       = _parse_product_types(product_type)
    cutoff_ts = pd.Timestamp(cutoff).normalize()

    job_id = create_job("simulation")
    if job_id is None:
        raise HTTPException(status_code=409, detail="A simulation is already in progress")

    def _cancel_check() -> bool:
        # Doubles as heartbeat: called between simulation steps
        try:
            touch_job(job_id)
            return job_cancel_requested(job_id)
        except Exception:
            return False

    def run():
        logger = JobLogger(job_id, flush_every=1)  # step logs are sparse; flush eagerly
        try:
            result = _run_segment_simulation(
                segment, cutoff_ts, horizon, model, pts,
                log_fn=logger.append, cancel_check=_cancel_check,
            )
            logger.flush()
            if result.get("cancelled"):
                finish_job(job_id, "cancelled", exit_code=0)
            elif result.get("error"):
                append_job_lines(job_id, [f"Error: {result['error']}"])
                finish_job(job_id, "failed", exit_code=-1)
            else:
                finish_job(job_id, "done", exit_code=0, result=result)
        except Exception as exc:
            logger.append(f"Error: {exc}")
            logger.flush()
            finish_job(job_id, "failed", exit_code=-1)

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": job_id}


@app.get("/segment-simulate-result/{job_id}")
def segment_simulate_result(job_id: str):
    """Retrieve the result of a completed simulation job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "done":
        raise HTTPException(409, f"Job not done yet (status: {job['status']})")
    return JSONResponse(job["result"])


@app.post("/cancel-simulation/{job_id}")
def cancel_simulation(job_id: str):
    """Signal a running simulation job to stop between steps."""
    job = request_job_cancel(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] not in ("running", "cancelling"):
        return {"status": job["status"]}
    return {"status": "cancelling"}


@app.get("/segment-simulate/{segment}")
def simulate_segment(
    segment: str,
    cutoff: str = Query(..., description="Cutoff date (YYYY-MM-DD, a Monday)"),
    horizon: int = Query(default=13, ge=1, le=52),
    model: str = Query(default="Auto"),
    product_type: str = Query(default="All"),
):
    """Run a fresh batch backtest simulation for all SKUs in a smooth segment."""
    if segment not in ("smooth_full", "smooth_short"):
        raise HTTPException(400, "Simulation is only supported for smooth_full and smooth_short.")
    pts       = _parse_product_types(product_type)
    cutoff_ts = pd.Timestamp(cutoff).normalize()
    result    = _run_segment_simulation(segment, cutoff_ts, horizon, model, pts)
    if "error" in result:
        raise HTTPException(500, detail=result["error"])
    return JSONResponse(result)


@app.get("/segmentation")
def get_segmentation():
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
    sql_total = "SELECT COUNT(DISTINCT link_master_sku) AS total FROM shipcore.fc_velocity_link_snapshot_forecast"
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


@app.get("/backtest-cycles")
def get_backtest_cycles(test: bool = Query(default=False)):
    """Return forecast runs where every forecasted week has already passed.
    Only these runs have complete actuals across the full horizon."""
    today = pd.Timestamp.today().normalize()
    days_back = today.dayofweek or 7
    last_monday = today - pd.Timedelta(days=days_back)
    table = "shipcore.fc_forward_forecasts_test" if test else "shipcore.fc_forward_forecasts"

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT forecast_date,
                   MIN(ds)                   AS horizon_start,
                   MAX(ds)                   AS horizon_end,
                   COUNT(DISTINCT ds)        AS horizon_weeks,
                   COUNT(DISTINCT unique_id) AS sku_count
            FROM {table}
            GROUP BY forecast_date
            HAVING MAX(ds) <= :last_monday
            ORDER BY forecast_date DESC
        """), {"last_monday": last_monday.date()}).fetchall()

    return JSONResponse([
        {
            "forecast_date":  str(r[0]),
            "horizon_start":  str(r[1]),
            "horizon_end":    str(r[2]),
            "horizon_weeks":  int(r[3]),
            "sku_count":      int(r[4]),
        }
        for r in rows
    ])


@app.get("/accuracy-history")
def get_accuracy_history(
    product_type: str = Query(default="All", description="Comma-separated product types, or 'All'"),
):
    """Pooled WAPE per stored forecast run, evaluated over the first K completed
    weeks of each run's horizon — for every K at once, so the client can switch
    the window without refetching. One run per training week (enforced at write).
    """
    product_types = _parse_product_types(product_type)

    def _compute() -> dict:
        pt_fcast = _product_type_where("f.unique_id", product_types)
        pt_snap  = _product_type_where("v.link_master_sku", product_types)

        today = pd.Timestamp.today().normalize()
        days_back = today.dayofweek or 7
        last_complete = today - pd.Timedelta(days=days_back)

        engine = get_engine()
        with engine.connect() as conn:
            fdf = pd.read_sql(text(f"""
                SELECT f.forecast_date, f.unique_id, f.history_length, f.ds, f.yhat, f.v1_yhat,
                       f.yhat_lo_70 AS lo, f.yhat_hi_70 AS hi
                FROM shipcore.fc_forward_forecasts f
                WHERE f.bucket = 'smooth'
                  AND f.ds <= :last_complete
                  AND {pt_fcast}
            """), conn, params={"last_complete": last_complete.date()}, parse_dates=["ds"])

            if fdf.empty:
                return {"last_complete_week": str(last_complete.date()), "series": []}

            # Weekly actuals labeled by W-MON convention: each week is labeled by the
            # Monday it ends on, so an order maps to the first Monday >= order_date.
            adf = pd.read_sql(text(f"""
                SELECT
                    v.link_master_sku AS unique_id,
                    (v.order_date + ((8 - EXTRACT(ISODOW FROM v.order_date))::int % 7) * INTERVAL '1 day')::date AS ds,
                    SUM(v.link_qty) AS y
                FROM shipcore.fc_velocity_link_snapshot_forecast v
                WHERE v.order_date > :from_date
                  AND {pt_snap}
                GROUP BY 1, 2
            """), conn, params={"from_date": (fdf["ds"].min() - pd.Timedelta(days=7)).date()}, parse_dates=["ds"])

        merged = fdf.merge(adf, on=["unique_id", "ds"], how="left")
        merged["y"] = merged["y"].fillna(0).astype(float)
        merged["yhat"] = merged["yhat"].clip(lower=0)
        merged["v1_yhat"] = merged["v1_yhat"].clip(lower=0)  # NaN stays NaN
        merged["segment"] = np.where(merged["history_length"] == "short", "smooth_short", "smooth_full")

        # Weekly band-coverage flags: P85 band (level-70 conformal) should contain
        # the actual ~70% of the time when calibrated.
        has_band = merged["lo"].notna() & merged["hi"].notna()
        merged["band_n"]   = has_band.astype(int)
        merged["band_hit"] = (has_band & (merged["y"] >= merged["lo"].clip(lower=0)) & (merged["y"] <= merged["hi"])).astype(int)

        merged = merged.sort_values(["forecast_date", "unique_id", "ds"])
        grp = merged.groupby(["forecast_date", "unique_id"])
        merged["week_index"] = grp.cumcount() + 1
        merged["yhat_cum"] = grp["yhat"].cumsum()
        merged["y_cum"]    = grp["y"].cumsum()
        merged["v1_cum"]   = grp["v1_yhat"].cumsum()
        merged["band_n_cum"]   = grp["band_n"].cumsum()
        merged["band_hit_cum"] = grp["band_hit"].cumsum()

        horizon_starts = merged.groupby("forecast_date")["ds"].min()

        def _wape(abs_err: float, demand: float) -> float | None:
            return round(abs_err / demand, 4) if demand > 0 else None

        series = []
        for (fd, k), snap in merged.groupby(["forecast_date", "week_index"]):
            for seg in ("all", "smooth_full", "smooth_short"):
                sub = snap if seg == "all" else snap[snap["segment"] == seg]
                if sub.empty:
                    continue
                demand = float(sub["y_cum"].sum())
                model_err = float((sub["yhat_cum"] - sub["y_cum"]).abs().sum())

                v1_sub = sub[sub["v1_cum"].notna()]
                demand_v1 = float(v1_sub["y_cum"].sum())
                model_err_v1 = float((v1_sub["yhat_cum"] - v1_sub["y_cum"]).abs().sum())
                v1_err = float((v1_sub["v1_cum"] - v1_sub["y_cum"]).abs().sum())

                band_n = int(sub["band_n_cum"].sum())
                band_hits = int(sub["band_hit_cum"].sum())

                series.append({
                    "forecast_date":  str(pd.Timestamp(fd).date()),
                    "horizon_start":  str(horizon_starts[fd].date()),
                    "segment":        seg,
                    "k":              int(k),
                    "n_skus":         int(sub["unique_id"].nunique()),
                    "demand_total":   round(demand),
                    "model_total":    round(float(sub["yhat_cum"].sum())),
                    "model_wape":     _wape(model_err, demand),
                    "n_v1":           int(v1_sub["unique_id"].nunique()),
                    "demand_total_v1": round(demand_v1),
                    "v1_total":       round(float(v1_sub["v1_cum"].sum())),
                    "model_wape_v1":  _wape(model_err_v1, demand_v1),
                    "v1_wape":        _wape(v1_err, demand_v1),
                    "coverage":       round(band_hits / band_n, 4) if band_n > 0 else None,
                    "n_band":         band_n,
                })

        series.sort(key=lambda r: (r["horizon_start"], r["segment"], r["k"]))
        return {"last_complete_week": str(last_complete.date()), "series": series}

    payload = _cached_response(f"accuracy:{product_type}", ttl=600, compute=_compute)
    return JSONResponse(payload)


@app.get("/history/{sku_id}")
def get_history(sku_id: str):
    """Weekly sales history only — for SKUs without a forecast (intermittent).
    Zero-padded from the global data start through the last completed week
    (same as the forecast chart's 'All' view), so gaps and pre-launch periods
    are visible rather than trimmed."""
    actuals = read_actuals(sku_id, n_weeks=None, pad_from=get_global_start())
    if actuals.empty:
        raise HTTPException(status_code=404, detail=f"No sales history found for SKU '{sku_id}'")

    today = pd.Timestamp.today().normalize()
    days_back = today.dayofweek or 7
    last_complete = today - pd.Timedelta(days=days_back)

    actuals = actuals[actuals["ds"] <= last_complete]
    if actuals.empty:
        raise HTTPException(status_code=404, detail=f"No completed sales weeks for SKU '{sku_id}'")

    return JSONResponse({
        "sku_id": sku_id,
        "dates":  actuals["ds"].dt.strftime("%Y-%m-%d").tolist(),
        "values": [int(v) for v in actuals["y"]],
    })


@app.get("/all-skus")
def get_all_skus(
    weeks: int = Query(default=10, ge=1, le=104, description="Demand lookback window in completed weeks"),
    product_type: str = Query(default="All", description="Comma-separated product types, or 'All'"),
):
    """Cross-segment SKU directory: demand, momentum, classification, and the
    latest run's forward forecast total for every SKU in the velocity universe.
    """
    product_types = _parse_product_types(product_type)
    pt_snap = _product_type_where("v.link_master_sku", product_types)

    today = pd.Timestamp.today().normalize()
    days_back = today.dayofweek or 7
    last_complete = today - pd.Timedelta(days=days_back)
    demand_start = last_complete - pd.Timedelta(weeks=weeks)
    recent_start = last_complete - pd.Timedelta(weeks=4)
    prior_start  = last_complete - pd.Timedelta(weeks=8)
    # Same 4 weeks one year (52 weeks) earlier, for seasonality-aware aggregates
    yoy_end   = last_complete - pd.Timedelta(weeks=52)
    yoy_start = last_complete - pd.Timedelta(weeks=56)

    sql = f"""
        WITH latest AS (
            SELECT unique_id,
                   MAX(history_length)      AS history_length,
                   MAX(selected_model)      AS selected_model,
                   SUM(GREATEST(yhat, 0))   AS forecast_total
            FROM shipcore.fc_forward_forecasts
            WHERE bucket = 'smooth'
              AND forecast_date = (SELECT MAX(forecast_date) FROM shipcore.fc_forward_forecasts)
            GROUP BY unique_id
        ),
        weekly AS (
            SELECT
                v.link_master_sku AS unique_id,
                (v.order_date + ((8 - EXTRACT(ISODOW FROM v.order_date))::int % 7) * INTERVAL '1 day')::date AS ds,
                SUM(v.link_qty) AS y
            FROM shipcore.fc_velocity_link_snapshot_forecast v
            WHERE {pt_snap}
            GROUP BY 1, 2
        ),
        m AS (
            SELECT unique_id,
                COUNT(*) FILTER (WHERE y > 0)                                                AS active_weeks,
                MAX(ds)  FILTER (WHERE y > 0)                                                AS last_sale_week,
                COALESCE(SUM(y) FILTER (WHERE ds > :demand_start), 0)                        AS demand_total,
                COALESCE(SUM(y) FILTER (WHERE ds > :recent_start), 0)                        AS recent4,
                COALESCE(SUM(y) FILTER (WHERE ds > :prior_start AND ds <= :recent_start), 0) AS prior4,
                COALESCE(SUM(y) FILTER (WHERE ds > :yoy_start AND ds <= :yoy_end), 0)        AS yoy4
            FROM weekly
            WHERE ds <= :last_complete
            GROUP BY unique_id
        )
        SELECT m.unique_id, m.active_weeks, m.last_sale_week,
               m.demand_total, m.recent4, m.prior4, m.yoy4,
               l.history_length, l.selected_model, l.forecast_total
        FROM m
        LEFT JOIN latest l USING (unique_id)
        ORDER BY m.demand_total DESC, m.unique_id
    """
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(sql), {
            "demand_start":  demand_start.date(),
            "recent_start":  recent_start.date(),
            "prior_start":   prior_start.date(),
            "yoy_start":     yoy_start.date(),
            "yoy_end":       yoy_end.date(),
            "last_complete": last_complete.date(),
        }).fetchall()

    skus = []
    for r in rows:
        hist = r[7]
        segment = "intermittent" if hist is None else ("smooth_short" if hist == "short" else "smooth_full")
        recent4, prior4 = float(r[4]), float(r[5])
        last_sale = r[2]
        skus.append({
            "unique_id":             r[0],
            "segment":               segment,
            "model":                 r[8],
            "active_weeks":          int(r[1]) if r[1] is not None else 0,
            "last_sale_week":        str(last_sale) if last_sale else None,
            "weeks_since_last_sale": int((last_complete.date() - last_sale).days // 7) if last_sale else None,
            "demand_total":          int(r[3]),
            "avg_weekly":            round(float(r[3]) / weeks, 1),
            "trend_pct":             round((recent4 - prior4) / prior4 * 100, 1) if prior4 > 0 else None,
            "recent_4w":             int(recent4),
            "prior_4w":              int(prior4),
            "yoy_4w":                int(r[6]),
            "forecast_total":        int(r[9]) if r[9] is not None else None,
        })

    return JSONResponse({
        "weeks":                  weeks,
        "period_start":           str((demand_start + pd.Timedelta(days=1)).date()),
        "period_end":             str(last_complete.date()),
        "forecast_horizon_weeks": HORIZON_WEEKS,
        "skus":                   skus,
    })


@app.get("/demand-trend")
def get_demand_trend(
    product_type: str = Query(default="All", description="Comma-separated product types, or 'All'"),
):
    """Aggregate weekly demand vs what stored runs predicted for those weeks.

    - actuals:   weekly demand totals per segment (last ~26 completed weeks)
    - predicted: forecast totals per (week, lead) for completed weeks, where
                 lead N = the run made N weeks before the target week
    - forward:   the latest run's remaining horizon with a P85 (level-70) band

    SKU membership and segment classification come from the latest run, and
    all runs are restricted to that SKU set so the lines share one universe.
    """
    product_types = _parse_product_types(product_type)

    def _compute() -> dict:
        pt_fcast = _product_type_where("f.unique_id", product_types)
        pt_snap  = _product_type_where("v.link_master_sku", product_types)

        today = pd.Timestamp.today().normalize()
        days_back = today.dayofweek or 7
        last_complete = today - pd.Timedelta(days=days_back)

        engine = get_engine()
        with engine.connect() as conn:
            fdf = pd.read_sql(text(f"""
                SELECT f.forecast_date, f.unique_id, f.history_length, f.ds, f.yhat,
                       f.yhat_lo_70 AS lo, f.yhat_hi_70 AS hi, f.v1_yhat
                FROM shipcore.fc_forward_forecasts f
                WHERE f.bucket = 'smooth' AND {pt_fcast}
            """), conn, parse_dates=["ds"])

            if fdf.empty:
                return {
                    "last_complete_week": str(last_complete.date()),
                    "forward_run_date": None,
                    "actuals": [], "predicted": [], "forward": [],
                }

            adf = pd.read_sql(text(f"""
                SELECT
                    v.link_master_sku AS unique_id,
                    (v.order_date + ((8 - EXTRACT(ISODOW FROM v.order_date))::int % 7) * INTERVAL '1 day')::date AS ds,
                    SUM(v.link_qty) AS y
                FROM shipcore.fc_velocity_link_snapshot_forecast v
                WHERE v.order_date > :from_date
                  AND {pt_snap}
                GROUP BY 1, 2
            """), conn, params={"from_date": (last_complete - pd.Timedelta(weeks=27)).date()}, parse_dates=["ds"])

        latest_fd = fdf["forecast_date"].max()
        seg_map = (
            fdf[fdf["forecast_date"] == latest_fd]
            .drop_duplicates("unique_id")
            .set_index("unique_id")["history_length"]
            .map(lambda h: "smooth_short" if h == "short" else "smooth_full")
        )

        fdf = fdf[fdf["unique_id"].isin(seg_map.index)].copy()
        fdf["segment"] = fdf["unique_id"].map(seg_map)
        fdf["yhat"] = fdf["yhat"].clip(lower=0)
        fdf["v1_yhat"] = fdf["v1_yhat"].clip(lower=0)  # NaN stays NaN
        horizon_start = fdf.groupby("forecast_date")["ds"].transform("min")
        fdf["lead"] = ((fdf["ds"] - horizon_start).dt.days // 7) + 1

        adf = adf[adf["unique_id"].isin(seg_map.index) & (adf["ds"] <= last_complete)].copy()
        adf["segment"] = adf["unique_id"].map(seg_map)

        actuals, predicted, forward = [], [], []
        for seg in ("all", "smooth_full", "smooth_short"):
            a_sub = adf if seg == "all" else adf[adf["segment"] == seg]
            for ds_, y in a_sub.groupby("ds")["y"].sum().items():
                actuals.append({"week": str(ds_.date()), "segment": seg, "y": int(y)})

            f_sub = fdf if seg == "all" else fdf[fdf["segment"] == seg]
            # One run per training week, so (ds, lead) maps to exactly one forecast_date.
            # Band sums fall back to yhat for SKUs without a stored interval, same as forward.
            past_src = f_sub[f_sub["ds"] <= last_complete].copy()
            past_src["lo_f"] = past_src["lo"].where(past_src["lo"].notna(), past_src["yhat"]).clip(lower=0)
            past_src["hi_f"] = past_src["hi"].where(past_src["hi"].notna(), past_src["yhat"]).clip(lower=0)
            past = (
                past_src
                .groupby(["ds", "lead", "forecast_date"])
                .agg(yhat=("yhat", "sum"), v1=("v1_yhat", lambda s: s.sum(min_count=1)),
                     lo=("lo_f", "sum"), hi=("hi_f", "sum"))
            )
            for (ds_, lead, fd), row in past.iterrows():
                predicted.append({
                    "week": str(ds_.date()), "lead": int(lead), "segment": seg,
                    "yhat": round(float(row["yhat"])),
                    "lo": round(float(row["lo"])), "hi": round(float(row["hi"])),
                    "v1": round(float(row["v1"])) if pd.notna(row["v1"]) else None,
                    "run_date": str(pd.Timestamp(fd).date()),
                })

            fwd = f_sub[(f_sub["forecast_date"] == latest_fd) & (f_sub["ds"] > last_complete)].copy()
            # SKUs without a stored band contribute their point forecast to the sum
            fwd["lo"] = fwd["lo"].where(fwd["lo"].notna(), fwd["yhat"]).clip(lower=0)
            fwd["hi"] = fwd["hi"].where(fwd["hi"].notna(), fwd["yhat"]).clip(lower=0)
            fwd_agg = fwd.groupby("ds").agg(
                yhat=("yhat", "sum"), lo=("lo", "sum"), hi=("hi", "sum"),
                v1=("v1_yhat", lambda s: s.sum(min_count=1)),
            )
            for ds_, row in fwd_agg.iterrows():
                forward.append({
                    "week": str(ds_.date()), "segment": seg,
                    "yhat": round(float(row["yhat"])), "lo": round(float(row["lo"])), "hi": round(float(row["hi"])),
                    "v1": round(float(row["v1"])) if pd.notna(row["v1"]) else None,
                })

        return {
            "last_complete_week": str(last_complete.date()),
            "forward_run_date": str(pd.Timestamp(latest_fd).date()),
            "actuals": actuals,
            "predicted": predicted,
            "forward": forward,
        }

    payload = _cached_response(f"trend:{product_type}", ttl=600, compute=_compute)
    return JSONResponse(payload)


@app.get("/segments")
def get_segments(
    weeks: int = Query(default=10, ge=1, le=52, description="Number of completed weeks to sum demand over"),
    product_type: str = Query(default="All", description="Comma-separated product types, or 'All'"),
):
    pts = None if product_type == "All" else [p.strip() for p in product_type.split(",") if p.strip()]
    return JSONResponse(read_segments(weeks, product_types=pts))


def _segment_detail_intermittent(weeks: int, product_types: list[str] | None = None) -> JSONResponse:
    """Compute intermittent SKU metrics purely from velocity data."""
    engine = get_engine()

    today = pd.Timestamp.today().normalize()
    days_back = today.dayofweek or 7
    last_monday = today - pd.Timedelta(days=days_back)
    recent_cutoff = last_monday - pd.Timedelta(weeks=weeks)

    pt_filter = _product_type_where("v.link_master_sku", product_types)

    sql = f"""
        WITH smooth_skus AS (
            SELECT DISTINCT unique_id
            FROM shipcore.fc_forward_forecasts
            WHERE bucket = 'smooth'
              AND forecast_date = (SELECT MAX(forecast_date) FROM shipcore.fc_forward_forecasts)
        ),
        weekly AS (
            SELECT
                v.link_master_sku                                        AS unique_id,
                DATE_TRUNC('week', v.order_date::timestamp)::date        AS week_start,
                SUM(v.link_qty)                                          AS week_qty
            FROM shipcore.fc_velocity_link_snapshot_forecast v
            WHERE v.link_master_sku NOT IN (SELECT unique_id FROM smooth_skus)
              AND {pt_filter}
            GROUP BY v.link_master_sku, DATE_TRUNC('week', v.order_date::timestamp)::date
        ),
        metrics AS (
            SELECT
                unique_id,
                SUM(CASE WHEN week_start > :recent_cutoff THEN week_qty ELSE 0 END)   AS units_recent,
                MAX(CASE WHEN week_qty > 0 THEN week_start END)                        AS last_sale_week,
                SUM(week_qty)                                                           AS total_units,
                COUNT(CASE WHEN week_qty > 0 THEN 1 END)                               AS nonzero_weeks
            FROM weekly
            GROUP BY unique_id
        )
        SELECT
            m.unique_id,
            COALESCE(m.units_recent, 0)                                                        AS units_recent,
            m.last_sale_week,
            CASE WHEN m.last_sale_week IS NOT NULL
                 THEN FLOOR((:today - m.last_sale_week) / 7.0)::int END                       AS weeks_since_last_sale,
            m.nonzero_weeks                                                                    AS event_count,
            CASE WHEN m.nonzero_weeks > 0
                 THEN ROUND(m.total_units::numeric / m.nonzero_weeks, 1) END                  AS avg_units_per_event
        FROM metrics m
        ORDER BY weeks_since_last_sale DESC NULLS LAST, m.unique_id
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql), {
            "recent_cutoff": recent_cutoff.date(),
            "today": today.date(),
        }).fetchall()

    return JSONResponse({
        "segment":           "intermittent",
        "weeks":             weeks,
        "forecast_run_date": str(last_monday.date()),
        "skus": [
            {
                "unique_id":             r[0],
                "units_recent":          int(r[1]) if r[1] is not None else 0,
                "last_sale_week":        str(r[2]) if r[2] is not None else None,
                "weeks_since_last_sale": int(r[3]) if r[3] is not None else None,
                "event_count":           int(r[4]) if r[4] is not None else None,
                "avg_units_per_event":   float(r[5]) if r[5] is not None else None,
            }
            for r in rows
        ],
    })


@app.get("/segment-detail/{segment}")
def get_segment_detail(
    segment: str,
    weeks: int = Query(default=10, ge=1, le=52),
    product_type: str = Query(default="All"),
    mode: str = Query(default="forward", description="'forward' = latest forecast; 'backtest' = evaluate a completed forecast run"),
    eval_date: str | None = Query(default=None, description="Backtest only: the forecast_date of the run to evaluate (YYYY-MM-DD)."),
    test: bool = Query(default=False, description="Use fc_forward_forecasts_test instead of the live table."),
    level: int = Query(default=70, description="Conformal interval level (40, 60, 70, 80, 90). Display labels: P70, P80, P85, P90, P95."),
):
    """Return per-SKU rows for a given segment (smooth_full | smooth_short | intermittent)."""
    _VALID_LEVELS = {40, 60, 70, 80, 90}
    if level not in _VALID_LEVELS:
        level = 70
    lo_col = f"yhat_lo_{level}"
    hi_col = f"yhat_hi_{level}"

    pts = _parse_product_types(product_type)
    if segment == "intermittent":
        return _segment_detail_intermittent(weeks, product_types=pts)

    if segment == "smooth_full":
        where_forward  = "f.bucket = 'smooth' AND f.history_length IN ('full', 'medium')"
        where_backtest = "f.bucket = 'smooth' AND f.history_length IN ('full', 'medium')"
    elif segment == "smooth_short":
        where_forward  = "f.bucket = 'smooth' AND f.history_length = 'short'"
        where_backtest = "f.bucket = 'smooth' AND f.history_length = 'short'"
    else:
        raise HTTPException(status_code=400, detail=f"Unknown segment '{segment}'")

    engine = get_engine()

    today = pd.Timestamp.today().normalize()
    days_back = today.dayofweek or 7
    last_monday = today - pd.Timedelta(days=days_back)

    fcast_table = "shipcore.fc_forward_forecasts_test" if test else "shipcore.fc_forward_forecasts"
    pt_fcast = _product_type_where("f.unique_id",     pts)
    pt_snap  = _product_type_where("link_master_sku", pts)

    if mode == "backtest":
        # Guard: caller must provide an eval_date from /backtest-cycles.
        if not eval_date:
            return JSONResponse({
                "segment": segment, "weeks": 0, "mode": mode,
                "period_start": "", "period_end": "", "skus": [],
                "backtest_unavailable": True, "earliest_forecast": None,
            })

        # Resolve the horizon for this specific forecast run.
        with engine.connect() as conn:
            horizon = conn.execute(text(f"""
                SELECT MIN(ds) AS h_start, MAX(ds) AS h_end, COUNT(DISTINCT ds) AS h_weeks
                FROM {fcast_table}
                WHERE forecast_date = :eval_date
            """), {"eval_date": eval_date}).fetchone()

        if not horizon or horizon[0] is None:
            raise HTTPException(400, f"No forecast data found for {eval_date}")

        horizon_start, horizon_end, horizon_weeks = horizon[0], horizon[1], int(horizon[2])

        # Evaluate this run's predictions against actual demand in the same weeks.
        # Segment membership comes from history_length stored in the table at seed time.
        sql = f"""
            WITH ranked AS (
                SELECT f.unique_id, f.bucket, f.history_length, f.selected_model,
                       f.yhat, f.{lo_col}, f.{hi_col}, f.confidence, f.active_weeks, f.v1_yhat
                FROM {fcast_table} f
                WHERE f.forecast_date = :eval_date
                  AND {where_backtest}
                  AND {pt_fcast}
            ),
            sku_agg AS (
                SELECT unique_id, bucket, history_length, selected_model, confidence,
                       MAX(active_weeks)                                                                                                           AS active_weeks,
                       SUM(ROUND(GREATEST(yhat, 0)))                                                                                             AS yhat_total,
                       SUM(CASE WHEN {lo_col} IS NOT NULL AND {lo_col} != 'NaN'::float8 THEN ROUND(GREATEST({lo_col}, 0)) END) AS yhat_lo_total,
                       SUM(CASE WHEN {hi_col} IS NOT NULL AND {hi_col} != 'NaN'::float8 THEN ROUND(GREATEST({hi_col}, 0)) END) AS yhat_hi_total,
                       SUM(CASE WHEN v1_yhat IS NOT NULL AND v1_yhat != 'NaN'::float8 THEN ROUND(GREATEST(v1_yhat, 0)) END)   AS v1_yhat_total
                FROM ranked
                GROUP BY unique_id, bucket, history_length, selected_model, confidence
            ),
            demand AS (
                SELECT link_master_sku, SUM(link_qty) AS demand_total
                FROM shipcore.fc_velocity_link_snapshot_forecast
                WHERE order_date > :eval_date
                  AND order_date <= :horizon_end
                  AND {pt_snap}
                GROUP BY link_master_sku
            )
            SELECT a.unique_id, a.bucket, a.history_length, a.selected_model, a.confidence,
                   COALESCE(a.yhat_total, 0)::int                                           AS yhat_total,
                   CASE WHEN a.yhat_lo_total IS NOT NULL THEN a.yhat_lo_total::int END      AS yhat_lo_total,
                   CASE WHEN a.yhat_hi_total IS NOT NULL THEN a.yhat_hi_total::int END      AS yhat_hi_total,
                   COALESCE(d.demand_total, 0)                                                 AS demand_total,
                   a.v1_yhat_total::int                                                     AS v1_yhat_total,
                   a.active_weeks::int                                                      AS active_weeks
            FROM sku_agg a
            LEFT JOIN demand d ON d.link_master_sku = a.unique_id
            ORDER BY a.yhat_total DESC NULLS LAST
        """
        params = {
            "eval_date":     eval_date,
            "horizon_start": horizon_start,
            "horizon_end":   horizon_end,
        }
        demand_start = pd.Timestamp(str(horizon_start))
        demand_end   = pd.Timestamp(str(horizon_end))
        weeks        = horizon_weeks
    else:
        demand_end   = last_monday
        demand_start = demand_end - pd.Timedelta(weeks=weeks)
        # Forward mode: latest forecast for the next N weeks, recent demand for context.
        sql = f"""
            WITH latest_dates AS (
                SELECT unique_id, MAX(forecast_date) AS latest_date
                FROM {fcast_table}
                GROUP BY unique_id
            ),
            ranked AS (
                SELECT f.unique_id, f.bucket, f.history_length, f.selected_model,
                       f.yhat, f.{lo_col}, f.{hi_col}, f.confidence,
                       ROW_NUMBER() OVER (PARTITION BY f.unique_id ORDER BY f.ds) AS week_num
                FROM {fcast_table} f
                INNER JOIN latest_dates ld
                    ON f.unique_id = ld.unique_id AND f.forecast_date = ld.latest_date
                WHERE {where_forward}
                  AND {pt_fcast}
            ),
            sku_agg AS (
                SELECT unique_id, bucket, history_length, selected_model, confidence,
                       SUM(ROUND(GREATEST(yhat, 0)))                                                                                             FILTER (WHERE week_num <= :weeks) AS yhat_total,
                       SUM(CASE WHEN {lo_col} IS NOT NULL AND {lo_col} != 'NaN'::float8 THEN ROUND(GREATEST({lo_col}, 0)) END) FILTER (WHERE week_num <= :weeks) AS yhat_lo_total,
                       SUM(CASE WHEN {hi_col} IS NOT NULL AND {hi_col} != 'NaN'::float8 THEN ROUND(GREATEST({hi_col}, 0)) END) FILTER (WHERE week_num <= :weeks) AS yhat_hi_total
                FROM ranked
                GROUP BY unique_id, bucket, history_length, selected_model, confidence
            ),
            demand AS (
                SELECT link_master_sku, SUM(link_qty) AS demand_total
                FROM shipcore.fc_velocity_link_snapshot_forecast
                WHERE order_date > :demand_start
                  AND order_date <= :demand_end
                  AND {pt_snap}
                GROUP BY link_master_sku
            )
            SELECT a.unique_id, a.bucket, a.history_length, a.selected_model, a.confidence,
                   COALESCE(a.yhat_total, 0)::int                                           AS yhat_total,
                   CASE WHEN a.yhat_lo_total IS NOT NULL THEN a.yhat_lo_total::int END      AS yhat_lo_total,
                   CASE WHEN a.yhat_hi_total IS NOT NULL THEN a.yhat_hi_total::int END      AS yhat_hi_total,
                   COALESCE(d.demand_total, 0)                                                 AS demand_total,
                   NULL::int                                                                 AS v1_yhat_total
            FROM sku_agg a
            LEFT JOIN demand d ON d.link_master_sku = a.unique_id
            ORDER BY a.yhat_total DESC NULLS LAST
        """
        params = {"weeks": weeks, "demand_start": demand_start.date(), "demand_end": demand_end.date()}

    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()

    # active_weeks: derived from sku_profiles.csv. In backtest mode, measure train_start → eval_date
    # so the displayed aw reflects history available at forecast time (not today's aw).
    # Segment filtering is done in SQL via history_length stored at seed time — not re-derived here.
    profiles_path = ROOT / "data" / "processed" / "sku_profiles.csv"
    active_weeks_map: dict[str, int] = {}
    current_bucket_map: dict[str, str] = {}  # uid → current bucket, used to drop stale DB rows
    if profiles_path.exists():
        prof = pd.read_csv(profiles_path, usecols=["unique_id", "active_weeks", "bucket"])
        active_weeks_map = dict(zip(prof["unique_id"], prof["active_weeks"].astype(int)))
        current_bucket_map = dict(zip(prof["unique_id"], prof["bucket"]))

    skus = []
    for r in rows:
        uid = r[0]
        # In backtest mode the DB now stores active_weeks at run time (r[10]).
        # Use it when available; fall back to current-profiles for rows written before this column existed.
        if mode == "backtest":
            aw = int(r[10]) if len(r) > 10 and r[10] is not None else active_weeks_map.get(uid)
        else:
            aw = active_weeks_map.get(uid)
        if mode == "forward" and segment in ("smooth_full", "smooth_short") and current_bucket_map:
            # Drop stale DB rows for SKUs reclassified since the last run.
            # If a SKU was reclassified to intermittent/low_volume, the pipeline
            # writes no new forecast — old smooth rows stay as "latest" in the DB.
            if current_bucket_map.get(uid) != "smooth":
                continue
        weeks_to_grad = max(0, SHORT_HISTORY_WEEKS - aw) if aw is not None else None
        skus.append({
            "unique_id":           uid,
            "bucket":              r[1],
            "history_length":      r[2],
            "selected_model":      r[3],
            "confidence":          r[4],
            "yhat_total":          int(r[5]) if r[5] is not None else 0,
            "yhat_lo_total":       int(r[6]) if r[6] is not None else None,
            "yhat_hi_total":       int(r[7]) if r[7] is not None else None,
            "demand_total":        int(r[8]) if r[8] is not None else 0,
            "active_weeks":        aw,
            "weeks_to_graduation": weeks_to_grad,
            "v1_yhat_total":       int(r[9]) if r[9] is not None else None,
        })

    # Join HorizonWAPE from selection.csv (not stored in DB) so the frontend
    # can display training error alongside the confidence badge.
    sel_path = OUTPUTS_REPORTS / "selection.csv"
    if sel_path.exists():
        sel_df = pd.read_csv(sel_path, usecols=["unique_id", "HorizonWAPE"])
        wape_map: dict[str, float | None] = {}
        for _, row in sel_df.iterrows():
            v = row["HorizonWAPE"]
            wape_map[row["unique_id"]] = float(v) if pd.notna(v) else None
        for sku in skus:
            sku["train_wape"] = wape_map.get(sku["unique_id"])
    else:
        for sku in skus:
            sku["train_wape"] = None

    if mode == "forward":
        with engine.connect() as conn:
            dr = conn.execute(text(f"""
                SELECT MAX(f.forecast_date) FROM {fcast_table} f
                WHERE {where_forward} AND {pt_fcast}
            """)).fetchone()
        forecast_run_date = str(dr[0]) if dr and dr[0] else None
    else:
        forecast_run_date = eval_date

    return JSONResponse({
        "segment":           segment,
        "weeks":             weeks,
        "mode":              mode,
        "period_start":      str(demand_start.date()),
        "period_end":        str(demand_end.date()),
        "forecast_run_date": forecast_run_date,
        "skus":              skus,
    })


@app.get("/sku-search")
def sku_search(q: str = Query(default="", min_length=1)):
    """Search SKUs by prefix/substring across all segments using sku_profiles.csv."""
    profiles_path = ROOT / "data" / "processed" / "sku_profiles.csv"
    if not profiles_path.exists():
        return JSONResponse([])

    prof = pd.read_csv(profiles_path, usecols=["unique_id", "bucket", "history_length", "active_weeks"])
    q_lower = q.strip().lower()
    matches = prof[prof["unique_id"].str.lower().str.contains(q_lower, regex=False)]

    def to_segment(row) -> str:
        if row["bucket"] == "smooth":
            return "smooth_full" if row["history_length"] in ("full", "medium") else "smooth_short"
        return "intermittent"

    results = [
        {
            "unique_id":    row["unique_id"],
            "segment":      to_segment(row),
            "active_weeks": int(row["active_weeks"]) if pd.notna(row["active_weeks"]) else None,
        }
        for _, row in matches.iterrows()
    ]
    return JSONResponse(results)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/run-forecast")
def run_forecast(horizon: int = Query(default=13, ge=1, le=104)):
    """Spawn a background forecast job and return a job_id to poll.
    Rejects with 409 if a forecast job is already running."""
    job_id = create_job("forecast")
    if job_id is None:
        raise HTTPException(status_code=409, detail="A forecast run is already in progress")

    def _run():
        logger = JobLogger(job_id)
        proc = None
        try:
            proc = subprocess.Popen(
                [sys.executable, str(ROOT / "scripts" / "run_forward_forecast.py"),
                 "--horizon", str(horizon)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(ROOT),
                start_new_session=True,   # own process group → killable with children
            )
            try:
                set_job_pgid(job_id, os.getpgid(proc.pid))
            except Exception:
                pass

            # Watcher: heartbeat + cancellation. Polls the DB every 2s while
            # the main loop below blocks on subprocess stdout.
            stop_watch = threading.Event()
            def _watch():
                while not stop_watch.wait(2.0):
                    try:
                        touch_job(job_id)
                        if job_cancel_requested(job_id):
                            try:
                                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                            except (ProcessLookupError, PermissionError):
                                pass
                            return
                    except Exception:
                        pass
            threading.Thread(target=_watch, daemon=True).start()

            for line in proc.stdout:
                logger.append(line.rstrip())
            proc.wait()
            stop_watch.set()
            logger.flush()

            if job_cancel_requested(job_id):
                finish_job(job_id, "cancelled", exit_code=proc.returncode)
            elif proc.returncode == 0:
                finish_job(job_id, "done", exit_code=0)
            else:
                finish_job(job_id, "failed", exit_code=proc.returncode)
        except Exception as exc:
            logger.append(f"Error: {exc}")
            logger.flush()
            finish_job(job_id, "failed", exit_code=-1)

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id}


@app.post("/cancel-forecast/{job_id}")
def cancel_forecast(job_id: str):
    """Request cancellation of a running forecast job (kills the whole
    process group via the job's watcher thread; also tries directly here)."""
    job = request_job_cancel(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in ("running", "cancelling"):
        return {"ok": False, "reason": "Job is not running"}
    # Fast path: if the process group still exists on this host, kill it now
    if job.get("pgid"):
        try:
            os.killpg(job["pgid"], signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    return {"ok": True}


@app.get("/forecast-status/{job_id}")
def forecast_status(job_id: str):
    """Poll the status of a running or completed forecast job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id":    job_id,
        "status":    job["status"],
        "lines":     job["lines"],
        "exit_code": job["exit_code"],
    }


@app.get("/forecast-last-run")
def forecast_last_run():
    """Return the most recent run_date and horizon from fc_forecast_history."""
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT run_date, horizon_weeks
            FROM shipcore.fc_forecast_history
            ORDER BY run_date DESC
            LIMIT 1
        """)).fetchone()
    if not row:
        return {"run_date": None, "horizon_weeks": None}
    return {"run_date": str(row[0]), "horizon_weeks": int(row[1])}


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]


@app.post("/chat")
async def chat(req: ChatRequest):
    """Agentic chat assistant — streams SSE events (status + delta + done)."""
    if not os.getenv("LLM_API_KEY"):
        raise HTTPException(status_code=503, detail="LLM_API_KEY not configured")

    messages = [{"role": m.role, "content": m.content} for m in req.messages[-20:]]
    if not messages or messages[-1]["role"] != "user":
        raise HTTPException(status_code=422, detail="Last message must be from the user")

    return StreamingResponse(
        iterate_in_threadpool(stream_chat(messages)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
