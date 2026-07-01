import sys
import copy
import uuid
import asyncio
import threading
import subprocess
from collections import defaultdict
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
from src.db import read_latest_forecast, read_actuals, read_segments, get_engine, get_global_start, _product_type_where
from src.profile import _detect_ramp_up
from src.models import get_models
from src.baselines import get_baselines
from src.deseasonalize import deseasonalize, reseasonalize

_jobs: dict[str, dict] = {}


def _parse_product_types(product_type: str) -> list[str] | None:
    """Parse a comma-separated product_type param into a list, or None for 'All'."""
    if product_type == "All":
        return None
    pts = [p.strip() for p in product_type.split(",") if p.strip()]
    return pts if pts else None

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

    pad = get_global_start() if weeks == 0 and not start else None
    actuals = read_actuals(sku_id, n_weeks=weeks if weeks > 0 else None, start_date=start or None, pad_from=pad)
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


# ── Segment simulation ───────────────────────────────────────────────────────

def _run_segment_simulation(
    segment: str,
    cutoff_ts: pd.Timestamp,
    horizon: int,
    model_param: str,
    pts: list[str] | None,
    log_fn=None,
    cancel_flag: threading.Event | None = None,
) -> dict:
    """Fit fresh models for all SKUs in a smooth segment and return per-SKU backtest results."""
    SHORT_THRESHOLD    = 52
    MIN_HISTORY_WEEKS  = 13

    def log(msg: str):
        if log_fn:
            log_fn(msg)

    def cancelled() -> bool:
        return cancel_flag is not None and cancel_flag.is_set()

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
        eligible     = prof[prof["aw"] >= SHORT_THRESHOLD].copy()
        hist_len_key = "full"
    else:  # smooth_short
        eligible     = prof[(prof["aw"] >= MIN_HISTORY_WEEKS) & (prof["aw"] < SHORT_THRESHOLD)].copy()
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
            FROM shipcore.fc_velocity_link_snapshot
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

    for uid in uid_list:
        grp         = all_demand[all_demand["unique_id"] == uid]
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
        eval_lookups[uid] = {r["ds"]: int(r["y"]) for _, r in eval_grp.iterrows()}

    valid_uids = list(train_frames.keys())
    if not valid_uids:
        return _empty()

    bucket     = "smooth"
    use_deseas = USE_SEASONAL_ADJUSTMENT and hist_len_key != "short"
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
    n_windows = max(0, min(_MAX_N_WINDOWS, (min_train - model_min) // horizon))

    if cancelled():
        return {"cancelled": True}

    # ── Model selection ───────────────────────────────────────────────────────
    uid_to_model: dict[str, str] = {}

    if model_param == "Auto" and hist_len_key != "short" and n_windows >= 1:
        log(f"Sim-Step 4: CV model selection ({n_windows} windows, {len(valid_uids)} SKUs)…")
        try:
            sf_cv = StatsForecast(models=copy.deepcopy(all_models_list), freq=FREQUENCY, n_jobs=-1)
            cv    = sf_cv.cross_validation(df=fit_data, h=horizon, n_windows=n_windows, step_size=horizon)
            if use_deseas:
                cv = reseasonalize(cv)
            non_meta = {"unique_id", "ds", "cutoff", "y", "bucket", "history_length"}
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
            log(f"  → selected models for {len(uid_to_model)} SKUs")
        except Exception as exc:
            log(f"  → CV selection failed ({exc}), falling back to selection.csv")
    else:
        if model_param == "Auto":
            # short history: WindowAverage is the safe fallback (V1 not available here)
            fixed = "WindowAverage"
            log(f"Sim-Step 4: Fitting models (short history → {fixed}, {len(valid_uids)} SKUs)…")
        else:
            fixed = model_param
            log(f"Sim-Step 4: Fitting models ({fixed}, {len(valid_uids)} SKUs)…")
        uid_to_model = {uid: fixed for uid in valid_uids}

    # Fill any gaps with selection.csv or AutoETS fallback
    if model_param == "Auto":
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
                uid_to_model[uid] = sel_map.get(uid, "AutoETS")

    if cancelled():
        return {"cancelled": True}

    # ── Forecast ──────────────────────────────────────────────────────────────
    log(f"  → Forecasting {len(valid_uids)} SKUs…")
    sf = StatsForecast(models=copy.deepcopy(all_models_list), freq=FREQUENCY, n_jobs=-1)
    try:
        if n_windows >= 1:
            pi    = ConformalIntervals(h=horizon, n_windows=n_windows)
            fcast = sf.forecast(df=fit_data, h=horizon, level=[_CONFORMAL_LEVEL], prediction_intervals=pi)
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

    for uid in valid_uids:
        uid_fcast = fcast[fcast["unique_id"] == uid] if "unique_id" in fcast.columns else fcast
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

        lo_vals = lo_s.values if lo_s is not None else [None] * len(uid_fcast)
        hi_vals = hi_s.values if hi_s is not None else [None] * len(uid_fcast)

        for ds_val, yhat_v, lo_v, hi_v in zip(uid_fcast["ds"].values, yhat_s.values, lo_vals, hi_vals):
            ds_ts = pd.Timestamp(ds_val)
            # Only count weeks where actual demand is available (same window for both sides)
            if ds_ts > today_ts:
                continue
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
            "weeks_to_graduation": max(0, SHORT_THRESHOLD - int(aw)) if aw is not None else None,
        })

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
async def start_segment_simulation(
    segment: str,
    cutoff: str = Query(..., description="Cutoff date (YYYY-MM-DD, a Monday)"),
    horizon: int = Query(default=13, ge=1, le=52),
    model: str = Query(default="Auto"),
    product_type: str = Query(default="All"),
):
    """Start a simulation job and return a job_id to poll."""
    if segment not in ("smooth_full", "smooth_short"):
        raise HTTPException(400, "Simulation is only supported for smooth_full and smooth_short.")

    pts         = _parse_product_types(product_type)
    cutoff_ts   = pd.Timestamp(cutoff).normalize()
    job_id      = str(uuid.uuid4())[:8]
    cancel_flag = threading.Event()

    _jobs[job_id] = {
        "status":       "running",
        "lines":        [],
        "exit_code":    None,
        "proc":         None,
        "cancel_event": cancel_flag,
        "result":       None,
    }

    def log_fn(msg: str):
        _jobs[job_id]["lines"].append(msg)

    def run():
        try:
            result = _run_segment_simulation(segment, cutoff_ts, horizon, model, pts, log_fn=log_fn, cancel_flag=cancel_flag)
            if result.get("cancelled"):
                _jobs[job_id]["status"] = "cancelled"
                _jobs[job_id]["exit_code"] = 0
            elif result.get("error"):
                _jobs[job_id]["lines"].append(f"Error: {result['error']}")
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["exit_code"] = -1
            else:
                _jobs[job_id]["result"]     = result
                _jobs[job_id]["status"]     = "done"
                _jobs[job_id]["exit_code"]  = 0
        except Exception as exc:
            _jobs[job_id]["lines"].append(f"Error: {exc}")
            _jobs[job_id]["status"]     = "failed"
            _jobs[job_id]["exit_code"]  = -1

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": job_id}


@app.get("/segment-simulate-result/{job_id}")
async def segment_simulate_result(job_id: str):
    """Retrieve the result of a completed simulation job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "done":
        raise HTTPException(409, f"Job not done yet (status: {job['status']})")
    return JSONResponse(job["result"])


@app.post("/cancel-simulation/{job_id}")
async def cancel_simulation(job_id: str):
    """Signal a running simulation job to stop between steps."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "running":
        return {"status": job["status"]}
    cancel_event = job.get("cancel_event")
    if cancel_event:
        cancel_event.set()
    return {"status": "cancelling"}


@app.get("/segment-simulate/{segment}")
async def simulate_segment(
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
    loop      = asyncio.get_event_loop()
    result    = await loop.run_in_executor(None, _run_segment_simulation, segment, cutoff_ts, horizon, model, pts)
    if "error" in result:
        raise HTTPException(500, detail=result["error"])
    return JSONResponse(result)


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


@app.get("/backtest-cycles")
async def get_backtest_cycles(test: bool = Query(default=False)):
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


@app.get("/segments")
async def get_segments(
    weeks: int = Query(default=10, ge=1, le=52, description="Number of completed weeks to sum demand over"),
    product_type: str = Query(default="All", description="Comma-separated product types, or 'All'"),
):
    pts = None if product_type == "All" else [p.strip() for p in product_type.split(",") if p.strip()]
    return JSONResponse(read_segments(weeks, product_types=pts))


async def _segment_detail_intermittent(weeks: int, product_types: list[str] | None = None) -> JSONResponse:
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
            FROM shipcore.fc_velocity_link_snapshot v
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
async def get_segment_detail(
    segment: str,
    weeks: int = Query(default=10, ge=1, le=52),
    product_type: str = Query(default="All"),
    mode: str = Query(default="forward", description="'forward' = latest forecast; 'backtest' = evaluate a completed forecast run"),
    eval_date: str | None = Query(default=None, description="Backtest only: the forecast_date of the run to evaluate (YYYY-MM-DD)."),
    test: bool = Query(default=False, description="Use fc_forward_forecasts_test instead of the live table."),
):
    """Return per-SKU rows for a given segment (smooth_full | smooth_short | intermittent)."""
    pts = _parse_product_types(product_type)
    if segment == "intermittent":
        return await _segment_detail_intermittent(weeks, product_types=pts)

    if segment == "smooth_full":
        where_forward  = "f.bucket = 'smooth' AND f.history_length IN ('full', 'medium')"
        where_backtest = "f.bucket = 'smooth'"   # re-classify by active_weeks in Python
    elif segment == "smooth_short":
        where_forward  = "f.bucket = 'smooth' AND f.history_length = 'short'"
        where_backtest = "f.bucket = 'smooth'"   # re-classify by active_weeks in Python
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
        # Fetch all smooth SKUs; segment membership is re-derived in Python by active_weeks_at_eval.
        sql = f"""
            WITH ranked AS (
                SELECT f.unique_id, f.bucket, f.history_length, f.selected_model,
                       f.yhat, f.yhat_lo, f.yhat_hi, f.confidence
                FROM {fcast_table} f
                WHERE f.forecast_date = :eval_date
                  AND {where_backtest}
                  AND {pt_fcast}
            ),
            sku_agg AS (
                SELECT unique_id, bucket, history_length, selected_model, confidence,
                       SUM(ROUND(GREATEST(yhat, 0)))                                           AS yhat_total,
                       SUM(CASE WHEN yhat_lo IS NOT NULL THEN ROUND(GREATEST(yhat_lo, 0)) END) AS yhat_lo_total,
                       SUM(CASE WHEN yhat_hi IS NOT NULL THEN ROUND(GREATEST(yhat_hi, 0)) END) AS yhat_hi_total
                FROM ranked
                GROUP BY unique_id, bucket, history_length, selected_model, confidence
            ),
            demand AS (
                SELECT link_master_sku, SUM(link_qty) AS demand_total
                FROM shipcore.fc_velocity_link_snapshot
                WHERE order_date >= :horizon_start
                  AND order_date <= :horizon_end
                  AND {pt_snap}
                GROUP BY link_master_sku
            )
            SELECT a.unique_id, a.bucket, a.history_length, a.selected_model, a.confidence,
                   COALESCE(a.yhat_total, 0)::int                                              AS yhat_total,
                   CASE WHEN a.yhat_lo_total IS NOT NULL THEN a.yhat_lo_total::int END         AS yhat_lo_total,
                   CASE WHEN a.yhat_hi_total IS NOT NULL THEN a.yhat_hi_total::int END         AS yhat_hi_total,
                   COALESCE(d.demand_total, 0)                                                 AS demand_total
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
                       f.yhat, f.yhat_lo, f.yhat_hi, f.confidence,
                       ROW_NUMBER() OVER (PARTITION BY f.unique_id ORDER BY f.ds) AS week_num
                FROM {fcast_table} f
                INNER JOIN latest_dates ld
                    ON f.unique_id = ld.unique_id AND f.forecast_date = ld.latest_date
                WHERE {where_forward}
                  AND {pt_fcast}
            ),
            sku_agg AS (
                SELECT unique_id, bucket, history_length, selected_model, confidence,
                       SUM(ROUND(GREATEST(yhat, 0)))                                           FILTER (WHERE week_num <= :weeks) AS yhat_total,
                       SUM(CASE WHEN yhat_lo IS NOT NULL THEN ROUND(GREATEST(yhat_lo, 0)) END) FILTER (WHERE week_num <= :weeks) AS yhat_lo_total,
                       SUM(CASE WHEN yhat_hi IS NOT NULL THEN ROUND(GREATEST(yhat_hi, 0)) END) FILTER (WHERE week_num <= :weeks) AS yhat_hi_total
                FROM ranked
                GROUP BY unique_id, bucket, history_length, selected_model, confidence
            ),
            demand AS (
                SELECT link_master_sku, SUM(link_qty) AS demand_total
                FROM shipcore.fc_velocity_link_snapshot
                WHERE order_date > :demand_start
                  AND order_date <= :demand_end
                  AND {pt_snap}
                GROUP BY link_master_sku
            )
            SELECT a.unique_id, a.bucket, a.history_length, a.selected_model, a.confidence,
                   COALESCE(a.yhat_total, 0)::int                                              AS yhat_total,
                   CASE WHEN a.yhat_lo_total IS NOT NULL THEN a.yhat_lo_total::int END         AS yhat_lo_total,
                   CASE WHEN a.yhat_hi_total IS NOT NULL THEN a.yhat_hi_total::int END         AS yhat_hi_total,
                   COALESCE(d.demand_total, 0)                                                 AS demand_total
            FROM sku_agg a
            LEFT JOIN demand d ON d.link_master_sku = a.unique_id
            ORDER BY a.yhat_total DESC NULLS LAST
        """
        params = {"weeks": weeks, "demand_start": demand_start.date(), "demand_end": demand_end.date()}

    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()

    SHORT_THRESHOLD = 52
    MIN_HISTORY_WEEKS = 20

    # active_weeks: always derived from sku_profiles.csv train_start (same pipeline/zero-filled data).
    # In backtest mode for any smooth segment, measure train_start → eval_date to get history
    # available at forecast time. This re-derives segment membership retroactively, overriding
    # whatever history_length was stored in the DB at eval_date.
    profiles_path = ROOT / "data" / "processed" / "sku_profiles.csv"
    active_weeks_map: dict[str, int] = {}
    current_bucket_map: dict[str, str] = {}  # uid → current bucket, used to drop stale DB rows
    if profiles_path.exists():
        if mode == "backtest" and segment in ("smooth_full", "smooth_short"):
            prof = pd.read_csv(profiles_path, usecols=["unique_id", "train_start"])
            eval_dt = pd.Timestamp(eval_date)
            for _, row in prof.iterrows():
                ts = pd.Timestamp(row["train_start"])
                active_weeks_map[row["unique_id"]] = max(0, (eval_dt - ts).days // 7)
        else:
            prof = pd.read_csv(profiles_path, usecols=["unique_id", "active_weeks", "bucket"])
            active_weeks_map = dict(zip(prof["unique_id"], prof["active_weeks"].astype(int)))
            current_bucket_map = dict(zip(prof["unique_id"], prof["bucket"]))

    skus = []
    for r in rows:
        uid = r[0]
        aw = active_weeks_map.get(uid)
        if mode == "backtest" and segment in ("smooth_full", "smooth_short"):
            # Re-classify by active_weeks at eval_date, ignoring history_length stored in DB.
            if aw is None or aw < MIN_HISTORY_WEEKS:
                continue  # too little history at eval_date — exclude entirely
            if segment == "smooth_full" and aw < SHORT_THRESHOLD:
                continue  # was short at eval_date — belongs in smooth_short
            if segment == "smooth_short" and aw >= SHORT_THRESHOLD:
                continue  # was full at eval_date — belongs in smooth_full
        elif mode == "forward" and segment in ("smooth_full", "smooth_short") and current_bucket_map:
            # Drop stale DB rows for SKUs reclassified since the last run.
            # If a SKU was reclassified to intermittent/low_volume, the pipeline
            # writes no new forecast — old smooth rows stay as "latest" in the DB.
            if current_bucket_map.get(uid) != "smooth":
                continue
        weeks_to_grad = max(0, SHORT_THRESHOLD - aw) if aw is not None else None
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
async def sku_search(q: str = Query(default="", min_length=1)):
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
async def run_forecast(horizon: int = Query(default=13, ge=1, le=104)):
    """Spawn a background forecast job and return a job_id to poll."""
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {"status": "running", "lines": [], "exit_code": None, "proc": None}

    def _run():
        try:
            proc = subprocess.Popen(
                [sys.executable, str(ROOT / "scripts" / "run_forward_forecast.py"),
                 "--horizon", str(horizon)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(ROOT),
            )
            _jobs[job_id]["proc"] = proc
            for line in proc.stdout:
                _jobs[job_id]["lines"].append(line.rstrip())
            proc.wait()
            _jobs[job_id]["exit_code"] = proc.returncode
            if _jobs[job_id].get("cancelled"):
                _jobs[job_id]["status"] = "cancelled"
            else:
                _jobs[job_id]["status"] = "done" if proc.returncode == 0 else "failed"
        except Exception as exc:
            _jobs[job_id]["lines"].append(f"Error: {exc}")
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["exit_code"] = -1

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id}


@app.post("/cancel-forecast/{job_id}")
async def cancel_forecast(job_id: str):
    """Terminate a running forecast job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "running":
        return {"ok": False, "reason": "Job is not running"}
    job["cancelled"] = True
    proc = job.get("proc")
    if proc:
        proc.terminate()
    return {"ok": True}


@app.get("/forecast-status/{job_id}")
async def forecast_status(job_id: str):
    """Poll the status of a running or completed forecast job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id":     job_id,
        "status":     job["status"],
        "lines":      job["lines"],
        "exit_code":  job["exit_code"],
    }


@app.get("/forecast-last-run")
async def forecast_last_run():
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
