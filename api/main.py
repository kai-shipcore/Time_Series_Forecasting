"""The forecast API. Every endpoint here serves the LightGBM track.

This file used to hold both forecasters. The statsforecast endpoints moved to
`api/legacy/` on 2026-08-13 and are no longer mounted, because the two screens
that called them were deleted the same day. That package is kept as a record of
the work; `api/legacy/__init__.py` explains what it was and
`src/legacy/__init__.py` covers its model half.

The imports here were pruned twice, so if something looks conspicuously absent
-- plotly, the conformal-interval constants, the V1 helpers, sqlalchemy -- it is
because only the legacy endpoints used it and it went with them.
"""

import sys
import os
import json
import signal
import threading
import subprocess
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from src.db import (
    create_job, touch_job, set_job_pgid, finish_job, get_job,
    job_cancel_requested, request_job_cancel, recover_orphaned_jobs, cleanup_old_jobs,
    ensure_jobs_table, ensure_indexes,
)
from pydantic import BaseModel
from api.common import JobLogger


app = FastAPI(title="Coverland Forecast API")


FORECAST_API_TOKEN = os.getenv("FORECAST_API_TOKEN")

@app.middleware("http")
async def _token_auth(request, call_next):
    if FORECAST_API_TOKEN and request.url.path != "/health":
        if request.headers.get("x-forecast-token") != FORECAST_API_TOKEN:
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
    return await call_next(request)


# The statsforecast router is NOT mounted, as of 2026-08-13.
#
# api/legacy/ still holds its sixteen endpoints and they still work. Nothing
# calls them: the two screens they served, the Demand Forecast page and SKU
# Planning's forecast tab, were deleted from the Next.js app on the same day.
# Serving endpoints nobody calls is not free here, because every one of them
# sits behind FORECAST_API_TOKEN on a host with no packet filtering, and two of
# them spawn a pipeline subprocess.
#
# The code is kept as a record of the work rather than deleted; see
# api/legacy/__init__.py. To bring it back, uncomment these two lines. The route
# expectation in outputs/reports/route_parity.json would then need re-recording
# with `scripts/check_route_parity.py --write`, which is deliberate friction: it
# makes remounting eighteen endpoints show up as a diff in review.
#
# from api.legacy import router as legacy_router  # noqa: E402
# app.include_router(legacy_router)


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



def _read_deployed_commit() -> str | None:
    """The git commit this process was started from, or None outside a deploy.

    Written by .github/workflows/ci-cd.yml into DEPLOY_PATH/.deployed_commit
    after the rsync and BEFORE the service restart, so the value describes the
    code that is about to run rather than the code that was running.

    Read ONCE, at import, deliberately. Reading per request would report
    whatever the file says now, which after a rewrite without a restart is a
    claim about code this process is not running. A stale-but-true answer is
    worth more than a fresh-but-wrong one, since the entire purpose of the field
    is to answer "is the thing serving traffic the thing that was deployed".

    Absent on a developer machine and on any clone, which is why it is optional
    rather than an error: nothing else about this service requires a deploy.
    """
    try:
        return (ROOT / ".deployed_commit").read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


DEPLOYED_COMMIT = _read_deployed_commit()


@app.get("/health")
async def health():
    """Liveness, plus whether the data this service reads is actually present.

    Still returns 200 when data is missing. The process is alive and the caller
    asked whether it is, so answering 503 would conflate "no server" with
    "server with no data", which are different problems with different fixes.
    The distinction is in the body, and it is the whole point of the endpoint:
    data/processed and outputs/reports are gitignored, so a fresh clone raises
    on every endpoint while looking perfectly healthy here.

    Left outside the token check with the rest of /health, so a client that has
    the token wrong can still find out the server is up.
    """
    try:
        status = _plan_data.readiness()
    except Exception as exc:  # pragma: no cover - readiness must never 500
        return {
            "status": "ok",
            "ready": None,
            "readiness_error": str(exc),
            "commit": DEPLOYED_COMMIT,
        }

    # Whether the accuracy report still describes the pinned snapshot and the
    # served population.
    #
    # Deliberately not part of `ready`. A drifted report serves correctly: the
    # grid renders, and its figures are real measurements of a real cohort. The
    # only thing wrong is that the cohort has moved on, which is a caption
    # problem rather than an outage, and failing readiness over it would take
    # both planning screens down and auto-start a server that was already up.
    #
    # It is here because this is the endpoint the cron already calls. The
    # 2026-08-11 re-profile went undetected for two weeks with every existing
    # check passing, because they all ask whether files are present and none
    # asked whether they agree.
    try:
        accuracy = _prov.drift_summary(_ML_DATA_SNAPSHOT)
    except Exception as exc:  # pragma: no cover - a caption must not 500 /health
        accuracy = {"ok": None, "known": False, "detail": f"drift check failed: {exc}"}

    # Whether the weekly pipeline is still delivering. Reported apart from the
    # accuracy check above because the two failures are unrelated and their
    # fixes are different: this is a cron that stopped, that is a report nobody
    # regenerated. Also not part of `ready`, for the same reason: last week's
    # forecast is a worse forecast, not an outage, and 503 here would take both
    # planning screens down over a forecast that still works.
    try:
        freshness = _prov.freshness_summary()
    except Exception as exc:  # pragma: no cover
        freshness = {"ok": None, "detail": f"freshness check failed: {exc}"}

    return {
        "status": "ok",
        "ready": status["ready"],
        "missing_required": status["missing_required"],
        "missing_optional": status["missing_optional"],
        "files": status["files"],
        "repo_root": status["repo_root"],
        "accuracy_report": accuracy,
        "data_freshness": freshness,
        # Which revision is actually serving. repo_root answers "started from
        # this directory", which a process left over from an earlier deploy also
        # satisfies; this answers "started from this commit", which it does not.
        # That difference is what BACKLOG 20 and 21 both turn on: a push that
        # never deployed and a deploy that never took the port are both a commit
        # mismatch here, and were both invisible before.
        "commit": DEPLOYED_COMMIT,
    }




@app.post("/planning/run-forecast")
def planning_run_forecast(horizon: int = Query(default=13, ge=13, le=104)):
    """Refresh the data and produce a new ML forward forecast, on demand.

    Runs scripts/ml_prepare_data.py --force: velocity sync, ingest, clean,
    profile, forward forecast. The same script the empty-machine path uses,
    with --force because here the files exist and rebuilding them is the point.

    This is now the same script the weekly cron runs. Until 2026-08-13 it was
    not: the cron ran the legacy statsforecast pipeline first, because that was
    the only thing that wrote sales_clean.parquet, and this button deliberately
    skipped it to avoid spending most of its runtime on a forecast no screen
    read. The cron now calls ml_prepare_data.py too, so the button and the
    weekly run finally do the same work, and the caveat that used to belong here
    -- that SKU Planning would still show the older forecast afterwards -- is
    gone with the page it was about.

    Profiling is in the pipeline rather than skipped alongside the model
    selection it sits next to. It writes sku_profiles.csv, which the planning
    table reads to decide which SKUs belong on the worklist; refreshing sales
    without it leaves segmentation describing last week while the forecast
    describes this week, which is the drift `demoted_since_forecast` counts.

    Async, returning a job_id to poll. Shares the "forecast" job type with
    prepare-data, so create_job refuses when either is already going: they write
    the same files, and two at once is a corrupted parquet rather than a slow
    afternoon. The legacy run was the third holder of that job type until it was
    retired on 2026-08-13.

    Horizon floors at 13 in the signature rather than in the script. Each run
    REPLACES the stored forecast for its training week, so a shorter one would
    clobber a full snapshot.
    """
    job_id = create_job("forecast")
    if job_id is None:
        raise HTTPException(status_code=409, detail="A forecast job is already in progress")

    def _run():
        logger = JobLogger(job_id)
        try:
            proc = subprocess.Popen(
                [sys.executable, str(ROOT / "scripts" / "ml_prepare_data.py"),
                 "--force", "--horizon", str(horizon)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                cwd=str(ROOT), start_new_session=True,   # own group → killable with children
            )
            try:
                set_job_pgid(job_id, os.getpgid(proc.pid))
            except Exception:
                pass

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
        except Exception as exc:  # noqa: BLE001 — a crashed thread must still close the job
            logger.append(f"Error: {exc}")
            logger.flush()
            finish_job(job_id, "failed", exit_code=-1)

    threading.Thread(target=_run, daemon=True).start()
    # Polled through the existing generic /forecast-status/{job_id} and
    # cancelled through /cancel-forecast/{job_id}; both read the jobs table by
    # id and know nothing about which pipeline produced the job.
    return {"job_id": job_id}


@app.post("/planning/prepare-data")
def planning_prepare_data(horizon: int = Query(default=13, ge=1, le=104)):
    """Build the ML data files from the database, for a machine that has none.

    The counterpart to /planning/run-forecast, and a separate endpoint rather
    than a flag on it because the two differ in intent rather than mechanics:
    this one builds files that are missing, that one rebuilds files that exist.
    Both now run scripts/ml_prepare_data.py, the second with --force.

    This paragraph used to warn against repairing missing ML data with the
    legacy /run-forecast endpoint, which regenerated sku_profiles.csv while
    writing its forecasts to shipcore.fc_forward_forecasts, moving segmentation
    underneath an unchanged ML forecast (docs/BACKLOG.md item 7). That endpoint
    was unmounted on 2026-08-13, so the trap is closed rather than avoided.

    Async, returning a job_id to poll, because this is minutes of work and a
    request cannot wait for it. Reuses the same job machinery the Run Forecast
    panel already polls, so the client side is a button rather than a mechanism.

    Refuses when the data is already there. The caller reaching this endpoint
    means readiness said something was missing, so a request that arrives when
    nothing is is either a stale page or a second click, and rebuilding live
    data from a page load is exactly what should not happen on the server.
    """
    status = _plan_data.readiness()
    if status["ready"] and not status["missing_required"]:
        raise HTTPException(
            status_code=409,
            detail="The data files are already present; nothing to prepare.",
        )

    job_id = create_job("forecast")
    if job_id is None:
        raise HTTPException(status_code=409, detail="A forecast job is already in progress")

    def _run():
        logger = JobLogger(job_id)
        try:
            proc = subprocess.Popen(
                [sys.executable, str(ROOT / "scripts" / "ml_prepare_data.py"),
                 "--horizon", str(horizon)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                cwd=str(ROOT), start_new_session=True,
            )
            try:
                set_job_pgid(job_id, os.getpgid(proc.pid))
            except Exception:
                pass
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


# /forecast-last-run moved to api/legacy/routes.py on 2026-08-13.
#
# It was grouped with the shared job endpoints when the tracks were split, on the
# strength of its name. That was wrong: it reads shipcore.fc_forecast_history,
# which only the statsforecast pipeline ever wrote and which nothing writes now.
# Left mounted it would have kept answering, with a run_date that silently aged,
# which is worse than the endpoint being absent.


# ─────────────────────────────────────────────────────────────────────────────
# Planning: the action list and the per-SKU detail view.
#
# These serve the Next.js planning page. They deliberately compute nothing of
# their own: every figure comes from src.planning, so there is one implementation
# of the recommended order quantity and no second place for it to disagree with
# itself. A Streamlit prototype used to render the same functions directly; it
# was retired on 2026-08-12, which removed a renderer rather than a calculation,
# as intended.
#
# These read a mixture. The v11 forward forecast lives in
# shipcore.ml_forward_forecasts, and falls back to a parquet when the database is
# unreachable. The older shipcore.fc_forward_forecasts still holds the last
# statsforecast output, frozen at the date that track stopped running. Inventory is a live
# query with a CSV fallback. Sales and the SKU profiles are still files only,
# and are what remains before this API could be deployed away from this repo.
# ─────────────────────────────────────────────────────────────────────────────

from src.planning import calc as _plan_calc  # noqa: E402
from src.planning import data as _plan_data  # noqa: E402
from src.planning import quality as _plan_quality  # noqa: E402


def _planning_params(
    lead_time_weeks: int,
    review_period_weeks: int,
    service_z: float,
    stockout_horizon_days: int,
) -> dict:
    """Merge request overrides onto the defaults, exactly as the sidebar does."""
    return {
        **_plan_calc.DEFAULT_PARAMS,
        "lead_time_weeks": int(lead_time_weeks),
        "review_period_weeks": int(review_period_weeks),
        "service_z": float(service_z),
        "stockout_horizon_days": int(stockout_horizon_days),
    }


def _jsonable(df: pd.DataFrame) -> list[dict]:
    """Rows as plain JSON. NaN and NaT become null rather than the literals
    "NaN"/"NaT", which are not valid JSON and which the browser would otherwise
    receive as strings and render as text."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    return out.replace({np.nan: None, pd.NaT: None}).to_dict(orient="records")


@app.get("/planning/action-list")
def planning_action_list(
    lead_time_weeks: int = Query(default=8, ge=1, le=52),
    review_period_weeks: int = Query(default=1, ge=1, le=13),
    service_z: float = Query(default=1.0, ge=0.0, le=4.0),
    stockout_horizon_days: int = Query(default=30, ge=1, le=365),
):
    """The action list: one row per forecastable SKU, with the recommended order
    quantity, its inputs, the priority, the stockout projection and the flags.

    The planning parameters are query arguments because they are the user's to
    choose: the same table at a 12-week lead time is a different worklist, and
    the recommendation has to move with them.
    """
    params = _planning_params(lead_time_weeks, review_period_weeks, service_z,
                              stockout_horizon_days)
    plan = _plan_calc.build_planning_table(params)
    flags = _plan_quality.flags_by_sku(plan)
    metrics = _plan_calc.overview_metrics(plan, params)

    rows = _jsonable(plan)
    for row in rows:
        row["flags"] = flags.get(str(row["unique_id"]), [])

    snapshot = _plan_data.forecast_snapshot_date()
    return JSONResponse({
        "params": params,
        "metrics": metrics,
        "rows": rows,
        "meta": {
            "sku_count": len(plan),
            # How many SKUs the other section holds. Returned here so the page
            # can label both halves of its toggle without fetching the
            # non-forecast payload, which is seven times the size and mostly
            # goes unread. It is the complement of this section by construction,
            # so the two always sum to the profiled universe.
            "not_forecast_count": max(int(len(_plan_data.load_profiles())) - len(plan), 0),
            # SKUs the forecast covered that the current profile has since
            # demoted to intermittent, and which are therefore not in `rows`.
            # Surfaced so a caller can reconcile against the forecast file
            # rather than finding the totals quietly short.
            "demoted_since_forecast": int(plan.attrs.get("demoted_since_forecast", 0)),
            "trained_through": snapshot.date().isoformat() if snapshot is not None else None,
            # Which model produced the forecast, and how far it reaches. The
            # header said when it was trained and nothing else, so a reader could
            # not tell which version they were looking at or where the horizon
            # ends, both of which the forecast file already carries.
            "model_version": (
                str(fc["model_version"].iloc[0])
                if not (fc := _plan_data.load_forecasts()).empty
                and "model_version" in fc.columns else None
            ),
            "horizon_end": (
                fc["ds"].max().date().isoformat() if not fc.empty else None
            ),
            "inventory_is_sample": bool(_plan_data.inventory_is_sample()),
        },
    })


@app.get("/planning/sku/{sku_id}")
def planning_sku_detail(
    sku_id: str,
    lead_time_weeks: int = Query(default=8, ge=1, le=52),
    review_period_weeks: int = Query(default=1, ge=1, le=13),
    service_z: float = Query(default=1.0, ge=0.0, le=4.0),
    stockout_horizon_days: int = Query(default=30, ge=1, le=365),
    history_weeks: int = Query(default=26, ge=4, le=260),
):
    """Everything the SKU detail view needs in one response: the planning row,
    the order-quantity breakdown as arithmetic, the plausible band, weekly
    history and forecast, and the backtest windows with their per-week
    predictions.

    One endpoint rather than several because the page shows them together and a
    partial render is worse than a slower one; the client should not have to
    orchestrate four calls to draw a single screen.
    """
    params = _planning_params(lead_time_weeks, review_period_weeks, service_z,
                              stockout_horizon_days)
    plan = _plan_calc.build_planning_table(params)
    match = plan[plan["unique_id"] == sku_id]
    if match.empty:
        # Three cases, not two. A SKU can be absent from this table because it
        # was in the forecast run and has since been demoted, or because it was
        # never forecast at all, or because it does not exist. The first two are
        # normal outcomes the page explains and draws history for; only the third
        # is an error. The previous wording called the middle case "Unknown SKU",
        # which is what every row on the non-forecast list is, so following a link
        # from there reported a failure instead of the page it was meant to reach.
        in_run = sku_id in set(_plan_data.load_forecasts()["unique_id"])
        profiles = _plan_data.load_profiles()
        profiled = sku_id in set(profiles["unique_id"])
        if in_run:
            detail = ("SKU is in the forecast run but the current segmentation classes it "
                      "intermittent, so it has no planning row")
        elif profiled:
            detail = ("SKU sells too irregularly to forecast weekly, so it has no planning "
                      "row. Its sales history is available")
        else:
            detail = "Unknown SKU"
        raise HTTPException(status_code=404, detail=detail)

    row = match.iloc[0]
    breakdown = _plan_calc.order_quantity_breakdown(row, params)
    # order_quantity_range removed with the band; see the note on "order" below.

    hist_all = _plan_data.sku_sales_history(sku_id)
    fc = _plan_data.sku_forecast(sku_id)
    windows = _plan_data.load_ml_accuracy_by_sku()
    version = _plan_data.load_forecasts()["model_version"].iloc[0] if not fc.empty else None
    all_versions = windows[windows["unique_id"] == sku_id]
    windows = all_versions
    if version is not None and "model_version" in windows.columns:
        windows = windows[windows["model_version"] == version]

    # This SKU's model against the spreadsheet, on the windows both were scored
    # on. The page draws both forecasts and lists both per week, and until now
    # said nothing about which had been closer FOR THIS SKU, which is the
    # question a purchaser deciding whether to trust the number is actually
    # asking. The portfolio answer is on the validation page and does not
    # transfer: V1's error is season-dependent, so a SKU can run opposite to it.
    #
    # Both errors are divided by the same actual, the one from the model's own
    # row, so the two WAPEs are like for like rather than each measured against
    # its own denominator. Same convention as the validation outlier lists.
    def _score(v: str) -> dict | None:
        rows = all_versions[all_versions["model_version"] == v]
        rows = rows[rows["y_total"] > 0]
        if rows.empty:
            return None
        actual = float(rows["y_total"].sum())
        return {
            "version": v,
            "wape": float(rows["ae"].sum() / actual),
            # Percentage points, positive meaning it forecast more than sold,
            # matching src/ml/evaluate.py and the validation grid.
            "bias_pct": float(100.0 * rows["bias"].sum() / actual),
            "ae_units": float(rows["ae"].sum()),
            "actual_units": actual,
            "n_windows": int(rows["window"].nunique()),
        }

    # History long enough to cover every backtest window drawn over it.
    #
    # `history_weeks` sizes the demand chart, and at its default of 26 it starts
    # in late January while the earliest backtest cutoff is the previous October.
    # Both charts read this one array, so the backtest chart was shading windows
    # and drawing predicted lines across sixteen weeks with no actuals beneath
    # them: the comparison the chart exists to make, missing exactly where the
    # oldest window sits. The four extra weeks put some observed demand to the
    # left of the earliest cutoff line, which is what that line divides.
    weeks_needed = history_weeks
    if not windows.empty and not hist_all.empty:
        earliest = pd.to_datetime(windows["cutoff"]).min()
        last_week = hist_all["ds"].max()
        if pd.notna(earliest) and pd.notna(last_week):
            weeks_needed = max(weeks_needed, int((last_week - earliest).days / 7) + 4)
    hist = hist_all.tail(weeks_needed)[["ds", "y"]]

    model_score = _score(version) if version else None
    baseline_score = _score("v1")
    comparison = None
    if model_score and baseline_score:
        shared = sorted(
            set(all_versions[all_versions["model_version"] == version]["window"])
            & set(all_versions[all_versions["model_version"] == "v1"]["window"])
        )
        comparison = {
            "model": model_score,
            "baseline": baseline_score,
            # Named rather than counted: a reader comparing two figures needs to
            # know they cover the same weeks, and "2 windows" does not say that.
            "windows": shared,
        }

    # The ordered SKU list, so the page can offer a selector and move between
    # SKUs without a round trip through the list. It is the planning table's own
    # order, which is the worklist order, so stepping through the selector walks
    # the same sequence the list shows. Roughly 10KB against a payload that
    # already carries weekly history, a forecast and every backtest week, so the
    # alternative of a second request costs more than the bytes.
    idx = plan.index[plan["unique_id"] == sku_id]
    position = int(plan.index.get_loc(idx[0])) if len(idx) else -1

    return JSONResponse({
        "params": params,
        "row": _jsonable(match)[0],
        "flags": _plan_quality.flags_by_sku(plan).get(sku_id, []),
        "skus": plan["unique_id"].tolist(),
        "position": position,
        "meta": {
            # Repeated from the action list rather than assumed: a user can land
            # here from a link without ever seeing that page, and a figure drawn
            # from sample inventory should say so wherever it appears.
            "inventory_is_sample": bool(_plan_data.inventory_is_sample()),
            "inventory_source": _plan_data.inventory_source(),
            # Same source as the action list's own figure, for the same reason.
            # Absent it the page had to infer the training week from the first
            # week of the horizon, which is one week later by construction, so
            # the two screens disagreed about how stale the forecast was.
            "trained_through": (
                snapshot.date().isoformat()
                if (snapshot := _plan_data.forecast_snapshot_date()) is not None
                else None
            ),
            # How many trailing weeks the demand chart should draw. `history` is
            # longer than this whenever a backtest window reaches further back,
            # so the client trims rather than the two charts disagreeing about
            # which weeks exist.
            "history_weeks": history_weeks,
        },
        # No plausible band. It flexed coverage demand by the SKU's error, which
        # is the same quantity safety stock adds, so at the default service level
        # its upper edge WAS the recommendation: the same figure presented twice,
        # once as a decision and once as a range that appeared to contain it.
        # Uncertainty is carried by the reliability card, which is measured.
        "order": {
            "total": int(row["recommended_order_qty"]),
            "breakdown": _jsonable(breakdown),
        },
        "history": _jsonable(hist),
        "forecast": _jsonable(fc),
        "comparison": comparison,
        "backtest": {
            "windows": _jsonable(windows.sort_values("cutoff")),
            "weekly": _jsonable(_plan_data.sku_backtest_weekly(sku_id, version)),
        },
    })


@app.get("/planning/sku/{sku_id}/history")
def planning_sku_history(
    sku_id: str,
    history_weeks: int = Query(default=104, ge=4, le=260),
):
    """Weekly actual demand for any SKU, forecastable or not.

    Separate from /planning/sku/{id} because that endpoint answers a planning
    question and correctly 404s for a SKU the model does not forecast: there is
    no order quantity, no coverage demand and no reliability without a forecast.
    Sales history exists regardless, and is the only thing the detail page can
    honestly show for an intermittent SKU.

    Reads the same weekly series the planning layer uses, so the line drawn here
    and the actuals drawn on a forecastable SKU's chart are the same measurement.
    """
    sales = _plan_data.load_sales()
    hist = sales[sales["unique_id"] == sku_id]
    if hist.empty:
        raise HTTPException(status_code=404, detail=f"No sales history for {sku_id}")

    hist = hist.sort_values("ds").tail(history_weeks)
    profiles = _plan_data.load_profiles()
    prof = profiles[profiles["unique_id"] == sku_id]
    return JSONResponse({
        "unique_id": sku_id,
        # Carried so the page can say why there is no forecast rather than only
        # that there is none.
        "bucket": (str(prof["bucket"].iloc[0]) if len(prof) and pd.notna(prof["bucket"].iloc[0])
                   else None),
        "weeks": int(len(hist)),
        "total_units": float(hist["y"].sum()),
        "history": _jsonable(hist[["ds", "y"]]),
    })


@app.get("/planning/not-forecast")
def planning_not_forecast(
    lead_time_weeks: int = Query(default=8, ge=1, le=52),
    review_period_weeks: int = Query(default=1, ge=1, le=13),
    service_z: float = Query(default=1.0, ge=0.0, le=4.0),
    stockout_horizon_days: int = Query(default=30, ge=1, le=365),
):
    """SKUs the model does not forecast: the intermittent tail.

    Roughly 87% of the SKU count and a fifth of recent unit volume. Nothing here
    is forecast-derived, and the payload deliberately carries no recommended
    order quantity: what can be stated without a forecast is recent demand, the
    rate it implies, the stock position, and how long that stock lasts at that
    rate. The reorder signal is a statement about timing, not a quantity.

    Only `lead_time_weeks` affects the result, through the reorder signal. The
    other parameters are accepted so a caller can forward the same query string
    it sends to the action list, rather than having to know which subset applies.
    """
    params = _planning_params(lead_time_weeks, review_period_weeks, service_z,
                              stockout_horizon_days)
    table = _plan_calc.build_not_forecast_table(params)
    return JSONResponse({
        "params": params,
        "metrics": _plan_calc.not_forecast_metrics(table, params),
        "rows": _jsonable(table),
        "meta": {
            "sku_count": int(len(table)),
            "window_weeks": _plan_calc.NOT_FORECAST_WEEKS,
            "inventory_is_sample": bool(_plan_data.inventory_is_sample()),
        },
    })


class DemandTrendRequest(BaseModel):
    """SKUs to aggregate over. Omitted or empty means every forecastable SKU."""
    skus: list[str] | None = None
    history_weeks: int = 26


@app.post("/planning/demand-trend")
def planning_demand_trend(req: DemandTrendRequest):
    """Weekly actuals and forward forecast, summed across a set of SKUs.

    POST rather than GET because the caller sends the SKU list its filters
    produced, which runs to hundreds of identifiers and past what a query string
    can carry reliably. The alternative, re-implementing the filters server-side,
    would put the same predicate in two places and let them drift.

    Aggregating server-side is not a convenience: the client holds one row per
    SKU with no weekly series in it, and shipping 400 SKUs of weekly history to
    the browser to sum it there would be far more data than the answer.
    """
    weeks = max(4, min(int(req.history_weeks), 260))
    fc = _plan_data.load_forecasts()
    sales = _plan_data.load_sales()
    if fc.empty:
        return JSONResponse({"actual": [], "forecast": [], "v1": [], "sku_count": 0})

    # Default population is the PLANNING TABLE, not the forecast file. Those
    # differ by the SKUs demoted to intermittent since the run, so defaulting to
    # the forecast would draw a chart over 447 SKUs above a table showing 432,
    # and the two would disagree for a reason nothing on screen explains. The
    # client always sends its filtered list, so this only governs a direct call,
    # which is exactly when nobody is watching for the discrepancy.
    ids = (
        set(req.skus) if req.skus
        else set(_plan_calc.build_planning_table(_plan_calc.DEFAULT_PARAMS)["unique_id"])
    )
    fc = fc[fc["unique_id"].isin(ids)]
    sales = sales[sales["unique_id"].isin(ids)]

    cutoff = sales["ds"].max() - pd.Timedelta(weeks=weeks) if not sales.empty else None
    hist = sales[sales["ds"] > cutoff] if cutoff is not None else sales
    actual = (hist.groupby("ds", as_index=False)["y"].sum()
              .rename(columns={"y": "value"})) if not hist.empty else pd.DataFrame(columns=["ds", "value"])
    model = (fc.groupby("ds", as_index=False)["yhat"].sum()
             .rename(columns={"yhat": "value"})) if not fc.empty else pd.DataFrame(columns=["ds", "value"])

    v1 = _plan_data.load_v1_forward()
    if not v1.empty:
        v1 = v1[v1["unique_id"].isin(ids)]
        v1_series = (v1.groupby("ds", as_index=False)["v1_yhat"].sum()
                     .rename(columns={"v1_yhat": "value"}))
        # V1 can cover fewer SKUs than the model. Where it does, the two lines
        # sum over different populations and are not directly comparable, so the
        # coverage is returned rather than left for the reader to assume.
        v1_coverage = float(len(ids & set(v1["unique_id"])) / max(len(ids), 1))
    else:
        v1_series, v1_coverage = pd.DataFrame(columns=["ds", "value"]), 0.0

    return JSONResponse({
        "actual": _jsonable(actual),
        "forecast": _jsonable(model),
        "v1": _jsonable(v1_series),
        "v1_coverage": v1_coverage,
        "sku_count": len(ids),
        "history_weeks": weeks,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Forecast validation: is the model better than the spreadsheet it replaces?
#
# Distinct from the demand-forecast page, which reports how the current model is
# doing, and from the action list, which is per-SKU and operational. This answers
# one question with a decision behind it, and it has to answer it honestly:
# where the spreadsheet still wins, and what the comparison cannot cover.
#
# Built to grow. Nothing here hardcodes a model version or a window count, so the
# final test window and every future run appear as they arrive.
# ─────────────────────────────────────────────────────────────────────────────

from config import ML_FINAL_TEST_CUTOFF as _ML_FINAL_TEST_CUTOFF  # noqa: E402
from config import ML_DATA_SNAPSHOT as _ML_DATA_SNAPSHOT  # noqa: E402
from src.ml.serving import history as _hist  # noqa: E402
from src.planning import provenance as _prov  # noqa: E402


def _model_card(version: str | None) -> dict | None:
    """What a registered model version is, for the page that reports on it.

    Read from the registry rather than restated in the web app, so a version
    this endpoint is not serving cannot be described as though it were. Returns
    None for a version with no registered class, which is the case for the V1
    spreadsheet baseline: it is a formula rather than a fitted model and is
    described where it is compared.

    `features` are the names the model is fitted on, per segment. They are
    deliberately raw rather than prettified: they are what appears in the design
    doc and the experiment scripts, and a reader following one to the other
    should find the same word.
    """
    try:
        from src.ml.serving.models import REGISTRY
        from src.ml.model import FEATURES_V1, FEATURES_V11_LONG
    except Exception:
        return None

    cls = REGISTRY.get(version or "")
    if cls is None:
        return None

    card = {
        "version": getattr(cls, "version", version),
        "description": getattr(cls, "description", None),
        # Only meaningful for the hybrid, which fits a different feature set per
        # segment. Guarded so a future single-model version reports nothing here
        # rather than a feature list it does not use.
        "features": None,
    }
    if getattr(cls, "version", None) in {"v11", "v14"}:
        card["features"] = {"short": list(FEATURES_V1), "long": list(FEATURES_V11_LONG)}
    return card


def _pooled(g: pd.DataFrame, err="ae", act="y_total") -> float:
    total = g[act].sum()
    return float(g[err].sum() / total) if total else float("nan")


# Default minimum volume for the per-SKU outlier lists, in units over a scored
# window. Stated here so the page and this endpoint cannot disagree about it.
#
# Why a threshold exists at all: the lists rank by the difference between the
# model's per-SKU WAPE and the baseline's, and that difference is bounded by the
# denominator. On the 2026-07-30 report the largest absolute delta is 4.94 in the
# 10-to-50-unit band against 0.48 above 500 units, so an unfiltered top-15 selects
# the smallest SKUs rather than the ones the model handles worst. The two lists
# together then carry 1.8% of scored demand, under a heading that tells a planner
# to read them first.
#
# Why 100 and not higher: 100 keeps 223 of 572 scored rows and 78% of scored
# demand eligible, which leaves a pool the top-15 is still an extreme of. The
# stricter candidates trade that away: 200 leaves 103 rows, 500 leaves 41, at
# which point a top-15 is over a third of everything eligible and the word
# "outlier" stops being accurate. Adjustable on the page, and displayed there,
# because the right number depends on what the reader is looking for.
OUTLIER_MIN_UNITS = 100


#: Where scripts/ml_41_final_test.py writes the quarantined window's result.
FINAL_TEST_RESULT = ROOT / "outputs" / "reports" / "final_test.json"


def _final_test_payload() -> dict:
    """The final test result, served from the file the runner wrote.

    The file is the source and this function does not restate its numbers. It
    passes `scores` and the provenance fields through unchanged, so a figure on
    the page and a figure in `final_test.json` cannot disagree without the file
    itself being wrong. That mattered enough to be worth the awkwardness: this
    project has three recorded instances of a number rotting because it was
    transcribed into prose rather than read from where it was measured.

    Two things are derived rather than passed through, both so the web app does
    not have to know a model version by name:

    `methods` names which key in `scores` plays which role. The comparison
    section above already reads versions out of its payload rather than naming
    v11 in a component, and this keeps that property here.

    `comparisons` flattens the `<model>_vs_<other>` blocks into a list. In the
    file they are keys containing a version name, which a TypeScript interface
    cannot describe without hardcoding that name. As a list each entry carries
    what it is comparing, so a future model changes the data and not the code.

    Returns `evaluated: False` when the file is absent, which is the honest
    answer on a fresh clone: the result is not regenerable, so a missing file
    means this checkout does not have it rather than that the test is pending.
    """
    fallback = {"cutoff": _ML_FINAL_TEST_CUTOFF, "evaluated": False}
    try:
        with FINAL_TEST_RESULT.open() as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        # Same posture as the rest of this endpoint: a section with no evidence
        # renders as "nothing here yet" rather than taking the page down.
        return fallback

    scores = raw.get("scores") or {}
    if not scores:
        return fallback

    # Roles by elimination. "baseline" and "V1" are fixed names written by the
    # runner; whatever else is in there is the model under test.
    spreadsheet = next((k for k in scores if k.upper() == "V1"), None)
    structural = next((k for k in scores if k.lower() == "baseline"), None)
    model = next((k for k in scores if k not in (spreadsheet, structural)), None)

    comparisons = []
    for key, block in raw.items():
        if "_vs_" not in key or not isinstance(block, dict):
            continue
        against = key.split("_vs_", 1)[1]
        against = spreadsheet if against.lower() == "v1" else against
        for segment, stats in block.items():
            if not isinstance(stats, dict) or "delta" not in stats:
                continue
            comparisons.append({
                "against": against,
                "segment": segment,
                "delta": stats.get("delta"),
                "se": stats.get("se"),
                "ci_lo": stats.get("ci_lo"),
                "ci_hi": stats.get("ci_hi"),
            })

    return {
        "cutoff": raw.get("cutoff") or _ML_FINAL_TEST_CUTOFF,
        "evaluated": True,
        "run_at": raw.get("run_at"),
        "commit": raw.get("commit"),
        "snapshot": raw.get("snapshot"),
        "test_weeks": raw.get("test_weeks") or [],
        "scores": scores,
        "methods": {
            "model": model,
            "spreadsheet": spreadsheet,
            "structural_baseline": structural,
        },
        "comparisons": comparisons,
        # The runner records pooled WAPE and the bootstrap only. The bias
        # figures for this window exist in ML_FORECAST_DESIGN.md Section 4.35
        # and are deliberately not copied in here, because a number typed into
        # a payload is exactly the kind that goes stale unnoticed. If they are
        # wanted on the page, ml_41_final_test.py should record them and this
        # passes them through like everything else.
        "has_bias": False,
    }


@app.get("/planning/validation")
def planning_validation(
    baseline: str = Query(default="v1", description="version to compare against"),
    top_n: int = Query(default=15, ge=1, le=100),
):
    """Everything the validation page needs, in one response.

    Sections with no evidence yet return empty lists rather than erroring, so the
    page can show what is coming without pretending it has arrived.
    """
    acc_path = ROOT / "outputs/reports/ml_accuracy.csv"
    sku_path = ROOT / "outputs/reports/ml_accuracy_by_sku.csv"
    served = _plan_calc  # noqa: F841 — kept for symmetry with the planning endpoints

    from src.ml.serving.models import CURRENT_BEST

    empty = {"grid": [], "headline": None, "versions": [], "windows": []}
    if not acc_path.exists():
        comparison = empty
    else:
        acc = pd.read_csv(acc_path)
        versions = sorted(acc["model_version"].unique().tolist())
        current = CURRENT_BEST if CURRENT_BEST in versions else (versions[-1] if versions else None)
        rows = []
        for (segment, window), g in acc.groupby(["segment", "window"]):
            cell = {"segment": segment, "window": window}
            for v in versions:
                sub = g[g["model_version"] == v]
                cell[v] = float(sub["pooled_wape"].iloc[0]) if len(sub) else None
                if v == current and len(sub):
                    cell["n_skus"] = int(sub["n_skus"].iloc[0])
                    cell["actual_units"] = float(sub["actual_units"].iloc[0])
                    cell["bias_pct"] = float(sub["bias_pct"].iloc[0])
            if cell.get(current) is not None and cell.get(baseline) is not None:
                cell["delta"] = cell[current] - cell[baseline]
                cell["winner"] = current if cell["delta"] < 0 else baseline
            rows.append(cell)

        # Unit-weighted, because pooled WAPE is unit-weighted within a cell and
        # averaging cells would let a small segment count as much as a large one.
        def weighted(v: str) -> float | None:
            s = acc[acc["model_version"] == v]
            s = s[s["segment"] != "TOTAL"]
            u = s["actual_units"].sum()
            return float((s["pooled_wape"] * s["actual_units"]).sum() / u) if u else None

        # Chronological, by the cutoff each window was scored at. Sorted by name
        # they read Dec-Feb, Mar-May, Oct-Dec, which is alphabetical order
        # presented as though it were time, and invites reading a trend across
        # columns that runs backwards.
        window_order = (
            acc[["window", "cutoff"]].drop_duplicates().sort_values("cutoff")["window"].tolist()
        )

        cur_w, base_w = weighted(current), weighted(baseline)
        comparison = {
            "grid": rows,
            "versions": versions,
            "current": current,
            "baseline": baseline,
            "windows": window_order,
            "headline": {
                "current": cur_w,
                "baseline": base_w,
                "improvement": (base_w - cur_w) / base_w if cur_w and base_w else None,
                "cells_won": sum(1 for r in rows if r.get("winner") == current),
                "cells_total": sum(1 for r in rows if "winner" in r),
            } if cur_w and base_w else None,
        }

    # Coverage: what the comparison can and cannot speak for.
    #
    # This is a join across the page's two clocks and it is the only figure that
    # is. `served` is live: today's planning table, which the Tuesday cron moves.
    # `scored` comes from the pinned accuracy report, which by design does not
    # move. Rendering their ratio as one percentage with one date, which is what
    # this did until 2026-08-14, states a fact about a single moment that never
    # existed.
    #
    # The gap has two causes and they are not interchangeable. The intended one
    # is promoted SKUs, whose training start moves with every profiling run and
    # which are therefore ineligible at any fixed cutoff (docs/BACKLOG.md item
    # 2). The unintended one is a report measured on a superseded population,
    # which is what `basis["drift"]` reports. The page attributed the whole gap
    # to the first cause in prose, so the second was invisible in exactly the
    # period it was true.
    #
    # Both dates travel with the counts now, so the caption can say which side
    # of the join each number comes from rather than implying they share one.
    plan = _plan_calc.build_planning_table(_plan_calc.DEFAULT_PARAMS)
    scored_ids = set()
    if sku_path.exists():
        sku_acc = pd.read_csv(sku_path)
        scored_ids = set(sku_acc.loc[sku_acc["model_version"] == comparison.get("current"),
                                     "unique_id"])
    served_ids = set(plan["unique_id"])
    accuracy_basis = _prov.accuracy_basis(_ML_DATA_SNAPSHOT)
    live_basis = _prov.live_basis()
    coverage = {
        "served": len(served_ids),
        "scored": len(served_ids & scored_ids),
        "unscored": len(served_ids - scored_ids),
        "share": (len(served_ids & scored_ids) / len(served_ids)) if served_ids else 0.0,
        # Which clock each side is on. The frontend renders these rather than
        # assuming both numbers are current, and cannot omit them by accident:
        # the caption reads one of two different sentences depending on drift.
        "served_as_of": live_basis["as_of"],
        "scored_as_of": accuracy_basis["computed_at"],
        "scored_snapshot": accuracy_basis["snapshot"],
        # Scored SKUs that are no longer served at all. Distinct from `unscored`,
        # which counts the other direction. A large value here means the report
        # is measuring products that have since left the forecast set, which is
        # the same staleness seen from the other end and the one the ratio hides
        # completely: dropping a scored SKU raises `share` by shrinking nothing
        # a reader can see.
        "scored_not_served": len(scored_ids - served_ids),
    }

    # Largest individual wins and losses, so the aggregate can be checked against
    # real products rather than taken on faith.
    #
    # The whole scored pool is returned rather than the two ranked lists. Ranking
    # here would fix the minimum volume at whatever this endpoint chose, and the
    # threshold is a judgement the reader has to be able to see and move; see
    # OUTLIER_MIN_UNITS above for why it exists. It is 572 rows and about 100 KB
    # on the current report, small enough that sending all of it costs less than
    # a round trip per adjustment.
    #
    # `top_n` is now how many rows the page displays per list, not how many are
    # sent. Ranking happens client-side, after the threshold is applied.
    outliers = {
        "rows": [],
        "top_n": top_n,
        "default_min_units": OUTLIER_MIN_UNITS,
        "scored_units": 0.0,
    }
    if sku_path.exists() and comparison.get("current"):
        s = pd.read_csv(sku_path)
        cur = s[s["model_version"] == comparison["current"]]
        base = s[s["model_version"] == baseline]
        j = cur.merge(base, on=["unique_id", "window"], suffixes=("_cur", "_base"))
        j = j[j["y_total_cur"] > 0]
        if len(j):
            j["wape_cur"] = j["ae_cur"] / j["y_total_cur"]
            j["wape_base"] = j["ae_base"] / j["y_total_cur"]
            j["delta"] = j["wape_cur"] - j["wape_base"]
            # Segment, in the same vocabulary every other section reports in:
            # medium and full collapse to long, matching src/ml/evaluate.py and
            # design Section 4.4. Carried per row so the page can ask whether a
            # regression is systematic (one segment losing) or scattered, which
            # is the difference between a model problem and a handful of odd
            # products. Taken from the current version's row, since the two
            # versions are scored on the same SKU-window and the label belongs
            # to the SKU rather than to whoever forecast it.
            j["segment"] = (
                j["bucket_cur"].fillna("?") + "/"
                + j["history_length_cur"].fillna("?").replace(
                    {"medium": "long", "full": "long"}
                )
            )
            cols = ["unique_id", "window", "segment", "y_total_cur",
                    "wape_cur", "wape_base", "delta"]
            # Both WAPEs use the actual from the current version's row, so the two
            # are divided by the same denominator and their difference is a like
            # for like comparison rather than an artefact of two different bases.
            outliers["rows"] = _jsonable(j.sort_values("delta", ascending=False)[cols])
            # The denominator for "what share of scored demand this list carries".
            # Taken over the unfiltered pool, since the point of the figure is to
            # measure the filtered list against everything that was scored.
            outliers["scored_units"] = float(j["y_total_cur"].sum())

    # Performance over time, from the accumulating history. Empty until enough
    # runs have been stored and their weeks have closed.
    try:
        perf = _hist.performance_by_run()
        run_index = _hist.runs()
    except Exception:
        perf, run_index = pd.DataFrame(), pd.DataFrame()

    return JSONResponse({
        "comparison": comparison,
        "coverage": coverage,
        "outliers": outliers,
        "over_time": {
            "runs": _jsonable(run_index) if len(run_index) else [],
            "performance": _jsonable(perf) if len(perf) else [],
            "last_complete_week": str(_hist.last_complete_week().date()),
        },
        # Where these figures came from and when.
        #
        # The page had none of this, which is the wrong way round: it is the
        # screen whose whole purpose is evidence, and it was the only one not
        # dating its own. The Action List says "Trained through" in its header,
        # and the absence here is how a forecast three weeks stale went
        # unnoticed while this page reported on it.
        #
        # Two dates, deliberately, because they answer different questions. The
        # snapshot is pinned so the accuracy figures cannot drift week to week;
        # a reader seeing the same numbers on two visits should know that is by
        # design rather than a broken refresh. `trained_through` is the served
        # forecast's own training week, which does move, and comparing the two
        # is how you see whether the model being validated is the one being
        # served.
        "meta": {
            # What "v11" actually is, served rather than written into the web
            # app. The description lives on the registered model class beside
            # the code that implements it, and the feature lists are the ones
            # the model is fitted on, so the page cannot describe a version it
            # is not serving. A short string on a screen is where a reader first
            # meets the model, and it was the one thing on the technical page
            # that assumed you already knew.
            "model": _model_card(comparison.get("current")),
            "snapshot": _ML_DATA_SNAPSHOT,
            # Recorded by ml_accuracy_report.py rather than inferred from the
            # summary file's mtime, which is what this was until 2026-08-14. An
            # mtime describes the filesystem, not the measurement: git checkout,
            # cp -r and every deploy rewrite it, so the page dated the report by
            # when the file last moved. Falls back to the mtime for a checkout
            # written before the manifest existed, and `basis.accuracy` says
            # which of the two a given figure is.
            "accuracy_computed": accuracy_basis["computed_at"],
            "trained_through": (
                snap.date().isoformat()
                if (snap := _plan_data.forecast_snapshot_date()) is not None else None
            ),
        },
        # Which clock each section runs on.
        #
        # The page has always had both kinds of figure and never said which was
        # which. Sections 02, 03 and 04 read data/processed and move with the
        # Tuesday cron, because they describe the business as it is now. Sections
        # 01, 05 and 06 read the pinned snapshot and do not move, because they
        # are measurements whose whole value is being comparable across model
        # versions. A reader with no way to tell them apart has to assume one or
        # the other, and either assumption is wrong for half the page.
        #
        # `accuracy.drift` is the part that could not be expressed at all
        # before: whether the pinned side has been superseded. It is reported
        # rather than repaired, and `src/planning/provenance.py` says why.
        "basis": {
            "live": live_basis,
            "accuracy": accuracy_basis,
        },
        # Served from outputs/reports/final_test.json. This was hardcoded to
        # `evaluated: False` with a comment reading "not yet run" until
        # 2026-08-14, which stopped being true on 2026-08-13 and left the last
        # section of the evidence page asserting the opposite of the result it
        # exists to report.
        "final_test": _final_test_payload(),
    })


@app.get("/planning/demand-patterns")
def planning_demand_patterns(weeks: int = Query(default=52, ge=13, le=260)):
    """Descriptive shape of demand, independent of any model.

    Weekly totals, concentration, and the segment mix. Nothing here is a
    forecast or an evaluation of one; it is the backdrop the rest is read
    against, and it is the one section that needs no model at all.
    """
    sales = _plan_data.load_sales()
    if sales.empty:
        return JSONResponse({"weekly": [], "concentration": [], "segments": [], "weeks": weeks})

    cutoff = sales["ds"].max() - pd.Timedelta(weeks=weeks)
    recent = sales[sales["ds"] > cutoff]

    # Split by whether the model forecasts the SKU at all. The intermittent tail
    # is 87% of the catalogue and about a fifth of volume, and its shape is a
    # fact about the business rather than about any model, so it belongs here
    # rather than in a section evaluating forecasts. Seeing the two side by side
    # is also the clearest statement of what the forecast does and does not
    # cover.
    served = set(_plan_data.load_forecasts()["unique_id"])
    recent = recent.assign(
        group=np.where(recent["unique_id"].isin(served), "forecast", "not_forecast")
    )
    wide = (recent.pivot_table(index="ds", columns="group", values="y",
                               aggfunc="sum", fill_value=0.0)
            .reset_index())
    for col in ("forecast", "not_forecast"):
        if col not in wide.columns:
            wide[col] = 0.0
    wide["units"] = wide["forecast"] + wide["not_forecast"]
    weekly = wide[["ds", "forecast", "not_forecast", "units"]]

    # Concentration: the share of demand carried by the top N% of SKUs. Answers
    # how much of the business a forecast has to get right to matter.
    per_sku = recent.groupby("unique_id")["y"].sum().sort_values(ascending=False)
    total = per_sku.sum()
    conc = []
    pareto = []
    if total > 0:
        cum = per_sku.cumsum() / total
        for pct in (0.05, 0.10, 0.20, 0.50):
            n = max(1, int(len(per_sku) * pct))
            conc.append({"sku_share": pct, "n_skus": n,
                         "demand_share": float(cum.iloc[n - 1])})

        # The same fact as `conc`, as a curve rather than four breakpoints.
        # Concentration is a distribution, and four rows make a reader add them
        # up mentally to see the shape; the curve states it directly.
        #
        # Downsampled to ~200 points. The catalogue is a few thousand SKUs and
        # the curve is smooth, so every point is bandwidth spent on detail no
        # screen can resolve. Sampled evenly by rank rather than by demand, so
        # the flat tail does not collapse to a couple of points.
        #
        # Both ends are pinned: the first point is (0, 0), which no SKU
        # produces, and the last is the final SKU, which even sampling can miss
        # and whose absence would leave the curve stopping short of 100%.
        n_sku = len(per_sku)
        cum_values = cum.to_numpy()
        step = max(1, n_sku // 200)
        idx = list(range(0, n_sku, step))
        if idx[-1] != n_sku - 1:
            idx.append(n_sku - 1)
        pareto = [{"sku_pct": 0.0, "demand_pct": 0.0}] + [
            {"sku_pct": (i + 1) / n_sku, "demand_pct": float(cum_values[i])}
            for i in idx
        ]

    prof = _plan_data.load_profiles()
    seg = []
    if not prof.empty:
        p = prof.copy()
        p["group"] = np.where(p["unique_id"].isin(served), "forecast", "not forecast")
        units = recent.groupby("unique_id")["y"].sum()
        p["units"] = p["unique_id"].map(units).fillna(0.0)
        for group, g in p.groupby("group"):
            seg.append({"group": group, "n_skus": int(len(g)),
                        "units": float(g["units"].sum())})

    return JSONResponse({
        "weekly": _jsonable(weekly),
        "concentration": conc,
        # The cumulative curve. `concentration` stays on the response: the chart
        # annotates one breakpoint and the interpretation line names it, and
        # both read it from here rather than recomputing it off the sampled
        # curve, which would put the stated figure a sample-interval away from
        # the true one.
        "pareto": pareto,
        "n_skus": int(len(per_sku)),
        "segments": seg,
        "weeks": weeks,
        # This section is live and should say so on its own, without the reader
        # having to hold the other payload's basis block in their head. `as_of`
        # is the newest week in the sales grid rather than today: they differ by
        # up to a week normally, and by however long the cron has been failing
        # otherwise, which is the case worth being able to see.
        "basis": _prov.live_basis(sales["ds"].max()),
    })


@app.get("/planning/demand-vs-forecast")
def planning_demand_vs_forecast(history_weeks: int = Query(default=26, ge=8, le=104)):
    """Weekly demand against what the stored runs predicted for those weeks.

    The ML counterpart of the old Demand Forecast page's trajectory chart, and
    read from the same kind of source: forecasts that were served before the
    outcome was known, scored as their weeks complete. That page reads
    `shipcore.fc_forward_forecasts` across many `forecast_date`s; this reads
    `ml_forecast_history`, which accumulates one entry per `week_of`. The two
    columns are named differently on purpose and mean different things: on the
    legacy side `forecast_date` is the calendar date of the run, on the ML side
    `week_of` is the training week. Both tables carry both concepts; only the
    legacy one had them named apart before 2026-08-12.

    Not the backtest windows. Those are a different claim, already answered by
    the comparison grid: the model refit at a cutoff and predicted forward
    knowing the modelling choices had been made across the whole period. This
    chart is about what the model actually said in advance, week by week.

    Consequence while the store is young: `predicted` is empty until runs
    accumulate, and the chart shows demand and the current forward horizon only.
    That is the honest state rather than a defect, and it fills itself.

    No prediction band. The legacy statsforecast model emits conformal
    intervals, which that chart drew around both the past predictions and the
    forward horizon and used as a calibration check. The LightGBM track emits a
    point forecast and nothing else.
    """
    from src.ml.serving.models import CURRENT_BEST

    sales = _plan_data.load_sales()
    scored = _hist.score_against_actuals(sales)

    fc = _plan_data.load_forecasts()
    version = None
    if not fc.empty and "model_version" in fc.columns:
        version = str(fc["model_version"].iloc[0])
    version = version or CURRENT_BEST

    def seg_of(df: pd.DataFrame) -> pd.Series:
        """The grid's vocabulary: medium and full report together as long,
        matching src/ml/evaluate.py and design Section 4.4."""
        bucket = df.get("bucket", pd.Series("?", index=df.index)).fillna("?")
        hist = df.get("history_length", pd.Series("?", index=df.index)).fillna("?")
        return bucket + "/" + hist.replace({"medium": "long", "full": "long"})

    def smooth_only(df: pd.DataFrame) -> pd.DataFrame:
        """Drop any row whose segment is not smooth.

        The model forecasts smooth SKUs and nothing else: serving/forecast.py
        filters to `bucket == "smooth"` and writes that literal, so a real run
        cannot produce another bucket. A row here saying otherwise is bad data,
        and charting it would show the model apparently predicting SKUs it
        declines to predict.

        Where it came from: seed_forecast_history.py stamped today's profile onto
        fabricated historical runs, so the 15 SKUs that were smooth on 2026-07-20
        and demoted since arrived labelled intermittent. That script is fixed and
        real runs supersede the fixture, but the guard stays, because the right
        response to a non-smooth row is to leave it out of a chart about the
        model rather than to trust whatever wrote it.
        """
        if df.empty or "segment" not in df.columns:
            return df
        return df[df["segment"].str.startswith("smooth/")]

    def by_segment(df: pd.DataFrame, aggs: dict, extra_keys: list | None = None) -> pd.DataFrame:
        keys = ["ds"] + (extra_keys or [])
        frames = []
        for seg, g in df.groupby("segment"):
            a = g.groupby(keys, as_index=False).agg(**aggs)
            a["segment"] = seg
            frames.append(a)
        a = df.groupby(keys, as_index=False).agg(**aggs)
        a["segment"] = "all"
        frames.append(a)
        return pd.concat(frames, ignore_index=True).sort_values(["segment"] + keys)

    # Forward horizon: the latest run only, which is what "the current forecast"
    # means and what the old chart drew beyond the marker.
    # Segment per SKU as the runs recorded it, not as the profile stands today.
    #
    # The actuals series used to take these from load_profiles(), which labels a
    # historical row with a current classification. That is the same error
    # corrected in the seed script and in the reliability sort: a SKU demoted
    # since the forecast ran arrived here labelled intermittent, so a chart about
    # what the model predicted showed segments the model does not predict.
    seg_by_sku: dict[str, str] = {}
    forward = pd.DataFrame(columns=["ds", "yhat", "v1", "n_skus", "segment"])
    forward_run_date = None
    fwd_skus: set = set()
    if not fc.empty:
        latest = fc["week_of"].max()
        fc = fc[fc["week_of"] == latest].copy()
        forward_run_date = latest.date().isoformat()
        fc["segment"] = seg_of(fc)
        fc = smooth_only(fc)
        # After the filter, so the SKU count the chart reports matches the SKUs
        # it actually draws.
        fwd_skus = set(fc["unique_id"])
        seg_by_sku.update(fc.drop_duplicates("unique_id").set_index("unique_id")["segment"])
        v1 = _plan_data.load_v1_forward()
        if not v1.empty:
            fc = fc.merge(v1[["unique_id", "ds", "v1_yhat"]], on=["unique_id", "ds"], how="left")
        else:
            fc["v1_yhat"] = float("nan")
        forward = by_segment(fc, {
            "yhat": ("yhat", "sum"),
            "v1": ("v1_yhat", "sum"),
            "n_skus": ("unique_id", "nunique"),
        })

    last_complete = _hist.last_complete_week()

    # Past predictions, per (week, lead), exactly as the old endpoint reports
    # them. Empty until the history store has runs whose weeks have closed.
    predicted = pd.DataFrame(columns=["ds", "lead", "yhat", "n_skus", "week_of", "segment"])
    leads: list = []
    history_version = None
    if not scored.empty:
        # One version at a time. The store is built to hold several so they can
        # be compared, which means summing them here would add two models'
        # predictions into a single line. Prefer the version the current forward
        # forecast came from; fall back to whichever version ran most recently,
        # so a store holding only sample or older rows still renders.
        available = scored["model_version"].unique().tolist()
        if version in available:
            history_version = version
        else:
            history_version = (
                scored.sort_values("week_of")["model_version"].iloc[-1]
            )
        scored = scored[scored["model_version"] == history_version].copy()
        scored["segment"] = seg_of(scored)
        scored = smooth_only(scored)
        # Only where the forward run did not already say. The forward forecast is
        # the newer statement of a SKU's segment, so it wins.
        for uid, seg in (
            scored.drop_duplicates("unique_id").set_index("unique_id")["segment"].items()
        ):
            seg_by_sku.setdefault(uid, seg)
        predicted = by_segment(
            scored,
            {
                "yhat": ("yhat", "sum"),
                "n_skus": ("unique_id", "nunique"),
                "week_of": ("week_of", "max"),
            },
            extra_keys=["lead"],
        )
        leads = sorted(int(x) for x in scored["lead"].unique())

    # Actuals over the same SKUs the forward horizon covers, so the demand line
    # and the forecast line describe one population rather than two.
    actual_skus = fwd_skus or (set(scored["unique_id"]) if not scored.empty else set())
    hist_sales = sales[sales["unique_id"].isin(actual_skus)].copy()
    start = last_complete - pd.Timedelta(weeks=history_weeks)
    hist_sales = hist_sales[(hist_sales["ds"] >= start) & (hist_sales["ds"] <= last_complete)]
    actuals = pd.DataFrame(columns=["ds", "y", "n_skus", "segment"])
    if not hist_sales.empty:
        hist_sales["segment"] = hist_sales["unique_id"].map(seg_by_sku)
        # A SKU with sales but in neither the forward run nor the scored history
        # has no segment this chart can honestly assign, so it is left out rather
        # than given today's label.
        hist_sales = hist_sales[hist_sales["segment"].notna()]
        hist_sales = smooth_only(hist_sales)
        actuals = by_segment(hist_sales, {
            "y": ("y", "sum"),
            "n_skus": ("unique_id", "nunique"),
        })

    segments = sorted(
        set(actuals["segment"].unique() if not actuals.empty else [])
        | set(forward["segment"].unique() if not forward.empty else [])
    ) if True else []
    segments = [s for s in segments if s != "all"]

    return JSONResponse({
        "actuals": _jsonable(actuals),
        "predicted": _jsonable(predicted),
        "forward": _jsonable(forward),
        "segments": segments,
        "leads": leads,
        "last_complete_week": last_complete.date().isoformat(),
        "forward_run_date": forward_run_date,
        "runs_stored": int(_hist.runs().shape[0]),
        "version": version,
        # Which version the predicted line is actually drawn from. Differs from
        # `version` when the store has no rows for the current model, which is
        # exactly the case while seeded sample data is standing in for real runs.
        "history_version": history_version,
        "has_intervals": False,
    })
