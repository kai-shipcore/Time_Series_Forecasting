#!/usr/bin/env bash
#
# Weekly forecast run, executed directly on the deployed server rather than a
# developer's Mac, so freshness does not depend on any individual machine being
# powered on. Writes straight into this checkout's data/processed and
# outputs/reports -- exactly where the running coverland-forecast-api service
# reads from -- so no push step is needed afterward the way
# scripts/push_data_to_server.sh is needed when the Mac produces the data.
#
# Cron (coverland user, on the server):
#   0 10 * * 2 cd /opt/coverland-forecast-api && scripts/run_forecast_cron.sh >> logs/forecast_cron.log 2>&1
#
# 10:00 UTC = 3am Pacific at the time this was set up. The server stays fixed
# UTC, so the Pacific wall-clock time drifts by an hour across the two DST
# transitions each year; re-adjust the cron line then if that matters.
#
# TUESDAY (day 2), not Monday, and this is load-bearing. A week runs Tuesday to
# Monday and is labelled by the Monday it ends on (src/weeks.py), so the bucket
# labelled Monday L is still open for the whole of Monday L. A Monday run can
# only use the bucket that ended the previous Monday, making every forecast a
# week staler than it needs to be. A Tuesday run picks up the week that closed
# hours earlier.
#
# This moved from Monday to Tuesday on 2026-08-06, together with clean.py
# reverting to closed="right" and last_complete_week restoring its extra Monday
# step. The three are one decision. If the week convention is ever changed
# again, this line changes with it, or the pipeline silently loses a week.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
[ -f .env ] && set -a && . ./.env && set +a

echo "=== $(date -u '+%Y-%m-%d %H:%M:%S UTC') : starting forecast run ==="

# One pipeline: sync, ingest, clean, profile, forecast.
#
# This used to be two, and the first of them was the statsforecast run, kept in
# the weekly job long after its forecasts stopped being used because it was the
# only thing that wrote data/processed/sales_clean.parquet. The ML run then read
# what it produced. That dependency was invisible in the worst way: deleting the
# statsforecast track would not have errored, the ML forecast would have carried
# on being served and quietly stopped moving.
#
# scripts/ml_prepare_data.py already did the whole ML-only sequence and had done
# since it was written, for the Action List's Run Forecast button. Pointing the
# cron at it removes the statsforecast dependency without adding a script, and
# gains the property the two-script version never had: it stages every artifact
# in a sibling directory and commits with os.replace only after the forecast has
# succeeded. A crash, a dropped SSH session or a failed step now leaves last
# week's files intact and being served, instead of a half-updated set where
# segmentation describes one week and sales describe another. That was the
# BACKLOG 15 fix, and until now the weekly run did not benefit from it.
#
# --force because it refuses to overwrite live files by default, which is the
# right default everywhere except here.
#
# The forecast step inside it runs ml_forward_forecast.py --snapshot live.
# "live" is load-bearing: the default is config.ML_DATA_SNAPSHOT, the pinned copy
# that exists so recorded evaluation figures cannot drift, and a weekly run
# against a frozen snapshot would produce the same forecast every week and look
# like it was working.
#
# The shipcore.fc_* tables are no longer written by this job. Their two screens
# were deleted in August 2026 and the statsforecast code is kept as a record
# under api/legacy/ and src/legacy/ rather than run. See docs/BACKLOG.md item 6.
ml_status=0

echo "--- forecast pipeline (sync, ingest, clean, profile, forecast) ---"
.venv/bin/python scripts/ml_prepare_data.py --force || ml_status=$?
if [ "$ml_status" -ne 0 ]; then
  echo "ERROR: the forecast pipeline failed with exit code $ml_status."
  echo "  Nothing was committed: the previous run's files are intact and still"
  echo "  being served, so the Action List and Forecast Validation show last"
  echo "  week's forecast rather than an error."
fi

if [ "$ml_status" -ne 0 ]; then
  exit "$ml_status"
fi

echo "Forecast runs complete. Checking readiness..."
body="$(curl --fail --silent --max-time 10 -H "x-forecast-token: ${FORECAST_API_TOKEN:-}" http://127.0.0.1:8000/health || true)"

if [ -z "$body" ]; then
  echo "WARNING: the service did not answer /health after the run."
  exit 1
fi

ready="$(printf '%s' "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("ready"))' 2>/dev/null)"
if [ "$ready" = "True" ]; then
  echo "Server reports ready."
else
  missing="$(printf '%s' "$body" | python3 -c 'import json,sys; print(", ".join(json.load(sys.stdin).get("missing_required") or []))' 2>/dev/null)"
  echo "WARNING: server still not ready after this run. Missing: ${missing:-unknown}"
  exit 1
fi

# Did this run actually advance the data?
#
# The pipeline exiting zero means it completed, not that it produced a newer
# week. An upstream table that has stopped receiving orders, a sync step that
# failed quietly, or a velocity snapshot nobody refreshed all produce a clean
# run over last week's data, and the result is served as though it were current.
#
# This is a non-zero exit, unlike the accuracy warning below, because it is the
# one condition where the forecast on the screen is wrong rather than merely
# described wrongly. Cron mails on it.
#
# The 2026-08-11 run is why. The week labelled 2026-08-10 closed the evening
# before and never arrived; the Action List and Forecast Validation both carried
# on serving a forecast trained through 2026-08-03, and it was found by hand
# three days later. DATA_AND_PIPELINE.md Section 4 predicted this exactly and
# left it as something a human should check against a calendar.
fresh_ok="$(printf '%s' "$body" | python3 -c 'import json,sys; print((json.load(sys.stdin).get("data_freshness") or {}).get("ok"))' 2>/dev/null)"
fresh_detail="$(printf '%s' "$body" | python3 -c 'import json,sys; print((json.load(sys.stdin).get("data_freshness") or {}).get("detail") or "")' 2>/dev/null)"
if [ "$fresh_ok" = "False" ]; then
  echo "ERROR: the run completed but the data did not advance."
  echo "  ${fresh_detail}"
  echo "  The forecast being served is stale. Check the velocity sync and the"
  echo "  upstream order table before trusting anything on the planning screens."
  exit 1
elif [ "$fresh_ok" = "True" ]; then
  echo "Data is current: ${fresh_detail}"
else
  echo "WARNING: could not determine whether the data advanced (${fresh_detail:-no detail})."
fi

# Does the pinned accuracy report still describe what is being served?
#
# This run just rewrote data/processed, which is what moves the served
# population. The accuracy report did not move and is not supposed to: it reads
# the snapshot named by config.ML_DATA_SNAPSHOT, so re-running it here would
# retrain three windows to produce identical bytes. What it needs instead is to
# be re-run when the snapshot is re-cut or the profiler is changed, and this is
# the check that notices either has happened.
#
# Why this exists. On 2026-08-11 the snapshot was re-cut with a re-profiled
# population, moving smooth/short from 382 SKUs to 247. Nothing re-ran the
# report, and for two weeks the Forecast Validation page reported accuracy for a
# cohort that no longer existed while every check in this script passed. They
# all ask whether files are present; none asked whether they agree.
#
# A warning rather than a non-zero exit, deliberately. The forecast this run
# produced is good and is being served; the accuracy caption beside it is what
# has gone stale, and exiting non-zero would mail the operator about a healthy
# forecast run. The page carries the same warning where a reader of the figures
# will see it, which is the place that actually matters.
acc_ok="$(printf '%s' "$body" | python3 -c 'import json,sys; print((json.load(sys.stdin).get("accuracy_report") or {}).get("ok"))' 2>/dev/null)"
if [ "$acc_ok" = "False" ]; then
  detail="$(printf '%s' "$body" | python3 -c 'import json,sys; print((json.load(sys.stdin).get("accuracy_report") or {}).get("detail") or "")' 2>/dev/null)"
  echo "WARNING: the accuracy report no longer matches what is being served."
  echo "  ${detail}"
  echo "  Fix: .venv/bin/python scripts/ml_accuracy_report.py, then commit"
  echo "  outputs/reports/ml_accuracy*.csv and ml_accuracy_meta.json."
  echo "  Forecast Validation sections 01 and 05, and the Action List's"
  echo "  reliability tiers, are computed from that report."
elif [ "$acc_ok" = "True" ]; then
  echo "Accuracy report matches the pinned snapshot and the served population."
else
  echo "NOTE: accuracy report provenance unknown (pre-manifest checkout, or the"
  echo "  report has never been run). Run scripts/ml_accuracy_report.py to record it."
fi

# Dated copy of the one artifact that cannot be rebuilt.
#
# Everything else this run wrote regenerates from the database. The accumulating
# history does not: it records what was predicted before the outcome was known,
# and re-running past versions against past cutoffs would produce backtest
# figures, which is a different and weaker claim.
#
# The run also writes it to shipcore.ml_forecast_history, which is the real fix
# and is backed up with the rest of the database. This is the belt to that
# braces, and it is what covers the case the table cannot cover: a run where the
# database was unreachable, which is exactly when the file is the only copy.
#
# Keeps the last 12 weeks. Older ones are strictly contained in newer ones,
# since the store only grows, so retention here is about surviving a corrupt
# write rather than about depth of history.
HIST="data/processed/ml_forecast_history.parquet"
if [ -f "$HIST" ]; then
  mkdir -p data/history_backups
  cp "$HIST" "data/history_backups/ml_forecast_history_$(date -u '+%Y-%m-%d').parquet"
  ls -1t data/history_backups/ml_forecast_history_*.parquet 2>/dev/null \
    | tail -n +13 | xargs -r rm --
  echo "History backed up: $(ls -1 data/history_backups | wc -l | tr -d ' ') copies retained."
else
  echo "WARNING: $HIST does not exist after a successful run; nothing to back up."
fi

# There is no second status to check any more. When the job was two pipelines this
# is where a legacy failure that the ML run had survived was turned into a non-zero
# exit, so cron would mail about it. One pipeline has one status, and it exits above.
#
# Note for anyone re-adding a step: `set -u` is on, so referring to a status
# variable that is never assigned aborts the script rather than being treated as
# empty. That is the desired behaviour and it is why this block was deleted rather
# than left pointing at a variable that no longer exists.
echo "=== $(date -u '+%Y-%m-%d %H:%M:%S UTC') : done ==="
