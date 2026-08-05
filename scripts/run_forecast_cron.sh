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
#   0 10 * * 1 cd /opt/coverland-forecast-api && scripts/run_forecast_cron.sh >> logs/forecast_cron.log 2>&1
#
# 10:00 UTC = 3am Pacific at the time this was set up. The server stays fixed
# UTC, so the Pacific wall-clock time drifts by an hour across the two DST
# transitions each year; re-adjust the cron line then if that matters.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
[ -f .env ] && set -a && . ./.env && set +a

echo "=== $(date -u '+%Y-%m-%d %H:%M:%S UTC') : starting forecast run ==="

# Two pipelines, in this order, because they are not independent.
#
# The legacy statsforecast run comes first and does the ingest: it pulls fresh
# order lines from the database and writes data/processed/sales_clean.parquet.
# It also writes the shipcore.fc_* tables, which SKU Planning still reads
# (docs/BACKLOG.md item 6 keeps that path in service pending a wider refactor),
# so it is not optional even though the two live planning screens no longer use
# its forecasts.
#
# The ML run comes second and has no ingest of its own. It reads the sales file
# the first run just refreshed, and writes ml_forward_forecasts plus a row per
# SKU into the accumulating ml_forecast_history. Until 2026-08-04 this script
# ran only the first of the two, so the Action List and Forecast Validation were
# serving whichever forecast someone had last produced by hand, and the history
# store never gained a real run. The readiness check and the history backup at
# the end of this file were both already written for the ML run, which is what
# made the omission easy to miss.
#
# --snapshot live is load-bearing. Without it the ML script defaults to
# config.ML_DATA_SNAPSHOT, the pinned copy that exists so recorded evaluation
# figures cannot drift. A weekly forward run against a frozen snapshot would
# produce the same forecast every week and look like it was working.
legacy_status=0
ml_status=0

echo "--- legacy statsforecast run (ingest + shipcore.fc_*) ---"
.venv/bin/python scripts/run_forward_forecast.py || legacy_status=$?
if [ "$legacy_status" -ne 0 ]; then
  echo "WARNING: legacy forecast run failed with exit code $legacy_status."
  echo "  SKU Planning will be stale, and the ML run below reads the sales file"
  echo "  this run refreshes, so its forecast may be built on older demand."
fi

echo "--- ML run (ml_forward_forecasts + ml_forecast_history) ---"
# Attempted even when the legacy run failed. The two serve different screens,
# and a failure in the legacy backtest stage says nothing about whether the ML
# forecast can be produced. The warning above records the caveat when it applies.
.venv/bin/python scripts/ml_forward_forecast.py --snapshot live || ml_status=$?
if [ "$ml_status" -ne 0 ]; then
  echo "ERROR: ML forecast run failed with exit code $ml_status."
  echo "  The Action List and Forecast Validation will serve the previous run."
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

# Non-zero when either pipeline failed, so cron mails on the Monday it breaks.
# The ML failure already exited above; this catches a legacy failure that the ML
# run survived, which is a real problem for SKU Planning even though the two
# screens this project owns are fine.
if [ "$legacy_status" -ne 0 ]; then
  echo "=== $(date -u '+%Y-%m-%d %H:%M:%S UTC') : done, with the legacy run failed ==="
  exit "$legacy_status"
fi

echo "=== $(date -u '+%Y-%m-%d %H:%M:%S UTC') : done ==="
