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

.venv/bin/python scripts/run_forward_forecast.py
status=$?
if [ "$status" -ne 0 ]; then
  echo "Forecast run failed with exit code $status"
  exit "$status"
fi

echo "Forecast run complete. Checking readiness..."
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

echo "=== $(date -u '+%Y-%m-%d %H:%M:%S UTC') : done ==="
