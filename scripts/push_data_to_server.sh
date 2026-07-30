#!/usr/bin/env bash
#
# Push the data the forecast API reads to the deployed server.
#
# The deploy pipeline ships code and deliberately excludes data/ and outputs/,
# because those are gitignored and a --delete rsync from a checkout that lacks
# them would wipe the server's copies. So code arrives from GitHub Actions and
# data arrives from here. Nothing owns both, which is the point: neither can
# silently destroy the other's files.
#
# Run this after the weekly forecast, from the machine that produced the files:
#
#   scripts/push_data_to_server.sh
#
# Configuration comes from the environment, or from .env alongside this repo:
#
#   FORECAST_DEPLOY_HOST   e.g. app.example.com          (required)
#   FORECAST_DEPLOY_USER   e.g. coverland                (required)
#   FORECAST_DEPLOY_PATH   e.g. /opt/coverland-forecast-api  (required)
#   FORECAST_DEPLOY_PORT   defaults to 22
#   FORECAST_DEPLOY_KEY    ssh key path, defaults to ssh's own resolution
#
# Exits non-zero if the server cannot serve afterwards, so a cron failure mail
# arrives on the Monday it breaks rather than a colleague discovering it on the
# Thursday.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
[ -f .env ] && set -a && . ./.env && set +a

: "${FORECAST_DEPLOY_HOST:?set FORECAST_DEPLOY_HOST in .env or the environment}"
: "${FORECAST_DEPLOY_USER:?set FORECAST_DEPLOY_USER in .env or the environment}"
: "${FORECAST_DEPLOY_PATH:?set FORECAST_DEPLOY_PATH in .env or the environment}"
PORT="${FORECAST_DEPLOY_PORT:-22}"

SSH_OPTS=(-p "$PORT" -o StrictHostKeyChecking=accept-new)
[ -n "${FORECAST_DEPLOY_KEY:-}" ] && SSH_OPTS+=(-i "$FORECAST_DEPLOY_KEY")

TARGET="${FORECAST_DEPLOY_USER}@${FORECAST_DEPLOY_HOST}"
SSH=(ssh "${SSH_OPTS[@]}" "$TARGET")

# Exactly the files src/planning/data.py reads. Listed explicitly rather than
# syncing whole directories, because outputs/ is 19 MB of experiment plots and
# CV dumps the server has no use for. Kept in step with _DATA_FILES there; the
# readiness check at the end is what catches them drifting apart.
FILES=(
  data/processed/ml_forward_forecasts.parquet
  data/processed/sales_clean.parquet
  data/processed/sku_profiles.csv
  data/processed/v1_forward_forecasts.parquet
  data/processed/ml_forecast_history.parquet
  outputs/reports/ml_accuracy.csv
  outputs/reports/ml_accuracy_by_sku.csv
  outputs/reports/ml_backtest_weekly.csv
  dashboard/data/inventory_snapshot.csv
)

echo "Pushing forecast data to ${TARGET}:${FORECAST_DEPLOY_PATH}"

present=()
for f in "${FILES[@]}"; do
  if [ -f "$f" ]; then
    present+=("$f")
  else
    # Not fatal here. Several are optional, and the readiness check below is
    # the authority on whether what arrived is enough to serve.
    echo "  skipping (absent locally): $f"
  fi
done

if [ ${#present[@]} -eq 0 ]; then
  echo "Nothing to push: none of the expected files exist locally." >&2
  exit 1
fi

# Directories first: rsync will not create intermediate paths on its own.
"${SSH[@]}" "mkdir -p '${FORECAST_DEPLOY_PATH}'/{data/processed,outputs/reports,dashboard/data}"

# --relative preserves each file's path under the destination root, so
# data/processed/x.parquet lands at <path>/data/processed/x.parquet.
# No --delete: this script is not the authority on what else lives there.
rsync -az --relative --checksum \
  -e "ssh ${SSH_OPTS[*]}" \
  "${present[@]}" \
  "${TARGET}:${FORECAST_DEPLOY_PATH}/"

echo "Pushed ${#present[@]} file(s)."

# Ask the server whether it can actually serve now. Pushing files that leave it
# still unable to answer is the failure worth catching, and only the server can
# say: it knows which paths it resolved and what it found there.
echo "Checking readiness..."
body="$("${SSH[@]}" "curl --fail --silent --max-time 10 http://127.0.0.1:8000/health" || true)"

if [ -z "$body" ]; then
  echo "The API did not answer /health. Files are in place; the service may need a restart." >&2
  exit 1
fi

ready="$(printf '%s' "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("ready"))')"

if [ "$ready" = "True" ]; then
  echo "Server reports ready."
  exit 0
fi

missing="$(printf '%s' "$body" | python3 -c 'import json,sys; print(", ".join(json.load(sys.stdin).get("missing_required") or []))')"
root="$(printf '%s' "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("repo_root") or "?")')"
echo "Server still cannot serve. Missing: ${missing:-unknown}" >&2
echo "It is reading from: ${root}" >&2
echo "If that path is not ${FORECAST_DEPLOY_PATH}, the service is running from a different checkout." >&2
exit 1
