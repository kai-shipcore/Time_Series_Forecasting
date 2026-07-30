#!/usr/bin/env bash
#
# Check that the deployed forecast API can actually serve, not merely that it
# started.
#
# Run after the first deploy and the first data push, and any time a planning
# page looks wrong. Every check runs on the server over SSH against
# 127.0.0.1:8000, because that is the address the Next.js process beside it
# uses, so a pass here means the app's own requests will succeed.
#
#   scripts/verify_deployment.sh
#
# Reads the same variables as scripts/push_data_to_server.sh:
#
#   FORECAST_DEPLOY_HOST, FORECAST_DEPLOY_USER, FORECAST_DEPLOY_PATH
#   FORECAST_DEPLOY_PORT  (default 22)
#   FORECAST_DEPLOY_KEY   (optional)
#
# Exits non-zero if any check fails, and prints what to fix rather than only
# what broke.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Sets TARGET, SSH_OPTS and SSH.
# shellcheck source=scripts/_deploy_env.sh
. "${REPO_ROOT}/scripts/_deploy_env.sh" || exit 1

FAILED=0
pass() { printf '  PASS  %s\n' "$1"; }
fail() { printf '  FAIL  %s\n' "$1"; [ $# -gt 1 ] && printf '        %s\n' "$2"; FAILED=1; }

# The token lives in the server's .env, never here. Sourcing it remotely keeps
# the value on the box rather than passing it through this machine's process
# list, where `ps` would expose it.
remote_get() {
  "${SSH[@]}" "set -a; . '${FORECAST_DEPLOY_PATH}/.env' 2>/dev/null; set +a;
    curl --fail --silent --max-time 20 \
      -H \"x-forecast-token: \${FORECAST_API_TOKEN:-}\" \
      'http://127.0.0.1:8000$1'" 2>/dev/null
}

jq_py() { python3 -c "import json,sys; d=json.load(sys.stdin); print($1)" 2>/dev/null; }

echo "Verifying ${FORECAST_DEPLOY_USER}@${FORECAST_DEPLOY_HOST}:${FORECAST_DEPLOY_PATH}"
echo

# 1. Service up, data present, and serving from the directory we deployed to.
echo "1. Service and data"
health="$("${SSH[@]}" "curl --fail --silent --max-time 10 http://127.0.0.1:8000/health" 2>/dev/null)"
if [ -z "$health" ]; then
  fail "the API is not answering /health" \
       "check: systemctl status coverland-forecast-api"
else
  pass "the API answers /health"

  ready="$(printf '%s' "$health" | jq_py 'd.get("ready")')"
  if [ "$ready" = "True" ]; then
    pass "it has the data it reads"
  else
    missing="$(printf '%s' "$health" | jq_py '", ".join(d.get("missing_required") or [])')"
    fail "it is running with no data. Missing: ${missing:-unknown}" \
         "fix: scripts/push_data_to_server.sh"
  fi

  root="$(printf '%s' "$health" | jq_py 'd.get("repo_root") or ""')"
  if [ "$root" = "$FORECAST_DEPLOY_PATH" ]; then
    pass "it is serving from ${FORECAST_DEPLOY_PATH}"
  else
    fail "it is serving from ${root:-unknown}, not ${FORECAST_DEPLOY_PATH}" \
         "the running service is a different checkout; pushing data will not reach it"
  fi
fi

# 2. The endpoint the Action List calls. A 401 here is the token mismatch that
#    is otherwise invisible: /health is exempt from the check, so the status
#    indicator would show the service up while every page failed.
echo
echo "2. Action List endpoint"
al="$(remote_get '/planning/action-list?lead_time_weeks=8&review_period_weeks=1&service_z=1.0&stockout_horizon_days=30')"
if [ -z "$al" ]; then
  fail "it did not answer, or rejected the token" \
       "FORECAST_API_TOKEN must be identical in the API's .env and the Next.js .env"
else
  rows="$(printf '%s' "$al" | jq_py 'len(d.get("rows") or [])')"
  if [ "${rows:-0}" -gt 0 ]; then
    pass "returns ${rows} rows"
  else
    fail "returned no rows" "the forecast file may be present but empty"
  fi

  trained="$(printf '%s' "$al" | jq_py 'd.get("meta",{}).get("trained_through") or "unknown"')"
  echo "        trained through ${trained}"

  sample="$(printf '%s' "$al" | jq_py 'd.get("meta",{}).get("inventory_is_sample")')"
  if [ "$sample" = "False" ]; then
    pass "inventory is live, not sample data"
  else
    fail "inventory is SAMPLE data" \
         "DB_* or COMMERCE_DB_* in ${FORECAST_DEPLOY_PATH}/.env are wrong or incomplete; both prefixes are required"
  fi
fi

# 3. The validation page's figures, which depend on a file the data push sends
#    separately from the forecast itself.
echo
echo "3. Forecast Validation endpoint"
val="$(remote_get '/planning/validation')"
if [ -z "$val" ]; then
  fail "it did not answer"
else
  cur="$(printf '%s' "$val" | jq_py 'd.get("comparison",{}).get("headline",{}).get("current")')"
  base="$(printf '%s' "$val" | jq_py 'd.get("comparison",{}).get("headline",{}).get("baseline")')"
  if [ "$cur" = "None" ] || [ -z "$cur" ]; then
    fail "no comparison figures" \
         "outputs/reports/ml_accuracy.csv did not arrive; re-run scripts/push_data_to_server.sh"
  else
    pass "comparison present: model ${cur} against baseline ${base}"
  fi
fi

echo
if [ "$FAILED" -eq 0 ]; then
  echo "All checks passed."
  echo
  echo "One check cannot be scripted, and it is the one that matters most:"
  echo "open /planning/action-list and /planning/forecast-validation from a"
  echo "colleague's machine, with nothing installed and no local server running."
  echo "That is the only test of whether this actually solved the problem."
  exit 0
fi

echo "Some checks failed. See the fix noted under each."
exit 1
