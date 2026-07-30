#!/usr/bin/env bash
#
# Report what is actually running on the deployment server.
#
#   scripts/server_topology.sh
#
# Answers the question that has to be settled before any cutover: which
# processes hold the forecast ports, which directory each is serving, and which
# revision each is running. Unit names cannot answer it, because two units can
# name the same code and only one can hold a port.
#
# Reads nothing and changes nothing. Safe to run against a live server.

set -uo pipefail

# `return 1` from a sourced file does not stop the caller, so the exit status is
# checked explicitly. Without this the script carried on to use TARGET and died
# on an unbound variable, burying the message that actually explained the
# problem under a shell error.
# shellcheck source=scripts/_deploy_env.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_deploy_env.sh" || exit 1

echo "Probing ${TARGET} (${FORECAST_DEPLOY_PATH})"
echo

"${SSH[@]}" 'bash -s' <<'REMOTE'
set -u

echo "=== init system ==="
if command -v systemctl >/dev/null 2>&1; then
  echo "  systemd"
else
  echo "  no systemctl on PATH. Not a systemd host, or a container without it."
  echo "  Services here are started some other way; check: ps aux | grep uvicorn"
fi

echo
echo "=== units matching coverland ==="
if command -v systemctl >/dev/null 2>&1; then
  systemctl list-unit-files --no-pager --no-legend 2>/dev/null \
    | grep -i coverland || echo "  none installed"
fi

echo
echo "=== what each unit runs ==="
if command -v systemctl >/dev/null 2>&1; then
  for u in $(systemctl list-unit-files --no-pager --no-legend 2>/dev/null \
             | awk '{print $1}' | grep -i coverland); do
    frag="$(systemctl show "$u" -p FragmentPath --value 2>/dev/null)"
    state="$(systemctl show "$u" -p ActiveState --value 2>/dev/null)"
    enabled="$(systemctl show "$u" -p UnitFileState --value 2>/dev/null)"
    echo "  --- ${u}  [${state}, ${enabled}]"
    [ -n "$frag" ] && [ -f "$frag" ] &&
      grep -E "^(ExecStart|WorkingDirectory|EnvironmentFile)" "$frag" | sed 's/^/      /'
  done
fi

echo
echo "=== listeners on 8000 and 8001 ==="
listeners="$( { sudo -n ss -lptnH 2>/dev/null || ss -lptnH 2>/dev/null; } | grep -E ':800[01] ' )"
if [ -z "$listeners" ]; then
  echo "  nothing listening on either port"
else
  printf '%s\n' "$listeners" | sed 's/^/  /'
fi

echo
echo "=== where those processes are running from ==="
pids="$(printf '%s\n' "$listeners" | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)"
if [ -z "$pids" ]; then
  echo "  (no pids visible; re-run with sudo access for process names)"
else
  for p in $pids; do
    cwd="$(readlink "/proc/$p/cwd" 2>/dev/null || echo '?')"
    cmd="$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)"
    echo "  pid ${p}"
    echo "      cwd: ${cwd}"
    echo "      cmd: ${cmd}"
  done
fi

echo
echo "=== what each port reports ==="
# The decisive test. A service built from the current revision reports
# repo_root; one that predates it does not. That distinguishes the two without
# trusting unit names, and names the directory each is really serving.
for port in 8000 8001; do
  body="$(curl -s --max-time 5 "http://127.0.0.1:${port}/health" 2>/dev/null)"
  if [ -z "$body" ]; then
    echo "  :${port}  no answer"
    continue
  fi
  root="$(printf '%s' "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("repo_root") or "")' 2>/dev/null)"
  ready="$(printf '%s' "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("ready"))' 2>/dev/null)"
  if [ -n "$root" ]; then
    echo "  :${port}  current revision, serving ${root}, ready=${ready}"
  else
    echo "  :${port}  answers, but reports no repo_root, so it predates this revision"
    echo "           ${body}"
  fi
done
REMOTE
