#!/usr/bin/env bash
# Step 1 of opening port 8000: make the service listen on all interfaces.
#
# Temporary operator script, not part of the pipeline. The leading underscore
# matches the convention already used by scripts/_debug_*.py and
# scripts/_deploy_env.sh for things that are not pipeline steps. Delete it once
# the port work is finished and DEPLOYMENT.md is updated.
#
# Run from the repo root on your Mac:
#     bash scripts/_open_port_bind.sh
#
# Why a file rather than a pasted command: zsh on this machine is leaking
# bracketed-paste markers ([200~ ... ~) into the command line, which corrupts
# any multi-line paste. Running a file avoids the terminal entirely.
#
# Why `ssh -t`: these commands use sudo, and sudo cannot prompt for a password
# without a TTY. Without -t it fails with "a terminal is required to read the
# password", which reads like a permissions problem and is not one.

set -uo pipefail

HOST="coverland@144.24.40.252"
UNIT="/etc/systemd/system/coverland-forecast-api.service"

echo "=== before ==="
ssh "$HOST" "systemctl show coverland-forecast-api -p ExecStart --value; ss -lntp | grep 8000 || true"

echo
echo "=== is the API token set? (only prints SET or MISSING, never the value) ==="
ssh "$HOST" 'if grep -qE "^FORECAST_API_TOKEN=.+" /opt/coverland-forecast-api/.env; then echo "token: SET"; else echo "token: MISSING - anyone reaching this port gets POST /chat and POST /run-forecast unauthenticated"; fi'

echo
read -r -p "Change the bind from 127.0.0.1 to 0.0.0.0? [y/N] " reply
[ "$reply" = "y" ] || { echo "aborted, nothing changed"; exit 0; }

echo
echo "=== applying ==="
ssh -t "$HOST" "
  set -e
  sudo cp '$UNIT' ~/unit.backup.\$(date +%F)
  sudo sed -i 's|--host 127.0.0.1|--host 0.0.0.0|' '$UNIT'
  sudo systemctl daemon-reload
  sudo systemctl restart coverland-forecast-api
"

echo
echo "=== after ==="
ssh "$HOST" "systemctl show coverland-forecast-api -p ExecStart --value; ss -lntp | grep 8000 || true"

cat <<'EOF'

Expect the ExecStart line to end --host 0.0.0.0 and ss to show 0.0.0.0:8000.
Loopback is included in 0.0.0.0, so the Next.js site on the same box keeps
working through AI_SERVICE_URL=http://localhost:8000 unchanged.

Still to do after this, and the port is NOT reachable until both are done:
  - host firewall: NOT NEEDED, iptables policy is ACCEPT with no REJECT (checked 2026-08-07)
  - Oracle Cloud VCN security list, ingress TCP 8000, in the web console

Rollback if anything looks wrong:
  ssh -t coverland@144.24.40.252 'sudo cp ~/unit.backup.* /etc/systemd/system/coverland-forecast-api.service && sudo systemctl daemon-reload && sudo systemctl restart coverland-forecast-api'
EOF
