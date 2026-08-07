#!/usr/bin/env bash
# Remove the unmanaged process holding port 8000 so the systemd unit can bind.
#
#     bash scripts/_fix_port_8000_squatter.sh
#
# Diagnosis (2026-08-07). The unit is crash-looping with [Errno 98] address
# already in use, every 8 seconds, because pid 3152116 holds 127.0.0.1:8000.
# That process was started Fri Aug 7 00:01:11 UTC as
#   .venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
# with no --workers 1 and a relative venv path, so it did not come from systemd,
# whose ExecStart uses an absolute path and passes --workers 1. Its PPID is 1,
# so whatever started it has exited.
#
# That is the shape of the Next.js app's on-demand start, which fires when
# AI_SERVICE_URL is localhost and port 8000 is not answering. DEPLOYMENT.md says
# to leave FORECAST_SERVER_DIR unset in production for exactly this reason: a
# second supervisor racing systemd for the port. Step 4 below checks for it,
# because killing the process without removing the cause invites a repeat.
#
# Downtime is bounded. systemd is already retrying every 8 seconds, so the gap
# between killing the squatter and the unit binding should be under that. The
# Planning pages will 503 briefly.

set -uo pipefail
HOST="coverland@144.24.40.252"
SQUATTER=3152116

echo "=== 1. was the squatter serving stale code? ==="
echo "    If the file timestamps are OLDER than the process start, it loaded"
echo "    the deployed code and nothing was stale. If NEWER, it was serving"
echo "    pre-deploy code and the guard was not live."
ssh "$HOST" "
  echo 'process started:'; ps -p $SQUATTER -o lstart= 2>/dev/null || echo '  gone'
  echo 'deployed file mtimes:'
  stat -c '  %y  %n' /opt/coverland-forecast-api/src/clean.py \
                      /opt/coverland-forecast-api/src/weeks.py \
                      /opt/coverland-forecast-api/src/ml/seasonal.py
"

echo
read -r -p "Kill pid $SQUATTER and let systemd take the port? [y/N] " reply
[ "$reply" = "y" ] || { echo "aborted, nothing changed"; exit 0; }

echo
echo "=== 2. terminating the squatter ==="
ssh "$HOST" "kill $SQUATTER 2>/dev/null; sleep 3; ps -p $SQUATTER >/dev/null 2>&1 && { echo 'still alive, sending SIGKILL'; kill -9 $SQUATTER; sleep 2; } ; echo done"

echo
echo "=== 3. waiting up to 20s for systemd to bind ==="
ssh "$HOST" '
  for i in $(seq 1 10); do
    if ss -lnt | grep -q "0.0.0.0:8000"; then echo "bound after ${i}x2s"; break; fi
    sleep 2
  done
  echo; systemctl is-active coverland-forecast-api
  ss -lntp | grep 8000 || echo "NOTHING LISTENING - see systemctl status"
  echo; curl -s --max-time 5 http://127.0.0.1:8000/health | head -c 120; echo
'

echo
echo "=== 4. the cause: is FORECAST_SERVER_DIR set for the Next.js app? ==="
echo "    Set, it will start a second forecast server the next time port 8000"
echo "    is briefly unanswered, and this recurs. DEPLOYMENT.md: leave it unset"
echo "    in production, because systemd already supervises the process."
ssh "$HOST" '
  for f in /opt/coverland-commerce/.env /opt/coverland-commerce/.env.local \
           /opt/demand-pilot/.env /opt/demand-pilot/.env.local; do
    [ -f "$f" ] && { echo "--- $f"; grep -n "FORECAST_SERVER_DIR\|AI_SERVICE_URL" "$f" || echo "  neither set"; }
  done
  echo "--- any .env under /opt mentioning FORECAST_SERVER_DIR:"
  sudo grep -rl "FORECAST_SERVER_DIR" /opt --include=".env*" 2>/dev/null || echo "  none found (or no sudo without a tty)"
'

cat <<'EOF'

If step 3 shows 0.0.0.0:8000 and "active", the unit owns the port and the bind
change has taken effect. Only then are the firewall layers worth touching.

If step 4 found FORECAST_SERVER_DIR set, remove it from that env file and
restart the Next.js app, or this happens again at the next deploy restart.
EOF
