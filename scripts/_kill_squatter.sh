#!/usr/bin/env bash
# Free port 8000 so the systemd unit can bind. Run whenever the diagnostics
# workflow reports 127.0.0.1:8000.
#
#     bash scripts/_kill_squatter.sh
#
# This is a workaround, not a fix. The unmanaged uvicorn returns after every
# deploy, because a deploy restarts the unit and something wins the race for the
# port during the gap. It has now happened twice: pid 3152116 at 00:01:11 on
# 2026-08-07, and pid 3269753 after the diagnostics workflow was pushed.
#
# The fix is to stop whatever starts it. See BACKLOG 21 and the pm2 step in
# .github/workflows/server-diagnostics.yml.
#
# Finds the pid rather than taking one as an argument, because the pid changes
# every time and hardcoding it is how the wrong process gets killed.

set -uo pipefail
HOST="coverland@144.24.40.252"

echo "=== before ==="
ssh "$HOST" 'ss -lntp | grep 8000 || echo "  nothing on 8000"; echo "unit: $(systemctl is-active coverland-forecast-api)"'

echo
read -r -p "Kill whatever holds 127.0.0.1:8000 and let systemd take it? [y/N] " reply
[ "$reply" = "y" ] || { echo "aborted"; exit 0; }

ssh "$HOST" '
  # Only ever the loopback listener. If the unit is correctly bound to 0.0.0.0
  # this matches nothing and the script does no harm.
  pid=$(ss -lntpH "sport = :8000" 2>/dev/null | grep "127.0.0.1:8000" | grep -o "pid=[0-9]*" | head -1 | cut -d= -f2)
  if [ -z "$pid" ]; then
    echo "no loopback listener on 8000 - nothing to kill"
    exit 0
  fi
  echo "killing pid $pid"
  ps -p "$pid" -o pid,lstart,cmd --no-headers
  kill "$pid" 2>/dev/null
  sleep 3
  ps -p "$pid" >/dev/null 2>&1 && { echo "still alive, SIGKILL"; kill -9 "$pid"; sleep 2; }

  for i in $(seq 1 10); do
    ss -lnt | grep -q "0.0.0.0:8000" && { echo "systemd bound after ${i}x2s"; break; }
    sleep 2
  done
'

echo
echo "=== after ==="
ssh "$HOST" 'echo "unit: $(systemctl is-active coverland-forecast-api)"; ss -lntp | grep 8000 || echo "  NOTHING LISTENING"'

echo
echo "Expect: unit active, 0.0.0.0:8000."
echo "This will come back at the next deploy until BACKLOG 21 is closed."
