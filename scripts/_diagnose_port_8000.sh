#!/usr/bin/env bash
# Who actually owns port 8000, and is the systemd unit running at all?
#
# Read-only. Changes nothing.
#
#     bash scripts/_diagnose_port_8000.sh
#
# Why this exists. On 2026-08-07 the bind was changed from 127.0.0.1 to
# 0.0.0.0 and applied cleanly: `systemctl show ExecStart` reflects the new
# value and the service restarted, pid 3259916 -> 3260617. But `ss` reported
# port 8000 held by pid 3152116 both before AND after, unchanged and older than
# either. Two different processes.
#
# The CI workflow already names this scenario: a process holding port 8000 that
# systemd does not manage means the unit fails to bind and crash-loops under
# Restart=always, while `systemctl restart` exits 0 and the port keeps
# answering. Deploys read green while the code they shipped is not running, and
# /health does not settle it because a stale process started from the same
# directory reports the same repo_root.
#
# If that is what is happening, every deploy since that process started has had
# no effect on what is being served, including the partial-trailing-week guard.

set -uo pipefail
HOST="coverland@144.24.40.252"

echo "=== 1. is the unit actually running, or restarting in a loop? ==="
echo "    (look at Active:, and at how recent the timestamp is)"
ssh "$HOST" 'systemctl status coverland-forecast-api --no-pager -l | head -20'

echo
echo "=== 2. the journal. 'address already in use' here confirms it ==="
ssh "$HOST" 'journalctl -u coverland-forecast-api -n 40 --no-pager | tail -25'

echo
echo "=== 3. what IS pid 3152116, and when did it start? ==="
echo "    (a start time older than the last deploy means it is serving stale code)"
ssh "$HOST" 'ps -p 3152116 -o pid,ppid,user,lstart,cmd --no-headers 2>/dev/null || echo "pid 3152116 is gone"'

echo
echo "=== 4. every uvicorn/python process, so nothing is hidden ==="
ssh "$HOST" 'ps -eo pid,ppid,user,lstart,cmd | grep -i "uvicorn\|api.main" | grep -v grep'

echo
echo "=== 5. everything listening on 8000, with sudo so no owner is hidden ==="
ssh -t "$HOST" 'sudo ss -lntp | grep 8000'

echo
echo "=== 6. what the running code actually reports about itself ==="
ssh "$HOST" 'curl -s --max-time 5 http://127.0.0.1:8000/health'

cat <<'EOF'


How to read this:

  Unit "active (running)" AND ss shows its pid   -> fine, the bind change simply
                                                     needs another restart.
  Unit restarting/failed, journal says "address already in use", and an older
  process owns 8000                              -> a stale process is squatting.
                                                     Every deploy since it started
                                                     has been cosmetic.

Do NOT kill anything yet. If a stale process is serving, it is currently the
only thing serving, and killing it before systemd can bind takes the Planning
pages down. The order matters and depends on what the above shows.
EOF
