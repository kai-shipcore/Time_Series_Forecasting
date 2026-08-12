#!/usr/bin/env bash
# Prove an aborted pipeline run cannot corrupt data/processed (BACKLOG item 15).
#
#     bash scripts/_test_staged_pipeline.sh
#
# Needs DB_* in .env: it runs the real ingest. A few minutes.
#
# How it kills, and why
# ---------------------
# Not on a timer. The first version slept 25 seconds and then sent SIGINT, which
# failed for two reasons worth keeping written down. Python defers signals while
# blocked in a C call, and at 25 seconds the run was inside a psycopg2 read, so
# the signal sat queued; and a fixed sleep against a step of unknown duration is
# a guess about where the boundary is. The run completed, the files legitimately
# changed, and the test reported a failure that was its own.
#
# This waits for the staging directory to contain at least one artifact, which is
# the exact moment a crash would have been destructive under the old code, then
# sends SIGKILL. SIGKILL cannot be deferred, caught or ignored, so it also
# simulates the harsher and likelier failures: a crash, a dropped SSH session, a
# power loss. A polite Ctrl-C is the easy case.
#
# What passing means
# ------------------
# data/processed is byte-identical to what it was before the run started. That
# is the whole guarantee. An orphaned data/.staging_* directory is EXPECTED
# after SIGKILL, because no cleanup handler runs when a process is killed
# outright; it is inert, costs disk only, and the next run makes its own. The
# test removes it so repeated runs stay clean.
#
# Safety
# ------
# data/processed is backed up first and restored on every exit path, including
# if this script is itself interrupted. data/snapshots is not touched by this
# pipeline. The database is only READ: the kill lands in step 2 or 3, and the
# only writes to shipcore.ml_* happen in step 4.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
[ -x "$PY" ] || { echo "no venv at $PY"; exit 1; }

PROC=data/processed
LOG=$(mktemp -t staged_run.XXXXXX)
BACKUP=$(mktemp -d)
trap 'echo; echo "restoring data/processed from backup"; rm -rf "$PROC"; cp -R "$BACKUP/processed" "$PROC"; rm -rf "$BACKUP"; rm -rf data/.staging_* 2>/dev/null' EXIT

echo "log:    $LOG"
echo "backup: $BACKUP"
cp -R "$PROC" "$BACKUP/processed"
before=$(find "$PROC" -type f -exec md5sum {} \; | sort)
echo "        $(echo "$before" | wc -l | tr -d ' ') files checksummed"
echo

echo "=== killing mid-write: data/processed must be byte-identical afterwards ==="
rm -rf data/.staging_* 2>/dev/null
"$PY" scripts/ml_prepare_data.py --force --no-sync >"$LOG" 2>&1 &
RUN=$!

# Wait for the run to be demonstrably mid-write rather than guessing a time.
staged=""
for i in $(seq 1 180); do
    if ! kill -0 "$RUN" 2>/dev/null; then
        echo "  the run exited on its own before writing anything staged."
        echo "  last 15 lines of the log:"; tail -15 "$LOG" | sed 's/^/    /'
        exit 1
    fi
    d=$(find data -maxdepth 1 -name ".staging_*" 2>/dev/null | head -1)
    if [ -n "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ]; then
        staged="$d"; break
    fi
    sleep 1
done
if [ -z "$staged" ]; then
    echo "  no staged artifact appeared within 180s. Log tail:"; tail -15 "$LOG" | sed 's/^/    /'
    exit 1
fi

echo "  staging populated after ${i}s: $(ls "$staged" | tr '\n' ' ')"
echo "  SIGKILL to pid $RUN  (cannot be deferred, caught or ignored)"
kill -9 "$RUN" 2>/dev/null
wait "$RUN" 2>/dev/null; echo "  reaped"

after=$(find "$PROC" -type f -exec md5sum {} \; | sort)
if [ "$before" = "$after" ]; then
    echo "  PASS: data/processed is byte-identical after being killed mid-write"
else
    echo "  FAIL: data/processed CHANGED"
    diff <(echo "$before") <(echo "$after") | sed 's/^/    /' | head -20
    echo "  log tail:"; tail -15 "$LOG" | sed 's/^/    /'
    exit 1
fi

orphan=$(find data -maxdepth 1 -name ".staging_*" 2>/dev/null)
if [ -n "$orphan" ]; then
    echo "  note: staging survived, which is expected after SIGKILL: $orphan"
    echo "        inert, and the next run creates its own. Removing it."
else
    echo "  note: no staging left (the kill landed between writes)"
fi

echo
echo "PASS. The guarantee holds: however violently the run dies, the previous"
echo "run's artifacts are intact and still being served."
echo "Restoring the backup so this test leaves nothing changed."
