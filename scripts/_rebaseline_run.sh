#!/usr/bin/env bash
# Re-run every experiment whose numbers appear in the design doc, on whatever
# snapshot config.ML_DATA_SNAPSHOT currently points at, and keep the output.
#
#     bash scripts/_rebaseline_run.sh 2026-08-03
#
# The argument is only the log directory name. It does NOT select the snapshot;
# config.py does that, and this script prints which one is active so a log can
# never be misfiled under the wrong label.
#
# Deliberately does not stop on the first failure. A batch that dies at step
# three leaves you with two logs and no idea whether the rest would have worked,
# and these are slow enough that finding out one at a time costs an afternoon.
# Exit codes are collected and printed together at the end.
#
# NOT run here, on purpose:
#   ml_04, ml_17, ml_18, ml_19   parameter searches whose chosen values are
#                                already in config.py
#   ml_26, ml_27, ml_28          the week-phase sweep behind Section 4.30
# These are searches, not version comparisons. Their conclusions are adopted and
# live in the code; re-running them costs many times the rest of this batch put
# together, and a re-baseline does not need them to be self-consistent. If you
# want them re-measured, that is a separate decision and a separate afternoon.

set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
[ -x "$PY" ] || { echo "no venv at $PY"; exit 1; }

LABEL="${1:-$(date +%F)}"
LOGS="docs/rebaseline_${LABEL}"
mkdir -p "$LOGS"

ACTIVE=$("$PY" -c 'from config import ML_DATA_SNAPSHOT as s; print(s)') || exit 1
echo "active snapshot : $ACTIVE"
echo "log directory   : $LOGS"
if [ "$ACTIVE" != "$LABEL" ]; then
    echo
    echo "  WARNING: the log label and the active snapshot differ."
    echo "  That is legal but it is usually a mistake. Ctrl-C now if it is."
    sleep 5
fi
echo

# name:script[:args]
RUNS=(
  "ml_01_naive_baseline"
  "ml_02_v1_benchmark"
  "ml_03_baseline_deseas"
  "ml_05_lgbm_v0"
  "ml_06_lgbm_v1"
  "ml_07_lgbm_v2"
  "ml_08_lgbm_v3"
  "ml_09_lgbm_v4"
  "ml_10_prototype_benchmark"
  "ml_13_holiday_window"
  "ml_14_lgbm_v7"
  "ml_15_lgbm_v8"
  "ml_16_lgbm_v9"
  "ml_22_v11_hybrid"
  "ml_23_v12_age"
  "ml_24_v13_accel"
  "ml_25_v14_min_child"
  "ml_29_v15_seasonal_blend --mode full"
  "ml_29_v15_seasonal_blend --mode holiday"
)

declare -a FAILED=()
START_ALL=$(date +%s)

for entry in "${RUNS[@]}"; do
    set -- $entry
    script="$1"; shift
    tag="$script"
    [ $# -gt 0 ] && tag="${script}$(echo "_$*" | tr -d ' -')"
    log="$LOGS/${tag}.log"

    printf '%-46s ' "$tag"
    start=$(date +%s)
    "$PY" "scripts/${script}.py" "$@" > "$log" 2>&1
    rc=$?
    dur=$(( $(date +%s) - start ))

    if [ $rc -eq 0 ]; then
        printf 'ok    %4ds\n' "$dur"
    else
        printf 'EXIT %-3d %4ds  -> %s\n' "$rc" "$dur" "$log"
        FAILED+=("$tag (exit $rc)")
    fi
done

# Last, and separately: this one is a regression check rather than a
# measurement, and until src/ml/reference.py is updated with the numbers the
# batch above just produced, it can only return "inconclusive". Running it here
# is what makes that visible rather than assumed.
echo
printf '%-46s ' "ml_12_seasonal_split_check"
"$PY" scripts/ml_12_seasonal_split_check.py > "$LOGS/ml_12_seasonal_split_check.log" 2>&1
rc=$?
case $rc in
  0) echo "PASS (data unchanged)" ;;
  2) echo "inconclusive (expected: reference.py not yet updated)" ;;
  *) echo "FAIL (exit $rc) -> $LOGS/ml_12_seasonal_split_check.log"
     FAILED+=("ml_12 (exit $rc)") ;;
esac

echo
echo "total $(( ($(date +%s) - START_ALL) / 60 )) min"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "all runs exited 0"
else
    echo "${#FAILED[@]} run(s) did not exit 0:"
    printf '  %s\n' "${FAILED[@]}"
fi
echo
echo "Next: read the logs, update the Section 6 tables and src/ml/reference.py"
echo "      (PROTOTYPE from ml_10, EXPECT_BASE from ml_03, EXPECT_V3 from ml_08),"
echo "      move REFERENCE_SNAPSHOT to $ACTIVE, then re-run ml_12 for a real verdict."
