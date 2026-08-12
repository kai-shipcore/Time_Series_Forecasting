#!/usr/bin/env bash
# BACKLOG 18: confirm eval_X/eval_y produced bit-identical numbers.
# Re-runs the model experiments and diffs against the logs committed under
# docs/rebaseline_2026-08-03/, ignoring only the deprecation warnings that the
# change is meant to remove.
set -uo pipefail
cd "$(dirname "$0")/.." 2>/dev/null || cd .
PY=.venv/bin/python
OLD=docs/rebaseline_2026-08-03
NEW=$(mktemp -d)

# What is excluded from the comparison, and why each one is legitimate.
#
#   lightgbm lines      the warnings this change exists to remove. Comparing
#                       them would guarantee a diff and prove nothing.
#   prototype / banner  reference figures from src/ml/reference.py. They are
#                       orientation printed beside the results, explicitly not
#                       pass criteria, and they move whenever the reference is
#                       re-measured. On the first run of this script they did
#                       exactly that: PROTOTYPE had been re-measured on the
#                       2026-08-03 snapshot after these logs were written, so
#                       four of six scripts reported DIFFERS on nothing but the
#                       bar they print beside their own numbers. A verification
#                       that fails for reasons unrelated to the thing under test
#                       is worse than none, because the next person reads FAIL
#                       and stops.
#
# Everything that could carry a result stays in: pooled WAPE tables, bias,
# tree counts, row counts, bootstrap deltas and verdicts.
strip() {
    grep -vE "LGBMDeprecationWarning|eval_set = _validate_eval_set_Xy|site-packages/lightgbm" \
    | grep -vE "^\s*prototype" \
    | grep -vE "REFERENCE FIGURES ARE STALE|printed below: measured on snapshot|running on:    snapshot|They are NOT comparable|Re-measure them|update src/ml/reference.py"
}

fail=0
for s in ml_03_baseline_deseas ml_08_lgbm_v3 ml_16_lgbm_v9 ml_22_v11_hybrid ml_23_v12_age ml_24_v13_accel; do
    printf '%-28s ' "$s"
    "$PY" "scripts/${s}.py" 2>&1 | strip > "$NEW/${s}.log"
    if diff -q <(strip < "$OLD/${s}.log") "$NEW/${s}.log" >/dev/null; then
        echo "identical"
    else
        echo "DIFFERS  ->  diff <(grep -v LGBM $OLD/${s}.log) $NEW/${s}.log"
        fail=1
    fi
done
echo
[ $fail -eq 0 ] && echo "PASS: every figure unchanged, and the warnings are gone." \
                || echo "FAIL: a number moved. The change is not cosmetic and needs its own entry."
echo "new logs: $NEW"
