"""Reference figures the experiment scripts print alongside their own results.

Why this module exists
----------------------
Seven scripts each carried their own copy of the same `PROTOTYPE` dictionary,
and `ml_12` carried two more, all measured on the 2026-07-20 snapshot and none
of them saying so. That is harmless while the snapshot never moves and actively
dangerous the moment it does: the script keeps printing 2026-07-20 numbers next
to results computed on newer data, in the same table, with nothing marking the
difference. Anyone reading the log compares across two vintages and calls it a
regression.

So the numbers live here once, the snapshot they were measured on travels with
them, and `warn_if_stale()` says so out loud when the two disagree. Re-baselining
means editing this file once and moving REFERENCE_SNAPSHOT forward with it.

These are reference points for orientation, never pass criteria. Nothing is
adopted or rejected by comparing against them.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ML_DATA_SNAPSHOT  # noqa: E402

# The snapshot every number below was measured on.
REFERENCE_SNAPSHOT = "2026-08-03-v2"

# The statistical prototype (scripts/ml_10), per window: (short, long) pooled WAPE.
#
# Re-measured 2026-08-11 on 2026-08-03-v2. That snapshot carries two profiling
# changes against 2026-08-03, and figures are NOT comparable across them:
#   - promoted SKUs get their detected smooth-history onset instead of a flat
#     13 weeks, which admitted 190 previously unscoreable SKUs
#   - the promotion bar was matched to the classification bar at 3.0, which
#     removed 127 SKUs below it
# Design doc Section 4.32.
#
# Previous values, for the record:
#   on 2026-08-03  Mar-May (0.2028, 0.1437)  Dec-Feb (0.2904, 0.2690)  Oct-Dec (0.4137, 0.0918)
#   on 2026-07-20  Mar-May (0.2014, 0.1411)  Dec-Feb (0.2863, 0.2737)  Oct-Dec (0.4251, 0.0911)
PROTOTYPE = {
    "Mar-May": (0.2053, 0.1435),
    "Dec-Feb": (0.2912, 0.2685),
    "Oct-Dec": (0.3972, 0.0918),
}

# scripts/ml_12 regression check, per window: (short, long) pooled WAPE.
#
# Both dictionaries were re-measured on 2026-08-03 on 2026-08-10. Read the
# design doc Section 4.31 before comparing them to what was here before: the
# previous EXPECT_BASE was not a 2026-07-20 measurement at all. Its long figures
# reproduce only under ML_HOLIDAY_END = (12, 31), the value superseded by v9, so
# they had been stale since v9 was adopted and nothing detected it. Previous
# values, for the record:
#   EXPECT_BASE  Mar-May (0.2097, 0.1321)  Dec-Feb (0.1788, 0.2764)  Oct-Dec (0.4861, 0.1209)
#   EXPECT_V3    Mar-May (0.1863, 0.1345)  Dec-Feb (0.1943, 0.3145)  Oct-Dec (0.1826, 0.1011)
# The old EXPECT_V3 reproduces under NEITHER holiday window; at least three
# things changed beneath it and no single one accounts for the gap. v3 is
# superseded and gates no decision, so it is re-measured rather than explained.
# Re-measured 2026-08-11 on 2026-08-03-v2. EXPECT_BASE cross-checks exactly
# against the baseline column of ml_22's own run, which is the confirmation that
# these were read out of the right table: v-base is the RAW series for short
# SKUs and the DESEASONALIZED one for long (Section 4.17), and ml_03 prints
# both columns, so taking the wrong one is an easy and silent mistake.
#
# Previous values, on 2026-08-03:
#   EXPECT_BASE  Mar-May (0.2114, 0.1313)  Dec-Feb (0.1893, 0.2175)  Oct-Dec (0.4755, 0.1213)
#   EXPECT_V3    Mar-May (0.1859, 0.1431)  Dec-Feb (0.2030, 0.2365)  Oct-Dec (0.2376, 0.0995)
EXPECT_BASE = {
    "Mar-May": (0.2141, 0.1311),
    "Dec-Feb": (0.1923, 0.2171),
    "Oct-Dec": (0.4605, 0.1215),
}
EXPECT_V3 = {
    "Mar-May": (0.1926, 0.1413),
    "Dec-Feb": (0.1994, 0.2321),
    "Oct-Dec": (0.2473, 0.0937),
}


def warn_if_stale() -> bool:
    """Print a banner when the active snapshot is not the reference one.

    Returns True when they match. Callers that are regression checks should use
    the return value to soften their verdict: a moved number means something
    different when the data underneath it also moved, and reporting "the factor
    sources differ" in that situation is a wrong diagnosis, not a cautious one.
    """
    if ML_DATA_SNAPSHOT == REFERENCE_SNAPSHOT:
        return True
    print(
        f"\n  !! REFERENCE FIGURES ARE STALE\n"
        f"     printed below: measured on snapshot {REFERENCE_SNAPSHOT}\n"
        f"     running on:    snapshot {ML_DATA_SNAPSHOT}\n"
        f"     They are NOT comparable to this run's numbers. Re-measure them\n"
        f"     (scripts/ml_10 for PROTOTYPE, ml_03 and ml_08 for ml_12's) and\n"
        f"     update src/ml/reference.py before reading anything into a gap.\n"
    )
    return False
