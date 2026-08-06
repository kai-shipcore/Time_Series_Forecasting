"""Seasonal factors owned by the ML track.

Why this exists
---------------
`src/deseasonalize.py` is shared by the statistical prototype (`backtest.py`,
`run_forward_forecast.py`) and, until now, by the ML track. That coupling means
changing a factor to improve the ML model also moves the prototype's forecasts,
which is the thing the ML model is measured against (design Section 1.6). A
change intended as an improvement would silently move the target at the same
time, and the comparison would be meaningless.

This module gives the ML track its own copy of the factor logic, read from
`ML_*` settings in `config.py`. Those default to exactly the prototype's values,
so this file changes no result on the day it lands; that is verified rather than
assumed (see below). Diverging them later is then a deliberate, isolated act.

Not affected either way: V1. `src/v1.py` and `scripts/compare_v1.py` each define
their own `SEASONAL` dict and never import from `deseasonalize`, so the
production formula is already independent of both tracks.

Verification contract
---------------------
While `ML_SEASONAL_MATCHES_PROTOTYPE` is True, `ml_factors(ds)` must return
exactly what `src.deseasonalize._factors(ds)` returns for every date. That is
asserted at import time in debug runs and checked directly by
`scripts/ml_12_seasonal_split_check.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    ML_HOLIDAY_END,
    ML_HOLIDAY_MULTIPLIER,
    ML_HOLIDAY_START,
    ML_USE_HOLIDAY_FLAG,
)
from src.deseasonalize import SEASONAL_BASE, SEASONAL_HOLIDAY  # noqa: E402

# The monthly index is still taken from deseasonalize: it is a shared statement
# about the business, not an ML-track choice. Only the holiday window, which is
# where the ML track has evidence of a problem, is independently configurable.
ML_SEASONAL = SEASONAL_HOLIDAY if ML_USE_HOLIDAY_FLAG else SEASONAL_BASE


def ml_is_holiday(ds: pd.Series) -> pd.Series:
    """True for weeks inside the ML track's holiday window.

    Membership is decided on the DAYS a week covers, not on its label. Under
    W-MON a label of 2025-12-22 covers Dec 16-22, so testing the label puts the
    effective boundary a week late and, worse, moves it year to year: a fixed
    (11,20)-(12,15) label test covered Nov 18 to Dec 08 in 2024 but Nov 17 to
    Dec 14 in 2025. A window pinned to real promotional dates cannot drift like
    that. A week counts when a majority of its seven days fall inside the range.

    The span is `[ds - 6, ds]`. It was written as `[ds - 7, ds - 1]` until
    2026-08-06, which is the Monday-to-Sunday span and not the one this project
    uses: the week labelled L runs Tuesday (L-6) through Monday L, so the label
    is the week's LAST day and belongs inside the range. The docstring described
    the intent correctly and the arithmetic was one day out.

    That error changed no result, which is why it survived. Checked over every
    W-MON label from 2024-06-17 to 2026-07-20: both spans flag exactly 8 weeks
    and disagree on none, because membership needs 4 of 7 days inside a window
    running Nov 20 to Dec 15 and a one-day shift never crosses that threshold.
    It is fixed anyway because it is only harmless for this window and this date
    range; moving ML_HOLIDAY_END or adding a year of data would have been enough
    to make it bite, and it would have surfaced as an unexplained accuracy
    change in whichever experiment happened to be running. See BACKLOG 19.

    Matches src/ml/serving/v1.py, which already used `ds - 6` to `ds`, and
    src/db.py, whose fetch window is also `ds - 6`. The prototype
    (src/deseasonalize.py) still uses the label test. That is why
    matches_prototype() will report False once ML_HOLIDAY_END is moved: the two
    are then deliberately different, which is the point of the split.
    """
    m0, d0 = ML_HOLIDAY_START
    m1, d1 = ML_HOLIDAY_END
    start = ds - pd.Timedelta(days=6)
    end = ds
    yr = end.dt.year.where(end.dt.month >= m0, end.dt.year - 1)
    p0 = pd.to_datetime(dict(year=yr, month=m0, day=d0))
    p1 = pd.to_datetime(dict(year=yr, month=m1, day=d1))
    lo = start.where(start > p0, p0)
    hi = end.where(end < p1, p1)
    days_in = (hi - lo).dt.days + 1
    return days_in >= 4


def ml_factors(ds: pd.Series) -> pd.Series:
    """Seasonal factor per date for the ML track.

    Priority matches the prototype: holiday window first, then monthly index.
    """
    monthly = ds.dt.month.map(ML_SEASONAL)
    if ML_USE_HOLIDAY_FLAG:
        monthly = monthly.where(~ml_is_holiday(ds), ML_HOLIDAY_MULTIPLIER)
    return monthly


def matches_prototype() -> bool:
    """Do the ML factors currently agree with the prototype's, everywhere?

    Checked over a multi-year daily range so window-boundary differences cannot
    hide between sampled dates.
    """
    from src.deseasonalize import _factors

    # W-MON labels only: the day-based membership rule is defined on the seven
    # days a week label covers, so evaluating it on arbitrary calendar dates
    # compares two different questions and always reports a difference.
    ds = pd.Series(pd.date_range("2024-01-01", "2027-12-31", freq="W-MON"))
    return bool((ml_factors(ds) == _factors(ds)).all())


if __name__ == "__main__":
    from config import HOLIDAY_END, HOLIDAY_MULTIPLIER, HOLIDAY_START

    print(f"  prototype window : {HOLIDAY_START} to {HOLIDAY_END} x{HOLIDAY_MULTIPLIER}")
    print(f"  ML window        : {ML_HOLIDAY_START} to {ML_HOLIDAY_END} "
          f"x{ML_HOLIDAY_MULTIPLIER}")
    print(f"  identical        : {matches_prototype()}")
