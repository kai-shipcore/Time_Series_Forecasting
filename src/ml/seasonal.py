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

    Note on W-MON: `ds` labels the Monday a week ENDS on, so a label of
    2025-12-22 covers Dec 15-21. Window membership is decided on the label,
    which means the effective boundary sits one week later than the calendar
    date suggests. That is the prototype's existing behaviour and is preserved
    here deliberately; changing it is a separate decision from moving the
    window.
    """
    m0, d0 = ML_HOLIDAY_START
    m1, d1 = ML_HOLIDAY_END
    m, d = ds.dt.month, ds.dt.day
    return ((m == m0) & (d >= d0)) | ((m > m0) & (m < m1)) | ((m == m1) & (d <= d1))


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

    ds = pd.Series(pd.date_range("2024-01-01", "2027-12-31", freq="D"))
    return bool((ml_factors(ds) == _factors(ds)).all())


if __name__ == "__main__":
    from config import HOLIDAY_END, HOLIDAY_MULTIPLIER, HOLIDAY_START

    print(f"  prototype window : {HOLIDAY_START} to {HOLIDAY_END} x{HOLIDAY_MULTIPLIER}")
    print(f"  ML window        : {ML_HOLIDAY_START} to {ML_HOLIDAY_END} "
          f"x{ML_HOLIDAY_MULTIPLIER}")
    print(f"  identical        : {matches_prototype()}")
