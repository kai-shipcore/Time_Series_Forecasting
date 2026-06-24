"""
Seasonal adjustment preprocessing for smooth SKUs.

──────────────────────────────────────────────────────────────────────────────
WHY THIS EXISTS
──────────────────────────────────────────────────────────────────────────────
Without this, AutoARIMA / AutoETS learn the seasonal pattern from ~2 years of
data (two Q4 cycles). That's too few cycles for a stable estimate, and it
causes systematic over-prediction in January: the model anchors on the Q4
peak and doesn't unwind fast enough.

The fix is the textbook deseasonalize → fit → reseasonalize approach:
  1. Divide every week's demand by its seasonal factor BEFORE fitting.
     The model now sees a "flat" series and only learns level and trend.
  2. Fit the model on the flat series.  (backtest.py handles this)
  3. Multiply each forecast week's prediction by that week's seasonal factor
     AFTER fitting.  Seasonality comes entirely from the known index — no
     double-counting, so October predictions are unaffected when you fix Jan.

──────────────────────────────────────────────────────────────────────────────
TWO-LEVEL SEASONAL INDEX
──────────────────────────────────────────────────────────────────────────────
Level 1 — monthly modifiers (SEASONAL_BASE):
  Standard months use a single multiplier per calendar month.

Level 2 — holiday window (USE_HOLIDAY_FLAG = True in config.py):
  Black Friday week through end of December is where the surge lives.
  A single HOLIDAY_MULTIPLIER is applied to every week inside the window
  (Nov 20 → Dec 31).  Nov/Dec monthly entries are zeroed (→1.0) so early
  November (Nov 1–19, pre-window) is treated as a baseline week — no
  double-counting.

  Multiplier rationale: run scripts/optimize_holiday_multiplier.py to find
  the CV-optimal value; current default (1.35) is a conservative starting
  point.  The concentrated 25-day window likely warrants a higher value than
  the old 42-day window did — the optimizer will confirm.

──────────────────────────────────────────────────────────────────────────────
HOW TO TOGGLE
──────────────────────────────────────────────────────────────────────────────
config.py:
  USE_SEASONAL_ADJUSTMENT = True / False   main on/off switch
  USE_HOLIDAY_FLAG        = True / False   holiday window on/off (requires above)
  HOLIDAY_START           = (11, 20)       (month, day) start of holiday window
  HOLIDAY_END             = (12, 31)       (month, day) end of holiday window
  HOLIDAY_MULTIPLIER      = 1.35           lift factor inside the window
"""
import pandas as pd
from config import (
    USE_HOLIDAY_FLAG,
    HOLIDAY_START,
    HOLIDAY_END,
    HOLIDAY_MULTIPLIER,
)

# ── Seasonal indices ───────────────────────────────────────────────────────────

# Original monthly modifiers — used when USE_HOLIDAY_FLAG is False.
SEASONAL_BASE: dict[int, float] = {
    1: 0.75,   # January   — post-holiday trough
    2: 0.80,   # February
    3: 0.90,   # March
    4: 0.95,   # April
    5: 1.00,   # May
    6: 1.00,   # June
    7: 1.00,   # July
    8: 1.00,   # August
    9: 1.00,   # September
    10: 1.10,  # October   — Q4 ramp
    11: 1.25,  # November
    12: 1.30,  # December  — peak
}

# Monthly modifiers with Nov/Dec zeroed — used alongside the holiday window.
# Nov 1-19 (pre-window) uses 1.0; window starts Nov 20 and runs through Dec 31.
SEASONAL_HOLIDAY: dict[int, float] = {
    **SEASONAL_BASE,
    11: 1.00,   # Nov 1-19 (pre-window) treated as baseline; window starts Nov 20
    12: 1.00,   # all of Dec inside holiday window; HOLIDAY_MULTIPLIER applied there
}

# Public alias — the "active" monthly index for external consumers (e.g. compare_v1.py)
SEASONAL = SEASONAL_HOLIDAY if USE_HOLIDAY_FLAG else SEASONAL_BASE

# Columns that are never factor-adjusted (metadata / actuals written by StatsForecast)
_META_COLS = {"unique_id", "ds", "cutoff", "y", "bucket", "history_length"}


# ── Factor helpers ─────────────────────────────────────────────────────────────

def _is_holiday(ds: pd.Series) -> pd.Series:
    """Boolean mask — True for weeks inside the holiday window."""
    h_m0, h_d0 = HOLIDAY_START
    h_m1, h_d1 = HOLIDAY_END
    m, d = ds.dt.month, ds.dt.day
    # Window does not cross year boundary (Nov–Dec), so a simple range check works.
    in_start_month = (m == h_m0) & (d >= h_d0)
    in_middle      = (m > h_m0) & (m < h_m1)
    in_end_month   = (m == h_m1) & (d <= h_d1)
    return in_start_month | in_middle | in_end_month


def _factors(ds: pd.Series) -> pd.Series:
    """
    Return the seasonal adjustment factor for every date in ds.

    Priority:
      1. Holiday window → HOLIDAY_MULTIPLIER  (if USE_HOLIDAY_FLAG)
      2. Monthly index  → SEASONAL_HOLIDAY or SEASONAL_BASE
    """
    monthly = ds.dt.month.map(SEASONAL_HOLIDAY if USE_HOLIDAY_FLAG else SEASONAL_BASE)
    if USE_HOLIDAY_FLAG:
        monthly = monthly.where(~_is_holiday(ds), HOLIDAY_MULTIPLIER)
    return monthly


# ── Public API ─────────────────────────────────────────────────────────────────

def deseasonalize(df: pd.DataFrame, date_col: str = "ds", y_col: str = "y") -> pd.DataFrame:
    """
    Divide each row's demand by the seasonal factor for that week.

    Call this on the training DataFrame before passing it to StatsForecast.
    The result is a "flat" series where the model learns only level and trend.
    """
    df = df.copy()
    df[y_col] = df[y_col] / _factors(df[date_col])
    return df


def reseasonalize(
    df: pd.DataFrame,
    date_col: str = "ds",
    forecast_cols: list[str] | None = None,
    y_col: str = "y",
) -> pd.DataFrame:
    """
    Multiply forecast columns (and the actuals column) by the seasonal factor
    for each row's date.

    Call this on the StatsForecast cross_validation output immediately after
    fitting.  Pass forecast_cols=None to auto-detect all non-metadata columns.

    Both the model forecasts and y (actual) are reseasonalized so downstream
    metric computation (selector.py) stays in the original demand scale.
    """
    df = df.copy()
    factor = _factors(df[date_col])

    if forecast_cols is None:
        forecast_cols = [c for c in df.columns if c not in _META_COLS]

    for col in [y_col] + list(forecast_cols):
        if col in df.columns:
            df[col] = df[col] * factor

    return df
