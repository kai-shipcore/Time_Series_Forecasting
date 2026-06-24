from pathlib import Path

# Paths
ROOT = Path(__file__).parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS_FORECASTS = ROOT / "outputs" / "forecasts"
OUTPUTS_REPORTS = ROOT / "outputs" / "reports"

# Forecast settings
FORECAST_HORIZON = 90       # days ahead to forecast
FREQUENCY = "W-MON"         # weekly, starting Monday

# Cross-validation / backtesting
TRIM_TRAILING_WEEKS = 0   # train through the last complete week
TEST_WEEKS = 10           # evaluation window ending at the trimmed tail
N_CV_SPLITS = 6

# Segmentation thresholds
ZERO_PCT_INTERMITTENT = 0.30    # fraction of zero weeks → intermittent bucket
CV_THRESHOLD = 1.5              # coefficient of variation cutoff for lumpy demand
MEAN_INTERMITTENT_CUTOFF = 3.0  # SKUs above this mean go to smooth even if high zero_pct

# Seasonal adjustment (see src/deseasonalize.py for full explanation)
# True  → deseasonalize training data before fitting, reseasonalize forecasts after.
#          Eliminates post-Q4 January over-prediction without affecting other months.
# False → original pipeline behaviour; models learn seasonality from data directly.
USE_SEASONAL_ADJUSTMENT = True

# Holiday window — weekly-resolution override for the pre-Christmas buying surge.
# When True, weeks inside the window use HOLIDAY_MULTIPLIER; all other weeks use
# their normal monthly factor (no zeroing needed — the window no longer spans Nov).
#
# Window rationale: Dec 1–14 captures the pre-Christmas gift/accessory buying peak;
# Dec 15–31 reverts to the normal December factor (1.30) as shipping cutoffs pass.
#
# Multiplier starting point: run scripts/optimize_holiday_multiplier.py to find the
# CV-optimal value. V1 prior (Nov×1.25 + Dec×1.30 over 61 days) concentrated into
# 14 days implies ~1.65; we start conservative at 1.35 until the optimiser runs.
USE_HOLIDAY_FLAG          = True
HOLIDAY_START             = (11, 20)   # (month, day)  Black Friday week
HOLIDAY_END               = (12, 31)   # (month, day)  end of December
HOLIDAY_MULTIPLIER        = 1.26       # CV-optimised: overall MAE 39.71, Jan bias +14.77 (window 11/20–12/31)

# Routing: short-history smooth SKUs route to V1 instead of a statistical model.
# Research showed V1's rolling-rate windows outperform any statistical model on
# ramp-up products with < 1 seasonal cycle of data (MAE 23.69 vs 27.27 for WA(12)).
ROUTE_SHORT_SMOOTH_TO_V1 = True

# Metric thresholds (used in select.py)
WAPE_ACCEPTABLE = 0.25          # flag SKUs above this in reports

# Intermittent inventory policy
LEAD_TIME_WEEKS = 2             # weeks from order placement to receipt
SERVICE_LEVEL = 0.95            # target in-stock probability during lead time
