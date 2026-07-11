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

# Conformal prediction interval levels.
# level=N in statsforecast means the CENTRAL N% interval: lower = P((100-N)/2), upper = P((100+N)/2).
# So level=70 → [P15, P85]; level=90 → [P5, P95]. The upper bound is NOT the Nth percentile.
CONFORMAL_LEVELS = [40, 60, 70, 80, 90]

# Segmentation thresholds
ZERO_PCT_INTERMITTENT = 0.30    # fraction of zero weeks → intermittent bucket
CV_THRESHOLD = 1.5              # coefficient of variation cutoff for lumpy demand
MEAN_INTERMITTENT_CUTOFF = 3.0  # SKUs above this mean go to smooth even if high zero_pct

# History-length boundaries (active training weeks)
SHORT_HISTORY_WEEKS   = 50   # smooth short/full boundary (< 50 → short)
MIN_SIM_HISTORY_WEEKS = 13   # minimum active weeks to be included in segment simulation

# Conformal prediction interval windows
MAX_CONFORMAL_WINDOWS = 5    # cap on calibration windows per run
MIN_CONFORMAL_WINDOWS = 2    # below this, skip PIs (1-window conformal is degenerate)

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
# False since Jul 2026: leak-free Apr–Jun backtest (321 short SKUs) showed
# WindowAverage(12) beats V1 in every tier×volume cell (median WAPE 0.188 vs
# 0.265, pooled 0.218 vs 0.291) — V1's 30–90d velocity windows average over the
# pre-ramp dormant period and systematically underforecast young SKUs (-13% bias).
# Short SKUs are also now deseasonalized like medium/full: the seasonal round-trip
# is safe for level-only models (a wash in mild seasons, Q4 protection in peak).
ROUTE_SHORT_SMOOTH_TO_V1 = False

# Metric thresholds (used in select.py)
WAPE_ACCEPTABLE = 0.25          # flag SKUs above this in reports

# Intermittent inventory policy
LEAD_TIME_WEEKS = 2             # weeks from order placement to receipt
SERVICE_LEVEL = 0.95            # target in-stock probability during lead time
