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

# ML track: pinned anchor for the evaluation windows (src/ml/dataset.py).
# This is the last TRAINING week of the quarantined final-test split. All
# rolling-origin windows are built by stepping back from this date, so the
# weekly data refresh cannot silently shift the dev/final windows. New data
# past the final-test window is ignored until this date is advanced on
# purpose (which requires re-baselining recorded results). Set to None to
# fall back to anchoring on the latest week in the data.
ML_FINAL_TEST_CUTOFF = "2026-05-04"

# ML track: pinned DATA snapshot for the evaluation windows (src/ml/dataset.py).
# ML_FINAL_TEST_CUTOFF pins which WEEKS each window covers; this pins what DATA
# is in them. The weekly cron rewrites data/processed/ in place, which revises
# recent actuals and the SKU profile snapshot, so a model evaluated last week
# and a model evaluated today are not measured on the same numbers even with
# identical windows (the v3 entry in the design doc records exactly this drift).
#
# When set to a folder name under data/snapshots/, the ML harness reads
# sales_clean.parquet and sku_profiles.csv from there instead of from
# data/processed/. Set to None to follow the live refreshed data.
#
# This affects the ML development track ONLY. The production pipeline
# (run_forward_forecast.py, backtest.py, the FastAPI app) continues to read
# data/processed/ and is unaffected by this setting. Advancing the snapshot is
# a deliberate act that requires re-baselining recorded results; create a new
# one with scripts/ml_snapshot_data.py.
#
# Advanced 2026-08-10, from "2026-07-20". Both snapshots are on disk and the
# older one is untouched, so every figure recorded before this date remains
# reproducible by setting this back. Reasons and consequences: design doc
# Section 4.31.
ML_DATA_SNAPSHOT = "2026-08-03-v2"
DATA_SNAPSHOTS = ROOT / "data" / "snapshots"

# Conformal prediction interval levels.
# level=N in statsforecast means the CENTRAL N% interval: lower = P((100-N)/2), upper = P((100+N)/2).
# So level=70 → [P15, P85]; level=90 → [P5, P95]. The upper bound is NOT the Nth percentile.
CONFORMAL_LEVELS = [40, 60, 70, 80, 90]

# Segmentation thresholds
ZERO_PCT_INTERMITTENT = 0.30    # fraction of zero weeks → intermittent bucket
CV_THRESHOLD = 1.5              # coefficient of variation cutoff for lumpy demand
# Below this mean → intermittent, as a filter INDEPENDENT of zero_pct. The
# comment here used to read "SKUs above this mean go to smooth even if high
# zero_pct", which describes a different rule: mean as an escape from the
# sparsity test rather than a second bar. src/profile.py:classify() tests
# zero_pct first and returns intermittent, so the mean never rescues anything.
# Corrected 2026-08-11. src/profile.py:RECENT_MEAN_UPGRADE is matched to this
# value so the two cannot disagree about the same judgement.
MEAN_INTERMITTENT_CUTOFF = 3.0

# History-length boundaries (active training weeks)
SHORT_HISTORY_WEEKS   = 50   # < 50 → short
MEDIUM_HISTORY_WEEKS  = 104  # 50–104 → medium; ≥ 104 → full (2+ seasonal cycles)
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

# ML track's own holiday window (src/ml/seasonal.py).
#
# The settings above are shared by the statistical prototype (backtest.py,
# run_forward_forecast.py). The prototype is the accuracy bar the ML model must
# clear (design Section 1.6), so changing a shared factor to improve the ML
# model would move the target at the same time and make the comparison
# meaningless. These ML_ settings let the two diverge deliberately.
#
# They default to the prototype's values, so the ML track's results are
# unchanged until one is edited. src/ml/seasonal.py:matches_prototype() asserts
# that equivalence over a multi-year daily range.
#
# V1 is unaffected either way: src/v1.py and scripts/compare_v1.py each carry
# their own SEASONAL dict and never import from deseasonalize.
ML_USE_HOLIDAY_FLAG   = USE_HOLIDAY_FLAG
ML_HOLIDAY_START      = HOLIDAY_START
ML_HOLIDAY_END        = (12, 15)   # v9: promotions run late Nov to mid Dec; see Section 6
ML_HOLIDAY_MULTIPLIER = HOLIDAY_MULTIPLIER
# v15, pre-registered in design doc Section 6. False keeps the factor read off
# the week LABEL, which is what every recorded figure was measured under. True
# averages the factor over the seven days the week actually covers, which fixes
# a week that is six-sevenths July taking August's multiplier, and removes the
# 4-of-7 majority cliff in the holiday window. Default False: turning this on
# re-baselines the Version Log, so it is a deliberate act, not a default.
ML_SEASONAL_BLEND     = "off"      # "off" (v11) | "holiday" (v16) | "full" (v15)

# Metric thresholds (used in select.py)
WAPE_ACCEPTABLE = 0.25          # flag SKUs above this in reports

# Intermittent inventory policy
LEAD_TIME_WEEKS = 2             # weeks from order placement to receipt
SERVICE_LEVEL = 0.95            # target in-stock probability during lead time
