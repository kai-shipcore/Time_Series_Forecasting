# Stage 5: Candidate model sets per bucket and history length (pure definition, no data).
# backtest.py runs these through cross-validation; selector.py picks the winner.
#
# history_length="full"/"medium" → seasonal models (season_length=52)
# history_length="short"         → non-seasonal models (season_length=1)
#   "short" SKUs have < 52 active weeks — no full annual cycle — so fitting
#   seasonality is fitting noise. Level + trend is the most we can extract.
from config import USE_SEASONAL_ADJUSTMENT
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    AutoTheta,
    CrostonOptimized,
    IMAPA,
    SeasonalNaive,
    WindowAverage,
    HistoricAverage,
    Naive,
)

SEASON_LENGTH = 52  # weeks in a year

# When deseasonalizing, smooth models use season_length=1 — the seasonal
# pattern has already been removed from the training data, so fitting a
# seasonal model would find noise rather than signal.
_SMOOTH_SL = 1 if USE_SEASONAL_ADJUSTMENT else SEASON_LENGTH

# Full seasonal model sets — used for "full" and "medium" history length.
# Medium uses the same set and lets CV decide whether the seasonal signal is real.
_MODEL_SETS = {
    "smooth": [
        AutoARIMA(season_length=_SMOOTH_SL),
        AutoETS(season_length=_SMOOTH_SL, damped=True),
        AutoTheta(season_length=_SMOOTH_SL),
        WindowAverage(window_size=8),
        Naive(),          # strong baseline on deseasonalized flat series
        HistoricAverage(),
    ],
    "low_volume": [
        SeasonalNaive(season_length=SEASON_LENGTH),
        WindowAverage(window_size=8),
        HistoricAverage(),
    ],
    "intermittent": [
        CrostonOptimized(),
        IMAPA(),
    ],
}

# Non-seasonal model sets for "short" history SKUs (< 52 active weeks).
# Intermittent SKUs are always "full" history — no short set needed.
_SHORT_MODEL_SETS = {
    "smooth": [
        AutoETS(season_length=1),       # level + trend, no seasonality
        WindowAverage(window_size=12),  # 12-week window; tighter windows are too noisy for ramp-up SKUs
    ],
    "low_volume": [
        WindowAverage(window_size=8),  # lumpy short-history: keep it simple
    ],
}


def get_models(bucket: str, history_length: str = "full") -> list:
    if bucket not in _MODEL_SETS:
        raise ValueError(f"Unknown bucket '{bucket}'. Expected one of: {list(_MODEL_SETS)}")
    if history_length == "short":
        if bucket not in _SHORT_MODEL_SETS:
            raise ValueError(f"No short model set for bucket '{bucket}'")
        return _SHORT_MODEL_SETS[bucket]
    # "medium" and "full" both use the seasonal model set
    return _MODEL_SETS[bucket]


if __name__ == "__main__":
    print("Seasonal model sets (full / medium history):")
    for bucket, models in _MODEL_SETS.items():
        print(f"  {bucket:<15} {[type(m).__name__ for m in models]}")
    print()
    print("Non-seasonal model sets (short history):")
    for bucket, models in _SHORT_MODEL_SETS.items():
        print(f"  {bucket:<15} {[type(m).__name__ for m in models]}")
