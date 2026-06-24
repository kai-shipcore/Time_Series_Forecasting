# Stage 6: Reference model sets per bucket and history length (pure definition, no data).
# Same interface as models.py — backtest.py merges candidates + baselines into one SF call.
# v1 formula will be added here once ported; slots are reserved below.
#
# history_length="full"/"medium" → SeasonalNaive(52) is the standard weekly baseline
# history_length="short"         → HistoricAverage; SeasonalNaive(52) is meaningless
#   when a SKU hasn't lived a full year — it would just repeat a near-zero value
from statsforecast.models import (
    Naive,
    SeasonalNaive,
    HistoricAverage,
)

SEASON_LENGTH = 52  # weeks in a year

# Seasonal baselines — "full" and "medium" history.
_BASELINE_SETS = {
    "smooth": [
        Naive(),
        SeasonalNaive(season_length=SEASON_LENGTH),
        # V1Formula(),  # TODO: add once ported
    ],
    "low_volume": [
        Naive(),
        SeasonalNaive(season_length=SEASON_LENGTH),
        # V1Formula(),
    ],
    "intermittent": [
        HistoricAverage(),
        # V1Formula(),
    ],
}

# Non-seasonal baselines — "short" history only.
# HistoricAverage gives the mean over active weeks: honest and hard to beat
# with only one partial seasonal cycle.
_SHORT_BASELINE_SETS = {
    "smooth": [
        Naive(),
        HistoricAverage(),
        # V1Formula(),
    ],
    "low_volume": [
        Naive(),
        HistoricAverage(),
        # V1Formula(),
    ],
}


def get_baselines(bucket: str, history_length: str = "full") -> list:
    if bucket not in _BASELINE_SETS:
        raise ValueError(f"Unknown bucket '{bucket}'. Expected one of: {list(_BASELINE_SETS)}")
    if history_length == "short":
        if bucket not in _SHORT_BASELINE_SETS:
            raise ValueError(f"No short baseline set for bucket '{bucket}'")
        return _SHORT_BASELINE_SETS[bucket]
    return _BASELINE_SETS[bucket]


if __name__ == "__main__":
    print("Seasonal baselines (full / medium history):")
    for bucket, models in _BASELINE_SETS.items():
        print(f"  {bucket:<15} {[type(m).__name__ for m in models]}")
    print()
    print("Non-seasonal baselines (short history):")
    for bucket, models in _SHORT_BASELINE_SETS.items():
        print(f"  {bucket:<15} {[type(m).__name__ for m in models]}")
