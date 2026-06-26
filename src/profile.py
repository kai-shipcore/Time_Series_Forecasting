# Stage 3: Compute per-SKU stats and classify into forecast buckets
import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

ZERO_PCT_INTERMITTENT = 0.30   # SKUs with ≥30% zero weeks → intermittent (hard floor)
CV_THRESHOLD = 1.5
MEAN_INTERMITTENT_CUTOFF = 3.0  # used for ramp-up detection only (not classification)

# Recent-activity overrides (last 13 complete weeks)
RECENT_WEEKS             = 13
RECENT_ZERO_PCT_UPGRADE  = 0.15  # if recent zero_pct below this AND mean ≥ threshold → promote to smooth/short
RECENT_MEAN_UPGRADE      = 2.0   # recent weekly mean must be ≥ this to promote
RECENT_MEAN_DOWNGRADE    = 2.0   # if recent weekly mean below this → demote to intermittent

# Ramp-up detection
RAMP_UP_RATIO = 3.0          # second-half mean must be this many times the first-half mean
RAMP_UP_MIN_DEMAND = MEAN_INTERMITTENT_CUTOFF  # ramp-up only meaningful above the intermittent threshold

# History length thresholds (weeks)
# < SHORT  → too little history; V1/simple rolling rate only
# SHORT–MEDIUM → one seasonal cycle; seasonal models are candidates but shaky
# > MEDIUM → 2+ cycles; full seasonal model set
SHORT_HISTORY_WEEKS = 52
MEDIUM_HISTORY_WEEKS = 104


def _history_length(active_weeks: int) -> str:
    if active_weeks < SHORT_HISTORY_WEEKS:
        return "short"
    if active_weeks < MEDIUM_HISTORY_WEEKS:
        return "medium"
    return "full"


def _trend_slope(y: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    return slope


def _detect_ramp_up(grp: pd.DataFrame) -> tuple[bool, float, pd.Timestamp]:
    """
    Returns (is_ramp_up, second_half_mean, train_start).
    Compares first-half mean to second-half mean. A seasonal SKU will have
    similar means across both halves; a true ramp-up will have a much higher
    second-half mean. For flagged SKUs, train_start is set to the first week
    where the 4-week rolling mean exceeds 25% of the second-half mean.
    """
    mid = len(grp) // 2
    first_half_mean = grp["y"].iloc[:mid].mean()
    second_half_mean = grp["y"].iloc[mid:].mean()

    if first_half_mean == 0 and second_half_mean == 0:
        return False, 0.0, grp["ds"].iloc[0]

    ratio = second_half_mean / first_half_mean if first_half_mean > 0 else np.inf
    # When first half is all zeros, any sustained second-half demand is a ramp-up —
    # but require second_half_mean >= 2.0 to exclude truly sparse intermittent SKUs.
    zero_first_half = first_half_mean == 0
    is_ramp_up = ratio >= RAMP_UP_RATIO and (
        (zero_first_half and second_half_mean >= 2.0) or
        (not zero_first_half and second_half_mean >= RAMP_UP_MIN_DEMAND)
    )

    if not is_ramp_up:
        return False, float(second_half_mean), grp["ds"].iloc[0]

    # Find first week where rolling mean exceeds 25% of second-half mean
    rolling = grp["y"].rolling(4, min_periods=1).mean()
    threshold = 0.25 * second_half_mean
    active = rolling >= threshold
    first_active_idx = active.idxmax() if active.any() else grp.index[0]
    train_start = grp.loc[first_active_idx, "ds"]

    # Require at least 13 active weeks after train_start; otherwise trimming isn't useful
    weeks_remaining = (grp["ds"].iloc[-1] - train_start).days / 7
    if weeks_remaining < 13:
        return False, float(second_half_mean), grp["ds"].iloc[0]

    return True, float(second_half_mean), train_start


def profile(df: pd.DataFrame) -> pd.DataFrame:
    data_end = df["ds"].max()
    stats = []
    for uid, grp in df.groupby("unique_id"):
        grp = grp.sort_values("ds").reset_index(drop=True)
        is_ramp_up, second_half_mean, train_start = _detect_ramp_up(grp)
        active_weeks = int(round((data_end - train_start).days / 7))

        # Stats and classification use only the active window (from train_start),
        # so ramp-up zeros don't misclassify a now-healthy SKU as intermittent.
        active_grp = grp[grp["ds"] >= train_start]
        y = active_grp["y"].values
        mean = y.mean()
        std = y.std()
        cv = std / mean if mean > 0 else np.inf
        zero_pct = (y == 0).mean()
        trend = _trend_slope(y)

        stats.append({
            "unique_id": uid,
            "mean": mean,
            "std": std,
            "cv": cv,
            "zero_pct": zero_pct,
            "trend": trend,
            "ramp_up": is_ramp_up,
            "second_half_mean": second_half_mean,
            "train_start": train_start,
            "active_weeks": active_weeks,
        })

    profiles = pd.DataFrame(stats)

    def classify(row):
        if row["zero_pct"] >= ZERO_PCT_INTERMITTENT:
            return "intermittent"
        if row["mean"] < MEAN_INTERMITTENT_CUTOFF:
            return "intermittent"
        if row["cv"] >= CV_THRESHOLD:
            return "low_volume"
        return "smooth"

    profiles["bucket"] = profiles.apply(classify, axis=1)
    profiles["history_length"] = profiles["active_weeks"].apply(_history_length)

    # ── Recent-activity overrides ─────────────────────────────────────────────
    recent_dates = sorted(df["ds"].unique())[-RECENT_WEEKS:]
    recent_df    = df[df["ds"].isin(recent_dates)]
    recent_stats = (
        recent_df.groupby("unique_id")["y"]
        .agg(recent_mean="mean", recent_zero_pct=lambda y: (y == 0).mean())
        .reset_index()
    )
    profiles = profiles.merge(recent_stats, on="unique_id", how="left")

    # Promote: intermittent → smooth/short if recent 13 weeks look smooth
    upgrade = (
        (profiles["bucket"] == "intermittent") &
        (profiles["recent_zero_pct"] < RECENT_ZERO_PCT_UPGRADE) &
        (profiles["recent_mean"] >= RECENT_MEAN_UPGRADE)
    )
    n_up = upgrade.sum()
    profiles.loc[upgrade, "bucket"]         = "smooth"
    profiles.loc[upgrade, "history_length"] = "short"
    profiles.loc[upgrade, "train_start"]    = recent_dates[0]
    profiles.loc[upgrade, "active_weeks"]   = RECENT_WEEKS

    # Demote: smooth/low_volume → intermittent if recently dormant
    downgrade = (
        profiles["bucket"].isin({"smooth", "low_volume"}) &
        (profiles["recent_mean"] < RECENT_MEAN_DOWNGRADE)
    )
    n_down = downgrade.sum()
    profiles.loc[downgrade, "bucket"] = "intermittent"

    if n_up or n_down:
        print(f"  Recent-activity overrides: +{n_up} promoted to smooth/short, -{n_down} demoted to intermittent")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(PROCESSED_DIR / "sku_profiles.csv", index=False)

    return profiles


if __name__ == "__main__":
    df = pd.read_parquet(PROCESSED_DIR / "sales_clean.parquet")
    profiles = profile(df)

    print(f"Total SKUs: {len(profiles)}")
    print()
    print("Bucket breakdown:")
    print(profiles["bucket"].value_counts().to_string())
    print()
    print("History length breakdown:")
    print(profiles["history_length"].value_counts().to_string())
    print()
    print("Bucket × history length:")
    print(profiles.groupby(["bucket", "history_length"]).size().to_string())
    print()
    print(f"Ramp-up SKUs: {profiles['ramp_up'].sum()} "
          f"(active_weeks min={profiles.loc[profiles['ramp_up'], 'active_weeks'].min()}, "
          f"max={profiles.loc[profiles['ramp_up'], 'active_weeks'].max()})")
