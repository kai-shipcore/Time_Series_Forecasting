# Stage 3: Compute per-SKU stats and classify into forecast buckets
import numpy as np
import pandas as pd
from pathlib import Path
from config import SHORT_HISTORY_WEEKS  # noqa: F401 — re-exported so importers don't break

# Follows config so a staged run (FORECAST_PROCESSED_DIR, BACKLOG item 15) writes
# where the rest of the pipeline is reading. Still a module-level name read at
# call time: scripts/ml_36 and promoted_sku_accuracy.py reassign it to a temp
# directory so an analysis cannot overwrite live data, and that must keep working.
from config import DATA_PROCESSED as _DEFAULT_PROCESSED  # noqa: E402

PROCESSED_DIR = _DEFAULT_PROCESSED

ZERO_PCT_INTERMITTENT = 0.30   # SKUs with ≥30% zero weeks → intermittent (hard floor)
CV_THRESHOLD = 1.5
# Used by classify() AND by ramp-up detection. The comment here previously said
# "not classification", and config.py's said SKUs above it "go to smooth even if
# high zero_pct". Both were wrong: classify() tests zero_pct first and returns
# intermittent, then tests mean as a second independent filter. Corrected rather
# than acted on, because the documented behaviour is a different rule and
# changing to it is a decision, not a typo fix.
MEAN_INTERMITTENT_CUTOFF = 3.0

# Recent-activity overrides (last 13 complete weeks)
RECENT_WEEKS             = 13
RECENT_ZERO_PCT_UPGRADE  = 0.20  # if recent zero_pct below this AND mean ≥ threshold → promote to smooth

# Matched to MEAN_INTERMITTENT_CUTOFF on 2026-08-11. It was 2.0, which is below
# the classification bar, so a SKU could fail one test and pass the other: a
# steady 2.5-unit SKU was called intermittent for being small, then promoted
# back on its recent window, then had its history truncated. Two bars for one
# judgement is what made that possible.
#
# A JUDGEMENT, not a measurement, and recorded as one. The development windows
# cannot locate this boundary: the population it governs numbered SEVEN SKUs at
# the Oct-2025 cutoff, because being regular is recent for these SKUs even when
# selling is not. The bands either side of 3.0 came back indistinguishable with
# point estimates favouring a trailing mean, which is consistent with a floor
# existing but does not say where.
#
# Its cost IS measured, so it can be revisited deliberately: 127 SKUs lose their
# forecast, smooth goes 467 to 340, and 3,978 of 58,842 smooth units stop being
# covered. That is 6.8% of forecast demand, from SKUs whose weekly means run
# from 2.0 to 2.92. They keep appearing in the Action List's Not-forecast
# section with a trailing actual-sales rate, so they are not invisible, only
# unforecast.
#
# The instrument that can settle this is shipcore.ml_forecast_history, which
# measures these SKUs as they are now rather than as they were when there were
# seven. See docs/HANDOVER.md finding 7.
RECENT_MEAN_UPGRADE      = 3.0

# Deliberately LOWER than the promotion bar, and this asymmetry is the point.
# Equal bars would make a SKU hovering at the threshold flip between smooth and
# intermittent week after week, and each flip removes or restores its forecast
# and resets train_start. The gap from 2.0 to 3.0 is a hysteresis band: a SKU
# must clear 3.0 to start being forecast and fall below 2.0 to stop.
RECENT_MEAN_DOWNGRADE    = 2.0

# Ramp-up detection
RAMP_UP_RATIO = 3.0          # second-half mean must be this many times the first-half mean
RAMP_UP_MIN_DEMAND = MEAN_INTERMITTENT_CUTOFF  # ramp-up only meaningful above the intermittent threshold

# History length thresholds (weeks)
# < SHORT  → too little history; fixed short default, no CV
# SHORT–MEDIUM → CV-capable (3+ ten-week windows on a 20-week training floor);
#                deseasonalized full model menu, seasonal signal still shaky
# > MEDIUM → 2+ cycles; full seasonal model set
# SHORT = 50 = 20-week training floor + 3 × 10-week CV windows: a SKU is
# "medium" exactly when it can support 3-window CV selection.
# SHORT_HISTORY_WEEKS is imported from config above.
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


def _smooth_onset(y: np.ndarray, dates: list) -> tuple:
    """How far back a promoted SKU's smooth behaviour actually runs.

    Returns (train_start, active_weeks).

    The promotion override used to assign three constants: `train_start` set to
    the start of the trailing RECENT_WEEKS window, `active_weeks` set to
    RECENT_WEEKS, and `history_length` set to "short". That is right for a SKU
    that genuinely just became forecastable and wrong for one that has been
    steady all along.

    It is wrong more often than not, because a SKU reaches the promotion path in
    two different ways. `classify()` sends a SKU to intermittent when it is
    sparse (zero_pct >= ZERO_PCT_INTERMITTENT) OR when it is merely small
    (mean < MEAN_INTERMITTENT_CUTOFF). A SKU selling a steady 2.5 units every
    week is not sporadic in any sense; it fails the second test only, then gets
    promoted on its recent window, then has its history cut to 13 weeks as
    though it had just appeared.

    Measured on the 2026-08-03 snapshot before this was written: 190 SKUs were
    promoted, 41% of the entire smooth set. Only 15 of them genuinely had 13
    weeks. The median had 34, the maximum had 111, which is the whole series,
    and 73 had at least 50 weeks yet were labelled "short" and so routed to the
    short model. 4,615 SKU-weeks of usable history were being discarded, because
    load_weekly trims each SKU's training data to train_start.

    The rule here is the promotion test applied to progressively longer trailing
    windows, stopping at the first length that fails. Stopping rather than
    scanning for the longest passing window anywhere is deliberate: a window that
    passes only after skipping a bad patch would silently include that patch.
    Conservative in the right direction, and it degenerates to RECENT_WEEKS for a
    SKU that really did just turn around.

    Side effect worth knowing, and it is a fix rather than a surprise:
    `train_start` stops being pinned to a window that slides forward every run
    and becomes a fixed historical date. That is what BACKLOG item 2 describes as
    eligibility being non-stationary, and it is why these SKUs could never appear
    in a backtest.
    """
    best = RECENT_WEEKS
    for k in range(RECENT_WEEKS, len(y) + 1):
        w = y[-k:]
        if (w == 0).mean() < RECENT_ZERO_PCT_UPGRADE and w.mean() >= RECENT_MEAN_UPGRADE:
            best = k
        else:
            break
    return dates[-best], best


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
    profiles.loc[upgrade, "bucket"] = "smooth"

    # Explicit marker, because the implicit one is gone. Downstream code used to
    # identify a promoted SKU by `active_weeks == RECENT_WEEKS`, which worked
    # only because promotion assigned that constant. Onset detection below gives
    # each promoted SKU its real length, so 175 of 190 no longer carry the
    # signature and every consumer of it would silently stop recognising them.
    # src/planning/calc.py is the one that matters: it hands promoted SKUs a
    # cohort error for safety-stock sizing, and losing them means under-sizing.
    #
    # A SKU cannot be both promoted and demoted in one run: promotion requires
    # recent_mean >= RECENT_MEAN_UPGRADE and demotion requires it below
    # RECENT_MEAN_DOWNGRADE, and the first is now the higher bar.
    profiles["promoted"] = upgrade

    # train_start, active_weeks and history_length are DETECTED per SKU rather
    # than assigned as constants. See _smooth_onset for what the constants cost.
    if n_up:
        piv = df.pivot_table(index="unique_id", columns="ds", values="y",
                             fill_value=0)
        piv = piv[sorted(piv.columns)]
        cols = list(piv.columns)
        starts, weeks = {}, {}
        for uid in profiles.loc[upgrade, "unique_id"]:
            start, k = _smooth_onset(piv.loc[uid].to_numpy(), cols)
            starts[uid], weeks[uid] = start, k
        ids = profiles.loc[upgrade, "unique_id"]
        profiles.loc[upgrade, "train_start"] = ids.map(starts).to_numpy()
        profiles.loc[upgrade, "active_weeks"] = ids.map(weeks).to_numpy()
        # Derived from the detected length by the same function every other SKU
        # goes through, rather than hardcoded "short". 73 SKUs on the 2026-08-03
        # snapshot have >= SHORT_HISTORY_WEEKS of smooth history and were being
        # labelled short, which routes them to the short model as well as
        # starving them of the data they have.
        profiles.loc[upgrade, "history_length"] = (
            ids.map(weeks).map(_history_length).to_numpy()
        )

    # Demote: smooth/low_volume → intermittent if recently dormant
    downgrade = (
        profiles["bucket"].isin({"smooth", "low_volume"}) &
        (profiles["recent_mean"] < RECENT_MEAN_DOWNGRADE)
    )
    n_down = downgrade.sum()
    profiles.loc[downgrade, "bucket"] = "intermittent"

    if n_up or n_down:
        # Says "smooth" rather than "smooth/short" because the history length is
        # now detected rather than assumed, and reports the spread so a run that
        # promotes everything to 13 weeks again is visible rather than silent.
        if n_up:
            got = profiles.loc[upgrade, "active_weeks"]
            spread = (f"{int(got.median())} weeks median, {int(got.min())}-"
                      f"{int(got.max())} range")
            lens = profiles.loc[upgrade, "history_length"].value_counts().to_dict()
        else:
            spread, lens = "", {}
        print(f"  Recent-activity overrides: +{n_up} promoted to smooth "
              f"({spread}; {lens}), -{n_down} demoted to intermittent")

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
