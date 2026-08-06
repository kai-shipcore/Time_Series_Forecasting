# Stage 2: Aggregate, reshape to long format (unique_id/ds/y), fill zeros, save to processed
import pandas as pd
from pathlib import Path

from src.weeks import drop_incomplete_weeks

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "sales_clean.parquet"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Aggregate to weekly totals per SKU (week starting Monday)
    df["order_date"] = pd.to_datetime(df["order_date"])
    # A week runs TUESDAY to MONDAY and is labelled by the Monday it ends on,
    # so bucket 2026-08-10 holds Tue 4 August through Mon 10 August. This is
    # pandas' default for W-MON; both arguments are written out anyway, because
    # the convention is a decision now and should not look like a default.
    #
    # This was changed to closed="left" (Mon-Sun) on 2026-08-05 on the grounds
    # that the documentation said Mon-Sun, and changed back on 2026-08-06 on
    # evidence. Both reasons are recorded because the second one is not obvious:
    #
    #   1. Experiment 27 swept all seven possible week phases. v11 scores best
    #      on Tue-Mon in seven of eight cells, consistently across seasons,
    #      while the comparators' optima wander. Leave-one-window-out selection
    #      picks Tuesday on every fold for an honest out-of-sample gain of
    #      0.0132 pooled WAPE, with selection optimism measured at 0.0001. The
    #      mechanism is not understood. See design doc Section 4.30.
    #   2. The SQL side of this stack has always been Tue-Mon. api/main.py and
    #      src/db.py bucket with (order_date + ((8 - ISODOW) % 7) days), which
    #      maps Monday to itself and Tue-Sun forward to the next Monday. Under
    #      closed="left" the Python ingest and the API's own queries disagreed
    #      about which week a Monday's orders belong to, silently.
    #
    # Consequence that travels with this line, and must not be separated from
    # it: bucket L is not complete until the end of Monday L, so the cron runs
    # on TUESDAY (scripts/run_forecast_cron.sh) and src/weeks.py steps back an
    # extra week when asked on a Monday. Change any one of the three and the
    # pipeline quietly trains on a part-finished week or throws away a good one.
    weekly = (
        df.groupby(
            [
                "link_master_sku",
                pd.Grouper(key="order_date", freq="W-MON", closed="right", label="right"),
            ],
            as_index=False,
        )["link_qty"]
        .sum()
        .rename(columns={"link_master_sku": "unique_id", "order_date": "ds", "link_qty": "y"})
    )

    # Drop the trailing bucket when its week has not finished.
    #
    # W-MON labels a week by the Monday it ends on, so a run on any day except
    # Monday emits a final bucket holding only the days elapsed so far. Nothing
    # here used to remove it: the grid below was built through max(ds), so a
    # Wednesday run trained on a three-day "week" and stamped it as complete.
    # See src/weeks.py for what that cost.
    #
    # Before the grid, deliberately. Trimming afterwards would be undone by the
    # reindex, which would re-create the dropped week as a row of zeros -- worse
    # than the partial week, not better.
    before = weekly["ds"].max()
    weekly = drop_incomplete_weeks(weekly)
    if weekly.empty:
        raise ValueError(
            "No complete weeks in the ingested data. The most recent order is in "
            f"the week labelled {before.date()}, which has not finished yet."
        )
    if weekly["ds"].max() < before:
        print(f"  Dropped incomplete trailing week {before.date()} "
              f"(training through {weekly['ds'].max().date()})")

    # Build a full regular grid so every SKU has a row for every week
    all_weeks = pd.date_range(weekly["ds"].min(), weekly["ds"].max(), freq="W-MON")
    all_skus = weekly["unique_id"].unique()
    grid = pd.MultiIndex.from_product([all_skus, all_weeks], names=["unique_id", "ds"])
    weekly = (
        weekly.set_index(["unique_id", "ds"])
        .reindex(grid, fill_value=0)
        .reset_index()
    )

    weekly["ds"] = pd.to_datetime(weekly["ds"])
    weekly = weekly.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(OUTPUT_PATH, index=False)
    weekly.to_csv(PROCESSED_DIR / "sales_clean.csv", index=False)
    print(f"Saved {len(weekly):,} rows to {OUTPUT_PATH}")

    return weekly


if __name__ == "__main__":
    from src.ingest import ingest
    df = ingest()
    clean_df = clean(df)
    print(f"SKUs: {clean_df['unique_id'].nunique()}")
    print(f"Date range: {clean_df['ds'].min().date()} to {clean_df['ds'].max().date()}")
    print(f"Shape: {clean_df.shape}")
    print(clean_df.head(10))
