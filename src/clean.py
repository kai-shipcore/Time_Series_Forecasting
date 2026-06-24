# Stage 2: Aggregate, reshape to long format (unique_id/ds/y), fill zeros, save to processed
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "sales_clean.parquet"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Aggregate to weekly totals per SKU (week starting Monday)
    df["order_date"] = pd.to_datetime(df["order_date"])
    weekly = (
        df.groupby(["link_master_sku", pd.Grouper(key="order_date", freq="W-MON")], as_index=False)["link_qty"]
        .sum()
        .rename(columns={"link_master_sku": "unique_id", "order_date": "ds", "link_qty": "y"})
    )

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
