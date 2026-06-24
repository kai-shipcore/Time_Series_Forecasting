# Stage 4: Route SKUs into model buckets based on saved profiles.
# Input:  data/processed/sku_profiles.csv  (written by profile.py)
# Output: dict mapping bucket name → list of unique_id strings
# profile.py owns classification logic; this stage owns routing only.
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROFILES_PATH = PROCESSED_DIR / "sku_profiles.csv"

BUCKETS = ("smooth", "low_volume", "intermittent")


def segment(profiles: pd.DataFrame) -> dict[str, list[str]]:
    return {
        bucket: profiles.loc[profiles["bucket"] == bucket, "unique_id"].tolist()
        for bucket in BUCKETS
    }


if __name__ == "__main__":
    profiles = pd.read_csv(PROFILES_PATH)
    groups = segment(profiles)

    print("Routing summary:")
    for bucket, skus in groups.items():
        ramp = profiles.loc[profiles["unique_id"].isin(skus), "ramp_up"].sum()
        print(f"  {bucket:<15} {len(skus):>5} SKUs  ({int(ramp)} ramp-up)")
