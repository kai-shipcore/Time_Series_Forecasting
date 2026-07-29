"""Registered, servable model versions.

Each class here is a faithful packaging of a version already decided and recorded
in the design doc's Section 6 version log. The logic lives in the frozen
primitives (src/ml/model.py); these classes only wire those primitives together
in the exact configuration the version log specifies, so a saved/served model
reproduces the recorded numbers.

CURRENT_BEST names the version the forward pipeline and dashboard use by default.
"""

from __future__ import annotations

import pandas as pd

from src.ml.dataset import stratified_val_skus
from src.ml.model import (
    FEATURES_V1,
    FEATURES_V11_LONG,
    RatioLGBM,
    long_sku_set,
)
from src.ml.serving.base import ForecastModel


class V11Hybrid(ForecastModel):
    """v11 (design doc Section 6, BEST).

    Two models instead of one:
      SHORT SKUs: the shared v9 model (FEATURES_V1, trained on all smooth SKUs),
        fully deseasonalized. Short predictions are identical to v9.
      LONG SKUs: a long-only model (FEATURES_V11_LONG = lead, y_last_r, lag_1_r,
        elev_long), fully deseasonalized, with early-stopping validation SKUs
        re-stratified within the long segment.
    The final forecast takes short SKUs from the shared model and long SKUs from
    the long model. This mirrors scripts/ml_22_v11_hybrid.py exactly.
    """

    version = "v11"
    description = (
        "Hybrid short/long. Short SKUs use the shared v9 model; long SKUs use a "
        "dedicated long-only model with an elevation feature. Both fully "
        "deseasonalized."
    )

    def fit(self, train: pd.DataFrame, profiles: pd.DataFrame, cutoff) -> "V11Hybrid":
        self.cutoff = pd.Timestamp(cutoff)
        self.long_uids = long_sku_set(profiles, cutoff) & set(train["unique_id"].unique())

        val_all = stratified_val_skus(train, profiles)
        long_train = train[train["unique_id"].isin(self.long_uids)]
        val_long = stratified_val_skus(long_train, profiles)

        self.shared = RatioLGBM(
            self.horizon, FEATURES_V1, deseas_features=True, deseas_all=True
        ).fit(train, profiles, cutoff, val_all)

        self.long = RatioLGBM(
            self.horizon, FEATURES_V11_LONG, deseas_features=True, deseas_all=True,
            uids=self.long_uids,
        ).fit(train, profiles, cutoff, val_long)
        return self

    def predict(self, train: pd.DataFrame, profiles: pd.DataFrame, cutoff) -> pd.DataFrame:
        longs = self.long_uids or (
            long_sku_set(profiles, cutoff) & set(train["unique_id"].unique())
        )
        p_short = self.shared.predict(train, profiles, cutoff)
        p_long = self.long.predict(train, profiles, cutoff)
        return pd.concat(
            [p_short[~p_short["unique_id"].isin(longs)], p_long],
            ignore_index=True,
        )


class V14MinChild(V11Hybrid):
    """v14 candidate: v11 with a lower min_child_samples on the shared model.

    Everything else is v11 exactly, including the dedicated long model, so only
    short predictions can move and the long segment is a control that must come
    back identical. See the design doc Section 6, v14, for the hypothesis and the
    pass criteria, both recorded before this ran.

    Subclasses rather than edits V11Hybrid, per the registry rule that existing
    versions are never modified: a change to v11's fit would silently re-baseline
    every number already recorded against it.
    """

    version = "v14"
    min_child_samples = 50

    def __init__(self, horizon: int, min_child_samples: int | None = None):
        super().__init__(horizon)
        if min_child_samples is not None:
            self.min_child_samples = int(min_child_samples)
        self.description = (
            f"v11 with min_child_samples={self.min_child_samples} on the shared "
            "(short-serving) model. Long model unchanged."
        )

    def fit(self, train: pd.DataFrame, profiles: pd.DataFrame, cutoff) -> "V14MinChild":
        self.cutoff = pd.Timestamp(cutoff)
        self.long_uids = long_sku_set(profiles, cutoff) & set(train["unique_id"].unique())

        val_all = stratified_val_skus(train, profiles)
        long_train = train[train["unique_id"].isin(self.long_uids)]
        val_long = stratified_val_skus(long_train, profiles)

        self.shared = RatioLGBM(
            self.horizon, FEATURES_V1, deseas_features=True, deseas_all=True,
            params={"min_child_samples": self.min_child_samples},
        ).fit(train, profiles, cutoff, val_all)

        # Untouched, so that any movement in the long segment would be a bug
        # rather than an effect.
        self.long = RatioLGBM(
            self.horizon, FEATURES_V11_LONG, deseas_features=True, deseas_all=True,
            uids=self.long_uids,
        ).fit(train, profiles, cutoff, val_long)
        return self


# Registry of servable versions. Add new versions here; do not edit existing ones.
REGISTRY: dict[str, type[ForecastModel]] = {
    V11Hybrid.version: V11Hybrid,
    V14MinChild.version: V14MinChild,
}

# The version the forward pipeline and dashboard use unless told otherwise.
CURRENT_BEST = V11Hybrid.version


def get_model(version: str | None = None, horizon: int = 13) -> ForecastModel:
    """Instantiate a registered version (defaults to CURRENT_BEST)."""
    version = version or CURRENT_BEST
    if version not in REGISTRY:
        raise KeyError(
            f"Unknown model version '{version}'. Registered: {sorted(REGISTRY)}."
        )
    return REGISTRY[version](horizon)
