"""The model interface every servable version implements.

A "version" is defined entirely by how it assembles predictions from the shared
primitives in src/ml (RatioLGBM, build_matrix, long_sku_set, the seasonal
factors). This layer does not contain modeling logic of its own; it packages a
decided version so it can be trained once, saved, and used to produce forward
forecasts for the dashboard.

To add a new version:
  1. Subclass ForecastModel, implement fit() and predict().
  2. Register it in serving.models.REGISTRY.
  3. Point CURRENT_BEST at it once it wins under the design doc's decision rule.
Adding a version must never change an existing version's output. The existing
versions compose frozen primitives, so this holds by construction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class ForecastModel(ABC):
    """Fit on history up to a cutoff, then predict a forward horizon.

    predict() returns a tidy frame with columns unique_id / ds / yhat, one row
    per SKU per forecast week, in the same shape src/ml/evaluate.score expects.
    """

    #: short identifier used in the registry, saved artifacts, and the dashboard.
    version: str = "base"
    #: one-line human description shown in the UI and saved metadata.
    description: str = ""

    def __init__(self, horizon: int):
        self.horizon = int(horizon)
        self.cutoff: pd.Timestamp | None = None
        #: SKUs the "long" sub-model owns, set during fit; used to label output.
        self.long_uids: set[str] = set()

    @abstractmethod
    def fit(self, train: pd.DataFrame, profiles: pd.DataFrame, cutoff) -> "ForecastModel":
        """Train on all weeks with ds <= cutoff. Returns self."""

    @abstractmethod
    def predict(self, train: pd.DataFrame, profiles: pd.DataFrame, cutoff) -> pd.DataFrame:
        """Return unique_id / ds / yhat for the horizon weeks after cutoff."""

    def served_by(self, unique_id: str) -> str:
        """Which sub-model produced this SKU's forecast (for reporting)."""
        return "long" if unique_id in self.long_uids else "shared"
