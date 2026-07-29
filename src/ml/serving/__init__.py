"""Servable ML forecasting: package a decided model version, train it, save it,
and produce per-SKU forward forecasts for the dashboard.

Public API:
    get_model(version=None, horizon=13) -> ForecastModel
    CURRENT_BEST                         -> str
    REGISTRY                             -> {version: class}
    forward_forecast(...)                -> (forecast_table, fitted_model)
    validate_version(...)                -> per-window per-segment pooled WAPE
    validate_version_detail(...)         -> the same, before aggregation (per-SKU)
    validate_version_weekly(...)         -> the same again, per SKU per test week
    save_model / load_model
    v1_forward(...)                      -> V1 decomposed onto a forecast grid
    validate_v1(...)                     -> V1's per-window per-segment pooled WAPE
    validate_v1_detail(...)              -> the same, before aggregation (per-SKU)
"""

from src.ml.serving.base import ForecastModel
from src.ml.serving.models import CURRENT_BEST, REGISTRY, get_model
from src.ml.serving.forecast import (
    forward_forecast,
    validate_version,
    validate_version_detail,
    validate_version_weekly,
    FORWARD_COLUMNS,
)
from src.ml.serving.persist import load_model, save_model
from src.ml.serving.v1 import v1_forward, validate_v1, validate_v1_detail

__all__ = [
    "ForecastModel",
    "CURRENT_BEST",
    "REGISTRY",
    "get_model",
    "forward_forecast",
    "validate_version",
    "validate_version_detail",
    "validate_version_weekly",
    "FORWARD_COLUMNS",
    "save_model",
    "load_model",
    "v1_forward",
    "validate_v1",
    "validate_v1_detail",
]
