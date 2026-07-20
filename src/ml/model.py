# ML model: global LightGBM ratio model built on the structural baseline.
#
# Model versions are feature lists (FEATURES_V0, FEATURES_V1, ...); every
# version is the same RatioLGBM class. Features are added one hypothesis at
# a time (design doc Section 5.1), judged by the Section 1.5 decision rule,
# and chronicled in the design doc's Section 6 version log.
#
# The structure (design Sections 4.5-4.10, 4.17):
#   LEVEL   trailing 12-week mean of the SKU's ADJUSTED series, where
#           adjusted = y / seasonal_factor for long SKUs, raw y for short
#           SKUs (per-segment seasonality, Section 4.17).
#   TARGET  ratio = adjusted_sales(anchor + lead) / level(anchor)
#   LOSS    L1 on the ratio, sample_weight = level  (equals the pooled-WAPE
#           numerator, Section 4.6)
#   OUTPUT  yhat = predicted_ratio x level x (target week's factor if long)
#   FLOOR   predicting ratio = 1.0 everywhere reproduces the structural
#           baseline exactly.
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deseasonalize import _factors  # noqa: E402
from src.ml.dataset import asof_history_length  # noqa: E402

EPS = 1e-9
MIN_ANCHOR_AGE_WEEKS = 4   # anchors younger than this have no usable level
WINSOR_Q = 0.995           # ratio-target clip quantile (computed on train)

# Feature sets by model version (design doc Section 6). Every version is the
# same RatioLGBM class with a different feature list.
FEATURES_V0 = ["lead"]
FEATURES_V1 = ["lead", "ramp_4_12", "y_last_r", "lag_1_r"]  # + ramp block


def long_sku_set(profiles: pd.DataFrame, cutoff) -> set[str]:
    """SKUs that are 'long' (medium or full) as of the cutoff."""
    seg = asof_history_length(profiles, cutoff).astype("object")
    return set(seg.index[seg.isin(["medium", "full"])])


def adjusted_series(weekly: pd.DataFrame, long_uids: set[str]) -> pd.DataFrame:
    """Add y_adj: seasonally adjusted sales for long SKUs, raw for short.

    This is the per-segment seasonal policy from design Section 4.17 applied
    to the training history. All levels and ratio targets are computed on
    y_adj, so the model never sees seasonal structure for long SKUs and sees
    raw reality for short ones.
    """
    df = weekly.sort_values(["unique_id", "ds"]).reset_index(drop=True).copy()
    is_long = df["unique_id"].isin(long_uids).to_numpy()
    factors = _factors(df["ds"]).to_numpy()
    df["y_adj"] = df["y"] / np.where(is_long, factors, 1.0)
    return df


def build_matrix(
    train: pd.DataFrame,
    horizon: int,
    cutoff,
    profiles: pd.DataFrame,
    for_training: bool,
) -> pd.DataFrame:
    """One row per (unique_id, anchor week, lead).

    for_training=True  -> anchors are all historical weeks with at least
        MIN_ANCHOR_AGE_WEEKS of history; each row carries the realized
        ratio target and its weight.
    for_training=False -> anchors are the cutoff week only; rows carry
        tgt_factor so predictions can be reseasonalized.

    """
    long_uids = long_sku_set(profiles, cutoff)
    df = adjusted_series(train, long_uids)

    g = df.groupby("unique_id")["y_adj"]
    df["level"] = (
        g.rolling(12, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df["weeks_live"] = df.groupby("unique_id").cumcount() + 1

    # Trajectory feature block: the SKU's recent path relative to its own
    # 12-week level, inclusive of the anchor week (a completed week at
    # prediction time). Scale-free.
    df["y_feat"] = df["y_adj"]
    gf = df.groupby("unique_id")["y_feat"]
    feat_level = (
        gf.rolling(12, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    denom = feat_level.clip(lower=EPS)
    roll4 = gf.rolling(4, min_periods=1).mean().reset_index(level=0, drop=True)
    df["ramp_4_12"] = roll4 / denom          # last month vs last quarter
    df["y_last_r"] = df["y_feat"] / denom    # anchor week vs level
    df["lag_1_r"] = gf.shift(1) / denom      # week before anchor vs level

    if for_training:
        anchors = df[df["weeks_live"] >= MIN_ANCHOR_AGE_WEEKS]
    else:
        anchors = df[df["ds"] == df["ds"].max()]

    targets = df[["unique_id", "ds", "y_adj"]].rename(
        columns={"ds": "tgt_ds", "y_adj": "tgt_y_adj"}
    )

    parts = []
    for h in range(1, horizon + 1):
        p = anchors.copy()
        p["lead"] = h
        p["tgt_ds"] = p["ds"] + pd.Timedelta(weeks=h)
        if for_training:
            p = p.merge(targets, on=["unique_id", "tgt_ds"], how="inner")
        parts.append(p)
    mat = pd.concat(parts, ignore_index=True)

    if for_training:
        mat["ratio"] = mat["tgt_y_adj"] / mat["level"].clip(lower=EPS)
        mat["weight"] = mat["level"]
    else:
        is_long = mat["unique_id"].isin(long_uids).to_numpy()
        tgt_factors = _factors(mat["tgt_ds"]).to_numpy()
        mat["tgt_factor"] = np.where(is_long, tgt_factors, 1.0)
    return mat


def structural_baseline(
    train: pd.DataFrame, test: pd.DataFrame, profiles: pd.DataFrame, cutoff
) -> pd.DataFrame:
    """The Section 3 baseline as a prediction frame (ratio = 1.0 everywhere).

    Library version of scripts/ml_03 logic so every experiment compares
    against the identical baseline.
    """
    mat = build_matrix(train, test["ds"].nunique(), cutoff, profiles, for_training=False)
    out = mat[["unique_id", "tgt_ds", "level", "tgt_factor"]].copy()
    out["yhat"] = out["level"] * out["tgt_factor"]
    return out.rename(columns={"tgt_ds": "ds"})[["unique_id", "ds", "yhat"]]


class RatioLGBM:
    """Global LightGBM on the ratio target. v0: lead-only features."""

    PARAMS = dict(
        objective="regression_l1",
        n_estimators=3000,          # cap; early stopping picks the real count
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=200,
        colsample_bytree=1.0,       # v0 has one feature; sampling is meaningless
        verbose=-1,
    )

    def __init__(
        self,
        horizon: int,
        features: list[str],
    ):
        # `features` is deliberately required, with no default. It used to
        # default to the current version's feature list, which meant an older
        # experiment script silently inherited whatever the newest version
        # used: ml_05 (v0, lead-only) began training on the v1 ramp block when
        # that block became the default, so v0 stopped being reproducible.
        # Every experiment must now name the feature set it is testing.
        self.horizon = horizon
        self.features = list(features)
        self.model = None
        self.clip_hi = None

    def fit(
        self,
        train: pd.DataFrame,
        profiles: pd.DataFrame,
        cutoff,
        val_uids: set,
    ) -> "RatioLGBM":
        import lightgbm as lgb

        mat = build_matrix(train, self.horizon, cutoff, profiles, for_training=True)

        self.clip_hi = float(mat["ratio"].quantile(WINSOR_Q))
        mat["ratio"] = mat["ratio"].clip(upper=self.clip_hi)

        is_val = mat["unique_id"].isin(val_uids)
        tr, va = mat[~is_val], mat[is_val]

        self.model = lgb.LGBMRegressor(**self.PARAMS)
        self.model.fit(
            tr[self.features], tr["ratio"],
            sample_weight=tr["weight"],
            eval_set=[(va[self.features], va["ratio"])],
            eval_sample_weight=[va["weight"]],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        return self

    def predict(
        self, train: pd.DataFrame, profiles: pd.DataFrame, cutoff
    ) -> pd.DataFrame:
        mat = build_matrix(train, self.horizon, cutoff, profiles, for_training=False)
        r_hat = np.clip(self.model.predict(mat[self.features]), 0.0, self.clip_hi)
        out = mat[["unique_id", "tgt_ds", "level", "tgt_factor"]].copy()
        out["yhat"] = r_hat * out["level"] * out["tgt_factor"]
        return out.rename(columns={"tgt_ds": "ds"})[["unique_id", "ds", "yhat"]]

    def importance(self) -> pd.DataFrame:
        return (
            pd.DataFrame({
                "feature": self.features,
                "gain": self.model.booster_.feature_importance("gain"),
            }).sort_values("gain", ascending=False).reset_index(drop=True)
        )
