# ML Stage 3: model-agnostic scoring.
#
# Every experiment — statistical baseline, LightGBM v1, v2 with GA4 — scores
# through this one function, so numbers are comparable by construction.
#
# Metric = the production convention (run_test_evaluation.py):
#   per SKU:  sum yhat and y over the test window  →  window totals
#   pooled WAPE per segment = Σ|yhat_total − y_total| / Σ y_total
# Heavier-demand SKUs count more; that is intentional (units ≈ dollars).
#
# Leakage guard: score() refuses predictions whose ds is outside the split's
# test window — a cheap tripwire that catches most off-by-one-week bugs.
from __future__ import annotations

import pandas as pd

from config import MIN_SIM_HISTORY_WEEKS
from src.ml.dataset import Split, asof_history_length, eligible_skus


def score(
    preds: pd.DataFrame,
    split: Split,
    profiles: pd.DataFrame,
    min_history_weeks: int | None = MIN_SIM_HISTORY_WEEKS,
) -> pd.DataFrame:
    """Score predictions against a split's test window.

    preds: unique_id / ds / yhat  (one row per SKU per test week; a model
           that predicts window totals directly may pass one row per SKU
           with ds = first test week and yhat = the total).
    Returns one row per segment (bucket × history_length) + TOTAL row:
    n_skus, actual_units, pooled WAPE, bias_pct.

    Eligibility: by default only SKUs with at least `min_history_weeks` weeks
    of history at the split's cutoff are scored (dataset.eligible_skus), so
    backtest windows do not score SKUs that had too little history at the
    time. Pass min_history_weeks=None to score every SKU (not recommended).
    """
    test_weeks = set(split.test["ds"].unique())
    bad = preds.loc[~preds["ds"].isin(test_weeks), "ds"].unique()
    if len(bad):
        raise ValueError(
            f"Predictions contain ds outside the test window: {sorted(bad)[:5]} "
            f"(test = {min(test_weeks).date()} → {max(test_weeks).date()})"
        )

    test = split.test
    if min_history_weeks is not None:
        keep = eligible_skus(profiles, split.cutoff, min_history_weeks)
        preds = preds[preds["unique_id"].isin(keep)]
        test = test[test["unique_id"].isin(keep)]

    # Warn loudly if the model failed to predict some eligible SKUs. They are
    # still scored (as a forecast of zero) so the metric stays honest, but a
    # silent gap here usually means a bug upstream, not a modeling choice.
    missing = set(test["unique_id"].unique()) - set(preds["unique_id"].unique())
    if missing:
        import warnings

        warnings.warn(
            f"score(): {len(missing)} eligible SKUs have no predictions and are "
            f"scored as zero forecasts (e.g., {sorted(missing)[:3]}).",
            stacklevel=2,
        )

    yhat_tot = preds.groupby("unique_id")["yhat"].sum().rename("yhat_total")
    y_tot = test.groupby("unique_id")["y"].sum().rename("y_total")
    df = pd.concat([yhat_tot, y_tot], axis=1).fillna(0.0).reset_index()
    df = df.merge(profiles[["unique_id", "bucket"]], on="unique_id", how="left")
    # Segment = history length AS OF the cutoff (not the snapshot), with
    # medium and full reported together as "long" per design Section 4.4.
    asof = asof_history_length(profiles, split.cutoff)
    df["history_length"] = (
        df["unique_id"].map(asof).astype("object")
        .replace({"medium": "long", "full": "long"})
    )
    df["ae"] = (df["yhat_total"] - df["y_total"]).abs()

    def _row(g: pd.DataFrame, label: str) -> dict:
        actual = g["y_total"].sum()
        return {
            "segment": label,
            "n_skus": len(g),
            "actual_units": round(actual, 1),
            "pooled_wape": round(g["ae"].sum() / max(actual, 1e-9), 4),
            "bias_pct": round((g["yhat_total"].sum() / max(actual, 1e-9) - 1) * 100, 1),
        }

    rows = [
        _row(g, f"{b}/{h}")
        for (b, h), g in df.groupby(["bucket", "history_length"])
    ]
    rows.append(_row(df, "TOTAL"))
    return pd.DataFrame(rows)


def bootstrap_delta(
    preds_a: pd.DataFrame,
    preds_b: pd.DataFrame,
    split: Split,
    profiles: pd.DataFrame,
    segment: str | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Paired SKU-bootstrap of the pooled-WAPE difference between two models.

    Answers: "is A's WAPE really lower than B's, or did the SKU draw get
    lucky?" Resamples SKUs with replacement (paired: each resample scores
    BOTH models on the same SKUs, so shared noise cancels) and reports the
    distribution of WAPE(A) − WAPE(B). Negative delta = A better.

    Decision rule (design doc §1.5): adopt A over B only if |mean_delta|
    exceeds 2 × se across the dev windows' evidence.

    segment: 'short' / 'long' / None (= all smooth pooled). Uses the same
    as-of segment labels and cutoff eligibility filter as score().
    Returns {delta, se, ci_lo, ci_hi} (95% CI).
    """
    import numpy as np

    keep = eligible_skus(profiles, split.cutoff, MIN_SIM_HISTORY_WEEKS)
    test = split.test[split.test["unique_id"].isin(keep)]
    y_tot = test.groupby("unique_id")["y"].sum()
    a_tot = preds_a.groupby("unique_id")["yhat"].sum()
    b_tot = preds_b.groupby("unique_id")["yhat"].sum()
    df = pd.concat(
        [y_tot.rename("y"), a_tot.rename("a"), b_tot.rename("b")], axis=1
    ).fillna(0.0)
    df = df[df.index.isin(keep)]
    if segment is not None:
        seg_map = (
            asof_history_length(profiles, split.cutoff)
            .astype("object")
            .replace({"medium": "long", "full": "long"})
        )
        df = df[df.index.map(seg_map) == segment]

    ae_a = (df["a"] - df["y"]).abs().to_numpy()
    ae_b = (df["b"] - df["y"]).abs().to_numpy()
    y = df["y"].to_numpy()

    point = ae_a.sum() / y.sum() - ae_b.sum() / y.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(df), size=(n_boot, len(df)))
    deltas = (ae_a[idx].sum(1) - ae_b[idx].sum(1)) / y[idx].sum(1)
    return {
        "delta": round(float(point), 4),
        "se": round(float(deltas.std()), 4),
        "ci_lo": round(float(np.percentile(deltas, 2.5)), 4),
        "ci_hi": round(float(np.percentile(deltas, 97.5)), 4),
    }


def ramp_cohort(
    split: Split,
    profiles: pd.DataFrame,
    ratio: float = 1.2,
    min_level: float = 1.0,
) -> set[str]:
    """SKUs that were visibly growing at the split's cutoff.

    Definition (design doc Section 1.4): trailing 4-week mean at least
    `ratio` (1.2) times the trailing 12-week mean, with the 12-week mean at
    least `min_level` (1.0) units/week to exclude near-zero noise. Both means
    use training data only. The cutoff eligibility filter is applied, so the
    cohort contains only SKUs that are scored at all.

    This is the growth-tracking diagnostic: trailing-average methods
    structurally underforecast exactly this cohort.
    """
    g = split.train.sort_values("ds").groupby("unique_id")["y"]
    m4 = g.apply(lambda s: s.tail(4).mean())
    m12 = g.apply(lambda s: s.tail(12).mean())
    growing = set(m12.index[(m12 >= min_level) & (m4 >= ratio * m12)])
    return growing & eligible_skus(profiles, split.cutoff, MIN_SIM_HISTORY_WEEKS)


def cohort_score(
    preds: pd.DataFrame,
    split: Split,
    skus: set[str],
) -> dict:
    """Pooled WAPE and bias for one model on one SKU subset.

    Same conventions as score() (window totals per SKU, pooled), returned as
    a dict so experiments can print cohort lines compactly:
    {n_skus, actual_units, pooled_wape, bias_pct}.
    """
    p = preds[preds["unique_id"].isin(skus)]
    t = split.test[split.test["unique_id"].isin(skus)]
    yhat_tot = p.groupby("unique_id")["yhat"].sum().rename("yhat")
    y_tot = t.groupby("unique_id")["y"].sum().rename("y")
    df = pd.concat([yhat_tot, y_tot], axis=1).fillna(0.0)
    actual = df["y"].sum()
    return {
        "n_skus": len(df),
        "actual_units": round(actual, 1),
        "pooled_wape": round((df["yhat"] - df["y"]).abs().sum() / max(actual, 1e-9), 4),
        "bias_pct": round((df["yhat"].sum() / max(actual, 1e-9) - 1) * 100, 1),
    }


def score_table(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine {model_name: score() output} into one comparison table."""
    parts = []
    for name, tbl in results.items():
        t = tbl.copy()
        t.insert(0, "model", name)
        parts.append(t)
    out = pd.concat(parts, ignore_index=True)
    return out.pivot_table(
        index="segment", columns="model", values="pooled_wape"
    ).round(4)
