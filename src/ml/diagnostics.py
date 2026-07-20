"""Structural diagnostics that a single headline metric hides.

Why this exists
---------------
Pooled WAPE is one number per segment. It cannot show you that a transform
applied BEFORE the model is mis-specified, because the error is absorbed into
the model's residuals and reported as "the model is a bit worse". Four model
versions regressed in the same segment-window (Dec-Feb long) before anyone
checked whether the seasonal transform underneath them was correct. It was not:
the December multiplier over-corrects long SKUs and under-corrects short ones,
in opposite directions.

`seasonal_fit` is the check that would have caught it, and it needs no model.
`residuals_by_month` is the same idea applied to a fitted model's predictions.
Run both alongside every version, not just the headline table.

Reading the output
------------------
Both functions report values normalized so that 1.00 means "as expected".
Deviations are only meaningful where the SKU count is large enough; months
below `min_skus` are flagged and should be ignored. Remember the standing data
limitation: roughly two seasonal cycles exist, so a per-month estimate rests on
two observations and is evidence for a hypothesis, not a measurement (design
doc Section 4.9).

Usage
-----
    .venv/bin/python -m src.ml.diagnostics          # seasonal fit, pinned data
    from src.ml.diagnostics import residuals_by_month
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ML_FINAL_TEST_CUTOFF  # noqa: E402
from src.ml.dataset import asof_history_length, load_weekly  # noqa: E402

MIN_SKUS = 30          # below this a month's estimate is not worth reading
FLAG_AT = 0.10         # |value - 1| above this is called out


def _segment(profiles: pd.DataFrame, cutoff) -> pd.Series:
    seg = asof_history_length(profiles, cutoff).astype("object")
    return seg.map(lambda s: "long" if s in ("medium", "full") else "short")


def seasonal_fit(
    weekly: pd.DataFrame | None = None,
    profiles: pd.DataFrame | None = None,
    cutoff: str | pd.Timestamp = ML_FINAL_TEST_CUTOFF,
    min_skus: int = MIN_SKUS,
) -> pd.DataFrame:
    """Does the seasonal transform actually flatten each segment's demand?

    Computes the demand-weighted mean of (deseasonalized target / level) per
    calendar month per segment, then normalizes by each segment's own mean so
    that growth cancels and only seasonal shape remains.

        1.00  the multiplier fits this month for this segment
        >1    under-corrected: real demand exceeds what the factor allowed for
        <1    over-corrected: the factor cut too hard

    A segment whose values swing far from 1.00 is being handed a mis-specified
    target, and no amount of model work will fix that. A LARGE GAP BETWEEN THE
    TWO SEGMENTS in the same month means one shared multiplier cannot serve
    both, which is a data problem masquerading as a modeling problem.

    Excludes weeks after `cutoff`, which defaults to the quarantined boundary,
    so this can be run freely during development.
    """
    from src.ml.model import build_matrix  # local import: avoids a cycle

    if weekly is None or profiles is None:
        weekly, profiles = load_weekly()
    smooth = set(profiles.loc[profiles["bucket"] == "smooth", "unique_id"])
    w = weekly[weekly["unique_id"].isin(smooth)]
    cutoff = pd.Timestamp(cutoff)
    w = w[w["ds"] <= cutoff]

    mat = build_matrix(w, 1, cutoff, profiles, for_training=True, deseas_all=True)
    mat["month"] = pd.to_datetime(mat["tgt_ds"]).dt.month

    rows = []
    for seg_val, seg_name in [(1, "long"), (0, "short")]:
        g = mat[mat["is_long"] == seg_val]
        if g.empty:
            continue
        per = g.groupby("month").apply(
            lambda d: pd.Series({
                "resid": np.average(d["ratio"], weights=d["weight"]),
                "n_skus": d["unique_id"].nunique(),
            }),
            include_groups=False,
        )
        per["resid"] = per["resid"] / per["resid"].mean()   # cancel growth
        per["segment"] = seg_name
        rows.append(per.reset_index())

    out = pd.concat(rows, ignore_index=True)
    out = out.pivot(index="month", columns="segment", values=["resid", "n_skus"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["gap"] = (out["resid_short"] / out["resid_long"]).round(2)
    out["thin"] = (out[["n_skus_short", "n_skus_long"]].min(axis=1) < min_skus)
    return out.round(3)


def print_seasonal_fit(tbl: pd.DataFrame, flag_at: float = FLAG_AT) -> None:
    """Human-readable seasonal_fit report, with the caveats attached."""
    print("Seasonal transform fit by month and segment")
    print("1.00 = multiplier fits. >1 under-corrected. <1 cut too hard.")
    print("'gap' = short/long; far from 1.00 means one multiplier cannot serve both.\n")
    print(f"{'mon':<5}{'long':>8}{'short':>8}{'gap':>7}{'n_long':>8}{'n_short':>9}   note")
    print("-" * 62)
    worst = []
    for m, r in tbl.iterrows():
        note = []
        if r["thin"]:
            note.append("thin, ignore")
        else:
            for seg in ("long", "short"):
                v = r[f"resid_{seg}"]
                if abs(v - 1) > flag_at:
                    note.append(f"{seg} {'under' if v > 1 else 'over'}-corrected")
            if abs(r["gap"] - 1) > 0.25:
                note.append("SEGMENTS DIVERGE")
                worst.append((m, r["gap"]))
        print(f"{int(m):<5}{r['resid_long']:>8.3f}{r['resid_short']:>8.3f}{r['gap']:>7.2f}"
              f"{int(r['n_skus_long']):>8}{int(r['n_skus_short']):>9}   {'; '.join(note)}")
    if worst:
        months = ", ".join(str(int(m)) for m, _ in sorted(worst, key=lambda x: -abs(x[1] - 1)))
        print(f"\n  Months where the two segments need different multipliers: {months}")
        print("  A shared factor cannot fit both. Consider per-segment multipliers,")
        print("  shrunk toward the hand-set values (design doc Section 5.2).")
    print("\n  Caveat: about two seasonal cycles exist, so each month rests on ~2")
    print("  observations. Treat as a hypothesis to test, not a measurement.")


def residuals_by_month(
    preds: pd.DataFrame,
    split,
    profiles: pd.DataFrame,
    min_skus: int = MIN_SKUS,
) -> pd.DataFrame:
    """Per-month bias and WAPE for one model, split by segment.

    The headline table gives one number per segment for the whole window. This
    breaks it out by calendar month, which is where a seasonal mis-specification
    shows up: a model can look mediocre overall while being badly wrong in one
    month and compensating in another.

    Returns one row per (segment, month) with n_skus, actual, pooled_wape and
    bias_pct. Months under `min_skus` are marked thin.
    """
    seg = _segment(profiles, split.cutoff)
    act = split.test.copy()
    act["month"] = pd.to_datetime(act["ds"]).dt.month
    pr = preds.copy()
    pr["month"] = pd.to_datetime(pr["ds"]).dt.month

    a = act.groupby(["unique_id", "month"])["y"].sum().rename("y")
    f = pr.groupby(["unique_id", "month"])["yhat"].sum().rename("yhat")
    df = pd.concat([a, f], axis=1).fillna(0.0).reset_index()
    df["segment"] = df["unique_id"].map(seg)
    df = df[df["segment"].notna()]

    rows = []
    for (s, m), g in df.groupby(["segment", "month"]):
        actual = g["y"].sum()
        rows.append({
            "segment": s,
            "month": int(m),
            "n_skus": g["unique_id"].nunique(),
            "actual": round(actual, 1),
            "pooled_wape": round((g["yhat"] - g["y"]).abs().sum() / max(actual, 1e-9), 4),
            "bias_pct": round((g["yhat"].sum() / max(actual, 1e-9) - 1) * 100, 1),
            "thin": g["unique_id"].nunique() < min_skus,
        })
    return pd.DataFrame(rows).sort_values(["segment", "month"]).reset_index(drop=True)


if __name__ == "__main__":
    print_seasonal_fit(seasonal_fit())
