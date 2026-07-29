#!/usr/bin/env python3
"""Generate the management-summary forecast charts.

Produces the two continuous-time weekly charts used in the management document
(Demand_Forecasting_Project_Summary.docx at the repo root): actual weekly demand
against the current spreadsheet formula and the new v11 hybrid model, one chart
for established (long-history) products and one for newer (short-history)
products, over the three development windows laid end to end.

Pipeline
--------
1. Load the pinned data snapshot (config.ML_DATA_SNAPSHOT) and keep smooth SKUs.
2. For each of the three development windows, fit the v9 shared model and the
   v11 long-only model and predict per week; compute the current formula's
   weekly forecast (its 70-day total spread across the weeks by the monthly
   seasonal factor) and the actual weekly demand. Only cutoff-eligible SKUs are
   scored, and the segment label is as-of the cutoff, matching the evaluation
   harness (src/ml/evaluate.py).
3. Aggregate to per-segment weekly totals over the contiguous 30-week span.
4. Render two Altair charts (monotone-smoothed), exported as PNG to
   outputs/reports/ via vl-convert.

The three windows are contiguous under W-MON, so the weeks concatenate into one
continuous timeline (Oct 2025 to May 2026). Each ten-week window is forecast by
the model trained at its start, so the forecast lines can step at the window
boundaries; those boundaries are drawn as labelled dashed rules. The newer chart
starts in Dec 2025 because too few short SKUs were eligible at the Oct cutoff
(14) to compare.

Dependencies
------------
Beyond the ML track, this needs `altair` and `vl-convert-python` (listed in
requirements.txt under the unpinned viz deps) and `lightgbm` (pinned). Run with
the repo venv:

    .venv/bin/pip install -r requirements.txt   # if lightgbm/altair are missing
    .venv/bin/python scripts/plot_management_forecast_charts.py
"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

import altair as alt
import pandas as pd

from compare_v1 import SEASONAL, build_cumsum_index, v1_forecast
from src.ml.dataset import (asof_history_length, dev_splits, eligible_skus,
                            load_weekly, stratified_val_skus)
from src.ml.model import (FEATURES_V1, FEATURES_V11_LONG, RatioLGBM,
                          long_sku_set)

OUTDIR = ROOT / "outputs" / "reports"
WINDOWS = ["Mar-May", "Dec-Feb", "Oct-Dec"]  # development windows, newest first

# ─────────────────────────────────────────────────────────────────────────────
# CHART CONFIG — edit these to change appearance. Nothing below this block
# normally needs touching.
# ─────────────────────────────────────────────────────────────────────────────

# Per-line styling. Rename a series, recolour it, or change its dash here.
# Dash [1, 0] = solid; e.g. [9, 4] = long dashes, [3, 3] = short dashes/dots.
#          internal column,  legend label,      colour,     dash pattern
SERIES = [
    ("actual",  "Actual demand",  "#111827", [1, 0]),
    ("current", "Current method", "#C0504D", [9, 4]),
    ("new",     "New model",      "#2E75B6", [3, 3]),
]

STYLE = {
    "y_title":        "Weekly demand (units)",   # y-axis label
    "x_date_format":  "%b %Y",                   # x-axis tick format (e.g. "%b %Y", "%b")
    "boundary_label": "new 10-week forecast",     # text printed at each retraining line
    "width":          640,                        # plot width  (px, before export scaling)
    "height":         250,                        # plot height (px, before export scaling)
    "scale_factor":   2.5,                        # PNG resolution multiplier
}

# One entry per chart to produce:
#   segment       "long" (established) or "short" (newer)
#   title         chart title; "" for none (the Word doc supplies its own caption)
#   start_week    earliest week to show, or None for all (newer starts 2025-12-22)
#   boundaries    dates to draw the dashed "new forecast" retraining lines
#   filename      output PNG name (written under outputs/reports/)
CHARTS = [
    {"segment": "long",  "title": "", "start_week": None,
     "boundaries": ["2025-12-22", "2026-03-02"], "filename": "management_chart_established.png"},
    {"segment": "short", "title": "", "start_week": "2025-12-22",
     "boundaries": ["2026-03-02"], "filename": "management_chart_newer.png"},
]

# Derived from SERIES (do not edit).
ORDER = [label for _c, label, _col, _d in SERIES]
COLORS = [col for _c, _l, col, _d in SERIES]
DASH = [d for _c, _l, _col, d in SERIES]
SERIES_NAMES = {col: label for col, label, _col, _d in SERIES}


def weekly_series() -> pd.DataFrame:
    """Per-segment weekly totals of actual demand, current formula, and v11."""
    weekly, profiles = load_weekly()
    smooth = set(profiles.loc[profiles["bucket"] == "smooth", "unique_id"])
    weekly = weekly[weekly["unique_id"].isin(smooth)]

    raw = pd.read_parquet(ROOT / "data" / "processed" / "orders_raw.parquet")
    raw["order_date"] = pd.to_datetime(raw["order_date"])
    raw = raw[raw["unique_id"].isin(smooth)]
    index = build_cumsum_index(raw)

    def v1_weekly(split, skus, keep):
        # Spread the current formula's 70-day total across the 10 test weeks by
        # each week's monthly seasonal factor, so the weekly sum equals the
        # scored horizon total and the line carries the method's seasonal shape.
        asof = split.cutoff - pd.Timedelta(days=1)
        tw = sorted(split.test["ds"].unique())
        fac = {d: SEASONAL[pd.Timestamp(d).month] for d in tw}
        fsum = sum(fac.values())
        rows = []
        for u in skus:
            if u not in keep:
                continue
            tot = v1_forecast(index, u, asof)
            rows += [{"unique_id": u, "ds": d, "v1": tot * fac[d] / fsum} for d in tw]
        return pd.DataFrame(rows)

    parts = {"actual": [], "current": [], "new": []}
    for split, _name in zip(dev_splits(weekly, n=3), WINDOWS):
        skus = sorted(split.train["unique_id"].unique())
        longs = long_sku_set(profiles, split.cutoff) & set(skus)
        keep = eligible_skus(profiles, split.cutoff)
        seg = (asof_history_length(profiles, split.cutoff).astype("object")
               .replace({"medium": "long", "full": "long"}))

        val_all = stratified_val_skus(split.train, profiles)
        val_long = stratified_val_skus(
            split.train[split.train["unique_id"].isin(longs)], profiles)
        m9 = RatioLGBM(split.horizon, FEATURES_V1, deseas_features=True,
                       deseas_all=True).fit(split.train, profiles, split.cutoff, val_all)
        p9 = m9.predict(split.train, profiles, split.cutoff)
        m_long = RatioLGBM(split.horizon, FEATURES_V11_LONG, deseas_features=True,
                           deseas_all=True, uids=longs).fit(
            split.train, profiles, split.cutoff, val_long)
        p11 = pd.concat(
            [p9[~p9["unique_id"].isin(longs)], m_long.predict(split.train, profiles, split.cutoff)],
            ignore_index=True)
        p11 = p11[p11["unique_id"].isin(keep)]
        act = split.test[split.test["unique_id"].isin(keep)][["unique_id", "ds", "y"]]
        v1w = v1_weekly(split, skus, keep)

        for frame, col, val in [(act, "actual", "y"), (v1w, "current", "v1"), (p11, "new", "yhat")]:
            g = frame.copy()
            g["segment"] = g["unique_id"].map(seg)
            parts[col].append(
                g.groupby(["segment", "ds"])[val].sum().reset_index().rename(columns={val: col}))

    wide = pd.concat(parts["actual"])
    for col in ("current", "new"):
        wide = wide.merge(pd.concat(parts[col]), on=["segment", "ds"], how="left")
    return wide.sort_values(["segment", "ds"])


def make_chart(sub: pd.DataFrame, boundaries: list[str], path: Path, title: str = "") -> None:
    m = sub.melt(id_vars=["ds"], value_vars=["actual", "current", "new"],
                 var_name="series", value_name="units")
    m["series"] = m["series"].map(SERIES_NAMES)

    color = alt.Color("series:N", sort=ORDER, title=None,
                      scale=alt.Scale(domain=ORDER, range=COLORS),
                      legend=alt.Legend(orient="top", direction="horizontal",
                                        symbolStrokeWidth=3, labelFontSize=12))
    dash = alt.StrokeDash("series:N", sort=ORDER, legend=None,
                          scale=alt.Scale(domain=ORDER, range=DASH))
    lines = alt.Chart(m).mark_line(interpolate="monotone", strokeWidth=2.6).encode(
        x=alt.X("ds:T", axis=alt.Axis(format=STYLE["x_date_format"], tickCount="month",
                                      title=None, labelAngle=0, grid=False, labelFontSize=11)),
        y=alt.Y("units:Q", title=STYLE["y_title"],
                axis=alt.Axis(format="~s", grid=True, gridColor="#EEEEEE", titleFontSize=11)),
        color=color, strokeDash=dash)

    bdf = pd.DataFrame({"b": pd.to_datetime(boundaries)})
    bdf["label"] = STYLE["boundary_label"]
    rules = alt.Chart(bdf).mark_rule(
        color="#8A9199", strokeDash=[5, 4], strokeWidth=1.3).encode(x="b:T")
    rule_lab = alt.Chart(bdf).mark_text(
        align="center", baseline="top", dy=1, fontSize=9.5, fontStyle="italic",
        color="#6B7280").encode(x="b:T", y=alt.value(2), text="label:N")

    props = {"width": STYLE["width"], "height": STYLE["height"]}
    if title:
        props["title"] = alt.TitleParams(title, fontSize=14, color="#1F2937", anchor="start", dx=40)
    ch = (rules + rule_lab + lines).properties(**props)
    ch = ch.configure_view(stroke=None).configure_axis(labelColor="#374151", titleColor="#374151")
    ch.save(str(path), scale_factor=STYLE["scale_factor"], engine="vl-convert")
    print("wrote", path)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    wide = weekly_series()
    for c in CHARTS:
        sub = wide[wide["segment"] == c["segment"]]
        if c["start_week"]:
            sub = sub[sub["ds"] >= c["start_week"]]
        make_chart(sub.sort_values("ds"), c["boundaries"], OUTDIR / c["filename"], c["title"])


if __name__ == "__main__":
    main()
