"""SKU Detail — everything about one SKU.

Answers three questions from the plan (Section 1.3): "how much should I order for
this SKU, and why that number", "is this forecast reliable for this particular
SKU", and "why is this number what it is".

Layout follows PLAN.md Section 3.1 and the agreed mockup. The order quantity and
its reliability sit side by side above the fold, because a user arrives here from
the Action List having already seen the recommended number and needs to know both
how it was derived and whether to trust it. The chart is supporting evidence and
sits below.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from lib import calc, data as D, quality, reliability as R, ui

ui.setup_page("SKU Detail")

#: Session key for the SKU selectbox. Named because it is written to before the
#: widget is built, which is the supported way to change a widget that already
#: holds state; `index=` cannot do it.
SKU_WIDGET_KEY = "sku_select"

params = ui.sidebar_params()
# build_planning_table attaches reliability itself, since safety stock depends on it.
plan = calc.build_planning_table(params)
skus = plan["unique_id"].tolist()

# ---------------------------------------------------------------------------
# Which SKU. Honours a selection handed over by the Action List.
# ---------------------------------------------------------------------------
ui.back_link("Action_List.py", "← Back to Action List")

# A SKU link in the action table arrives as ?sku=... in the URL. It is consumed
# once and remembered, rather than read on every rerun: if it were re-applied
# each time, the query string would keep overriding the selectbox and the SKU
# could not be changed from this page. Comparing against the last consumed value
# means a later click on a different SKU still takes effect. Nothing is written
# back to the URL, which keeps this free of rerun side effects.
#
# The incoming SKU is written to the SELECTBOX'S OWN KEY, not to a separate
# variable fed through `index=`. That distinction is the whole fix for a bug
# where a clicked SKU appeared for an instant and then snapped back to the
# previous one: `index` supplies a default only while the widget has no stored
# state, and Streamlit keeps that state across page navigation, so once the
# selectbox had ever been used its stored value won and the argument was
# ignored. Setting the key is the only way to move a widget that already exists.
# It must happen before the widget is created; Streamlit rejects the write
# afterwards.
_incoming = st.query_params.get(ui.SKU_QUERY_PARAM)
if _incoming and _incoming != st.session_state.get("_sku_from_url"):
    st.session_state["_sku_from_url"] = _incoming
    if _incoming in skus:
        st.session_state[SKU_WIDGET_KEY] = _incoming

# Repair the stored value before the widget reads it. It can fall outside the
# list when the served set changes underneath a live session, which happens
# whenever a SKU is demoted to intermittent between forecast runs.
if st.session_state.get(SKU_WIDGET_KEY) not in skus:
    st.session_state[SKU_WIDGET_KEY] = skus[0]

sku = st.selectbox("SKU", skus, key=SKU_WIDGET_KEY, label_visibility="collapsed")

if _incoming and _incoming not in skus:
    st.warning(f"No SKU matching “{_incoming}”. Showing {sku} instead.")

row = plan[plan["unique_id"] == sku].iloc[0]
fc = D.sku_forecast(sku)
fc_row = fc.iloc[0] if not fc.empty else None
snap = D.forecast_snapshot_date()
trained = snap.date().isoformat() if snap is not None else "unknown"

ui.sku_header(row, fc_row, trained)
ui.quality_flags_inline(quality.flags_by_sku(plan).get(sku, []))
ui.sample_data_warning()

# Sits above the order card deliberately. The caveat is about the number in that
# card, so it has to be read before it, not after.
ui.runs_high_note(row, len(fc) if not fc.empty else 13)

# ---------------------------------------------------------------------------
# The decision, and whether it can be trusted. Side by side, deliberately.
# ---------------------------------------------------------------------------
left, right = st.columns(2)
with left:
    ui.order_card(
        calc.order_quantity_breakdown(row, params),
        int(row["recommended_order_qty"]),
        int(params["lead_time_weeks"]),
        int(params["review_period_weeks"]),
        band=calc.order_quantity_range(row),
        error=row.get("error_used"),
    )
with right:
    acc = D.load_ml_accuracy_by_sku()
    version = R.served_version()
    windows = acc[(acc["unique_id"] == sku)]
    if version is not None and "model_version" in windows.columns:
        windows = windows[windows["model_version"] == version]
    ui.reliability_card(
        row.get("wape"), str(row.get("tier", "none")),
        windows, int(row["recommended_order_qty"]),
    )

st.write("")

# ---------------------------------------------------------------------------
# How the forecast performed when it was tested. The reliability card gives the
# number; this gives the reason. Open by default when the SKU's error is in the
# poor tier, because that is exactly when the number alone is not enough and the
# user is going to want to see what happened.
# ---------------------------------------------------------------------------
_tier = str(row.get("tier", "none"))
if not windows.empty:
    with st.expander(
        f"How the forecast was tested · {len(windows)} "
        f"backtest window{'s' if len(windows) != 1 else ''}",
        expanded=(_tier == "poor"),
    ):
        _hist = D.sku_sales_history(sku)
        if _hist.empty:
            st.info("No sales history to chart for this SKU.")
        else:
            _win = windows.sort_values("cutoff")
            _weekly = D.sku_backtest_weekly(sku, version)
            ui.backtest_strip(_win)
            st.altair_chart(
                ui.backtest_chart(_hist, _win, weekly=_weekly),
                use_container_width=True,
            )
            if _weekly.empty:
                st.caption(
                    f"Shaded blocks are the {ui.BACKTEST_WEEKS}-week windows the model "
                    "was scored on, and the grey dashed line at the left edge of each is "
                    "the cutoff: everything to its left is what the model had seen when "
                    "it made that prediction. Per-week predictions have not been "
                    "generated for this model version, so the prediction is shown as a "
                    "flat average per week. Run scripts/ml_backtest_weekly.py to replace "
                    "it with the actual weekly curve."
                )
            else:
                st.caption(
                    f"Shaded blocks are the {ui.BACKTEST_WEEKS}-week windows the model "
                    "was scored on. The grey dashed line at the left edge of each is the "
                    "cutoff: everything to its left is what the model had seen when it "
                    "made that prediction, everything inside the block is what it had "
                    "not. The dashed line with points is what it predicted week by week; "
                    "the solid line is what actually happened. Hover a point for the "
                    "week, the lead, and the predicted value."
                )

st.write("")

# ---------------------------------------------------------------------------
# Inventory position.
# ---------------------------------------------------------------------------
eta = str(row.get("inbound_eta") or "").strip()
inbound = float(row.get("confirmed_inbound") or 0)
days = float(row.get("days_to_stockout", float("inf")))
stockout = str(row.get("estimated_stockout_date") or "")
if stockout:
    try:
        stockout_label = pd.Timestamp(stockout).strftime("%-d %b")
    except Exception:
        stockout_label = stockout
else:
    stockout_label = "—"
when = "" if not pd.notna(days) or days == float("inf") else (
    " (today)" if days < 1 else f" ({days:.0f} day{'s' if days >= 2 else ''})")

ui.stat_strip([
    # Available to sell, i.e. physical stock less what is already allocated to
    # unshipped orders. Not the same as on-hand, and labelled accordingly.
    ("available", f"{float(row['available_inventory']):,.0f}", False),
    ("preorder backlog", f"{float(row['preorder_backlog']):,.0f}", False),
    ("confirmed inbound", f"{inbound:,.0f}" if inbound else "—", False),
    (f"stocks out{when}", stockout_label, bool(pd.notna(days) and days <= 14)),
    ("30-day sales", f"{float(row['recent_units']):,.0f}", False),
    ("avg per day", f"{float(row['avg_daily_sales']):,.1f}", False),
])

st.write("")

# ---------------------------------------------------------------------------
# Actual demand against both forecasts.
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="dfx" style="font-size:9.5px;letter-spacing:.08em;text-transform:uppercase;'
    'opacity:.6;margin-bottom:2px">Weekly demand · actual and forecast</div>',
    unsafe_allow_html=True,
)

HIST_WEEKS = 26
hist = D.sku_sales_history(sku).tail(HIST_WEEKS)[["ds", "y"]]
hist = hist.rename(columns={"y": "value"}).assign(series="Actual sales")

parts = [hist]
if not fc.empty:
    # Join the forecast lines to the last actual so the chart reads continuously.
    bridge = hist.tail(1)
    model = fc[["ds", "yhat"]].rename(columns={"yhat": "value"}).assign(series="Model forecast")
    parts.append(pd.concat([bridge.assign(series="Model forecast"), model]))
    if "v1_yhat" in fc.columns and fc["v1_yhat"].notna().any():
        v1 = (fc.loc[fc["v1_yhat"].notna(), ["ds", "v1_yhat"]]
              .rename(columns={"v1_yhat": "value"}).assign(series="Spreadsheet (V1)"))
        parts.append(pd.concat([bridge.assign(series="Spreadsheet (V1)"), v1]))
    # For a SKU whose demand is falling, the flat 4-week average is the figure the
    # model was measured against and lost to. Drawn only for those SKUs: adding a
    # fourth line to every chart would cost more in clutter than it returns.
    if row.get("forecast_runs_high") and pd.notna(row.get("wa4")):
        wa4 = pd.DataFrame({"ds": fc["ds"], "value": float(row["wa4"])}).assign(
            series="Recent average (4wk)")
        parts.append(pd.concat([bridge.assign(series="Recent average (4wk)"), wa4]))

chart_df = pd.concat(parts, ignore_index=True)
st.altair_chart(
    ui.demand_chart(chart_df,
                    boundary=hist["ds"].max() if not fc.empty else None,
                    height=400),
    width="stretch",
)
st.caption(
    f"Last {len(hist)} weeks of actual demand, then the {len(fc)}-week forecast. "
    "The dashed vertical line marks where history ends."
)

# ---------------------------------------------------------------------------
# The same figures as numbers. Collapsed: the chart answers the shape question,
# the table answers the per-week planning question, which is asked less often.
# ---------------------------------------------------------------------------
with st.expander("Weekly figures"):
    if fc.empty:
        st.info("No forward forecast for this SKU.")
    else:
        wk = fc[["ds", "yhat"]].rename(columns={"ds": "Week", "yhat": "Model forecast"})
        if "v1_yhat" in fc.columns:
            wk["Spreadsheet (V1)"] = fc["v1_yhat"].to_numpy()
        wk["Week"] = pd.to_datetime(wk["Week"]).dt.date
        st.dataframe(wk.round(1), width="stretch", hide_index=True)
