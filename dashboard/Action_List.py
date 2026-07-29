"""Action List — the operational home screen.

Answers two questions from the plan (Section 1.3): "what needs my attention right
now" and "when will this SKU run out". Everything else is deferred to SKU Detail.

Layout follows PLAN.md Section 3.1: summary counts act as filters rather than as a
separate overview screen, so the figures are the way into the work instead of a
page the user reads and then navigates away from.

Run with:  .venv/bin/streamlit run dashboard/Action_List.py
"""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from lib import calc, data as D, quality, reliability as R, ui

ui.setup_page("Action List")
ui.forecast_caption()
ui.sample_data_warning()

# ---------------------------------------------------------------------------
# Data. build_planning_table does the joins and the order-quantity maths; this
# page adds no calculations of its own.
# ---------------------------------------------------------------------------
params = ui.sidebar_params()
# build_planning_table attaches reliability itself, since safety stock depends on it.
plan = calc.build_planning_table(params)
metrics = calc.overview_metrics(plan, params)
all_flags = quality.flags_by_sku(plan)

# ---------------------------------------------------------------------------
# Summary counts, doubling as the primary filter.
# ---------------------------------------------------------------------------
FOCUS_ALL = "All forecast SKUs"
FOCUS_OPTIONS = {
    FOCUS_ALL: "forecasted",
    "Preorder priority": "preorder",
    "Out of stock": "out of stock",
    "Best sellers at risk": "best seller risk",
    f"Stocks out ≤{metrics['horizon_days']}d": "out soon",
}
focus = st.session_state.get("focus", FOCUS_ALL)

ui.chip_row([
    ("forecasted", metrics["forecasted_skus"], focus == FOCUS_ALL),
    ("preorder", metrics["preorder_priority"], focus == "Preorder priority"),
    ("out of stock", metrics["out_of_stock"], focus == "Out of stock"),
    ("best seller risk", metrics["best_sellers_at_risk"], focus == "Best sellers at risk"),
    (f"out ≤{metrics['horizon_days']}d", metrics["stockout_within_horizon"],
     focus.startswith("Stocks out")),
    ("units rec.", metrics["total_recommended_order_qty"], False),
])

focus = st.radio(
    "Focus", list(FOCUS_OPTIONS), horizontal=True, key="focus",
    label_visibility="collapsed",
)

# ---------------------------------------------------------------------------
# Filters.
# ---------------------------------------------------------------------------
c1, c2, c3, c4, c5, c6 = st.columns([2.2, 1.3, 1.3, 1.1, 1.3, 1.1])
query = c1.text_input("Search", "", placeholder="Search SKU or product name…",
                      label_visibility="collapsed").strip().lower()
prio = c2.selectbox("Priority", ["Priority: all"] + ui.PRIORITY_ORDER,
                    label_visibility="collapsed")
hist = c3.selectbox("History", ["History: all", "short", "long"],
                    label_visibility="collapsed")
rel = c4.selectbox("Reliability", ["Reliability: all"] + R.TIER_ORDER,
                   label_visibility="collapsed")
cats = ["Category: all"] + sorted(plan["product_category"].dropna().unique().tolist())
cat = c5.selectbox("Category", cats, label_visibility="collapsed")
size_label = c6.selectbox("Rows per page", ["20 per page", "50 per page",
                                            "100 per page", "Show all"],
                          index=1, label_visibility="collapsed")
page_size = None if size_label == "Show all" else int(size_label.split()[0])

# Sort. The default is the worklist order: priority first, most urgent within it.
# Any column can replace it, and reset restores the default in one click.
SORTS = {
    "Priority, then urgency": (["priority", "days_to_stockout"], [True, True]),
    "SKU": (["unique_id"], [True]),
    "Category": (["product_category", "priority"], [True, True]),
    "Available": (["available_inventory"], [False]),
    "Preorder backlog": (["preorder_backlog"], [False]),
    "Confirmed inbound": (["confirmed_inbound"], [False]),
    "30-day sales": (["recent_units"], [False]),
    "13-week forecast": (["forecast_total"], [False]),
    "Stocks out soonest": (["days_to_stockout"], [True]),
    "Order quantity": (["recommended_order_qty"], [False]),
    "Reliability (worst first)": (["error_used"], [False]),
}
s1, s2, s3 = st.columns([2.4, 1.4, 1.0])
sort_by = s1.selectbox("Sort by", list(SORTS), key="sort_by", label_visibility="collapsed")
sort_dir = s2.radio("Direction", ["Default", "Ascending", "Descending"], key="sort_dir",
                    horizontal=True, label_visibility="collapsed")
def _reset_sort() -> None:
    # Must run as a callback: Streamlit refuses writes to a widget's key once the
    # widget has been instantiated, and the button is created after both of them.
    # Callbacks fire before the next rerun builds the widgets, so this is allowed.
    st.session_state["sort_by"] = "Priority, then urgency"
    st.session_state["sort_dir"] = "Default"


s3.button("Reset sort", width="stretch", on_click=_reset_sort,
          disabled=(sort_by == "Priority, then urgency" and sort_dir == "Default"))

view = plan
if focus == "Preorder priority":
    view = view[view["priority_label"] == "Preorder"]
elif focus == "Out of stock":
    view = view[view["available_inventory"] <= 0]
elif focus == "Best sellers at risk":
    view = view[view["best_seller_at_risk"]]
elif focus.startswith("Stocks out"):
    view = view[view["days_to_stockout"] <= params["stockout_horizon_days"]]

if query:
    hay = (view["unique_id"].astype(str).str.lower() + " "
           + view["product_name"].astype(str).str.lower())
    view = view[hay.str.contains(query, na=False)]
if prio != "Priority: all":
    view = view[view["priority_label"] == prio]
if hist != "History: all":
    view = view[view["history_group"] == hist]
if rel != "Reliability: all":
    view = view[view["tier"] == rel]
if cat != "Category: all":
    view = view[view["product_category"] == cat]

cols, ascending = SORTS[sort_by]
if sort_dir != "Default":
    ascending = [sort_dir == "Ascending"] * len(cols)
view = view.sort_values(cols, ascending=ascending, kind="mergesort")

sort_note = sort_by.lower() + ("" if sort_dir == "Default" else f", {sort_dir.lower()}")
st.caption(
    f"{len(view):,} of {len(plan):,} SKUs · sorted by {sort_note} · "
    f"{int(view['recommended_order_qty'].sum()):,} units recommended in this view"
)

ui.quality_summary({k: v for k, v in all_flags.items() if k in set(view["unique_id"])},
                   len(view))

# ---------------------------------------------------------------------------
# Pagination. The page resets whenever the filtered set changes, otherwise a
# user who is on page 5 and then narrows the filters lands on an empty page and
# reads it as "no results".
# ---------------------------------------------------------------------------
signature = (focus, query, prio, hist, rel, cat, size_label, sort_by, sort_dir)
if st.session_state.get("_filter_signature") != signature:
    st.session_state["_filter_signature"] = signature
    st.session_state["page"] = 1

total = len(view)
if page_size is None or total <= page_size:
    page, n_pages, page_view = 1, 1, view
else:
    n_pages = math.ceil(total / page_size)
    page = min(max(1, st.session_state.get("page", 1)), n_pages)
    st.session_state["page"] = page
    start = (page - 1) * page_size
    page_view = view.iloc[start:start + page_size]

    p1, p2, p3 = st.columns([1, 3, 1])
    if p1.button("← Previous", disabled=page <= 1, width="stretch"):
        st.session_state["page"] = page - 1
        st.rerun()
    first, last = (page - 1) * page_size + 1, min(page * page_size, total)
    p2.markdown(
        f"<div style='text-align:center;font-size:12px;opacity:.75;padding-top:6px'>"
        f"Showing {first:,}–{last:,} of {total:,} · page {page} of {n_pages}</div>",
        unsafe_allow_html=True,
    )
    if p3.button("Next →", disabled=page >= n_pages, width="stretch"):
        st.session_state["page"] = page + 1
        st.rerun()

# ---------------------------------------------------------------------------
# Portfolio demand. Follows the filters, so it always describes the SKUs in the
# list below rather than a fixed total that would disagree with them.
# ---------------------------------------------------------------------------
# Placement: between the filters and the table. The filters drive both, so a
# control must not sit below its own output; and buried under the table it would
# be scrolled past. Collapsible, so anyone working the list daily can reclaim the
# vertical space and put the table back at the top.
with st.expander(f"Demand across these {len(view):,} SKUs", expanded=True):
    ids = set(view["unique_id"])
    HIST_WEEKS = 26
    sales = D.load_sales()
    sales = sales[sales["unique_id"].isin(ids)]
    actual = (sales[sales["ds"] > sales["ds"].max() - pd.Timedelta(weeks=HIST_WEEKS)]
            .groupby("ds", as_index=False)["y"].sum()
            .rename(columns={"y": "value"}).assign(series="Actual sales"))

    fcast = D.load_forecasts()
    fcast = fcast[fcast["unique_id"].isin(ids)]
    model_tot = (fcast.groupby("ds", as_index=False)["yhat"].sum()
             .rename(columns={"yhat": "value"}).assign(series="Model forecast"))

    v1 = D.load_v1_forward()
    v1 = v1[v1["unique_id"].isin(ids)] if not v1.empty else v1

    if actual.empty or model_tot.empty:
        st.info("Not enough data to chart this selection.")
    else:
        # Join the forecast lines onto the last actual so the chart reads as one
        # continuous series rather than two floating fragments.
        bridge = actual.tail(1)
        parts = [actual, pd.concat([bridge.assign(series="Model forecast"), model_tot])]
        v1_note = ""
        if not v1.empty:
            v1_tot = (v1.groupby("ds", as_index=False)["v1_yhat"].sum()
                      .rename(columns={"v1_yhat": "value"}).assign(series="Spreadsheet (V1)"))
            parts.append(pd.concat([bridge.assign(series="Spreadsheet (V1)"), v1_tot]))
            # V1 is a separate artifact and can cover fewer SKUs than the model.
            # Where it does, the two lines are sums over different populations,
            # so say so rather than letting them look directly comparable.
            covered = float(view["unique_id"].isin(set(v1["unique_id"])).mean())
            if covered < 1:
                v1_note = (f" V1 covers {covered:.0%} of these SKUs, so its line is "
                           "not directly comparable.")
        st.altair_chart(
            ui.demand_chart(pd.concat(parts, ignore_index=True),
                            boundary=actual["ds"].max(), height=420),
            width="stretch",
        )
        st.caption(
            f"Last {len(actual)} weeks of actual demand, then the {fcast['ds'].nunique()}-week "
            f"forecast, summed across the {len(view):,} SKUs in the current filter. "
            f"The dashed line marks where history ends.{v1_note}"
        )

# ---------------------------------------------------------------------------
# The table. Rows on the current page only; it still scrolls in both directions
# with a pinned header and pinned SKU column.
# ---------------------------------------------------------------------------
if view.empty:
    st.info("No SKUs match these filters.")
else:
    ui.action_table(page_view, flags=all_flags)
    ui.reliability_legend(R.tier_counts(plan))

# ---------------------------------------------------------------------------
# Export. There is no drill-down control here any more: every SKU in the table is
# a link to its detail page, and an anchor is tab-focusable, so a widget beside it
# would only duplicate a path that already works by mouse and by keyboard. To
# reach a SKU that is not on the current page, search or filter to it rather than
# paging to it.
# ---------------------------------------------------------------------------
st.divider()

# Export covers every filtered row, not just the visible page, since the point of
# exporting is to take the whole worklist elsewhere.
st.download_button(
    f"⇩ Export all {total:,} filtered rows (CSV)",
    view.drop(columns=["best_seller", "stockout_soon", "best_seller_at_risk"],
              errors="ignore").to_csv(index=False).encode("utf-8"),
    file_name="action_list.csv",
    mime="text/csv",
)
