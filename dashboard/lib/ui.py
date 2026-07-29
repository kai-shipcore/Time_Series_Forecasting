"""Shared presentation helpers.

Holds the stylesheet and the HTML builders for the pieces that Streamlit's own
widgets cannot express: the summary chip row, and the grouped, horizontally
scrollable action table with its pinned first column. Anything Streamlit renders
well natively is left to Streamlit.

Design notes, kept here because they explain choices that look arbitrary in code:

- The table is custom HTML because the column banding (position / demand /
  action) and the coloured priority and reliability markers carry meaning that a
  plain dataframe cannot show. The cost is that sorting is not free and has to be
  driven by controls above the table. Row clicks, however, are free: this HTML is
  rendered by `st.markdown(unsafe_allow_html=True)` into the main document rather
  than into an iframe, so an ordinary `<a href>` in a cell is an ordinary link.
  The whole first cell of each row is one, pointing at the detail page with the
  SKU in the query string. Anchors are natively tab-focusable, so this is the
  keyboard path as well as the mouse one and needs no widget alongside it.
- The first column is pinned. Ten columns overflow a narrow window, and a
  horizontally scrolled table with no anchor leaves the reader unable to tell
  which SKU a row belongs to.
- Status is never carried by colour alone. Every priority badge and reliability
  marker pairs its colour with a glyph and a text label, so the table survives
  being printed, screenshotted or read by someone with colour blindness.

Development note: editing this file usually requires restarting the Streamlit
server, not just saving. Streamlit re-runs the page script on save, but modules
it imports are already in `sys.modules` and are not reliably re-imported. The
stylesheet and the colour tables here are module-level, so a saved change to
them can appear to do nothing until the server is restarted. Changes to the page
scripts themselves do take effect on save as normal.
"""

from __future__ import annotations

import html
from urllib.parse import quote

import pandas as pd
import streamlit as st

from src.planning import data as D, reliability as R

APP_TITLE = "Demand Forecasting"

# URL slug Streamlit derives from pages/1_SKU_Detail.py: the numeric ordering
# prefix and the .py extension are dropped, underscores are kept. Renaming that
# file changes this value.
SKU_DETAIL_SLUG = "SKU_Detail"
SKU_QUERY_PARAM = "sku"

# Column-band colours. Three hues from outside the status palette (which owns
# red, amber, green and the priority magenta), so a band colour reads as "these
# columns belong together" and never as a judgement about the values.
#
# These are applied as INLINE styles, not via the stylesheet. Streamlit passes
# custom HTML through DOMPurify before it reaches the DOM, and whether a <style>
# block and its class hooks survive that is not verifiable from Python. An
# inline style attribute needs no stylesheet and outranks the theme's own table
# header styling, so it renders the same either way.
BAND = {
    "pos": "#7fb0e3",   # POSITION, blue
    "dem": "#5cc4ba",   # DEMAND, teal
    "act": "#a3a6ee",   # ACTION, indigo
}


BAND_TINT = {"pos": "rgba(127,176,227,.13)", "dem": "rgba(92,196,186,.13)",
             "act": "rgba(163,166,238,.13)"}

# Order-breakdown colours: green adds to the quantity to buy, red subtracts.
ADD_COLOUR = "#5aa86f"
SUB_COLOUR = "#e0736e"

# Forecast-miss scale. Diverging from accurate at the centre, with the two
# directions on deliberately different palettes because they are not equally
# costly: under-forecasting causes stockouts and lost sales, so it escalates
# through the warning colours, while over-forecasting only ties up cash and
# runs cool. A single red/green pair would have said the two were opposites of
# the same thing, which they are not.
#
# Bands are set against the observed distribution of 572 SKU-window misses: the
# central band holds about a third of them and each outer band stays populated
# (12.8%, 19.1% under; 14.3%, 10.5%, 10.0% over).
MISS_SCALE = [
    (-0.50, "#e0736e", "under >50%"),
    (-0.25, "#e09a4a", "under 25–50%"),
    (-0.10, "#d9c04a", "under 10–25%"),
    (0.10,  "#5aa86f", "within ±10%"),
    (0.25,  "#4fb3a6", "over 10–25%"),
    (0.50,  "#5b9dd9", "over 25–50%"),
    (float("inf"), "#4a7fc4", "over >50%"),
]


def miss_colour(pct: float) -> str:
    """Colour for a signed forecast miss, by direction and severity."""
    for upper, colour, _ in MISS_SCALE:
        if pct < upper:
            return colour
    return MISS_SCALE[-1][1]


# Demand chart series, shared by the portfolio chart and the per-SKU chart so a
# line means the same thing on both screens.
DEMAND_COLOURS = {
    "Actual sales": "#e8e8e8",
    "Model forecast": "#a3a6ee",
    "Spreadsheet (V1)": "#5cc4ba",
    # Amber, matching the caveat callout it belongs to. Only drawn for SKUs whose
    # demand is falling, where this flat line is the figure the model was
    # measured against and lost to.
    "Recent average (4wk)": "#e0a750",
}
DEMAND_DASHES = {
    "Actual sales": [1, 0],
    "Model forecast": [5, 3],
    "Spreadsheet (V1)": [2, 3],
    "Recent average (4wk)": [4, 4],
}


def demand_chart(chart_df, boundary=None, height: int = 230,
                 y_title: str = "units / week"):
    """Actual demand against the forecasts, on a tidy ds / value / series frame.

    Shared so the portfolio view and the single-SKU view cannot drift apart in
    colour, dash pattern or the marking of where history ends.

    Hovering reports every series at the nearest week rather than requiring the
    cursor to land on a line. A tooltip attached to the lines themselves only
    fires within a couple of pixels of a 1.8px stroke, which in practice means it
    never fires; the usual remedy is a transparent full-height rule that captures
    the pointer, snaps to the nearest x, and carries the tooltip for all series.
    """
    import altair as alt
    import pandas as _pd

    order = [s for s in DEMAND_COLOURS if s in set(chart_df["series"])]
    base = alt.Chart(chart_df)
    colour = alt.Color(
        "series:N", title=None,
        scale=alt.Scale(domain=order, range=[DEMAND_COLOURS[s] for s in order]),
        legend=alt.Legend(orient="bottom"),
    )

    lines = base.mark_line(strokeWidth=2).encode(
        x=alt.X("ds:T", title=None),
        y=alt.Y("value:Q", title=y_title),
        color=colour,
        strokeDash=alt.StrokeDash(
            "series:N", legend=None,
            scale=alt.Scale(domain=order, range=[DEMAND_DASHES[s] for s in order]),
        ),
    )

    hover = alt.selection_point(
        nearest=True, on="pointerover", fields=["ds"], empty=False, clear="pointerout",
    )
    # Dots appear on every series at the hovered week, so the reader can see
    # which points the numbers refer to.
    dots = lines.mark_point(size=60, filled=True, opacity=0).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
    )
    # One pivoted rule carries the whole tooltip: week, then each series.
    crosshair = (
        base.transform_pivot("series", value="value", groupby=["ds"])
        .mark_rule(color="#9a9a9a", strokeWidth=1)
        .encode(
            x="ds:T",
            opacity=alt.condition(hover, alt.value(0.45), alt.value(0)),
            tooltip=[alt.Tooltip("ds:T", title="Week", format="%d %b %Y")]
            + [alt.Tooltip(f"{s}:Q", title=s, format=",.0f") for s in order],
        )
        .add_params(hover)
    )

    layers = [lines, dots, crosshair]
    if boundary is not None:
        layers.insert(0, alt.Chart(_pd.DataFrame({"ds": [boundary]}))
                      .mark_rule(strokeDash=[4, 4], opacity=.45, color="#9a9a9a",
                                 strokeWidth=1.5)
                      .encode(x="ds:T"))
    return (
        alt.layer(*layers)
        .properties(height=height)
        .configure_view(strokeWidth=0)
        .configure_axis(grid=True, gridOpacity=.12, labelFontSize=11, titleFontSize=11)
        .configure_legend(labelFontSize=11, symbolStrokeWidth=2)
    )


def _band_th(band: str, label: str, *, colspan: int = 1, sep: bool = True,
             group: bool = False) -> str:
    """One colour-coded header cell.

    All band styling lives here rather than being split between this function
    and the stylesheet, so there is a single place to change a band's
    appearance. The tint is an inset shadow rather than a background colour
    because these cells are sticky, and a translucent background would let
    scrolled rows show through them.
    """
    c = BAND[band]
    css = f"box-shadow:inset 0 0 0 999px {BAND_TINT[band]};"
    css += f"border-bottom:2px solid {c};" if group else f"border-bottom:1px solid {c}66;"
    if sep:
        css += "border-left:1px solid rgba(128,128,128,.35);padding-left:11px;"
    cs = f' colspan="{colspan}"' if colspan > 1 else ""
    return (f'<th{cs} style="{css}">'
            f'<span style="color:{c};font-weight:600">{label}</span></th>')

# Priority presentation. Glyphs pair with colour so neither is load-bearing alone.
PRIORITY_ORDER = ["Preorder", "No Stock", "Best Seller", "Routine"]
PRIORITY_GLYPH = {
    "Preorder": "◆", "No Stock": "●", "Best Seller": "★", "Routine": "○",
}
PRIORITY_CLASS = {
    "Preorder": "b-pre", "No Stock": "b-nos", "Best Seller": "b-best",
    "Routine": "b-rou",
}

_CSS = """
<style>
  .dfx{font-size:12px}
  .dfx-warn{font-size:11px;color:var(--text-color,#9a9a9a);opacity:.85;
    border:1px solid rgba(128,128,128,.35);border-left:3px solid #d9a441;
    border-radius:4px;padding:6px 9px;margin:2px 0 12px}
  .dfx-chips{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:6px}
  .dfx-chip{border:1px solid rgba(128,128,128,.35);border-radius:6px;padding:6px 9px;
    flex:1 1 92px;min-width:92px}
  .dfx-chip.on{border-color:#5b9dd9;background:rgba(91,157,217,.12)}
  .dfx-chip-n{font-size:15px;font-weight:600;line-height:1.1}
  .dfx-chip-l{font-size:9.5px;opacity:.7;margin-top:2px;letter-spacing:.02em}
  /* Scrolls in both directions. The header rows pin to the top and the SKU
     column pins to the left, so neither the column meaning nor the row identity
     is lost while scrolling a long list. */
  .dfx-tw{overflow:auto;max-height:68vh;border:1px solid rgba(128,128,128,.35);border-radius:6px}
  .dfx-tw table{border-collapse:separate;border-spacing:0;min-width:760px;width:100%;
    font-variant-numeric:tabular-nums;font-size:12px}
  .dfx-tw th,.dfx-tw td{white-space:nowrap}
  /* Fixed header-row heights so the second row's sticky offset is exact. */
  .dfx-grp th{font-size:9px;letter-spacing:.08em;text-transform:uppercase;opacity:.7;
    font-weight:600;padding:0 11px;height:28px;box-sizing:border-box;text-align:center;
    border-bottom:1px solid rgba(128,128,128,.35);background:var(--background-color,#0e1117);
    position:sticky;top:0;z-index:3}
  .dfx-grp th.g1{text-align:left}
  .dfx-h2 th{font-size:10px;opacity:.7;font-weight:500;padding:0 11px;height:32px;
    box-sizing:border-box;text-align:right;
    border-bottom:1px solid rgba(128,128,128,.35);background:var(--background-color,#0e1117);
    position:sticky;top:28px;z-index:3}
  .dfx-h2 th.l{text-align:left}

  /* Column-band colours are applied inline by _band_th(), not here, so that a
     band's appearance has a single definition. The one rule needed at this
     level is that a banded header must not inherit the dimmed opacity above,
     or the colour is muted. */
  .dfx-grp th[style],.dfx-h2 th[style]{opacity:1}
  .dfx-tw td{padding:7px 11px;text-align:right;border-bottom:1px solid rgba(128,128,128,.18)}
  .dfx-tw td.l{text-align:left}
  .dfx-sep{border-left:1px solid rgba(128,128,128,.35)}
  .dfx-stick{position:sticky;left:0;z-index:2;background:var(--background-color,#0e1117);
    border-right:1px solid rgba(128,128,128,.35)}
  /* Header cells of the pinned column stick in both axes, so they must sit
     above both the row-sticky and the column-sticky cells. */
  .dfx-grp th.dfx-stick,.dfx-h2 th.dfx-stick{z-index:5}
  /* Row hover is a translucent white wash rather than a fixed colour, so it lifts
     whatever background sits beneath it instead of assuming one.
     The pinned column is the awkward case: it paints its own opaque background so
     it can sit over the scrolling cells, so a translucent background-COLOR on the
     row would be painted out by it. Applying the wash as a background-IMAGE on
     that cell instead layers it over the opaque base, which composites to exactly
     the same result as the transparent cells and keeps the band continuous. */
  .dfx-tw tbody tr:hover td{background-color:rgba(255,255,255,.06)}
  .dfx-tw tbody tr:hover td.dfx-stick{
    background-image:linear-gradient(rgba(255,255,255,.06),rgba(255,255,255,.06))}

  .dfx-sku{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:10.5px;line-height:1.3}
  /* The whole first cell is the link, not just the SKU text, so the click target
     is the full row height rather than one short string. The negative margin
     cancels the cell's own padding and the padding then puts it back, which makes
     the anchor fill the cell edge to edge instead of leaving dead borders around
     it. A real anchor also keeps the browser behaviours a link should have: tab
     focus, middle-click to a new tab, a copyable URL, and a status-bar preview. */
  .dfx-cell{display:block;margin:-7px -11px;padding:7px 11px}

  /* Streamlit styles anchors globally, with selectors of the form ".stMarkdown a"
     and "[data-testid=stMarkdownContainer] a". Those are specificity 0-1-1 and a
     bare ".dfx-cell" is 0-1-0, so the plain class loses and every SKU renders in
     the theme's link colour, underlined, permanently. The reset below is
     qualified with the element and marked important deliberately: it overrides a
     third-party stylesheet that is not ours to edit, which is the case important
     exists for. Every state is listed, or the theme recolours the cell on hover
     and drags the subtitle with it.
     Note text-decoration in particular: it propagates from the anchor to its
     descendants and cannot be cancelled by a child, so it has to be killed here
     rather than on the span. */
  .dfx-tw a.dfx-cell,
  .dfx-tw a.dfx-cell:link,
  .dfx-tw a.dfx-cell:visited,
  .dfx-tw a.dfx-cell:hover,
  .dfx-tw a.dfx-cell:focus,
  .dfx-tw a.dfx-cell:active{color:inherit!important;text-decoration:none!important}

  /* Both link cues are held back until the pointer is over that one cell. A
     column of 50 permanently underlined SKUs reads as 50 things already
     highlighted, which is noise; the affordance is only useful at the moment it
     is aimed at. The border is present but transparent when idle so that showing
     it does not shift the text by a pixel. A direct colour on the span beats the
     inherited one above without needing important of its own, because any
     declaration outranks inheritance. */
  .dfx-id{border-bottom:1px dotted transparent}
  .dfx-tw a.dfx-cell:hover .dfx-id{color:#7fb0e3;border-bottom-color:currentColor}
  .dfx-seg{font-size:9px;opacity:.6;margin-top:1px}
  /* Backtest window results. Colour rides on a filled chip rather than on text,
     because coloured glyphs on a chart background were unreadable. */
  .dfx-btrow{display:flex;flex-wrap:wrap;gap:8px;margin:2px 0 8px}
  .dfx-bt{display:inline-flex;align-items:center;gap:8px;font-size:10.5px;
    padding:4px 9px;border-radius:6px;border:1px solid rgba(128,128,128,.28);
    background:rgba(128,128,128,.07)}
  .dfx-bt b{font-weight:600;letter-spacing:.02em}
  .dfx-bt-n{opacity:.72}
  .dfx-bt-p{font-weight:600;padding:1px 6px;border-radius:9px;border:1px solid}

  /* Plausible-requirement chip, under the recommended quantity. Tinted with the
     same accent as the headline figure so the two read as one statement: the
     number, and how far it could reasonably move. */
  .dfx-band{display:inline-flex;align-items:baseline;gap:7px;margin:7px 0 3px;
    padding:3px 9px;border-radius:5px;background:rgba(163,166,238,.11);
    border:1px solid rgba(163,166,238,.30)}
  .dfx-band-l{font-size:8.5px;letter-spacing:.07em;text-transform:uppercase;opacity:.7}
  .dfx-band-v{font-size:13px;font-weight:600;color:#a3a6ee;
    font-variant-numeric:tabular-nums}
  .dfx-band-e{font-size:9.5px;opacity:.62}
  .dfx-band-n{font-size:10px;opacity:.6;line-height:1.45;margin:0 0 6px}

  /* Caveat callout. Amber rather than red: this is a known limitation of the
     forecast on a known pattern, not an error or a failure. */
  .dfx-warn{font-size:11.5px;line-height:1.5;padding:9px 12px;border-radius:6px;
    border:1px solid rgba(224,167,80,.45);background:rgba(224,167,80,.09);
    margin:2px 0 10px}
  .dfx-warn b{color:#e0a750}

  .dfx-badge{display:inline-block;font-size:9.5px;padding:1px 6px;border-radius:9px;border:1px solid}
  .b-pre{color:#c084d9;border-color:#7e5490;background:rgba(192,132,217,.11)}
  .b-nos{color:#e0736e;border-color:#8f4b48;background:rgba(224,115,110,.11)}
  .b-best{color:#e0a750;border-color:#8f6c34;background:rgba(224,167,80,.11)}
  .b-rou{color:#9a9a9a;border-color:#5a5a5a;background:rgba(154,154,154,.08)}
  .dfx-qty{font-weight:600;font-size:13px}
  .dfx-urg{font-weight:600;font-size:11.5px}
  .u-now{color:#e0736e}.u-soon{color:#e0a750}
  .dfx-udate{font-size:9px;opacity:.6;margin-top:1px}
  .dfx-rel{display:inline-flex;align-items:center;gap:4px;justify-content:flex-end;font-size:11px}
  .dfx-dot{width:6px;height:6px;border-radius:50%;flex:0 0 auto}
  .d-good{background:#5aa86f}.d-fair{background:#d9a441}.d-poor{background:#e0736e}
  .d-none{background:transparent;border:1px solid currentColor;opacity:.5}
  .dfx-relnone{opacity:.6;font-style:italic;font-size:10px}
  .dfx-dash{opacity:.35}
  .dfx-foot{margin-top:8px;font-size:10.5px;opacity:.7;display:flex;gap:11px;
    flex-wrap:wrap;align-items:center}
  .dfx-key{display:inline-flex;align-items:center;gap:4px}
</style>
"""


# Screens from PLAN.md Section 3.1, in navigation order. Unbuilt ones are listed
# so the intended shape of the application is visible while it is being built.
SCREENS = [
    ("Action List", True),
    ("SKU Detail", True),
    ("Forecast Validation", False),
    ("Data Quality", False),
]


def sidebar_nav(current: str) -> None:
    """Application identity, plus the screens still to come.

    Streamlit renders its own links for the pages that exist, at the top of the
    sidebar. This adds the app name and lists the unbuilt screens beneath, so the
    intended shape of the application stays visible while it is being built.
    """
    with st.sidebar:
        st.markdown(f"### {APP_TITLE}")
        st.caption("Inventory & purchasing")
        pending = [name for name, built in SCREENS if not built]
        if pending:
            st.divider()
            st.markdown(
                "<span style='font-size:11px;opacity:.45'>Not built yet: "
                + ", ".join(pending) + "</span>",
                unsafe_allow_html=True,
            )


# Service level to the z multiplier on forecast error. Higher service means more
# buffer: at z=1.0 roughly one order cycle in six runs short, at z=1.65 about one
# in twenty, assuming errors are roughly symmetric.
SERVICE_LEVELS = {
    "84% (z=1.0)": 1.00,
    "90% (z=1.28)": 1.28,
    "95% (z=1.65)": 1.65,
    "98% (z=2.05)": 2.05,
}


def sidebar_params() -> dict:
    """Planning assumptions, shared by every page.

    The widgets carry keys, so Streamlit persists them across page navigation and
    both screens compute from the same assumptions. Changing any of these
    recomputes the planning table, so the order quantities, stockout dates and
    headline counts all move together.
    """
    from . import calc  # local import: calc imports data/reliability, not ui

    defaults = calc.DEFAULT_PARAMS
    with st.sidebar:
        st.divider()
        st.markdown("**Planning assumptions**")
        lead = st.slider(
            "Lead time (weeks)", 1, 20, int(defaults["lead_time_weeks"]),
            key="p_lead_time_weeks",
            help="Supplier plus transit. Drives how much demand an order must cover.",
        )
        review = st.slider(
            "Reorder cycle (weeks)", 1, 8, int(defaults["review_period_weeks"]),
            key="p_review_period_weeks",
            help="How often orders are placed. An order covers the lead time plus "
                 "this, so stock lasts until the next one arrives.",
        )
        service = st.selectbox(
            "Service level", list(SERVICE_LEVELS), index=0, key="p_service_level",
            help="How much safety stock to hold against forecast error. Higher "
                 "service means more buffer and more stock.",
        )
        st.caption(f"Orders cover **{lead + review} weeks** ({lead}w lead + {review}w cycle)")

    params = {
        **defaults,
        "lead_time_weeks": int(lead),
        "review_period_weeks": int(review),
        "service_z": float(SERVICE_LEVELS[service]),
    }
    st.session_state["params"] = params
    return params


def back_link(target: str, label: str) -> None:
    """Inline link to another page, degrading quietly if it cannot resolve.

    `st.page_link` reads the app's page registry, which is only populated when
    Streamlit is running the whole app. Executing a single page directly (as the
    AppTest harness does) leaves the registry without the entry, and the call
    raises. The link is a convenience: the sidebar already offers the same
    navigation, so a failure here should not take the page down with it.
    """
    try:
        st.page_link(target, label=label)
    except Exception:  # pragma: no cover - registry unavailable outside a full app
        st.caption(label)


def setup_page(subtitle: str | None = None) -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📦", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)
    if subtitle:
        sidebar_nav(subtitle)
    st.title(subtitle or APP_TITLE)


def forecast_caption() -> None:
    """Vintage of the forecast currently loaded."""
    fc = D.load_forecasts()
    snap = D.forecast_snapshot_date()
    if snap is None or fc.empty:
        st.caption("No forecast loaded.")
        return
    ver = fc["model_version"].iloc[0] if "model_version" in fc.columns else "unknown"
    st.caption(
        f"Forecast trained {snap.date().isoformat()} · model {ver} · "
        f"horizon to {fc['ds'].max().date().isoformat()}"
    )


def sample_data_warning() -> None:
    """State plainly which fields on screen are not real, when that is the case."""
    if D.inventory_is_sample():
        st.markdown(
            '<div class="dfx-warn">⚠ Inventory, preorder backlog, confirmed inbound '
            "and ETA are <strong>sample values</strong>. Forecast, sales history and "
            "reliability are real.</div>",
            unsafe_allow_html=True,
        )


def chip_row(chips: list[tuple[str, object, bool]]) -> None:
    """Summary counts. `chips` is (label, value, highlighted)."""
    parts = ['<div class="dfx-chips">']
    for label, value, on in chips:
        v = f"{value:,}" if isinstance(value, (int, float)) else str(value)
        parts.append(
            f'<div class="dfx-chip{" on" if on else ""}">'
            f'<div class="dfx-chip-n">{html.escape(v)}</div>'
            f'<div class="dfx-chip-l">{html.escape(label.upper())}</div></div>'
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _num(v, dash_on_zero: bool = False) -> str:
    if v is None or pd.isna(v):
        return '<span class="dfx-dash">—</span>'
    v = float(v)
    if dash_on_zero and v == 0:
        return '<span class="dfx-dash">—</span>'
    return f"{v:,.0f}"


def _urgency(days, date_str) -> str:
    """Days-to-stockout as the headline, calendar date beneath it."""
    if days is None or pd.isna(days) or days == float("inf"):
        return '<span class="dfx-dash">—</span>'
    d = float(days)
    cls = "u-now" if d <= 2 else ("u-soon" if d <= 14 else "")
    txt = "now" if d < 1 else (f"{d:.0f} day" if d < 2 else f"{d:.0f} days")
    date_html = ""
    if isinstance(date_str, str) and date_str:
        try:
            date_html = (
                f'<div class="dfx-udate">'
                f"{pd.Timestamp(date_str).strftime('%-d %b')}</div>"
            )
        except Exception:
            date_html = ""
    return f'<div class="dfx-urg {cls}">{txt}</div>{date_html}'


def _reliability(wape, tier) -> str:
    if tier == "none":
        return ('<span class="dfx-rel"><span class="dfx-dot d-none"></span>'
                '<span class="dfx-relnone">no history</span></span>')
    return (f'<span class="dfx-rel"><span class="dfx-dot d-{tier}"></span>'
            f"{R.display(wape)}</span>")


def action_table(view: pd.DataFrame, flags: dict | None = None) -> None:
    """The banded, scrollable action table. Expects reliability columns attached."""
    head = (
        '<div class="dfx-tw"><table>'
        '<thead class="dfx-grp"><tr>'
        '<th class="g1 dfx-stick">SKU</th>'
        '<th>PRIORITY</th>'
        + _band_th("pos", "POSITION", colspan=3, group=True)
        + _band_th("dem", "DEMAND", colspan=2, group=True)
        + _band_th("act", "ACTION", colspan=3, group=True)
        + '</tr></thead>'
        '<thead class="dfx-h2"><tr>'
        '<th class="l dfx-stick">SKU</th><th class="l">Priority</th>'
        # "Available", not "On hand": the figure is SUM(available) from the
        # warehouse table, which is physical stock less units already allocated to
        # unshipped orders. Calling it on-hand overstates what it is by whatever
        # is currently allocated.
        + _band_th("pos", "Available")
        + _band_th("pos", "Preord.", sep=False)
        + _band_th("pos", "Inbound", sep=False)
        + _band_th("dem", "30d")
        + _band_th("dem", "13w fcst", sep=False)
        + _band_th("act", "Stocks out")
        + _band_th("act", "Order", sep=False)
        + _band_th("act", "Reliability", sep=False)
        + '</tr></thead><tbody>'
    )
    rows = []
    for _, r in view.iterrows():
        label = r.get("priority_label", "Routine")
        badge = (
            f'<span class="dfx-badge {PRIORITY_CLASS.get(label, "b-rou")}">'
            f'{PRIORITY_GLYPH.get(label, "○")} {html.escape(str(label))}</span>'
        )
        name = r.get("product_name")
        parts = [str(r.get("product_category", "")), str(r.get("history_group", ""))]
        if isinstance(name, str) and name:
            parts.append(html.escape(name))
        sub = " · ".join([x for x in parts if x])
        # A caveat is worth most next to the row it concerns, so flagged SKUs
        # carry a marker here; the full wording is in the title attribute and on
        # the SKU detail page.
        sku_flags = (flags or {}).get(str(r["unique_id"]), [])
        flag_mark = (
            f'<span title="{html.escape("; ".join(sku_flags))}" '
            f'style="color:#e0736e;margin-left:5px">⚑</span>' if sku_flags else ""
        )
        uid = str(r["unique_id"])
        # Relative href on purpose. From "/" it resolves to "/SKU_Detail" and from
        # "/SKU_Detail" it replaces the last segment, so it is also correct under a
        # server base path, which an absolute "/SKU_Detail" would break.
        # target="_self" keeps the navigation in this tab. The anchor wraps the
        # whole cell, so the flag marker sits inside it; that is harmless, because
        # the innermost title wins on hover and the flag keeps its own wording.
        rows.append(
            "<tr>"
            f'<td class="l dfx-stick">'
            f'<a class="dfx-cell" href="{SKU_DETAIL_SLUG}?sku={quote(uid)}" '
            f'target="_self" title="Open SKU detail">'
            f'<div class="dfx-sku"><span class="dfx-id">{html.escape(uid)}</span>'
            f'{flag_mark}</div>'
            f'<div class="dfx-seg">{sub}</div></a></td>'
            f'<td class="l">{badge}</td>'
            f'<td class="dfx-sep">{_num(r.get("available_inventory"))}</td>'
            f'<td>{_num(r.get("preorder_backlog"), dash_on_zero=True)}</td>'
            f'<td>{_num(r.get("confirmed_inbound"), dash_on_zero=True)}</td>'
            f'<td class="dfx-sep">{_num(r.get("recent_units"))}</td>'
            f'<td>{_num(r.get("forecast_total"))}</td>'
            f'<td class="dfx-sep">{_urgency(r.get("days_to_stockout"), r.get("estimated_stockout_date"))}</td>'
            f'<td class="dfx-qty">{_num(r.get("recommended_order_qty"))}</td>'
            f'<td>{_reliability(r.get("wape"), r.get("tier", "none"))}</td>'
            "</tr>"
        )
    st.markdown(head + "".join(rows) + "</tbody></table></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SKU Detail building blocks.
# ---------------------------------------------------------------------------
TIER_COLOUR = {"good": "#5aa86f", "fair": "#d9a441", "poor": "#e0736e",
               "none": "rgba(154,154,154,.9)"}


def sku_header(row: pd.Series, fc_row: pd.Series | None, trained: str) -> None:
    """SKU identity: the code, then the badges that qualify how to read it."""
    prio = str(row.get("priority_label", "Routine"))
    bits = [
        f'<span class="dfx-badge {PRIORITY_CLASS.get(prio, "b-rou")}">'
        f'{PRIORITY_GLYPH.get(prio, "○")} {html.escape(prio)}</span>'
    ]
    bits.append(f'<span class="dfx-badge b-rou">{html.escape(str(row.get("history_group","")))}'
                " history</span>")

    # Priority is derived from inventory quantities. While those are simulated,
    # say so next to the badge rather than relying only on the banner above: the
    # badge reads as a classification of the product, so the caveat has to travel
    # with it.
    if D.inventory_is_sample():
        bits.append('<span class="dfx-badge b-rou" style="opacity:.6">from sample stock</span>')

    tail = []
    cat = str(row.get("product_category") or "").strip()
    if cat:
        tail.append(html.escape(cat))
    if fc_row is not None:
        served = str(fc_row.get("served_by", ""))
        which = "long model" if served == "long" else "shared model"
        tail.append(f"forecast by {html.escape(str(fc_row.get('model_version','')))} ({which})")
    tail.append(f"trained {trained}")

    name = str(row.get("product_name") or "").strip()
    name_html = f'<div style="font-size:13px;opacity:.8">{html.escape(name)}</div>' if name else ""
    st.markdown(
        f'<div class="dfx"><div style="font-family:ui-monospace,monospace;font-size:17px;'
        f'font-weight:600">{html.escape(str(row["unique_id"]))}</div>{name_html}'
        f'<div style="margin-top:5px">{"".join(bits)}'
        f'<span style="font-size:11px;opacity:.65"> {html.escape(" · ".join(tail))}</span>'
        "</div></div>",
        unsafe_allow_html=True,
    )


def _order_band_html(total: int, band: tuple[int, int] | None,
                     error: float | None) -> str:
    """The plausible requirement, stated under the recommended quantity.

    Turns the reliability figure from advice into arithmetic: rather than
    "±46%, treat it as a midpoint", it says what the requirement could actually
    be. Where the recommendation sits above the band, that is a deliberate
    consequence of the chosen service level and is labelled as such rather than
    left looking like an inconsistency.
    """
    if band is None:
        return '<div style="margin-bottom:9px"></div>'
    low, high = band
    err_txt = (f'<span class="dfx-band-e">±{error:.0%} error</span>'
               if error is not None and pd.notna(error) else "")
    note = ""
    if total > high:
        note = ('<div class="dfx-band-n">Recommended above this band to hold the '
                'chosen service level.</div>')
    # A chip rather than a third line of small grey text. It previously sat
    # between two captions of the same size and opacity and was read straight
    # past. It qualifies the headline figure, so it is placed directly beneath it
    # and given enough contrast to register as a second number rather than as
    # more caption.
    return (
        '<div class="dfx-band">'
        '<span class="dfx-band-l">plausible</span>'
        f'<span class="dfx-band-v">{low:,}–{high:,}</span>'
        f"{err_txt}</div>{note}"
    )


def order_card(breakdown: pd.DataFrame, total: int, lead_weeks: int,
               review_weeks: int, band: tuple[int, int] | None = None,
               error: float | None = None) -> None:
    """The recommended order shown as arithmetic, so it can be checked by hand."""
    # Rendered as arithmetic rather than as a list of figures: a leading operator
    # column and the running signs make the table itself the formula, so no
    # separate explanation of how the number is reached is needed.
    rows = []
    first_term = True
    for _, r in breakdown.iterrows():
        comp, units = str(r["Component"]), float(r["Units"])
        sign = r.get("Sign")
        # Named is_total, not total: `total` is this function's parameter (the
        # headline quantity), and reassigning it here silently replaced it with a
        # boolean, so the headline rendered as 1 for every SKU.
        is_total = sign == 0
        aside = sign is None or pd.isna(sign)

        if is_total:
            op, shown = "=", units
        elif aside:
            op, shown = "", units
        else:
            op = "" if first_term else ("−" if sign < 0 else "+")
            shown = abs(units)
            first_term = False

        cell = "padding:6px 10px;font-size:11.5px"
        if is_total:
            cell += ";border-top:1px solid rgba(128,128,128,.45);font-weight:600"
        # The operator and its figure share a colour, so a row reads as one
        # movement: green adds to what must be bought, red takes away from it.
        # The total is left in the default colour, being neither.
        if is_total or aside:
            term_colour = ""
        else:
            term_colour = f";color:{ADD_COLOUR if sign > 0 else SUB_COLOUR}"
        row_style = ";opacity:.55;font-style:italic" if aside else ""

        rows.append(
            f'<tr>'
            f'<td style="{cell};padding-left:2px;padding-right:0;width:16px;'
            f'text-align:center{term_colour or ";opacity:.7"}{row_style}">{op}</td>'
            f'<td style="{cell};padding-left:6px{row_style}">{html.escape(comp)}</td>'
            f'<td style="{cell};padding-right:10px;text-align:center'
            f'{term_colour}{row_style}">{shown:,.0f}</td>'
            f"</tr>"
        )
    # Assembled in named pieces rather than one long chain of adjacent literals:
    # mixing implicit concatenation with a `+` call silently breaks the join.
    head = (
        '<div class="dfx" style="border:1px solid rgba(128,128,128,.35);border-radius:6px;'
        'padding:11px 13px;height:100%">'
        '<div style="font-size:9.5px;letter-spacing:.08em;text-transform:uppercase;'
        'opacity:.6;margin-bottom:8px">Recommended order</div>'
        f'<div style="font-size:30px;font-weight:700;line-height:1;color:#a3a6ee">{total:,}</div>'
    )
    # The band goes directly under the headline, before the caption explaining
    # what the quantity covers. It qualifies the number, so it belongs adjacent
    # to it; behind the caption it became the middle of three grey lines.
    caption = (
        f'<div style="font-size:10.5px;opacity:.6;margin:3px 0 2px">units, covering a '
        f"{lead_weeks}-week lead time plus a {review_weeks}-week reorder cycle</div>"
    )
    table = (
        '<table style="width:100%;border-collapse:collapse;'
        f'font-variant-numeric:tabular-nums">{"".join(rows)}</table>'
    )
    foot = (
        '<div style="font-size:10px;opacity:.5;margin-top:8px;line-height:1.5">'
        "what to buy = what is owed + what will sell + a cushion for forecast error, "
        "less what is already here or on its way</div></div>"
    )
    st.markdown(head + _order_band_html(total, band, error) + caption + table + foot,
                unsafe_allow_html=True)


#: Length of each backtest window, in weeks. Mirrors config.TEST_WEEKS, which the
#: ML harness uses to build its evaluation splits. If that changes, the shaded
#: regions here would be drawn the wrong width, so it is named rather than typed
#: into the geometry below.
BACKTEST_WEEKS = 10


def runs_high_note(row: pd.Series, horizon_weeks: int) -> None:
    """Callout for a SKU the model forecasts well above its recent selling rate.

    Leads with the comparison for this SKU, which is checkable against the chart
    below, rather than with a claim about a class of SKUs. The backtest evidence
    is added only when the SKU is also in a falling state, because that is the
    pattern the evidence was measured on; quoting it for a SKU that is merely
    forecast high would be borrowing authority the measurement does not give.

    It does not change the recommendation. Routing these SKUs to a 4-week average
    was tested and failed the adoption rule, so substituting silently would
    assert more than the evidence supports.
    """
    if not row.get("forecast_runs_high"):
        return
    wa4 = float(row.get("wa4") or 0.0)
    fc_total = float(row.get("forecast_total") or 0.0)
    fc_wk = float(row.get("forecast_per_week") or 0.0)
    excess = float(row.get("forecast_excess") or 0.0)
    ratio = row.get("forecast_over_recent")
    state = str(row.get("demand_state", "unknown"))

    if wa4 <= 0:
        lead = (f"<b>Nothing has sold in the last 4 weeks</b>, and the model forecasts "
                f"<b>{fc_total:,.0f}</b> units over the next {horizon_weeks}.")
    else:
        lead = (f"<b>The forecast is {float(ratio):.1f}x the recent selling rate.</b> "
                f"The last 4 weeks averaged {wa4:,.1f} units a week; the model forecasts "
                f"{fc_wk:,.1f} a week, <b>{excess:,.0f}</b> units more across "
                f"{horizon_weeks} weeks than the recent rate implies.")

    evidence = ""
    if state in ("falling", "collapsing"):
        evidence = (" Demand is also " + html.escape(state) + ", and on the development "
                    "windows the model over-forecast that pattern by roughly three times, "
                    "with a plain 4-week average beating it.")

    st.markdown(
        f'<div class="dfx dfx-warn">{lead}{evidence} The recommended quantity below '
        'still uses the model: substituting the recent rate was tested and did not meet '
        'the bar to be adopted, so the disagreement is shown rather than resolved.</div>',
        unsafe_allow_html=True,
    )


def backtest_strip(windows: pd.DataFrame) -> None:
    """Per-window results as a row of chips above the backtest chart.

    These were once drawn as text inside the plot and were unreadable there:
    coloured text sits on whatever the chart happens to put behind it, competes
    with the demand line, and collides with its neighbours when windows are close
    together. As HTML the contrast is controlled, the same MISS_SCALE the action
    table uses applies, and the colour lands on a filled chip rather than on thin
    glyphs, which is where colour is legible.
    """
    chips = []
    for _, w in windows.iterrows():
        y_tot, yhat = float(w["y_total"]), float(w["yhat_total"])
        err = (yhat - y_tot) / y_tot if y_tot else None
        colour = miss_colour(err) if err is not None else "#9a9a9a"
        pct = f"{err:+.0%}" if err is not None else "n/a"
        chips.append(
            f'<span class="dfx-bt">'
            f'<b>{html.escape(str(w["window"]))}</b>'
            f'<span class="dfx-bt-n">predicted {yhat:,.0f} · actual {y_tot:,.0f}</span>'
            f'<span class="dfx-bt-p" style="background:{colour}22;color:{colour};'
            f'border-color:{colour}66">{pct}</span></span>'
        )
    st.markdown(f'<div class="dfx dfx-btrow">{"".join(chips)}</div>',
                unsafe_allow_html=True)


def backtest_chart(history: pd.DataFrame, windows: pd.DataFrame,
                   weekly: pd.DataFrame | None = None, height: int = 320):
    """Weekly demand against what the model predicted, per backtest window.

    Answers "how did this SKU's forecast go wrong", which a single error
    percentage cannot. Each window is shaded from its cutoff to ten weeks past
    it, so the training data the model saw sits immediately left of the result it
    produced and a large miss can be read back to the shape that caused it.

    `weekly` carries the per-week predictions from
    outputs/reports/ml_backtest_weekly.csv, whose totals are verified against the
    stored accuracy report before that file is written. When it is present the
    prediction is drawn as a real line inside each window, one segment per window
    so nothing is connected across the gaps between them. When it is absent the
    chart falls back to a flat level per window, which is the predicted total
    spread evenly and is labelled as an average rather than presented as a shape
    the record does not contain.

    Window results are not drawn on the plot; see backtest_strip().
    """
    import altair as alt

    hist = history.rename(columns={"y": "value"})[["ds", "value"]].copy()
    hist["ds"] = pd.to_datetime(hist["ds"])
    has_weekly = weekly is not None and not weekly.empty

    bands, rules, levels = [], [], []
    for _, w in windows.iterrows():
        cut = pd.Timestamp(w["cutoff"])
        start, end = cut, cut + pd.Timedelta(weeks=BACKTEST_WEEKS)
        bands.append({"start": start, "end": end, "window": str(w["window"])})
        rules.append({"cutoff": cut, "window": str(w["window"])})
        if not has_weekly:
            levels.append({"start": start, "end": end, "window": str(w["window"]),
                           "level": float(w["yhat_total"]) / BACKTEST_WEEKS})

    layers = []
    band_df = pd.DataFrame(bands)
    if not band_df.empty:
        layers.append(
            alt.Chart(band_df).mark_rect(opacity=.13, fill="#7fb0e3").encode(
                x="start:T", x2="end:T",
                tooltip=[alt.Tooltip("window:N", title="backtest window")],
            )
        )
        layers.append(
            alt.Chart(pd.DataFrame(rules)).mark_rule(
                strokeDash=[4, 3], color="#9a9a9a", strokeWidth=1).encode(
                x=alt.X("cutoff:T"),
                tooltip=[alt.Tooltip("cutoff:T", title="trained through"),
                         alt.Tooltip("window:N", title="window")],
            )
        )

    layers.append(
        alt.Chart(hist).mark_line(color=DEMAND_COLOURS["Actual sales"],
                                  strokeWidth=1.8).encode(
            x=alt.X("ds:T", title=None),
            y=alt.Y("value:Q", title="units / week"),
        )
    )

    if has_weekly:
        wk = weekly[["ds", "yhat", "window", "lead"]].copy()
        wk["ds"] = pd.to_datetime(wk["ds"])

        # Bridge. A prediction line that begins at the first forecast week starts
        # in mid-air, detached from the history it was made from. Each window's
        # line is therefore anchored to the actual value at its own cutoff, the
        # last week the model saw, so it leaves the demand curve rather than
        # appearing beside it. The anchor is a real observation, not a prediction,
        # so it is added to the line only; the point markers below use the
        # unbridged rows and every dot on the chart remains a genuine forecast.
        actual_at = hist.set_index("ds")["value"]
        bridge = []
        for win, grp in wk.groupby("window"):
            cut = grp["ds"].min() - pd.Timedelta(weeks=1)
            if cut in actual_at.index:
                bridge.append({"ds": cut, "yhat": float(actual_at.loc[cut]),
                               "window": win, "lead": 0})
        wk_line = (pd.concat([pd.DataFrame(bridge), wk], ignore_index=True)
                   if bridge else wk).sort_values(["window", "ds"])

        layers.append(
            alt.Chart(wk_line).mark_line(
                color=DEMAND_COLOURS["Model forecast"], strokeWidth=1.8,
                strokeDash=DEMAND_DASHES["Model forecast"]).encode(
                x="ds:T", y=alt.Y("yhat:Q"),
                detail="window:N",   # one line per window, never joined across gaps
            )
        )
        # Markers are for reading position, not for hitting with a pointer: the
        # crosshair below owns hovering. Still enlarged from 22, since at that
        # size they read as noise on the line rather than as weekly points.
        layers.append(
            alt.Chart(wk).mark_point(
                color=DEMAND_COLOURS["Model forecast"], size=45, filled=True,
                opacity=.95).encode(
                x="ds:T", y=alt.Y("yhat:Q"), detail="window:N",
            )
        )
    elif levels:
        layers.append(
            alt.Chart(pd.DataFrame(levels)).mark_rule(
                strokeDash=[2, 4], color="#e0a750", strokeWidth=1.4, opacity=.85).encode(
                x="start:T", x2="end:T", y=alt.Y("level:Q"),
                tooltip=[alt.Tooltip("level:Q", title="predicted, average per week",
                                     format=",.1f"),
                         alt.Tooltip("window:N", title="window")],
            )
        )

    # Nearest-week crosshair. A tooltip on the lines themselves only fires within
    # a couple of pixels of a 1.8px stroke, so it effectively never fires. One
    # rule carries the selection and the tooltip together: attaching `nearest` to
    # a mark is what builds the Voronoi hit region that makes hovering anywhere
    # snap to a week, so the mark that owns the selection has to be the mark that
    # owns the tooltip. Splitting them across layers silently reduces the target
    # back to the 1px rule itself.
    #
    # That constrains the tooltip to one field list for every week, because
    # Vega-Lite cannot vary the rows by datum. Weeks outside a backtest window
    # have no prediction, and the pivot leaves those null, which formats as NaN.
    # They are therefore substituted for an em dash in the spec rather than left
    # to format: the row still appears, but it reads as "not applicable here"
    # instead of as a broken number. Dropping the rows entirely would mean giving
    # up the snap, which is the worse trade.
    tidy = hist.rename(columns={"value": "v"}).assign(series="Actual")[["ds", "series", "v"]]
    if has_weekly:
        tidy = pd.concat([
            tidy,
            wk.rename(columns={"yhat": "v"}).assign(series="Predicted")[["ds", "series", "v"]],
            wk.rename(columns={"lead": "v"}).assign(series="Lead")[["ds", "series", "v"]],
        ], ignore_index=True)

    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["ds"],
                                  empty=False, clear="pointerout")
    crosshair = (
        alt.Chart(tidy).transform_pivot("series", value="v", groupby=["ds"])
    )
    tips = [alt.Tooltip("ds:T", title="Week ending", format="%d %b %Y"),
            alt.Tooltip("Actual:Q", title="Actual units", format=",.0f")]
    if has_weekly:
        crosshair = crosshair.transform_calculate(
            pred_txt="isValid(datum.Predicted) ? format(datum.Predicted, ',.1f') : '—'",
            lead_txt="isValid(datum.Lead) ? format(datum.Lead, ',.0f') : '—'",
        )
        tips += [alt.Tooltip("pred_txt:N", title="Predicted"),
                 alt.Tooltip("lead_txt:N", title="Weeks after cutoff")]

    layers.append(
        crosshair.mark_rule(color="#9a9a9a", strokeWidth=1).encode(
            x="ds:T",
            opacity=alt.condition(nearest, alt.value(.45), alt.value(0)),
            tooltip=tips,
        ).add_params(nearest)
    )

    return alt.layer(*layers).properties(height=height).configure_view(strokeOpacity=0)


def reliability_card(wape, tier: str, windows: pd.DataFrame, order_qty: int) -> None:
    """Forecast reliability, with the per-window evidence behind the headline.

    A single error percentage hides the thing a planner most needs: whether the
    misses run one way or both. The window table is what turns "±46%" into a
    decision about how much to trust the recommended quantity.
    """
    colour = TIER_COLOUR.get(tier, "#9a9a9a")
    if tier == "none" or wape is None or pd.isna(wape):
        head = (f'<div style="font-size:19px;font-weight:600;color:{colour}">No history</div>'
                '<div style="font-size:11px;opacity:.7;margin-top:7px;line-height:1.45">'
                "This SKU was too new to appear in any backtest window, so its forecast "
                "has never been scored. That is not a sign the forecast is wrong, only "
                "that it is unverified. Treat it with more caution than a SKU with a "
                "measured track record.</div>")
        body = ""
    else:
        n = len(windows)
        signs = set()
        rows = []
        for _, w in windows.iterrows():
            actual, pred = float(w["y_total"]), float(w["yhat_total"])
            pct = (pred - actual) / actual if actual else 0.0
            signs.add(pct >= 0)
            cls = miss_colour(pct)
            rows.append(
                f'<tr><td style="text-align:left;padding:6px 12px 6px 8px;'
                f'border-top:1px solid rgba(128,128,128,.18)">{html.escape(str(w["window"]))}</td>'
                f'<td style="text-align:right;padding:6px 12px;border-top:1px solid rgba(128,128,128,.18)">'
                f"{pred:,.0f}</td>"
                f'<td style="text-align:right;padding:6px 12px;border-top:1px solid rgba(128,128,128,.18)">'
                f"{actual:,.0f}</td>"
                # A bar beside the figure, so severity is comparable at a glance
                # without reading three percentages. Length is |miss| capped at
                # 100%, which keeps an extreme outlier from flattening the rest.
                f'<td style="text-align:right;padding:6px 8px 6px 12px;'
                f'border-top:1px solid rgba(128,128,128,.18)">'
                f'<span style="display:inline-flex;align-items:center;gap:7px;'
                f'justify-content:flex-end">'
                f'<span style="width:42px;height:5px;border-radius:3px;'
                f'background:rgba(128,128,128,.18);position:relative;display:inline-block">'
                f'<span style="position:absolute;right:0;top:0;height:100%;'
                f'width:{min(abs(pct), 1.0) * 100:.0f}%;background:{cls};'
                f'border-radius:3px;display:block"></span></span>'
                f'<span style="color:{cls}">{pct:+.0%}</span></span></td></tr>'
            )
        both = len(signs) > 1
        direction = (
            f"It has missed in <strong>both</strong> directions, so treat {order_qty:,} "
            "as a midpoint rather than a floor."
            if both else
            ("It has consistently forecast <strong>low</strong>, so the true requirement "
             f"is likely above {order_qty:,}."
             if False in signs else
             "It has consistently forecast <strong>high</strong>, so the true requirement "
             f"is likely below {order_qty:,}.")
        )
        head = (
            f'<div style="display:flex;align-items:baseline;gap:8px">'
            f'<span style="font-size:26px;font-weight:700;line-height:1;color:{colour}">'
            f"±{wape:.0%}</span>"
            f'<span style="font-size:11px;color:{colour}">● {html.escape(tier)}</span></div>'
            f'<div style="font-size:11px;opacity:.7;margin:7px 0 9px;line-height:1.45">'
            f"Off by {wape:.0%} on average across {n} backtest "
            f"window{'s' if n != 1 else ''}. {direction}</div>"
        )
        body = (
            '<table style="width:100%;border-collapse:collapse;font-size:11px;'
            'font-variant-numeric:tabular-nums">'
            '<tr><th style="text-align:left;font-size:9.5px;opacity:.6;font-weight:500;'
            'padding:0 12px 8px 8px;text-transform:uppercase;letter-spacing:.05em">Window</th>'
            '<th style="text-align:right;font-size:9.5px;opacity:.6;font-weight:500;'
            'padding:0 12px 8px;text-transform:uppercase;letter-spacing:.05em">Forecast</th>'
            '<th style="text-align:right;font-size:9.5px;opacity:.6;font-weight:500;'
            'padding:0 12px 8px;text-transform:uppercase;letter-spacing:.05em">Actual</th>'
            '<th style="text-align:right;font-size:9.5px;opacity:.6;font-weight:500;'
            f'padding:0 8px 8px 12px;text-transform:uppercase;letter-spacing:.05em">Miss</th></tr>{"".join(rows)}</table>'
            # Seven colours are not self-explanatory, so the scale is stated. It
            # doubles as the reminder that under and over are not equally costly.
            '<div style="margin-top:9px;font-size:9px;opacity:.6;line-height:1.6">'
            'MISS SCALE &nbsp;'
            + " ".join(
                f'<span style="color:{c};font-weight:600">■</span>'
                f'<span style="opacity:.85">&nbsp;{html.escape(lbl)}</span>&nbsp;&nbsp;'
                for _, c, lbl in MISS_SCALE
            )
            + '<br><span style="opacity:.8">Warm = under-forecast, which risks a '
            'stockout. Cool = over-forecast, which only ties up stock.</span></div>'
        )
    st.markdown(
        '<div class="dfx" style="border:1px solid rgba(128,128,128,.35);border-radius:6px;'
        'padding:11px 13px;height:100%">'
        '<div style="font-size:9.5px;letter-spacing:.08em;text-transform:uppercase;'
        'opacity:.6;margin-bottom:8px">How reliable is this forecast</div>'
        f"{head}{body}</div>",
        unsafe_allow_html=True,
    )


def quality_summary(flags: dict[str, list[str]], n_rows: int) -> None:
    """How many SKUs in view carry a data-quality caveat, and which."""
    if not flags:
        return
    counts: dict[str, int] = {}
    for labels in flags.values():
        for lab in labels:
            counts[lab] = counts.get(lab, 0) + 1
    parts = " · ".join(f"{n:,} {html.escape(lab)}"
                       for lab, n in sorted(counts.items(), key=lambda kv: -kv[1]))
    st.markdown(
        f'<div class="dfx dfx-warn">⚑ {len(flags):,} of {n_rows:,} SKUs here carry a '
        f"data-quality caveat &nbsp;·&nbsp; {parts}</div>",
        unsafe_allow_html=True,
    )


def quality_flags_inline(labels: list[str]) -> None:
    """The caveats for one SKU, shown where its numbers are read."""
    if not labels:
        return
    chips = "".join(
        f'<span class="dfx-badge b-nos" style="margin-right:4px">⚑ {html.escape(l)}</span>'
        for l in labels
    )
    st.markdown(f'<div class="dfx" style="margin:2px 0 10px">{chips}</div>',
                unsafe_allow_html=True)


def stat_strip(items: list[tuple[str, str, bool]]) -> None:
    """Row of small figures. Each item is (label, value, urgent)."""
    cells = "".join(
        f'<div style="border:1px solid rgba(128,128,128,.35);border-radius:6px;'
        f'padding:7px 10px;flex:1 1 108px">'
        f'<div style="font-size:16px;font-weight:600;line-height:1.1'
        f'{";color:#e0736e" if urgent else ""}">{html.escape(str(value))}</div>'
        f'<div style="font-size:9.5px;opacity:.6;margin-top:2px">{html.escape(label.upper())}</div>'
        "</div>"
        for label, value, urgent in items
    )
    st.markdown(
        f'<div class="dfx" style="display:flex;gap:8px;flex-wrap:wrap">{cells}</div>',
        unsafe_allow_html=True,
    )


def reliability_legend(counts: dict[str, int]) -> None:
    keys = "".join(
        f'<span class="dfx-key"><span class="dfx-dot d-{t}"></span>'
        f"{html.escape(R.TIER_LABEL[t])} ({counts.get(t, 0)})</span>"
        for t in R.TIER_ORDER
    )
    st.markdown(
        '<div class="dfx-foot"><span>Reliability = this SKU\'s typical forecast '
        f"error in backtests.</span>{keys}</div>",
        unsafe_allow_html=True,
    )
