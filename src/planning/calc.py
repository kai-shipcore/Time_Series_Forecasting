"""Planning calculations: model group, forecast aggregation, priority logic,
stockout dates, and the recommended-order-quantity breakdown.

All formulas are intentionally simple and transparent for a Phase-1 prototype.
The exact definitions are documented in dashboard/README.md and mirrored in the
SKU Detail breakdown so displayed values can be traced by hand.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd

from src.planning._cache import cache as _cache

from src.planning import data as D, reliability as R

# Default planning parameters (overridable from the UI).
#: A forecast this many times the recent weekly rate is worth flagging, provided
#: the excess is also material. Both are display thresholds; neither changes what
#: is forecast or what is ordered.
RUNS_HIGH_RATIO = 1.5
#: and the excess must be at least this many units across the forecast horizon,
#: so the flag does not fire on SKUs where the ratio is large but the quantity
#: is negligible. At 20 units it selects roughly 22 of 447 SKUs; lowering it to
#: 10 more than doubles that, almost all of them trivial.
RUNS_HIGH_EXCESS_UNITS = 20

#: `active_weeks` value that marks a SKU promoted from intermittent to
#: smooth/short. `src/profile.py` assigns RECENT_WEEKS there rather than a
#: measured count, so the constant is a marker, not a coincidence.
PROMOTED_ACTIVE_WEEKS = 13
#: Pooled WAPE of promoted SKUs across the three development windows, used as
#: their safety-stock fallback when they have no measured error of their own.
#: Measured July 2026 at 0.2397 over 118 scored SKU-windows, against 0.1912 for
#: the rest of the short segment; worse in all three windows, and distinguishable
#: from sampling noise in Dec-Feb. Reproduce with
#: `scripts/promoted_sku_accuracy.py`, which re-profiles as-of each cutoff to
#: identify which SKUs were promoted at the time. Refresh it when the pinned
#: windows move, since it is a measurement and will drift with them.
PROMOTED_ERROR_FALLBACK = 0.24

DEFAULT_PARAMS = {
    # An order must cover demand until the NEXT order can arrive, which is the
    # lead time plus however long it is until the next ordering round. Covering
    # only the lead time leaves a shortfall of one review period every cycle.
    "lead_time_weeks": 8,          # supplier + transit
    "review_period_weeks": 1,      # how often orders are placed
    # Safety stock is z x the SKU's own forecast error over the coverage window.
    # z = 1.0 is roughly an 84% service level on a normal error distribution;
    # 1.65 is roughly 95%.
    "service_z": 1.0,
    "best_seller_top_pct": 0.20,   # top X% of SKUs by recent units = best seller
    "stockout_horizon_days": 30,   # window for "at risk" flags
}


def history_group(bucket: str, history_length: str) -> str:
    """Collapse history length into the project's two segments: short vs long.

    smooth/short = fewer than 50 active weeks; smooth/long = 50+ (medium or full).
    Matches the terminology in docs/ML_FORECAST_DESIGN.md.
    """
    if bucket != "smooth":
        return "n/a"
    return "short" if history_length == "short" else "long"


def _coverage_demand(fc_sku: pd.DataFrame, weeks: int) -> float:
    """Forecast demand over the coverage window (lead time + review period).

    Sums yhat across the first ``weeks`` forecast weeks; if the horizon is
    shorter than that, the remaining weeks are padded with the horizon's weekly
    average.
    """
    if fc_sku.empty:
        return 0.0
    yhat = fc_sku.sort_values("ds")["yhat"].to_numpy()
    weekly_avg = float(yhat.mean())
    if weeks <= len(yhat):
        return float(yhat[:weeks].sum())
    return float(yhat.sum() + weekly_avg * (weeks - len(yhat)))


@_cache(show_spinner=False)
def _weekly_forecasts() -> dict[str, np.ndarray]:
    """The forecast curve per SKU, as an array of weekly units."""
    fc = D.load_forecasts()
    if fc.empty:
        return {}
    fc = fc.sort_values(["unique_id", "ds"])
    return {uid: g["yhat"].to_numpy(dtype=float)
            for uid, g in fc.groupby("unique_id")}


def _days_until_consumed(weekly: np.ndarray, stock: float) -> float:
    """Days until the forecast curve consumes `stock` units.

    Walks the forecast week by week and interpolates within the week the stock
    runs out, so a SKU heading into a seasonal peak runs down faster than a flat
    average would suggest. Past the end of the horizon, demand continues at the
    horizon's average weekly rate.
    """
    if stock <= 0:
        return 0.0
    if weekly.size == 0:
        return float("inf")
    cum = 0.0
    for i, w in enumerate(weekly):
        if w <= 0:
            continue
        if cum + w >= stock:
            return i * 7.0 + 7.0 * (stock - cum) / w
        cum += w
    avg = float(weekly.mean())
    if avg <= 0:
        return float("inf")
    return weekly.size * 7.0 + 7.0 * (stock - cum) / avg


def _forecast_aggregates(coverage_weeks: int) -> pd.DataFrame:
    fc = D.load_forecasts()
    if fc.empty:
        return pd.DataFrame(
            columns=["unique_id", "forecast_total", "coverage_demand"]
        )
    rows = []
    for uid, g in fc.groupby("unique_id"):
        rows.append(
            {
                "unique_id": uid,
                "forecast_total": float(g["yhat"].sum()),
                "coverage_demand": _coverage_demand(g, coverage_weeks),
                # Segmentation as of the forecast run (not the current profile
                # snapshot), so a later reclassification cannot blank a SKU that
                # actually has a forecast.
                "bucket": g["bucket"].iloc[0],
                "history_length": g["history_length"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


@_cache(show_spinner=False)
def build_planning_table(params: dict | None = None) -> pd.DataFrame:
    """Join forecasts, segmentation, recent sales and inventory into one
    per-SKU planning row with priorities and recommended order quantities.

    Covers forecastable SKUs only. Any SKU the current profile snapshot classes
    as intermittent is excluded, including one that was smooth when the forecast
    ran and has since been demoted; ``df.attrs["demoted_since_forecast"]`` counts
    the latter. Intermittent SKUs have no forecast by design, so nothing here
    applies to them: no coverage demand, no order quantity, no stockout date.

    Bringing them into the worklist is a separate piece of work and needs its own
    basis, since every quantity on this screen is derived from a forecast they do
    not have. See docs/BACKLOG.md.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    lead_weeks = int(p["lead_time_weeks"])
    review_weeks = int(p["review_period_weeks"])
    coverage_weeks = lead_weeks + review_weeks
    z = float(p["service_z"])

    fc_agg = _forecast_aggregates(coverage_weeks)
    # history_length comes from the forecast run itself (in fc_agg), because it
    # describes the model that produced the number. The BUCKET is taken from the
    # current profile snapshot instead, and deliberately not from the forecast:
    # `forward_forecast` writes the literal "smooth" on every row, since only
    # smooth SKUs are modelled at all, so the forecast's own bucket column can
    # never disagree with itself and is useless as a check.
    profiles = D.load_profiles()[["unique_id", "mean", "active_weeks", "bucket"]]
    recent = D.recent_sales(weeks=4)
    inv = D.load_inventory()

    df = fc_agg.drop(columns=["bucket"], errors="ignore").merge(
        profiles, on="unique_id", how="left")

    # Drop SKUs the profiler has since demoted to intermittent. A forecast is
    # produced weekly against the segmentation of that moment; by the time it is
    # read, further sales have arrived and some SKUs no longer qualify. Serving
    # their forecast means showing a number the project's own segmentation says
    # should not exist, and the next run will drop them anyway. The window
    # reopens every week, so this is a standing reconciliation rather than a
    # one-off cleanup.
    #
    # The count is kept on the frame rather than printed: silently shrinking a
    # table is how a caller ends up puzzled by totals that will not reconcile.
    n_demoted = int(df["bucket"].eq("intermittent").sum())
    df = df[~df["bucket"].eq("intermittent")].copy()
    df = df.merge(recent, on="unique_id", how="left")
    df = df.merge(inv.drop(columns=["is_sample"], errors="ignore"), on="unique_id", how="left")

    df["recent_units"] = df["recent_units"].fillna(0.0)
    df["avg_daily_sales"] = df["avg_daily_sales"].fillna(0.0)
    for col in ["available_inventory", "preorder_backlog", "confirmed_inbound"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce").fillna(0.0)

    # Draft container coverage. Absence is a real zero, like confirmed inbound
    # and for the same reason: the figure comes from container line items, so a
    # SKU with no matching draft row genuinely has nothing drafted. The column is
    # only ever missing outright on the sample path, and whether that source can
    # be trusted is already answered once for the whole inventory block by
    # `inventory_source`, rather than per column.
    #
    # Kept out of the loop above because those three feed the order formula and
    # this one deliberately does not: a draft is not a commitment, so crediting
    # it would under-order the SKUs someone has already acted on.
    if "draft_inbound" not in df.columns:
        df["draft_inbound"] = 0.0
    df["draft_inbound"] = pd.to_numeric(df["draft_inbound"], errors="coerce").fillna(0.0)
    if "draft_eta" not in df.columns:
        df["draft_eta"] = ""
    df["draft_eta"] = df["draft_eta"].fillna("")

    df["history_group"] = [
        history_group(b, h) for b, h in zip(df["bucket"], df["history_length"])
    ]
    # Category comes from the SKU prefix, so it is real for every SKU and needs
    # no inventory export.
    df["product_category"] = [D.product_category(u) for u in df["unique_id"]]

    # ----- Forecast uncertainty, per SKU -------------------------------------
    # Safety stock is sized from how wrong this SKU's forecast has actually been,
    # not from a flat number of days. A SKU the model predicts to within 13%
    # needs far less cushion than one that misses by 46%, and the flat rule had
    # it backwards, holding the most stock against the best-predicted SKUs.
    rel = R.per_sku()[["unique_id", "wape", "n_windows", "tier"]]
    df = df.merge(rel, on="unique_id", how="left")
    df["tier"] = df["tier"].fillna("none")
    df["n_windows"] = df["n_windows"].fillna(0).astype(int)
    # A SKU with no backtest history is not assumed accurate. It inherits its
    # segment's median error, so "unmeasured" costs a normal amount of cushion
    # rather than none at all.
    # Two different fallbacks, because "unmeasured" is not one population.
    #
    # Most unmeasured SKUs are unmeasured because they were promoted from
    # intermittent to smooth/short, which resets their training start to the
    # trailing 13 weeks and makes them ineligible for every pinned backtest
    # window (see docs/BACKLOG.md item 2). Promoted SKUs that COULD be scored
    # historically came in at 0.24 pooled against 0.19 for the rest of the short
    # segment, consistently worse in all three windows. Handing them the segment
    # median would size their safety stock with a number now known to be too low
    # for them specifically, and they carry a fifth of all recommended units.
    #
    # A promoted SKU is identifiable by active_weeks sitting exactly at the
    # promotion constant: profile.py assigns RECENT_WEEKS rather than a measured
    # count. Anything else unmeasured falls back to its segment median as before.
    seg_median = df.groupby("history_group")["wape"].transform("median")
    overall = df["wape"].median()
    promoted = df["active_weeks"].eq(PROMOTED_ACTIVE_WEEKS)
    fallback = seg_median.where(~promoted, PROMOTED_ERROR_FALLBACK)
    df["error_used"] = df["wape"].fillna(fallback).fillna(overall).fillna(0.0)
    # Kept so the UI can say which SKUs are running on a cohort estimate rather
    # than their own measured error, instead of showing a number with no
    # provenance.
    df["error_is_measured"] = df["wape"].notna()
    df["error_basis"] = np.where(
        df["wape"].notna(), "measured",
        np.where(promoted, "promoted cohort", "segment median"),
    )

    # ----- Where the forecast disagrees with recent sales ---------------------
    # The condition worth warning about is the forecast standing well above the
    # rate the SKU is currently selling at. That is measured directly rather
    # than inferred from the shape of history: an earlier version of this flag
    # used the ramp feature as a proxy and both missed cases and invented them,
    # because ramp describes how history moved, not whether the forecast agrees
    # with where demand now is. The clearest miss sold 0.25 units a week against
    # a 2.7 forecast, 10.9 times over, while its ramp read "rising" because the
    # 12-week average behind it was lower still.
    #
    # Two conditions, because a ratio alone is meaningless at low volume. A SKU
    # selling a quarter of a unit a week can be forecast ten times over and it
    # still amounts to nothing anyone should act on, so the excess must also be
    # material in units over the horizon.
    #
    # This is surfaced, not acted on. The recommended order quantity still comes
    # from the model: routing these SKUs to a 4-week average was tested and
    # failed the adoption rule in design doc Section 1.5, so the dashboard shows
    # the disagreement rather than quietly resolving it.
    ramp = D.sku_ramp()
    df = df.merge(ramp, on="unique_id", how="left")
    horizon = max(int(D.load_forecasts()["ds"].nunique() or 1), 1)
    df["forecast_per_week"] = df["forecast_total"] / horizon
    # Units the forecast adds over what the recent rate, carried flat, implies.
    df["forecast_excess"] = df["forecast_total"] - df["wa4"].fillna(0.0) * horizon
    with np.errstate(divide="ignore", invalid="ignore"):
        df["forecast_over_recent"] = df["forecast_per_week"] / df["wa4"].replace(0, np.nan)
    # A SKU with no sales at all in four weeks has no ratio, and is included on
    # the excess alone: "forecast N, sold nothing recently" is the same warning.
    df["forecast_runs_high"] = (
        ((df["forecast_over_recent"] >= RUNS_HIGH_RATIO) | (df["wa4"].fillna(0.0) <= 0))
        & (df["forecast_excess"] >= RUNS_HIGH_EXCESS_UNITS)
    )
    # Descriptive only: it words the callout and is not what triggers it.
    df["demand_state"] = np.select(
        [df["ramp"].isna(), df["ramp"] < 0.40, df["ramp"] < D.COLLAPSE_RAMP,
         df["ramp"] < 1.10],
        ["unknown", "collapsing", "falling", "steady"],
        default="rising",
    )

    # ----- Inbound that actually lands in time -------------------------------
    # Stock already on the water only helps if it arrives inside the window this
    # order is meant to cover. Inbound with no ETA cannot be confirmed to arrive
    # in time, so it is not credited; that is visible as its own column.
    today = pd.Timestamp(_dt.date.today())
    coverage_end = today + pd.Timedelta(weeks=coverage_weeks)
    eta = pd.to_datetime(df.get("inbound_eta"), errors="coerce")
    df["inbound_eta_days"] = (eta - today).dt.days
    in_window = eta.notna() & (eta <= coverage_end)
    df["inbound_in_window"] = np.where(in_window, df["confirmed_inbound"], 0.0)
    df["inbound_excluded"] = df["confirmed_inbound"] - df["inbound_in_window"]

    # ----- Recommended order quantity ----------------------------------------
    # Every component is rounded to whole units BEFORE the total is taken, so the
    # figures shown in the breakdown add up exactly to the recommended quantity.
    # Rounding only the total lets the displayed lines disagree with it by a unit
    # or two, which destroys the point of showing the arithmetic at all.
    df["safety_stock"] = (z * df["error_used"] * df["coverage_demand"]).round()
    df["coverage_demand"] = df["coverage_demand"].round()
    for col in ("preorder_backlog", "available_inventory",
                "confirmed_inbound", "inbound_in_window", "inbound_excluded"):
        df[col] = df[col].round()

    df["recommended_order_qty"] = (
        df["preorder_backlog"]
        + df["coverage_demand"]
        + df["safety_stock"]
        - df["available_inventory"]
        - df["inbound_in_window"]
    ).clip(lower=0).astype(int)

    # ----- Days of cover, from the forecast, respecting when inbound lands ----
    # Stock is depleted against the SKU's own forecast curve rather than a flat
    # trailing average, so a SKU running into a seasonal peak is shown running
    # out sooner. Inbound only extends cover if it arrives before the shelf
    # empties; arriving later makes it a refill after a stockout, not a
    # prevention of one.
    curves = _weekly_forecasts()
    on_hand = df["available_inventory"].to_numpy(dtype=float)
    inbound = df["confirmed_inbound"].to_numpy(dtype=float)
    eta_days = df["inbound_eta_days"].to_numpy(dtype=float)

    days_cover, before_inbound = [], []
    for uid, oh, inb, eta in zip(df["unique_id"], on_hand, inbound, eta_days):
        curve = curves.get(uid, np.array([], dtype=float))
        t_on_hand = _days_until_consumed(curve, oh)
        in_time = inb > 0 and np.isfinite(eta) and eta <= t_on_hand
        days_cover.append(
            _days_until_consumed(curve, oh + inb) if in_time else t_on_hand
        )
        before_inbound.append(bool(inb > 0 and not in_time))

    df["days_to_stockout"] = np.array(days_cover, dtype=float)
    df["stockout_before_inbound"] = before_inbound
    df["estimated_stockout_date"] = [
        (today + pd.Timedelta(days=float(d))).date().isoformat()
        if np.isfinite(d)
        else ""
        for d in days_cover
    ]

    # Best seller = top X% by recent units among forecasted SKUs.
    top_pct = float(p["best_seller_top_pct"])
    if len(df) and df["recent_units"].max() > 0:
        threshold = df["recent_units"].quantile(1 - top_pct)
        df["best_seller"] = df["recent_units"] >= max(threshold, 1e-9)
    else:
        df["best_seller"] = False

    # Priority: 1 Preorder, 2 No Stock, 3 Best Seller (lowest number wins).
    def _priority(row):
        if row["preorder_backlog"] > 0:
            return 1, "Preorder"
        if row["available_inventory"] <= 0:
            return 2, "No Stock"
        if row["best_seller"]:
            return 3, "Best Seller"
        return 99, "Routine"

    pr = df.apply(_priority, axis=1, result_type="expand")
    df["priority"] = pr[0]
    df["priority_label"] = pr[1]

    horizon = float(p["stockout_horizon_days"])
    df["stockout_soon"] = df["days_to_stockout"] <= horizon
    df["best_seller_at_risk"] = df["best_seller"] & df["stockout_soon"]

    # ----- Stock running out before the replacement lands ---------------------
    # Two columns on this table are computed on assumptions that contradict each
    # other. days_to_stockout ignores inbound entirely; the recommended quantity
    # credits it as though it were already on the shelf. So a SKU can show
    # "out in 12 days" beside "order 0" while a container is 40 days away, and
    # nothing on the row explains the 28 days in between.
    #
    # The order quantity is not wrong. With an 8-week lead time a purchase order
    # placed today lands later than a container already booked, so ordering more
    # cannot close the gap. What is wrong is saying nothing: this is a service
    # failure the data can already see, and the action it calls for is expediting
    # or reallocating, not buying.
    eta = pd.to_datetime(df.get("inbound_eta"), errors="coerce")
    df["days_to_inbound"] = (eta - today).dt.days.astype("float64")
    finite_stockout = np.isfinite(df["days_to_stockout"])
    df["supply_gap_days"] = np.where(
        finite_stockout & df["days_to_inbound"].notna()
        & (df["days_to_inbound"] > df["days_to_stockout"]),
        df["days_to_inbound"] - df["days_to_stockout"],
        np.nan,
    )
    df["has_supply_gap"] = df["supply_gap_days"].notna()
    # Whether ordering could actually help. A gap shorter than the lead time
    # cannot be closed by a new order, which is the difference between "buy
    # more" and "expedite what is already coming".
    lead_days = lead_weeks * 7
    df["gap_closable_by_order"] = (
        df["has_supply_gap"] & (df["days_to_inbound"] > lead_days)
    )

    # kind="mergesort": stable, so SKUs tied on both keys (common -- most
    # Routine/Preorder SKUs currently share recommended_order_qty=0) keep a
    # fixed relative order instead of shuffling under quicksort's tie-breaking.
    out = df.sort_values(
        ["priority", "recommended_order_qty"], ascending=[True, False], kind="mergesort"
    )
    # Set last: pandas drops `attrs` across most merges and assignments, so a
    # value attached earlier in this function would silently arrive as None.
    out.attrs["demoted_since_forecast"] = n_demoted
    return out


def overview_metrics(plan: pd.DataFrame, params: dict | None = None) -> dict:
    """Headline counts for the Inventory Overview page."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    horizon = float(p["stockout_horizon_days"])
    return {
        "forecasted_skus": int(len(plan)),
        "preorder_priority": int((plan["priority"] == 1).sum()),
        "out_of_stock": int((plan["available_inventory"] <= 0).sum()),
        "best_sellers_at_risk": int(plan["best_seller_at_risk"].sum()),
        "total_recommended_order_qty": int(plan["recommended_order_qty"].sum()),
        "stockout_within_horizon": int((plan["days_to_stockout"] <= horizon).sum()),
        # SKUs that run dry before their booked container lands, and the demand
        # that falls into those windows. Reported separately from the stockout
        # count because the action differs: these already have stock coming and
        # cannot be helped by ordering more.
        "supply_gap": int(plan.get("has_supply_gap", pd.Series(dtype=bool)).sum()),
        "supply_gap_backlog": int(
            plan.loc[plan.get("has_supply_gap", pd.Series(dtype=bool)).fillna(False),
                     "preorder_backlog"].sum()
        ) if "has_supply_gap" in plan.columns else 0,
        "horizon_days": int(horizon),
    }


def order_quantity_range(row: pd.Series) -> tuple[int, int]:
    """The plausible requirement, low to high, given this SKU's forecast error.

    Only the demand term is uncertain, so the band is coverage demand flexed by
    the SKU's measured error; the preorder backlog, stock on hand and inbound are
    known quantities and are not flexed. Safety stock is deliberately excluded:
    it is the cushion we chose to add on top of expected demand, so including it
    would count the same uncertainty twice and inflate the upper end.

    The recommended quantity therefore sits inside this band, at the point the
    chosen service level puts it.
    """
    pre = float(row["preorder_backlog"])
    cover = float(row["coverage_demand"])
    avail = float(row["available_inventory"])
    inbound = float(row.get("inbound_in_window", 0.0))
    err = float(row.get("error_used") or 0.0)

    low = max(0.0, pre + cover * (1 - err) - avail - inbound)
    high = max(0.0, pre + cover * (1 + err) - avail - inbound)
    return int(round(low)), int(round(high))


def order_quantity_breakdown(row: pd.Series, params: dict | None = None) -> pd.DataFrame:
    """Signed line-item breakdown of the recommended order quantity for one SKU.

    Labelled so the arithmetic can be checked by hand, including which inbound
    was credited: inbound arriving after the coverage window is deliberately not
    subtracted, because it cannot cover the demand this order is for.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    weeks = int(p["lead_time_weeks"]) + int(p["review_period_weeks"])
    pre = float(row["preorder_backlog"])
    cover = float(row["coverage_demand"])
    safety = float(row["safety_stock"])
    avail = float(row["available_inventory"])
    inbound = float(row.get("inbound_in_window", row.get("confirmed_inbound", 0.0)))
    excluded = float(row.get("inbound_excluded", 0.0))
    # Prefer the stored quantity so this table can never disagree with the one
    # shown on the Action List; fall back to recomputing it if absent.
    roq = float(row["recommended_order_qty"]) if "recommended_order_qty" in row \
        else max(0.0, pre + cover + safety - avail - inbound)

    err = row.get("error_used")
    err_txt = f" (±{float(err):.0%} error)" if err is not None and pd.notna(err) else ""
    # `Sign` states each line's role in the arithmetic: +1 adds, -1 subtracts,
    # 0 is the total, None is an informational aside. It is carried explicitly
    # rather than inferred from the number, because a subtracted line whose value
    # happens to be zero would otherwise read as an addition.
    rows = [
        {"Component": "Preorder demand", "Units": round(pre), "Sign": 1},
        {"Component": f"Demand over {weeks} weeks", "Units": round(cover), "Sign": 1},
        {"Component": f"Safety stock{err_txt}", "Units": round(safety), "Sign": 1},
        {"Component": "Available inventory", "Units": -round(avail), "Sign": -1},
        {"Component": "Confirmed inbound (arrives in time)",
         "Units": -round(inbound), "Sign": -1},
    ]
    if excluded > 0:
        rows.append({"Component": "…inbound arriving too late, not counted",
                     "Units": round(excluded), "Sign": None})
    # An aside for the same reason the line above is one: it is real and it is
    # not in the sum. Drafted units are not subtracted because a draft can be
    # cancelled, so the total below is the requirement if it is. What the
    # requirement becomes if it is not is on the card rather than here, because
    # a second figure with an equals sign in this table would read as a second
    # answer to the same question instead of the answer to a different one.
    draft = float(row.get("draft_inbound", 0.0) or 0.0)
    if draft > 0:
        rows.append({"Component": "…drafted, not committed, not subtracted",
                     "Units": round(draft), "Sign": None})
    rows.append({"Component": "Recommended order quantity", "Units": round(roq), "Sign": 0})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# SKUs the model does not forecast.
#
# 87% of the SKU count and about a fifth of recent unit volume is intermittent:
# demand too sporadic for a weekly forecast, so segmentation excludes it by
# design. Excluding it from the planning screen as well leaves a fifth of the
# business invisible, which is why this exists.
#
# Nothing here is forecast-derived, and the columns say so. There is no coverage
# demand, no safety stock, no recommended order quantity and no reliability
# tier, because none of those can be computed without a forecast. What can be
# stated honestly is what a SKU has recently sold, what is in stock, and how long
# that stock lasts at the recent rate. The reorder flag is arithmetic on those,
# not a recommendation: it says stock runs out before a replacement could arrive,
# which is a fact about timing rather than a quantity to buy.
#
# Keeping this in a separate function with different column names, rather than
# padding the forecast table with blanks, is the point. A cover figure derived
# from a 13-week average and one derived from a scored forecast are not the same
# kind of number, and putting them under one heading would invite reading them
# as though they were.
# ─────────────────────────────────────────────────────────────────────────────

#: Weeks of recent history the demand rate is averaged over. Matches the window
#: profiling itself uses to decide whether a SKU is intermittent, so the rate and
#: the classification describe the same period.
NOT_FORECAST_WEEKS = 13


@_cache(show_spinner=False)
def build_not_forecast_table(params: dict | None = None) -> pd.DataFrame:
    """One row per SKU the model does not forecast.

    Returns unique_id, product_name, product_category, recent demand over
    NOT_FORECAST_WEEKS weeks and the weekly and daily rates implied by it, the
    stock position, days of cover at that rate, the week of the last sale, and
    a reorder flag.

    ``days_of_cover`` is NaN where nothing has sold recently: dividing by a zero
    rate would give infinity, which reads on a screen as "never runs out" when
    the truth is that the question does not apply.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    lead_days = int(p["lead_time_weeks"]) * 7

    profiles = D.load_profiles()
    if profiles.empty:
        return pd.DataFrame()
    # Defined by absence from the PLANNING TABLE, not from the forecast file.
    # Those differ by the SKUs demoted to intermittent since the run: they are in
    # the forecast, so keying on it would exclude them here, while
    # build_planning_table drops them there. Fifteen SKUs fell through that gap
    # on the first attempt. Keying on what the other section actually shows makes
    # the two a partition by construction rather than by coincidence.
    served = set(build_planning_table(p)["unique_id"])
    df = profiles[~profiles["unique_id"].isin(served)].copy()
    if df.empty:
        return pd.DataFrame()

    sales = D.load_sales()
    if not sales.empty:
        cutoff = sales["ds"].max() - pd.Timedelta(weeks=NOT_FORECAST_WEEKS)
        recent = sales[sales["ds"] > cutoff]
        agg = recent.groupby("unique_id")["y"].sum().rename("recent_units")
        last = (sales[sales["y"] > 0].groupby("unique_id")["ds"].max()
                .rename("last_sale_week"))
        df = df.merge(agg, on="unique_id", how="left").merge(last, on="unique_id", how="left")
    else:
        df["recent_units"] = np.nan
        df["last_sale_week"] = pd.NaT

    df["recent_units"] = df["recent_units"].fillna(0.0)
    df["weekly_rate"] = df["recent_units"] / NOT_FORECAST_WEEKS
    df["daily_rate"] = df["recent_units"] / (NOT_FORECAST_WEEKS * 7.0)

    inv = D.load_inventory().drop(columns=["is_sample", "source"], errors="ignore")
    df = df.merge(inv, on="unique_id", how="left")

    # Left blank, not zeroed. A SKU absent from the inventory source has no
    # record, which is a different statement from a record showing none, and the
    # screen should not turn the first into the second.
    for col in ("available_inventory", "preorder_backlog", "confirmed_inbound"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    with np.errstate(divide="ignore", invalid="ignore"):
        df["days_of_cover"] = np.where(
            df["daily_rate"] > 0,
            df["available_inventory"] / df["daily_rate"].replace(0, np.nan),
            np.nan,
        )

    # Stock runs out before a replacement could land. Stated as timing, not as a
    # quantity: how much to buy would need a demand model this SKU does not have.
    df["reorder_signal"] = (
        df["days_of_cover"].notna()
        & (df["days_of_cover"] < lead_days)
        & (df["recent_units"] > 0)
    )
    df["product_category"] = [D.product_category(u) for u in df["unique_id"]]
    df["last_sale_week"] = pd.to_datetime(df["last_sale_week"], errors="coerce")

    cols = ["unique_id", "product_name", "product_category", "bucket",
            "recent_units", "weekly_rate", "daily_rate", "last_sale_week",
            "available_inventory", "preorder_backlog", "confirmed_inbound",
            "inbound_eta", "days_of_cover", "reorder_signal", "active_weeks",
            "zero_pct"]
    out = df[[c for c in cols if c in df.columns]].copy()
    # Sorted by what needs attention: flagged first, then by how little cover is
    # left, then by recent volume so the larger SKUs lead among equals.
    return out.sort_values(
        ["reorder_signal", "days_of_cover", "recent_units"],
        ascending=[False, True, False], kind="mergesort",
    ).reset_index(drop=True)


def not_forecast_metrics(table: pd.DataFrame, params: dict | None = None) -> dict:
    """Headline counts for the non-forecast section."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    if table.empty:
        return {"skus": 0, "selling": 0, "dormant": 0, "reorder_signal": 0,
                "out_of_stock": 0, "recent_units": 0,
                "lead_time_days": int(p["lead_time_weeks"]) * 7}
    return {
        "skus": int(len(table)),
        "selling": int((table["recent_units"] > 0).sum()),
        "dormant": int((table["recent_units"] <= 0).sum()),
        "reorder_signal": int(table["reorder_signal"].sum()),
        "out_of_stock": int((table["available_inventory"] <= 0).sum()),
        "recent_units": int(table["recent_units"].sum()),
        "lead_time_days": int(p["lead_time_weeks"]) * 7,
    }
