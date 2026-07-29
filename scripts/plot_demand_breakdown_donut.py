#!/usr/bin/env python3
"""Demand-breakdown donut for the management summary.

Renders the "Share of demand, last 90 days" donut used in
Demand_Forecasting_Project_Summary.docx. It shows the same figures as the
"Demand breakdown by SKU group" table in that document: the share of total unit
demand carried by each SKU group over the trailing 90 days, with the two
regular-selling groups (established + newer) making up roughly 80% of demand.

The values below mirror the table in the document. Update them together with the
table when the data snapshot is refreshed. They are kept as plain constants so
the chart needs no database access.

Run with the repo venv:
    .venv/bin/python scripts/plot_demand_breakdown_donut.py [output.png]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── data: mirrors the demand-breakdown table (last 90 days) ──────────────────
#   name, SKU count, demand (units), colour
GROUPS = [
    ("Long history",  81,    21_745, "#1F4E79"),  # smooth, long history
    ("Short history", 366,   34_816, "#2E75B6"),  # smooth, short history
    ("Intermittent",  3_002, 14_452, "#D9D9D9"),  # sporadic, not forecast
]
TITLE = "Share of demand, last 90 days"
CENTER = "80%\nregular-\nselling"   # established + newer share of demand


def main(out: Path) -> None:
    names   = [g[0] for g in GROUPS]
    skus    = [g[1] for g in GROUPS]
    demand  = [g[2] for g in GROUPS]
    colours = [g[3] for g in GROUPS]
    total   = sum(demand)
    shares  = [d / total for d in demand]

    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
    ax.pie(
        demand, colors=colours, startangle=90, counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
    )
    ax.text(0, 0, CENTER, ha="center", va="center",
            fontsize=12, color="#374151", linespacing=1.3)
    ax.set_aspect("equal")
    ax.set_title(TITLE, fontsize=14, fontweight="bold", color="#1F2937", pad=12)

    handles = [
        Patch(facecolor=c, edgecolor="none",
              label=f"{n}\n{s:,} SKUs\n{round(sh * 100)}% of demand")
        for n, c, s, sh in zip(names, colours, skus, shares)
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, handlelength=1.1, handleheight=1.1,
              labelspacing=1.3, fontsize=10, borderaxespad=0)

    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    default = (Path(__file__).resolve().parent.parent
               / "outputs" / "reports" / "demand_breakdown_donut.png")
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    out.parent.mkdir(parents=True, exist_ok=True)
    main(out)
