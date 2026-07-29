#!/usr/bin/env bash
# Regenerate Machine_Learning_Demand_Forecast_Proposal.docx from the Markdown master.
#
# Edit Machine_Learning_Demand_Forecast_Proposal.md, then run this script.
#
# Steps:
#   1. pandoc renders the Markdown to docx, styled by custom-reference.docx.
#      --columns=20 forces explicit table column widths; without it, narrow
#      tables collapse to a single column in LibreOffice.
#   2. finalize_docx.py adds solid table gridlines and a shaded header row,
#      keeps each table on one page, and tightens bullet spacing.
#      (needs python-docx: pip install python-docx)
#
# Charts are read from outputs/reports/. Regenerate them first if the data changed:
#   .venv/bin/python scripts/plot_management_forecast_charts.py
#   .venv/bin/python scripts/plot_demand_breakdown_donut.py
set -euo pipefail
cd "$(dirname "$0")"
pandoc Machine_Learning_Demand_Forecast_Proposal.md \
  --columns=20 \
  --reference-doc=custom-reference.docx \
  -o Machine_Learning_Demand_Forecast_Proposal.docx
python3 finalize_docx.py Machine_Learning_Demand_Forecast_Proposal.docx
echo "Wrote Machine_Learning_Demand_Forecast_Proposal.docx"
