# CLAUDE.md

Guidance for AI assistants working in this repository.

## Project state

The active work is the LightGBM demand-forecasting track in `src/ml/` and
`scripts/ml_*.py`. Before doing anything, read:

1. `docs/ML_FORECAST_DESIGN.md`: goals, metrics, evaluation protocol, every design
   decision with its evidence (Section 4), the model version log (Section 6). This is the
   source of truth for what has been decided and why. Do not re-litigate settled
   decisions without new evidence; do not contradict them silently.
2. `docs/CODEBASE_GUIDE.md`: what every file does and how data flows from raw sales to a
   scored forecast.

## Working rules

- One hypothesis at a time. Every model change is evaluated on the development windows
  through `src/ml/evaluate.py` and judged by the design doc's Section 1.5 decision rule.
  Results are recorded in the Decision Log (Section 4) and Version Log (Section 6),
  including rejections.
- Pass criteria for a new version are stated in the version log BEFORE the experiment
  runs.
- Raw per-segment results are shown to the user before any summary or interpretation.
- The final test window is quarantined (design doc Section 2.2). Never evaluate against
  it during development.
- Writing style for documents: plain professional prose, no decorative phrasing, no
  dashes connecting statements (dashes in date ranges are fine).

## Work log (required habit)

`docs/WORKLOG.md` is the running record of completed work, used to generate day-by-day
progress summaries for the user's manager.

- Whenever a notable piece of work is completed (a document, a harness change, a model
  version evaluated, a bug fixed), append one dated line describing it in plain terms.
- When the user asks for a progress summary, read all entries after the latest
  "SUMMARY PRODUCED" marker, write a numbered list with one plain-language sentence per
  item (audience: a non-technical manager), then append a new marker line with the date.

## Environment notes

- Run scripts with the repo's venv (`.venv/bin/python`) on the user's machine; plain
  `python` does not exist there.
- The database is not reachable from the assistant's sandbox. DB data is accessed through
  export scripts (see `scripts/export_forecast_history.py`) that snapshot tables to
  `data/processed/`; the user runs them locally.
- `load_dotenv` calls should use `override=True`; the user's shell has stale DB_*
  exports that otherwise shadow `.env` (documented incident: truncated password).
- The weekly cron refreshes `sales_clean.parquet`; evaluation windows are pinned via
  `ML_FINAL_TEST_CUTOFF` in `config.py` and must not silently move.
