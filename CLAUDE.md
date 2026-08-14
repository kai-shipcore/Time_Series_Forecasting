# CLAUDE.md

Guidance for AI assistants working in this repository.

## Project state

The active work is the LightGBM demand-forecasting track in `src/ml/` and
`scripts/ml_*.py`.

**Start with the four-document set.** Rewritten 2026-08-14 as a considered set rather than
an accumulation. Each has a Korean counterpart with the `_KO` suffix.

1. `docs/OVERVIEW.md`: what the project is, how the pieces fit, the evaluation protocol and
   the measured results. **The entry point; read it first.** Its Section 6 is the
   authoritative performance table, and it states the provenance of each column, which the
   older documents got wrong.
2. `docs/MODEL.md`: architecture, features, segmentation rules, what was rejected, how to
   run an experiment.
3. `docs/DATA_AND_PIPELINE.md`: data sources, the week convention, the weekly cron, health
   checks and troubleshooting.
4. `docs/SCREENS.md`: the Action List and Forecast Validation pages, for whoever maintains
   them. Includes the order formula and the known defects.

`docs/FUTURE_IMPROVEMENTS.md` is deliberately standalone: everything identified and not
done, grouped by what blocks it. Read it before proposing an improvement, because most
obvious ideas are already there and several have a recorded reason they were rejected.

**The record**, kept as reference rather than as reading:

- `docs/ML_FORECAST_DESIGN.md`: every design decision with its evidence (Section 4) and the
   model version log (Section 6). The source of truth for what was decided and why. Do not
   re-litigate settled decisions without new evidence; do not contradict them silently.
- `docs/CODEBASE_GUIDE.md`: what every file does and how data flows from raw sales to a
   scored forecast.
- `docs/MODEL_PRIMER.md`: plain-language explainer for anyone without an ML background.
   Covers how trees work, what the features do, and the version narrative.
- `docs/archive/BACKLOG.md`: item-by-item work log, including closed items with their
   reasoning.
- `docs/archive/WORKLOG.md`: running record of completed work for progress summaries.
- `docs/archive/PROJECT_WRITEUP.md`: personal interview study notes, not a reference.
- `docs/archive/` also holds completed task briefs (DEPLOY_TASK, PLANNING_PLAN,
   PLANNING_REQUIREMENTS, VERIFICATION_2026-08-13), the superseded V1_FORMULA (now
   `src/v1.py`), rebaseline logs, and older rebaseline logs.

**Known defects** are collected in `docs/SCREENS.md` Section 4. All accuracy-affecting
defects (final test rendering, stale accuracy report, untracked final_test.json) were fixed
on 2026-08-14.

`Machine_Learning_Demand_Forecast_Proposal.md` and its `_KO` counterpart are the summary
written for management. They are built to `.docx` by `build_docx.sh`, which currently
handles the English file only. **Edit the markdown, never the docx**: the docx is a build
artifact and rebuilding it discards anything typed into it directly.

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

`docs/archive/WORKLOG.md` is the running record of completed work, used to generate
day-by-day progress summaries for the user's manager.

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
  `ML_FINAL_TEST_CUTOFF` in `config.py` and must not silently move. The ML track reads its
  inputs from the snapshot named by `ML_DATA_SNAPSHOT`, not from `data/processed/`, so the
  refresh cannot move results. Advancing it re-baselines every recorded number.
- ML dependencies are pinned to exact versions in `requirements.txt`, because results are
  compared at the third decimal. Note `lightgbm` is not currently installed in `.venv`;
  run `.venv/bin/pip install -r requirements.txt` before any `ml_*` script.
