# Work Log

Running record of completed work on the ML forecasting project, one dated line per item.
Used to generate day-by-day progress summaries for reporting. When a summary is produced,
a marker line is added; the next summary covers everything after the latest marker.

Entry style: date, then a plain one-line description of what was completed. Technical
detail lives in the design document and codebase guide, not here.

---

- 2026-07-17: Created the master design document (goals, metrics, data and split
  protocol, decision log, feature backlog) and revised it into a professional format.
- 2026-07-17: Created the codebase guide documenting every file and the full data flow
  from raw sales to scored forecast.
- 2026-07-17: Audited and hardened the evaluation harness: as-of eligibility and segment
  labels, pinned evaluation windows, stratified validation SKUs, segment-merge reporting,
  missing-prediction warnings, ramp cohort moved into the harness.
- 2026-07-17: Established the structural baseline (per-segment seasonality; alpha search
  confirmed no seasonal adjustment for short SKUs) with floor numbers on three dev
  windows.
- 2026-07-17: Built and evaluated model versions v0 (lead-only, rejected), v1 (ramp
  block, rejected), and v2 (deseasonalized trajectory features, rejected); each
  rejection's mechanism identified and logged; v3 direction defined.

--- SUMMARY PRODUCED 2026-07-20 (covering all entries above) ---

- 2026-07-20: Set up the work log process and repository instructions (CLAUDE.md) so
  future work sessions follow the same documentation and reporting habits.

- 2026-07-20: Built and evaluated model version v3 (fully seasonally consistent ML path;
  met two of three pass criteria, best version so far, blocked by one segment's
  regression in the post-holiday window).

- 2026-07-20: Pinned model development to a frozen copy of the sales data so that
  accuracy comparisons between model versions are measured on identical numbers. The
  weekly automated refresh was quietly changing the figures underneath the comparison,
  which had already shifted recorded results. The automated refresh continues to run
  unchanged for the production forecast; only the development track reads the frozen
  copy. Added a tool to create and check these frozen copies, and confirmed all six
  previously recorded baseline accuracy figures reproduce exactly.

- 2026-07-20: Put the forecasting work under version control for the first time and saved
  each model version as its own labelled checkpoint, so any earlier version can be
  retrieved and re-run later. None of this work had been backed up before.

- 2026-07-20: Found and fixed a fault that had made the first model version impossible to
  re-run: it was picking up a later version's settings rather than its own. Checked all
  four versions re-run correctly, and locked the software library versions so results
  cannot shift between machines.

- 2026-07-20: Re-measured every model version's accuracy on the frozen data so all four
  are scored on identical figures and can be compared directly. Previously the earlier
  versions were measured on older data than the latest one, so the same model showed
  different numbers in different tables. None of the conclusions changed. Also wrote down
  the rule for when an accuracy difference counts as real rather than chance, which had
  been applied but never actually defined.

- 2026-07-20: Built and evaluated model version v4, which tells the model whether each
  product is long-established or new. It fixed the problem it was designed to fix, giving
  the best result any version has achieved for established products in the post-holiday
  period, but it did so by shifting effort away from newer products and making those
  noticeably worse. Rejected on that trade. The finding points clearly at the next step:
  the model needs to be prevented from favouring one group over the other, rather than
  simply being told the two groups differ.
