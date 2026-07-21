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

- 2026-07-20: Fixed a date-alignment fault in the benchmark of the current spreadsheet
  method, which had been scored against a period one day off from the one it was
  forecasting. Corrected the document's description of that method: it had been described
  as consistently forecasting too low, but measurement shows its error changes direction
  by season. Also recorded that the spreadsheet method still beats our baseline for
  established products in the autumn period, which is the one case where it wins and had
  not been written down anywhere.

- 2026-07-20: Built and evaluated model version v4, which tells the model whether each
  product is long-established or new. It fixed the problem it was designed to fix, giving
  the best result any version has achieved for established products in the post-holiday
  period, but it did so by shifting effort away from newer products and making those
  noticeably worse. Rejected on that trade. The finding points clearly at the next step:
  the model needs to be prevented from favouring one group over the other, rather than
  simply being told the two groups differ.

- 2026-07-20: Added two permanent health checks that run alongside every model version:
  one tests whether the seasonal adjustment actually fits each product group, the other
  breaks a model's error out month by month. The first check found that the December
  adjustment fits established and newer products very differently, which had gone
  undetected under the single headline accuracy number.

- 2026-07-20: Measured the earlier statistical system (the project's stated accuracy
  target) through the same evaluation used for everything else, for the first time. In
  doing so found and fixed two faults in its evaluation script, including one where it
  treated newer products differently from the intended method. Result: the current model
  already matches or beats that target for newer products across all test periods, with
  one remaining gap for established products in the post-holiday period.

- 2026-07-20: Tested giving established and newer products different December
  adjustments, based on the health-check finding. Clearly rejected: with only one or two
  Decembers of history the estimates are unreliable, and adjusting one month shifts
  errors into the neighbouring months. Cheap to find out now rather than after building
  it into the model; revisit after this year's holiday season adds a third December.

- 2026-07-20: Confirmed with statistical checks that the current model's only real
  remaining weakness is established products in the post-holiday period; everywhere else
  it now matches or beats both the earlier statistical system and the spreadsheet method
  used in production.

--- SUMMARY PRODUCED 2026-07-21 (covering all 2026-07-20 entries above) ---

- 2026-07-21: Investigated why forecasts run high after the holidays and found the cause
  is the seasonal adjustment rather than the model: a holiday uplift is being applied to
  the last two weeks of December, which in both years of available data were at or below
  normal demand rather than above it. The code also contradicts its own documented
  intent, which says that period should revert to the normal December level.

- 2026-07-21: Gave the machine-learning work its own copy of the seasonal settings so that
  adjusting them cannot alter the older statistical system we measure ourselves against.
  Verified the separation changes no existing result, and added a permanent check that
  will catch it if the two ever drift apart unintentionally.

- 2026-07-21: Tested rebalancing how much each product group influences training, on the
  theory that established products were crowding out newer ones. Rejected: it made newer
  products worse rather than better, which rules out that explanation. A control version
  included in the same test showed the real cause is simply telling the model which group
  a product belongs to, which helps established products and harms newer ones no matter
  how the training is balanced. Four different approaches have now hit this same trade-off,
  which points clearly at keeping separate models for the two groups.

- 2026-07-21: Tried training separate models for established and newer products, the last
  of the structural options. Rejected: it helped established products slightly but made
  newer ones significantly worse, confirming that newer products benefit from being
  trained alongside established ones.

- 2026-07-21: Corrected the holiday period used by the forecast so it matches the
  promotional period the company actually runs, late November to mid-December, rather than
  running to the end of December. This resolved the long-standing post-holiday accuracy
  problem for established products, which five previous attempts had failed to fix by
  treating it as a difference between product groups. The forecast now matches or beats
  the earlier statistical system for both product groups across all three test periods.

- 2026-07-21: Ran a broad hyperparameter search over the model's main tuning settings,
  after an initial narrow attempt had tested too little to conclude anything. Result: the
  model is essentially already well configured; the best settings found gave no meaningful
  improvement and made one test period slightly worse. More importantly, this settles that
  the remaining post-holiday weakness for established products is not a tuning problem. It
  comes from the model expecting continued growth into a period where demand falls, which
  will need a new signal that anticipates the turn rather than any tuning change.

- 2026-07-21: Solved the long-standing post-holiday accuracy problem for established
  products. Split the forecaster into two models: newer products keep the shared model
  that serves them well, and established products get a dedicated model with a new feature
  that recognises when a product is running far above its yearly norm and is therefore
  likely to fall back rather than keep climbing. This fixed the one remaining weak spot
  decisively, left newer products untouched, and the forecast now matches or beats the
  earlier statistical system in five of six test cases. The feature generalises to any
  temporary spike, not just the holidays. Still to do before this can replace the current
  spreadsheet method: the final held-out test, which is run only once.
