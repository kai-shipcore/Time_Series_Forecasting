# Handover: findings and caveats

> **FIGURES SUPERSEDED, 2026-08-12. The findings hold; the numbers are two snapshots old.**
>
> Written against the `2026-08-03-onset` run. Since then the promotion and classification
> thresholds were matched at 3.0, moving the smooth set from 467 SKUs to 340 and changing
> every accuracy figure. Current numbers: `ML_FORECAST_DESIGN.md` Sections 4.32 and 6.
>
> To be rewritten immediately before handover, deliberately once rather than after each
> change.

Written 2026-08-11. Read this before `ML_FORECAST_DESIGN.md`, which is long and
assumes you already know what is contested.

This file holds two things: what was learned that is not a model version, and
what is wrong or unproven about the model that is. Both matter more than the
version log, because the version log records which experiments passed and this
records what the passes and failures mean.

Evidence pointers are to scripts and log directories that exist in the repo.
Every number below can be re-derived; none of it needs to be taken on trust.

---

## Part 1. Findings

### 1. The evaluation had been excluding 41% of the forecastable catalogue

The single most important item here.

`src/profile.py` promotes an intermittent SKU to smooth when its recent 13 weeks
look steady, and then assigned three constants: `train_start` to the start of
that 13-week window, `active_weeks` to 13, `history_length` to `short`. The
first of those pinned every promoted SKU's history to a date that moves forward
every profiling run and sits in the future relative to every backtest cutoff. So
all 190 promoted SKUs, 41% of the 467 smooth SKUs, had negative history at every
development window and were silently dropped from every figure ever recorded.

Only 15 of the 190 genuinely had 13 weeks. The median had 34, the maximum 111,
which is the entire series. 73 had at least 50 weeks and were still labelled
`short`, so they were routed to the short model as well as starved of the data
they had. 4,615 SKU-weeks of usable history were being discarded, because
`load_weekly` trims each SKU's training frame to `train_start`.

Fixed 2026-08-11 by detecting the actual onset (`_smooth_onset`). The scored
population went from 266/251/66 SKUs across the three windows to 356/327/103.

**Consequence you must understand before reading any older figure.** Every
number recorded before 2026-08-11 was measured on the easier 59% of the
catalogue, and nobody knew. v11's margin over the structural baseline shrank in
all three windows once the excluded SKUs appeared, and reversed in one:

| smooth/long, v11 minus baseline | before | after |
|---|---|---|
| Mar-May | +0.0024 | +0.0168 |
| Dec-Feb | −0.0728 | −0.0584 |
| Oct-Dec | −0.0212 | −0.0139 |

Evidence: `docs/rebaseline_2026-08-03/` versus `docs/rebaseline_2026-08-03-onset/`.

### 2. The model's value is concentrated, and a wide middle band is unproven

`scripts/ml_37_band_worth_forecasting.py` scores every SKU against the
structural baseline, binned by trailing weekly demand, bootstrapped within band.

| band | SKU-windows | units | model | baseline | difference | verdict |
|---|---|---|---|---|---|---|
| <2 | 73 | 1,914 | 0.3571 | 0.4411 | −0.0840 | model better |
| 2–2.5 | 84 | 2,366 | 0.3198 | 0.2985 | +0.0213 | indistinguishable |
| 2.5–3 | 57 | 1,636 | 0.3301 | 0.2816 | +0.0485 | indistinguishable |
| 3–4 | 99 | 3,769 | 0.2436 | 0.2499 | −0.0062 | indistinguishable |
| 4–6 | 125 | 6,096 | 0.2888 | 0.2560 | +0.0328 | indistinguishable |
| 6–10 | 131 | 10,277 | 0.2394 | 0.2187 | +0.0207 | indistinguishable |
| 10+ | 217 | 64,981 | 0.1336 | 0.1592 | −0.0255 | model better |

The model beats a trailing 12-week mean above 10 units a week and below 2.
Between 2 and 10 it cannot be shown to beat it, and four of those five bands
have point estimates favouring the baseline. The 4–6 band is 1.9 standard errors
in the baseline's favour, which misses the significance bar but not by much.

This does not mean v11 is worthless. The 10+ band carries 71% of units and
pooled WAPE is demand-weighted, so the aggregate wins are real and are driven by
that band. It means the value is concentrated, and that a wide middle band is
being forecast with no demonstrated benefit.

The `<2` band favouring the model is counterintuitive and **not explained**. The
plausible story is that these are recently-onset SKUs with rising demand, where
a trailing mean under-forecasts and the ramp features correct for it. That is a
hypothesis. It has not been tested.

### 3. The success metric cannot see improvements to the unproven band

Follows directly from finding 2, and it is the one that should shape what anyone
works on next.

Suppose a change fixed the 2–10 band perfectly, bringing every band to at least
baseline parity. Pooled WAPE across the three development windows would move
from about 0.1736 to about 0.1679, an improvement of roughly **0.006**. Section
1.5 requires 0.01 with a consistent sign, and the single-window noise floor is
±0.011 to ±0.014.

So a change that completely solved the problem would be rejected by the criteria
as written, because pooled WAPE is demand-weighted and those bands hold 29% of
units. Anyone intending to work on low and mid-volume accuracy has to change the
criterion first, and pre-register it, or they will measure a real improvement
against a rule that cannot detect it and then argue about the result.

### 4. The long model degrades under any perturbation

Five changes of different kinds, in different directions, in different parts of
the pipeline, each costing the long segment roughly half a point to a point:

| perturbation | smooth/long cost |
|---|---|
| seasonal blend, both halves (v15) | +0.0059 |
| monthly index blended only | +0.0085 |
| holiday window blended only (v16) | +0.0054 |
| week boundary shifted one day (Section 4.30) | ~3x comparators' sensitivity |
| exogenous channel-mix feature (v17) | +0.0120 |

The consistent reading is not that each change was individually wrong. It is a
model at a narrow optimum on a small sample, 55 to 62 SKUs depending on window,
where nearly any added degree of freedom costs more in variance than it returns
in signal. For a segment that size this is expected behaviour rather than a
defect, and it is the strongest argument in the repository against continuing to
add features.

### 5. A recorded baseline figure had been stale since v9

The Section 6 v-base table did not reproduce. Four of its six cells matched a
fresh run; Mar-May long and Dec-Feb long did not. Setting `ML_HOLIDAY_END` back
to `(12, 31)`, its value before v9, reproduced all six exactly including the
bias percentages.

v-base applies the seasonal round-trip to long SKUs, so the holiday window is
one of its inputs. When v9 moved that window the baseline moved with it and the
table was never re-measured. Dec-Feb long was recorded as 0.2764 against a true
0.2167, overstating by 0.06 every margin quoted over the baseline in that window
from v9 onward.

No verdict changed. Comparisons computed inside a script recompute the baseline
and were never affected; only figures transcribed into the document went stale.

**The reason it survived is the more useful part.** `ml_12` is the regression
check positioned to catch exactly this, and it had been failing since v9 for an
unrelated and legitimate reason: it required the ML and prototype factor sources
to agree on every pinned week, which the v9 window split makes impossible on the
four weeks covering Dec 16 to 31. A check that always fails is a check nobody
reads. Both were fixed 2026-08-10.

Evidence: `scripts/ml_33_holiday_window_audit.py`.

### 6. The evaluation selects its population using future information

`asof_history_length` and `eligible_skus` are as-of the cutoff, deliberately and
with stated reasons. The **bucket**, smooth versus intermittent, is not. It is
taken from the present-day profile.

Measured: at the 2025-10-06 cutoff, 121 SKUs were classifiable as smooth from
data available then, while the harness scores against today's 467. So 349 SKUs
enter that window on the strength of behaviour that had not happened yet, and 3
that were smooth at the time are excluded because they are not smooth now.

The direction is not neutral. SKUs admitted by hindsight are the ones that went
on to sell smoothly, which is the easier population. The valid half of
`scripts/ml_34_asof_bucket_audit.py` shows the long segment degrading under as-of
buckets in all three windows.

**Not fixed.** The design doc scopes this out as a deeper change and it remains
so. It is recorded here because it is a leak, not merely a coverage gap, and
because it applies to the final test window as much as the development ones.

Caution: `ml_34` has a bug. Its TOTAL column and margin table are contaminated
because the test frame was not restricted to the scored population, so roughly
3,000 intermittent SKUs were scored as zero forecasts. The per-segment numbers
are sound and reproduce the recorded run exactly. Fix the test frame before
reusing it.

### 7. Promoted SKUs cannot be validated by backtest, only forward

Related to 1 and 6, and it constrains what is answerable.

At the 2025-10-06 cutoff, of the 190 promoted SKUs, 22 had not sold at all, 78
had under 13 weeks of history, 83 had history but were genuinely sparse then,
and **7** were regular with a mean below the classification cutoff. The
population that the entire mean-threshold argument concerns numbered seven.

It is not that these SKUs are recent. 83 of 190 first sold before July 2025.
Being *regular* is what is recent for them. A backtest cannot measure how well
regular low-volume SKUs are forecast when there were barely any at the time.

The only instrument that can reach them is `shipcore.ml_forecast_history`, which
accumulates what was predicted before the outcome was known. It received its
first real rows on 2026-08-11, 6,071 of them. In a few months it can answer the
threshold question directly, by binning served forecasts by demand and comparing
against actuals. That is a defined experiment, not an open argument.

### 8. Channel mix does not help, and the model used it heavily

v17 added the trailing 12-week Amazon FBA share to the long model, the first
exogenous input tried. Rejected: the long segment regressed in all three
windows, mean +0.0120 against a bar requiring a 0.0100 improvement.

The pre-registered escape hatch does not apply. It said near-zero feature gain
would mean the model ignored the feature and the null said nothing about channel
mix. Instead the share ranked first by gain in two windows and second in the
third. The model reached for it and was made worse by it.

The precondition test used to justify running it was necessary and insufficient,
and that is worth remembering. The share was checked for within-SKU movement, on
the reasoning that a fixed per-SKU value is a fingerprint a tree will memorise.
It passed at 70% within-SKU variation. The remaining 30% is cross-sectional
identity, and across 53 SKUs that was enough.

### 9. The ingest was dropping a partial week at one end and keeping one at the other

`drop_incomplete_weeks` trimmed the tail. Nothing trimmed the head. The source
orders begin on 2024-06-17, which under Tuesday-to-Monday weeks is the last day
of the bucket labelled 2024-06-17, so that bucket held one day of seven and was
stamped as a full week: 32 units against neighbours of 280 to 415.

The tail case was caught a year ago because a short final week looks wrong. The
head case survived because a short first week looks like a launch. Fixed
2026-08-10 by `drop_leading_partial_week`, which tests the calendar rather than
inferring anything from the unit count.

### 10. The week convention is Tuesday to Monday, on evidence, mechanism unexplained

Section 4.30 has the detail. Seven phases were swept; v11 scores best on Tuesday
to Monday in seven of eight cells, consistently across seasons, while the
comparators' optima wander. Leave-one-window-out selection picks Tuesday on
every fold for an out-of-sample gain of 0.0132, with selection optimism measured
at 0.0001.

**Why the mechanism is not understood is itself a finding.** An arbitrary
one-day shift of a bucket boundary should not move a model by that much. It is
the same sensitivity finding 4 describes from a different angle.

Three things encode this convention and are only correct together: `clean.py`
`closed="right"`, `last_complete_week` stepping back an extra week on Mondays,
and the cron running Tuesday. Change one alone and the pipeline either trains on
a part-finished week or discards a good one, and neither failure announces
itself.

---

## Part 2. Caveats on the model itself

**The long model rests on 55 to 62 SKUs**, depending on window, and 23 of them
are correlated `CC-CN-03`/`CC-CP-03` variants. Its effective sample is smaller
than the count suggests and its intervals overstate confidence.

**v11 is now behind the structural baseline on Mar-May long** (+0.0168) after the
population fix. It was a tie before. The Dec-Feb win survives and remains
significant, so v11 still passes its criteria, but the margin is narrower than
the version log has ever shown.

**The Dec-Feb win rests on two observed Decembers.** The elevation feature is
trained on every elevated-then-reverted episode across all long SKUs and weeks
rather than on December alone, which helps, but the headline result is still two
seasons.

**`elev_long` cannot distinguish a temporary spike from a genuine new plateau**,
so a long SKU that truly breaks out to a higher level will be under-forecast.
Acceptable because long SKUs are mature and rarely ramp, and `y_last_r` and
`lag_1_r` still carry gradual growth. Not acceptable if the catalogue changes
character.

**Point forecasts only.** No prediction intervals. The ML tables have no
`yhat_lo_*` / `yhat_hi_*` columns. `_MIGRATE_PI_SQL` in `src/db.py` is the
pattern for adding them.

**Pooled WAPE is demand-weighted**, so every headline number is dominated by
high-volume SKUs. This is intentional, units approximate dollars, but it means
the headline is not a statement about the typical SKU. See findings 2 and 3.

**The final test window was evaluated once during development, on 2026-08-11, by
mistake.** `scripts/ml_34_asof_bucket_audit.py` included it in its window list
and fitted v11 on it. The numbers seen were smooth/short 0.2093 and smooth/long
0.1353 under today's buckets. Nothing was tuned on it; it was one measurement,
not a search. It is recorded because the value of that window comes from being
used once, and anyone reading a final test result is entitled to know it had
been observed. As of writing, the final test has **not** been run as the formal
gate.

---

## Part 3. Data caveats

**The source table is restated continuously.** `fc_velocity_link_snapshot_forecast`
disagreed with a three-week-old snapshot on 73 of 110 weeks, 0.4% of units.
Snapshots are point-in-time and the pinned one is itself a restatement of 2024
data. This is the ordinary caveat, but it means "re-run it and see" is never
exactly reproducible against a live query.

**No stockout or censored-demand correction.** The model trains on units sold.
During a stockout, sold is capped by available rather than by wanted, and the
model reads that as demand falling. Understates the forecast, which understates
the order, which makes the next stockout likelier. BACKLOG 4. Blocked: nothing
records stockout or restock dates.

**No inventory history**, only a current snapshot, which is why stockout-aware
demotion (BACKLOG 1) cannot be done as-of and would leak into backtests.

---

## Part 4. What I would do next, in order

1. **Run the final test once**, on whatever model is current, and report it
   whichever way it lands including the exposure noted above. It is the
   deliverable and everything gating it is settled.
2. **Rewrite the Section 6 tables against `2026-08-03-onset`.** They are
   currently measured on a snapshot whose profiling has a known defect. Every
   table needs re-deriving from `docs/rebaseline_2026-08-03-onset/`, and
   `src/ml/reference.py` re-measured with it.
3. **Decide whether to change the evaluation criterion** before attempting any
   further accuracy work on low and mid-volume SKUs. Finding 3 is the reason.
4. **Leave the demand thresholds alone until the forward data can settle them.**
   `MEAN_INTERMITTENT_CUTOFF` at 3.0 and `RECENT_MEAN_UPGRADE` at 2.0 are
   inconsistent, both were chosen rather than measured, and the evidence
   available today does not support either value over the other.
5. **Consider as-of bucket recomputation** (finding 6). It is a real leak, it is
   cheap to compute at one second per cutoff, and it is the last input in the
   harness still using future information.

## Part 5. Known and accepted, not to be re-raised as findings

Assessed 2026-08-10 and accepted on the grounds that the repository is internal
to about three developers: `Archive.zip` containing zipped `.env` files,
credentials in `.claude/settings.local.json` tracked since April, and a
partially exposed `FORECAST_API_TOKEN`. Recorded so the next person inherits the
judgement rather than rediscovering the files and not knowing whether anyone had
looked.
