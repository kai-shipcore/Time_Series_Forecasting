"""The W-MON week convention, in one place.

Extracted from src/ml/serving/history.py so the ingest can use the same rule.
It lived there because scoring was the first thing that needed it; cleaning
needs it too, and a second copy of a calendar calculation is how two parts of a
pipeline come to disagree about which week it is.

The convention, verified against pandas rather than assumed:

    A week is labelled by the Monday it ENDS on. `pd.Grouper(freq="W-MON")`
    is right-closed and right-labelled, so the bucket labelled Monday L spans
    Tuesday (L-6) through Monday L inclusive.

So bucket 2026-08-03 holds Tue 28 July through Mon 3 August. Much of docs/ and
the project notes have said "Mon -> Sun"; the LABEL semantics there are right
and the span is one day out. Rather than the docs being corrected, the code was
briefly changed to match them on 2026-08-05 and changed back on 2026-08-06,
because Tue-Mon turned out to be measurably better for the model and to be what
the SQL in api/main.py and src/db.py had been doing all along. Design doc
Section 4.30 carries the evidence. The documentation is what was wrong.

Three things encode this convention and are only correct together:

    src/clean.py          closed="right" on the W-MON grouper
    last_complete_week    steps back an extra week when asked on a Monday
    the cron              runs Tuesday, because bucket L is open all Monday

Changing one alone produces a pipeline that either trains on a part-finished
week or discards a finished one, and neither failure announces itself.
"""

from __future__ import annotations

import pandas as pd


def last_complete_week(today: pd.Timestamp | None = None) -> pd.Timestamp:
    """The most recent week that has finished, as its W-MON label.

    Bucket L spans Tuesday (L-6) through Monday L, so it is still open for the
    whole of Monday L. On a Monday the answer is therefore the Monday BEFORE
    today; on any other day it is the Monday just past, whose bucket closed at
    the end of that day.

    The extra Monday step is the part that looks like an off-by-one and is not.
    It was removed on 2026-08-05 when src/clean.py briefly used closed="left",
    where it genuinely would have discarded a good week, and restored on
    2026-08-06 with the binning. The two must always move together: with
    closed="right" and no extra step, every Tuesday cron run would train on a
    Monday-only fragment stamped as a full week.

    Derived from the calendar rather than from `max(ds)` in the sales file.
    Those usually agree, but not always: an ingest running mid-week emits a
    partial week under next Monday's label, and treating that as complete reads
    two days of orders as a full week of demand.
    """
    today = pd.Timestamp(today or pd.Timestamp.today().normalize()).normalize()
    monday = today - pd.Timedelta(days=today.dayofweek)  # Monday == 0
    if today.dayofweek == 0:
        monday -= pd.Timedelta(days=7)
    return monday


def drop_incomplete_weeks(
    weekly: pd.DataFrame,
    today: pd.Timestamp | None = None,
    ds_col: str = "ds",
) -> pd.DataFrame:
    """Remove W-MON buckets whose week has not finished yet.

    The bug this exists for: `clean()` grouped orders into W-MON buckets and
    kept everything through the last bucket containing an order, so a pipeline
    run on a Wednesday produced a final bucket holding three days of sales and
    stamped it as a full week. `TRIM_TRAILING_WEEKS = 0` carried the comment
    "train through the last complete week", which described an intention the
    code did not implement.

    The consequence was not cosmetic. That partial week is one of the twelve in
    the trailing mean the model's target is a ratio to, and one of the four in
    `ramp_4_12`, where a short week hits the numerator hardest. Every SKU reads
    as falling and every forecast comes out low. It only stayed hidden because
    runs had landed on Mondays, where the last bucket happens to be complete.
    """
    if weekly.empty:
        return weekly
    cutoff = last_complete_week(today)
    return weekly[pd.to_datetime(weekly[ds_col]) <= cutoff]
