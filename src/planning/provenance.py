"""Where each figure on the Forecast Validation page came from, and when.

The page mixes two kinds of number and, before this module, said so nowhere.

**Live** figures describe the business as it is now: weekly demand, which SKUs
are forecast, how much volume they carry. They read `data/processed`, which the
Tuesday cron rewrites, so they move every week and are supposed to.

**Pinned** figures are measurements of model quality: the comparison grid, the
per-SKU breakdown, the final test. They read the snapshot named by
`config.ML_DATA_SNAPSHOT`, which exists precisely so a recorded result cannot
drift between runs. They are supposed to sit still.

Neither kind is wrong. The failure is presenting them together with one date, or
with no date, so a reader cannot tell which they are looking at. That is not
hypothetical: `coverage` in `/planning/validation` intersected a live served set
with a scored set read from a pinned report, and rendered the result as a single
percentage on the headline card.

**The specific incident this module is built around.** On 2026-08-11 the
snapshot was re-cut as `2026-08-03-v2` with a re-profiled population, moving
smooth/short from 382 SKUs to 247. `scripts/ml_accuracy_report.py` was not
re-run, so the served figures described a cohort that no longer existed. Nothing
detected it for two weeks, because every check in the system asks whether files
are present and none asked whether they agree.

A note on what this module deliberately does not do. It reports drift; it does
not repair it. Regenerating the accuracy report is a retraining pass over three
windows whose output is compared against the design doc at the third decimal,
and starting one from a health check or a cron would move published figures with
no human deciding they should move. See `docs/archive/BACKLOG.md` and the header of
`scripts/ml_accuracy_report.py` for why the weekly cron is the wrong trigger:
the report reads pinned inputs, so a weekly run rewrites identical bytes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.planning import data as _data

REPO_ROOT = Path(__file__).resolve().parents[2]
ACCURACY_META = REPO_ROOT / "outputs" / "reports" / "ml_accuracy_meta.json"
ACCURACY_SUMMARY = REPO_ROOT / "outputs" / "reports" / "ml_accuracy.csv"

#: How far the live population may drift from the evaluated one before the page
#: says so. A share rather than a count, so it does not need revisiting as the
#: catalogue grows.
#:
#: 0.05 because ordinary week-to-week promotion moves a handful of SKUs out of a
#: few hundred, which is low single digits, while both changes that actually
#: mattered were an order of magnitude larger: the 2026-08-11 re-profile moved
#: 42% of the forecastable cohort and the 2026-08-12 threshold alignment moved
#: 27%. Low enough to catch those with room to spare, high enough not to put a
#: banner on the page every week.
POPULATION_DRIFT_TOLERANCE = 0.05

#: Buckets the model does not forecast, and which therefore cannot drift in any
#: way this measurement cares about.
#:
#: `serving/forecast.py` filters to `bucket == "smooth"` and writes that literal,
#: so smooth is the whole served cohort and everything else is the tail. Named
#: as an exclusion rather than as `{"smooth"}` so that adding a forecast bucket
#: later is picked up here by default instead of being silently excluded from
#: the drift check by a set nobody remembered to update.
NON_FORECAST_BUCKETS = ("intermittent",)


def accuracy_manifest() -> dict | None:
    """Provenance recorded by `scripts/ml_accuracy_report.py`, or None.

    None has two causes worth distinguishing, and the caller cannot tell them
    apart from here: a checkout predating the manifest, or a report that has
    never been run. Both mean the same thing to the page, which is that the
    figures cannot be dated, so both are reported the same way.
    """
    if not ACCURACY_META.exists():
        return None
    try:
        return json.loads(ACCURACY_META.read_text())
    except Exception:
        # A corrupt manifest is not worth taking the page down for. It means the
        # figures are undateable, which is exactly what a missing one means.
        return None


def _segment_counts(profiles: pd.DataFrame) -> dict[str, int]:
    """Population by `bucket/history_length`, the profiler's own vocabulary.

    Not the grid's vocabulary: `medium` and `full` are kept apart here rather
    than collapsed to `long`. This is a comparison between two profiling runs,
    so it wants the labels profiling actually wrote, and collapsing them would
    hide a medium/full reclassification that moved no SKU across the smooth
    boundary but did change which model each one is served by.
    """
    if profiles is None or profiles.empty or "bucket" not in profiles.columns:
        return {}
    hist = profiles.get("history_length", pd.Series("?", index=profiles.index))
    key = profiles["bucket"].fillna("?") + "/" + hist.fillna("?")
    return {str(k): int(v) for k, v in key.value_counts().sort_index().items()}


def live_population() -> dict:
    """The cohort being served right now, from `data/processed/sku_profiles.csv`.

    Read through `data.load_profiles()` rather than the file, so it follows the
    same cache and the same fallbacks as everything else the page renders and
    cannot disagree with the numbers beside it.
    """
    try:
        profiles = _data.load_profiles()
    except Exception:
        return {"total": 0, "segments": {}}
    return {"total": int(len(profiles)), "segments": _segment_counts(profiles)}


def _forecastable(segments: dict[str, int]) -> dict[str, int]:
    """Drop the segments the model never forecasts."""
    return {k: v for k, v in segments.items()
            if not k.split("/", 1)[0] in NON_FORECAST_BUCKETS}


def _drift_share(live: dict, scored: dict) -> float | None:
    """How far the served cohort has moved from the measured one.

    Total absolute per-segment movement across the **forecastable** cohort,
    over the size of that cohort as the report measured it.

    Two decisions here, both of which the obvious implementation gets wrong.

    **The tail is excluded.** Intermittent SKUs are 87% of the catalogue and are
    never forecast and never scored, so movement among them cannot affect any
    figure on the page. Including them does not merely add noise, it suppresses
    the signal: measured over the whole catalogue the 2026-08-11 re-profile
    reads as 3.8% and would sit under any tolerance loose enough to be usable,
    which is to say the check would have stayed silent through the exact
    incident it exists to catch. Over the forecastable cohort the same event
    reads 42.1%, against the 41% the repo documents.

    **The movement is not halved.** Treating a reclassification as one event
    rather than two is right within a closed set, and this set is not closed: a
    SKU demoted from smooth to intermittent leaves the cohort altogether and is
    a loss of coverage rather than half of a swap. Halving would report the same
    event as 21%.
    """
    live_seg = _forecastable(live.get("segments", {}))
    scored_seg = _forecastable(scored.get("segments", {}))
    base = sum(scored_seg.values())
    if not base:
        return None
    keys = set(live_seg) | set(scored_seg)
    moved = sum(abs(live_seg.get(k, 0) - scored_seg.get(k, 0)) for k in keys)
    return moved / base


def accuracy_basis(config_snapshot: str) -> dict:
    """What the pinned sections were measured on, and whether it is still current.

    `config_snapshot` is passed in rather than imported so this module stays
    testable without the config, and so the caller cannot accidentally compare
    the manifest against a different snapshot than the one the service is
    actually configured for.

    The returned `drift` block is the answer to two separate questions, kept
    separate because they have different fixes:

    - `snapshot_stale`: the report was measured on a snapshot other than the one
      pinned now. The fix is to re-run the report.
    - `population_stale`: the cohort being served has moved away from the cohort
      that was measured, whether or not the snapshot name changed. The fix is
      also to re-run the report, but this one can be true while the snapshot
      matches, which is the case a name comparison alone would miss.
    """
    meta = accuracy_manifest()
    live = live_population()

    if meta is None:
        # Dated by mtime as a last resort, and labelled as such. This is the
        # pre-manifest behaviour, kept only so an old checkout degrades to what
        # it did before rather than to nothing.
        mtime = None
        if ACCURACY_SUMMARY.exists():
            mtime = pd.Timestamp(ACCURACY_SUMMARY.stat().st_mtime,
                                 unit="s").date().isoformat()
        return {
            "kind": "pinned",
            "snapshot": None,
            "computed_at": mtime,
            "computed_at_is_mtime": True,
            "commit": None,
            "windows": [],
            "scored_skus": None,
            "population": None,
            "live_population": live,
            "drift": {
                "known": False,
                "snapshot_stale": None,
                "population_stale": None,
                "config_snapshot": config_snapshot,
                "report_snapshot": None,
                "population_drift": None,
                "tolerance": POPULATION_DRIFT_TOLERANCE,
            },
        }

    scored_pop = meta.get("population") or {}
    share = _drift_share(live, scored_pop)
    return {
        "kind": "pinned",
        "snapshot": meta.get("snapshot"),
        "computed_at": meta.get("run_at"),
        "computed_at_is_mtime": False,
        "commit": meta.get("commit"),
        "windows": meta.get("windows") or [],
        "scored_skus": meta.get("scored_skus"),
        "population": scored_pop,
        "live_population": live,
        "drift": {
            "known": True,
            "snapshot_stale": meta.get("snapshot") != config_snapshot,
            "population_stale": bool(share is not None and share > POPULATION_DRIFT_TOLERANCE),
            "config_snapshot": config_snapshot,
            "report_snapshot": meta.get("snapshot"),
            "population_drift": share,
            "tolerance": POPULATION_DRIFT_TOLERANCE,
        },
    }


def live_basis(as_of: pd.Timestamp | None = None) -> dict:
    """What the live sections describe, and whether that is as recent as it should be.

    `as_of` is the newest week present in the sales grid rather than today's
    date. `expected` is what the calendar says the newest complete week is, from
    `src.weeks.last_complete_week`, which is the same rule the ingest and the
    scorer use. `weeks_behind` is the gap.

    **The gap is the whole point of this function.** The pinned half of the page
    now announces when it has been superseded, and the live half needed the same
    thing for the opposite failure: not a report that was never regenerated, but
    a cron that stopped delivering. `DATA_AND_PIPELINE.md` Section 4 states the
    problem exactly and then leaves it to the reader, saying a stale forecast
    looks identical to a healthy one from the outside and that you should check
    `trained_through` against the calendar yourself. A check a human has to
    remember to run is one nobody runs, and on 2026-08-11 nobody did: the
    Tuesday run did not deliver the week labelled 2026-08-10 and nothing said
    so for three days.

    Derived from the calendar rather than from today's date directly, because
    the two disagree in exactly the case that matters. An ingest running
    mid-week writes a partial bucket under next Monday's label, and comparing
    against `today` would read those two days as a full week and report the
    pipeline as current.
    """
    if as_of is None:
        try:
            sales = _data.load_sales()
            as_of = sales["ds"].max() if not sales.empty else None
        except Exception:
            as_of = None

    expected = None
    behind = None
    try:
        from src.weeks import last_complete_week

        expected = last_complete_week()
        if as_of is not None:
            # Whole weeks, floored at zero. A negative gap means the grid holds a
            # week the calendar says is still open, which is a different fault
            # (a mid-week ingest stamped as complete) and is not staleness, so it
            # reports as zero here rather than as a negative number nothing reads.
            delta = (expected - pd.Timestamp(as_of)).days
            behind = max(0, delta // 7)
    except Exception:
        pass

    return {
        "kind": "live",
        "as_of": str(pd.Timestamp(as_of).date()) if as_of is not None else None,
        "expected_week": str(pd.Timestamp(expected).date()) if expected is not None else None,
        "weeks_behind": behind,
        "source": "data/processed",
    }


def freshness_summary() -> dict:
    """Whether the weekly pipeline is delivering, for `/health` and the cron.

    Separate from `drift_summary` because the two failures are unrelated and
    have different fixes: this one is a pipeline that has stopped, that one is a
    report nobody regenerated. Collapsing them into one status would make the
    remedy ambiguous at exactly the moment someone needs it.
    """
    basis = live_basis()
    behind = basis["weeks_behind"]
    if behind is None:
        return {"ok": None, "detail": "cannot determine the served week"}
    if behind == 0:
        return {"ok": True, "detail": f"data through {basis['as_of']}, current"}
    return {
        "ok": False,
        "detail": (
            f"served data ends {basis['as_of']} but the last complete week is "
            f"{basis['expected_week']}, {behind} week(s) behind"
        ),
        "fix": "scripts/ml_prepare_data.py --force",
    }


def drift_summary(config_snapshot: str) -> dict:
    """The one-line form, for `/health` and the cron.

    Deliberately not an exception and deliberately not a failed readiness check.
    A drifted report still serves: the grid renders, the figures are real
    measurements of a real cohort, and the only thing wrong is that the cohort
    has moved on. Failing readiness would take the page down over a caption.
    """
    basis = accuracy_basis(config_snapshot)
    d = basis["drift"]
    if not d["known"]:
        return {"ok": True, "known": False,
                "detail": "accuracy report has no provenance manifest; "
                          "re-run scripts/ml_accuracy_report.py to record one"}

    problems = []
    if d["snapshot_stale"]:
        problems.append(
            f"accuracy report was measured on snapshot {d['report_snapshot']} "
            f"but ML_DATA_SNAPSHOT is {d['config_snapshot']}"
        )
    if d["population_stale"]:
        problems.append(
            f"served population has drifted {d['population_drift']:.0%} from the "
            f"{basis['scored_skus']} SKUs the report scored "
            f"(tolerance {d['tolerance']:.0%})"
        )
    return {
        "ok": not problems,
        "known": True,
        "detail": "; ".join(problems) or "accuracy report matches the pinned snapshot "
                                         "and the served population",
        "fix": "scripts/ml_accuracy_report.py" if problems else None,
    }
