#!/usr/bin/env python3
"""THE FINAL TEST. Runs once, on the quarantined window. Read Section 4.34 first.

    .venv/bin/python scripts/ml_41_final_test.py

No database. Refuses to run if the preconditions do not hold, and refuses to
overwrite a result that already exists.

This is the only script in the repository that touches the window pinned by
ML_FINAL_TEST_CUTOFF. Everything else uses dev_splits(), which excludes it.

Why the preflight is not ceremony
---------------------------------
The window can be spent once. A run against a stale snapshot, a mismatched
LightGBM, a half-committed working tree or a harness that leaks does not produce
a weaker result, it produces a result that cannot be interpreted at all, and the
window is gone either way. Each check below corresponds to something that has
actually gone wrong in this project during the last week:

  seasonal blend off      ML_SEASONAL_BLEND has three modes and two of them were
                          rejected; testing under the wrong one is silent.
  snapshot == reference   the reference figures went stale under a moved config
                          value once already (Section 4.31).
  manifest checksums      the pinned data is meant to be immutable; if it has
                          moved, nothing recorded is comparable.
  lightgbm pinned         results are compared at the third decimal.
  clean working tree      a result must be attributable to a commit.
  harness integrity       the model must be refitted per window, trained only on
                          data at or before the cutoff, and order-independent
                          (scripts/ml_38).
  no prior result         single use is the whole point.
"""
from __future__ import annotations

import json
import hashlib
import pathlib
import subprocess
import sys
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from config import ML_DATA_SNAPSHOT, ML_FINAL_TEST_CUTOFF  # noqa: E402
from src.ml.dataset import (data_dir, final_test_split,  # noqa: E402
                            load_weekly, stratified_val_skus)
from src.ml.evaluate import bootstrap_delta, is_significant, score, score_table  # noqa: E402
from src.ml.model import (FEATURES_V1, FEATURES_V11_LONG, RatioLGBM,  # noqa: E402
                          long_sku_set, structural_baseline)
sys.path.insert(0, str(ROOT / "scripts"))
from compare_v1 import build_cumsum_index, v1_forecast  # noqa: E402

def _raw_path() -> pathlib.Path:
    """The order lines V1 is computed from, preferring the pinned copy.

    V1 became the final test's primary comparator on 2026-08-13, and until then
    orders_raw.parquet lived only in data/processed, which the weekly ingest
    rewrites. A one-shot test whose primary comparator drifts is not reproducible,
    so a frozen copy was placed beside the pinned inputs in the snapshot directory
    and is preferred here.

    It is deliberately NOT added to manifest.json: snapshots are written read-only
    by scripts/ml_snapshot_data.py and are meant to be immutable, so rewriting a
    manifest after the fact would undermine the guarantee it exists to give. The
    file's md5 is recorded in final_test.json instead, which is where it matters
    for this run.

    Falls back to data/processed with a warning rather than failing, because a
    fresh clone has no snapshot copy and the fallback is still correct, merely
    not pinned.
    """
    pinned = data_dir() / "orders_raw.parquet"
    if pinned.exists():
        return pinned
    live = ROOT / "data" / "processed" / "orders_raw.parquet"
    print(f"WARNING: no pinned orders_raw.parquet in {data_dir()}; using {live}, "
          "which the weekly ingest rewrites. The V1 figure will not be reproducible.")
    return live


def v1_predictions(index: dict, split, skus: list[str]) -> "pd.DataFrame":
    """One row per SKU: yhat = V1's 70-day total, ds = first test week.

    Lifted from scripts/ml_02_v1_benchmark.py so the primary criterion is scored
    by the same production V1 implementation every other comparison uses.

    As-of is `cutoff - 1 day`, NOT the cutoff. Under W-MON a week labelled `ds`
    covers [ds-7, ds-1], so a cutoff label of 2026-05-04 means training ends
    Sunday 2026-05-03. v1_forecast treats its date argument as the last day of
    available history, so passing the cutoff label would read one day of the test
    period as history and shift the whole 70-day span a day late.
    """
    asof = split.cutoff - pd.Timedelta(days=1)
    first_week = split.test["ds"].min()
    return pd.DataFrame([
        {"unique_id": uid, "ds": first_week, "yhat": v1_forecast(index, uid, asof)}
        for uid in skus
    ])
from src.ml.reference import REFERENCE_SNAPSHOT  # noqa: E402

RESULT = ROOT / "outputs" / "reports" / "final_test.json"


def preflight() -> list[str]:
    """Every reason this run would be uninterpretable. Empty list means go."""
    bad: list[str] = []

    from config import ML_SEASONAL_BLEND
    if ML_SEASONAL_BLEND != "off":
        bad.append(f"ML_SEASONAL_BLEND is {ML_SEASONAL_BLEND!r}, expected 'off' "
                   f"(v15 and v16 were both rejected)")

    if ML_DATA_SNAPSHOT != REFERENCE_SNAPSHOT:
        bad.append(f"snapshot {ML_DATA_SNAPSHOT} but reference figures are from "
                   f"{REFERENCE_SNAPSHOT}; re-measure src/ml/reference.py first")

    # The pinned data must be byte-identical to what the manifest recorded.
    import hashlib
    man = data_dir() / "manifest.json"
    if not man.is_file():
        bad.append(f"{man} missing; cannot prove the snapshot is unchanged")
    else:
        m = json.loads(man.read_text())
        for name, rec in m.get("files", {}).items():
            p = data_dir() / name
            if not p.is_file():
                bad.append(f"{name} missing from the snapshot")
                continue
            h = hashlib.md5(p.read_bytes()).hexdigest()
            if h != rec.get("md5"):
                bad.append(f"{name} has changed since the manifest was written "
                           f"({h[:8]} vs {rec['md5'][:8]})")

    import lightgbm
    pin = next((l.split("==")[1].strip()
                for l in (ROOT / "requirements.txt").read_text().splitlines()
                if l.strip().startswith("lightgbm==")), None)
    if pin and lightgbm.__version__ != pin:
        bad.append(f"lightgbm {lightgbm.__version__} but requirements pins {pin}")

    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        bad.append("working tree is not clean, so the result would not be "
                   f"attributable to a commit:\n      "
                   + "\n      ".join(dirty.splitlines()[:6]))

    r = subprocess.run([sys.executable, "scripts/ml_38_training_integrity.py"],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        bad.append("harness integrity checks FAILED; see ml_38 output")

    if RESULT.is_file():
        prev = json.loads(RESULT.read_text())
        bad.append(f"the final test has ALREADY been run, on {prev.get('run_at')}, "
                   f"commit {str(prev.get('commit'))[:8]}. Delete "
                   f"{RESULT.relative_to(ROOT)} deliberately if you mean to re-run "
                   f"it, and record why. Running it twice does not un-run it.")
    return bad


def main() -> int:
    print("FINAL TEST — preflight\n")
    bad = preflight()
    for b in bad:
        print(f"  BLOCKED: {b}")
    if bad:
        print("\nNot running. Fix the above; the window is spent either way.")
        return 1
    print("  all preconditions hold\n")

    weekly, profiles = load_weekly()
    smooth = set(profiles.loc[profiles["bucket"] == "smooth", "unique_id"])
    ws = weekly[weekly["unique_id"].isin(smooth)]
    split = final_test_split(ws)
    print(f"{'=' * 72}\nQUARANTINED WINDOW  {split}\n")

    longs = long_sku_set(profiles, split.cutoff) & set(split.train["unique_id"])
    val_all = stratified_val_skus(split.train, profiles)
    val_long = stratified_val_skus(
        split.train[split.train["unique_id"].isin(longs)], profiles)

    m = RatioLGBM(split.horizon, FEATURES_V1, deseas_features=True,
                  deseas_all=True).fit(split.train, profiles, split.cutoff, val_all)
    preds = m.predict(split.train, profiles, split.cutoff)
    mL = RatioLGBM(split.horizon, FEATURES_V11_LONG, deseas_features=True,
                   deseas_all=True, uids=longs).fit(
        split.train, profiles, split.cutoff, val_long)
    preds = pd.concat([preds[~preds["unique_id"].isin(longs)],
                       mL.predict(split.train, profiles, split.cutoff)],
                      ignore_index=True)
    base = structural_baseline(split.train, split.test, profiles, split.cutoff)

    # V1, the spreadsheet the company runs on, is the primary comparison.
    raw_path = _raw_path()
    raw = pd.read_parquet(raw_path)
    raw["order_date"] = pd.to_datetime(raw["order_date"])
    raw = raw[raw["unique_id"].isin(smooth)]
    v1 = v1_predictions(build_cumsum_index(raw), split,
                        sorted(split.train["unique_id"].unique()))

    results = {"v11": score(preds, split, profiles),
               "V1": score(v1, split, profiles),
               "baseline": score(base, split, profiles)}
    print(f"  long model: {mL.n_train_rows:,} rows, {len(val_long)} val SKUs, "
          f"{mL.model.best_iteration_} trees\n")
    print("RAW per-segment results:")
    print(score_table(results).to_string())
    print("\nbias% by segment:")
    print(pd.DataFrame({n: t.set_index("segment")["bias_pct"]
                        for n, t in results.items()}).to_string())

    # Section 4.34's go/no-go. V1 is the spreadsheet in production, so this is
    # the only comparison that decides whether v11 ships. The baseline delta is
    # printed after it as context, not as a criterion.
    print(f"\n{'=' * 72}\nPRIMARY CRITERION: v11 beats V1, the production spreadsheet\n")
    verdict = {}
    for seg in ("short", "long"):
        d = bootstrap_delta(preds, v1, split, profiles, segment=seg)
        verdict[seg] = d
        print(f"  smooth/{seg:<6} {d['delta']:+.4f}  se {d['se']:.4f}  "
              f"95% CI [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]  "
              f"{'SIGNIFICANT' if is_significant(d) else 'not distinguishable from noise'}"
              f"   {'v11 ahead' if d['delta'] < 0 else 'V1 ahead'}")

    tot = {n: t.set_index("segment")["pooled_wape"].get("TOTAL")
           for n, t in results.items()}
    print(f"\n  TOTAL       v11 {tot['v11']:.4f}   V1 {tot['V1']:.4f}   "
          f"{'v11 ahead' if tot['v11'] < tot['V1'] else 'V1 ahead'}")

    print(f"\n{'-' * 72}\nContext, not a criterion: v11 against the structural baseline\n")
    context = {}
    for seg in ("short", "long"):
        d = bootstrap_delta(preds, base, split, profiles, segment=seg)
        context[seg] = d
        print(f"  smooth/{seg:<6} {d['delta']:+.4f}  se {d['se']:.4f}  "
              f"   {'v11 ahead' if d['delta'] < 0 else 'baseline ahead'}")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    payload = {
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": commit, "snapshot": ML_DATA_SNAPSHOT,
        # Which order file produced the V1 figure, so the primary comparison is
        # reproducible even though this input is not in the snapshot manifest.
        "v1_orders_raw": {
            "path": str(raw_path.relative_to(ROOT)),
            "md5": hashlib.md5(raw_path.read_bytes()).hexdigest(),
        },
        "cutoff": str(split.cutoff.date()),
        "test_weeks": [str(w.date()) for w in sorted(split.test["ds"].unique())],
        "scores": {n: t.set_index("segment")["pooled_wape"].to_dict()
                   for n, t in results.items()},
        "v11_vs_v1": {k: v for k, v in verdict.items()},
        "v11_vs_baseline": {k: v for k, v in context.items()},
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nrecorded in {RESULT.relative_to(ROOT)}")
    print("Write it up in Section 4.34's result entry whichever way it landed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
