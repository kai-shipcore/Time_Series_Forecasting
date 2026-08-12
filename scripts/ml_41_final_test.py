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

    results = {"baseline": score(base, split, profiles),
               "v11": score(preds, split, profiles)}
    print(f"  long model: {mL.n_train_rows:,} rows, {len(val_long)} val SKUs, "
          f"{mL.model.best_iteration_} trees\n")
    print("RAW per-segment results:")
    print(score_table(results).to_string())
    print("\nbias% by segment:")
    print(pd.DataFrame({n: t.set_index("segment")["bias_pct"]
                        for n, t in results.items()}).to_string())

    print(f"\n{'=' * 72}\nPRIMARY CRITERION: v11 beats the structural baseline\n")
    verdict = {}
    for seg in ("short", "long"):
        d = bootstrap_delta(preds, base, split, profiles, segment=seg)
        verdict[seg] = d
        print(f"  smooth/{seg:<6} {d['delta']:+.4f}  se {d['se']:.4f}  "
              f"95% CI [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]  "
              f"{'SIGNIFICANT' if is_significant(d) else 'not distinguishable from noise'}"
              f"   {'v11 ahead' if d['delta'] < 0 else 'baseline ahead'}")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    payload = {
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": commit, "snapshot": ML_DATA_SNAPSHOT,
        "cutoff": str(split.cutoff.date()),
        "test_weeks": [str(w.date()) for w in sorted(split.test["ds"].unique())],
        "scores": {n: t.set_index("segment")["pooled_wape"].to_dict()
                   for n, t in results.items()},
        "v11_vs_baseline": {k: v for k, v in verdict.items()},
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nrecorded in {RESULT.relative_to(ROOT)}")
    print("Write it up in Section 4.34's result entry whichever way it landed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
