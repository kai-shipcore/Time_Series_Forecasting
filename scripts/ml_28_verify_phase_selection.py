#!/usr/bin/env python3
"""Verify the leave-one-window-out result that the Tue-Mon adoption rests on.

Why this exists as a script. The original result was computed inline, on one
CSV, on the TOTAL segment only, with one selection metric, and it was then used
to justify a structural change to the pipeline. That is more weight than an
unreviewed throwaway can carry. This re-derives it with the choices made
explicit and varied, so the conclusion either survives them or does not.

The claim under test
--------------------
"Choosing the week phase on two development windows and paying for it on the
third gives an honest out-of-sample gain of 0.0132 pooled WAPE over Monday,
with selection optimism of 0.0001, and Tuesday is chosen on every fold."

Four things about that were choices rather than findings, and each is varied:

  1. Which sweep. The pinned-profile run and the --reprofile run. The original
     read whichever file existed, which by then was the reprofiled one.
  2. Which rows. TOTAL only, or the segments scored separately.
  3. How windows combine during selection. Unit-weighted pooling (correct for
     pooled WAPE, since sum(wape_i * units_i) / sum(units_i) reconstructs
     sum|e| / sum(y) exactly) or an unweighted mean of window WAPEs, which is
     not the same thing and would let the smallest window count as much as the
     largest.
  4. Whether the Section 1.5 inadmissible cell participates. smooth/short at
     Oct-Dec has 14 eligible SKUs. It is excluded from the segment view. It
     cannot be excluded from TOTAL, whose SKUs are pooled before scoring, and
     that asymmetry is reported rather than hidden.

Also reported, because "optimism is 0.0001" is close to vacuous on its own:
with three folds and the same phase winning all three, selection cannot be
optimistic by construction. What makes the result meaningful is the MARGIN by
which the chosen phase wins on the folds used to choose it, and whether it also
wins on the fold it did not see. Both are printed.

    .venv/bin/python scripts/ml_28_verify_phase_selection.py
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

from ml_26_week_boundary_ab import EXCLUDED_CELLS, _rel  # noqa: E402
from ml_27_week_phase_sweep import OUT, OUT_REPROFILE, PHASES  # noqa: E402

INCUMBENT = "Mon"   # what production would use without this decision


def pooled(sub: pd.DataFrame) -> float:
    """Unit-weighted combination, which for pooled WAPE is exact.

    wape_i = sum|e|_i / sum(y)_i and units_i = sum(y)_i, so
    sum(wape_i * units_i) / sum(units_i) = sum(sum|e|_i) / sum(sum(y)_i).
    """
    return (sub["pooled_wape"] * sub["actual_units"]).sum() / sub["actual_units"].sum()


def unweighted(sub: pd.DataFrame) -> float:
    """Plain mean of window WAPEs. Not pooled WAPE. Included as a robustness
    check, not because it is the right combination."""
    return sub["pooled_wape"].mean()


def loo(d: pd.DataFrame, combine, windows: list[str]) -> dict:
    """Leave-one-window-out phase selection."""
    sel, inc, best, picks, margins = [], [], [], [], []
    for held in windows:
        tr, te = d[d.window != held], d[d.window == held]
        scores = {p: combine(tr[tr.phase == p]) for p in PHASES}
        pick = min(scores, key=scores.get)
        ordered = sorted(scores.values())
        margins.append(ordered[1] - ordered[0])      # winner vs runner-up, on the fold used to choose
        on_held = {p: combine(te[te.phase == p]) for p in PHASES}
        sel.append(on_held[pick])
        inc.append(on_held[INCUMBENT])
        best.append(min(on_held.values()))
        picks.append(pick)
    n = len(windows)
    return {
        "picks": picks,
        "selected": sum(sel) / n,
        "incumbent": sum(inc) / n,
        "hindsight": sum(best) / n,
        "gain": (sum(inc) - sum(sel)) / n,
        "optimism": (sum(sel) - sum(best)) / n,
        "mean_margin": sum(margins) / n,
        "won_held_out": sum(1 for a, b in zip(sel, best) if abs(a - b) < 1e-12),
    }


def main() -> int:
    sources = [("pinned profiles", OUT), ("re-profiled", OUT_REPROFILE)]
    missing = [p for _, p in sources if not p.exists()]
    if missing:
        print("Missing sweep output(s): " + ", ".join(_rel(p) for p in missing))
        print("Run scripts/ml_27_week_phase_sweep.py with and without --reprofile first.")
        return 1

    print(f"{'source':<16}{'rows':<14}{'combine':<12}"
          f"{'picks':<18}{'gain':>9}{'optimism':>10}{'margin':>9}")
    print("-" * 88)

    verdicts = []
    for (label, path), rows, (cname, combine) in product(
        sources, ["TOTAL", "segments"],
        [("unit-wtd", pooled), ("unweighted", unweighted)],
    ):
        d = pd.read_csv(path)
        d = d[d.model == "v11"]
        if rows == "TOTAL":
            d = d[d.segment == "TOTAL"]
        else:
            d = d[d.segment != "TOTAL"]
            d = d[[(w, s) not in EXCLUDED_CELLS
                   for w, s in zip(d.window, d.segment)]]
        windows = sorted(d.window.unique())
        r = loo(d, combine, windows)
        verdicts.append(r)
        print(f"{label:<16}{rows:<14}{cname:<12}"
              f"{'/'.join(r['picks']):<18}{r['gain']:>+9.4f}"
              f"{r['optimism']:>10.4f}{r['mean_margin']:>9.4f}")

    print("\n  gain      = incumbent (Mon) minus the phase chosen without seeing the held-out window")
    print("  optimism  = chosen minus best-in-hindsight on the held-out window")
    print("  margin    = winner vs runner-up on the folds used to choose, averaged")

    all_tue = all(set(v["picks"]) == {"Tue"} for v in verdicts)
    gains = [v["gain"] for v in verdicts]
    opts = [v["optimism"] for v in verdicts]
    print(f"\n  Tuesday chosen in every fold of every variant: {all_tue}")
    print(f"  gain across variants:     {min(gains):+.4f} to {max(gains):+.4f}")
    print(f"  optimism across variants: {min(opts):.4f} to {max(opts):.4f}")

    print("\n  Caveats that survive whatever the numbers say:")
    print("   - three folds, sharing SKUs and overlapping training data, so they")
    print("     are not independent and the optimism estimate has its own error")
    print("     that this design cannot quantify.")
    print("   - the mechanism is unknown, so nothing here says the advantage")
    print("     transfers to data these windows do not cover.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
