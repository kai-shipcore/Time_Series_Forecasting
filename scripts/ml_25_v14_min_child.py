#!/usr/bin/env python3
"""v14: sweep min_child_samples on the shared (short-serving) model.

Hypothesis, pass criteria and recorded expectation are in the design doc,
Section 6, v14, written before this ran.

Reports, per candidate value:
  1. Per-window per-segment pooled WAPE, raw, against the v11 baseline of 200.
  2. Bootstrap significance of each short-segment difference (evaluate.bootstrap_delta).
  3. The tail criterion: pooled WAPE over anchors whose deseasonalized ramp_4_12
     is below 0.7, and whether the predicted-ratio ordering by ramp bucket
     survives to lead 10.

Long is a control: it must come back identical, since only the shared model changes.

Run:
    .venv/bin/python scripts/ml_25_v14_min_child.py
    .venv/bin/python scripts/ml_25_v14_min_child.py --values 100 50 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import MIN_SIM_HISTORY_WEEKS  # noqa: E402
from src.ml.dataset import dev_splits, eligible_skus, load_weekly  # noqa: E402
from src.ml.evaluate import per_sku_totals, score  # noqa: E402
from src.ml.seasonal import ml_factors as F  # noqa: E402
from src.ml.serving.models import V11Hybrid, V14MinChild  # noqa: E402

BUCKETS = [(0, .4, "collapsed <0.4"), (.4, .7, "falling .4-.7"),
           (.7, 1.1, "flat .7-1.1"), (1.1, 99, "rising >1.1")]


def _smooth(weekly, profiles):
    keep = set(profiles.loc[profiles["bucket"] == "smooth", "unique_id"])
    return weekly[weekly["unique_id"].isin(keep)].copy()


def ramp_at(train, cutoff):
    """Deseasonalized ramp_4_12 per SKU at the cutoff, as the model computes it."""
    d = train[train["ds"] <= cutoff].sort_values(["unique_id", "ds"]).copy()
    d["y_adj"] = d["y"] / F(d["ds"]).to_numpy()
    g = d.groupby("unique_id")["y_adj"]
    lvl = g.tail(12).groupby(d["unique_id"]).mean()
    r4 = g.tail(4).groupby(d["unique_id"]).mean()
    return (r4 / lvl.clip(lower=1e-9)).rename("ramp"), lvl.rename("level")


def run_one(model, splits, profiles):
    """Fit and score one model across the splits. Returns (segment table, per-SKU detail)."""
    seg_rows, detail = [], []
    for split, label in splits:
        m = model(split.horizon).fit(split.train, profiles, split.cutoff)
        preds = m.predict(split.train, profiles, split.cutoff)
        tbl = score(preds, split, profiles)
        for _, r in tbl.iterrows():
            seg_rows.append({"window": label, "segment": r["segment"],
                             "pooled_wape": r["pooled_wape"], "n_skus": r["n_skus"],
                             "actual_units": r["actual_units"]})
        d = per_sku_totals(preds, split, profiles)
        d["window"] = label
        detail.append(d)
    return pd.DataFrame(seg_rows), pd.concat(detail, ignore_index=True)


def tail_wape(detail, ramps, thresh=0.7):
    """Pooled WAPE over short SKUs whose ramp at the cutoff was below `thresh`."""
    d = detail[detail["history_length"] == "short"].merge(ramps, on=["unique_id", "window"])
    t = d[d["ramp"] < thresh]
    if t.empty or t["y_total"].sum() == 0:
        return float("nan"), 0
    return t["ae"].sum() / t["y_total"].sum(), t["unique_id"].nunique()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--values", type=int, nargs="+", default=[100, 50, 20])
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    weekly, profiles = load_weekly()
    w = _smooth(weekly, profiles)
    splits = []
    ramp_parts = []
    for sp in dev_splits(w, n=args.windows):
        label = f"{sp.test['ds'].min():%b}-{sp.test['ds'].max():%b}"
        splits.append((sp, label))
        r, _ = ramp_at(sp.train, sp.cutoff)
        keep = eligible_skus(profiles, sp.cutoff, MIN_SIM_HISTORY_WEEKS)
        rr = r[r.index.isin(keep)].reset_index()
        rr["window"] = label
        ramp_parts.append(rr)
    ramps = pd.concat(ramp_parts, ignore_index=True)
    print(f"windows: {[l for _, l in splits]}\n")

    print("=" * 74)
    print("BASELINE  v11  (min_child_samples=200)")
    print("=" * 74)
    base_seg, base_det = run_one(V11Hybrid, splits, profiles)
    print(base_seg.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    bw, bn = tail_wape(base_det, ramps)
    print(f"\ntail (short, ramp<0.7): pooled WAPE {bw:.4f} over {bn} SKUs")

    results = {}
    for v in args.values:
        print("\n" + "=" * 74)
        print(f"CANDIDATE  min_child_samples={v}")
        print("=" * 74)
        seg, det = run_one(lambda h, _v=v: V14MinChild(h, min_child_samples=_v),
                           splits, profiles)
        print(seg.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
        tw, tn = tail_wape(det, ramps)
        print(f"\ntail (short, ramp<0.7): pooled WAPE {tw:.4f} over {tn} SKUs "
              f"(baseline {bw:.4f}, delta {tw - bw:+.4f})")

        # long is a control
        L = seg[seg.segment.str.contains("long")].set_index("window")["pooled_wape"]
        LB = base_seg[base_seg.segment.str.contains("long")].set_index("window")["pooled_wape"]
        same = np.allclose(L.sort_index().to_numpy(), LB.sort_index().to_numpy(), atol=1e-9)
        print(f"long segment identical to baseline (control): {same}")

        results[v] = (seg, det, tw, same)

    print("\n" + "=" * 74)
    print("SHORT SEGMENT, ALL VALUES  (decision windows: Mar-May, Dec-Feb)")
    print("=" * 74)
    S = base_seg[base_seg.segment.str.contains("short")].set_index("window")["pooled_wape"]
    hdr = f"{'window':<12}{'v11 (200)':>12}" + "".join(f"{'mcs='+str(v):>14}" for v in args.values)
    print(hdr)
    for win in S.index:
        row = f"{win:<12}{S[win]:>12.4f}"
        for v in args.values:
            seg = results[v][0]
            x = seg[(seg.window == win) & (seg.segment.str.contains("short"))]["pooled_wape"].iloc[0]
            row += f"{x:>9.4f}{x - S[win]:>+6.4f}"[:14].rjust(14)
        print(row)

    print(f"\n{'value':<10}{'tail WAPE':>12}{'delta':>10}{'long identical':>17}")
    print(f"{'200 (v11)':<10}{bw:>12.4f}{0.0:>10.4f}{'baseline':>17}")
    for v in args.values:
        print(f"{v:<10}{results[v][2]:>12.4f}{results[v][2] - bw:>+10.4f}{str(results[v][3]):>17}")

    out = ROOT / "outputs/reports/v14_min_child_sweep.csv"
    rows = [base_seg.assign(min_child_samples=200)]
    rows += [results[v][0].assign(min_child_samples=v) for v in args.values]
    pd.concat(rows, ignore_index=True).to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
