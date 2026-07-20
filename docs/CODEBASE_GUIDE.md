# LightGBM Forecasting: Codebase and Pipeline Guide

**Purpose:** a developer reference for the machine-learning forecasting track. It explains
every file, how data moves from raw sales to a scored accuracy table, and how the code
works.

**Relationship to the design document:** `ML_FORECAST_DESIGN.md` records what was decided
and why. This document records where the code lives and how it runs. When a design decision
is referenced here, it is cited by its number (for example, Design Section 4.10).

**Status:** the harness files (`dataset.py`, `evaluate.py`) are stable and documented in
full below. The model files (`model.py`, `features.py`) are being restructured for the
restart baseline (Design Section 4.10) and will be documented in full once that work
lands. Their current role is summarized in the file inventory.

---

## 1. File Inventory

All model code lives in `src/ml/`. All runnable experiments live in `scripts/ml_*.py`. The
experiments import from `src/ml/` and add nothing to it, so the library stays reusable.

| File | Lines | Role |
|---|---|---|
| `src/ml/__init__.py` | 8 | Package marker and a short description of the layout. |
| `src/ml/dataset.py` | ~230 | Loads the pinned snapshot data, applies the shared cleaning rules, builds rolling-origin train/test splits, and selects the stratified validation SKUs. Fully documented below. |
| `src/ml/evaluate.py` | ~127 | Scores any model's predictions into a per-segment pooled-WAPE and bias table, guards against data leakage, and provides the bootstrap significance test. Fully documented below. |
| `src/ml/model.py` | ~200 | The current model code (rewritten at the restart, Design Section 4.10): the per-segment seasonal adjustment, the training-matrix builder, the `structural_baseline` prediction function, and `RatioLGBM` (the LightGBM wrapper: fit, predict, feature importance). All modeling continues from this file; pre-restart model code was deleted. |
| `scripts/ml_snapshot_data.py` | ~200 | Creates a dated, read-only copy of the two ML inputs under `data/snapshots/<date>/` with a checksum manifest, and verifies an existing snapshot against it (`--verify`). Supports Design Section 4.21. |
| `scripts/ml_00_harness_check.py` | ~63 | Runs the baseline moving averages (WA12, WA8, naive) through the harness to confirm the scorer reproduces known production numbers. Contains no machine learning; retained as evidence for Design Section 4.2. |
| `scripts/ml_02_v1_benchmark.py` | ~80 | Runs the legacy V1 formula through the harness for reference, using the production V1 implementation imported from `scripts/compare_v1.py`. |
| `scripts/ml_03_baseline_deseas.py` | ~90 | The restart baseline experiment: deseasonalized WA12 versus raw WA12 (Design Section 4.17 evidence). |
| `scripts/ml_04_alpha_search.py` | ~85 | Grid search for the age-damping exponent; concluded no seasonal adjustment for short SKUs (Design Section 4.17 evidence). |
| `scripts/ml_05_lgbm_v0.py` | ~80 | LightGBM v0 (lead-only feature) versus the structural baseline; rejected (Design Section 4.18). |
| `scripts/ml_06_lgbm_v1.py` | ~95 | LightGBM v1 (lead + ramp block) versus v0 and baseline, with the learned-response probe; rejected (Design Section 4.19). |
| `scripts/ml_07_lgbm_v2.py` | ~85 | LightGBM v2 (deseasonalized trajectory features) versus v1 and baseline; rejected (Design Section 4.20). |
| `scripts/ml_08_lgbm_v3.py` | ~80 | LightGBM v3 (fully deseasonalized ML path) versus v2 and baseline; closest yet, blocked by a long-segment regression (Design Section 4.20). |

Supporting scripts outside `src/ml/` that the track depends on:

| File | Role |
|---|---|
| `scripts/export_forecast_history.py` | Exports forecast tables from the database to `data/processed/*.parquet` for offline analysis. Discovers table schemas at runtime. |
| `scripts/compare_v1.py` | The production V1 formula. `ml_02` imports its `v1_forecast` and cumulative-sum index functions so the benchmark uses the exact production logic. |
| `config.py` | Central configuration (paths, `TEST_WEEKS`, `TRIM_TRAILING_WEEKS`, segmentation thresholds, seasonal settings). The harness reads split-related settings from here. |
| `src/deseasonalize.py` | Provides the production seasonal factors. The restart baseline (Design Section 4.10) will use these to deseasonalize the target. |

---

## 2. End-to-End Pipeline Flow

The flow from raw data to a scored result has four stages. Each stage is one function or
object, and the data passed between them is always a plain pandas DataFrame.

```
data/snapshots/<ML_DATA_SNAPSHOT>/sales_clean.parquet   (weekly sales per SKU)
data/snapshots/<ML_DATA_SNAPSHOT>/sku_profiles.csv      (segment labels per SKU)
             │        pinned copies; the weekly cron refreshes data/processed/
             │        only, so ML results do not move between runs (Design 4.21)
             ▼
  [1] dataset.load_weekly()          load + apply shared cleaning rules
             │   returns (weekly, profiles)
             ▼
  [2] dataset.dev_splits(weekly)     build rolling-origin train/test windows
             │   returns [Split, Split, Split]
             ▼
      for each Split:
             │
             ├─ dataset.stratified_val_skus(split.train, profiles)   pick validation SKUs
             │
             ├─ [3] the model under test:
             │        fit on split.train  →  predict the test weeks
             │        returns predictions: unique_id / ds / yhat
             │
             ▼
  [4] evaluate.score(predictions, split, profiles)
                 returns a per-segment table: n_skus, actual_units, pooled_wape, bias_pct
```

Every model in the project, whether it is a moving average, the V1 formula, or LightGBM,
plugs into the same stages 1, 2, and 4. Only stage 3 changes. This is what makes the
comparison fair: the data, the splits, and the scoring are identical across models, so any
difference in the result comes from the model alone.

---

## 3. `src/ml/dataset.py`: Loading, Splitting, Validation Selection

This file has four public pieces: `load_weekly`, the `Split` object, the split builders
(`make_splits`, `dev_splits`, `final_test_split`), and `stratified_val_skus`.

### 3.1 `data_dir()` and `load_weekly()`

```python
def data_dir(snapshot: str | None = ML_DATA_SNAPSHOT) -> Path:
    if snapshot is None:
        return DATA_PROCESSED
    path = DATA_SNAPSHOTS / str(snapshot)
    ...   # raises FileNotFoundError naming available snapshots if absent
    return path
```

`data_dir` decides where the ML track reads from. With `ML_DATA_SNAPSHOT` set (currently
2026-07-20) it returns the pinned snapshot directory; with it set to `None` it falls back
to the live processed files. A missing or incomplete snapshot raises immediately, listing
the snapshots that do exist, so a misconfiguration cannot quietly fall through to live
data. This is the data-content half of reproducibility, complementing the window anchor of
Design Section 4.14; see Design Section 4.21 for why both pins are needed. Production code
does not call this function and is unaffected.

```python
def load_weekly(snapshot: str | None = ML_DATA_SNAPSHOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    src = data_dir(snapshot)
    weekly = pd.read_parquet(src / "sales_clean.parquet")
    profiles = pd.read_csv(src / "sku_profiles.csv")
    weekly["ds"] = pd.to_datetime(weekly["ds"])
    profiles["train_start"] = pd.to_datetime(profiles["train_start"])
```

It reads the two artifacts. `weekly` has three columns: `unique_id` (the SKU),
`ds` (the week, labeled by the Monday it ends on), and `y` (units sold that week). It is a
complete grid, so every SKU has a row for every week, with zero where there were no sales.
`profiles` has one row per SKU with its segment labels and its `train_start` date.

```python
    if TRIM_TRAILING_WEEKS:
        keep = sorted(weekly["ds"].unique())[:-TRIM_TRAILING_WEEKS]
        weekly = weekly[weekly["ds"].isin(keep)]
```

This drops the most recent `TRIM_TRAILING_WEEKS` weeks (currently 0). Recent weeks can be
unreliable because orders register late. The setting is read from `config.py`, the same
value the statistical prototype uses.

```python
    starts = profiles.set_index("unique_id")["train_start"]
    weekly = weekly.copy()
    weekly["_ts"] = weekly["unique_id"].map(starts)
    weekly = weekly[weekly["_ts"].isna() | (weekly["ds"] >= weekly["_ts"])]
    weekly = weekly.drop(columns="_ts").sort_values(["unique_id", "ds"])
    return weekly.reset_index(drop=True), profiles
```

This removes the pre-launch weeks of SKUs that started partway through the data window. For
each row it looks up that SKU's `train_start` and keeps the row only if its week is on or
after that date. A SKU with no recorded `train_start` is kept in full (the `isna` check).
Pre-launch weeks are zeros that are not real demand history, and any average or lag
computed over them would be wrong. Both cleaning rules match `src/backtest.py` exactly, so
the LightGBM track and the statistical prototype always train on identical data.

### 3.2 The `Split` object

```python
@dataclass
class Split:
    cutoff: pd.Timestamp   # last TRAINING week (inclusive)
    train: pd.DataFrame    # all rows with ds <= cutoff
    test: pd.DataFrame     # the `horizon` weeks strictly after cutoff
    horizon: int
```

A `Split` is one backtest scenario. `cutoff` is the last week the model is allowed to see.
`train` is everything up to and including the cutoff. `test` is the block of weeks
immediately after it. The rule the whole project depends on is that a model may use only
data with `ds <= cutoff`; `evaluate.score` re-checks this later so a mistake cannot pass
silently.

### 3.3 `make_splits`, `dev_splits`, `final_test_split`

```python
def make_splits(weekly, horizon=TEST_WEEKS, n_splits=1, anchor=None) -> list[Split]:
    all_weeks = [pd.Timestamp(w) for w in sorted(weekly["ds"].unique())]
    if anchor is None:
        anchor_idx = len(all_weeks) - horizon - 1     # legacy: anchor on latest data
    else:
        anchor = pd.Timestamp(anchor)
        if anchor not in all_weeks:
            raise ValueError(...)                      # anchor date must exist in the data
        anchor_idx = all_weeks.index(anchor)
    splits = []
    for i in range(n_splits):
        cutoff_idx = anchor_idx - i * horizon
        start = cutoff_idx + 1                         # first test week
        end = start + horizon                          # one past the last test week
        if cutoff_idx < 0 or end > len(all_weeks):
            raise ValueError(...)                      # window runs off either end of the data
        cutoff = all_weeks[cutoff_idx]
        test_weeks = all_weeks[start:end]
        splits.append(Split(cutoff, weekly[weekly.ds <= cutoff].copy(),
                            weekly[weekly.ds.isin(test_weeks)].copy(), horizon))
    return splits
```

`make_splits` produces rolling-origin windows, most recent first. `anchor` is the last
training week of split 0. Windows are built by stepping the cutoff back `horizon` weeks per
split, so the anchor pins every window to a fixed date. This matters because the source
data refreshes weekly (Design Section 4.14): without a fixed anchor, every refresh would
shift all windows and make earlier results irreproducible. Weeks in the data after split
0's test window are ignored. When `anchor` is `None`, the function falls back to anchoring
on the latest week in the data. If a window would run off either end of the data, it raises
an error rather than returning a malformed split.

Two thin wrappers enforce the evaluation protocol (Design Section 2.2):

```python
def final_test_split(weekly) -> Split:
    return make_splits(weekly, n_splits=1, anchor=ML_FINAL_TEST_CUTOFF)[0]

def dev_splits(weekly, n=3) -> list[Split]:
    return make_splits(weekly, n_splits=n + 1, anchor=ML_FINAL_TEST_CUTOFF)[1:]
```

Both wrappers pass the pinned anchor from `config.py` (`ML_FINAL_TEST_CUTOFF`, currently
2026-05-04). `final_test_split` returns only the anchored final window, which is reserved
for the single final evaluation. `dev_splits` returns the windows before it, which is where
all iteration happens. Because experiments call `dev_splits`, they cannot accidentally
touch the quarantined window.

### 3.4 `stratified_val_skus`

This selects the SKUs held out inside training for LightGBM's early stopping (Design
Sections 2.3 and 4.13). The goal is a held-out set that represents the whole portfolio, not
a random draw that might miss the high-volume SKUs.

```python
    vol = weekly.groupby("unique_id")["y"].mean()
    seg = (profiles.set_index("unique_id")["history_length"]
           .reindex(vol.index)
           .map({"short": "short", "medium": "long", "full": "long"})
           .fillna("short"))
    df = pd.DataFrame({"uid": vol.index, "vol": vol.to_numpy(), "seg": seg.to_numpy()})
```

It computes each SKU's average weekly demand as a volume measure, and labels each SKU as
`short` or `long` (merging medium and full into long, matching the reporting segments).

```python
    rng = np.random.default_rng(seed)
    chosen = []
    for _, g in df.groupby("seg"):
        g = g.copy()
        if len(g) >= n_tiers:
            g["tier"] = pd.qcut(g["vol"].rank(method="first"), n_tiers, labels=False)
        else:
            g["tier"] = 0
        for _, cell in g.groupby("tier"):
            n = min(len(cell), max(min_per_cell, round(len(cell) * frac)))
            chosen.extend(rng.choice(cell["uid"].to_numpy(), size=n, replace=False))
    return set(chosen)
```

Within each segment it splits the SKUs into `n_tiers` volume tiers (terciles by default).
`pd.qcut` on the rank produces equal-sized tiers even when many SKUs share the same volume.
From each of the six resulting cells (two segments times three tiers) it draws a
proportional `frac` (15%), never fewer than `min_per_cell` SKUs. The result is a set of SKU
IDs. Because the draw is proportional within every cell, the held-out set mirrors the
portfolio's segment and volume mix; because it is drawn per cell, no cell can be missed by
chance.

### 3.5 `eligible_skus` and `asof_history_length`

These two functions make the scoring "as of" the cutoff rather than as of today (Design
Section 4.15). Both are used automatically inside `evaluate.score`.

```python
def eligible_skus(profiles, cutoff, min_weeks=MIN_SIM_HISTORY_WEEKS) -> set:
    ts = pd.to_datetime(profiles.set_index("unique_id")["train_start"])
    weeks_at_cutoff = (pd.Timestamp(cutoff) - ts).dt.days / 7
    return set(ts.index[weeks_at_cutoff >= min_weeks])
```

`eligible_skus` returns the SKUs that had at least `min_weeks` (13) weeks of history at the
cutoff, computed from each SKU's `train_start`. A backtest window must not score a SKU that
had almost no history at the time, or that had not launched yet; doing so scores it as a
near-zero forecast against its later real demand and inflates the error. This mirrors the
statistical prototype's per-window inclusion rule.

```python
def asof_history_length(profiles, cutoff) -> pd.Series:
    ts = pd.to_datetime(profiles.set_index("unique_id")["train_start"])
    weeks = (pd.Timestamp(cutoff) - ts).dt.days / 7
    return pd.cut(weeks, bins=[-inf, SHORT_HISTORY_WEEKS, MEDIUM_HISTORY_WEEKS, inf],
                  labels=["short", "medium", "full"], right=False)
```

`asof_history_length` returns each SKU's history-length label as it was at the cutoff, using
the same 50 and 104-week thresholds as the profiler. Because a SKU only gains history over
time, its label today is always at least as long as it was at any past cutoff, so scoring by
today's label would push SKUs that were young at the time into the wrong segment. Note that
with only two years of data, no SKU reaches 104 weeks at any cutoff, so the "full" label
does not appear in backtest windows.

The `train_start` field both functions rely on is a stable per-SKU property (the SKU's
launch or ramp-start week), which is why using it for an as-of calculation is valid. The
`bucket` label (smooth versus intermittent) is not recomputed as of the cutoff; that is a
documented limitation.

---

## 4. `src/ml/evaluate.py`: Scoring, Leakage Guard, Significance

This file is model-agnostic. It never knows which model produced the predictions, which is
why the same code can score a moving average, V1, and LightGBM identically.

### 4.1 `score`

```python
def score(preds, split, profiles) -> pd.DataFrame:
    test_weeks = set(split.test["ds"].unique())
    bad = preds.loc[~preds["ds"].isin(test_weeks), "ds"].unique()
    if len(bad):
        raise ValueError(f"Predictions contain ds outside the test window: ...")
```

`preds` is a DataFrame with `unique_id`, `ds`, and `yhat`. The first thing `score` does is
reject any prediction dated outside the split's test window. This is the leakage guard. If
a model accidentally predicted a training week, or a horizon was misaligned by one week,
the run stops here instead of producing a wrong but plausible number.

After the leakage guard, `score` restricts both the predictions and the test actuals to the
SKUs returned by `eligible_skus` (Section 3.5), unless `min_history_weeks=None` is passed.
This is on by default so an experiment cannot forget it. The segment each SKU is reported
under comes from `asof_history_length`, not from the snapshot label, with medium and full
reported together as `long` (Design Section 4.4). If any eligible SKU has no predictions,
`score` emits a warning naming the count and example SKUs, then scores them as zero
forecasts; a silent gap usually indicates an upstream bug rather than a modeling choice.
`bootstrap_delta` accepts `segment='short'` or `'long'` and uses the same as-of labels.

```python
    yhat_tot = preds.groupby("unique_id")["yhat"].sum().rename("yhat_total")
    y_tot = split.test.groupby("unique_id")["y"].sum().rename("y_total")
    df = pd.concat([yhat_tot, y_tot], axis=1).fillna(0.0).reset_index()
    df = df.merge(profiles[["unique_id", "bucket", "history_length"]],
                  on="unique_id", how="left")
    df["ae"] = (df["yhat_total"] - df["y_total"]).abs()
```

It sums each SKU's forecast and actual over the test window to get one forecast total and
one actual total per SKU, attaches the segment labels, and computes the absolute error of
each SKU's total. Summing to totals first is the production convention: the metric measures
the whole-window forecast, not each week separately.

```python
    def _row(g, label) -> dict:
        actual = g["y_total"].sum()
        return {
            "segment": label,
            "n_skus": len(g),
            "actual_units": round(actual, 1),
            "pooled_wape": round(g["ae"].sum() / max(actual, 1e-9), 4),
            "bias_pct": round((g["yhat_total"].sum() / max(actual, 1e-9) - 1) * 100, 1),
        }
    rows = [_row(g, f"{b}/{h}") for (b, h), g in df.groupby(["bucket", "history_length"])]
    rows.append(_row(df, "TOTAL"))
    return pd.DataFrame(rows)
```

`_row` computes the two headline numbers for any group of SKUs. Pooled WAPE is the sum of
absolute errors divided by the sum of actual demand, so high-volume SKUs carry more weight
(Design Section 1.3). Bias is total forecast over total actual, minus one, expressed as a
percent, so a positive value means over-forecasting. The function builds one row per
segment plus a `TOTAL` row, and returns the table. The `max(actual, 1e-9)` avoids division
by zero if a segment happens to have no demand.

### 4.2 `bootstrap_delta`

This answers whether a WAPE difference between two models is real or could be an artifact of
which SKUs happen to be in the segment (Design Sections 1.5 and 4.12).

```python
    y_tot = split.test.groupby("unique_id")["y"].sum()
    a_tot = preds_a.groupby("unique_id")["yhat"].sum()
    b_tot = preds_b.groupby("unique_id")["yhat"].sum()
    df = pd.concat([y_tot.rename("y"), a_tot.rename("a"), b_tot.rename("b")], axis=1).fillna(0.0)
    if segment is not None:
        seg_map = profiles.set_index("unique_id")["history_length"]
        df = df[df.index.map(seg_map) == segment]
```

It builds a per-SKU table of actual, model A's total, and model B's total, optionally
restricted to one segment.

```python
    ae_a = (df["a"] - df["y"]).abs().to_numpy()
    ae_b = (df["b"] - df["y"]).abs().to_numpy()
    y = df["y"].to_numpy()
    point = ae_a.sum() / y.sum() - ae_b.sum() / y.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(df), size=(n_boot, len(df)))
    deltas = (ae_a[idx].sum(1) - ae_b[idx].sum(1)) / y[idx].sum(1)
    return {"delta": ..., "se": ..., "ci_lo": ..., "ci_hi": ...}
```

`point` is the actual WAPE difference (negative means A is better). The bootstrap then draws
1,000 resamples of the SKU population with replacement. `idx` is a 1,000-by-N matrix of
random SKU indices; each row is one resampled portfolio. The draw is paired: both models are
scored on the same resampled SKUs, so noise common to both cancels and only the difference
is measured. The spread of the 1,000 resampled differences (`se`) is how much the WAPE gap
moves purely from SKU sampling. A gap much larger than this spread is real; a gap smaller
than it is not distinguishable from chance.

### 4.3 `ramp_cohort` and `cohort_score`

These implement the growth-tracking diagnostic (Design Section 1.4) inside the harness, so
every experiment reports it identically.

```python
def ramp_cohort(split, profiles, ratio=1.2, min_level=1.0) -> set:
    g = split.train.sort_values("ds").groupby("unique_id")["y"]
    m4 = g.apply(lambda s: s.tail(4).mean())
    m12 = g.apply(lambda s: s.tail(12).mean())
    growing = set(m12.index[(m12 >= min_level) & (m4 >= ratio * m12)])
    return growing & eligible_skus(profiles, split.cutoff, MIN_SIM_HISTORY_WEEKS)
```

`ramp_cohort` returns the SKUs that were visibly growing at the cutoff: trailing 4-week
average at least 1.2 times the trailing 12-week average, with the 12-week average at least
1 unit per week to exclude near-zero noise. Both averages use training data only, and the
standard eligibility filter is applied. This cohort is where trailing-average methods
underforecast most severely, so it is the targeted test of growth tracking.

`cohort_score` computes pooled WAPE and bias for one model restricted to one SKU set, using
the same window-total conventions as `score`, and returns a dict
(`n_skus`, `actual_units`, `pooled_wape`, `bias_pct`) for compact printing. One reading
note: in the window whose cutoff falls in December, most of the portfolio qualifies as
"ramping" because demand is seasonally rising into Q4, so that window's cohort is large and
less informative than the others.

### 4.4 `score_table`

```python
def score_table(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ...
    return out.pivot_table(index="segment", columns="model", values="pooled_wape").round(4)
```

A formatting helper. Given a dictionary mapping model names to their `score` tables, it
returns one comparison table with segments as rows and models as columns, showing pooled
WAPE. This is the side-by-side view printed at the end of an experiment.

---

## 5. Model Files and Experiment Scripts

`src/ml/model.py` and `src/ml/features.py` are being restructured for the restart baseline
(Design Section 4.10), which changes the target definition and removes the seasonal
features. They will be documented here in full once that work is complete, to avoid
describing code that is about to change. Until then, the design document (Sections 4.5
through 4.10) describes their intended behavior.

The experiment scripts (`scripts/ml_00`, `ml_01`, `ml_02`) each follow the same shape: load
the data, build the development splits, run one or more models, and print the per-segment
tables through `evaluate.score` and `score_table`. They will also be documented in full
alongside the model files.
