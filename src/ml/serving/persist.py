"""Save and load a fitted model so serving is decoupled from training.

A fitted model holds only trees and configuration (no training data), so the
bundle is small. predict() rebuilds its feature matrix from whatever history it
is given at call time. A JSON sidecar records what was trained, for inspection.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.ml.serving.base import ForecastModel


def save_model(model: ForecastModel, path: str | Path) -> Path:
    """Pickle a fitted model to `path` and write a `.json` metadata sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    meta = {
        "version": model.version,
        "description": model.description,
        "horizon": model.horizon,
        "cutoff": None if model.cutoff is None else str(model.cutoff.date()),
        "n_long_uids": len(model.long_uids),
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return path


def load_model(path: str | Path) -> ForecastModel:
    """Load a fitted model saved by save_model()."""
    return joblib.load(Path(path))
