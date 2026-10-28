"""Cumulant + gradient-boosting baseline.

Gradient boosting on 14 hand-built features is a genuinely competitive
classical approach, and it is the number the CNN has to justify itself
against. If the CNN cannot beat this, the CNN does not belong in the repo.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rfmc.dataset import IQDataset
from rfmc.features import FEATURE_NAMES, extract


def build_model(seed: int = 7) -> Pipeline:
    """Standardise features, then boost."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=400,
                    learning_rate=0.1,
                    max_leaf_nodes=31,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=seed,
                ),
            ),
        ]
    )


def fit(train: IQDataset, sps: int, seed: int = 7) -> tuple[Pipeline, dict[str, Any]]:
    """Extract features, fit the model, and report timing."""
    t0 = time.perf_counter()
    features = extract(train.x, sps)
    feat_time = time.perf_counter() - t0

    model = build_model(seed)
    t0 = time.perf_counter()
    model.fit(features, train.y)
    fit_time = time.perf_counter() - t0

    info = {
        "n_features": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "feature_extraction_seconds": round(feat_time, 2),
        "fit_seconds": round(fit_time, 2),
        "n_train": int(len(train)),
    }
    return model, info


def predict(model: Pipeline, data: IQDataset, sps: int) -> np.ndarray:
    """Class predictions for a dataset."""
    return model.predict(extract(data.x, sps))


def save(model: Pipeline, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load(path: str | Path) -> Pipeline:
    return joblib.load(path)
