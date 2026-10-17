"""Inference layer: loads the trained artifacts once and exposes predict()."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import FEATURES

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# --- Configuration (override with environment variables) -------------------
# Which class label means "malignant" in YOUR training data.
# Common encodings: M=1 / B=0 (default here) or sklearn's load_breast_cancer (0=malignant).
MALIGNANT_CLASS = os.getenv("MALIGNANT_CLASS", "1")
# Decision threshold on P(malignant). Lower it to favour recall.
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
MODEL_PATH = os.getenv("MODEL_PATH")  # optional explicit path


def find_model_path() -> Path | None:
    if MODEL_PATH:
        p = Path(MODEL_PATH)
        return p if p.exists() else None
    for pattern in ("*.joblib", "*.pkl"):
        for p in sorted(MODELS_DIR.glob(pattern)):
            if "scaler" not in p.stem.lower():
                return p
    return None


def _find_scaler() -> Path | None:
    for pattern in ("*scaler*.joblib", "*scaler*.pkl"):
        for p in sorted(MODELS_DIR.glob(pattern)):
            return p
    return None


@lru_cache(maxsize=1)
def get_artifacts():
    path = find_model_path()
    if path is None:
        raise FileNotFoundError(
            f"No model found in {MODELS_DIR}. Save a fitted sklearn Pipeline with "
            "joblib.dump(pipeline, 'models/model.joblib') or set MODEL_PATH."
        )
    model = joblib.load(path)
    # If the model is not a Pipeline, optionally apply a separately saved scaler.
    scaler = None
    if not hasattr(model, "steps"):
        sp = _find_scaler()
        scaler = joblib.load(sp) if sp else None
    log.info("Loaded model %s (%s)", path.name, type(model).__name__)
    return path, model, scaler


def _validate(features: dict) -> None:
    missing = [f for f in FEATURES if f not in features]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    for f in FEATURES:
        v = features[f]
        if v is None or not np.isfinite(float(v)) or float(v) < 0:
            raise ValueError(f"Invalid value for {f}: {v!r} (must be a finite number >= 0)")


def _to_frame(model, features: dict):
    row = [[float(features[f]) for f in FEATURES]]
    names = getattr(model, "feature_names_in_", None)
    if names is not None and len(names) == len(FEATURES):
        # Same order as training, whatever the training column names were.
        return pd.DataFrame(row, columns=list(names))
    return np.array(row)


def predict(features: dict) -> dict:
    _validate(features)
    _, model, scaler = get_artifacts()
    x = _to_frame(model, features)
    if scaler is not None:
        x = scaler.transform(x)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        classes = [str(c) for c in model.classes_]
        if MALIGNANT_CLASS not in classes:
            raise RuntimeError(
                f"MALIGNANT_CLASS={MALIGNANT_CLASS!r} not in model classes {classes}"
            )
        p_mal = float(proba[classes.index(MALIGNANT_CLASS)])
    else:  # models without probabilities
        p_mal = float(str(model.predict(x)[0]) == MALIGNANT_CLASS)

    label = "Malignant" if p_mal >= THRESHOLD else "Benign"
    return {
        "label": label,
        "malignant_probability": round(p_mal, 4),
        "benign_probability": round(1 - p_mal, 4),
        "threshold": THRESHOLD,
    }


def model_info() -> dict:
    path, model, scaler = get_artifacts()
    return {
        "model_file": path.name,
        "model_type": type(model).__name__,
        "n_features": len(FEATURES),
        "features": FEATURES,
        "classes": [str(c) for c in getattr(model, "classes_", [])],
        "malignant_class": MALIGNANT_CLASS,
        "threshold": THRESHOLD,
        "separate_scaler": scaler is not None,
    }
