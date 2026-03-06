"""
Model loading and prediction logic for the diabetes risk API.

Uses ONNX Runtime for lightweight inference (no xgboost/sklearn needed at runtime).
Isotonic calibration is done via numpy interpolation from saved threshold arrays.
"""

import os
import numpy as np
import onnxruntime as ort

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")

# ── Loaded at import time (module-level singletons) ───────────────────────
_onnx_session = None
_iso_x = None
_iso_y = None


def load_models():
    """Load the ONNX model and isotonic calibration data from disk."""
    global _onnx_session, _iso_x, _iso_y
    _onnx_session = ort.InferenceSession(
        os.path.join(MODEL_DIR, "diabetes_xgboost.onnx")
    )
    cal = np.load(os.path.join(MODEL_DIR, "isotonic_calibration.npz"))
    _iso_x = cal["X_thresholds"]
    _iso_y = cal["y_thresholds"]


def _isotonic_predict(raw_prob: float) -> float:
    """Apply isotonic calibration via numpy interpolation."""
    return float(np.interp(raw_prob, _iso_x, _iso_y))


def build_feature_vector(
    age_group: float,
    bmi: float,
    high_bp: float,
    smoker: float,
    high_cholesterol: float,
    physical_activity: float,
    general_health: float,
    mental_health: float,
) -> np.ndarray:
    """Build a single-row numpy array with all 13 features."""
    return np.array(
        [[
            bmi, age_group, high_bp, smoker, high_cholesterol,
            physical_activity, general_health, mental_health,
            bmi * age_group, bmi * high_bp, age_group * high_bp,
            high_cholesterol * bmi, general_health * bmi,
        ]],
        dtype=np.float32,
    )


def predict(
    age_group: float,
    bmi: float,
    high_bp: float,
    smoker: float,
    high_cholesterol: float,
    physical_activity: float,
    general_health: float,
    mental_health: float,
) -> dict:
    """
    Run inference and return risk percentage + level.
    Uses isotonic calibration for well-calibrated probabilities.
    """
    features = build_feature_vector(
        age_group, bmi, high_bp, smoker, high_cholesterol,
        physical_activity, general_health, mental_health,
    )

    # ONNX returns [labels, probabilities]; probabilities is a list of dicts
    results = _onnx_session.run(None, {"features": features})
    raw_prob = float(results[1][0][1])

    cal_prob = _isotonic_predict(raw_prob)
    risk_pct = round(cal_prob * 100, 1)

    if risk_pct <= 30:
        level = "Low"
    elif risk_pct <= 60:
        level = "Moderate"
    else:
        level = "High"

    return {"risk_percentage": risk_pct, "risk_level": level}
