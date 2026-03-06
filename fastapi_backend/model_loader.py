"""
Model loading and prediction logic for the diabetes risk API.
"""

import os
import numpy as np
import pandas as pd
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")

# ── Loaded at import time (module-level singletons) ───────────────────────
_xgb_model = None
_iso_reg = None


def load_models():
    """Load the trained XGBoost model and isotonic calibrator from disk."""
    global _xgb_model, _iso_reg
    _xgb_model = joblib.load(os.path.join(MODEL_DIR, "diabetes_xgboost.pkl"))
    _iso_reg = joblib.load(os.path.join(MODEL_DIR, "isotonic_calibrator.pkl"))


def build_feature_vector(
    age_group: float,
    bmi: float,
    high_bp: float,
    smoker: float,
    high_cholesterol: float,
    physical_activity: float,
    general_health: float,
    mental_health: float,
) -> pd.DataFrame:
    """Build a single-row DataFrame with all 13 features."""
    features = {
        "bmi": bmi,
        "age_group": age_group,
        "high_bp": high_bp,
        "smoker": smoker,
        "high_cholesterol": high_cholesterol,
        "physical_activity": physical_activity,
        "general_health": general_health,
        "mental_health": mental_health,
        "bmi_age": bmi * age_group,
        "bmi_bp": bmi * high_bp,
        "age_bp": age_group * high_bp,
        "chol_bmi": high_cholesterol * bmi,
        "health_bmi": general_health * bmi,
    }
    return pd.DataFrame([features]).astype(np.float64)


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
    df = build_feature_vector(
        age_group, bmi, high_bp, smoker, high_cholesterol,
        physical_activity, general_health, mental_health,
    )

    raw_prob = _xgb_model.predict_proba(df)[:, 1][0]
    cal_prob = float(_iso_reg.predict([raw_prob])[0])
    risk_pct = round(cal_prob * 100, 1)

    if risk_pct <= 30:
        level = "Low"
    elif risk_pct <= 60:
        level = "Moderate"
    else:
        level = "High"

    return {"risk_percentage": risk_pct, "risk_level": level}
