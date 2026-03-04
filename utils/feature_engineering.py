"""
Feature engineering utilities for the BRFSS diabetes risk model.

Centralises all column selection, renaming, cleaning, and interaction-term
creation so the same transformations can be applied in both the training
notebook and the production inference path.
"""

import pandas as pd
import numpy as np

# ── Raw BRFSS columns we use ───────────────────────────────────────────────
BRFSS_COLUMNS = [
    "DIABETE3",
    "_BMI5",
    "_AGEG5YR",
    "BPHIGH4",
    "SMOKE100",
    "_RFCHOL",
    "_TOTINDA",
    "GENHLTH",
    "MENTHLTH",
]

RENAME_MAP = {
    "DIABETE3": "diabetes",
    "_BMI5": "bmi",
    "_AGEG5YR": "age_group",
    "BPHIGH4": "high_bp",
    "SMOKE100": "smoker",
    "_RFCHOL": "high_cholesterol",
    "_TOTINDA": "physical_activity",
    "GENHLTH": "general_health",
    "MENTHLTH": "mental_health",
}

FEATURE_COLS = [
    "bmi", "age_group", "high_bp", "smoker", "high_cholesterol",
    "physical_activity", "general_health", "mental_health",
    "bmi_age", "bmi_bp", "age_bp", "chol_bmi", "health_bmi",
]


def select_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    """Select the BRFSS columns we need and rename to friendly names."""
    return df[BRFSS_COLUMNS].rename(columns=RENAME_MAP)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standard BRFSS cleaning pipeline:
      1. Scale BMI (raw value is ×100).
      2. Keep only diabetes = 1 (yes) or 2 (no), recode to 1/0.
      3. Replace 7 / 9 (don't know / refused) with NA, then drop.
    """
    df = df.copy()
    df["bmi"] = df["bmi"] / 100
    df = df[df["diabetes"].isin([1, 2])]
    df["diabetes"] = df["diabetes"].replace({1: 1, 2: 0})
    df = df.replace([7, 9], pd.NA).dropna()
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the five interaction terms used by the model."""
    df = df.copy()
    df["bmi_age"]    = df["bmi"] * df["age_group"]
    df["bmi_bp"]     = df["bmi"] * df["high_bp"]
    df["age_bp"]     = df["age_group"] * df["high_bp"]
    df["chol_bmi"]   = df["high_cholesterol"] * df["bmi"]
    df["health_bmi"] = df["general_health"] * df["bmi"]
    return df


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
    """
    Build a single-row DataFrame with all 13 features from raw clinical inputs.
    Used at inference time (risk calculator / API).
    """
    features = {
        "bmi":               bmi,
        "age_group":         age_group,
        "high_bp":           high_bp,
        "smoker":            smoker,
        "high_cholesterol":  high_cholesterol,
        "physical_activity": physical_activity,
        "general_health":    general_health,
        "mental_health":     mental_health,
        "bmi_age":           bmi * age_group,
        "bmi_bp":            bmi * high_bp,
        "age_bp":            age_group * high_bp,
        "chol_bmi":          high_cholesterol * bmi,
        "health_bmi":        general_health * bmi,
    }
    return pd.DataFrame([features]).astype(np.float64)
