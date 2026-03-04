# Healthcare Risk Prediction — End-to-End ML Pipeline

A production-grade machine learning system for predicting **diabetes risk** using the CDC BRFSS 2015 health survey dataset (441,456 respondents).

---

## Project Highlights

| Metric | Value |
|---|---|
| **ROC AUC** | 0.879 |
| **Brier Score (calibrated)** | 0.034 |
| **Calibration improvement** | 78% lower Brier |
| **Class imbalance handling** | `scale_pos_weight` (no data discarded) |
| **Explainability** | SHAP TreeExplainer |

---

## Repository Structure

```
Healthcare_risk_prediction/
│
├── notebooks/
│   └── brfss_cleaning.ipynb      # Full pipeline: cleaning → training → evaluation → SHAP
│
├── models/
│   ├── diabetes_xgboost.pkl      # Trained XGBoost model
│   ├── isotonic_calibrator.pkl   # Probability calibrator
│   └── shap_explainer.pkl        # SHAP TreeExplainer
│
├── app/
│   └── risk_assistant.py         # CLI + Gradio diabetes risk calculator
│
├── utils/
│   └── feature_engineering.py    # Reusable feature pipeline
│
├── data_raw/                     # Raw BRFSS SAS file (not tracked — 1.1 GB)
├── data_processed/               # Cleaned CSV (not tracked)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ML Pipeline

### 1. Data Cleaning
- Source: [CDC BRFSS 2015](https://www.cdc.gov/brfss/) (`LLCP2015.XPT`, SAS format)
- 9 variables selected, renamed, cleaned (7/9 → NA → drop)
- BMI rescaled from ×100 integer encoding

### 2. Feature Engineering
- 5 interaction features: `bmi_age`, `bmi_bp`, `age_bp`, `chol_bmi`, `health_bmi`
- 13 total features used by the model

### 3. Class Imbalance
- **`scale_pos_weight`** computed as `count(negative) / count(positive)`
- Full training set retained (30,024 rows) — no downsampling

### 4. Models
| Model | ROC AUC |
|---|---|
| Random Forest (baseline) | 0.871 |
| **XGBoost + early stopping** | **0.879** |

XGBoost config: `n_estimators=800, max_depth=6, lr=0.03, subsample=0.8, colsample=0.8, early_stopping_rounds=50`

### 5. Probability Calibration
- **Isotonic regression** maps raw XGBoost scores → real-world probabilities
- Brier score: 0.156 → 0.034 (78% improvement)
- Calibrated probabilities match true prevalence

### 6. Explainability (SHAP)
- `TreeExplainer` for global and per-patient feature attribution
- Summary plot, bar plot, dependence plot (BMI × Age interaction)
- Risk calculator shows top 5 contributing factors per patient

### 7. Risk Calculator Output
```
====================================================
   DIABETES RISK ASSESSMENT
====================================================

   Estimated Diabetes Risk : 0.72  (72.0%)
   Risk Level              : HIGH
   Model Confidence        : calibrated probability
   (Raw model score        : 0.31)

   Top Contributing Factors:
   ------------------------------------------
   + Bmi Age..........................  43.5%
   + Age Group........................  13.2%
   + General Health...................  11.3%
   + Health Bmi.......................   8.8%
   + Chol Bmi.........................   5.1%

====================================================
```

---

## Quick Start

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data
Download [BRFSS 2015 SAS file](https://www.cdc.gov/brfss/annual_data/2015/files/LLCP2015XPT.zip) into `data_raw/`.

### Train
Open and run `notebooks/brfss_cleaning.ipynb` end-to-end.

### Risk Calculator
```bash
# CLI mode
python app/risk_assistant.py

# Gradio web UI (requires: pip install gradio)
python app/risk_assistant.py --ui
```

---

## Tech Stack

- **Python 3.13**
- pandas, NumPy, scikit-learn, XGBoost, SHAP, matplotlib
- Gradio (optional — interactive UI)
- joblib (model serialisation)

---

## Evaluation Strategy

- Stratified 80/20 train-test split
- 5-fold cross-validation (Mean AUC: 0.860 ± 0.010)
- Confusion matrix, precision/recall/F1
- ROC curve comparison (XGBoost vs Random Forest)
- **Reliability diagram** — calibration curve + Brier score
- SHAP global + local explanations

---

## Author

**Aryan Mishra**  
Data Science Student — focused on ML Engineering & Deployment

---

## License

This project is for educational and portfolio purposes.
