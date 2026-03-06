"""
FastAPI Diabetes Risk Prediction Service.

Run locally:
    uvicorn fastapi_backend.main:app --reload --port 8000

Vercel deployment:
    Served via api/index.py which re-exports this app.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastapi_backend.schemas import PredictionRequest, PredictionResponse
from fastapi_backend.model_loader import load_models, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models at startup."""
    load_models()
    yield


app = FastAPI(
    title="Diabetes Risk Prediction API",
    description="Predicts diabetes risk from health indicators using a trained XGBoost model.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
@app.get("/api")
def root():
    return {"service": "Diabetes Risk Prediction API", "status": "running"}


@app.post("/predict", response_model=PredictionResponse)
@app.post("/api/predict", response_model=PredictionResponse)
def make_prediction(data: PredictionRequest):
    """
    Predict diabetes risk from health indicators.

    Returns risk percentage (0-100) and risk level (Low/Moderate/High).
    """
    result = predict(
        age_group=data.age,
        bmi=data.bmi,
        high_bp=data.bp,
        smoker=data.smoker,
        high_cholesterol=data.cholesterol,
        physical_activity=data.activity,
        general_health=data.health,
        mental_health=data.mental,
    )
    return PredictionResponse(**result)
