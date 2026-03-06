"""
FastAPI Diabetes Risk Prediction Service.

Run locally:
    uvicorn fastapi_backend.main:app --reload --port 8000

Vercel deployment:
    Served via api/index.py which re-exports this app.
"""

import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from fastapi_backend.schemas import PredictionRequest, PredictionResponse
from fastapi_backend.model_loader import load_models, predict

# ── Rate limiting ──────────────────────────────────────────────────────────
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60"))
_request_log: dict[str, list[float]] = defaultdict(list)


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

# ── CORS ───────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ── Rate-limit middleware ──────────────────────────────────────────────────
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = now - 60
    # Prune old entries
    _request_log[client_ip] = [t for t in _request_log[client_ip] if t > window]
    if len(_request_log[client_ip]) >= RATE_LIMIT:
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests. Try again later."},
        )
    _request_log[client_ip].append(now)
    return await call_next(request)


@app.get("/")
def serve_index():
    """Serve the frontend UI."""
    index_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public", "index.html")
    return FileResponse(index_path)


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


# ── Serve public/ static assets (CSS, JS, images if any) ──────────────────
_PUBLIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public")
if os.path.isdir(_PUBLIC_DIR):
    app.mount("/", StaticFiles(directory=_PUBLIC_DIR), name="static")
