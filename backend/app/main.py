"""FastAPI application entrypoint for UPI Fraud Prediction & Segmentation."""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.predictor import FraudPredictorService
from app.schemas import BatchPredictionResponse, PredictionResponse, TransactionInput

predictor: FraudPredictorService | None = None

FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global predictor
    predictor = FraudPredictorService()
    yield
    predictor = None


app = FastAPI(
    title="UPI Fraud Prediction & Segmentation API",
    description="Full-stack ML inference API powered by XGBoost, Logistic Regression, and KMeans.",
    version="1.0.0",
    lifespan=lifespan,
)

cors_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in cors_origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_predictor() -> FraudPredictorService:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model service is not initialized.")
    return predictor


@app.get("/api/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "models_loaded": predictor is not None,
    }


@app.get("/api/metadata")
def get_metadata() -> dict:
    return _get_predictor().get_metadata()


@app.post("/api/predict", response_model=PredictionResponse)
def predict_transaction(payload: TransactionInput) -> PredictionResponse:
    return _get_predictor().predict_single(payload)


@app.post("/api/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    file: UploadFile = File(...),
    model_type: str = Query(default="xgboost", pattern="^(xgboost|logistic_regression)$"),
) -> BatchPredictionResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a valid .csv file.")

    raw_bytes = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {exc}") from exc

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV contains no rows.")

    try:
        return _get_predictor().predict_batch(df, model_type=model_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# Serve built React frontend in production if frontend/dist exists
if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found.")
        target = FRONTEND_DIST / full_path
        if full_path and target.is_file():
            return FileResponse(str(target))
        return FileResponse(str(FRONTEND_DIST / "index.html"))
