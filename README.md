# UPI Shield — Full-Stack UPI Fraud Prediction & Customer Segmentation

A full-stack machine learning web application built with **FastAPI** (backend) and **React + Vite** (frontend) for real-time UPI transaction fraud risk scoring and KMeans customer segmentation.

## Project Architecture

- **ML Pipeline (`backend/train_models.py`, `backend/app/features.py`)**
  - Reproduces the feature engineering from `feature_engineering.ipynb` (`log_amount`, `is_high_value`, `is_late_night`, `time_slot`, `is_weekend`, `bank_device`, `bank_category`).
  - Trains and exports **XGBoost** (`scale_pos_weight`) and **Logistic Regression** (`class_weight="balanced"`) pipelines to `backend/modal/upi_fraud_artifact.joblib`.
  - Integrates the pre-trained **KMeans (K=5)** customer segmentation model (`backend/modal/upi_kmeans_artifact.joblib`).
- **FastAPI Backend (`backend/app/`)**
  - `GET /api/health` — Service and model readiness check
  - `GET /api/metadata` — Categorical dropdown options, model evaluation metrics, dataset stats, and KMeans cluster profiles
  - `POST /api/predict` — Single transaction fraud probability, risk tier (`LOW`/`MEDIUM`/`HIGH`), customer segment, and behavioral risk signals
  - `POST /api/predict/batch` — Batch CSV scoring (up to 5,000 rows)
  - Automatically serves `frontend/dist` when built for production.
- **React Frontend (`frontend/`)**
  - **Live Fraud Predictor**: Interactive form with quick-fill presets, real-time risk gauge, customer segment badge, and risk drivers.
  - **Batch CSV Scoring**: Upload transaction CSVs, inspect flagged rows, and export predictions.
  - **Model & Cluster Analytics**: Dataset KPIs, XGBoost vs Logistic Regression comparison, and 5 KMeans customer segments.

## Local Development

### 1. Backend (FastAPI)
```bash
# Optional: retrain and export fraud models
PYTHONPATH=backend /home/saurabh/anaconda3/bin/python3 backend/train_models.py

# Start FastAPI server
PYTHONPATH=backend /home/saurabh/anaconda3/bin/uvicorn app.main:app --reload --port 8000
```

### 2. Frontend (React + Vite)
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` (Vite proxies `/api` requests to `http://127.0.0.1:8000`).

## Production Deployment

### Option A: Single-Server Production (FastAPI serving React build)
```bash
cd frontend && npm run build && cd ..
PYTHONPATH=backend /home/saurabh/anaconda3/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000`.

### Option B: Docker
```bash
docker build -t upi-fraud-app .
docker run -p 8000:8000 upi-fraud-app
```
