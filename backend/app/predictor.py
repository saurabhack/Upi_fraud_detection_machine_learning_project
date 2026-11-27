"""ML model loading and inference service for UPI fraud detection and customer segmentation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.features import CLUSTER_COLUMNS, engineer_features
from app.schemas import BatchPredictionResponse, PredictionResponse, RiskFactor, TransactionInput

BACKEND_DIR = Path(__file__).resolve().parent.parent
MODAL_DIR = BACKEND_DIR / "modal"
FRAUD_ARTIFACT_PATH = MODAL_DIR / "upi_fraud_artifact.joblib"
KMEANS_ARTIFACT_PATH = MODAL_DIR / "upi_kmeans_artifact.joblib"
METADATA_PATH = MODAL_DIR / "model_metadata.json"

REQUIRED_BATCH_COLUMNS = {
    "transaction_type",
    "merchant_category",
    "amount_inr",
    "transaction_status",
    "sender_age_group",
    "receiver_age_group",
    "sender_state",
    "sender_bank",
    "receiver_bank",
    "device_type",
    "network_type",
    "hour",
    "day_of_week",
}


class FraudPredictorService:
    """Loads both the fraud classification pipelines and KMeans segmentation artifact."""

    def __init__(self) -> None:
        if not FRAUD_ARTIFACT_PATH.exists():
            raise FileNotFoundError(
                f"Fraud model artifact not found at {FRAUD_ARTIFACT_PATH}. Run train_models.py first."
            )
        if not KMEANS_ARTIFACT_PATH.exists():
            raise FileNotFoundError(
                f"KMeans artifact not found at {KMEANS_ARTIFACT_PATH}."
            )

        self.fraud_artifact: dict[str, Any] = joblib.load(FRAUD_ARTIFACT_PATH)
        self.kmeans_artifact: dict[str, Any] = joblib.load(KMEANS_ARTIFACT_PATH)
        self.metadata: dict[str, Any] = (
            json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
        )

        self.xgb_pipeline = self.fraud_artifact["xgb_pipeline"]
        self.lr_pipeline = self.fraud_artifact["lr_pipeline"]
        self.kmeans_scaler = self.kmeans_artifact["scaler"]
        self.kmeans_model = self.kmeans_artifact["kmeans"]
        self.segment_names: dict[int, str] = self.kmeans_artifact["segment_names"]

    def get_metadata(self) -> dict[str, Any]:
        return {
            "categorical_options": self.fraud_artifact.get("categorical_options", {}),
            "metrics": self.fraud_artifact.get("metrics", {}),
            "dataset_stats": self.fraud_artifact.get("dataset_stats", {}),
            "cluster_profiles": self.fraud_artifact.get("cluster_profiles", []),
        }

    @staticmethod
    def _classify_risk(prob: float) -> tuple[str, bool]:
        if prob >= 0.65:
            return "HIGH", True
        if prob >= 0.35:
            return "MEDIUM", False
        return "LOW", False

    @staticmethod
    def _extract_risk_factors(tx: TransactionInput, feat_row: pd.Series) -> list[RiskFactor]:
        factors: list[RiskFactor] = []
        if tx.amount_inr >= 10000:
            factors.append(
                RiskFactor(
                    label="High-Value Transaction",
                    impact="high",
                    detail=f"Amount ₹{tx.amount_inr:,.0f} exceeds ₹10,000 threshold (is_high_value=1).",
                )
            )
        if int(feat_row["is_late_night"]) == 1:
            factors.append(
                RiskFactor(
                    label="Late-Night Window",
                    impact="high",
                    detail=f"Transaction at {tx.hour:02d}:00 falls in the 00:00–05:59 late-night slot.",
                )
            )
        if tx.transaction_status == "FAILED":
            factors.append(
                RiskFactor(
                    label="Failed Status Flag",
                    impact="medium",
                    detail="Failed transactions often correlate with repeated probing or blocked attempts.",
                )
            )
        if tx.device_type == "Web" or tx.network_type == "3G":
            factors.append(
                RiskFactor(
                    label="Elevated Channel Risk",
                    impact="medium",
                    detail=f"Channel combination {tx.device_type} over {tx.network_type} carries higher anomaly rate.",
                )
            )
        if tx.sender_bank != tx.receiver_bank:
            factors.append(
                RiskFactor(
                    label="Inter-Bank Transfer",
                    impact="low",
                    detail=f"Cross-bank transfer from {tx.sender_bank} to {tx.receiver_bank}.",
                )
            )
        if not factors:
            factors.append(
                RiskFactor(
                    label="Normal Behavioral Profile",
                    impact="low",
                    detail="Amount, time slot, and channel align with typical UPI usage.",
                )
            )
        return factors

    def predict_single(self, tx: TransactionInput) -> PredictionResponse:
        raw_df = pd.DataFrame([tx.model_dump(exclude={"model_type"})])
        features_df = engineer_features(raw_df)

        pipeline = self.xgb_pipeline if tx.model_type == "xgboost" else self.lr_pipeline
        prob = float(pipeline.predict_proba(features_df)[0][1])
        risk_level, is_fraud = self._classify_risk(prob)

        cluster_scaled = self.kmeans_scaler.transform(features_df[CLUSTER_COLUMNS])
        cluster_id = int(self.kmeans_model.predict(cluster_scaled)[0])
        segment = self.segment_names.get(cluster_id, f"Cluster {cluster_id}")

        feat_row = features_df.iloc[0]
        engineered = {
            "log_amount": round(float(feat_row["log_amount"]), 4),
            "is_high_value": int(feat_row["is_high_value"]),
            "is_late_night": int(feat_row["is_late_night"]),
            "is_weekend": int(feat_row["is_weekend"]),
            "time_slot": str(feat_row["time_slot"]),
            "bank_device": str(feat_row["bank_device"]),
            "bank_category": str(feat_row["bank_category"]),
        }

        return PredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=round(prob, 4),
            risk_score=int(round(prob * 100)),
            risk_level=risk_level,
            model_used=tx.model_type,
            cluster_id=cluster_id,
            customer_segment=segment,
            engineered_features=engineered,
            risk_factors=self._extract_risk_factors(tx, feat_row),
        )

    def predict_batch(
        self, df: pd.DataFrame, model_type: str = "xgboost", max_rows: int = 5000
    ) -> BatchPredictionResponse:
        missing = REQUIRED_BATCH_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

        batch_df = df.head(max_rows).copy()
        features_df = engineer_features(batch_df)

        pipeline = self.xgb_pipeline if model_type == "xgboost" else self.lr_pipeline
        probs = pipeline.predict_proba(features_df)[:, 1]
        clusters = self.kmeans_model.predict(
            self.kmeans_scaler.transform(features_df[CLUSTER_COLUMNS])
        )

        results: list[dict[str, Any]] = []
        flagged_count = 0
        high_risk_count = 0

        for idx, prob in enumerate(probs):
            prob_val = float(prob)
            risk_level, is_fraud = self._classify_risk(prob_val)
            if is_fraud:
                flagged_count += 1
            if risk_level == "HIGH":
                high_risk_count += 1
            cid = int(clusters[idx])
            row_in = batch_df.iloc[idx]
            results.append(
                {
                    "row_index": idx + 1,
                    "amount_inr": float(row_in["amount_inr"]),
                    "transaction_type": str(row_in["transaction_type"]),
                    "merchant_category": str(row_in["merchant_category"]),
                    "sender_bank": str(row_in["sender_bank"]),
                    "hour": int(row_in["hour"]),
                    "fraud_probability": round(prob_val, 4),
                    "risk_score": int(round(prob_val * 100)),
                    "risk_level": risk_level,
                    "is_fraud": is_fraud,
                    "customer_segment": self.segment_names.get(cid, f"Cluster {cid}"),
                }
            )

        return BatchPredictionResponse(
            total_count=len(results),
            flagged_fraud_count=flagged_count,
            high_risk_count=high_risk_count,
            model_used=model_type,
            results=results,
        )
