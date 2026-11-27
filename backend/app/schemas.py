"""Pydantic request and response schemas for the UPI Fraud Prediction API."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    transaction_type: str = Field(default="P2P", description="P2P, P2M, Bill Payment, or Recharge")
    merchant_category: str = Field(default="Food", description="Merchant category")
    amount_inr: float = Field(default=500.0, gt=0, le=10000000, description="Transaction amount in INR")
    transaction_status: str = Field(default="SUCCESS", description="SUCCESS or FAILED")
    sender_age_group: str = Field(default="26-35", description="Sender age bracket")
    receiver_age_group: str = Field(default="26-35", description="Receiver age bracket")
    sender_state: str = Field(default="Maharashtra", description="Sender state")
    sender_bank: str = Field(default="SBI", description="Sender bank")
    receiver_bank: str = Field(default="HDFC", description="Receiver bank")
    device_type: str = Field(default="Android", description="Android, iOS, or Web")
    network_type: str = Field(default="4G", description="3G, 4G, 5G, or WiFi")
    hour: int = Field(default=14, ge=0, le=23, description="Transaction hour (0-23)")
    day_of_week: str = Field(default="Monday", description="Day of week")
    model_type: Literal["xgboost", "logistic_regression"] = Field(
        default="xgboost", description="Model used for fraud scoring"
    )


class RiskFactor(BaseModel):
    label: str
    impact: Literal["high", "medium", "low"]
    detail: str


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_score: int
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    model_used: str
    cluster_id: int
    customer_segment: str
    engineered_features: dict[str, Any]
    risk_factors: list[RiskFactor]


class BatchPredictionResponse(BaseModel):
    total_count: int
    flagged_fraud_count: int
    high_risk_count: int
    model_used: str
    results: list[dict[str, Any]]
