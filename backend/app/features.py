"""Shared feature engineering logic matching feature_engineering.ipynb."""

from __future__ import annotations

import numpy as np
import pandas as pd

WEEKEND_DAYS = {"Saturday", "Sunday"}

FEATURE_COLUMNS = [
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
    "is_weekend",
    "hour",
    "day_of_week",
    "time_slot",
    "log_amount",
    "is_high_value",
    "is_late_night",
    "bank_device",
    "bank_category",
]

CLUSTER_COLUMNS = ["log_amount", "hour", "is_high_value"]


def compute_time_slot(hour: int) -> str:
    if hour <= 5:
        return "Late_Night"
    if hour <= 11:
        return "Morning"
    if hour <= 17:
        return "Afternoon"
    return "Evening"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact feature engineering transformations from feature_engineering.ipynb."""
    out = df.copy()
    out["amount_inr"] = pd.to_numeric(out["amount_inr"], errors="coerce").fillna(0.0)
    out["hour"] = pd.to_numeric(out["hour"], errors="coerce").fillna(0).astype(int).clip(0, 23)

    if "is_weekend" not in out.columns:
        out["is_weekend"] = out["day_of_week"].isin(WEEKEND_DAYS).astype(int)
    else:
        out["is_weekend"] = pd.to_numeric(out["is_weekend"], errors="coerce").fillna(0).astype(int)

    out["log_amount"] = np.log1p(out["amount_inr"])
    out["is_high_value"] = (out["amount_inr"] >= 10000).astype(int)
    out["is_late_night"] = out["hour"].between(0, 5).astype(int)
    out["time_slot"] = pd.cut(
        out["hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=["Late_Night", "Morning", "Afternoon", "Evening"],
    ).astype(str)
    out["bank_device"] = out["sender_bank"].astype(str) + "_" + out["device_type"].astype(str)
    out["bank_category"] = out["sender_bank"].astype(str) + "_" + out["merchant_category"].astype(str)

    return out[FEATURE_COLUMNS]
