"""Train and export the UPI fraud detection models alongside the existing KMeans artifact."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from app.features import CLUSTER_COLUMNS, FEATURE_COLUMNS, engineer_features

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "preprocessed" / "upi_features.csv"
MODAL_DIR = ROOT_DIR / "backend" / "modal"
KMEANS_PATH = MODAL_DIR / "upi_kmeans_artifact.joblib"
FRAUD_ARTIFACT_PATH = MODAL_DIR / "upi_fraud_artifact.joblib"
METADATA_PATH = MODAL_DIR / "model_metadata.json"

INPUT_CATEGORICAL_COLS = [
    "transaction_type",
    "merchant_category",
    "transaction_status",
    "sender_age_group",
    "receiver_age_group",
    "sender_state",
    "sender_bank",
    "receiver_bank",
    "device_type",
    "network_type",
    "day_of_week",
]


def evaluate_model(name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    pr_auc = float(average_precision_score(y_test, y_prob))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    fraud_metrics = report.get("1", {})
    return {
        "name": name,
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "accuracy": round(float(report.get("accuracy", 0.0)), 4),
        "fraud_precision": round(float(fraud_metrics.get("precision", 0.0)), 4),
        "fraud_recall": round(float(fraud_metrics.get("recall", 0.0)), 4),
        "fraud_f1": round(float(fraud_metrics.get("f1-score", 0.0)), 4),
        "confusion_matrix": cm,
    }


def main() -> None:
    print(f"Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    X = engineer_features(df)
    y = df["fraud_flag"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    lr_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ]
    )
    print("Training Logistic Regression...")
    lr_pipeline.fit(X_train, y_train)

    fraud_ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    xgb_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.05,
                    scale_pos_weight=fraud_ratio,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    print("Training XGBoost...")
    xgb_pipeline.fit(X_train, y_train)

    lr_metrics = evaluate_model("Logistic Regression (Balanced)", lr_pipeline, X_test, y_test)
    xgb_metrics = evaluate_model("XGBoost", xgb_pipeline, X_test, y_test)
    print("LR Metrics:", lr_metrics)
    print("XGB Metrics:", xgb_metrics)

    # Load existing KMeans artifact to compute cluster profile stats
    kmeans_artifact = joblib.load(KMEANS_PATH)
    scaler = kmeans_artifact["scaler"]
    kmeans = kmeans_artifact["kmeans"]
    segment_names = kmeans_artifact["segment_names"]

    X_cluster = X[CLUSTER_COLUMNS]
    clusters = kmeans.predict(scaler.transform(X_cluster))
    df_profile = df.copy()
    df_profile["cluster"] = clusters
    df_profile["segment"] = df_profile["cluster"].map(segment_names)

    cluster_profiles = []
    for cid, grp in df_profile.groupby("cluster"):
        cluster_profiles.append(
            {
                "cluster_id": int(cid),
                "segment_name": segment_names.get(int(cid), f"Cluster {cid}"),
                "txn_count": int(len(grp)),
                "share_pct": round(float(len(grp) / len(df_profile) * 100), 2),
                "median_amount_inr": round(float(grp["amount_inr"].median()), 2),
                "mean_amount_inr": round(float(grp["amount_inr"].mean()), 2),
                "avg_hour": round(float(grp["hour"].mean()), 2),
                "fraud_rate_pct": round(float(grp["fraud_flag"].mean() * 100), 3),
                "top_category": str(grp["merchant_category"].mode()[0]),
                "top_bank": str(grp["sender_bank"].mode()[0]),
            }
        )

    categorical_options = {
        col: sorted(df[col].dropna().astype(str).unique().tolist())
        for col in INPUT_CATEGORICAL_COLS
    }

    dataset_stats = {
        "total_transactions": int(len(df)),
        "fraud_count": int(y.sum()),
        "legit_count": int((y == 0).sum()),
        "fraud_rate_pct": round(float(y.mean() * 100), 3),
        "avg_amount_inr": round(float(df["amount_inr"].mean()), 2),
        "median_amount_inr": round(float(df["amount_inr"].median()), 2),
        "max_amount_inr": round(float(df["amount_inr"].max()), 2),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }

    MODAL_DIR.mkdir(parents=True, exist_ok=True)
    fraud_artifact = {
        "xgb_pipeline": xgb_pipeline,
        "lr_pipeline": lr_pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_options": categorical_options,
        "metrics": {
            "xgboost": xgb_metrics,
            "logistic_regression": lr_metrics,
        },
        "dataset_stats": dataset_stats,
        "cluster_profiles": cluster_profiles,
    }
    joblib.dump(fraud_artifact, FRAUD_ARTIFACT_PATH)

    metadata = {
        "categorical_options": categorical_options,
        "metrics": fraud_artifact["metrics"],
        "dataset_stats": dataset_stats,
        "cluster_profiles": cluster_profiles,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"Saved fraud artifact to {FRAUD_ARTIFACT_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")


if __name__ == "__main__":
    main()
