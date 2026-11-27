"""API integration tests for the UPI Fraud Prediction FastAPI backend."""

from fastapi.testclient import TestClient

from app.main import app


def test_health_and_metadata():
    with TestClient(app) as client:
        res = client.get("/api/health")
        assert res.status_code == 200
        assert res.json() == {"status": "ok", "models_loaded": True}

        meta = client.get("/api/metadata")
        assert meta.status_code == 200
        data = meta.json()
        assert "categorical_options" in data
        assert "metrics" in data
        assert "dataset_stats" in data
        assert len(data["cluster_profiles"]) == 5


def test_predict_single_xgboost_and_lr():
    payload = {
        "transaction_type": "P2P",
        "merchant_category": "Education",
        "amount_inr": 25000,
        "transaction_status": "FAILED",
        "sender_age_group": "26-35",
        "receiver_age_group": "46-55",
        "sender_state": "Maharashtra",
        "sender_bank": "SBI",
        "receiver_bank": "HDFC",
        "device_type": "Web",
        "network_type": "3G",
        "hour": 2,
        "day_of_week": "Sunday",
        "model_type": "xgboost",
    }
    with TestClient(app) as client:
        res = client.post("/api/predict", json=payload)
        assert res.status_code == 200
        body = res.json()
        assert 0.0 <= body["fraud_probability"] <= 1.0
        assert 0 <= body["risk_score"] <= 100
        assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
        assert body["customer_segment"]
        assert body["engineered_features"]["is_high_value"] == 1
        assert body["engineered_features"]["is_late_night"] == 1
        assert len(body["risk_factors"]) >= 3

        payload["model_type"] = "logistic_regression"
        res_lr = client.post("/api/predict", json=payload)
        assert res_lr.status_code == 200
        assert res_lr.json()["model_used"] == "logistic_regression"


def test_predict_batch_csv():
    csv_content = (
        "transaction_type,merchant_category,amount_inr,transaction_status,"
        "sender_age_group,receiver_age_group,sender_state,sender_bank,receiver_bank,"
        "device_type,network_type,hour,day_of_week\n"
        "P2P,Food,350,SUCCESS,26-35,26-35,Maharashtra,SBI,SBI,Android,4G,19,Monday\n"
        "P2M,Shopping,18500,FAILED,18-25,36-45,Delhi,HDFC,ICICI,Web,3G,2,Sunday\n"
    )
    with TestClient(app) as client:
        res = client.post(
            "/api/predict/batch?model_type=xgboost",
            files={"file": ("sample.csv", csv_content.encode("utf-8"), "text/csv")},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["total_count"] == 2
        assert len(data["results"]) == 2
