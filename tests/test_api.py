import numpy as np
from fastapi.testclient import TestClient

from heart_ml import api


class DummyModel:
    def predict(self, frame):
        return np.array([1])

    def predict_proba(self, frame):
        return np.array([[0.2, 0.8]])


def test_health_endpoint():
    client = TestClient(api.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_response(monkeypatch):
    monkeypatch.setattr(api, "load_model", lambda: DummyModel())
    client = TestClient(api.app)
    body = {
        "age": 57, "sex": 1, "cp": 2, "trestbps": 140, "chol": 240,
        "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
        "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 3
    }
    response = client.post("/predict", json=body)
    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] == 1
    assert payload["confidence"] == 0.8
    assert "disclaimer" in payload


def test_predict_rejects_extra_field():
    client = TestClient(api.app)
    body = {
        "age": 57, "sex": 1, "cp": 2, "trestbps": 140, "chol": 240,
        "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
        "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 3, "unexpected": 9
    }
    response = client.post("/predict", json=body)
    assert response.status_code == 422
