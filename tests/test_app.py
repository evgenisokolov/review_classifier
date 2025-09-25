import pytest
from starlette.testclient import TestClient

from src.serving.app import app

client = TestClient(app)

def test_predict_endpoint_returns_classes_and_probabilities(monkeypatch):
    # Patch embedding_model and xgb_model for speed and determinism
    from src.serving import app as app_module

    class DummyEmbedder:
        def encode(self, texts, **kwargs):
            import numpy as np
            return np.zeros((len(texts), 10))  # fake 10-dim embeddings

    class DummyXGBModel:
        def predict_proba(self, X):
            import numpy as np
            # return dummy probabilities for 3 classes
            probs = np.full((X.shape[0], 3), 0.0)
            probs[:, 0] = 1.0
            return probs

    monkeypatch.setattr(app_module, "embedding_model", DummyEmbedder())
    monkeypatch.setattr(app_module, "xgb_model", DummyXGBModel())

    response = client.post("/predict", json={"texts": ["foo", "bar"]})
    assert response.status_code == 200
    data = response.json()
    assert "classes" in data and "probabilities" in data
    assert len(data["classes"]) == 2
    assert len(data["probabilities"]) == 2
    # because our dummy always picks class 0
    assert all(c == 0 for c in data["classes"])

def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    # prometheus content type
    assert "text/plain" in r.headers["content-type"]
    assert b"http_requests_total" in r.content
