import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient
from prometheus_client import CollectorRegistry

from src.serving.middleware import create_logging_middleware
from src.utils.logger import init_logger

def test_logging_middleware_increments_metrics():
    app = FastAPI()
    logger = init_logger("test_logger")

    # Create a fresh, isolated registry for this test
    test_registry = CollectorRegistry()

    # Add the middleware with the new registry
    app.middleware("http")(create_logging_middleware(logger, registry=test_registry))

    @app.get("/ping")
    def ping():
        return {"pong": True}

    client = TestClient(app)

    # Call endpoint
    r = client.get("/ping")
    assert r.status_code == 200
    assert r.json() == {"pong": True}

    metrics = list(test_registry.collect())

    http_requests_metric = None
    for m in metrics:
        if m.name == "http_requests":
            http_requests_metric = m
            break
    
    assert http_requests_metric is not None, "Metric 'http_requests' not found in registry"

    found_ping_sample = False
    for sample in http_requests_metric.samples:
        if sample.name == 'http_requests_total' and sample.labels.get('endpoint') == '/ping':
            found_ping_sample = True
            break
    
    assert found_ping_sample, "Sample for /ping not found in 'http_requests_total' metric"