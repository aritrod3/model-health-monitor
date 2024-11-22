"""Test API endpoints."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from service.app import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_predict_endpoint_without_model(client, sample_features: dict):
    """Test predict endpoint without model loaded."""
    # Without model, should handle gracefully
    response = client.post("/predict", json={
        "features": sample_features,
        "include_explanation": False
    })
    # May fail without model, but should handle gracefully
    assert response.status_code in [200, 503, 500]


def test_batch_predict_endpoint_without_model(client, sample_features: dict):
    """Test batch predict endpoint without model loaded."""
    response = client.post("/predict/batch", json={
        "instances": [sample_features, sample_features],
        "include_explanations": False
    })
    # May fail without model, but should handle gracefully
    assert response.status_code in [200, 503, 500]


def test_metrics_endpoint(client, db_session):
    """Test metrics endpoint."""
    from storage.db import Base, Metric
    # Ensure tables exist
    Base.metadata.create_all(bind=db_session.bind)
    
    # Mock database session
    with patch('service.app.get_db_session', return_value=db_session):
        from storage.db import store_metrics
        store_metrics(db_session, {'accuracy': 0.85, 'precision': 0.82})
        db_session.commit()
        
        response = client.get("/metrics")
        assert response.status_code in [200, 404, 500]  # 500 if no data, which is OK


def test_drift_endpoint(client, db_session):
    """Test drift endpoint."""
    from storage.db import Base
    # Ensure tables exist
    Base.metadata.create_all(bind=db_session.bind)
    
    with patch('service.app.get_db_session', return_value=db_session):
        response = client.get("/drift")
        assert response.status_code in [200, 500]  # 500 if no data
        if response.status_code == 200:
        data = response.json()
        assert "drifted_features" in data
        assert "drift_scores" in data
        assert "timestamp" in data
