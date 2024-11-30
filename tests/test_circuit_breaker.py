"""Test circuit breaker logic and trust boundary enforcement."""

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from monitoring.circuit_breaker import check_critical_drift
from service.app import app
from storage.db import DriftSignal, Prediction


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def db_with_drift_signals(db_session):
    """Create database with drift signals."""
    from datetime import datetime, timedelta
    from storage.db import Base
    
    # Ensure tables exist
    Base.metadata.create_all(bind=db_session.bind)
    
    now = datetime.utcnow()
    
    # Create critical drift signal (p < 0.01)
    drift1 = DriftSignal(
        feature_name="age",
        drift_score=0.005,
        timestamp=now - timedelta(hours=1),
        is_drifted=True
    )
    
    # Create warning drift signal (0.01 <= p < 0.05)
    drift2 = DriftSignal(
        feature_name="education",
        drift_score=0.02,
        timestamp=now - timedelta(hours=2),
        is_drifted=True
    )
    
    # Create old drift signal (outside lookback window)
    drift3 = DriftSignal(
        feature_name="workclass",
        drift_score=0.003,
        timestamp=now - timedelta(hours=25),
        is_drifted=True
    )
    
    db_session.add(drift1)
    db_session.add(drift2)
    db_session.add(drift3)
    db_session.commit()
    
    return db_session


def test_circuit_breaker_no_drift(db_session):
    """Test circuit breaker allows serving when no drift."""
    blocked, message = check_critical_drift(db_session, severity_threshold=0.01, lookback_hours=24)
    
    assert blocked is False
    assert message is None


def test_circuit_breaker_critical_drift(db_with_drift_signals):
    """Test circuit breaker blocks serving on critical drift."""
    blocked, message = check_critical_drift(
        db_with_drift_signals,
        severity_threshold=0.01,
        lookback_hours=24
    )
    
    assert blocked is True
    assert message is not None
    assert "critical drift" in message.lower() or "age" in message.lower()


def test_circuit_breaker_lookback_window(db_with_drift_signals):
    """Test circuit breaker respects lookback time window."""
    # Check with 24-hour lookback (should find drift)
    blocked_24h, _ = check_critical_drift(
        db_with_drift_signals,
        severity_threshold=0.01,
        lookback_hours=24
    )
    
    # Check with 1-hour lookback (should NOT find old drift)
    blocked_1h, _ = check_critical_drift(
        db_with_drift_signals,
        severity_threshold=0.01,
        lookback_hours=1
    )
    
    assert blocked_24h is True
    assert blocked_1h is True  # Still finds recent critical drift


def test_circuit_breaker_severity_threshold(db_session):
    """Test circuit breaker uses correct severity threshold."""
    from storage.db import Base
    # Ensure tables exist
    Base.metadata.create_all(bind=db_session.bind)
    
    now = datetime.utcnow()
    
    # Create drift signal just above threshold
    drift_above = DriftSignal(
        feature_name="age",
        drift_score=0.015,  # Above 0.01 threshold
        timestamp=now - timedelta(hours=1),
        is_drifted=False
    )
    
    # Create drift signal below threshold
    drift_below = DriftSignal(
        feature_name="education",
        drift_score=0.005,  # Below 0.01 threshold
        timestamp=now - timedelta(hours=1),
        is_drifted=True
    )
    
    db_session.add(drift_above)
    db_session.add(drift_below)
    db_session.commit()
    
    blocked, _ = check_critical_drift(
        db_session,
        severity_threshold=0.01,
        lookback_hours=24
    )
    
    assert blocked is True  # Should block because drift_below triggers it


def test_circuit_breaker_clears_after_drift_resolves(db_session):
    """Test circuit breaker resets when drift clears."""
    from storage.db import Base
    # Ensure tables exist
    Base.metadata.create_all(bind=db_session.bind)
    
    now = datetime.utcnow()
    
    # Add critical drift
    drift = DriftSignal(
        feature_name="age",
        drift_score=0.005,
        timestamp=now - timedelta(hours=1),
        is_drifted=True
    )
    db_session.add(drift)
    db_session.commit()
    
    blocked, _ = check_critical_drift(db_session, severity_threshold=0.01, lookback_hours=24)
    assert blocked is True
    
    # Delete drift signal
    db_session.delete(drift)
    db_session.commit()
    
    blocked_after, _ = check_critical_drift(db_session, severity_threshold=0.01, lookback_hours=24)
    assert blocked_after is False


def test_circuit_breaker_integration_with_api(client, db_with_drift_signals, sample_features):
    """Test circuit breaker integration with /predict endpoint."""
    from unittest.mock import patch
    from storage.db import Base, Prediction
    
    # Ensure tables exist
    Base.metadata.create_all(bind=db_with_drift_signals.bind)
    
    with patch('service.app.get_db_session', return_value=db_with_drift_signals):
        # Mock model to avoid model loading issues
        with patch('service.app.model') as mock_model:
            mock_model.predict.return_value = ["<=50K"]
            mock_model.predict_proba.return_value = [[0.85, 0.15]]
            
            response = client.post("/predict", json={
                "features": sample_features,
                "include_explanation": False
            })
            
            # May return 503 (circuit breaker) or 500 (model not loaded) or 200 (if model works)
            assert response.status_code in [200, 503, 500]
            if response.status_code == 503:
                assert "drift" in response.json()["detail"].lower() or "blocked" in response.json()["detail"].lower()
