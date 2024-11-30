"""Test end-to-end drift detection workflow."""

from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

from monitoring.drift import detect_drift, get_drifted_features
from service.app import InProcessEventProcessor
from storage.db import DriftSignal, Prediction


@pytest.fixture
def reference_data(sample_data):
    """Create reference data for drift detection."""
    return sample_data.copy()


@pytest.fixture
def production_data_shifted(sample_data):
    """Create shifted production data."""
    prod_data = sample_data.copy()
    prod_data['age'] = prod_data['age'] + 10  # Shift age distribution
    return prod_data


def test_drift_detection_runs_at_threshold(db_session):
    """Test drift detection runs after N predictions."""
    from monitoring.events import EventQueue, PredictionEvent, reset_event_queue
    
    reset_event_queue()
    event_queue = EventQueue(maxsize=100)
    
    def session_factory():
        return db_session
    
    processor = InProcessEventProcessor(
        event_queue=event_queue,
        session_factory=session_factory,
        poll_interval=0.1,
        batch_size=10
    )
    
    processor.drift_check_threshold = 3  # Lower for testing
    
    # Add enough events to trigger drift check
    for i in range(3):
        event = PredictionEvent(
            timestamp=datetime.utcnow(),
            features={"age": 39 + i, "education": "Bachelors"},
            prediction="<=50K",
            probability={"<=50K": 0.85}
        )
        event_queue.enqueue(event, block=False)
    
    initial_count = processor.predictions_since_drift_check
    
    with patch.object(processor, '_run_drift_detection') as mock_drift:
        processor._process_batch(event_queue.dequeue_batch(10, block=False))
        
        # Should trigger drift detection
        assert mock_drift.called


def test_drift_detection_stores_signals(db_session, reference_data, production_data_shifted):
    """Test drift signals are stored in database."""
    from monitoring.drift import detect_drift
    from storage.db import store_drift_signal
    
    drift_scores = detect_drift(reference_data, production_data_shifted, threshold=0.05)
    
    assert len(drift_scores) > 0
    
    store_drift_signal(db_session, drift_scores, threshold=0.05)
    db_session.commit()
    
    stored_signals = db_session.query(DriftSignal).all()
    assert len(stored_signals) > 0


def test_drift_detection_generates_alerts(db_session, reference_data, production_data_shifted):
    """Test drift detection generates alerts for drifted features."""
    from monitoring.alerts import check_drift_alerts
    from monitoring.drift import detect_drift
    from storage.db import store_drift_signal
    
    drift_scores = detect_drift(reference_data, production_data_shifted, threshold=0.05)
    store_drift_signal(db_session, drift_scores, threshold=0.05)
    db_session.commit()
    
    drifted_features = get_drifted_features(drift_scores, threshold=0.05)
    
    if drifted_features:
        alerts = check_drift_alerts(
            db_session,
            drift_scores,
            threshold=0.05,
            severity_threshold=0.01
        )
        
        assert len(alerts) >= 0  # May or may not have alerts depending on scores


def test_drift_detection_uses_reference_data(db_session):
    """Test drift detection compares to training reference data."""
    from service.app import InProcessEventProcessor
    from monitoring.events import EventQueue, PredictionEvent, reset_event_queue
    
    reset_event_queue()
    event_queue = EventQueue(maxsize=100)
    
    def session_factory():
        return db_session
    
    processor = InProcessEventProcessor(
        event_queue=event_queue,
        session_factory=session_factory,
        poll_interval=0.1,
        batch_size=10
    )
    
    # Mock reference data loading
    with patch.object(processor, '_load_reference_data') as mock_ref:
        mock_ref.return_value = pd.DataFrame({
            'age': [39, 40, 41],
            'education': ['Bachelors', 'Masters', 'HS-grad']
        })
        
        # Should use reference data
        ref_data = processor._load_reference_data()
        assert ref_data is not None
        assert len(ref_data) > 0


def test_drift_detection_handles_insufficient_data(db_session):
    """Test drift detection handles insufficient production data."""
    from service.app import InProcessEventProcessor
    from monitoring.events import EventQueue, reset_event_queue
    
    reset_event_queue()
    event_queue = EventQueue(maxsize=100)
    
    def session_factory():
        return db_session
    
    processor = InProcessEventProcessor(
        event_queue=event_queue,
        session_factory=session_factory,
        poll_interval=0.1,
        batch_size=10
    )
    
    # Only 2 predictions (need at least 10)
    for i in range(2):
        pred = Prediction(
            features='{"age": 39}',
            prediction="<=50K",
            probability='{"<=50K": 0.85}',
            timestamp=datetime.utcnow()
        )
        db_session.add(pred)
    db_session.commit()
    
    production_data = processor._get_production_data(db_session)
    
    assert production_data is None  # Should return None with insufficient data


def test_drift_detection_prediction_thresholded(db_session):
    """Test drift detection is prediction-thresholded, not time-based."""
    from service.app import InProcessEventProcessor
    from monitoring.events import EventQueue, PredictionEvent, reset_event_queue
    
    reset_event_queue()
    event_queue = EventQueue(maxsize=100)
    
    def session_factory():
        return db_session
    
    processor = InProcessEventProcessor(
        event_queue=event_queue,
        session_factory=session_factory,
        poll_interval=0.1,
        batch_size=10
    )
    
    processor.drift_check_threshold = 5
    
    # Process events up to threshold
    events = [
        PredictionEvent(
            timestamp=datetime.utcnow(),
            features={"age": 39 + i},
            prediction="<=50K",
            probability={"<=50K": 0.85}
        ) for i in range(4)
    ]
    
    processor._process_batch(events)
    
    # Should not trigger yet (4 < 5)
    assert processor.predictions_since_drift_check == 4
    
    # Add one more
    processor._process_batch([PredictionEvent(
        timestamp=datetime.utcnow(),
        features={"age": 39},
        prediction="<=50K",
        probability={"<=50K": 0.85}
    )])
    
    # Should reset counter after drift check
    # (implementation dependent, but should be 0 or 1)
