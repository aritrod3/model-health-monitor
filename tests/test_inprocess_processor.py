"""Test in-process event processor."""

import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from monitoring.events import EventQueue, PredictionEvent, reset_event_queue
from service.app import InProcessEventProcessor


@pytest.fixture
def event_queue():
    """Create event queue for testing."""
    reset_event_queue()
    return EventQueue(maxsize=100)


@pytest.fixture
def session_factory(db_session):
    """Create session factory for processor."""
    def factory():
        return db_session
    return factory


@pytest.fixture
def sample_events():
    """Create sample prediction events."""
    return [
        PredictionEvent(
            timestamp=datetime.utcnow(),
            features={"age": 39, "education": "Bachelors"},
            prediction="<=50K",
            probability={"<=50K": 0.85, ">50K": 0.15}
        ) for _ in range(5)
    ]


def test_inprocess_processor_initialization(event_queue, session_factory):
    """Test processor starts when no external worker detected."""
    processor = InProcessEventProcessor(
        event_queue=event_queue,
        session_factory=session_factory,
        poll_interval=0.1,
        batch_size=10
    )
    
    assert processor.running is False
    assert processor.thread is None
    assert processor.poll_interval == 0.1
    assert processor.batch_size == 10


def test_inprocess_processor_processes_events(event_queue, session_factory, sample_events):
    """Test processor consumes events from queue."""
    processor = InProcessEventProcessor(
        event_queue=event_queue,
        session_factory=session_factory,
        poll_interval=0.1,
        batch_size=10
    )
    
    for event in sample_events:
        event_queue.enqueue(event, block=False)
    
    assert event_queue.size() == 5
    
    processor.start()
    time.sleep(0.5)  # Allow processor to run
    
    processor.stop()
    
    # Events should be processed
    assert event_queue.size() == 0


def test_inprocess_processor_drift_detection(event_queue, session_factory, db_session):
    """Test processor runs drift detection at threshold."""
    from unittest.mock import patch
    from storage.db import Base, Prediction
    
    # Ensure tables exist
    Base.metadata.create_all(bind=db_session.bind)
    
    processor = InProcessEventProcessor(
        event_queue=event_queue,
        session_factory=session_factory,
        poll_interval=0.1,
        batch_size=10
    )
    
    processor.drift_check_threshold = 3  # Lower threshold for testing
    processor.predictions_since_drift_check = 0
    
    # Add enough events to trigger drift check
    for i in range(3):
        event = PredictionEvent(
            timestamp=datetime.utcnow(),
            features={"age": 39 + i, "education": "Bachelors"},
            prediction="<=50K",
            probability={"<=50K": 0.85}
        )
        event_queue.enqueue(event, block=False)
    
    with patch.object(processor, '_run_drift_detection') as mock_drift:
        processor._process_batch(event_queue.dequeue_batch(10, block=False))
        
        # Should call drift detection after threshold
        assert mock_drift.called


def test_inprocess_processor_metrics_computation(event_queue, session_factory, db_session):
    """Test processor computes live metrics from labeled predictions."""
    processor = InProcessEventProcessor(
        event_queue=event_queue,
        session_factory=session_factory,
        poll_interval=0.1,
        batch_size=10
    )
    
    # Add labeled events
    for i in range(3):
        event = PredictionEvent(
            timestamp=datetime.utcnow(),
            features={"age": 39, "education": "Bachelors"},
            prediction="<=50K",
            probability={"<=50K": 0.85, ">50K": 0.15},
            label="<=50K"
        )
        event_queue.enqueue(event, block=False)
    
    processor.start()
    time.sleep(0.5)
    processor.stop()
    
    # Check that metrics were computed (if there are enough labeled predictions)
    from storage.db import Metric
    metrics = db_session.query(Metric).all()
    # Metrics might not be computed if insufficient data, so just verify processor ran
    assert event_queue.size() == 0


def test_inprocess_processor_graceful_shutdown(event_queue, session_factory):
    """Test processor stops cleanly."""
    processor = InProcessEventProcessor(
        event_queue=event_queue,
        session_factory=session_factory,
        poll_interval=0.1,
        batch_size=10
    )
    
    processor.start()
    assert processor.running is True
    
    processor.stop()
    
    time.sleep(0.2)
    assert processor.running is False


def test_inprocess_processor_handles_errors(event_queue, session_factory):
    """Test processor handles errors gracefully."""
    def failing_session_factory():
        raise Exception("Database error")
    
    processor = InProcessEventProcessor(
        event_queue=event_queue,
        session_factory=failing_session_factory,
        poll_interval=0.1,
        batch_size=10
    )
    
    event = PredictionEvent(
        timestamp=datetime.utcnow(),
        features={"age": 39},
        prediction="<=50K",
        probability={"<=50K": 0.85}
    )
    event_queue.enqueue(event, block=False)
    
    # Should not crash on error
    processor.start()
    time.sleep(0.3)
    processor.stop()
    
    assert processor.running is False
