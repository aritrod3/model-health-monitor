"""Test system performance under load."""

import time
from datetime import datetime

import pytest

from monitoring.events import EventQueue, PredictionEvent, reset_event_queue
from monitoring.worker import MonitoringWorker


@pytest.fixture
def event_queue():
    """Create event queue for testing."""
    reset_event_queue()
    return EventQueue(maxsize=1000)


@pytest.fixture
def sample_events():
    """Create sample prediction events."""
    return [
        PredictionEvent(
            timestamp=datetime.utcnow(),
            features={"age": 39 + i, "education": "Bachelors"},
            prediction="<=50K",
            probability={"<=50K": 0.85, ">50K": 0.15}
        ) for i in range(100)
    ]


def test_high_throughput_prediction_handling(event_queue, sample_events):
    """Test system handles high prediction rate."""
    start_time = time.time()
    
    # Enqueue 100 events
    for event in sample_events:
        event_queue.enqueue(event, block=False)
    
    enqueue_time = time.time() - start_time
    
    # Should enqueue quickly
    assert enqueue_time < 1.0
    assert event_queue.size() == 100


def test_event_queue_performance(event_queue, sample_events):
    """Test event queue performance under load."""
    # Enqueue events
    for event in sample_events:
        event_queue.enqueue(event, block=False)
    
    start_time = time.time()
    
    # Dequeue in batches
    total_dequeued = 0
    while total_dequeued < len(sample_events):
        batch = event_queue.dequeue_batch(10, block=False)
        total_dequeued += len(batch)
        if not batch:
            break
    
    dequeue_time = time.time() - start_time
    
    assert total_dequeued == len(sample_events)
    assert dequeue_time < 1.0  # Should be fast


def test_worker_batch_processing_performance(event_queue, sample_events, db_session):
    """Test worker batch processing efficiency."""
    from storage.db import Base
    # Ensure tables exist
    Base.metadata.create_all(bind=db_session.bind)
    
    worker = MonitoringWorker(
        event_queue=event_queue,
        poll_interval=0.1,
        batch_size=50,
        db_path="sqlite:///:memory:"
    )
    
    # Patch SessionLocal to use our test session
    worker.SessionLocal = lambda: db_session
    
    # Enqueue events (limit to available)
    events_to_test = sample_events[:min(50, len(sample_events))]
    for event in events_to_test:
        event_queue.enqueue(event, block=False)
    
    start_time = time.time()
    processed = worker.run_once()
    process_time = time.time() - start_time
    
    assert processed == len(events_to_test)
    assert process_time < 10.0  # Should process batch reasonably quickly


def test_database_write_performance(db_session):
    """Test database write performance."""
    from storage.db import Prediction
    
    start_time = time.time()
    
    # Write 50 predictions
    for i in range(50):
        pred = Prediction(
            features='{"age": 39}',
            prediction="<=50K",
            probability='{"<=50K": 0.85}',
            timestamp=datetime.utcnow()
        )
        db_session.add(pred)
    
    db_session.commit()
    write_time = time.time() - start_time
    
    assert write_time < 2.0  # Should write reasonably quickly


def test_drift_detection_performance():
    """Test drift detection performance with large datasets."""
    import pandas as pd
    
    # Create larger datasets
    reference_data = pd.DataFrame({
        'age': range(1000),
        'education': ['Bachelors'] * 1000
    })
    
    production_data = pd.DataFrame({
        'age': range(1000, 2000),
        'education': ['Masters'] * 1000
    })
    
    from monitoring.drift import detect_drift
    
    start_time = time.time()
    drift_scores = detect_drift(reference_data, production_data)
    drift_time = time.time() - start_time
    
    assert drift_time < 5.0  # Should complete reasonably quickly
    assert len(drift_scores) > 0
