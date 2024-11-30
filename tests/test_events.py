"""Test event queue and async architecture."""

import threading
import time
from datetime import datetime

import pytest

from monitoring.events import EventQueue, PredictionEvent, get_event_queue, reset_event_queue


@pytest.fixture
def event_queue():
    """Create event queue for testing."""
    reset_event_queue()
    return EventQueue(maxsize=100)


@pytest.fixture
def sample_event():
    """Create sample prediction event."""
    return PredictionEvent(
        timestamp=datetime.utcnow(),
        features={"age": 39, "education": "Bachelors"},
        prediction="<=50K",
        probability={"<=50K": 0.85, ">50K": 0.15}
    )


def test_event_queue_enqueue_dequeue(event_queue, sample_event):
    """Test basic enqueue/dequeue operations."""
    assert event_queue.is_empty()
    assert event_queue.size() == 0
    
    event_queue.enqueue(sample_event, block=False)
    assert not event_queue.is_empty()
    assert event_queue.size() == 1
    
    dequeued = event_queue.dequeue(block=False)
    assert dequeued is not None
    assert dequeued.prediction == sample_event.prediction
    assert event_queue.is_empty()


def test_event_queue_batch_operations(event_queue, sample_event):
    """Test batch dequeue operations."""
    for _ in range(5):
        event_queue.enqueue(sample_event, block=False)
    
    assert event_queue.size() == 5
    
    batch = event_queue.dequeue_batch(3, block=False)
    assert len(batch) == 3
    assert event_queue.size() == 2
    
    batch = event_queue.dequeue_batch(10, block=False)
    assert len(batch) == 2
    assert event_queue.is_empty()


def test_event_queue_thread_safety(event_queue, sample_event):
    """Test concurrent access to event queue."""
    def enqueue_events():
        for _ in range(50):
            event_queue.enqueue(sample_event, block=False)
            time.sleep(0.001)
    
    def dequeue_events():
        dequeued = 0
        while dequeued < 50:
            events = event_queue.dequeue_batch(5, block=False)
            dequeued += len(events)
            time.sleep(0.001)
    
    thread1 = threading.Thread(target=enqueue_events)
    thread2 = threading.Thread(target=dequeue_events)
    
    thread1.start()
    thread2.start()
    
    thread1.join()
    thread2.join()
    
    assert event_queue.is_empty() or event_queue.size() <= 5


def test_event_queue_maxsize_enforcement():
    """Test queue size limits and full queue behavior."""
    reset_event_queue()
    queue = EventQueue(maxsize=5)
    
    sample_event = PredictionEvent(
        timestamp=datetime.utcnow(),
        features={"age": 39},
        prediction="<=50K",
        probability={"<=50K": 0.85}
    )
    
    for i in range(5):
        queue.enqueue(sample_event, block=False)
    
    assert queue.size() == 5
    assert queue.is_full()
    
    try:
        queue.enqueue(sample_event, block=False)
        assert False, "Should raise exception when queue is full"
    except Exception:
        pass


def test_prediction_event_serialization(sample_event):
    """Test PredictionEvent dataclass serialization."""
    assert sample_event.timestamp is not None
    assert sample_event.features == {"age": 39, "education": "Bachelors"}
    assert sample_event.prediction == "<=50K"
    assert sample_event.probability == {"<=50K": 0.85, ">50K": 0.15}
    assert sample_event.explanation is None
    assert sample_event.model_version == "v1"  # Default value is "v1"
    assert sample_event.label is None


def test_reset_event_queue():
    """Test global queue reset functionality."""
    queue1 = get_event_queue(maxsize=100)
    queue1.enqueue(PredictionEvent(
        timestamp=datetime.utcnow(),
        features={"age": 39},
        prediction="<=50K",
        probability={"<=50K": 0.85}
    ), block=False)
    
    assert queue1.size() > 0
    
    reset_event_queue()
    
    queue2 = get_event_queue(maxsize=100)
    assert queue2.size() == 0
