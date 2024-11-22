"""
Event queue abstraction for asynchronous monitoring pipeline.

Provides thread-safe event queue for decoupling prediction serving from
monitoring computation (drift detection, metrics aggregation, alerting).
"""

import logging
import queue
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PredictionEvent:
    """Event representing a single prediction for monitoring pipeline."""
    
    timestamp: datetime
    features: Dict[str, Any]
    prediction: Any
    probability: Optional[Dict[str, float]] = None
    explanation: Optional[Dict[str, float]] = None
    model_version: str = "v1"
    label: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "features": self.features,
            "prediction": self.prediction,
            "probability": self.probability,
            "explanation": self.explanation,
            "model_version": self.model_version,
            "label": self.label,
        }


class EventQueue:
    """
    Thread-safe in-memory event queue for prediction events.
    
    Wraps Python's queue.Queue to provide a simple interface for
    enqueueing prediction events and dequeuing them for processing
    by background workers.
    """
    
    def __init__(self, maxsize: int = 10000):
        """
        Initialize event queue.
        
        Args:
            maxsize: Maximum queue size. If full, enqueue will block
                    until space is available. Set to 0 for unlimited.
        """
        self._queue = queue.Queue(maxsize=maxsize)
        self.maxsize = maxsize
        logger.info(f"EventQueue initialized with maxsize={maxsize}")
    
    def enqueue(self, event: PredictionEvent, block: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Enqueue a prediction event.
        
        Args:
            event: PredictionEvent to enqueue
            block: If True, block until space is available. If False, raise queue.Full
            timeout: Maximum time to wait if block=True (None = wait indefinitely)
            
        Returns:
            True if event was enqueued successfully, False if queue was full and block=False
            
        Raises:
            queue.Full: If queue is full and block=False
        """
        try:
            self._queue.put(event, block=block, timeout=timeout)
            logger.debug(f"Event enqueued: prediction={event.prediction}, timestamp={event.timestamp}")
            return True
        except queue.Full:
            logger.warning(f"Event queue full (size={self.size()}), event dropped")
            return False
    
    def dequeue(self, block: bool = True, timeout: Optional[float] = None) -> Optional[PredictionEvent]:
        """
        Dequeue a prediction event.
        
        Args:
            block: If True, block until an event is available. If False, return None if empty
            timeout: Maximum time to wait if block=True (None = wait indefinitely)
            
        Returns:
            PredictionEvent if available, None if queue is empty and block=False
        """
        try:
            event = self._queue.get(block=block, timeout=timeout)
            logger.debug(f"Event dequeued: prediction={event.prediction}, timestamp={event.timestamp}")
            return event
        except queue.Empty:
            return None
    
    def dequeue_batch(self, batch_size: int, block: bool = True, timeout: Optional[float] = None) -> List[PredictionEvent]:
        """
        Dequeue multiple events in a batch.
        
        Args:
            batch_size: Maximum number of events to dequeue
            block: If True, block until at least one event is available
            timeout: Maximum time to wait for first event if block=True
            
        Returns:
            List of PredictionEvent objects (may be empty if queue is empty and block=False)
        """
        events = []
        
        if block:
            first_event = self.dequeue(block=True, timeout=timeout)
            if first_event:
                events.append(first_event)
        
        while len(events) < batch_size:
            event = self.dequeue(block=False)
            if event is None:
                break
            events.append(event)
        
        if events:
            logger.debug(f"Dequeued batch of {len(events)} events")
        
        return events
    
    def size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()
    
    def is_full(self) -> bool:
        """Check if queue is full."""
        return self._queue.full()


global_event_queue: Optional[EventQueue] = None


def get_event_queue(maxsize: int = 10000) -> EventQueue:
    """
    Get or create global event queue instance.
    
    Args:
        maxsize: Maximum queue size (only used on first call)
        
    Returns:
        Global EventQueue instance
    """
    global global_event_queue
    if global_event_queue is None:
        global_event_queue = EventQueue(maxsize=maxsize)
    return global_event_queue


def reset_event_queue():
    """Reset global event queue (useful for testing)."""
    global global_event_queue
    global_event_queue = None
