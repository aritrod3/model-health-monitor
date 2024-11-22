"""
Background worker for processing prediction events.

Consumes events from the event queue and handles:
- Storing predictions to database
- Aggregating metrics
- Running drift detection
- Generating alerts
"""

import json
import logging
import signal
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy.orm import Session

from config.loader import get_config
from monitoring.alerts import check_drift_alerts, check_performance_alerts
from monitoring.circuit_breaker import check_critical_drift
from monitoring.drift import detect_drift, get_drifted_features
from monitoring.events import EventQueue, PredictionEvent, get_event_queue
from monitoring.metrics import compute_metrics_from_labeled_predictions, record_metrics
from model.train import load_training_data, prepare_features
from storage.db import get_latest_metrics, get_predictions, initialize_database, store_drift_signal, store_prediction

logger = logging.getLogger(__name__)


class MonitoringWorker:
    """
    Background worker that processes prediction events asynchronously.
    
    Polls event queue, processes batches of events, and stores them to database.
    Designed to run in a separate process from the serving API.
    """
    
    def __init__(
        self,
        event_queue: Optional[EventQueue] = None,
        poll_interval: float = 5.0,
        batch_size: int = 50,
        db_path: Optional[str] = None,
        monitoring_interval: float = 300.0,
        monitoring_min_predictions: int = 100
    ):
        """
        Initialize monitoring worker.
        
        Args:
            event_queue: EventQueue instance (if None, uses global queue)
            poll_interval: Seconds between queue polls when queue is empty
            batch_size: Maximum number of events to process per batch
            db_path: Database connection string (if None, uses config)
            monitoring_interval: Seconds between monitoring runs (default: 300s = 5min)
            monitoring_min_predictions: Minimum predictions needed before running monitoring
        """
        self.event_queue = event_queue or get_event_queue()
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self.monitoring_interval = monitoring_interval
        self.monitoring_min_predictions = monitoring_min_predictions
        self.running = False
        self.shutdown_requested = False
        
        config = get_config()
        if db_path is None:
            db_path = config.get("database.path", "sqlite:///monitor.db")
        
        self.config = config
        self.SessionLocal = initialize_database(db_path)
        self.last_monitoring_time = None
        self.reference_data = None
        
        project_root = Path(__file__).parent.parent.parent
        self.stats_file = project_root / "worker_stats.json"
        
        self.stats = {
            "total_processed": 0,
            "total_failed": 0,
            "last_processed_time": None,
            "last_error_time": None,
            "last_error": None,
            "batches_processed": 0,
            "monitoring_runs": 0,
            "monitoring_errors": 0
        }
        
        logger.info(f"MonitoringWorker initialized: poll_interval={poll_interval}s, batch_size={batch_size}, "
                   f"monitoring_interval={monitoring_interval}s")
    
    def _retry_with_backoff(self, func, max_retries: int = 3, initial_delay: float = 1.0):
        """
        Retry a function with exponential backoff.
        
        Args:
            func: Function to retry (callable that takes no args)
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds (doubles each retry)
            
        Returns:
            Result of func() if successful
            
        Raises:
            Exception: If all retries fail
        """
        delay = initial_delay
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func()
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    logger.warning(f"Retry attempt {attempt + 1}/{max_retries} after {delay:.2f}s: {e}")
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"All {max_retries + 1} attempts failed")
        
        raise last_exception
    
    def process_event(self, event: PredictionEvent, session: Session) -> None:
        """
        Process a single prediction event with retry logic.
        
        Args:
            event: PredictionEvent to process
            session: Database session
        """
        def _store():
            store_prediction(
                session=session,
                features=event.features,
                prediction=event.prediction,
                probability=event.probability,
                explanation=event.explanation,
                label=event.label,
                timestamp=event.timestamp
            )
        
        try:
            self._retry_with_backoff(_store, max_retries=3, initial_delay=0.5)
            logger.debug(f"Stored prediction event: prediction={event.prediction}, label={event.label}, timestamp={event.timestamp}")
        except Exception as e:
            logger.error(f"Failed to store prediction event after retries: {e}", exc_info=True)
            self.stats["last_error_time"] = datetime.utcnow()
            self.stats["last_error"] = str(e)
            raise
    
    def process_batch(self, events: List[PredictionEvent]) -> int:
        """
        Process a batch of prediction events.
        
        Args:
            events: List of PredictionEvent objects to process
            
        Returns:
            Number of successfully processed events
        """
        if not events:
            return 0
        
        session = self.SessionLocal()
        processed = 0
        
        try:
            for event in events:
                try:
                    self.process_event(event, session)
                    processed += 1
                except Exception as e:
                    logger.error(f"Error processing event: {e}", exc_info=True)
                    continue
            
            session.commit()
            logger.info(f"Processed batch: {processed}/{len(events)} events stored")
            
            self.stats["total_processed"] += processed
            self.stats["total_failed"] += (len(events) - processed)
            self.stats["last_processed_time"] = datetime.utcnow()
            self.stats["batches_processed"] += 1
            
            labeled_count = sum(1 for event in events if event.label is not None)
            if labeled_count > 0:
                try:
                    window_size = self.config.get("monitoring.metrics.live_window", 1000)
                    live_metrics = compute_metrics_from_labeled_predictions(session, window_size=window_size)
                    if live_metrics:
                        record_metrics(session, live_metrics, timestamp=datetime.utcnow())
                        logger.info(f"Updated live metrics from {labeled_count} labeled predictions in batch")
                except Exception as e:
                    logger.warning(f"Failed to compute live metrics: {e}", exc_info=True)
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}", exc_info=True)
            session.rollback()
            self.stats["total_failed"] += len(events)
            self.stats["last_error_time"] = datetime.utcnow()
            self.stats["last_error"] = str(e)
        finally:
            session.close()
        
        return processed
    
    def run_once(self) -> int:
        """
        Process one batch of events from the queue.
        
        Returns:
            Number of events processed
        """
        if self.shutdown_requested:
            return 0
        
        events = self.event_queue.dequeue_batch(
            batch_size=self.batch_size,
            block=False
        )
        
        if not events:
            return 0
        
        return self.process_batch(events)
    
    def _load_reference_data(self) -> Optional[pd.DataFrame]:
        """Load reference training data for drift detection."""
        if self.reference_data is not None:
            return self.reference_data
        
        try:
            project_root = Path(__file__).parent.parent.parent
            train_data_path = project_root / self.config.get("data.train_path", "data/raw/adult_train.csv")
            target_col = self.config.get("data.target_column", "income")
            
            if not train_data_path.exists():
                logger.warning(f"Reference data not found at {train_data_path}, skipping drift detection")
                return None
            
            df_train = load_training_data(train_data_path)
            X_ref, _ = prepare_features(df_train, target_col=target_col)
            self.reference_data = X_ref
            logger.info(f"Loaded reference data: {len(X_ref)} rows, {len(X_ref.columns)} features")
            return X_ref
        except Exception as e:
            logger.error(f"Failed to load reference data: {e}", exc_info=True)
            return None
    
    def _get_production_data(self, session: Session, limit: int = 1000) -> Optional[pd.DataFrame]:
        """
        Get recent production predictions and convert to DataFrame for drift detection.
        
        Args:
            session: Database session
            limit: Maximum number of predictions to retrieve
            
        Returns:
            DataFrame with production features, or None if insufficient data
        """
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=7)
            
            predictions_df = get_predictions(session, start_date=start_date, limit=limit)
            
            if predictions_df.empty or len(predictions_df) < self.monitoring_min_predictions:
                logger.debug(f"Insufficient production data: {len(predictions_df)} predictions "
                           f"(need {self.monitoring_min_predictions})")
                return None
            
            features_list = []
            for _, row in predictions_df.iterrows():
                try:
                    features = json.loads(row['features']) if isinstance(row['features'], str) else row['features']
                    features_list.append(features)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse features from prediction: {e}")
                    continue
            
            if not features_list:
                return None
            
            prod_df = pd.DataFrame(features_list)
            logger.debug(f"Extracted {len(prod_df)} production feature vectors")
            return prod_df
        except Exception as e:
            logger.error(f"Failed to get production data: {e}", exc_info=True)
            return None
    
    def run_monitoring(self) -> None:
        """
        Run monitoring tasks: drift detection, metrics, and alerts.
        
        This method is called periodically to:
        1. Detect drift between reference and production data
        2. Store drift signals
        3. Check and generate alerts
        """
        if self.shutdown_requested:
            return
        
        try:
            session = self.SessionLocal()
            
            try:
                reference_data = self._load_reference_data()
                if reference_data is None:
                    logger.debug("Skipping monitoring: reference data unavailable")
                    return
                
                production_data = self._get_production_data(session)
                if production_data is None:
                    logger.debug("Skipping monitoring: insufficient production data")
                    return
                
                logger.info(f"Running drift detection: {len(reference_data)} reference vs {len(production_data)} production")
                
                drift_threshold = self.config.get("monitoring.drift.threshold", 0.05)
                drift_method = self.config.get("monitoring.drift.method", "ks")
                
                drift_scores = detect_drift(
                    reference_data=reference_data,
                    production_data=production_data,
                    threshold=drift_threshold,
                    method=drift_method
                )
                
                if drift_scores:
                    store_drift_signal(session, drift_scores, threshold=drift_threshold)
                    drifted_features = get_drifted_features(drift_scores, threshold=drift_threshold)
                    
                    if drifted_features:
                        logger.warning(f"Detected drift in {len(drifted_features)} features: {drifted_features}")
                    
                    severity_threshold = self.config.get("monitoring.drift.severity_threshold", 0.01)
                    drift_alerts = check_drift_alerts(
                        session,
                        drift_scores,
                        threshold=drift_threshold,
                        severity_threshold=severity_threshold
                    )
                    
                    if drift_alerts:
                        logger.warning(f"Generated {len(drift_alerts)} drift alerts")
                
                baseline_metrics = get_latest_metrics(session)
                if baseline_metrics:
                    logger.debug(f"Baseline metrics available: {list(baseline_metrics.keys())}")
                    logger.debug("Performance alerting requires current metrics with labels (not available in production)")
                
                logger.info("Monitoring cycle complete")
                self.stats["monitoring_runs"] += 1
                
            except Exception as e:
                logger.error(f"Monitoring cycle failed: {e}", exc_info=True)
                self.stats["monitoring_errors"] += 1
                self.stats["last_error_time"] = datetime.utcnow()
                self.stats["last_error"] = f"Monitoring: {str(e)}"
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Monitoring cycle failed (outer): {e}", exc_info=True)
            self.stats["monitoring_errors"] += 1
            self.stats["last_error_time"] = datetime.utcnow()
            self.stats["last_error"] = f"Monitoring: {str(e)}"
    
    def _save_stats(self) -> None:
        """Save worker statistics to file for API access."""
        try:
            stats = self.get_stats()
            with open(self.stats_file, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save stats to file: {e}")
    
    def get_stats(self) -> Dict:
        """Get worker statistics."""
        stats = self.stats.copy()
        stats["queue_size"] = self.event_queue.size()
        stats["queue_maxsize"] = self.event_queue.maxsize
        stats["queue_is_full"] = self.event_queue.is_full()
        stats["running"] = self.running
        stats["last_monitoring_time"] = self.last_monitoring_time.isoformat() if self.last_monitoring_time else None
        stats["last_processed_time"] = stats["last_processed_time"].isoformat() if stats["last_processed_time"] else None
        stats["last_error_time"] = stats["last_error_time"].isoformat() if stats["last_error_time"] else None
        return stats
    
    def run(self) -> None:
        """
        Main worker loop.
        
        Continuously polls event queue and processes events until shutdown is requested.
        Handles graceful shutdown on SIGTERM/SIGINT.
        """
        self.running = True
        logger.info("MonitoringWorker started")
        
        total_processed = 0
        consecutive_empty_polls = 0
        
        try:
            while self.running and not self.shutdown_requested:
                try:
                    processed = self.run_once()
                    
                    if processed > 0:
                        total_processed += processed
                        consecutive_empty_polls = 0
                        self._save_stats()
                    else:
                        consecutive_empty_polls += 1
                        if consecutive_empty_polls % 10 == 0:
                            logger.debug(f"Queue empty, waiting {self.poll_interval}s (total processed: {total_processed})")
                            self._save_stats()
                        time.sleep(self.poll_interval)
                    
                    if self.shutdown_requested:
                        break
                    
                    current_time = time.time()
                    if (self.last_monitoring_time is None or 
                        current_time - self.last_monitoring_time >= self.monitoring_interval):
                        self.run_monitoring()
                        self.last_monitoring_time = current_time
                        self._save_stats()
                
                except KeyboardInterrupt:
                    logger.info("Received keyboard interrupt, shutting down...")
                    self.shutdown_requested = True
                    break
                except Exception as e:
                    logger.error(f"Error in worker loop: {e}", exc_info=True)
                    time.sleep(self.poll_interval)
        
        finally:
            self.running = False
            self._save_stats()
            logger.info(f"MonitoringWorker stopped (total events processed: {total_processed})")
    
    def stop(self) -> None:
        """Request graceful shutdown."""
        logger.info("Shutdown requested")
        self.shutdown_requested = True
        self.running = False


def setup_signal_handlers(worker: MonitoringWorker) -> None:
    """
    Setup signal handlers for graceful shutdown.
    
    Args:
        worker: MonitoringWorker instance to shutdown on signal
    """
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down worker...")
        worker.stop()
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
