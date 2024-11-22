"""
FastAPI application for model inference and monitoring.

Provides REST endpoints for predictions, explanations, metrics, and drift detection.
"""

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader
from sqlalchemy.orm import Session

from model.predict import load_model_for_inference, predict_batch, predict_single
from monitoring.alerts import check_drift_alerts, check_performance_alerts, get_recent_alerts
from monitoring.circuit_breaker import check_critical_drift
from monitoring.drift import detect_drift, get_drifted_features
from monitoring.events import EventQueue, PredictionEvent, get_event_queue, reset_event_queue
from monitoring.metrics import compute_metrics_from_labeled_predictions, get_metrics_history, record_metrics
from model.train import load_training_data, prepare_features
from storage.db import get_latest_metrics, store_drift_signal
from service.middleware import LoggingMiddleware
from service.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    DriftResponse,
    MetricsResponse,
    PredictionRequest,
    PredictionResponse,
)
from storage.db import initialize_database, get_predictions, store_prediction
from config.loader import get_config
from config.logging_setup import setup_logging

# Load configuration
config = get_config()

# Set up logging with config
log_level = config.get("logging.level", "INFO")
log_file = config.get("logging.file")
setup_logging(log_level=log_level, log_file=log_file)

logger = logging.getLogger(__name__)

# Global variables for model and database session
MODEL_PATH = Path(config.get("model.path", "model/model.pkl"))
model = None
preprocessor = None
label_encoder = None
SessionLocal = None
event_queue = None
async_mode = None
inprocess_processor = None
processor_thread = None

# Templates
jinja_env = Environment(loader=FileSystemLoader("service/templates"))

app = FastAPI(
    title="Model Health Monitor API",
    description="API for tabular classifier inference and monitoring",
    version="0.1.0"
)

# Add logging middleware
app.add_middleware(LoggingMiddleware)


def get_db_session() -> Session:
    """Get database session."""
    global SessionLocal
    if SessionLocal is None:
        db_path = config.get("database.path", "sqlite:///monitor.db")
        SessionLocal = initialize_database(db_path)
    return SessionLocal()


def _is_worker_running() -> bool:
    """Check if external worker process is running."""
    project_root = Path(__file__).parent.parent
    stats_file = project_root / "worker_stats.json"
    if stats_file.exists():
        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                return stats.get("running", False)
        except Exception:
            return False
    return False


class InProcessEventProcessor:
    """In-process event processor that runs as a background thread."""
    
    def __init__(self, event_queue: EventQueue, session_factory, poll_interval: float = 1.0, batch_size: int = 50):
        self.event_queue = event_queue
        self.session_factory = session_factory
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self.running = False
        self.thread = None
        self.config = get_config()
        self.predictions_since_drift_check = 0
        self.drift_check_threshold = self.config.get("monitoring.drift.check_threshold", 100)
    
    def start(self):
        """Start the processor thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info("In-process event processor started")
    
    def stop(self):
        """Stop the processor thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        logger.info("In-process event processor stopped")
    
    def _run(self):
        """Main processing loop."""
        while self.running:
            try:
                events = self.event_queue.dequeue_batch(self.batch_size, block=False)
                if events:
                    self._process_batch(events)
                else:
                    time.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"Error in in-process event processor: {e}", exc_info=True)
                time.sleep(self.poll_interval)
    
    def _process_batch(self, events: List[PredictionEvent]):
        """Process a batch of events."""
        if not events:
            return
        
        session = self.session_factory()
        processed = 0
        
        try:
            for event in events:
                try:
                    store_prediction(
                        session=session,
                        features=event.features,
                        prediction=event.prediction,
                        probability=event.probability,
                        explanation=event.explanation,
                        label=event.label,
                        timestamp=event.timestamp
                    )
                    processed += 1
                except Exception as e:
                    logger.error(f"Error storing event: {e}", exc_info=True)
                    continue
            
            session.commit()
            logger.debug(f"In-process processor: stored {processed}/{len(events)} events")
            
            self.predictions_since_drift_check += processed
            
            labeled_count = sum(1 for event in events if event.label is not None)
            if labeled_count > 0:
                try:
                    window_size = self.config.get("monitoring.metrics.live_window", 1000)
                    live_metrics = compute_metrics_from_labeled_predictions(session, window_size=window_size)
                    if live_metrics:
                        record_metrics(session, live_metrics, timestamp=datetime.utcnow())
                        logger.debug(f"Updated live metrics from {labeled_count} labeled predictions")
                except Exception as e:
                    logger.warning(f"Failed to compute live metrics: {e}", exc_info=True)
            
            if self.predictions_since_drift_check >= self.drift_check_threshold:
                logger.info(f"Drift check threshold reached: {self.predictions_since_drift_check} >= {self.drift_check_threshold}")
                try:
                    self._run_drift_detection(session)
                    self.predictions_since_drift_check = 0
                except Exception as e:
                    logger.error(f"Failed to run drift detection: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Batch processing failed: {e}", exc_info=True)
            session.rollback()
        finally:
            session.close()
    
    def _load_reference_data(self):
        """Load reference training data for drift detection."""
        try:
            train_path = self.config.get("data.train_path", "data/raw/adult_train.csv")
            target_col = self.config.get("data.target_column", "income")
            logger.debug(f"Loading reference data from: {train_path}")
            # Convert string path to Path object
            train_path = Path(train_path)
            df = load_training_data(train_path)
            if df is None or df.empty:
                logger.warning(f"Reference data is None or empty from {train_path}")
                return None
            # prepare_features returns (X, y) tuple, we only need X (features)
            X_features, _ = prepare_features(df, target_col=target_col)
            logger.debug(f"Reference data prepared: {len(X_features)} rows, {len(X_features.columns)} columns")
            return X_features
        except Exception as e:
            logger.error(f"Failed to load reference data for drift detection: {e}", exc_info=True)
            return None
    
    def _get_production_data(self, session, limit: int = 1000):
        """Get recent production predictions for drift detection."""
        try:
            from storage.db import Prediction as PredictionModel
            from sqlalchemy import func
            
            recent_predictions = session.query(PredictionModel).order_by(
                PredictionModel.timestamp.desc()
            ).limit(limit).all()
            
            if len(recent_predictions) < 10:
                return None
            
            import pandas as pd
            import json
            
            features_list = []
            for pred in recent_predictions:
                try:
                    features = json.loads(pred.features) if isinstance(pred.features, str) else pred.features
                    features_list.append(features)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse features from prediction: {e}")
                    continue
            
            if not features_list:
                return None
            
            prod_df = pd.DataFrame(features_list)
            return prod_df
        except Exception as e:
            logger.warning(f"Failed to get production data: {e}")
            return None
    
    def _run_drift_detection(self, session):
        """Run drift detection between reference and production data."""
        logger.info("Running drift detection...")
        reference_data = self._load_reference_data()
        if reference_data is None:
            logger.warning("Skipping drift detection: reference data unavailable")
            return
        
        logger.debug(f"Reference data loaded: {len(reference_data)} rows, {len(reference_data.columns)} columns")
        
        production_data = self._get_production_data(session)
        if production_data is None:
            logger.warning("Skipping drift detection: insufficient production data (need at least 10 predictions)")
            return
        
        logger.debug(f"Production data loaded: {len(production_data)} rows, {len(production_data.columns)} columns")
        
        drift_threshold = self.config.get("monitoring.drift.threshold", 0.05)
        drift_method = self.config.get("monitoring.drift.method", "ks")
        
        logger.info(f"Running drift detection: method={drift_method}, threshold={drift_threshold}")
        drift_scores = detect_drift(
            reference_data=reference_data,
            production_data=production_data,
            threshold=drift_threshold,
            method=drift_method
        )
        
        logger.info(f"Drift detection completed: {len(drift_scores)} features tested")
        if drift_scores:
            logger.debug(f"Drift scores: {drift_scores}")
            from storage.db import store_drift_signal
            drifted_features = get_drifted_features(drift_scores, threshold=drift_threshold)
            
            if drifted_features:
                limited_drift_scores = {}
                limited_drift_features = []
                for feature in drifted_features:
                    if len(limited_drift_features) < 1:
                        limited_drift_scores[feature] = drift_scores[feature]
                        limited_drift_features.append(feature)
                
                if limited_drift_scores:
                    store_drift_signal(session, limited_drift_scores, threshold=drift_threshold)
                    logger.info(f"Stored drift for {len(limited_drift_features)} feature(s): {limited_drift_features}")
                    
                    severity_threshold = self.config.get("monitoring.drift.severity_threshold", 0.001)
                    drift_alerts = check_drift_alerts(
                        session,
                        limited_drift_scores,
                        threshold=drift_threshold,
                        severity_threshold=severity_threshold
                    )
                    if drift_alerts:
                        logger.info(f"Generated {len(drift_alerts)} drift alerts")
                else:
                    logger.info("No features detected as drifted (all p-values >= threshold)")
            else:
                logger.info("No features detected as drifted (all p-values >= threshold)")
            
            session.commit()
        else:
            logger.warning("No drift scores returned from detect_drift()")


def _start_inprocess_processor():
    """Start the in-process event processor."""
    global inprocess_processor, processor_thread, event_queue, SessionLocal
    if inprocess_processor is not None:
        return
    
    poll_interval = config.get("monitoring.worker.poll_interval", 1.0)
    batch_size = config.get("monitoring.worker.batch_size", 50)
    
    def session_factory():
        return SessionLocal()
    
    inprocess_processor = InProcessEventProcessor(
        event_queue=event_queue,
        session_factory=session_factory,
        poll_interval=poll_interval,
        batch_size=batch_size
    )
    inprocess_processor.start()


def load_model_if_needed():
    """Load model if not already loaded."""
    global model, preprocessor, label_encoder
    if model is None and MODEL_PATH.exists():
        try:
            model, preprocessor, label_encoder = load_model_for_inference(MODEL_PATH)
            logger.info(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
    elif not MODEL_PATH.exists():
        logger.warning(f"Model file not found at {MODEL_PATH}")


@app.on_event("startup")
async def startup_event():
    """Initialize database, load model, and initialize event queue on startup."""
    global SessionLocal, event_queue, async_mode
    db_path = config.get("database.path", "sqlite:///monitor.db")
    SessionLocal = initialize_database(db_path)
    logger.info(f"Database initialized at {db_path}")
    
    try:
        from storage.db import Prediction, Metric, DriftSignal, Alert, Base
        import os
        from sqlalchemy import inspect
        
        session = SessionLocal()
        try:
            logger.info("Resetting database on startup to ensure clean state...")
            
            from sqlalchemy import text
            
            inspector = inspect(session.bind)
            existing_tables = inspector.get_table_names()
            
            for table in existing_tables:
                session.execute(text(f"DROP TABLE IF EXISTS {table}"))
            
            session.commit()
            
            Base.metadata.create_all(bind=session.bind)
            session.commit()
            
            logger.info("Database reset complete: tables recreated with current schema")
        except Exception as e:
            logger.error(f"Error resetting database: {e}", exc_info=True)
            session.rollback()
        finally:
            session.close()
        
        project_root = Path(__file__).parent.parent
        stats_file = project_root / "worker_stats.json"
        if stats_file.exists():
            try:
                os.remove(stats_file)
                logger.info("Deleted worker stats file")
            except Exception as e:
                logger.warning(f"Failed to delete worker stats file: {e}")
    except Exception as e:
        logger.error(f"Error during startup reset: {e}", exc_info=True)
    
    load_model_if_needed()
    
    async_mode = config.get("monitoring.async_mode", True)
    if async_mode:
        queue_maxsize = config.get("monitoring.event_queue_maxsize", 10000)
        
        global inprocess_processor
        if inprocess_processor is not None:
            logger.info("Stopping existing in-process processor...")
            inprocess_processor.stop()
            inprocess_processor = None
        
        reset_event_queue()
        event_queue = get_event_queue(maxsize=queue_maxsize)
        logger.info(f"Async monitoring enabled with event queue (maxsize={queue_maxsize}, reset on startup)")
        
        if not _is_worker_running():
            logger.info("External worker not detected, starting in-process event processor")
            _start_inprocess_processor()
        else:
            logger.info("External worker detected, using separate worker process")
    else:
        logger.info("Sync monitoring mode (events processed in request path)")


@app.on_event("shutdown")
async def shutdown_event():
    """Reset database and stop in-process processor on server shutdown."""
    global inprocess_processor
    
    if inprocess_processor is not None:
        inprocess_processor.stop()
        inprocess_processor = None
    
    global SessionLocal
    if SessionLocal is None:
        return
    
    try:
        from storage.db import Base, Prediction, Metric, DriftSignal, Alert
        import os
        
        session = SessionLocal()
        try:
            logger.info("Resetting database on shutdown...")
            
            session.query(Prediction).delete()
            session.query(Metric).delete()
            session.query(DriftSignal).delete()
            session.query(Alert).delete()
            
            session.commit()
            logger.info("Database reset complete: all predictions, metrics, drift signals, and alerts cleared")
        except Exception as e:
            logger.error(f"Error resetting database: {e}", exc_info=True)
            session.rollback()
        finally:
            session.close()
        
        project_root = Path(__file__).parent.parent
        stats_file = project_root / "worker_stats.json"
        if stats_file.exists():
            try:
                os.remove(stats_file)
                logger.info("Cleared worker stats file")
            except Exception as e:
                logger.warning(f"Failed to remove worker stats file: {e}")
    except Exception as e:
        logger.error(f"Failed to reset database on shutdown: {e}", exc_info=True)


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Dashboard home page."""
    template = jinja_env.get_template("dashboard.html")
    return HTMLResponse(content=template.render(request=request))


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    
    Returns trust status: distinguishes "model is running" vs "model is safe to trust".
    Circuit breaker state is checked to determine if serving is blocked.
    
    Returns:
        Status dictionary with model, database, event queue, and trust status
    """
    model_status = "loaded" if model is not None else "not_loaded"
    
    trust_status = "trusted"
    circuit_breaker_active = False
    trust_message = None
    
    circuit_breaker_enabled = config.get("monitoring.circuit_breaker.enabled", True)
    if circuit_breaker_enabled and model is not None:
        session = get_db_session()
        try:
            severity_threshold = config.get("monitoring.circuit_breaker.severity_threshold", 0.01)
            lookback_hours = config.get("monitoring.circuit_breaker.lookback_hours", 24)
            
            is_critical, message = check_critical_drift(
                session,
                severity_threshold=severity_threshold,
                lookback_hours=lookback_hours
            )
            
            if is_critical:
                trust_status = "blocked"
                circuit_breaker_active = True
                trust_message = message
        except Exception as e:
            logger.warning(f"Error checking circuit breaker in health check: {e}")
            trust_status = "unknown"
            trust_message = "Unable to determine trust status"
        finally:
            session.close()
    
    response = {
        "status": "healthy" if trust_status != "blocked" else "degraded",
        "model": model_status,
        "database": "connected",
        "async_mode": async_mode if async_mode is not None else False,
        "trust_status": trust_status,
        "circuit_breaker_active": circuit_breaker_active
    }
    
    if trust_message:
        response["trust_message"] = trust_message
    
    if async_mode and event_queue is not None:
        response["event_queue"] = {
            "size": event_queue.size(),
            "maxsize": event_queue.maxsize,
            "is_full": event_queue.is_full()
        }
    
    return response


@app.get("/api/dashboard/stats")
def get_dashboard_stats() -> Dict[str, Any]:
    """Get dashboard statistics."""
    session = get_db_session()
    try:
        # Get total predictions count
        from sqlalchemy import func, distinct
        from storage.db import Prediction as PredictionModel, DriftSignal
        from datetime import timedelta
        
        total_predictions = session.query(func.count(PredictionModel.id)).scalar() or 0
        
        # Get active alerts (recent alerts with warning or critical severity)
        alerts = get_recent_alerts(session, limit=100)
        active_alerts = len([a for a in alerts if a.severity in ['warning', 'critical']])
        
        # Get drifted features count (features with drift detected in last 24 hours)
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        drifted_features_query = session.query(distinct(DriftSignal.feature_name)).filter(
            DriftSignal.timestamp >= cutoff_time,
            DriftSignal.is_drifted == True
        )
        drifted_features_count = drifted_features_query.count()
        
        model_status = "loaded" if model is not None else "not_loaded"
        
        queue_size = 0
        if async_mode and event_queue is not None:
            queue_size = event_queue.size()
        
        return {
            "model_status": model_status,
            "total_predictions": total_predictions,
            "active_alerts": active_alerts,
            "drifted_features": drifted_features_count,
            "queue_size": queue_size
        }
    finally:
        session.close()


@app.get("/api/dashboard/alerts")
def get_dashboard_alerts(limit: int = 10) -> list:
    """Get recent alerts for dashboard."""
    session = get_db_session()
    try:
        from datetime import timezone, timedelta
        alerts = get_recent_alerts(session, limit=limit)
        pst = timezone(timedelta(hours=-8))
        
        result = []
        for alert in alerts:
            alert_timestamp = datetime.fromisoformat(alert.timestamp.replace('Z', '+00:00'))
            if alert_timestamp.tzinfo is None:
                alert_timestamp = alert_timestamp.replace(tzinfo=timezone.utc)
            pst_timestamp = alert_timestamp.astimezone(pst)
            
            result.append({
                "timestamp": pst_timestamp.isoformat(),
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message
            })
        return result
    finally:
        session.close()


@app.get("/api/dashboard/drift")
def get_dashboard_drift() -> Dict[str, Any]:
    """Get drift status for dashboard."""
    session = get_db_session()
    try:
        from storage.db import DriftSignal
        from sqlalchemy import func, distinct
        from datetime import timedelta
        
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        recent_drift = session.query(DriftSignal).filter(
            DriftSignal.timestamp >= cutoff_time
        ).order_by(DriftSignal.timestamp.desc()).first()
        
        if recent_drift:
            drift_scores = {}
            all_drift = session.query(DriftSignal).filter(
                DriftSignal.timestamp == recent_drift.timestamp
            ).all()
            
            for drift in all_drift:
                drift_scores[drift.feature_name] = drift.drift_score
            
            drifted_features = get_drifted_features(drift_scores, threshold=0.05)
            
            from datetime import timezone, timedelta
            pst = timezone(timedelta(hours=-8))
            pst_timestamp = recent_drift.timestamp.replace(tzinfo=timezone.utc).astimezone(pst)
            
            return {
                "drifted_features": drifted_features,
                "drift_scores": drift_scores,
                "timestamp": pst_timestamp.isoformat()
            }
        
        pst = timezone(timedelta(hours=-8))
        pst_now = datetime.now(pst)
        return {
            "drifted_features": [],
            "drift_scores": {},
            "timestamp": pst_now.isoformat()
        }
    finally:
        session.close()


@app.post("/api/dashboard/undo-stream")
def undo_stream(request: Dict[str, Any]) -> Dict[str, Any]:
    """Delete predictions created since a given timestamp."""
    session = get_db_session()
    try:
        from storage.db import Prediction, Metric
        
        since_time_str = request.get("since")
        if not since_time_str:
            raise HTTPException(status_code=400, detail="Missing 'since' timestamp")
        
        from datetime import timedelta
        
        since_time = datetime.fromisoformat(since_time_str.replace('Z', '+00:00'))
        if since_time.tzinfo:
            since_time = since_time.replace(tzinfo=None) - timedelta(hours=since_time.utcoffset().total_seconds() / 3600)
        
        deleted = session.query(Prediction).filter(
            Prediction.timestamp >= since_time
        ).delete()
        
        session.query(Metric).filter(
            Metric.timestamp >= since_time
        ).delete()
        
        session.commit()
        
        return {
            "deleted": deleted,
            "since": since_time.isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid timestamp format: {str(e)}")
    except Exception as e:
        session.rollback()
        logger.error(f"Error undoing stream: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to undo stream: {str(e)}")
    finally:
        session.close()


@app.post("/api/dashboard/reset")
def reset_database(request: Request = None) -> Dict[str, Any]:
    """Reset database (clear all predictions, metrics, drift signals, and alerts)."""
    global inprocess_processor, event_queue
    
    logger.info("Resetting database via API request...")
    
    if inprocess_processor is not None:
        logger.info("Stopping in-process processor before reset...")
        inprocess_processor.stop()
        inprocess_processor = None
    
    if event_queue is not None:
        logger.info("Clearing event queue before reset...")
        queue_size = event_queue.size()
        for _ in range(queue_size):
            try:
                event_queue.dequeue(block=False)
            except:
                break
        reset_event_queue()
        queue_maxsize = config.get("monitoring.event_queue_maxsize", 10000)
        event_queue = get_event_queue(maxsize=queue_maxsize)
        logger.info(f"Event queue cleared: removed {queue_size} events")
    
    session = get_db_session()
    try:
        from storage.db import Prediction, Metric, DriftSignal, Alert, Base
        from sqlalchemy import inspect, text
        
        inspector = inspect(session.bind)
        existing_tables = inspector.get_table_names()
        
        for table in existing_tables:
            session.execute(text(f"DROP TABLE IF EXISTS {table}"))
        
        session.commit()
        
        Base.metadata.create_all(bind=session.bind)
        session.commit()
        
        logger.info("Database reset complete via API")
        
        if async_mode and event_queue is not None and not _is_worker_running():
            logger.info("Restarting in-process processor after reset...")
            _start_inprocess_processor()
        
        return {
            "status": "success",
            "message": "Database reset complete. All predictions, metrics, drift signals, and alerts have been cleared."
        }
    except Exception as e:
        session.rollback()
        logger.error(f"Error resetting database via API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reset database: {str(e)}")
    finally:
        session.close()


@app.get("/worker/health")
def worker_health() -> Dict[str, Any]:
    """
    Get worker health and statistics.
    
    Returns worker statistics from the background worker process.
    If worker is not running, returns status indicating worker is offline.
    """
    project_root = Path(__file__).parent.parent
    stats_file = project_root / "worker_stats.json"
    
    if not stats_file.exists():
        return {
            "status": "offline",
            "message": "Worker not running or stats file not found",
            "queue_size": event_queue.size() if event_queue else 0,
            "queue_maxsize": event_queue.maxsize if event_queue else 0
        }
    
    try:
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        stats["status"] = "online" if stats.get("running", False) else "offline"
        return stats
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to read worker stats: {e}")
        return {
            "status": "error",
            "message": f"Failed to read worker stats: {str(e)}",
            "queue_size": event_queue.size() if event_queue else 0,
            "queue_maxsize": event_queue.maxsize if event_queue else 0
        }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Generate prediction for single instance.
    
    Circuit breaker: If critical drift is detected, serving is halted (503) to prevent
    unreliable predictions. This implements an explicit trust boundary.
    
    In async mode (default), prediction is returned immediately and monitoring
    (drift detection, metrics, alerts) is handled by background workers.
    In sync mode, prediction is stored synchronously before returning.
    
    Args:
        request: Prediction request with features
        
    Returns:
        Prediction response with class, probabilities, and optional explanation
        
    Raises:
        HTTPException: 503 if circuit breaker active (critical drift), 500 if prediction fails
    """
    if model is None:
        load_model_if_needed()
        if model is None:
            raise HTTPException(status_code=503, detail="Model not available")
    
    circuit_breaker_enabled = config.get("monitoring.circuit_breaker.enabled", True)
    model_status = "healthy"
    is_critical = False
    circuit_breaker_message = None
    
    if circuit_breaker_enabled:
        session = get_db_session()
        try:
            severity_threshold = config.get("monitoring.circuit_breaker.severity_threshold", 0.01)
            lookback_hours = config.get("monitoring.circuit_breaker.lookback_hours", 24)
            
            is_critical, circuit_breaker_message = check_critical_drift(
                session,
                severity_threshold=severity_threshold,
                lookback_hours=lookback_hours
            )
            
            if is_critical:
                model_status = "blocked"
                logger.warning(f"Circuit breaker: blocking prediction request due to critical drift")
                raise HTTPException(
                    status_code=503,
                    detail=circuit_breaker_message or "Model serving halted: critical drift detected"
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error checking circuit breaker: {e}", exc_info=True)
            model_status = "unknown"
        finally:
            session.close()
    
    try:
        # Generate prediction (serving plane - fast path)
        prediction, probabilities, explanation = predict_single(
            model=model,
            preprocessor=preprocessor,
            label_encoder=label_encoder,
            features=request.features,
            include_explanation=request.include_explanation
        )
        
        # Handle monitoring based on async mode
        if async_mode and event_queue is not None:
            # Async mode: enqueue event for background worker (fire-and-forget)
            event = PredictionEvent(
                timestamp=datetime.utcnow(),
                features=request.features,
                prediction=prediction,
                probability=probabilities,
                explanation=explanation,
                model_version=config.get("model.version", "v1"),
                label=request.label
            )
            
            # Non-blocking enqueue (will drop event if queue is full)
            enqueued = event_queue.enqueue(event, block=False)
            if not enqueued:
                logger.warning("Event queue full, prediction event dropped (serving continues)")
                if model_status == "healthy":
                    model_status = "degraded"
        else:
            # Sync mode: store prediction synchronously (backward compatibility)
            session = get_db_session()
            try:
                store_prediction(
                    session=session,
                    features=request.features,
                    prediction=prediction,
                    probability=probabilities,
                    explanation=explanation,
                    label=request.label
                )
            finally:
                session.close()
        
        # Compute and update metrics if label provided (sync mode only)
        # In async mode, worker will compute metrics when processing events
        if request.label is not None and not async_mode:
            session = get_db_session()
            try:
                window_size = config.get("monitoring.metrics.live_window", 1000)
                live_metrics = compute_metrics_from_labeled_predictions(session, window_size=window_size)
                if live_metrics:
                    record_metrics(session, live_metrics, timestamp=datetime.utcnow())
                    logger.debug(f"Updated live metrics from {window_size} most recent labeled predictions")
            except Exception as e:
                logger.warning(f"Failed to compute live metrics: {e}", exc_info=True)
            finally:
                session.close()
        
        return PredictionResponse(
            prediction=prediction,
            probability=probabilities,
            explanation=explanation,
            model_status=model_status
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch_endpoint(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Generate predictions for batch of instances.
    
    Args:
        request: Batch prediction request
        
    Returns:
        Batch prediction response
        
    Raises:
        HTTPException: If batch prediction fails
    """
    if model is None:
        load_model_if_needed()
        if model is None:
            raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        # Generate predictions
        predictions, probabilities_list, explanations = predict_batch(
            model=model,
            preprocessor=preprocessor,
            label_encoder=label_encoder,
            features_list=request.instances,
            include_explanations=request.include_explanations
        )
        
        return BatchPredictionResponse(
            predictions=predictions,
            probabilities=probabilities_list,
            explanations=explanations
        )
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics_endpoint() -> MetricsResponse:
    """
    Retrieve current model performance metrics.
    
    Returns metrics as N/A if no metrics have been computed yet.
    Metrics start as N/A and update live as labeled predictions come in.
    
    Returns:
        Metrics response with performance indicators (N/A if not yet computed)
    """
    try:
        session = get_db_session()
        try:
            metrics = get_latest_metrics(session)
            
            if not metrics:
                metrics = {
                    "accuracy": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "roc_auc": None
                }
            
            return MetricsResponse(
                metrics=metrics,
                timestamp=datetime.utcnow().isoformat()
            )
        finally:
            session.close()
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")


@app.get("/drift", response_model=DriftResponse)
def check_drift_endpoint() -> DriftResponse:
    """
    Check for data drift in production data.
    
    This endpoint requires reference data and recent production predictions.
    For a full implementation, you would:
    1. Load reference training data
    2. Get recent production predictions from database
    3. Extract features from recent predictions
    4. Run drift detection
    
    Returns:
        Drift response with drifted features and scores
        
    Raises:
        HTTPException: If drift detection fails
    """
    try:
        # For demo purposes, this is a placeholder
        # In production, you would:
        # 1. Load reference data from training set
        # 2. Get recent production data from database
        # 3. Run detect_drift()
        
        session = get_db_session()
        try:
            # Get recent predictions (as example production data)
            recent_predictions = get_predictions(session, limit=100)
            
            if recent_predictions.empty:
                return DriftResponse(
                    drifted_features=[],
                    drift_scores={},
                    timestamp=datetime.utcnow().isoformat()
                )
            
            # This is a simplified version - in production you'd have reference data
            logger.warning("Drift detection requires reference data. Returning empty results.")
            
            return DriftResponse(
                drifted_features=[],
                drift_scores={},
                timestamp=datetime.utcnow().isoformat()
            )
        finally:
            session.close()
    except Exception as e:
        logger.error(f"Drift detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Drift detection failed: {str(e)}")


@app.post("/debug/inject_drift")
def inject_drift(
    feature_name: str = "age",
    drift_score: float = 0.005,
    severity_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    DEBUG: Manually inject a drift signal to trigger circuit breaker.
    
    This endpoint is for demonstration/testing only. It injects a drift signal
    directly into the database, which will immediately trigger the circuit breaker
    if drift_score < severity_threshold.
    
    Use this to test the trust boundary: after injection, /health will show
    trust_status: "blocked" and /predict will return 503.
    
    Args:
        feature_name: Feature name to mark as drifted
        drift_score: Drift score (p-value). Must be < severity_threshold to trigger circuit breaker
        severity_threshold: Override config threshold (default: uses config value)
        
    Returns:
        Status indicating drift signal was injected and whether circuit breaker is now active
    """
    if severity_threshold is None:
        severity_threshold = config.get("monitoring.circuit_breaker.severity_threshold", 0.01)
    
    if drift_score >= severity_threshold:
        raise HTTPException(
            status_code=400,
            detail=f"drift_score ({drift_score}) must be < severity_threshold ({severity_threshold}) to trigger circuit breaker"
        )
    
    session = get_db_session()
    try:
        drift_threshold = config.get("monitoring.drift.threshold", 0.05)
        
        store_drift_signal(
            session=session,
            drift_scores={feature_name: drift_score},
            threshold=drift_threshold,
            timestamp=datetime.utcnow()
        )
        
        if drift_score < severity_threshold:
            check_drift_alerts(
                session,
                {feature_name: drift_score},
                threshold=drift_threshold,
                severity_threshold=severity_threshold
            )
        
        is_critical, message = check_critical_drift(
            session,
            severity_threshold=severity_threshold,
            lookback_hours=config.get("monitoring.circuit_breaker.lookback_hours", 24)
        )
        
        return {
            "status": "injected",
            "message": f"Drift signal injected for feature '{feature_name}' with score {drift_score}",
            "circuit_breaker_active": is_critical,
            "trust_status": "blocked" if is_critical else "trusted",
            "detail": message if is_critical else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to inject drift: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to inject drift: {str(e)}")
    finally:
        session.close()


@app.post("/debug/clear_drift")
def clear_drift() -> Dict[str, Any]:
    """
    DEBUG: Clear recent drift signals to reset circuit breaker.
    
    Removes drift signals from the last 24 hours, effectively resetting
    the circuit breaker state. System will return to "trusted" status.
    
    Returns:
        Status indicating drift signals were cleared
    """
    from storage.db import DriftSignal
    from datetime import timedelta
    
    session = get_db_session()
    try:
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        deleted = session.query(DriftSignal).filter(
            DriftSignal.timestamp >= cutoff_time
        ).delete()
        
        session.commit()
        
        is_critical, _ = check_critical_drift(
            session,
            severity_threshold=config.get("monitoring.circuit_breaker.severity_threshold", 0.01),
            lookback_hours=config.get("monitoring.circuit_breaker.lookback_hours", 24)
        )
        
        return {
            "status": "cleared",
            "message": f"Cleared {deleted} drift signal(s) from last 24 hours",
            "circuit_breaker_active": is_critical,
            "trust_status": "blocked" if is_critical else "trusted"
        }
    except Exception as e:
        logger.error(f"Failed to clear drift: {e}", exc_info=True)
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to clear drift: {str(e)}")
    finally:
        session.close()
