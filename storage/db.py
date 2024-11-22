"""
Database interface module.

Provides abstractions for storing predictions, metrics, drift signals, and alerts.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    Index,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from typing import Callable

Base = declarative_base()


class Prediction(Base):
    """Prediction table model."""
    
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    features = Column(Text, nullable=False)  # JSON string
    prediction = Column(Text, nullable=False)
    probability = Column(Text, nullable=True)  # JSON string
    explanation = Column(Text, nullable=True)  # JSON string
    label = Column(Text, nullable=True, index=True)  # Optional ground truth label
    
    __table_args__ = (Index("idx_predictions_timestamp", "timestamp"),)


class Metric(Base):
    """Metric table model."""
    
    __tablename__ = "metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    
    __table_args__ = (
        Index("idx_metrics_timestamp", "timestamp"),
        Index("idx_metrics_name", "metric_name"),
    )


class DriftSignal(Base):
    """Drift signal table model."""
    
    __tablename__ = "drift_signals"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    feature_name = Column(String(100), nullable=False, index=True)
    drift_score = Column(Float, nullable=False)
    is_drifted = Column(Boolean, default=False, index=True)
    prediction_count_at_detection = Column(Integer, nullable=True, index=True)
    acknowledged = Column(Boolean, default=False, index=True)
    acknowledged_at = Column(DateTime, nullable=True, index=True)
    
    __table_args__ = (
        Index("idx_drift_timestamp", "timestamp"),
        Index("idx_drift_feature", "feature_name"),
        Index("idx_drift_drifted", "is_drifted"),
        Index("idx_drift_acknowledged", "acknowledged"),
    )


class Alert(Base):
    """Alert table model."""
    
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    alert_type = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)
    message = Column(Text, nullable=False)
    alert_metadata = Column(Text, nullable=True)  # JSON string (renamed from metadata - SQLAlchemy reserved word)
    
    __table_args__ = (
        Index("idx_alerts_timestamp", "timestamp"),
        Index("idx_alerts_type", "alert_type"),
        Index("idx_alerts_severity", "severity"),
    )


def initialize_database(db_path: str = "sqlite:///monitor.db") -> Callable[[], Session]:
    """
    Initialize database connection and create tables.
    
    Args:
        db_path: Database connection string (default: SQLite)
        
    Returns:
        Sessionmaker for creating database sessions
    """
    engine = create_engine(db_path, echo=False)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal


def store_prediction(
    session: Session,
    features: Dict[str, Any],
    prediction: Any,
    probability: Optional[Dict[str, float]] = None,
    explanation: Optional[Dict[str, float]] = None,
    label: Optional[str] = None,
    timestamp: Optional[datetime] = None
) -> None:
    """
    Store prediction with metadata.
    
    Args:
        session: Database session
        features: Input features
        prediction: Model prediction
        probability: Optional prediction probabilities
        explanation: Optional feature importance
        label: Optional ground truth label
        timestamp: Optional timestamp (defaults to now)
    """
    if timestamp is None:
        timestamp = datetime.utcnow()
    
    pred_record = Prediction(
        timestamp=timestamp,
        features=json.dumps(features),
        prediction=str(prediction),
        probability=json.dumps(probability) if probability else None,
        explanation=json.dumps(explanation) if explanation else None,
        label=str(label) if label else None,
    )
    
    session.add(pred_record)
    session.commit()


def store_metrics(
    session: Session,
    metrics: Dict[str, float],
    timestamp: Optional[datetime] = None
) -> None:
    """
    Store performance metrics.
    
    Args:
        session: Database session
        metrics: Dictionary of metric names to values
        timestamp: Optional timestamp (defaults to now)
    """
    if timestamp is None:
        timestamp = datetime.utcnow()
    
    for metric_name, metric_value in metrics.items():
        metric_record = Metric(
            timestamp=timestamp,
            metric_name=metric_name,
            metric_value=float(metric_value),
        )
        session.add(metric_record)
    
    session.commit()


def store_drift_signal(
    session: Session,
    drift_scores: Dict[str, float],
    threshold: float = 0.05,
    timestamp: Optional[datetime] = None,
    prediction_count: Optional[int] = None
) -> None:
    """
    Store drift detection results.
    
    Args:
        session: Database session
        drift_scores: Dictionary of feature names to drift scores (p-values or PSI)
        threshold: Threshold for determining if feature has drifted (default: 0.05)
        timestamp: Optional timestamp (defaults to now)
        prediction_count: Optional prediction count at detection time (for threshold-based recovery)
    """
    if timestamp is None:
        timestamp = datetime.utcnow()
    
    if prediction_count is None:
        from sqlalchemy import func
        prediction_count = session.query(func.count(Prediction.id)).scalar() or 0
    
    for feature_name, drift_score in drift_scores.items():
        # For p-values: lower = more drift. For PSI: higher = more drift.
        # Assuming p-values here (lower threshold = more drift)
        is_drifted = drift_score < threshold
        
        drift_record = DriftSignal(
            timestamp=timestamp,
            feature_name=feature_name,
            drift_score=float(drift_score),
            is_drifted=is_drifted,
            prediction_count_at_detection=prediction_count,
        )
        session.add(drift_record)
    
    session.commit()


def get_predictions(
    session: Session,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Retrieve stored predictions within date range.
    
    Args:
        session: Database session
        start_date: Start date for query
        end_date: End date for query
        limit: Optional maximum number of records
        
    Returns:
        DataFrame with predictions and metadata
    """
    query = session.query(Prediction)
    
    if start_date:
        query = query.filter(Prediction.timestamp >= start_date)
    if end_date:
        query = query.filter(Prediction.timestamp <= end_date)
    
    query = query.order_by(Prediction.timestamp.desc())
    
    if limit:
        query = query.limit(limit)
    
    predictions = query.all()
    
    # Convert to list of dicts
    records = []
    for pred in predictions:
        record = {
            "id": pred.id,
            "timestamp": pred.timestamp,
            "features": json.loads(pred.features),
            "prediction": pred.prediction,
            "probability": json.loads(pred.probability) if pred.probability else None,
            "explanation": json.loads(pred.explanation) if pred.explanation else None,
        }
        records.append(record)
    
    if not records:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=["id", "timestamp", "features", "prediction", "probability", "explanation"])
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    return df


def get_latest_metrics(session: Session) -> Dict[str, float]:
    """
    Retrieve most recent metrics.
    
    Args:
        session: Database session
        
    Returns:
        Dictionary of latest metric values
    """
    # Get the most recent timestamp
    latest_timestamp = session.query(Metric.timestamp).order_by(Metric.timestamp.desc()).first()
    
    if latest_timestamp is None:
        return {}
    
    # Get all metrics from that timestamp
    metrics = session.query(Metric).filter(Metric.timestamp == latest_timestamp[0]).all()
    
    return {metric.metric_name: metric.metric_value for metric in metrics}
