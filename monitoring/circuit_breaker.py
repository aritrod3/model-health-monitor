"""
Circuit breaker for ML serving.

Implements trust boundary: stops serving predictions when critical drift is detected,
preventing unreliable model outputs from being used in production.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

from sqlalchemy import and_
from sqlalchemy.orm import Session

from storage.db import DriftSignal

logger = logging.getLogger(__name__)


def check_critical_drift(
    session: Session,
    severity_threshold: float = 0.01,
    lookback_hours: int = 24,
    recovery_predictions: Optional[int] = None,
    auto_acknowledge_seconds: Optional[int] = None
) -> Tuple[bool, Optional[str]]:
    """
    Check if critical drift is currently active.
    
    Circuit breaker logic: if any feature has drift score below severity_threshold
    within the lookback window, the model is considered unreliable.
    
    Args:
        session: Database session
        severity_threshold: P-value threshold for critical drift (default: 0.01)
        lookback_hours: Hours to look back for drift signals (default: 24, fallback if recovery_predictions not set)
        recovery_predictions: If set, ignore drift signals older than this many predictions (threshold-based recovery)
        
    Returns:
        Tuple of (is_critical_drift, message)
        - is_critical_drift: True if critical drift detected
        - message: Human-readable explanation if drift detected, None otherwise
    """
    try:
        from sqlalchemy import func
        from storage.db import Prediction as PredictionModel
        
        current_prediction_count = session.query(func.count(PredictionModel.id)).scalar() or 0
        
        if recovery_predictions is not None and recovery_predictions > 0:
            cutoff_prediction_count = current_prediction_count - recovery_predictions
            
            recent_critical_drift = session.query(DriftSignal).filter(
                and_(
                    DriftSignal.prediction_count_at_detection >= cutoff_prediction_count,
                    DriftSignal.drift_score < severity_threshold,
                    DriftSignal.is_drifted == True,
                    DriftSignal.acknowledged == False
                )
            ).all()
        else:
            cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)
            
            recent_critical_drift = session.query(DriftSignal).filter(
                and_(
                    DriftSignal.timestamp >= cutoff_time,
                    DriftSignal.drift_score < severity_threshold,
                    DriftSignal.is_drifted == True,
                    DriftSignal.acknowledged == False
                )
            ).all()
        
        if auto_acknowledge_seconds is not None and auto_acknowledge_seconds > 0:
            now = datetime.utcnow()
            for drift in recent_critical_drift:
                age_seconds = (now - drift.timestamp).total_seconds()
                if age_seconds >= auto_acknowledge_seconds:
                    drift.acknowledged = True
                    drift.acknowledged_at = now
                    session.commit()
                    logger.info(f"Auto-acknowledged drift signal for feature '{drift.feature_name}' after {age_seconds:.1f} seconds")
            
            recent_critical_drift = [d for d in recent_critical_drift if not d.acknowledged]
        
        if not recent_critical_drift:
            return False, None
        
        drifted_features = [signal.feature_name for signal in recent_critical_drift]
        max_drift = min([signal.drift_score for signal in recent_critical_drift])
        
        message = (
            f"Circuit breaker active: critical drift detected in {len(drifted_features)} feature(s) "
            f"({', '.join(drifted_features[:5])}{'...' if len(drifted_features) > 5 else ''}). "
            f"Minimum drift score: {max_drift:.4f} (threshold: {severity_threshold}). "
            f"Model serving halted to prevent unreliable predictions."
        )
        
        logger.warning(f"Circuit breaker triggered: {message}")
        return True, message
        
    except Exception as e:
        logger.error(f"Error checking critical drift: {e}", exc_info=True)
        return False, None
