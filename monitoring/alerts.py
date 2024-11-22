"""
Alerting module.

Provides interfaces for generating and managing alerts based on drift and performance degradation.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel
from sqlalchemy.orm import Session

from storage.db import Alert as AlertModel, store_drift_signal

logger = logging.getLogger(__name__)


class Alert(BaseModel):
    """Alert data model."""
    
    alert_type: str
    severity: str
    message: str
    timestamp: str
    metadata: Optional[Dict] = None


def check_drift_alerts(
    session: Session,
    drift_scores: Dict[str, float],
    threshold: float = 0.05,
    severity_threshold: float = 0.01
) -> List[Alert]:
    """
    Generate alerts based on drift detection results.
    
    Args:
        session: Database session
        drift_scores: Dictionary of feature names to drift p-values
        threshold: P-value threshold for warning alerts
        severity_threshold: P-value threshold for critical alerts
        
    Returns:
        List of drift alerts
    """
    alerts = []
    timestamp = datetime.utcnow().isoformat()
    
    for feature_name, p_value in drift_scores.items():
        if p_value < severity_threshold:
            # Critical drift
            alert = Alert(
                alert_type="drift",
                severity="critical",
                message=f"Critical drift detected in feature '{feature_name}' (p-value: {p_value:.4f})",
                timestamp=timestamp,
                metadata={"feature": feature_name, "p_value": float(p_value), "threshold": severity_threshold}
            )
            alerts.append(alert)
        elif p_value < threshold:
            # Warning drift
            alert = Alert(
                alert_type="drift",
                severity="warning",
                message=f"Drift detected in feature '{feature_name}' (p-value: {p_value:.4f})",
                timestamp=timestamp,
                metadata={"feature": feature_name, "p_value": float(p_value), "threshold": threshold}
            )
            alerts.append(alert)
    
    # Store alerts in database
    for alert in alerts:
        store_alert(session, alert)
    
    if alerts:
        logger.warning(f"Generated {len(alerts)} drift alerts")
    
    return alerts


def check_performance_alerts(
    session: Session,
    current_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    degradation_threshold: float = 0.05,
    critical_threshold: float = 0.15
) -> List[Alert]:
    """
    Generate alerts based on performance degradation.
    
    Args:
        session: Database session
        current_metrics: Current performance metrics
        baseline_metrics: Baseline metrics
        degradation_threshold: Relative degradation threshold for warnings
        critical_threshold: Relative degradation threshold for critical alerts
        
    Returns:
        List of performance alerts
    """
    from monitoring.metrics import compare_to_baseline
    
    alerts = []
    timestamp = datetime.utcnow().isoformat()
    
    # Check for degradation
    for metric_name in current_metrics:
        if metric_name not in baseline_metrics:
            continue
        
        current_val = current_metrics[metric_name]
        baseline_val = baseline_metrics[metric_name]
        
        if baseline_val == 0:
            continue
        
        relative_change = (current_val - baseline_val) / baseline_val
        
        if relative_change < -critical_threshold:
            # Critical degradation
            alert = Alert(
                alert_type="performance",
                severity="critical",
                message=f"Critical performance degradation in '{metric_name}': "
                       f"{current_val:.4f} vs baseline {baseline_val:.4f} "
                       f"({relative_change*100:.2f}% change)",
                timestamp=timestamp,
                metadata={
                    "metric": metric_name,
                    "current_value": float(current_val),
                    "baseline_value": float(baseline_val),
                    "relative_change": float(relative_change),
                    "threshold": critical_threshold
                }
            )
            alerts.append(alert)
        elif relative_change < -degradation_threshold:
            # Warning degradation
            alert = Alert(
                alert_type="performance",
                severity="warning",
                message=f"Performance degradation in '{metric_name}': "
                       f"{current_val:.4f} vs baseline {baseline_val:.4f} "
                       f"({relative_change*100:.2f}% change)",
                timestamp=timestamp,
                metadata={
                    "metric": metric_name,
                    "current_value": float(current_val),
                    "baseline_value": float(baseline_val),
                    "relative_change": float(relative_change),
                    "threshold": degradation_threshold
                }
            )
            alerts.append(alert)
    
    # Store alerts in database
    for alert in alerts:
        store_alert(session, alert)
    
    if alerts:
        logger.warning(f"Generated {len(alerts)} performance alerts")
    
    return alerts


def send_alert(alert: Alert, channel: str = "log") -> None:
    """
    Send alert through specified channel.
    
    Args:
        alert: Alert object to send
        channel: Alert channel ("log", "email", "webhook", etc.)
    """
    if channel == "log":
        if alert.severity == "critical":
            logger.critical(f"[ALERT] {alert.alert_type.upper()} - {alert.severity.upper()}: {alert.message}")
        elif alert.severity == "warning":
            logger.warning(f"[ALERT] {alert.alert_type.upper()} - {alert.severity.upper()}: {alert.message}")
        else:
            logger.info(f"[ALERT] {alert.alert_type.upper()}: {alert.message}")
    elif channel == "email":
        # TODO: Implement email sending
        logger.info(f"Email alert not implemented, logging instead: {alert.message}")
        send_alert(alert, channel="log")
    elif channel == "webhook":
        # TODO: Implement webhook
        logger.info(f"Webhook alert not implemented, logging instead: {alert.message}")
        send_alert(alert, channel="log")
    else:
        logger.warning(f"Unknown alert channel '{channel}', using log")
        send_alert(alert, channel="log")


def store_alert(session: Session, alert: Alert) -> None:
    """
    Store alert in database.
    
    Args:
        session: Database session
        alert: Alert object to store
    """
    try:
        timestamp = datetime.fromisoformat(alert.timestamp.replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        timestamp = datetime.utcnow()
    
    alert_record = AlertModel(
        timestamp=timestamp,
        alert_type=alert.alert_type,
        severity=alert.severity,
        message=alert.message,
        alert_metadata=json.dumps(alert.metadata) if alert.metadata else None
    )
    
    session.add(alert_record)
    session.commit()


def get_recent_alerts(
    session: Session,
    limit: int = 100,
    alert_type: Optional[str] = None
) -> List[Alert]:
    """
    Retrieve recent alerts.
    
    Args:
        session: Database session
        limit: Maximum number of alerts to return
        alert_type: Optional filter by alert type
        
    Returns:
        List of recent alerts
    """
    query = session.query(AlertModel)
    
    if alert_type:
        query = query.filter(AlertModel.alert_type == alert_type)
    
    query = query.order_by(AlertModel.timestamp.desc()).limit(limit)
    
    alert_records = query.all()
    
    alerts = []
    for alert_record in alert_records:
        try:
            metadata = json.loads(alert_record.alert_metadata) if alert_record.alert_metadata else None
        except (json.JSONDecodeError, TypeError):
            metadata = None
        
        alert = Alert(
            alert_type=alert_record.alert_type,
            severity=alert_record.severity,
            message=alert_record.message,
            timestamp=alert_record.timestamp.isoformat(),
            metadata=metadata
        )
        alerts.append(alert)
    
    return alerts
