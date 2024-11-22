#!/usr/bin/env python3
"""
Monitoring workflow script.

Retrieves recent predictions, runs drift detection, computes metrics, checks alerts.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta

from monitoring.alerts import check_drift_alerts, check_performance_alerts, get_recent_alerts
from monitoring.drift import detect_drift, get_drifted_features
from monitoring.metrics import compare_to_baseline, get_latest_metrics, get_metrics_history
from model.train import load_training_data, prepare_features
from storage.db import get_predictions, initialize_database

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Run monitoring checks."""
    project_root = Path(__file__).parent.parent
    train_data_path = project_root / "data" / "raw" / "adult_train.csv"
    
    logger.info("Starting monitoring workflow")
    
    # Initialize database
    SessionLocal = initialize_database()
    session = SessionLocal()
    
    try:
        # Get recent predictions (production data)
        logger.info("Retrieving recent predictions...")
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)  # Last 7 days
        
        recent_predictions = get_predictions(session, start_date=start_date, limit=1000)
        
        if recent_predictions.empty:
            logger.warning("No recent predictions found for monitoring")
            return
        
        logger.info(f"Found {len(recent_predictions)} recent predictions")
        
        # Load reference data for drift detection
        logger.info("Loading reference data...")
        df_train = load_training_data(train_data_path)
        X_ref, _ = prepare_features(df_train, target_col="income")
        
        # Extract features from predictions for drift detection
        # Note: In production, you'd have separate feature storage
        # For now, this is a placeholder that requires feature data
        logger.warning("Drift detection requires feature data from predictions")
        logger.info("Monitoring workflow complete (drift detection requires feature storage)")
        
        # Get latest metrics and baseline
        logger.info("Checking metrics...")
        latest_metrics = get_latest_metrics(session)
        if latest_metrics:
            logger.info(f"Latest metrics: {latest_metrics}")
        
        # Get recent alerts
        recent_alerts = get_recent_alerts(session, limit=10)
        if recent_alerts:
            logger.warning(f"Found {len(recent_alerts)} recent alerts")
            for alert in recent_alerts[:5]:
                logger.info(f"Alert: {alert.alert_type} - {alert.severity}: {alert.message}")
        else:
            logger.info("No recent alerts")
    
    finally:
        session.close()
    
    logger.info("Monitoring workflow complete")


if __name__ == "__main__":
    main()
