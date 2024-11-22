#!/usr/bin/env python3
"""
Training workflow script.

Loads data, trains model, evaluates, and saves model with baseline metrics.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime

from config.loader import get_config
from model.evaluate import compute_baseline_metrics, evaluate_model
from model.train import load_training_data, prepare_features, save_model, train_model
from monitoring.metrics import record_metrics
from storage.db import initialize_database

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Train model and save with baseline metrics."""
    config = get_config()
    project_root = Path(__file__).parent.parent
    
    # Get paths from config
    train_data_path = project_root / config.get("data.train_path", "data/raw/adult_train.csv")
    test_data_path = project_root / config.get("data.test_path", "data/raw/adult_test.csv")
    model_path = project_root / config.get("model.path", "model/model.pkl")
    model_dir = model_path.parent
    target_col = config.get("data.target_column", "income")
    
    # Get model config
    model_type = config.get("model.type", "random_forest")
    model_config = {
        "type": model_type,
        **config.get(f"model.hyperparameters.{model_type}", {})
    }
    
    # Create model directory
    model_dir.mkdir(exist_ok=True)
    
    logger.info("Starting model training workflow")
    
    # Load training data
    logger.info(f"Loading training data from {train_data_path}")
    df_train = load_training_data(train_data_path)
    
    # Prepare features
    X_train, y_train = prepare_features(df_train, target_col=target_col)
    
    # Train model
    logger.info(f"Training {model_type} model...")
    model, preprocessor, label_encoder = train_model(X_train, y_train, model_config=model_config)
    logger.info("Model training complete")
    
    # Save model
    logger.info(f"Saving model to {model_path}")
    save_model(model, preprocessor, label_encoder, model_path)
    
    # Evaluate on test set
    logger.info(f"Loading test data from {test_data_path}")
    df_test = load_training_data(test_data_path)
    X_test, y_test = prepare_features(df_test, target_col=target_col)
    
    # Initialize database
    db_path = config.get("database.path", "sqlite:///monitor.db")
    SessionLocal = initialize_database(db_path)
    
    # Compute baseline metrics
    logger.info("Computing baseline metrics...")
    baseline_metrics = compute_baseline_metrics(model, preprocessor, label_encoder, X_test, y_test)
    logger.info(f"Baseline metrics: {baseline_metrics}")
    
    # Store baseline metrics in database
    logger.info("Storing baseline metrics in database...")
    session = SessionLocal()
    try:
        record_metrics(session, baseline_metrics, timestamp=datetime.utcnow())
        logger.info("Baseline metrics stored successfully")
    finally:
        session.close()
    
    logger.info("Training workflow complete!")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Baseline metrics: {baseline_metrics}")


if __name__ == "__main__":
    main()
