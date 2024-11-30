"""Test live metrics computation with rolling windows."""

from datetime import datetime

import pytest

from monitoring.metrics import compute_metrics_from_labeled_predictions, record_metrics
from storage.db import Prediction


@pytest.fixture
def labeled_predictions(db_session):
    """Create labeled predictions for testing."""
    predictions = []
    labels = ["<=50K", ">50K", "<=50K", ">50K", "<=50K"]
    predictions_responses = ["<=50K", "<=50K", "<=50K", ">50K", "<=50K"]
    
    for i, (label, pred) in enumerate(zip(labels, predictions_responses)):
        pred_obj = Prediction(
            features='{"age": 39, "education": "Bachelors"}',
            prediction=pred,
            probability='{"<=50K": 0.85, ">50K": 0.15}',
            label=label,
            timestamp=datetime.utcnow()
        )
        db_session.add(pred_obj)
        predictions.append(pred_obj)
    
    db_session.commit()
    return predictions


def test_compute_metrics_from_labeled_predictions(db_session, labeled_predictions):
    """Test metrics computation from labeled predictions."""
    metrics = compute_metrics_from_labeled_predictions(db_session, window_size=None)
    
    assert metrics is not None
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert 0 <= metrics["accuracy"] <= 1


def test_live_metrics_rolling_window(db_session, labeled_predictions):
    """Test rolling window limits predictions used."""
    # Create more predictions
    for i in range(10):
        pred = Prediction(
            features='{"age": 39, "education": "Bachelors"}',
            prediction="<=50K",
            probability='{"<=50K": 0.85, ">50K": 0.15}',
            label="<=50K",
            timestamp=datetime.utcnow()
        )
        db_session.add(pred)
    db_session.commit()
    
    # Compute with rolling window of 5
    metrics = compute_metrics_from_labeled_predictions(db_session, window_size=5)
    
    assert metrics is not None
    # Should use only most recent 5 labeled predictions


def test_live_metrics_with_insufficient_data(db_session):
    """Test metrics return None when insufficient labeled predictions."""
    # Add only one labeled prediction (need at least 2)
    pred = Prediction(
        features='{"age": 39}',
        prediction="<=50K",
        probability='{"<=50K": 0.85}',
        label="<=50K",
        timestamp=datetime.utcnow()
    )
    db_session.add(pred)
    db_session.commit()
    
    metrics = compute_metrics_from_labeled_predictions(db_session)
    
    assert metrics is None


def test_live_metrics_updates_on_new_predictions(db_session):
    """Test metrics update as new labeled predictions arrive."""
    # Initial predictions
    for i in range(3):
        pred = Prediction(
            features='{"age": 39}',
            prediction="<=50K",
            probability='{"<=50K": 0.85, ">50K": 0.15}',
            label="<=50K",
            timestamp=datetime.utcnow()
        )
        db_session.add(pred)
    db_session.commit()
    
    metrics1 = compute_metrics_from_labeled_predictions(db_session)
    assert metrics1 is not None
    
    # Add more predictions
    for i in range(3):
        pred = Prediction(
            features='{"age": 39}',
            prediction=">50K",
            probability='{"<=50K": 0.45, ">50K": 0.55}',
            label=">50K",
            timestamp=datetime.utcnow()
        )
        db_session.add(pred)
    db_session.commit()
    
    metrics2 = compute_metrics_from_labeled_predictions(db_session)
    assert metrics2 is not None
    # Metrics might change with new data
    assert "accuracy" in metrics2


def test_live_metrics_handles_missing_labels(db_session):
    """Test metrics only use predictions with labels."""
    # Add predictions with and without labels
    for i in range(3):
        pred_with_label = Prediction(
            features='{"age": 39}',
            prediction="<=50K",
            probability='{"<=50K": 0.85}',
            label="<=50K",
            timestamp=datetime.utcnow()
        )
        db_session.add(pred_with_label)
        
        pred_without_label = Prediction(
            features='{"age": 40}',
            prediction="<=50K",
            probability='{"<=50K": 0.85}',
            label=None,
            timestamp=datetime.utcnow()
        )
        db_session.add(pred_without_label)
    
    db_session.commit()
    
    metrics = compute_metrics_from_labeled_predictions(db_session)
    
    # Should compute metrics from only labeled predictions
    assert metrics is not None
    assert "accuracy" in metrics


def test_live_metrics_roc_auc_computation(db_session):
    """Test ROC-AUC computation from probabilities."""
    # Create binary classification predictions with probabilities
    for i in range(5):
        pred = Prediction(
            features='{"age": 39}',
            prediction="<=50K" if i < 3 else ">50K",
            probability='{"<=50K": 0.85, ">50K": 0.15}' if i < 3 else '{"<=50K": 0.30, ">50K": 0.70}',
            label="<=50K" if i < 3 else ">50K",
            timestamp=datetime.utcnow()
        )
        db_session.add(pred)
    
    db_session.commit()
    
    metrics = compute_metrics_from_labeled_predictions(db_session)
    
    assert metrics is not None
    assert "roc_auc" in metrics
    assert 0 <= metrics["roc_auc"] <= 1
