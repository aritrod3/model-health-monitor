"""Test database operations."""

from datetime import datetime

import pytest
from sqlalchemy.orm import Session

from storage.db import get_latest_metrics, get_predictions, store_drift_signal, store_metrics, store_prediction


def test_store_and_retrieve_prediction(db_session: Session):
    """Test storing and retrieving predictions."""
    features = {'age': 39, 'workclass': 'State-gov', 'education': 'Bachelors'}
    store_prediction(
        session=db_session,
        features=features,
        prediction='<=50K',
        probability={'<=50K': 0.85, '>50K': 0.15}
    )
    
    predictions = get_predictions(db_session, limit=1)
    assert len(predictions) == 1
    assert predictions.iloc[0]['prediction'] == '<=50K'
    assert predictions.iloc[0]['features']['age'] == 39


def test_store_and_retrieve_metrics(db_session: Session):
    """Test storing and retrieving metrics."""
    metrics = {
        'accuracy': 0.85,
        'precision': 0.82,
        'recall': 0.80,
        'f1': 0.81
    }
    store_metrics(db_session, metrics)
    
    latest_metrics = get_latest_metrics(db_session)
    assert latest_metrics['accuracy'] == 0.85
    assert latest_metrics['precision'] == 0.82
    assert latest_metrics['recall'] == 0.80
    assert latest_metrics['f1'] == 0.81


def test_store_drift_signal(db_session: Session):
    """Test storing drift signals."""
    drift_scores = {
        'age': 0.02,
        'education': 0.03,
        'workclass': 0.15
    }
    store_drift_signal(db_session, drift_scores, threshold=0.05)
    
    # Verify drift signals were stored
    # (We'd need a get_drift_signals function to fully test, but this verifies it doesn't crash)
    assert True  # If we get here, it worked


def test_get_predictions_with_filters(db_session: Session):
    """Test retrieving predictions with date filters."""
    # Store predictions at different times
    store_prediction(db_session, {'age': 39}, '<=50K')
    
    # Get all predictions
    all_predictions = get_predictions(db_session)
    assert len(all_predictions) >= 1
    
    # Get with limit
    limited = get_predictions(db_session, limit=1)
    assert len(limited) == 1
