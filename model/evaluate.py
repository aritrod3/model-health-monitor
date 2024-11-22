"""
Evaluation module for model performance assessment.

Provides metrics calculation and validation utilities.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

import logging

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute classification performance metrics.
    
    Args:
        y_true: True labels (can be encoded or original)
        y_pred: Predicted labels (encoded)
        y_proba: Optional predicted probabilities (for ROC-AUC)
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }
    
    # Add ROC-AUC if probabilities are provided
    if y_proba is not None:
        try:
            # Handle binary and multiclass
            if y_proba.ndim == 1:
                # Binary classification
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))
            else:
                # Multiclass - use weighted average
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba, average='weighted', multi_class='ovr'))
        except ValueError as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")
            metrics['roc_auc'] = 0.0
    
    return metrics


def evaluate_model(
    model: BaseEstimator,
    preprocessor: ColumnTransformer,
    label_encoder: LabelEncoder,
    X: pd.DataFrame,
    y: pd.Series
) -> Dict[str, float]:
    """
    Evaluate model on test dataset.
    
    Args:
        model: Trained classifier
        preprocessor: Preprocessing pipeline
        label_encoder: Target label encoder
        X: Feature matrix
        y: True labels (original format)
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Preprocess features
    X_processed = preprocessor.transform(X)
    
    # Encode target labels
    y_encoded = label_encoder.transform(y)
    
    # Generate predictions
    y_pred = model.predict(X_processed)
    y_proba = model.predict_proba(X_processed)
    
    # Compute metrics
    metrics = compute_metrics(y_encoded, y_pred, y_proba)
    
    logger.info(f"Model evaluation metrics: {metrics}")
    
    return metrics


def compute_baseline_metrics(
    model: BaseEstimator,
    preprocessor: ColumnTransformer,
    label_encoder: LabelEncoder,
    X_baseline: pd.DataFrame,
    y_baseline: pd.Series
) -> Dict[str, float]:
    """
    Compute baseline performance metrics from training/validation data.
    
    Used as reference for production performance comparison.
    
    Args:
        model: Trained classifier
        preprocessor: Preprocessing pipeline
        label_encoder: Target label encoder
        X_baseline: Baseline feature matrix
        y_baseline: Baseline labels (original format)
        
    Returns:
        Dictionary of baseline metrics
    """
    # Use evaluate_model for baseline metrics
    return evaluate_model(model, preprocessor, label_encoder, X_baseline, y_baseline)
