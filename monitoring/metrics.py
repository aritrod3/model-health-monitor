"""
Metrics collection and tracking module.

Provides interfaces for storing and retrieving model performance metrics over time.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from sqlalchemy.orm import Session

from storage.db import Metric, Prediction

logger = logging.getLogger(__name__)


def record_metrics(
    session: Session,
    metrics: Dict[str, float],
    timestamp: Optional[datetime] = None
) -> None:
    """
    Record performance metrics with timestamp.
    
    Args:
        session: Database session
        metrics: Dictionary of metric names to values
        timestamp: Optional timestamp (defaults to now)
    """
    from storage.db import store_metrics
    store_metrics(session, metrics, timestamp)
    logger.info(f"Recorded {len(metrics)} metrics at {timestamp or datetime.utcnow()}")


def get_metrics_history(
    session: Session,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    metric_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Retrieve metrics history within date range.
    
    Args:
        session: Database session
        start_date: Start date for query
        end_date: End date for query
        metric_names: Optional list of metric names to filter
        
    Returns:
        DataFrame with timestamp index and metric columns
    """
    query = session.query(Metric)
    
    if start_date:
        query = query.filter(Metric.timestamp >= start_date)
    if end_date:
        query = query.filter(Metric.timestamp <= end_date)
    if metric_names:
        query = query.filter(Metric.metric_name.in_(metric_names))
    
    query = query.order_by(Metric.timestamp.asc())
    
    metrics_records = query.all()
    
    if not metrics_records:
        return pd.DataFrame()
    
    # Convert to DataFrame
    records = [
        {
            'timestamp': m.timestamp,
            'metric_name': m.metric_name,
            'metric_value': m.metric_value
        }
        for m in metrics_records
    ]
    
    df = pd.DataFrame(records)
    
    # Pivot to have metrics as columns
    if len(df) > 0:
        df_pivot = df.pivot(index='timestamp', columns='metric_name', values='metric_value')
        df_pivot = df_pivot.sort_index()
        return df_pivot
    else:
        return pd.DataFrame()


def compute_metric_trend(
    session: Session,
    metric_name: str,
    window_size: int = 7
) -> Dict[str, float]:
    """
    Compute trend statistics for a metric over recent window.
    
    Args:
        session: Database session
        metric_name: Name of metric to analyze
        window_size: Number of days for trend window
        
    Returns:
        Dictionary with trend statistics (mean, std, slope, etc.)
    """
    from datetime import timedelta
    
    # Get metrics from recent window
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=window_size)
    
    df = get_metrics_history(session, start_date=start_date, end_date=end_date, metric_names=[metric_name])
    
    if df.empty or metric_name not in df.columns:
        logger.warning(f"No data found for metric '{metric_name}' in window")
        return {
            'mean': 0.0,
            'std': 0.0,
            'slope': 0.0,
            'min': 0.0,
            'max': 0.0,
            'count': 0
        }
    
    metric_values = df[metric_name].dropna()
    
    if len(metric_values) < 2:
        mean_val = float(metric_values.iloc[0]) if len(metric_values) == 1 else 0.0
        return {
            'mean': mean_val,
            'std': 0.0,
            'slope': 0.0,
            'min': mean_val,
            'max': mean_val,
            'count': len(metric_values)
        }
    
    # Compute statistics
    mean_val = float(metric_values.mean())
    std_val = float(metric_values.std())
    min_val = float(metric_values.min())
    max_val = float(metric_values.max())
    
    # Compute slope (linear trend)
    # Use index as x values (timestamps)
    x = np.arange(len(metric_values))
    y = metric_values.values
    slope = float(np.polyfit(x, y, 1)[0]) if len(y) > 1 else 0.0
    
    return {
        'mean': mean_val,
        'std': std_val,
        'slope': slope,
        'min': min_val,
        'max': max_val,
        'count': len(metric_values)
    }


def compare_to_baseline(
    current_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    threshold: float = 0.05
) -> Dict[str, bool]:
    """
    Compare current metrics to baseline and identify significant degradation.
    
    Args:
        current_metrics: Current performance metrics
        baseline_metrics: Baseline/reference metrics
        threshold: Relative degradation threshold (default: 5%)
        
    Returns:
        Dictionary mapping metric names to degradation flags
    """
    degradation_flags = {}
    
    for metric_name in current_metrics:
        if metric_name not in baseline_metrics:
            logger.warning(f"Metric '{metric_name}' not found in baseline metrics")
            degradation_flags[metric_name] = False
            continue
        
        current_val = current_metrics[metric_name]
        baseline_val = baseline_metrics[metric_name]
        
        if baseline_val == 0:
            logger.warning(f"Baseline value for '{metric_name}' is zero, skipping comparison")
            degradation_flags[metric_name] = False
            continue
        
        # For metrics where higher is better (accuracy, precision, recall, F1, ROC-AUC)
        # Degradation = current < baseline * (1 - threshold)
        # For metrics where lower is better, invert the logic
        relative_change = (current_val - baseline_val) / baseline_val
        
        # Assume higher is better (most ML metrics)
        is_degraded = relative_change < -threshold
        
        degradation_flags[metric_name] = is_degraded
        
        if is_degraded:
            logger.warning(f"Metric '{metric_name}' degraded: {current_val:.4f} vs baseline {baseline_val:.4f} "
                         f"({relative_change*100:.2f}% change)")
    
    return degradation_flags


def compute_metrics_from_labeled_predictions(
    session: Session,
    window_size: Optional[int] = None
) -> Optional[Dict[str, float]]:
    """
    Compute metrics from labeled predictions in database.
    
    Uses rolling window if specified, otherwise uses all labeled predictions.
    
    Args:
        session: Database session
        window_size: Optional number of most recent labeled predictions to use
        
    Returns:
        Dictionary of metrics (accuracy, precision, recall, f1, roc_auc) or None if insufficient data
    """
    from datetime import timedelta
    
    query = session.query(Prediction).filter(Prediction.label.isnot(None))
    
    if window_size is not None:
        query = query.order_by(Prediction.timestamp.desc()).limit(window_size)
    else:
        query = query.order_by(Prediction.timestamp.asc())
    
    labeled_predictions = query.all()
    
    if len(labeled_predictions) < 2:
        logger.debug(f"Insufficient labeled predictions ({len(labeled_predictions)}) for metrics computation")
        return None
    
    y_true_list = []
    y_pred_list = []
    y_proba_list = []
    
    for pred in labeled_predictions:
        try:
            y_true_list.append(pred.label)
            y_pred_list.append(pred.prediction)
            
            if pred.probability:
                prob_dict = json.loads(pred.probability)
                prob_values = list(prob_dict.values())
                y_proba_list.append(prob_values)
        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            logger.warning(f"Failed to parse prediction {pred.id}: {e}")
            continue
    
    if len(y_true_list) < 2:
        return None
    
    try:
        label_encoder = LabelEncoder()
        all_labels = list(set(y_true_list + y_pred_list))
        label_encoder.fit(all_labels)
        
        y_true_encoded = label_encoder.transform(y_true_list)
        y_pred_encoded = label_encoder.transform(y_pred_list)
        
        metrics = {
            'accuracy': float(accuracy_score(y_true_encoded, y_pred_encoded)),
            'precision': float(precision_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)),
        }
        
        if y_proba_list and len(y_proba_list) == len(y_true_list):
            try:
                y_proba_array = np.array(y_proba_list)
                if y_proba_array.ndim == 2 and y_proba_array.shape[1] >= 2:
                    if len(all_labels) == 2:
                        metrics['roc_auc'] = float(roc_auc_score(y_true_encoded, y_proba_array[:, 1]))
                    else:
                        metrics['roc_auc'] = float(roc_auc_score(y_true_encoded, y_proba_array, average='weighted', multi_class='ovr'))
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not compute ROC-AUC: {e}")
                metrics['roc_auc'] = 0.0
        
        logger.info(f"Computed metrics from {len(y_true_list)} labeled predictions: {metrics}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error computing metrics from labeled predictions: {e}", exc_info=True)
        return None
