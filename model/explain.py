"""
Explanation module for model decision interpretation.

Provides interfaces for generating feature importance and instance-level explanations.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def explain_instance(
    model: BaseEstimator,
    preprocessor: ColumnTransformer,
    instance: pd.Series,
    feature_names: List[str],
    method: str = "permutation"
) -> Dict[str, float]:
    """
    Generate feature importance for a single prediction.
    
    Args:
        model: Trained classifier
        preprocessor: Preprocessing pipeline
        instance: Single feature vector (pd.Series with feature names)
        feature_names: List of feature names (in original order)
        method: Explanation method ("permutation" or "feature_importance")
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    # Convert single instance to DataFrame for preprocessing
    instance_df = instance.to_frame().T
    
    # Preprocess instance
    instance_processed = preprocessor.transform(instance_df)
    
    if method == "feature_importance":
        # Use model's built-in feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            # Map to original feature names
            # Note: preprocessing may change feature order/count
            feature_importance_dict = {}
            # For simplicity, map by position if feature count matches
            # This is a simplified approach - full implementation would track
            # how preprocessing transforms feature names
            if len(importances) == len(feature_names):
                feature_importance_dict = dict(zip(feature_names, importances))
            else:
                # Fall back to indexed names if counts don't match
                feature_importance_dict = {f"feature_{i}": imp for i, imp in enumerate(importances)}
        else:
            logger.warning("Model does not have feature_importances_, using permutation method")
            return explain_instance(model, preprocessor, instance, feature_names, method="permutation")
    else:
        # Permutation importance (requires background data - simplified version)
        # For single instance, use a simple approximation
        # In production, you'd use SHAP or a proper permutation importance with background data
        logger.warning("Permutation importance for single instance is approximated. "
                      "Consider using SHAP or providing background data.")
        
        # Simple approximation: use feature importances if available, otherwise equal weights
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            if len(importances) == len(feature_names):
                feature_importance_dict = dict(zip(feature_names, importances))
            else:
                feature_importance_dict = {f"feature_{i}": imp for i, imp in enumerate(importances)}
        else:
            # Fallback: equal importance
            feature_importance_dict = {name: 1.0 / len(feature_names) for name in feature_names}
    
    return feature_importance_dict


def explain_batch(
    model: BaseEstimator,
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    method: str = "permutation"
) -> pd.DataFrame:
    """
    Generate feature importance for a batch of predictions.
    
    Args:
        model: Trained classifier
        preprocessor: Preprocessing pipeline
        X: Feature matrix
        method: Explanation method ("permutation" or "feature_importance")
        
    Returns:
        DataFrame with feature importance scores per instance
    """
    # For batch explanations, use global feature importance
    # Individual instance explanations would require SHAP or similar
    logger.info(f"Generating batch explanations for {len(X)} instances using {method} method")
    
    if method == "feature_importance" and hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        # Create DataFrame with same feature importance for all instances
        # (This is a simplified approach - proper instance-level would use SHAP)
        importance_dict = {f"feature_{i}": imp for i, imp in enumerate(importances)}
        # Repeat for all instances
        explanations = [importance_dict.copy() for _ in range(len(X))]
        return pd.DataFrame(explanations)
    else:
        # Use global importance
        global_importance = get_global_importance(model, X, preprocessor, method=method)
        # Repeat for all instances (simplified - proper implementation would use SHAP)
        explanations = [global_importance.copy() for _ in range(len(X))]
        return pd.DataFrame(explanations)


def get_global_importance(
    model: BaseEstimator,
    X: pd.DataFrame,
    preprocessor: ColumnTransformer,
    method: str = "permutation"
) -> Dict[str, float]:
    """
    Compute global feature importance across dataset.
    
    Args:
        model: Trained classifier
        X: Feature matrix
        preprocessor: Preprocessing pipeline
        method: Importance computation method ("permutation" or "feature_importance")
        
    Returns:
        Dictionary mapping feature names to global importance scores
    """
    feature_names = X.columns.tolist()
    
    if method == "feature_importance":
        # Use model's built-in feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            # Map to original feature names
            if len(importances) == len(feature_names):
                return dict(zip(feature_names, importances))
            else:
                # Preprocessing changed feature count - use indexed names
                logger.warning(f"Feature count mismatch: model has {len(importances)}, "
                             f"original features: {len(feature_names)}")
                return {f"feature_{i}": imp for i, imp in enumerate(importances)}
        else:
            logger.warning("Model does not have feature_importances_, using permutation method")
            method = "permutation"
    
    if method == "permutation":
        # Compute permutation importance
        X_processed = preprocessor.transform(X)
        
        # For binary classification, use positive class probability
        if hasattr(model, 'predict_proba'):
            scoring = 'neg_log_loss'  # Works for classification
        else:
            scoring = 'accuracy'
        
        # Compute permutation importance
        # Note: This requires y_true, but for global importance we can use predictions
        # In practice, you'd want actual labels for more accurate importance
        try:
            y_pred = model.predict(X_processed)
            perm_importance = permutation_importance(
                model,
                X_processed,
                y_pred,
                n_repeats=10,
                random_state=42,
                n_jobs=-1,
                scoring=scoring
            )
            importances = perm_importance.importances_mean
            
            # Map to original feature names
            if len(importances) == len(feature_names):
                return dict(zip(feature_names, importances))
            else:
                return {f"feature_{i}": imp for i, imp in enumerate(importances)}
        except Exception as e:
            logger.warning(f"Permutation importance computation failed: {e}. Using feature_importances_ if available.")
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                if len(importances) == len(feature_names):
                    return dict(zip(feature_names, importances))
                else:
                    return {f"feature_{i}": imp for i, imp in enumerate(importances)}
            else:
                # Fallback: equal importance
                return {name: 1.0 / len(feature_names) for name in feature_names}
    
    # Default fallback
    return {name: 1.0 / len(feature_names) for name in feature_names}
