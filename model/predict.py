"""
Prediction module for model inference.

Provides interfaces for generating predictions with optional explanations.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

from model.explain import explain_instance, get_global_importance
from model.train import load_model

logger = logging.getLogger(__name__)


def predict_single(
    model: BaseEstimator,
    preprocessor: ColumnTransformer,
    label_encoder: LabelEncoder,
    features: Dict[str, Any],
    include_explanation: bool = False
) -> Tuple[Any, Dict[str, float], Optional[Dict[str, float]]]:
    """
    Generate prediction for a single instance.
    
    Args:
        model: Trained classifier
        preprocessor: Preprocessing pipeline
        label_encoder: Target label encoder
        features: Feature dictionary
        include_explanation: Whether to generate explanation
        
    Returns:
        Tuple of (prediction, probabilities_dict, explanation_dict)
    """
    # Convert features dict to DataFrame
    feature_df = pd.DataFrame([features])
    
    # Preprocess features
    feature_processed = preprocessor.transform(feature_df)
    
    # Generate prediction
    prediction_encoded = model.predict(feature_processed)[0]
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]
    
    # Generate probabilities
    probabilities_encoded = model.predict_proba(feature_processed)[0]
    probabilities = {
        label_encoder.inverse_transform([i])[0]: float(prob)
        for i, prob in enumerate(probabilities_encoded)
    }
    
    # Generate explanation if requested
    explanation = None
    if include_explanation:
        feature_series = pd.Series(features)
        feature_names = list(features.keys())
        explanation = explain_instance(
            model,
            preprocessor,
            feature_series,
            feature_names,
            method="feature_importance"
        )
    
    return prediction, probabilities, explanation


def predict_batch(
    model: BaseEstimator,
    preprocessor: ColumnTransformer,
    label_encoder: LabelEncoder,
    features_list: List[Dict[str, Any]],
    include_explanations: bool = False
) -> Tuple[List[Any], List[Dict[str, float]], Optional[List[Dict[str, float]]]]:
    """
    Generate predictions for a batch of instances.
    
    Args:
        model: Trained classifier
        preprocessor: Preprocessing pipeline
        label_encoder: Target label encoder
        features_list: List of feature dictionaries
        include_explanations: Whether to generate explanations
        
    Returns:
        Tuple of (predictions_list, probabilities_list, explanations_list)
    """
    # Convert features list to DataFrame
    feature_df = pd.DataFrame(features_list)
    
    # Preprocess features
    feature_processed = preprocessor.transform(feature_df)
    
    # Generate predictions
    predictions_encoded = model.predict(feature_processed)
    predictions = label_encoder.inverse_transform(predictions_encoded).tolist()
    
    # Generate probabilities
    probabilities_encoded = model.predict_proba(feature_processed)
    probabilities_list = []
    for probs in probabilities_encoded:
        prob_dict = {
            label_encoder.inverse_transform([i])[0]: float(prob)
            for i, prob in enumerate(probs)
        }
        probabilities_list.append(prob_dict)
    
    # Generate explanations if requested
    explanations = None
    if include_explanations:
        # Use global importance for all instances (simplified)
        # For proper instance-level explanations, use SHAP
        global_importance = get_global_importance(
            model,
            feature_df,
            preprocessor,
            method="feature_importance"
        )
        explanations = [global_importance.copy() for _ in range(len(features_list))]
    
    return predictions, probabilities_list, explanations


def load_model_for_inference(model_path: Path) -> Tuple[BaseEstimator, ColumnTransformer, LabelEncoder]:
    """
    Load model for inference.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Tuple of (model, preprocessor, label_encoder)
    """
    return load_model(model_path)
