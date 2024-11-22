"""
Training module for tabular classifier.

Provides interfaces for model training, validation, and persistence.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

logger = logging.getLogger(__name__)


def load_training_data(data_path: Path) -> pd.DataFrame:
    """
    Load training data from CSV file.
    
    Args:
        data_path: Path to training CSV file
        
    Returns:
        DataFrame containing training data
        
    Raises:
        FileNotFoundError: If data_path does not exist
        ValueError: If file is empty or cannot be read
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    try:
        df = pd.read_csv(data_path, encoding="utf-8", na_values=[" ?", "?"])
    except UnicodeDecodeError:
        # Try latin-1 encoding as fallback
        logger.warning(f"UTF-8 decoding failed for {data_path}, trying latin-1")
        df = pd.read_csv(data_path, encoding="latin-1", na_values=[" ?", "?"])
    
    if df.empty:
        raise ValueError(f"Data file is empty: {data_path}")
    
    # Basic validation
    if df.columns.duplicated().any():
        logger.warning("Duplicate column names found in data")
    
    logger.info(f"Loaded data: {len(df)} rows, {len(df.columns)} columns from {data_path}")
    
    return df


def prepare_features(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target from training dataframe.
    
    Args:
        df: Training dataframe
        target_col: Name of target column
        
    Returns:
        Tuple of (features, target)
        
    Raises:
        ValueError: If target column does not exist
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data. Available columns: {list(df.columns)}")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Check for missing values (warn, but don't fail)
    missing_features = X.isnull().sum()
    if missing_features.any():
        logger.warning(f"Missing values found in features:\n{missing_features[missing_features > 0]}")
    
    if y.isnull().any():
        logger.warning(f"Missing values found in target column: {y.isnull().sum()}")
    
    logger.info(f"Prepared features: {len(X.columns)} features, {len(X)} samples")
    
    return X, y


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_config: Optional[Dict[str, Any]] = None
) -> Tuple[BaseEstimator, ColumnTransformer, LabelEncoder]:
    """
    Train a tabular classifier with preprocessing pipeline.
    
    Args:
        X: Feature matrix
        y: Target vector
        model_config: Optional model hyperparameters
        
    Returns:
        Tuple of (trained_model, preprocessing_pipeline, label_encoder)
    """
    if model_config is None:
        model_config = {}
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    logger.info(f"Categorical columns: {len(categorical_cols)}, Numerical columns: {len(numerical_cols)}")
    
    # Create preprocessing pipeline
    # Handle categorical: impute missing with 'missing', then ordinal encode
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    # Handle numerical: impute missing with median
    numerical_transformer = SimpleImputer(strategy='median')
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols),
        ],
        remainder='passthrough'
    )
    
    # Fit preprocessing pipeline
    X_processed = preprocessor.fit_transform(X)
    
    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Get model type and hyperparameters
    model_type = model_config.get('type', 'random_forest')
    model_params = {k: v for k, v in model_config.items() if k != 'type'}
    
    # Default hyperparameters
    if model_type == 'gradient_boosting':
        default_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42
        }
        default_params.update(model_params)
        model = GradientBoostingClassifier(**default_params)
    else:  # random_forest
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(model_params)
        model = RandomForestClassifier(**default_params)
    
    # Train model
    logger.info(f"Training {model_type} classifier...")
    model.fit(X_processed, y_encoded)
    logger.info("Model training complete")
    
    return model, preprocessor, label_encoder


def save_model(
    model: BaseEstimator,
    preprocessor: ColumnTransformer,
    label_encoder: LabelEncoder,
    model_path: Path
) -> None:
    """
    Persist trained model, preprocessing pipeline, and label encoder to disk.
    
    Args:
        model: Trained classifier
        preprocessor: Preprocessing pipeline
        label_encoder: Target label encoder
        model_path: Path to save model
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_package = {
        'model': model,
        'preprocessor': preprocessor,
        'label_encoder': label_encoder
    }
    
    joblib.dump(model_package, model_path)
    logger.info(f"Model saved to {model_path}")


def load_model(model_path: Path) -> Tuple[BaseEstimator, ColumnTransformer, LabelEncoder]:
    """
    Load persisted model, preprocessing pipeline, and label encoder from disk.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Tuple of (model, preprocessor, label_encoder)
        
    Raises:
        FileNotFoundError: If model_path does not exist
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model_package = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")
    
    return model_package['model'], model_package['preprocessor'], model_package['label_encoder']
