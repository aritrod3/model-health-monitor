"""
Training module for tabular classifier.

Provides interfaces for model training, validation, and persistence.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from sklearn.base import BaseEstimator


def load_training_data(data_path: Path) -> pd.DataFrame:
    """
    Load training data from CSV file.
    
    Args:
        data_path: Path to training CSV file
        
    Returns:
        DataFrame containing training data
        
    Raises:
        FileNotFoundError: If data_path does not exist
    """
    pass


def prepare_features(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target from training dataframe.
    
    Args:
        df: Training dataframe
        target_col: Name of target column
        
    Returns:
        Tuple of (features, target)
    """
    pass


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_config: Optional[Dict[str, Any]] = None
) -> BaseEstimator:
    """
    Train a tabular classifier.
    
    Args:
        X: Feature matrix
        y: Target vector
        model_config: Optional model hyperparameters
        
    Returns:
        Trained classifier
    """
    pass


def save_model(model: BaseEstimator, model_path: Path) -> None:
    """
    Persist trained model to disk.
    
    Args:
        model: Trained classifier
        model_path: Path to save model
    """
    pass


def load_model(model_path: Path) -> BaseEstimator:
    """
    Load persisted model from disk.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded classifier
    """
    pass
