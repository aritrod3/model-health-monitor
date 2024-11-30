"""Test training workflow."""

from pathlib import Path

import pandas as pd
import pytest

from model.evaluate import compute_baseline_metrics, evaluate_model
from model.train import load_training_data, prepare_features, save_model, train_model


def test_load_training_data(temp_data_dir: Path, sample_data: pd.DataFrame):
    """Test loading training data."""
    # Save sample data to file
    csv_path = temp_data_dir / "test_data.csv"
    sample_data.to_csv(csv_path, index=False)
    
    # Load it
    df = load_training_data(csv_path)
    assert len(df) == len(sample_data)
    assert list(df.columns) == list(sample_data.columns)


def test_prepare_features(sample_data: pd.DataFrame):
    """Test feature preparation."""
    X, y = prepare_features(sample_data, target_col='income')
    
    assert 'income' not in X.columns
    assert 'income' in sample_data.columns
    assert len(X) == len(y)
    assert len(y) == len(sample_data)


def test_train_model(sample_data: pd.DataFrame):
    """Test model training."""
    X, y = prepare_features(sample_data, target_col='income')
    
    # Train model
    model, preprocessor, label_encoder = train_model(X, y)
    
    assert model is not None
    assert preprocessor is not None
    assert label_encoder is not None


def test_save_and_load_model(temp_model_dir: Path, sample_data: pd.DataFrame):
    """Test model saving and loading."""
    from model.train import load_model
    
    X, y = prepare_features(sample_data, target_col='income')
    model, preprocessor, label_encoder = train_model(X, y)
    
    # Save model
    model_path = temp_model_dir / "test_model.pkl"
    save_model(model, preprocessor, label_encoder, model_path)
    
    assert model_path.exists()
    
    # Load model
    loaded_model, loaded_preprocessor, loaded_label_encoder = load_model(model_path)
    
    assert loaded_model is not None
    assert loaded_preprocessor is not None
    assert loaded_label_encoder is not None


def test_evaluate_model(sample_data: pd.DataFrame):
    """Test model evaluation."""
    X, y = prepare_features(sample_data, target_col='income')
    
    # Train model
    model, preprocessor, label_encoder = train_model(X, y)
    
    # Evaluate
    metrics = evaluate_model(model, preprocessor, label_encoder, X, y)
    
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 0 <= metrics['accuracy'] <= 1


def test_compute_baseline_metrics(sample_data: pd.DataFrame):
    """Test baseline metrics computation."""
    X, y = prepare_features(sample_data, target_col='income')
    
    # Train model
    model, preprocessor, label_encoder = train_model(X, y)
    
    # Compute baseline
    baseline_metrics = compute_baseline_metrics(model, preprocessor, label_encoder, X, y)
    
    assert 'accuracy' in baseline_metrics
    assert len(baseline_metrics) > 0
