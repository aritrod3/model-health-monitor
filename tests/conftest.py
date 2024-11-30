"""Pytest fixtures for testing."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest
from sqlalchemy.orm import Session, sessionmaker

from storage.db import Base, initialize_database


@pytest.fixture
def temp_db() -> Generator[sessionmaker, None, None]:
    """Create temporary test database."""
    # Use in-memory SQLite for tests
    db_path = "sqlite:///:memory:"
    SessionLocal = initialize_database(db_path)
    
    # Create tables
    engine = SessionLocal().bind
    Base.metadata.create_all(bind=engine)
    
    yield SessionLocal
    
    # Cleanup
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(temp_db: sessionmaker) -> Generator[Session, None, None]:
    """Create database session for testing."""
    session = temp_db()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample data for testing."""
    data = {
        'age': [39, 50, 38, 53, 28],
        'workclass': ['State-gov', 'Self-emp-not-inc', 'Private', 'Private', 'Private'],
        'fnlwgt': [77516, 83311, 215646, 234721, 338409],
        'education': ['Bachelors', 'Bachelors', 'HS-grad', '11th', 'Bachelors'],
        'education-num': [13, 13, 9, 7, 13],
        'marital-status': ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-civ-spouse', 'Married-civ-spouse'],
        'occupation': ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Handlers-cleaners', 'Prof-specialty'],
        'relationship': ['Not-in-family', 'Husband', 'Not-in-family', 'Husband', 'Wife'],
        'race': ['White', 'White', 'White', 'Black', 'Black'],
        'sex': ['Male', 'Male', 'Male', 'Male', 'Female'],
        'capital-gain': [2174, 0, 0, 0, 0],
        'capital-loss': [0, 0, 0, 0, 0],
        'hours-per-week': [40, 13, 40, 40, 40],
        'native-country': ['United-States', 'United-States', 'United-States', 'United-States', 'Cuba'],
        'income': ['<=50K', '<=50K', '<=50K', '<=50K', '<=50K']
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_features() -> dict:
    """Sample feature dictionary for prediction."""
    return {
        'age': 39,
        'workclass': 'State-gov',
        'fnlwgt': 77516,
        'education': 'Bachelors',
        'education-num': 13,
        'marital-status': 'Never-married',
        'occupation': 'Adm-clerical',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 2174,
        'capital-loss': 0,
        'hours-per-week': 40,
        'native-country': 'United-States'
    }


@pytest.fixture
def temp_model_dir(tmp_path: Path) -> Path:
    """Create temporary directory for test models."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create temporary directory for test data."""
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    return data_dir
