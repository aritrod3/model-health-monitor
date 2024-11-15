"""
FastAPI application for model inference and monitoring.

Provides REST endpoints for predictions, explanations, metrics, and drift detection.
"""

from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException

from service.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    DriftResponse,
    MetricsResponse,
    PredictionRequest,
    PredictionResponse,
)

app = FastAPI(
    title="Model Health Monitor API",
    description="API for tabular classifier inference and monitoring",
    version="0.1.0"
)


@app.get("/health")
def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    
    Returns:
        Status dictionary
    """
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Generate prediction for single instance.
    
    Args:
        request: Prediction request with features
        
    Returns:
        Prediction response with class, probabilities, and optional explanation
        
    Raises:
        HTTPException: If prediction fails
    """
    pass


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Generate predictions for batch of instances.
    
    Args:
        request: Batch prediction request
        
    Returns:
        Batch prediction response
        
    Raises:
        HTTPException: If batch prediction fails
    """
    pass


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics() -> MetricsResponse:
    """
    Retrieve current model performance metrics.
    
    Returns:
        Metrics response with performance indicators
        
    Raises:
        HTTPException: If metrics retrieval fails
    """
    pass


@app.get("/drift", response_model=DriftResponse)
def check_drift() -> DriftResponse:
    """
    Check for data drift in production data.
    
    Returns:
        Drift response with drifted features and scores
        
    Raises:
        HTTPException: If drift detection fails
    """
    pass
