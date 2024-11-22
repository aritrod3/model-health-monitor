"""
API request/response schemas.

Defines Pydantic models for service endpoints.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint."""
    
    features: Dict[str, Any] = Field(..., description="Feature dictionary")
    include_explanation: bool = Field(default=False, description="Generate explanation")
    explanation_method: Optional[str] = Field(default="shap", description="Explanation method")
    label: Optional[str] = Field(None, description="Optional ground truth label for computing live metrics")


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    
    prediction: Any = Field(..., description="Predicted class")
    probability: Optional[Dict[str, float]] = Field(None, description="Class probabilities")
    explanation: Optional[Dict[str, float]] = Field(None, description="Feature importance")
    model_status: str = Field(default="healthy", description="Model trust status: healthy, degraded, or blocked")


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction endpoint."""
    
    instances: List[Dict[str, Any]] = Field(..., description="List of feature dictionaries")
    include_explanations: bool = Field(default=False, description="Generate explanations")


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction endpoint."""
    
    predictions: List[Any] = Field(..., description="List of predicted classes")
    probabilities: Optional[List[Dict[str, float]]] = Field(None, description="List of class probabilities")
    explanations: Optional[List[Dict[str, float]]] = Field(None, description="List of feature importance dicts")


class MetricsResponse(BaseModel):
    """Response schema for metrics endpoint."""
    
    metrics: Dict[str, Optional[float]] = Field(..., description="Performance metrics (None if N/A)")
    timestamp: str = Field(..., description="Metrics computation timestamp")


class DriftResponse(BaseModel):
    """Response schema for drift detection endpoint."""
    
    drifted_features: List[str] = Field(..., description="Features with detected drift")
    drift_scores: Dict[str, float] = Field(..., description="Drift scores per feature")
    timestamp: str = Field(..., description="Drift detection timestamp")
