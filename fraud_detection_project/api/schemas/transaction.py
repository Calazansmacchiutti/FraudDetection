"""
API Schemas

Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class RiskLevel(str, Enum):
    """Risk level enumeration."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class TransactionInput(BaseModel):
    """Input schema for a single transaction."""
    
    V1: float = Field(..., description="PCA component V1")
    V2: float = Field(..., description="PCA component V2")
    V3: float = Field(..., description="PCA component V3")
    V4: float = Field(..., description="PCA component V4")
    V5: float = Field(..., description="PCA component V5")
    V6: float = Field(..., description="PCA component V6")
    V7: float = Field(..., description="PCA component V7")
    V8: float = Field(..., description="PCA component V8")
    V9: float = Field(..., description="PCA component V9")
    V10: float = Field(..., description="PCA component V10")
    V11: float = Field(..., description="PCA component V11")
    V12: float = Field(..., description="PCA component V12")
    V13: float = Field(..., description="PCA component V13")
    V14: float = Field(..., description="PCA component V14")
    V15: float = Field(..., description="PCA component V15")
    V16: float = Field(..., description="PCA component V16")
    V17: float = Field(..., description="PCA component V17")
    V18: float = Field(..., description="PCA component V18")
    V19: float = Field(..., description="PCA component V19")
    V20: float = Field(..., description="PCA component V20")
    V21: float = Field(..., description="PCA component V21")
    V22: float = Field(..., description="PCA component V22")
    V23: float = Field(..., description="PCA component V23")
    V24: float = Field(..., description="PCA component V24")
    V25: float = Field(..., description="PCA component V25")
    V26: float = Field(..., description="PCA component V26")
    V27: float = Field(..., description="PCA component V27")
    V28: float = Field(..., description="PCA component V28")
    Amount: float = Field(..., ge=0, description="Transaction amount")
    Time: Optional[float] = Field(None, description="Time in seconds (optional)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "V1": -1.359807,
                "V2": -0.072781,
                "V3": 2.536347,
                "V4": 1.378155,
                "V5": -0.338321,
                "V6": 0.462388,
                "V7": 0.239599,
                "V8": 0.098698,
                "V9": 0.363787,
                "V10": 0.090794,
                "V11": -0.551600,
                "V12": -0.617801,
                "V13": -0.991390,
                "V14": -0.311169,
                "V15": 1.468177,
                "V16": -0.470401,
                "V17": 0.207971,
                "V18": 0.025791,
                "V19": 0.403993,
                "V20": 0.251412,
                "V21": -0.018307,
                "V22": 0.277838,
                "V23": -0.110474,
                "V24": 0.066928,
                "V25": 0.128539,
                "V26": -0.189115,
                "V27": 0.133558,
                "V28": -0.021053,
                "Amount": 149.62
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for prediction."""
    
    transaction_id: str = Field(..., description="Unique transaction identifier")
    is_fraud: bool = Field(..., description="Whether transaction is predicted as fraud")
    probability: float = Field(..., ge=0, le=1, description="Fraud probability")
    risk_level: RiskLevel = Field(..., description="Risk level classification")
    model_used: str = Field(..., description="Model used for prediction")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "txn_abc123",
                "is_fraud": False,
                "probability": 0.0234,
                "risk_level": "LOW",
                "model_used": "xgboost",
                "timestamp": "2025-01-15T10:30:00Z"
            }
        }


class BatchPredictionInput(BaseModel):
    """Input schema for batch predictions."""
    
    transactions: List[TransactionInput] = Field(
        ..., 
        min_length=1,
        max_length=1000,
        description="List of transactions to predict"
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    
    predictions: List[PredictionResponse]
    total_count: int
    fraud_count: int
    processing_time_ms: float


class ModelInfo(BaseModel):
    """Model information schema."""
    
    name: str
    type: str
    is_loaded: bool


class HealthResponse(BaseModel):
    """Health check response schema."""
    
    status: str
    version: str
    models_loaded: int
    timestamp: datetime
