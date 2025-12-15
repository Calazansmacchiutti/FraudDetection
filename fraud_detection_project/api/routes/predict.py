"""
Prediction Routes

API endpoints for fraud prediction.
"""

from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
import pandas as pd
import numpy as np
import uuid
import time
import logging

from api.schemas.transaction import (
    TransactionInput,
    PredictionResponse,
    BatchPredictionInput,
    BatchPredictionResponse,
    RiskLevel
)

logger = logging.getLogger("fraud_detection.api.predict")

router = APIRouter()


def get_models():
    """Get loaded models from main app."""
    from api.main import models
    return models


def get_risk_level(probability: float) -> RiskLevel:
    """Determine risk level from probability."""
    if probability >= 0.7:
        return RiskLevel.HIGH
    elif probability >= 0.3:
        return RiskLevel.MEDIUM
    else:
        return RiskLevel.LOW


def prepare_features(transaction: TransactionInput, scaler=None) -> pd.DataFrame:
    """Prepare transaction features for prediction."""
    # Convert to dict and create DataFrame
    data = transaction.model_dump()
    
    # Remove Time if present (we'll create derived features)
    time_val = data.pop("Time", None)
    
    # Create DataFrame
    df = pd.DataFrame([data])
    
    # Add engineered features
    if time_val is not None:
        hour = (time_val / 3600) % 24
        df["Hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["Hour_cos"] = np.cos(2 * np.pi * hour / 24)
    else:
        df["Hour_sin"] = 0.0
        df["Hour_cos"] = 0.0
    
    df["Amount_log"] = np.log1p(df["Amount"])
    
    # PCA magnitude
    top_pca = ["V14", "V12", "V10", "V17", "V16", "V3", "V7", "V11"]
    df["PCA_magnitude"] = np.sqrt((df[top_pca] ** 2).sum(axis=1))
    
    # Interaction
    df["V14_V12"] = df["V14"] * df["V12"]
    
    # Scale if scaler available
    if scaler is not None:
        feature_cols = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else df.columns
        df_scaled = pd.DataFrame(
            scaler.transform(df[feature_cols]),
            columns=feature_cols
        )
        return df_scaled
    
    return df


@router.post("/predict", response_model=PredictionResponse)
async def predict_single(
    transaction: TransactionInput,
    model_name: str = "xgboost",
    threshold: float = 0.5
):
    """
    Predict fraud for a single transaction.
    
    Parameters
    ----------
    transaction : TransactionInput
        Transaction data
    model_name : str
        Model to use for prediction
    threshold : float
        Classification threshold
        
    Returns
    -------
    PredictionResponse
        Prediction result
    """
    models = get_models()
    
    if not models:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please check server logs."
        )
    
    if model_name not in models:
        available = [k for k in models.keys() if k != "scaler"]
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not found. Available: {available}"
        )
    
    try:
        # Prepare features
        scaler = models.get("scaler")
        X = prepare_features(transaction, scaler)
        
        # Get model and predict
        model = models[model_name]
        probability = float(model.predict_proba(X)[0])
        is_fraud = probability >= threshold
        
        # Create response
        response = PredictionResponse(
            transaction_id=f"txn_{uuid.uuid4().hex[:12]}",
            is_fraud=is_fraud,
            probability=round(probability, 6),
            risk_level=get_risk_level(probability),
            model_used=model_name,
            timestamp=datetime.utcnow()
        )
        
        logger.info(
            f"Prediction: fraud={is_fraud}, prob={probability:.4f}, "
            f"risk={response.risk_level}"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    batch: BatchPredictionInput,
    model_name: str = "xgboost",
    threshold: float = 0.5
):
    """
    Predict fraud for multiple transactions.
    
    Parameters
    ----------
    batch : BatchPredictionInput
        Batch of transactions
    model_name : str
        Model to use
    threshold : float
        Classification threshold
        
    Returns
    -------
    BatchPredictionResponse
        Batch prediction results
    """
    models = get_models()
    
    if not models or model_name not in models:
        raise HTTPException(
            status_code=503,
            detail="Model not available"
        )
    
    start_time = time.time()
    predictions = []
    fraud_count = 0
    
    try:
        model = models[model_name]
        scaler = models.get("scaler")
        
        for transaction in batch.transactions:
            X = prepare_features(transaction, scaler)
            probability = float(model.predict_proba(X)[0])
            is_fraud = probability >= threshold
            
            if is_fraud:
                fraud_count += 1
            
            predictions.append(PredictionResponse(
                transaction_id=f"txn_{uuid.uuid4().hex[:12]}",
                is_fraud=is_fraud,
                probability=round(probability, 6),
                risk_level=get_risk_level(probability),
                model_used=model_name,
                timestamp=datetime.utcnow()
            ))
        
        processing_time = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions),
            fraud_count=fraud_count,
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
