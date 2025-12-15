"""
Fraud Detection API

FastAPI application for real-time fraud detection.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Optional

from api.routes import predict, health
from api.schemas.transaction import ModelInfo

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud_detection.api")

# Global model storage
models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    # Startup
    logger.info("Starting Fraud Detection API...")
    
    # Load models
    try:
        from src.models import XGBoostModel, IsolationForestModel
        
        models_dir = Path("models")
        
        if (models_dir / "xgboost_model.pkl").exists():
            models["xgboost"] = XGBoostModel.load(models_dir / "xgboost_model.pkl")
            logger.info("Loaded XGBoost model")
        
        if (models_dir / "isolation_forest_model.pkl").exists():
            models["isolation_forest"] = IsolationForestModel.load(
                models_dir / "isolation_forest_model.pkl"
            )
            logger.info("Loaded Isolation Forest model")
        
        # Load scaler
        import joblib
        if (models_dir / "scaler.pkl").exists():
            models["scaler"] = joblib.load(models_dir / "scaler.pkl")
            logger.info("Loaded scaler")
        
        logger.info(f"Loaded {len(models)} model components")
        
    except Exception as e:
        logger.warning(f"Could not load models: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Fraud Detection API...")
    models.clear()


# Create FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection using machine learning",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, prefix="/api/v1", tags=["Predictions"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/api/v1/models", response_model=list[ModelInfo])
async def list_models():
    """List available models."""
    available_models = []
    
    for name, model in models.items():
        if name != "scaler":
            available_models.append(ModelInfo(
                name=name,
                type=getattr(model, "model_type", "unknown"),
                is_loaded=True
            ))
    
    return available_models


def get_models():
    """Get loaded models (for dependency injection)."""
    return models
