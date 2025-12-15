"""
Health Check Routes

API endpoints for health monitoring.
"""

from fastapi import APIRouter
from datetime import datetime

from api.schemas.transaction import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check API health status.
    
    Returns
    -------
    HealthResponse
        Health status information
    """
    from api.main import models
    
    models_count = len([k for k in models.keys() if k != "scaler"])
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=models_count,
        timestamp=datetime.utcnow()
    )


@router.get("/ready")
async def readiness_check():
    """
    Check if API is ready to serve requests.
    
    Returns
    -------
    dict
        Readiness status
    """
    from api.main import models
    
    is_ready = len(models) > 0
    
    return {
        "ready": is_ready,
        "message": "API ready" if is_ready else "Models not loaded"
    }


@router.get("/live")
async def liveness_check():
    """
    Check if API is alive.
    
    Returns
    -------
    dict
        Liveness status
    """
    return {"alive": True}
