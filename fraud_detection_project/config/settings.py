"""
Application Settings

This module contains all configuration settings for the fraud detection system.
Settings can be overridden via environment variables.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"


@dataclass
class DataConfig:
    """Data-related configuration."""
    
    raw_data_path: Path = DATA_DIR / "raw" / "creditcard.csv"
    processed_data_path: Path = DATA_DIR / "processed"
    test_size: float = 0.2
    random_state: int = 42
    
    # Feature columns
    pca_features: list = field(default_factory=lambda: [f"V{i}" for i in range(1, 29)])
    target_column: str = "Class"
    time_column: str = "Time"
    amount_column: str = "Amount"


@dataclass
class ModelConfig:
    """Model-related configuration."""
    
    # Model paths
    models_dir: Path = MODELS_DIR
    default_model: str = "xgboost"
    
    # XGBoost parameters
    xgboost_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "aucpr",
        "use_label_encoder": False
    })
    
    # Random Forest parameters
    random_forest_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 10,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1
    })
    
    # Isolation Forest parameters
    isolation_forest_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "contamination": 0.002,
        "random_state": 42,
        "n_jobs": -1
    })
    
    # K-Means parameters
    kmeans_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_clusters": 8,
        "random_state": 42,
        "n_init": 10
    })


@dataclass
class PredictionConfig:
    """Prediction-related configuration."""
    
    default_threshold: float = 0.5
    high_risk_threshold: float = 0.7
    medium_risk_threshold: float = 0.3
    
    # Risk levels
    risk_levels: Dict[str, str] = field(default_factory=lambda: {
        "high": "HIGH",
        "medium": "MEDIUM",
        "low": "LOW"
    })


@dataclass
class APIConfig:
    """API-related configuration."""
    
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    api_version: str = "v1"
    api_prefix: str = "/api/v1"
    
    # CORS settings
    allowed_origins: list = field(default_factory=lambda: ["*"])
    allowed_methods: list = field(default_factory=lambda: ["GET", "POST"])
    allowed_headers: list = field(default_factory=lambda: ["*"])


@dataclass
class LogConfig:
    """Logging configuration."""
    
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_dir: Path = LOGS_DIR
    log_file: str = "fraud_detection.log"
    max_bytes: int = 10_000_000  # 10 MB
    backup_count: int = 5


class Settings:
    """Main settings class combining all configurations."""
    
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.prediction = PredictionConfig()
        self.api = APIConfig()
        self.log = LogConfig()
        
        # Ensure directories exist
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            DATA_DIR / "raw",
            DATA_DIR / "processed",
            DATA_DIR / "external",
            MODELS_DIR,
            LOGS_DIR,
            REPORTS_DIR / "figures",
            REPORTS_DIR / "metrics"
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, model_name: str) -> Path:
        """Get the path for a specific model."""
        return self.model.models_dir / f"{model_name}_model.pkl"
    
    def get_scaler_path(self) -> Path:
        """Get the path for the scaler."""
        return self.model.models_dir / "scaler.pkl"


# Global settings instance
settings = Settings()
