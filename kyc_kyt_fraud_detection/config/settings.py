"""
Application Settings

Centralized configuration for KYC/KYT Loan Default Prediction system.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any


# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"


@dataclass
class DataConfig:
    """Data-related configuration."""

    raw_data_dir: Path = DATA_DIR / "raw"
    processed_data_dir: Path = DATA_DIR / "processed"

    # Data files
    loan_file: str = "loan.csv"
    trans_file: str = "trans.csv"

    # Split configuration
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True

    # Target configuration
    target_column: str = "is_default"
    status_column: str = "status"
    default_status: str = "D"  # Status value indicating default


@dataclass
class ModelConfig:
    """Model-related configuration."""

    # Model paths
    models_dir: Path = MODELS_DIR
    default_model: str = "random_forest"  # Best performer from tuning

    # Scaling
    scaler_type: str = "robust"  # 'robust' or 'standard'

    # Random Forest (Optuna-tuned params)
    random_forest_tuned: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 173,
        "max_depth": 5,
        "min_samples_split": 9,
        "min_samples_leaf": 5,
        "max_features": None,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1
    })

    # XGBoost (Optuna-tuned params)
    xgboost_tuned: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 358,
        "max_depth": 6,
        "learning_rate": 0.038805,
        "min_child_weight": 3,
        "subsample": 0.946990,
        "colsample_bytree": 0.611372,
        "gamma": 2.083614,
        "reg_alpha": 0.001823,
        "reg_lambda": 0.041021,
        "scale_pos_weight": 25.217325,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "aucpr",
        "use_label_encoder": False
    })

    # Autoencoder (Optuna-tuned params)
    autoencoder_tuned: Dict[str, Any] = field(default_factory=lambda: {
        "encoding_dim": 8,
        "n_layers": 1,
        "dropout": 0.247,
        "l2_reg": 1.37e-06,
        "learning_rate": 0.00659,
        "batch_size": 32,
        "epochs": 100,
        "patience": 10
    })


@dataclass
class OptunaConfig:
    """Optuna hyperparameter optimization configuration."""

    n_trials: int = 50
    cv_folds: int = 5
    metric: str = "average_precision"
    direction: str = "maximize"
    random_state: int = 42

    # Parallel execution
    n_jobs: int = 1  # Set to -1 for parallel optimization (may be unstable)


@dataclass
class PredictionConfig:
    """Prediction-related configuration."""

    default_threshold: float = 0.5
    high_risk_threshold: float = 0.8
    critical_risk_threshold: float = 0.6
    medium_risk_threshold: float = 0.3

    # Risk levels
    risk_levels: Dict[str, str] = field(default_factory=lambda: {
        "critical": "CRITICAL",
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
    log_file: str = "kyc_kyt_fraud_detection.log"
    max_bytes: int = 10_000_000  # 10 MB
    backup_count: int = 5


class Settings:
    """Main settings class combining all configurations."""

    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.optuna = OptunaConfig()
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
            MODELS_DIR,
            LOGS_DIR,
            REPORTS_DIR / "figures",
            REPORTS_DIR / "optuna_studies",
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

    def get_loan_file_path(self) -> Path:
        """Get the full path to loan data file."""
        return self.data.raw_data_dir / self.data.loan_file

    def get_trans_file_path(self) -> Path:
        """Get the full path to transaction data file."""
        return self.data.raw_data_dir / self.data.trans_file


# Global settings instance
settings = Settings()
