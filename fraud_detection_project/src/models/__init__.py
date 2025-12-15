"""
Models Module

This module provides machine learning models for fraud detection.
"""

from src.models.base import BaseFraudModel
from src.models.supervised import (
    SupervisedFraudModel,
    RandomForestModel,
    XGBoostModel
)
from src.models.unsupervised import (
    UnsupervisedFraudModel,
    IsolationForestModel,
    KMeansAnomalyModel
)
from src.models.ensemble import (
    EnsembleFraudDetector,
    HybridFraudDetector
)

__all__ = [
    "BaseFraudModel",
    "SupervisedFraudModel",
    "RandomForestModel",
    "XGBoostModel",
    "UnsupervisedFraudModel",
    "IsolationForestModel",
    "KMeansAnomalyModel",
    "EnsembleFraudDetector",
    "HybridFraudDetector"
]
