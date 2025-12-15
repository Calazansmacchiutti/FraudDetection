"""
Unsupervised Models Module

This module provides unsupervised anomaly detection models for fraud detection.
"""

import time
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import logging

from src.models.base import BaseFraudModel

logger = logging.getLogger("fraud_detection.models.unsupervised")


class UnsupervisedFraudModel(BaseFraudModel):
    """
    Base class for unsupervised fraud detection models.
    
    Unsupervised models detect anomalies without using labels.
    """
    
    def __init__(self, model_name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the unsupervised model.
        
        Parameters
        ----------
        model_name : str
            Name of the model
        params : dict, optional
            Model hyperparameters
        """
        super().__init__(model_name=model_name, model_type="unsupervised")
        self.params = params or {}


class IsolationForestModel(UnsupervisedFraudModel):
    """
    Isolation Forest model for anomaly detection.
    
    Isolation Forest isolates anomalies through random partitioning.
    Points that require fewer partitions to isolate are considered anomalies.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Isolation Forest model.
        
        Parameters
        ----------
        params : dict, optional
            Model hyperparameters
        """
        default_params = {
            "n_estimators": 100,
            "contamination": 0.002,
            "random_state": 42,
            "n_jobs": -1
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(model_name="isolation_forest", params=default_params)
        self.model = IsolationForest(**self.params)
    
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        train_on_normal_only: bool = True
    ) -> "IsolationForestModel":
        """
        Fit the model on training data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y : pd.Series, optional
            Training labels (used to filter normal samples if train_on_normal_only=True)
        train_on_normal_only : bool
            Whether to train only on normal (non-fraud) samples
            
        Returns
        -------
        IsolationForestModel
            Fitted model (self)
        """
        logger.info("Training Isolation Forest model...")
        
        self._feature_names = list(X.columns)
        
        # Filter to normal samples if labels provided
        if train_on_normal_only and y is not None:
            X_train = X[y == 0]
            logger.info(f"Training on {len(X_train):,} normal samples only")
        else:
            X_train = X
        
        start_time = time.time()
        self.model.fit(X_train)
        self.training_time = time.time() - start_time
        self.is_fitted = True
        
        logger.info(
            f"Isolation Forest trained in {self.training_time:.2f} seconds"
        )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features for prediction
            
        Returns
        -------
        np.ndarray
            Predicted labels (0=normal, 1=fraud)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Isolation Forest returns 1 for inliers, -1 for outliers
        predictions = self.model.predict(X)
        # Convert: -1 (outlier/anomaly) -> 1 (fraud), 1 (inlier) -> 0 (normal)
        return (predictions == -1).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomaly probability.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features for prediction
            
        Returns
        -------
        np.ndarray
            Anomaly probabilities (higher = more likely fraud)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get anomaly scores (more negative = more anomalous)
        scores = self.model.score_samples(X)
        
        # Normalize to [0, 1] where higher = more anomalous
        proba = 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        return proba


class KMeansAnomalyModel(UnsupervisedFraudModel):
    """
    K-Means based anomaly detection model.
    
    Points far from cluster centroids are considered anomalies.
    """
    
    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        anomaly_percentile: float = 99.0
    ):
        """
        Initialize K-Means anomaly model.
        
        Parameters
        ----------
        params : dict, optional
            Model hyperparameters
        anomaly_percentile : float
            Percentile threshold for anomaly detection
        """
        default_params = {
            "n_clusters": 8,
            "random_state": 42,
            "n_init": 10
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(model_name="kmeans", params=default_params)
        self.model = KMeans(**self.params)
        self.anomaly_percentile = anomaly_percentile
        self.distance_threshold: float = 0.0
    
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> "KMeansAnomalyModel":
        """
        Fit the model on training data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y : pd.Series, optional
            Training labels (not used, included for API consistency)
            
        Returns
        -------
        KMeansAnomalyModel
            Fitted model (self)
        """
        logger.info("Training K-Means anomaly model...")
        
        self._feature_names = list(X.columns)
        
        start_time = time.time()
        self.model.fit(X)
        
        # Calculate distance threshold
        distances = self._calculate_distances(X)
        self.distance_threshold = np.percentile(distances, self.anomaly_percentile)
        
        self.training_time = time.time() - start_time
        self.is_fitted = True
        
        logger.info(
            f"K-Means trained in {self.training_time:.2f} seconds. "
            f"Distance threshold: {self.distance_threshold:.4f}"
        )
        
        return self
    
    def _calculate_distances(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate distances to nearest centroid."""
        distances = self.model.transform(X)
        return distances.min(axis=1)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features for prediction
            
        Returns
        -------
        np.ndarray
            Predicted labels (0=normal, 1=fraud)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        distances = self._calculate_distances(X)
        return (distances > self.distance_threshold).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomaly probability based on distance.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features for prediction
            
        Returns
        -------
        np.ndarray
            Anomaly probabilities (higher = more likely fraud)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        distances = self._calculate_distances(X)
        
        # Normalize distances to [0, 1]
        proba = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)
        
        return proba
