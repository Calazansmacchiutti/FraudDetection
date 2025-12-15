"""
Supervised Models Module

This module provides supervised learning models for fraud detection.
"""

import time
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import logging

from src.models.base import BaseFraudModel

logger = logging.getLogger("fraud_detection.models.supervised")


class SupervisedFraudModel(BaseFraudModel):
    """
    Supervised fraud detection model.
    
    This class wraps supervised classifiers (Random Forest, XGBoost)
    for fraud detection.
    """
    
    SUPPORTED_MODELS = {
        "random_forest": RandomForestClassifier,
        "xgboost": XGBClassifier
    }
    
    def __init__(
        self,
        model_name: str = "xgboost",
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the supervised model.
        
        Parameters
        ----------
        model_name : str
            Name of the model ('random_forest' or 'xgboost')
        params : dict, optional
            Model hyperparameters
        """
        super().__init__(model_name=model_name, model_type="supervised")
        
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Supported models: {list(self.SUPPORTED_MODELS.keys())}"
            )
        
        self.params = params or {}
        self._init_model()
    
    def _init_model(self) -> None:
        """Initialize the underlying model."""
        model_class = self.SUPPORTED_MODELS[self.model_name]
        self.model = model_class(**self.params)
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None
    ) -> "SupervisedFraudModel":
        """
        Fit the model on training data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y : pd.Series
            Training labels
        sample_weight : np.ndarray, optional
            Sample weights
            
        Returns
        -------
        SupervisedFraudModel
            Fitted model (self)
        """
        logger.info(f"Training {self.model_name} model...")
        
        self._feature_names = list(X.columns)
        
        start_time = time.time()
        
        if sample_weight is not None:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)
        
        self.training_time = time.time() - start_time
        self.is_fitted = True
        
        logger.info(
            f"{self.model_name} trained in {self.training_time:.2f} seconds"
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
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of fraud.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features for prediction
            
        Returns
        -------
        np.ndarray
            Fraud probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance.
        
        Returns
        -------
        pd.DataFrame
            Feature importance dataframe
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        importance = pd.DataFrame({
            "feature": self._feature_names,
            "importance": self.model.feature_importances_
        })
        
        return importance.sort_values("importance", ascending=False)


class RandomForestModel(SupervisedFraudModel):
    """Random Forest model for fraud detection."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Random Forest model.
        
        Parameters
        ----------
        params : dict, optional
            Model hyperparameters
        """
        default_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(model_name="random_forest", params=default_params)


class XGBoostModel(SupervisedFraudModel):
    """XGBoost model for fraud detection."""
    
    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        scale_pos_weight: Optional[float] = None
    ):
        """
        Initialize XGBoost model.
        
        Parameters
        ----------
        params : dict, optional
            Model hyperparameters
        scale_pos_weight : float, optional
            Weight for positive class (for imbalanced data)
        """
        default_params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "aucpr",
            "use_label_encoder": False
        }
        
        if scale_pos_weight is not None:
            default_params["scale_pos_weight"] = scale_pos_weight
        
        if params:
            default_params.update(params)
        
        super().__init__(model_name="xgboost", params=default_params)
    
    @classmethod
    def with_auto_weight(
        cls,
        y_train: pd.Series,
        params: Optional[Dict[str, Any]] = None
    ) -> "XGBoostModel":
        """
        Create XGBoost model with automatic class weight calculation.
        
        Parameters
        ----------
        y_train : pd.Series
            Training labels
        params : dict, optional
            Additional parameters
            
        Returns
        -------
        XGBoostModel
            Model instance with calculated scale_pos_weight
        """
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")
        
        return cls(params=params, scale_pos_weight=scale_pos_weight)
