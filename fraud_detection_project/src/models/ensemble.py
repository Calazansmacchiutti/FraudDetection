"""
Ensemble Models Module

This module provides ensemble methods combining multiple models.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging

from src.models.base import BaseFraudModel

logger = logging.getLogger("fraud_detection.models.ensemble")


class EnsembleFraudDetector:
    """
    Ensemble fraud detector combining multiple models.
    
    This class combines predictions from multiple models using
    weighted averaging or voting.
    
    Attributes
    ----------
    models : list
        List of (name, model, weight) tuples
    combination_method : str
        Method for combining predictions ('weighted_average' or 'voting')
    """
    
    def __init__(
        self,
        combination_method: str = "weighted_average"
    ):
        """
        Initialize the ensemble detector.
        
        Parameters
        ----------
        combination_method : str
            Method for combining predictions
        """
        self.models: List[tuple] = []
        self.combination_method = combination_method
    
    def add_model(
        self,
        name: str,
        model: BaseFraudModel,
        weight: float = 1.0
    ) -> "EnsembleFraudDetector":
        """
        Add a model to the ensemble.
        
        Parameters
        ----------
        name : str
            Name for the model
        model : BaseFraudModel
            Model instance
        weight : float
            Weight for this model's predictions
            
        Returns
        -------
        EnsembleFraudDetector
            Self for chaining
        """
        if not model.is_fitted:
            raise ValueError(f"Model '{name}' must be fitted before adding to ensemble")
        
        self.models.append((name, model, weight))
        logger.info(f"Added model '{name}' with weight {weight}")
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud probability using ensemble.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features for prediction
            
        Returns
        -------
        np.ndarray
            Ensemble fraud probabilities
        """
        if not self.models:
            raise ValueError("No models in ensemble. Add models first.")
        
        if self.combination_method == "weighted_average":
            return self._weighted_average_proba(X)
        elif self.combination_method == "voting":
            return self._voting_proba(X)
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
    
    def _weighted_average_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate weighted average of probabilities."""
        total_weight = sum(w for _, _, w in self.models)
        
        ensemble_proba = np.zeros(len(X))
        
        for name, model, weight in self.models:
            proba = model.predict_proba(X)
            ensemble_proba += (weight / total_weight) * proba
        
        return ensemble_proba
    
    def _voting_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate voting-based probability."""
        predictions = []
        
        for name, model, weight in self.models:
            pred = model.predict(X)
            predictions.append(pred * weight)
        
        # Average votes
        total_weight = sum(w for _, _, w in self.models)
        return np.sum(predictions, axis=0) / total_weight
    
    def predict(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Make predictions using ensemble.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features for prediction
        threshold : float
            Classification threshold
            
        Returns
        -------
        np.ndarray
            Predicted labels
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def get_model_contributions(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get individual model contributions for predictions.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features for prediction
            
        Returns
        -------
        pd.DataFrame
            DataFrame with each model's probability
        """
        contributions = {}
        
        for name, model, weight in self.models:
            contributions[f"{name}_proba"] = model.predict_proba(X)
            contributions[f"{name}_weight"] = weight
        
        contributions["ensemble_proba"] = self.predict_proba(X)
        
        return pd.DataFrame(contributions)


class HybridFraudDetector:
    """
    Hybrid detector combining supervised and unsupervised models.
    
    This detector uses a supervised model for known patterns and
    an unsupervised model for detecting novel anomalies.
    """
    
    def __init__(
        self,
        supervised_model: BaseFraudModel,
        unsupervised_model: BaseFraudModel,
        supervised_weight: float = 0.7,
        unsupervised_weight: float = 0.3
    ):
        """
        Initialize the hybrid detector.
        
        Parameters
        ----------
        supervised_model : BaseFraudModel
            Trained supervised model
        unsupervised_model : BaseFraudModel
            Trained unsupervised model
        supervised_weight : float
            Weight for supervised model
        unsupervised_weight : float
            Weight for unsupervised model
        """
        self.supervised_model = supervised_model
        self.unsupervised_model = unsupervised_model
        self.supervised_weight = supervised_weight
        self.unsupervised_weight = unsupervised_weight
        
        # Normalize weights
        total = supervised_weight + unsupervised_weight
        self.supervised_weight /= total
        self.unsupervised_weight /= total
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud probability.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features for prediction
            
        Returns
        -------
        np.ndarray
            Combined fraud probabilities
        """
        supervised_proba = self.supervised_model.predict_proba(X)
        unsupervised_proba = self.unsupervised_model.predict_proba(X)
        
        combined_proba = (
            self.supervised_weight * supervised_proba +
            self.unsupervised_weight * unsupervised_proba
        )
        
        return combined_proba
    
    def predict(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Make predictions.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features for prediction
        threshold : float
            Classification threshold
            
        Returns
        -------
        np.ndarray
            Predicted labels
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def get_breakdown(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get detailed breakdown of predictions.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features for prediction
            
        Returns
        -------
        pd.DataFrame
            Breakdown of predictions by model
        """
        return pd.DataFrame({
            "supervised_proba": self.supervised_model.predict_proba(X),
            "unsupervised_proba": self.unsupervised_model.predict_proba(X),
            "combined_proba": self.predict_proba(X),
            "prediction": self.predict(X)
        })
