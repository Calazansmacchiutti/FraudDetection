"""
Base Model Module

Abstract base class for all loan default prediction models.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Union
import numpy as np
import pandas as pd
import joblib
import logging

logger = logging.getLogger("kyc_kyt.models.base")


class BaseDefaultModel(ABC):
    """
    Abstract base class for loan default prediction models.

    All models should inherit from this class and implement
    the required methods.

    Attributes
    ----------
    model : object
        The underlying model object
    model_name : str
        Name of the model
    model_type : str
        Type of the model ('supervised' or 'unsupervised')
    is_fitted : bool
        Whether the model has been fitted
    training_time : float
        Time taken to train the model (seconds)
    feature_names : list
        Names of features used for training
    """

    def __init__(self, model_name: str, model_type: str):
        """
        Initialize the base model.

        Parameters
        ----------
        model_name : str
            Name of the model
        model_type : str
            Type of the model ('supervised' or 'unsupervised')
        """
        self.model = None
        self.model_name = model_name
        self.model_type = model_type
        self.is_fitted = False
        self.training_time: float = 0.0
        self.feature_names: list = []
        self.metadata: Dict = {}

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> "BaseDefaultModel":
        """
        Fit the model on training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y : pd.Series, optional
            Training labels (required for supervised models)

        Returns
        -------
        BaseDefaultModel
            Fitted model (self)
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Parameters
        ----------
        X : pd.DataFrame
            Features for prediction

        Returns
        -------
        np.ndarray
            Predicted labels (0 or 1)
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of default.

        Parameters
        ----------
        X : pd.DataFrame
            Features for prediction

        Returns
        -------
        np.ndarray
            Probability of default (0 to 1)
        """
        pass

    def predict_with_threshold(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Make predictions using a custom threshold.

        Parameters
        ----------
        X : pd.DataFrame
            Features for prediction
        threshold : float
            Classification threshold (default 0.5)

        Returns
        -------
        np.ndarray
            Predicted labels based on threshold
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def get_risk_level(self, probability: float) -> str:
        """
        Get risk level based on probability.

        Parameters
        ----------
        probability : float
            Default probability

        Returns
        -------
        str
            Risk level ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
        """
        if probability >= 0.8:
            return "CRITICAL"
        elif probability >= 0.6:
            return "HIGH"
        elif probability >= 0.3:
            return "MEDIUM"
        else:
            return "LOW"

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the model to disk.

        Parameters
        ----------
        path : str or Path
            Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model": self.model,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "is_fitted": self.is_fitted,
            "training_time": self.training_time,
            "feature_names": self.feature_names,
            "metadata": self.metadata
        }

        joblib.dump(state, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseDefaultModel":
        """
        Load a model from disk.

        Parameters
        ----------
        path : str or Path
            Path to the saved model

        Returns
        -------
        BaseDefaultModel
            Loaded model
        """
        state = joblib.load(path)

        # Create instance without calling __init__ of subclass
        instance = object.__new__(cls)
        instance.model = state["model"]
        instance.model_name = state["model_name"]
        instance.model_type = state["model_type"]
        instance.is_fitted = state["is_fitted"]
        instance.training_time = state["training_time"]
        instance.feature_names = state.get("feature_names", [])
        instance.metadata = state.get("metadata", {})

        logger.info(f"Model loaded from {path}")

        return instance

    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.

        Returns
        -------
        dict
            Model parameters
        """
        if hasattr(self.model, "get_params"):
            return self.model.get_params()
        return {}

    def set_metadata(self, **kwargs) -> None:
        """
        Set metadata for the model.

        Parameters
        ----------
        **kwargs
            Metadata key-value pairs
        """
        self.metadata.update(kwargs)

    def get_metadata(self) -> Dict:
        """
        Get model metadata.

        Returns
        -------
        dict
            Model metadata
        """
        return self.metadata

    def __repr__(self) -> str:
        fitted_str = "fitted" if self.is_fitted else "not fitted"
        return (
            f"{self.__class__.__name__}("
            f"name='{self.model_name}', "
            f"type='{self.model_type}', "
            f"{fitted_str})"
        )
