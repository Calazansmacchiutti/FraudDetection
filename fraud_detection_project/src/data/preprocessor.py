"""
Data Preprocessor Module

This module provides data preprocessing and transformation utilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
import joblib
import logging

logger = logging.getLogger("fraud_detection.data.preprocessor")


class DataPreprocessor:
    """
    Data preprocessor for credit card transaction data.
    
    This class handles data cleaning, transformation, and splitting
    for model training and inference.
    
    Attributes
    ----------
    scaler : sklearn scaler
        Fitted scaler for feature normalization
    feature_columns : list
        List of feature column names
    is_fitted : bool
        Whether the preprocessor has been fitted
    """
    
    def __init__(self, scaler_type: str = "robust"):
        """
        Initialize the DataPreprocessor.
        
        Parameters
        ----------
        scaler_type : str
            Type of scaler to use ('robust' or 'standard')
        """
        self.scaler_type = scaler_type
        self.scaler = self._create_scaler(scaler_type)
        self.feature_columns: List[str] = []
        self.is_fitted: bool = False
    
    def _create_scaler(self, scaler_type: str):
        """Create scaler based on type."""
        if scaler_type == "robust":
            return RobustScaler()
        elif scaler_type == "standard":
            return StandardScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    def fit(self, X: pd.DataFrame) -> "DataPreprocessor":
        """
        Fit the preprocessor on training data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training features
            
        Returns
        -------
        DataPreprocessor
            Fitted preprocessor (self)
        """
        logger.info("Fitting preprocessor...")
        
        self.feature_columns = list(X.columns)
        self.scaler.fit(X)
        self.is_fitted = True
        
        logger.info(f"Preprocessor fitted on {len(self.feature_columns)} features")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features to transform
            
        Returns
        -------
        pd.DataFrame
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        # Ensure columns match
        missing_cols = set(self.feature_columns) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Transform and return as DataFrame
        X_scaled = self.scaler.transform(X[self.feature_columns])
        
        return pd.DataFrame(
            X_scaled,
            columns=self.feature_columns,
            index=X.index
        )
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features to fit and transform
            
        Returns
        -------
        pd.DataFrame
            Transformed features
        """
        return self.fit(X).transform(X)
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save preprocessor to file.
        
        Parameters
        ----------
        path : str or Path
            Path to save the preprocessor
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "scaler": self.scaler,
            "scaler_type": self.scaler_type,
            "feature_columns": self.feature_columns,
            "is_fitted": self.is_fitted
        }
        
        joblib.dump(state, path)
        logger.info(f"Preprocessor saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "DataPreprocessor":
        """
        Load preprocessor from file.
        
        Parameters
        ----------
        path : str or Path
            Path to the saved preprocessor
            
        Returns
        -------
        DataPreprocessor
            Loaded preprocessor
        """
        state = joblib.load(path)
        
        preprocessor = cls(scaler_type=state["scaler_type"])
        preprocessor.scaler = state["scaler"]
        preprocessor.feature_columns = state["feature_columns"]
        preprocessor.is_fitted = state["is_fitted"]
        
        logger.info(f"Preprocessor loaded from {path}")
        
        return preprocessor


def prepare_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and test sets.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    test_size : float
        Proportion of data for test set
    random_state : int
        Random seed for reproducibility
    stratify : bool
        Whether to stratify by target
        
    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    stratify_col = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col
    )
    
    logger.info(
        f"Data split: {len(X_train):,} train, {len(X_test):,} test "
        f"(stratified={stratify})"
    )
    
    return X_train, X_test, y_train, y_test
