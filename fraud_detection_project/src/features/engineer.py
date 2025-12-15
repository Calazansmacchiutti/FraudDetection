"""
Feature Engineering Module

This module provides feature engineering utilities for fraud detection.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger("fraud_detection.features.engineer")


class FeatureEngineer:
    """
    Feature engineer for credit card transaction data.
    
    This class creates derived features to improve model performance.
    
    Attributes
    ----------
    feature_names : list
        List of engineered feature names
    config : dict
        Feature engineering configuration
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the FeatureEngineer.
        
        Parameters
        ----------
        config : dict, optional
            Feature engineering configuration
        """
        self.config = config or {}
        self.feature_names: List[str] = []
        
        # Default top PCA features for magnitude calculation
        self.top_pca_features = self.config.get(
            "top_pca_features",
            ["V14", "V12", "V10", "V17", "V16", "V3", "V7", "V11"]
        )
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all engineered features.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with raw features
            
        Returns
        -------
        pd.DataFrame
            Dataframe with engineered features added
        """
        logger.info("Engineering features...")
        
        df = df.copy()
        
        # Time-based features
        df = self._create_time_features(df)
        
        # Amount-based features
        df = self._create_amount_features(df)
        
        # PCA-based features
        df = self._create_pca_features(df)
        
        # Interaction features
        df = self._create_interaction_features(df)
        
        logger.info(f"Created {len(self.feature_names)} new features")
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        if "Time" not in df.columns:
            logger.warning("'Time' column not found, skipping time features")
            return df
        
        # Hour of day (cyclical encoding)
        df["Hour"] = (df["Time"] / 3600) % 24
        df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
        df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
        
        self.feature_names.extend(["Hour_sin", "Hour_cos"])
        logger.debug("Created time features: Hour_sin, Hour_cos")
        
        return df
    
    def _create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create amount-based features."""
        if "Amount" not in df.columns:
            logger.warning("'Amount' column not found, skipping amount features")
            return df
        
        # Log-transformed amount
        df["Amount_log"] = np.log1p(df["Amount"])
        
        self.feature_names.append("Amount_log")
        logger.debug("Created amount features: Amount_log")
        
        return df
    
    def _create_pca_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create PCA-based features."""
        # Check if top PCA features exist
        available_features = [f for f in self.top_pca_features if f in df.columns]
        
        if len(available_features) < 2:
            logger.warning("Not enough PCA features found, skipping PCA features")
            return df
        
        # PCA magnitude (L2 norm of top discriminative features)
        df["PCA_magnitude"] = np.sqrt((df[available_features] ** 2).sum(axis=1))
        
        self.feature_names.append("PCA_magnitude")
        logger.debug("Created PCA features: PCA_magnitude")
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        # V14 * V12 interaction (two most discriminative features)
        if "V14" in df.columns and "V12" in df.columns:
            df["V14_V12"] = df["V14"] * df["V12"]
            self.feature_names.append("V14_V12")
            logger.debug("Created interaction features: V14_V12")
        
        return df
    
    def get_feature_list(
        self,
        include_original: bool = True,
        exclude_cols: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get list of features to use for modeling.
        
        Parameters
        ----------
        include_original : bool
            Whether to include original V1-V28 and Amount features
        exclude_cols : list, optional
            Columns to exclude
            
        Returns
        -------
        list
            List of feature names
        """
        exclude_cols = exclude_cols or ["Time", "Hour", "Class"]
        
        features = []
        
        if include_original:
            # Original PCA features
            features.extend([f"V{i}" for i in range(1, 29)])
            features.append("Amount")
        
        # Engineered features
        features.extend(self.feature_names)
        
        # Remove excluded columns
        features = [f for f in features if f not in exclude_cols]
        
        return features


def compute_feature_statistics(
    X: pd.DataFrame,
    y: pd.Series
) -> pd.DataFrame:
    """
    Compute feature statistics by class.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target labels
        
    Returns
    -------
    pd.DataFrame
        Statistics for each feature
    """
    stats = []
    
    for col in X.columns:
        fraud_values = X.loc[y == 1, col]
        normal_values = X.loc[y == 0, col]
        
        normal_mean = normal_values.mean()
        normal_std = normal_values.std()
        fraud_mean = fraud_values.mean()
        
        stats.append({
            "feature": col,
            "mean_normal": normal_mean,
            "std_normal": normal_std,
            "mean_fraud": fraud_mean,
            "std_fraud": fraud_values.std(),
            "mean_diff": abs(fraud_mean - normal_mean),
            "standardized_diff": abs(fraud_mean - normal_mean) / (normal_std + 1e-8)
        })
    
    stats_df = pd.DataFrame(stats)
    stats_df = stats_df.sort_values("standardized_diff", ascending=False)
    
    return stats_df
