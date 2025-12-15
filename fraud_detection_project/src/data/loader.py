"""
Data Loader Module

This module provides utilities for loading and handling data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger("fraud_detection.data.loader")


class DataLoader:
    """
    Data loader for credit card transaction data.
    
    This class handles loading, validation, and basic inspection
    of the credit card fraud dataset.
    
    Attributes
    ----------
    data_path : Path
        Path to the data file
    df : pd.DataFrame
        Loaded dataframe
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize the DataLoader.
        
        Parameters
        ----------
        data_path : Path, optional
            Path to the CSV file. If not provided, uses default from settings.
        """
        from config.settings import settings
        
        self.data_path = Path(data_path) if data_path else settings.data.raw_data_path
        self.df: Optional[pd.DataFrame] = None
        self._metadata: Dict[str, Any] = {}
    
    def load(self, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Parameters
        ----------
        **kwargs
            Additional arguments passed to pd.read_csv
            
        Returns
        -------
        pd.DataFrame
            Loaded dataframe
            
        Raises
        ------
        FileNotFoundError
            If the data file does not exist
        """
        if not self.data_path.exists():
            logger.error(f"Data file not found: {self.data_path}")
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}. "
                "Please download from https://www.kaggle.com/mlg-ulb/creditcardfraud"
            )
        
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path, **kwargs)
        
        # Store metadata
        self._metadata = {
            "n_rows": len(self.df),
            "n_columns": len(self.df.columns),
            "columns": list(self.df.columns),
            "memory_usage": self.df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        logger.info(
            f"Loaded {self._metadata['n_rows']:,} rows, "
            f"{self._metadata['n_columns']} columns, "
            f"{self._metadata['memory_usage']:.2f} MB"
        )
        
        return self.df
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the loaded data.
        
        Returns
        -------
        dict
            Dictionary containing summary statistics
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load() first.")
        
        n_fraud = self.df["Class"].sum()
        n_normal = len(self.df) - n_fraud
        
        summary = {
            "total_transactions": len(self.df),
            "n_features": len(self.df.columns) - 1,
            "n_fraud": int(n_fraud),
            "n_normal": int(n_normal),
            "fraud_ratio": n_fraud / len(self.df),
            "imbalance_ratio": n_normal / n_fraud if n_fraud > 0 else float("inf"),
            "missing_values": int(self.df.isnull().sum().sum()),
            "duplicates": int(self.df.duplicated().sum()),
            "memory_mb": self._metadata.get("memory_usage", 0)
        }
        
        return summary
    
    def print_summary(self) -> None:
        """Print formatted summary of the dataset."""
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print("DATASET SUMMARY".center(60))
        print("=" * 60)
        print(f"\nTotal transactions:     {summary['total_transactions']:,}")
        print(f"Number of features:     {summary['n_features']}")
        print(f"\nClass Distribution:")
        print(f"  Legitimate (Class 0): {summary['n_normal']:,} ({100 - summary['fraud_ratio']*100:.2f}%)")
        print(f"  Fraudulent (Class 1): {summary['n_fraud']:,} ({summary['fraud_ratio']*100:.4f}%)")
        print(f"  Imbalance ratio:      {summary['imbalance_ratio']:.0f}:1")
        print(f"\nData Quality:")
        print(f"  Missing values:       {summary['missing_values']}")
        print(f"  Duplicate records:    {summary['duplicates']}")
        print(f"  Memory usage:         {summary['memory_mb']:.2f} MB")
        print("=" * 60 + "\n")


def load_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Convenience function to load data.
    
    Parameters
    ----------
    data_path : Path, optional
        Path to the data file
        
    Returns
    -------
    pd.DataFrame
        Loaded dataframe
    """
    loader = DataLoader(data_path)
    return loader.load()


def split_features_target(
    df: pd.DataFrame,
    target_col: str = "Class",
    drop_cols: Optional[list] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features and target.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the target column
    drop_cols : list, optional
        Additional columns to drop from features
        
    Returns
    -------
    tuple
        (X, y) tuple of features and target
    """
    drop_cols = drop_cols or []
    all_drop_cols = [target_col] + drop_cols
    
    X = df.drop(columns=[c for c in all_drop_cols if c in df.columns])
    y = df[target_col]
    
    return X, y
