"""
Data Validator Module

This module provides data validation utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger("fraud_detection.data.validator")


@dataclass
class ValidationResult:
    """Result of data validation."""
    
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    statistics: Dict[str, Any]
    
    def __str__(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        result = f"Validation {status}\n"
        
        if self.errors:
            result += f"Errors ({len(self.errors)}):\n"
            for error in self.errors:
                result += f"  - {error}\n"
        
        if self.warnings:
            result += f"Warnings ({len(self.warnings)}):\n"
            for warning in self.warnings:
                result += f"  - {warning}\n"
        
        return result


class DataValidator:
    """
    Data validator for credit card transaction data.
    
    This class validates data structure, types, and values
    to ensure data quality before model training or inference.
    """
    
    # Expected feature columns
    REQUIRED_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    OPTIONAL_COLUMNS = ["Time", "Class"]
    
    def __init__(self):
        """Initialize the DataValidator."""
        self.validation_result: Optional[ValidationResult] = None
    
    def validate(self, df: pd.DataFrame, for_training: bool = True) -> ValidationResult:
        """
        Validate the dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to validate
        for_training : bool
            Whether validation is for training (requires Class column)
            
        Returns
        -------
        ValidationResult
            Validation result with errors and warnings
        """
        errors = []
        warnings = []
        statistics = {}
        
        # Check required columns
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check target column for training
        if for_training and "Class" not in df.columns:
            errors.append("Missing target column 'Class' for training")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        if total_missing > 0:
            warnings.append(f"Found {total_missing} missing values")
            statistics["missing_by_column"] = missing_counts[missing_counts > 0].to_dict()
        
        # Check data types
        for col in self.REQUIRED_COLUMNS:
            if col in df.columns and not np.issubdtype(df[col].dtype, np.number):
                errors.append(f"Column '{col}' should be numeric, got {df[col].dtype}")
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_counts = np.isinf(df[numeric_cols]).sum()
        total_inf = inf_counts.sum()
        if total_inf > 0:
            warnings.append(f"Found {total_inf} infinite values")
        
        # Check Amount range
        if "Amount" in df.columns:
            if (df["Amount"] < 0).any():
                errors.append("Found negative values in 'Amount' column")
            statistics["amount_range"] = {
                "min": float(df["Amount"].min()),
                "max": float(df["Amount"].max()),
                "mean": float(df["Amount"].mean())
            }
        
        # Check Class values for training
        if for_training and "Class" in df.columns:
            unique_classes = df["Class"].unique()
            if not set(unique_classes).issubset({0, 1}):
                errors.append(f"Invalid class values: {unique_classes}. Expected [0, 1]")
            
            # Class distribution
            class_counts = df["Class"].value_counts().to_dict()
            statistics["class_distribution"] = class_counts
            
            # Check for extreme imbalance
            if len(class_counts) == 2:
                minority_ratio = min(class_counts.values()) / sum(class_counts.values())
                if minority_ratio < 0.001:
                    warnings.append(
                        f"Extreme class imbalance detected: minority class is "
                        f"{minority_ratio:.4%} of data"
                    )
        
        # Check for duplicates
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            warnings.append(f"Found {n_duplicates} duplicate rows")
        statistics["n_duplicates"] = int(n_duplicates)
        
        # General statistics
        statistics["n_rows"] = len(df)
        statistics["n_columns"] = len(df.columns)
        
        # Create result
        is_valid = len(errors) == 0
        self.validation_result = ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            statistics=statistics
        )
        
        # Log result
        if is_valid:
            logger.info("Data validation passed")
        else:
            logger.error(f"Data validation failed: {errors}")
        
        if warnings:
            logger.warning(f"Validation warnings: {warnings}")
        
        return self.validation_result


def validate_transaction(transaction: Dict[str, float]) -> ValidationResult:
    """
    Validate a single transaction for inference.
    
    Parameters
    ----------
    transaction : dict
        Transaction data as dictionary
        
    Returns
    -------
    ValidationResult
        Validation result
    """
    errors = []
    warnings = []
    
    # Check required fields
    required_fields = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    missing_fields = set(required_fields) - set(transaction.keys())
    
    if missing_fields:
        errors.append(f"Missing required fields: {missing_fields}")
    
    # Check value types
    for key, value in transaction.items():
        if key in required_fields:
            if not isinstance(value, (int, float)):
                errors.append(f"Field '{key}' must be numeric, got {type(value).__name__}")
            elif np.isnan(value) or np.isinf(value):
                errors.append(f"Field '{key}' contains invalid value: {value}")
    
    # Check Amount
    if "Amount" in transaction:
        if transaction["Amount"] < 0:
            errors.append("Amount cannot be negative")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        statistics={"n_fields": len(transaction)}
    )
