"""
Pytest configuration and fixtures.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_transaction():
    """Sample transaction data."""
    return {
        "V1": -1.359807,
        "V2": -0.072781,
        "V3": 2.536347,
        "V4": 1.378155,
        "V5": -0.338321,
        "V6": 0.462388,
        "V7": 0.239599,
        "V8": 0.098698,
        "V9": 0.363787,
        "V10": 0.090794,
        "V11": -0.551600,
        "V12": -0.617801,
        "V13": -0.991390,
        "V14": -0.311169,
        "V15": 1.468177,
        "V16": -0.470401,
        "V17": 0.207971,
        "V18": 0.025791,
        "V19": 0.403993,
        "V20": 0.251412,
        "V21": -0.018307,
        "V22": 0.277838,
        "V23": -0.110474,
        "V24": 0.066928,
        "V25": 0.128539,
        "V26": -0.189115,
        "V27": 0.133558,
        "V28": -0.021053,
        "Amount": 149.62
    }


@pytest.fixture
def sample_dataset():
    """Sample dataset for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create features
    data = {f"V{i}": np.random.randn(n_samples) for i in range(1, 29)}
    data["Time"] = np.random.uniform(0, 172800, n_samples)
    data["Amount"] = np.random.uniform(0, 500, n_samples)
    data["Class"] = np.random.choice([0, 1], n_samples, p=[0.99, 0.01])
    
    return pd.DataFrame(data)


@pytest.fixture
def trained_model(sample_dataset):
    """Trained XGBoost model for testing."""
    from src.data import split_features_target, DataPreprocessor
    from src.models import XGBoostModel
    
    X, y = split_features_target(sample_dataset, drop_cols=["Time"])
    
    preprocessor = DataPreprocessor()
    X_scaled = preprocessor.fit_transform(X)
    
    model = XGBoostModel(params={"n_estimators": 10, "max_depth": 3})
    model.fit(X_scaled, y)
    
    return model, preprocessor
