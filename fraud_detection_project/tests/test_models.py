"""
Tests for fraud detection models.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestDataLoader:
    """Tests for data loading functionality."""
    
    def test_split_features_target(self):
        """Test splitting features and target."""
        from src.data import split_features_target
        
        df = pd.DataFrame({
            "V1": [1, 2, 3],
            "V2": [4, 5, 6],
            "Amount": [100, 200, 300],
            "Class": [0, 1, 0]
        })
        
        X, y = split_features_target(df)
        
        assert "Class" not in X.columns
        assert len(X) == 3
        assert len(y) == 3
        assert list(y) == [0, 1, 0]


class TestDataPreprocessor:
    """Tests for data preprocessing."""
    
    def test_preprocessor_fit_transform(self):
        """Test preprocessor fit and transform."""
        from src.data import DataPreprocessor
        
        X = pd.DataFrame({
            "V1": [1, 2, 3, 4, 5],
            "V2": [10, 20, 30, 40, 50],
            "Amount": [100, 200, 300, 400, 500]
        })
        
        preprocessor = DataPreprocessor(scaler_type="robust")
        X_scaled = preprocessor.fit_transform(X)
        
        assert preprocessor.is_fitted
        assert X_scaled.shape == X.shape
        assert list(X_scaled.columns) == list(X.columns)


class TestFeatureEngineer:
    """Tests for feature engineering."""
    
    def test_engineer_features(self):
        """Test feature creation."""
        from src.features import FeatureEngineer
        
        df = pd.DataFrame({
            "Time": [0, 3600, 7200],
            "Amount": [100, 200, 300],
            "V1": [1, 2, 3],
            "V3": [4, 5, 6],
            "V7": [7, 8, 9],
            "V10": [10, 11, 12],
            "V11": [13, 14, 15],
            "V12": [16, 17, 18],
            "V14": [19, 20, 21],
            "V16": [22, 23, 24],
            "V17": [25, 26, 27],
            "Class": [0, 1, 0]
        })
        
        engineer = FeatureEngineer()
        df_new = engineer.engineer_features(df)
        
        assert "Hour_sin" in df_new.columns
        assert "Hour_cos" in df_new.columns
        assert "Amount_log" in df_new.columns
        assert "PCA_magnitude" in df_new.columns


class TestModels:
    """Tests for model classes."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 30
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"V{i}" for i in range(1, n_features + 1)]
        )
        y = pd.Series(np.random.randint(0, 2, n_samples))
        
        return X, y
    
    def test_xgboost_model(self, sample_data):
        """Test XGBoost model training and prediction."""
        from src.models import XGBoostModel
        
        X, y = sample_data
        
        model = XGBoostModel(params={"n_estimators": 10, "max_depth": 3})
        model.fit(X, y)
        
        assert model.is_fitted
        
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})
        
        proba = model.predict_proba(X)
        assert len(proba) == len(X)
        assert all(0 <= p <= 1 for p in proba)
    
    def test_random_forest_model(self, sample_data):
        """Test Random Forest model."""
        from src.models import RandomForestModel
        
        X, y = sample_data
        
        model = RandomForestModel(params={"n_estimators": 10, "max_depth": 3})
        model.fit(X, y)
        
        assert model.is_fitted
        
        importance = model.get_feature_importance()
        assert len(importance) == X.shape[1]
    
    def test_isolation_forest_model(self, sample_data):
        """Test Isolation Forest model."""
        from src.models import IsolationForestModel
        
        X, y = sample_data
        
        model = IsolationForestModel(params={"n_estimators": 10})
        model.fit(X, y, train_on_normal_only=True)
        
        assert model.is_fitted
        
        predictions = model.predict(X)
        assert len(predictions) == len(X)


class TestMetrics:
    """Tests for evaluation metrics."""
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        from src.evaluation import MetricsCalculator
        
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
        y_proba = np.array([0.1, 0.6, 0.8, 0.9, 0.3, 0.2, 0.7, 0.1])
        
        calc = MetricsCalculator()
        metrics = calc.calculate_metrics(y_true, y_pred, y_proba)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert "average_precision" in metrics
        
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
    
    def test_confusion_matrix(self):
        """Test confusion matrix calculation."""
        from src.evaluation import MetricsCalculator
        
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        
        calc = MetricsCalculator()
        cm = calc.calculate_confusion_matrix(y_true, y_pred)
        
        assert cm["true_positives"] == 1
        assert cm["true_negatives"] == 1
        assert cm["false_positives"] == 1
        assert cm["false_negatives"] == 1


class TestDataValidator:
    """Tests for data validation."""
    
    def test_valid_data(self):
        """Test validation of valid data."""
        from src.data import DataValidator
        
        df = pd.DataFrame({
            **{f"V{i}": np.random.randn(100) for i in range(1, 29)},
            "Amount": np.random.uniform(0, 1000, 100),
            "Class": np.random.randint(0, 2, 100)
        })
        
        validator = DataValidator()
        result = validator.validate(df)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_missing_columns(self):
        """Test validation with missing columns."""
        from src.data import DataValidator
        
        df = pd.DataFrame({
            "V1": [1, 2, 3],
            "Amount": [100, 200, 300],
            "Class": [0, 1, 0]
        })
        
        validator = DataValidator()
        result = validator.validate(df)
        
        assert not result.is_valid
        assert len(result.errors) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
