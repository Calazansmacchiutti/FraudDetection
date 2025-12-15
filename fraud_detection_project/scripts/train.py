#!/usr/bin/env python
"""
Model Training Script

This script trains all fraud detection models.

Usage:
    python scripts/train.py
    python scripts/train.py --model xgboost
    python scripts/train.py --output models/
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime

from src.data import DataLoader, DataPreprocessor, split_features_target, prepare_train_test_split
from src.features import FeatureEngineer
from src.models import (
    XGBoostModel, RandomForestModel,
    IsolationForestModel, KMeansAnomalyModel
)
from src.evaluation import MetricsCalculator, print_evaluation_report
from config.settings import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("fraud_detection.train")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train fraud detection models"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(settings.data.raw_data_path),
        help="Path to training data"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["all", "xgboost", "random_forest", "isolation_forest", "kmeans"],
        default="all",
        help="Model to train"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(settings.model.models_dir),
        help="Output directory for models"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size"
    )
    
    return parser.parse_args()


def train_models(args):
    """Main training function."""
    logger.info("=" * 70)
    logger.info("FRAUD DETECTION MODEL TRAINING")
    logger.info("=" * 70)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("\n1. Loading data...")
    loader = DataLoader(args.data)
    df = loader.load()
    loader.print_summary()
    
    # Feature engineering
    logger.info("\n2. Engineering features...")
    engineer = FeatureEngineer()
    df = engineer.engineer_features(df)
    
    # Prepare features and target
    X, y = split_features_target(df, drop_cols=["Time", "Hour"])
    
    # Train-test split
    logger.info("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        X, y, test_size=args.test_size
    )
    
    # Preprocessing
    logger.info("\n4. Preprocessing features...")
    preprocessor = DataPreprocessor(scaler_type="robust")
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    # Save preprocessor
    preprocessor.save(output_dir / "scaler.pkl")
    
    # Initialize metrics calculator
    metrics_calc = MetricsCalculator()
    
    # Train models
    logger.info("\n5. Training models...")
    
    models_to_train = []
    if args.model == "all":
        models_to_train = ["xgboost", "random_forest", "isolation_forest", "kmeans"]
    else:
        models_to_train = [args.model]
    
    trained_models = {}
    
    for model_name in models_to_train:
        logger.info(f"\n--- Training {model_name} ---")
        
        if model_name == "xgboost":
            model = XGBoostModel.with_auto_weight(y_train)
            model.fit(X_train_scaled, y_train)
            
        elif model_name == "random_forest":
            model = RandomForestModel()
            model.fit(X_train_scaled, y_train)
            
        elif model_name == "isolation_forest":
            model = IsolationForestModel()
            model.fit(X_train_scaled, y_train, train_on_normal_only=True)
            
        elif model_name == "kmeans":
            model = KMeansAnomalyModel()
            model.fit(X_train_scaled)
        
        # Save model
        model_path = output_dir / f"{model_name}_model.pkl"
        model.save(model_path)
        trained_models[model_name] = model
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        
        print_evaluation_report(
            y_test.values, y_pred, y_proba, model_name.upper()
        )
        
        metrics_calc.evaluate_model(
            model_name, y_test.values, y_pred, y_proba
        )
    
    # Model comparison
    logger.info("\n6. Model Comparison")
    comparison = metrics_calc.compare_models()
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (sorted by Average Precision)")
    print("=" * 70)
    print(comparison.to_string())
    print("=" * 70)
    
    # Save comparison
    comparison.to_csv(output_dir / "model_comparison.csv")
    
    logger.info(f"\nModels saved to: {output_dir}")
    logger.info("Training complete!")
    
    return trained_models


if __name__ == "__main__":
    args = parse_args()
    train_models(args)
