#!/usr/bin/env python
"""
Model Training Script

Trains loan default prediction models with optional Optuna optimization.

Usage:
    python scripts/train.py
    python scripts/train.py --model xgboost
    python scripts/train.py --optimize --n-trials 50
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src.data.loader import CzechBankDataLoader
from src.data.aggregator import TransactionAggregator
from src.data.validator import DataValidator
from src.features.engineering import FeatureEngineer
from src.models.supervised import (
    RandomForestDefaultModel,
    XGBoostDefaultModel
)
from src.models.unsupervised import AutoencoderAnomalyModel
from src.models.optimizer import OptunaOptimizer
from src.evaluation.metrics import (
    MetricsCalculator,
    print_evaluation_report
)
from src.evaluation.statistical_tests import print_discrimination_report
from config.settings import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("kyc_kyt.train")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train KYC/KYT loan default prediction models"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["all", "xgboost", "random_forest", "autoencoder"],
        default="all",
        help="Model to train (default: all)"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Perform hyperparameter optimization with Optuna"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials (default: 50)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(settings.model.models_dir),
        help="Output directory for models"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(settings.data.raw_data_dir),
        help="Directory containing data files"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size (default: 0.2)"
    )

    return parser.parse_args()


def main(args):
    """Main training function."""
    logger.info("=" * 70)
    logger.info("KYC/KYT LOAN DEFAULT PREDICTION - MODEL TRAINING")
    logger.info("=" * 70)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =================================================================
    # 1. LOAD DATA
    # =================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 1: LOADING DATA")
    logger.info("="*70)

    data_loader = CzechBankDataLoader(args.data_dir)
    loan_df, trans_df = data_loader.load_all()
    data_loader.print_summary(loan_df, trans_df)

    # =================================================================
    # 2. VALIDATE DATA
    # =================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 2: VALIDATING DATA")
    logger.info("="*70)

    validator = DataValidator()
    validator.validate_loans(loan_df)
    validator.validate_transactions(trans_df)

    # =================================================================
    # 3. AGGREGATE TRANSACTIONS
    # =================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 3: AGGREGATING TRANSACTIONS")
    logger.info("="*70)

    aggregator = TransactionAggregator()
    trans_agg = aggregator.aggregate(trans_df)

    validator.validate_merge(loan_df, trans_agg)
    validator.print_report()

    # =================================================================
    # 4. CREATE TARGET VARIABLE
    # =================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 4: CREATING TARGET VARIABLE")
    logger.info("="*70)

    loan_df['is_default'] = (
        loan_df[settings.data.status_column] == settings.data.default_status
    ).astype(int)

    default_rate = loan_df['is_default'].mean()
    logger.info(f"Default rate: {default_rate*100:.2f}%")
    logger.info(f"Imbalance ratio: {(1-default_rate)/default_rate:.1f}:1")

    # =================================================================
    # 5. FEATURE ENGINEERING
    # =================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 5: FEATURE ENGINEERING")
    logger.info("="*70)

    engineer = FeatureEngineer()
    df_engineered = engineer.engineer_features(loan_df, trans_agg)

    # Statistical analysis
    print_discrimination_report(
        df_engineered,
        settings.data.target_column,
        top_n=15
    )

    # =================================================================
    # 6. PREPARE MODELING DATA
    # =================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 6: PREPARING MODELING DATA")
    logger.info("="*70)

    X, y, feature_cols = engineer.prepare_modeling_data(df_engineered)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=settings.data.random_state,
        stratify=y if settings.data.stratify else None
    )

    # Scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)

    logger.info(f"Training set: {len(X_train)} samples ({y_train.sum()} defaults)")
    logger.info(f"Test set:     {len(X_test)} samples ({y_test.sum()} defaults)")
    logger.info(f"Features:     {len(feature_cols)}")

    # Save scaler
    scaler_path = output_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to: {scaler_path}")

    # =================================================================
    # 7. TRAIN MODELS
    # =================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 7: TRAINING MODELS")
    logger.info("="*70)

    models_to_train = []
    if args.model == "all":
        models_to_train = ["random_forest", "xgboost", "autoencoder"]
    else:
        models_to_train = [args.model]

    trained_models = {}
    metrics_calc = MetricsCalculator()

    for model_name in models_to_train:
        logger.info(f"\n{'='*70}")
        logger.info(f"TRAINING: {model_name.upper()}")
        logger.info(f"{'='*70}")

        # Hyperparameter optimization
        if args.optimize and model_name in ["xgboost", "random_forest"]:
            logger.info(f"\nRunning Optuna optimization ({args.n_trials} trials)...")

            optimizer = OptunaOptimizer(
                n_trials=args.n_trials,
                **vars(settings.optuna)
            )

            if model_name == "xgboost":
                best_params = optimizer.optimize_xgboost(X_train_scaled, y_train)
                model = XGBoostDefaultModel(params=best_params, use_tuned_params=False)
            else:  # random_forest
                best_params = optimizer.optimize_random_forest(X_train_scaled, y_train)
                model = RandomForestDefaultModel(
                    params=best_params,
                    use_tuned_params=False
                )

            # Save optimization history
            history = optimizer.get_optimization_history()
            history.to_csv(
                output_dir.parent / "reports" / "optuna_studies" / f"{model_name}_optimization.csv"
            )

        else:
            # Use pre-tuned parameters
            if model_name == "xgboost":
                model = XGBoostDefaultModel(use_tuned_params=True)
            elif model_name == "random_forest":
                model = RandomForestDefaultModel(use_tuned_params=True)
            elif model_name == "autoencoder":
                model = AutoencoderAnomalyModel(use_tuned_params=True)

        # Train model
        if model_name == "autoencoder":
            model.fit(X_train_scaled, y_train, train_on_normal_only=True)
        else:
            model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)

        print_evaluation_report(y_test.values, y_pred, y_proba, model_name.upper())

        metrics_calc.evaluate_model(
            y_test.values,
            y_pred,
            y_proba,
            model_name
        )

        # Save model
        model_path = output_dir / f"{model_name}_model.pkl"
        model.save(model_path)
        logger.info(f"Model saved to: {model_path}")

        trained_models[model_name] = model

    # =================================================================
    # 8. MODEL COMPARISON
    # =================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 8: MODEL COMPARISON")
    logger.info("="*70)

    comparison = metrics_calc.compare_models()
    print("\n" + "="*70)
    print("MODEL COMPARISON (sorted by Average Precision)")
    print("="*70)
    print(comparison.to_string(index=False))
    print("="*70)

    # Save comparison
    comparison.to_csv(output_dir.parent / "reports" / "metrics" / "model_comparison.csv")

    best_model_name = metrics_calc.get_best_model()
    logger.info(f"\n🏆 Best Model: {best_model_name}")

    # =================================================================
    # 9. FEATURE IMPORTANCE (Best Model)
    # =================================================================
    if best_model_name in ["random_forest", "xgboost"]:
        logger.info("\n" + "="*70)
        logger.info(f"STEP 9: FEATURE IMPORTANCE ({best_model_name.upper()})")
        logger.info("="*70)

        best_model = trained_models[best_model_name]
        importance_df = best_model.get_feature_importance()

        print("\nTop 20 Most Important Features:")
        print(importance_df.head(20).to_string(index=False))

        # Save feature importance
        importance_df.to_csv(
            output_dir.parent / "reports" / "metrics" / f"{best_model_name}_feature_importance.csv",
            index=False
        )

    # =================================================================
    # TRAINING COMPLETE
    # =================================================================
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*70)
    logger.info(f"Models saved to: {output_dir}")
    logger.info(f"Reports saved to: {output_dir.parent / 'reports'}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
