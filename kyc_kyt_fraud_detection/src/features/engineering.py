"""
Feature Engineering Module

Main pipeline for combining KYC and KYT features.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
import logging

from .kyc_features import KYCFeatureEngineer
from .kyt_features import KYTFeatureEngineer

logger = logging.getLogger("kyc_kyt.features.engineering")


class FeatureEngineer:
    """
    Main feature engineering pipeline.

    Combines KYC and KYT features and prepares data for modeling.
    """

    def __init__(self):
        """Initialize feature engineer."""
        self.kyc_engineer = KYCFeatureEngineer()
        self.kyt_engineer = KYTFeatureEngineer()
        self.feature_columns = []

    def engineer_features(
        self,
        loan_df: pd.DataFrame,
        trans_stats: pd.DataFrame,
        merge_key: str = 'account_id'
    ) -> pd.DataFrame:
        """
        Engineer complete feature set combining KYC and KYT.

        Parameters
        ----------
        loan_df : pd.DataFrame
            Loan dataframe
        trans_stats : pd.DataFrame
            Aggregated transaction statistics
        merge_key : str
            Key to merge on

        Returns
        -------
        pd.DataFrame
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering pipeline...")

        # Step 1: Create KYC features
        df_with_kyc = self.kyc_engineer.create_features(loan_df)

        # Step 2: Create KYT features
        trans_with_kyt = self.kyt_engineer.create_features(trans_stats)

        # Step 3: Merge loan data with transaction features
        logger.info(f"Merging on '{merge_key}'...")
        df = df_with_kyc.merge(
            trans_with_kyt,
            left_on=merge_key,
            right_index=True,
            how='left'
        )

        # Step 4: Create combined features (KYC + KYT interactions)
        df = self._create_interaction_features(df)

        # Step 5: Handle missing values
        df = self._handle_missing_values(df)

        logger.info(
            f"Feature engineering complete: {df.shape[1]} total columns"
        )

        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between KYC and KYT.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with KYC and KYT features

        Returns
        -------
        pd.DataFrame
            DataFrame with interaction features added
        """
        logger.info("Creating KYC-KYT interaction features...")

        # Loan to Balance Ratio (key risk indicator)
        if 'amount' in df.columns and 'balance_mean' in df.columns:
            df['loan_to_balance'] = df['amount'] / (df['balance_mean'].abs() + 1)

        # Payment burden relative to transaction volume
        if 'payments' in df.columns and 'amount_mean' in df.columns:
            df['payment_to_avg_transaction'] = df['payments'] / (
                df['amount_mean'].abs() + 1
            )

        # Loan size vs transaction history
        if 'amount' in df.columns and 'amount_sum' in df.columns:
            df['loan_to_transaction_history'] = df['amount'] / (
                df['amount_sum'].abs() + 1
            )

        # Balance stability with payment burden
        if 'balance_stability' in df.columns and 'payment_to_amount' in df.columns:
            df['stability_payment_ratio'] = df['balance_stability'] * (
                1 - df['payment_to_amount']
            )

        # Risk score (combined indicator)
        risk_indicators = []

        if 'had_negative_balance' in df.columns:
            risk_indicators.append('had_negative_balance')

        if 'balance_cv' in df.columns:
            # High balance volatility is risky
            df['high_balance_volatility'] = (
                df['balance_cv'] > df['balance_cv'].quantile(0.75)
            ).astype(int)
            risk_indicators.append('high_balance_volatility')

        if 'loan_to_balance' in df.columns:
            # High loan-to-balance ratio is risky
            df['high_loan_to_balance'] = (
                df['loan_to_balance'] > df['loan_to_balance'].quantile(0.75)
            ).astype(int)
            risk_indicators.append('high_loan_to_balance')

        if risk_indicators:
            df['risk_flag_count'] = df[risk_indicators].sum(axis=1)

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in features.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with features

        Returns
        -------
        pd.DataFrame
            DataFrame with missing values handled
        """
        # Log missing value counts
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(
                f"Missing values detected in {(missing_counts > 0).sum()} columns"
            )

        # Fill numeric missing values with 0
        # (These are typically from accounts without transaction history)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        return df

    def prepare_modeling_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'is_default',
        exclude_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare data for modeling.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with all features
        target_col : str
            Target column name
        exclude_cols : List[str], optional
            Columns to exclude from features

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series, List[str]]
            (X, y, feature_columns)
        """
        if exclude_cols is None:
            exclude_cols = [
                'loan_id', 'account_id', 'date', 'status',
                'is_default', 'duration_bin'  # categorical
            ]

        # Select feature columns
        feature_cols = [
            c for c in df.columns
            if c not in exclude_cols
            and df[c].dtype in ['int64', 'float64']
            and df[c].std() > 0  # Remove zero-variance features
        ]

        X = df[feature_cols].fillna(0)
        y = df[target_col] if target_col in df.columns else None

        self.feature_columns = feature_cols

        logger.info(
            f"Prepared modeling data: {len(feature_cols)} features, "
            f"{len(X)} samples"
        )

        return X, y, feature_cols

    def get_feature_importance_summary(
        self,
        importance_dict: dict
    ) -> pd.DataFrame:
        """
        Create summary of feature importance by category.

        Parameters
        ----------
        importance_dict : dict
            Feature importance scores

        Returns
        -------
        pd.DataFrame
            Summary by feature category
        """
        importance_df = pd.DataFrame({
            'feature': list(importance_dict.keys()),
            'importance': list(importance_dict.values())
        })

        # Categorize features
        def categorize(feature_name):
            if feature_name in self.kyc_engineer.get_feature_names():
                return 'KYC'
            elif feature_name in self.kyt_engineer.get_feature_names():
                return 'KYT'
            elif any(kw in feature_name for kw in ['amount', 'duration', 'payment']):
                return 'KYC (base)'
            elif any(kw in feature_name for kw in ['balance', 'transaction', 'tx']):
                return 'KYT (base)'
            else:
                return 'Interaction'

        importance_df['category'] = importance_df['feature'].apply(categorize)

        # Summary by category
        summary = importance_df.groupby('category')['importance'].agg([
            'mean', 'sum', 'count'
        ]).sort_values('sum', ascending=False)

        return summary
