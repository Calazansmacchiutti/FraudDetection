"""
KYT Features Module

Know Your Transaction (KYT) features based on transaction behavior patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger("kyc_kyt.features.kyt")


class KYTFeatureEngineer:
    """
    Creates KYT (Know Your Transaction) behavioral features.

    These features capture transaction patterns and financial behavior.
    """

    def __init__(self):
        """Initialize KYT feature engineer."""
        self.feature_names = []

    def create_features(self, trans_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Create KYT features from aggregated transaction statistics.

        Parameters
        ----------
        trans_stats : pd.DataFrame
            Aggregated transaction statistics (from TransactionAggregator)

        Returns
        -------
        pd.DataFrame
            DataFrame with KYT features
        """
        logger.info("Creating KYT features...")

        df = trans_stats.copy()
        created_features = []

        # Transaction Volume Features
        if 'n_transactions' in df.columns:
            # Transaction frequency (normalized by account age could be added)
            df['tx_frequency'] = df['n_transactions']
            created_features.append('tx_frequency')

        # Amount-based Features
        if 'amount_std' in df.columns and 'amount_mean' in df.columns:
            # Coefficient of Variation (volatility)
            df['amount_cv'] = df['amount_std'] / (df['amount_mean'].abs() + 1)
            created_features.append('amount_cv')

        if 'amount_max' in df.columns and 'amount_mean' in df.columns:
            # Max-to-mean ratio (detects large outlier transactions)
            df['amount_max_ratio'] = df['amount_max'] / (df['amount_mean'].abs() + 1)
            created_features.append('amount_max_ratio')

        if 'amount_min' in df.columns and 'amount_mean' in df.columns:
            # Min-to-mean ratio
            df['amount_min_ratio'] = df['amount_min'] / (df['amount_mean'].abs() + 1)
            created_features.append('amount_min_ratio')

        if 'amount_sum' in df.columns and 'n_transactions' in df.columns:
            # Average transaction size
            df['avg_transaction_size'] = df['amount_sum'] / (df['n_transactions'] + 1)
            created_features.append('avg_transaction_size')

        # Balance-based Features
        if 'balance_std' in df.columns and 'balance_mean' in df.columns:
            # Balance stability (inverse of CV)
            df['balance_stability'] = df['balance_mean'].abs() / (df['balance_std'] + 1)
            created_features.append('balance_stability')

            # Balance coefficient of variation
            df['balance_cv'] = df['balance_std'] / (df['balance_mean'].abs() + 1)
            created_features.append('balance_cv')

        if 'balance_max' in df.columns and 'balance_min' in df.columns:
            # Balance range (volatility indicator)
            df['balance_range'] = df['balance_max'] - df['balance_min']
            created_features.append('balance_range')

            # Balance range ratio
            df['balance_range_ratio'] = df['balance_range'] / (
                df['balance_mean'].abs() + 1
            )
            created_features.append('balance_range_ratio')

        if 'balance_min' in df.columns:
            # Negative balance flag (important risk indicator)
            df['had_negative_balance'] = (df['balance_min'] < 0).astype(int)
            created_features.append('had_negative_balance')

            # Severity of negative balance
            df['negative_balance_depth'] = df['balance_min'].clip(upper=0).abs()
            created_features.append('negative_balance_depth')

        if 'balance_median' in df.columns and 'balance_mean' in df.columns:
            # Skewness indicator (mean vs median)
            df['balance_skew_indicator'] = (
                df['balance_mean'] - df['balance_median']
            ) / (df['balance_std'] + 1)
            created_features.append('balance_skew_indicator')

        # Transaction Type Features (if available)
        type_cols = [c for c in df.columns if c.startswith('pct_type_')]
        if len(type_cols) > 1:
            # Transaction diversity (entropy-like measure)
            type_df = df[type_cols]
            df['tx_type_diversity'] = -1 * (
                type_df * np.log(type_df + 1e-10)
            ).sum(axis=1)
            created_features.append('tx_type_diversity')

        self.feature_names = created_features
        logger.info(f"Created {len(created_features)} KYT features")

        return df

    def get_feature_names(self) -> list:
        """
        Get names of created features.

        Returns
        -------
        list
            Feature names
        """
        return self.feature_names


def create_kyt_features(trans_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to create KYT features.

    Parameters
    ----------
    trans_stats : pd.DataFrame
        Aggregated transaction statistics

    Returns
    -------
    pd.DataFrame
        DataFrame with KYT features added
    """
    engineer = KYTFeatureEngineer()
    return engineer.create_features(trans_stats)
