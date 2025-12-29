"""
KYC Features Module

Know Your Customer (KYC) features based on loan characteristics and customer info.
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger("kyc_kyt.features.kyc")


class KYCFeatureEngineer:
    """
    Creates KYC (Know Your Customer) features.

    These features are based on loan application data and customer demographics.
    """

    def __init__(self):
        """Initialize KYC feature engineer."""
        self.feature_names = []

    def create_features(self, loan_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create KYC features from loan data.

        Parameters
        ----------
        loan_df : pd.DataFrame
            Loan dataframe

        Returns
        -------
        pd.DataFrame
            DataFrame with KYC features added
        """
        logger.info("Creating KYC features...")

        df = loan_df.copy()
        created_features = []

        # Loan Ratio Features
        if 'payments' in df.columns and 'amount' in df.columns:
            # Payment to amount ratio (monthly payment burden)
            df['payment_to_amount'] = df['payments'] / (df['amount'] + 1)
            created_features.append('payment_to_amount')

        if 'amount' in df.columns and 'duration' in df.columns:
            # Loan amount per month
            df['amount_per_month'] = df['amount'] / (df['duration'] + 1)
            created_features.append('amount_per_month')

        if 'payments' in df.columns and 'duration' in df.columns:
            # Total expected payment over loan lifetime
            df['total_expected_payment'] = df['payments'] * df['duration']
            created_features.append('total_expected_payment')

            # Overpayment ratio (total paid / original amount)
            if 'amount' in df.columns:
                df['overpayment_ratio'] = df['total_expected_payment'] / (
                    df['amount'] + 1
                )
                created_features.append('overpayment_ratio')

                # Implied interest (total overpayment)
                df['implied_interest'] = df['total_expected_payment'] - df['amount']
                created_features.append('implied_interest')

                # Implied interest rate (approximate)
                df['implied_interest_rate'] = (
                    df['implied_interest'] / (df['amount'] * df['duration'] + 1)
                ) * 100
                created_features.append('implied_interest_rate')

        # Loan Size Categories
        if 'amount' in df.columns:
            # Log-transformed amount (reduce skewness)
            df['log_amount'] = np.log1p(df['amount'])
            created_features.append('log_amount')

            # Amount percentile within dataset
            df['amount_percentile'] = df['amount'].rank(pct=True)
            created_features.append('amount_percentile')

        # Duration Categories
        if 'duration' in df.columns:
            # Short-term vs long-term loan
            df['is_short_term'] = (df['duration'] <= 12).astype(int)
            df['is_medium_term'] = (
                (df['duration'] > 12) & (df['duration'] <= 36)
            ).astype(int)
            df['is_long_term'] = (df['duration'] > 36).astype(int)
            created_features.extend(['is_short_term', 'is_medium_term', 'is_long_term'])

            # Duration bins
            df['duration_bin'] = pd.cut(
                df['duration'],
                bins=[0, 12, 24, 36, 48, 60, np.inf],
                labels=['0-12', '13-24', '25-36', '37-48', '49-60', '60+']
            )

        self.feature_names = created_features
        logger.info(f"Created {len(created_features)} KYC features")

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


def create_kyc_features(loan_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to create KYC features.

    Parameters
    ----------
    loan_df : pd.DataFrame
        Loan dataframe

    Returns
    -------
    pd.DataFrame
        DataFrame with KYC features added
    """
    engineer = KYCFeatureEngineer()
    return engineer.create_features(loan_df)
