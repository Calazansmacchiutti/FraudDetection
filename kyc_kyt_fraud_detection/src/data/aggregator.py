"""
Transaction Aggregator Module

Aggregates transaction-level data to account-level features (KYT).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("kyc_kyt.data.aggregator")


class TransactionAggregator:
    """
    Aggregates transaction data by account_id.

    Creates KYT (Know Your Transaction) features from raw transactions.
    """

    def __init__(self):
        """Initialize transaction aggregator."""
        self.aggregated_features = None

    def aggregate(
        self,
        trans_df: pd.DataFrame,
        account_col: str = 'account_id'
    ) -> pd.DataFrame:
        """
        Aggregate transactions by account.

        Parameters
        ----------
        trans_df : pd.DataFrame
            Transaction dataframe
        account_col : str
            Column name for account identifier

        Returns
        -------
        pd.DataFrame
            Aggregated features indexed by account_id
        """
        logger.info(f"Aggregating {len(trans_df):,} transactions...")

        # Basic transaction statistics
        agg_dict = {
            'trans_id': 'count',
            'amount': ['mean', 'std', 'min', 'max', 'sum', 'median'],
            'balance': ['mean', 'std', 'min', 'max', 'median']
        }

        aggregated = trans_df.groupby(account_col).agg(agg_dict)

        # Flatten column names
        aggregated.columns = [
            'n_transactions',
            'amount_mean', 'amount_std', 'amount_min', 'amount_max',
            'amount_sum', 'amount_median',
            'balance_mean', 'balance_std', 'balance_min', 'balance_max',
            'balance_median'
        ]

        # Add transaction type percentages if available
        if 'type' in trans_df.columns:
            type_pct = trans_df.groupby(account_col)['type'].value_counts(
                normalize=True
            ).unstack(fill_value=0)
            type_pct.columns = [f'pct_type_{c}' for c in type_pct.columns]
            aggregated = aggregated.join(type_pct)

        # Add operation percentages if available
        if 'operation' in trans_df.columns:
            op_pct = trans_df.groupby(account_col)['operation'].value_counts(
                normalize=True
            ).unstack(fill_value=0)
            op_pct.columns = [f'pct_op_{c}' for c in op_pct.columns]
            aggregated = aggregated.join(op_pct)

        # Fill NaN with 0 for standard deviation (accounts with 1 transaction)
        aggregated['amount_std'] = aggregated['amount_std'].fillna(0)
        aggregated['balance_std'] = aggregated['balance_std'].fillna(0)

        self.aggregated_features = aggregated

        logger.info(
            f"Aggregated to {len(aggregated):,} accounts with "
            f"{aggregated.shape[1]} features"
        )

        return aggregated

    def get_feature_names(self) -> List[str]:
        """
        Get list of aggregated feature names.

        Returns
        -------
        List[str]
            Feature names
        """
        if self.aggregated_features is None:
            return []
        return list(self.aggregated_features.columns)

    def get_statistics(self) -> Dict:
        """
        Get summary statistics of aggregated features.

        Returns
        -------
        Dict
            Statistics dictionary
        """
        if self.aggregated_features is None:
            return {}

        stats = {
            'n_accounts': len(self.aggregated_features),
            'n_features': self.aggregated_features.shape[1],
            'avg_transactions_per_account': self.aggregated_features[
                'n_transactions'
            ].mean(),
            'median_transactions_per_account': self.aggregated_features[
                'n_transactions'
            ].median()
        }

        return stats


def aggregate_transactions(
    trans_df: pd.DataFrame,
    account_col: str = 'account_id'
) -> pd.DataFrame:
    """
    Convenience function to aggregate transactions.

    Parameters
    ----------
    trans_df : pd.DataFrame
        Transaction dataframe
    account_col : str
        Column name for account identifier

    Returns
    -------
    pd.DataFrame
        Aggregated features indexed by account_id
    """
    aggregator = TransactionAggregator()
    return aggregator.aggregate(trans_df, account_col)
