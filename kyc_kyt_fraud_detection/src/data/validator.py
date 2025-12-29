"""
Data Validator Module

Validates data quality and performs integrity checks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("kyc_kyt.data.validator")


class DataValidator:
    """
    Validates data quality for loan and transaction datasets.
    """

    def __init__(self):
        """Initialize data validator."""
        self.validation_results = {}

    def validate_loans(self, loan_df: pd.DataFrame) -> Dict:
        """
        Validate loan dataset.

        Parameters
        ----------
        loan_df : pd.DataFrame
            Loan dataframe

        Returns
        -------
        Dict
            Validation results
        """
        logger.info("Validating loan data...")

        results = {
            'table': 'loans',
            'n_rows': len(loan_df),
            'n_cols': len(loan_df.columns),
            'issues': []
        }

        # Check required columns
        required_cols = ['loan_id', 'account_id', 'date', 'amount',
                        'duration', 'payments', 'status']
        missing_cols = [c for c in required_cols if c not in loan_df.columns]
        if missing_cols:
            results['issues'].append(f"Missing columns: {missing_cols}")

        # Check duplicates
        if 'loan_id' in loan_df.columns:
            n_duplicates = loan_df['loan_id'].duplicated().sum()
            if n_duplicates > 0:
                results['issues'].append(f"Duplicate loan_ids: {n_duplicates}")

        # Check missing values
        missing_counts = loan_df.isnull().sum()
        if missing_counts.sum() > 0:
            missing_info = missing_counts[missing_counts > 0].to_dict()
            results['issues'].append(f"Missing values: {missing_info}")

        # Check status values
        if 'status' in loan_df.columns:
            valid_statuses = {'A', 'B', 'C', 'D'}
            invalid_statuses = set(loan_df['status'].unique()) - valid_statuses
            if invalid_statuses:
                results['issues'].append(
                    f"Invalid status values: {invalid_statuses}"
                )

        # Check numeric ranges
        if 'amount' in loan_df.columns:
            if (loan_df['amount'] <= 0).any():
                results['issues'].append("Negative or zero loan amounts found")

        if 'duration' in loan_df.columns:
            if (loan_df['duration'] <= 0).any():
                results['issues'].append("Negative or zero durations found")

        results['is_valid'] = len(results['issues']) == 0

        self.validation_results['loans'] = results
        return results

    def validate_transactions(self, trans_df: pd.DataFrame) -> Dict:
        """
        Validate transaction dataset.

        Parameters
        ----------
        trans_df : pd.DataFrame
            Transaction dataframe

        Returns
        -------
        Dict
            Validation results
        """
        logger.info("Validating transaction data...")

        results = {
            'table': 'transactions',
            'n_rows': len(trans_df),
            'n_cols': len(trans_df.columns),
            'issues': []
        }

        # Check required columns
        required_cols = ['trans_id', 'account_id', 'amount', 'balance']
        missing_cols = [c for c in required_cols if c not in trans_df.columns]
        if missing_cols:
            results['issues'].append(f"Missing columns: {missing_cols}")

        # Check duplicates
        if 'trans_id' in trans_df.columns:
            n_duplicates = trans_df['trans_id'].duplicated().sum()
            if n_duplicates > 0:
                results['issues'].append(
                    f"Duplicate trans_ids: {n_duplicates}"
                )

        # Check missing values (some are expected in Czech Bank data)
        missing_pct = trans_df.isnull().sum().sum() / (
            trans_df.shape[0] * trans_df.shape[1]
        ) * 100
        if missing_pct > 20:
            results['issues'].append(
                f"High missing percentage: {missing_pct:.1f}%"
            )

        results['is_valid'] = len(results['issues']) == 0

        self.validation_results['transactions'] = results
        return results

    def validate_merge(
        self,
        loan_df: pd.DataFrame,
        trans_agg: pd.DataFrame,
        merge_key: str = 'account_id'
    ) -> Dict:
        """
        Validate merge between loans and aggregated transactions.

        Parameters
        ----------
        loan_df : pd.DataFrame
            Loan dataframe
        trans_agg : pd.DataFrame
            Aggregated transaction features
        merge_key : str
            Key to merge on

        Returns
        -------
        Dict
            Validation results
        """
        logger.info("Validating merge compatibility...")

        results = {
            'operation': 'merge',
            'issues': []
        }

        # Check key existence
        if merge_key not in loan_df.columns:
            results['issues'].append(
                f"Merge key '{merge_key}' not in loan data"
            )
            results['is_valid'] = False
            return results

        # Get accounts
        loan_accounts = set(loan_df[merge_key].unique())
        trans_accounts = set(trans_agg.index.unique())

        # Check overlap
        overlap = loan_accounts.intersection(trans_accounts)
        missing_in_trans = loan_accounts - trans_accounts
        extra_in_trans = trans_accounts - loan_accounts

        results['n_loan_accounts'] = len(loan_accounts)
        results['n_trans_accounts'] = len(trans_accounts)
        results['n_overlap'] = len(overlap)
        results['overlap_pct'] = len(overlap) / len(loan_accounts) * 100

        if missing_in_trans:
            results['issues'].append(
                f"{len(missing_in_trans)} loan accounts have no transactions"
            )

        if results['overlap_pct'] < 50:
            results['issues'].append(
                f"Low overlap: {results['overlap_pct']:.1f}%"
            )

        results['is_valid'] = len(results['issues']) == 0

        self.validation_results['merge'] = results
        return results

    def print_report(self) -> None:
        """Print validation report."""
        print("=" * 60)
        print("DATA VALIDATION REPORT")
        print("=" * 60)

        for name, results in self.validation_results.items():
            print(f"\n{name.upper()}:")
            print(f"  Status: {'✅ PASS' if results['is_valid'] else '❌ FAIL'}")

            if 'n_rows' in results:
                print(f"  Rows: {results['n_rows']:,}")
            if 'n_cols' in results:
                print(f"  Columns: {results['n_cols']}")

            if results['issues']:
                print(f"  Issues:")
                for issue in results['issues']:
                    print(f"    • {issue}")
            else:
                print(f"  No issues found")

        print("=" * 60)
