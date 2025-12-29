"""
Data Loader Module

Loads and processes Czech Bank dataset (loan.csv and trans.csv).
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger("kyc_kyt.data.loader")


class CzechBankDataLoader:
    """
    Loader for Czech Bank dataset.

    Handles semicolon-separated CSV files with quoted strings.
    """

    def __init__(self, data_dir: Path):
        """
        Initialize data loader.

        Parameters
        ----------
        data_dir : Path
            Directory containing loan.csv and trans.csv
        """
        self.data_dir = Path(data_dir)
        self.loan_file = self.data_dir / "loan.csv"
        self.trans_file = self.data_dir / "trans.csv"

    def _load_csv(self, filepath: Path) -> pd.DataFrame:
        """
        Load CSV file with Czech Bank format.

        Parameters
        ----------
        filepath : Path
            Path to CSV file

        Returns
        -------
        pd.DataFrame
            Loaded dataframe
        """
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath, sep=';', quotechar='"', low_memory=False)

        # Clean column names
        df.columns = df.columns.str.strip().str.replace('"', '')

        # Clean string columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip().str.replace('"', '')

        return df

    def load_loans(self) -> pd.DataFrame:
        """
        Load loan dataset.

        Returns
        -------
        pd.DataFrame
            Loan data with columns: loan_id, account_id, date, amount,
            duration, payments, status
        """
        logger.info(f"Loading loans from {self.loan_file}")
        df = self._load_csv(self.loan_file)
        logger.info(f"Loaded {len(df):,} loans")
        return df

    def load_transactions(self) -> pd.DataFrame:
        """
        Load transaction dataset.

        Returns
        -------
        pd.DataFrame
            Transaction data with columns: trans_id, account_id, date, type,
            operation, amount, balance, k_symbol, bank, account
        """
        logger.info(f"Loading transactions from {self.trans_file}")
        df = self._load_csv(self.trans_file)
        logger.info(f"Loaded {len(df):,} transactions")
        return df

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both loan and transaction datasets.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (loan_df, trans_df)
        """
        return self.load_loans(), self.load_transactions()

    def print_summary(self, loan_df: pd.DataFrame, trans_df: pd.DataFrame) -> None:
        """
        Print summary statistics of loaded data.

        Parameters
        ----------
        loan_df : pd.DataFrame
            Loan dataframe
        trans_df : pd.DataFrame
            Transaction dataframe
        """
        print("=" * 60)
        print("DATA LOADING SUMMARY")
        print("=" * 60)

        print(f"\nLoan Table:")
        print(f"  Records:       {len(loan_df):>10,}")
        print(f"  Columns:       {len(loan_df.columns):>10}")
        print(f"  Missing:       {loan_df.isnull().sum().sum():>10,}")
        print(f"  Duplicates:    {loan_df.duplicated().sum():>10,}")
        print(f"  Memory:        {loan_df.memory_usage(deep=True).sum()/1024**2:>10.2f} MB")

        print(f"\nTransaction Table:")
        print(f"  Records:       {len(trans_df):>10,}")
        print(f"  Columns:       {len(trans_df.columns):>10}")
        print(f"  Missing:       {trans_df.isnull().sum().sum():>10,}")
        print(f"  Duplicates:    {trans_df.duplicated().sum():>10,}")
        print(f"  Memory:        {trans_df.memory_usage(deep=True).sum()/1024**2:>10.2f} MB")

        if 'status' in loan_df.columns:
            print(f"\nLoan Status Distribution:")
            for status in ['A', 'B', 'C', 'D']:
                count = (loan_df['status'] == status).sum()
                pct = count / len(loan_df) * 100
                marker = "🔴" if status == 'D' else "🟢"
                print(f"  {marker} {status}: {count:>4} ({pct:>5.1f}%)")


def load_czech_bank_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load Czech Bank data.

    Parameters
    ----------
    data_dir : Path
        Directory containing loan.csv and trans.csv

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (loan_df, trans_df)
    """
    loader = CzechBankDataLoader(data_dir)
    return loader.load_all()
