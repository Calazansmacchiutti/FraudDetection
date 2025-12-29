"""
Statistical Tests Module

Statistical tests for feature discrimination power (Cohen's d, KS test).
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from scipy.stats import ks_2samp
import logging

logger = logging.getLogger("kyc_kyt.evaluation.statistical_tests")


def calculate_cohens_d(
    good_values: np.ndarray,
    bad_values: np.ndarray
) -> float:
    """
    Calculate Cohen's d effect size.

    Parameters
    ----------
    good_values : np.ndarray
        Values for good loans
    bad_values : np.ndarray
        Values for defaults

    Returns
    -------
    float
        Cohen's d (effect size)
    """
    pooled_std = np.sqrt(
        (good_values.std()**2 + bad_values.std()**2) / 2
    )

    cohens_d = abs(bad_values.mean() - good_values.mean()) / (pooled_std + 1e-8)

    return cohens_d


def calculate_discrimination(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate discrimination power for each feature.

    Uses Cohen's d and Kolmogorov-Smirnov test.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and target
    target_col : str
        Target column name
    feature_cols : List[str], optional
        Feature columns to test (defaults to all numeric)

    Returns
    -------
    pd.DataFrame
        Discrimination statistics sorted by Cohen's d
    """
    logger.info("Calculating feature discrimination...")

    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in feature_cols if c != target_col]

    results = []

    for feat in feature_cols:
        if feat not in df.columns:
            continue

        good = df[df[target_col] == 0][feat].dropna()
        bad = df[df[target_col] == 1][feat].dropna()

        # Skip if insufficient data
        if len(good) < 10 or len(bad) < 5:
            continue

        # Cohen's d (effect size)
        cohens_d = calculate_cohens_d(good.values, bad.values)

        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = ks_2samp(good, bad)

        # Mean difference
        mean_diff = bad.mean() - good.mean()
        mean_diff_pct = (mean_diff / (good.mean() + 1e-8)) * 100

        results.append({
            'Feature': feat,
            'Cohens_d': cohens_d,
            'KS_Stat': ks_stat,
            'KS_pval': ks_pval,
            'Mean_Good': good.mean(),
            'Mean_Bad': bad.mean(),
            'Diff': mean_diff,
            'Diff_Pct': mean_diff_pct
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Cohens_d', ascending=False)

    logger.info(f"Calculated discrimination for {len(results_df)} features")

    return results_df


def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d effect size.

    Parameters
    ----------
    d : float
        Cohen's d value

    Returns
    -------
    str
        Interpretation
    """
    if d >= 0.8:
        return "Large"
    elif d >= 0.5:
        return "Medium"
    elif d >= 0.2:
        return "Small"
    else:
        return "Negligible"


def get_top_discriminative_features(
    df: pd.DataFrame,
    target_col: str,
    top_n: int = 15,
    min_cohens_d: float = 0.2
) -> pd.DataFrame:
    """
    Get top discriminative features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and target
    target_col : str
        Target column name
    top_n : int
        Number of top features to return
    min_cohens_d : float
        Minimum Cohen's d threshold

    Returns
    -------
    pd.DataFrame
        Top discriminative features
    """
    discrimination = calculate_discrimination(df, target_col)
    discrimination = discrimination[discrimination['Cohens_d'] >= min_cohens_d]
    top_features = discrimination.head(top_n)

    # Add interpretation
    top_features['Effect_Size'] = top_features['Cohens_d'].apply(interpret_cohens_d)

    return top_features


def print_discrimination_report(
    df: pd.DataFrame,
    target_col: str,
    top_n: int = 15
) -> None:
    """
    Print discrimination report.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and target
    target_col : str
        Target column name
    top_n : int
        Number of top features to show
    """
    print("=" * 80)
    print("FEATURE DISCRIMINATION ANALYSIS")
    print("=" * 80)

    discrimination = calculate_discrimination(df, target_col)
    top_features = discrimination.head(top_n)

    print(f"\nTop {top_n} Most Discriminative Features:")
    print("-" * 80)
    print(f"{'Feature':<25} {'Cohen\\'s d':<10} {'Effect':<12} {'KS Stat':<10} {'p-value':<10}")
    print("-" * 80)

    for _, row in top_features.iterrows():
        effect = interpret_cohens_d(row['Cohens_d'])
        print(
            f"{row['Feature']:<25} "
            f"{row['Cohens_d']:<10.4f} "
            f"{effect:<12} "
            f"{row['KS_Stat']:<10.4f} "
            f"{row['KS_pval']:<10.4e}"
        )

    print("-" * 80)
    print("\nEffect Size Interpretation:")
    print("  • Large (d ≥ 0.8):      Strong discriminator")
    print("  • Medium (0.5 ≤ d < 0.8): Moderate discriminator")
    print("  • Small (0.2 ≤ d < 0.5):  Weak discriminator")
    print("  • Negligible (d < 0.2):   Very weak")
    print("=" * 80)
