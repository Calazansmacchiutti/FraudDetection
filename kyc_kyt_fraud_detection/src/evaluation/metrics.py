"""
Metrics Module

Evaluation metrics for loan default prediction models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
import logging

logger = logging.getLogger("kyc_kyt.evaluation.metrics")


class MetricsCalculator:
    """
    Calculate and track evaluation metrics for default prediction models.
    """

    def __init__(self):
        """Initialize metrics calculator."""
        self.model_results = {}

    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        model_name: str = "Model"
    ) -> Dict:
        """
        Evaluate model performance.

        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        y_proba : np.ndarray
            Predicted probabilities
        model_name : str
            Name of the model

        Returns
        -------
        Dict
            Evaluation metrics
        """
        metrics = {
            'Model': model_name,
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1': f1_score(y_true, y_pred, zero_division=0),
            'ROC_AUC': roc_auc_score(y_true, y_proba),
            'Avg_Precision': average_precision_score(y_true, y_proba)
        }

        self.model_results[model_name] = metrics

        return metrics

    def compare_models(self) -> pd.DataFrame:
        """
        Compare all evaluated models.

        Returns
        -------
        pd.DataFrame
            Comparison table sorted by Average Precision
        """
        if not self.model_results:
            return pd.DataFrame()

        df = pd.DataFrame(list(self.model_results.values()))
        df = df.sort_values('Avg_Precision', ascending=False)

        return df

    def get_best_model(self, metric: str = 'Avg_Precision') -> str:
        """
        Get name of best model by metric.

        Parameters
        ----------
        metric : str
            Metric to use for comparison

        Returns
        -------
        str
            Best model name
        """
        if not self.model_results:
            return None

        best_model = max(
            self.model_results.items(),
            key=lambda x: x[1][metric]
        )

        return best_model[0]


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model"
) -> Dict:
    """
    Convenience function to evaluate a model.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_proba : np.ndarray
        Predicted probabilities
    model_name : str
        Name of the model

    Returns
    -------
    Dict
        Evaluation metrics
    """
    calculator = MetricsCalculator()
    return calculator.evaluate_model(y_true, y_pred, y_proba, model_name)


def print_metrics(metrics: Dict) -> None:
    """
    Print metrics in a formatted way.

    Parameters
    ----------
    metrics : Dict
        Metrics dictionary
    """
    print(f"\n📊 {metrics['Model']}:")
    print(f"   Precision:     {metrics['Precision']:.4f}")
    print(f"   Recall:        {metrics['Recall']:.4f}")
    print(f"   F1:            {metrics['F1']:.4f}")
    print(f"   ROC-AUC:       {metrics['ROC_AUC']:.4f}")
    print(f"   Avg Precision: {metrics['Avg_Precision']:.4f}")


def print_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model"
) -> None:
    """
    Print comprehensive evaluation report.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_proba : np.ndarray
        Predicted probabilities
    model_name : str
        Name of the model
    """
    print("=" * 60)
    print(f"EVALUATION REPORT: {model_name}")
    print("=" * 60)

    # Metrics
    metrics = evaluate_model(y_true, y_pred, y_proba, model_name)
    print_metrics(metrics)

    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"   TN: {cm[0,0]:>4}  |  FP: {cm[0,1]:>4}")
    print(f"   FN: {cm[1,0]:>4}  |  TP: {cm[1,1]:>4}")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Good', 'Default']))

    print("=" * 60)
