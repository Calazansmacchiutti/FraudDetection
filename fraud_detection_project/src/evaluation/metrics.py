"""
Evaluation Metrics Module

This module provides evaluation metrics for fraud detection models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve
)
import logging

logger = logging.getLogger("fraud_detection.evaluation.metrics")


class MetricsCalculator:
    """
    Calculator for fraud detection metrics.
    
    This class computes various evaluation metrics appropriate
    for imbalanced classification problems.
    """
    
    def __init__(self):
        """Initialize the MetricsCalculator."""
        self.results: Dict[str, Dict[str, Any]] = {}
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate classification metrics.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        y_proba : np.ndarray, optional
            Predicted probabilities
            
        Returns
        -------
        dict
            Dictionary of metrics
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            metrics["average_precision"] = average_precision_score(y_true, y_proba)
        
        return metrics
    
    def calculate_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, int]:
        """
        Calculate confusion matrix components.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
            
        Returns
        -------
        dict
            Confusion matrix components
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        return {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "total_fraud": int(tp + fn),
            "total_normal": int(tn + fp)
        }
    
    def calculate_business_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        amounts: Optional[np.ndarray] = None,
        cost_fp: float = 10.0,
        cost_fn: float = 100.0
    ) -> Dict[str, float]:
        """
        Calculate business-oriented metrics.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        amounts : np.ndarray, optional
            Transaction amounts
        cost_fp : float
            Cost of false positive (investigation cost)
        cost_fn : float
            Cost of false negative (fraud loss)
            
        Returns
        -------
        dict
            Business metrics
        """
        cm = self.calculate_confusion_matrix(y_true, y_pred)
        
        total_fraud = cm["total_fraud"]
        tp = cm["true_positives"]
        fn = cm["false_negatives"]
        fp = cm["false_positives"]
        
        metrics = {
            "fraud_detection_rate": tp / total_fraud if total_fraud > 0 else 0,
            "fraud_miss_rate": fn / total_fraud if total_fraud > 0 else 0,
            "false_alert_count": fp,
            "total_cost": fp * cost_fp + fn * cost_fn
        }
        
        if amounts is not None:
            fraud_mask = y_true == 1
            detected_mask = (y_true == 1) & (y_pred == 1)
            missed_mask = (y_true == 1) & (y_pred == 0)
            
            metrics["total_fraud_amount"] = float(amounts[fraud_mask].sum())
            metrics["detected_fraud_amount"] = float(amounts[detected_mask].sum())
            metrics["missed_fraud_amount"] = float(amounts[missed_mask].sum())
        
        return metrics
    
    def evaluate_model(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        amounts: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Parameters
        ----------
        model_name : str
            Name of the model
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        y_proba : np.ndarray, optional
            Predicted probabilities
        amounts : np.ndarray, optional
            Transaction amounts
            
        Returns
        -------
        dict
            Complete evaluation results
        """
        results = {
            "model_name": model_name,
            "classification_metrics": self.calculate_metrics(y_true, y_pred, y_proba),
            "confusion_matrix": self.calculate_confusion_matrix(y_true, y_pred)
        }
        
        if amounts is not None:
            results["business_metrics"] = self.calculate_business_metrics(
                y_true, y_pred, amounts
            )
        
        self.results[model_name] = results
        
        return results
    
    def compare_models(
        self,
        metrics_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare evaluated models.
        
        Parameters
        ----------
        metrics_list : list, optional
            List of metrics to include
            
        Returns
        -------
        pd.DataFrame
            Comparison dataframe
        """
        if not self.results:
            raise ValueError("No models evaluated. Call evaluate_model first.")
        
        metrics_list = metrics_list or [
            "precision", "recall", "f1", "roc_auc", "average_precision"
        ]
        
        comparison = []
        for model_name, results in self.results.items():
            row = {"model": model_name}
            for metric in metrics_list:
                if metric in results["classification_metrics"]:
                    row[metric] = results["classification_metrics"][metric]
            comparison.append(row)
        
        df = pd.DataFrame(comparison).set_index("model")
        
        return df.sort_values("average_precision", ascending=False)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1"
) -> tuple:
    """
    Find optimal classification threshold.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities
    metric : str
        Metric to optimize ('f1', 'precision', 'recall')
        
    Returns
    -------
    tuple
        (optimal_threshold, optimal_metric_value)
    """
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_score = 0
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


def print_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Print formatted evaluation report.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_proba : np.ndarray, optional
        Predicted probabilities
    model_name : str
        Name of the model
        
    Returns
    -------
    dict
        Evaluation metrics
    """
    calculator = MetricsCalculator()
    metrics = calculator.calculate_metrics(y_true, y_pred, y_proba)
    cm = calculator.calculate_confusion_matrix(y_true, y_pred)
    
    print(f"\n{'='*70}")
    print(f"EVALUATION REPORT: {model_name}")
    print(f"{'='*70}")
    
    print(f"\nClassification Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    
    if y_proba is not None:
        print(f"\nRanking Metrics:")
        print(f"  ROC-AUC:           {metrics['roc_auc']:.4f}")
        print(f"  Average Precision: {metrics['average_precision']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives (TN):  {cm['true_negatives']:,}")
    print(f"  False Positives (FP): {cm['false_positives']:,}")
    print(f"  False Negatives (FN): {cm['false_negatives']:,}")
    print(f"  True Positives (TP):  {cm['true_positives']:,}")
    
    total_fraud = cm['total_fraud']
    tp = cm['true_positives']
    fn = cm['false_negatives']
    
    print(f"\nBusiness Impact:")
    print(f"  Total frauds in test set:  {total_fraud:,}")
    print(f"  Frauds detected:           {tp:,} ({tp/total_fraud*100:.1f}%)")
    print(f"  Frauds missed:             {fn:,} ({fn/total_fraud*100:.1f}%)")
    print(f"  False alerts:              {cm['false_positives']:,}")
    
    print(f"{'='*70}\n")
    
    return metrics
