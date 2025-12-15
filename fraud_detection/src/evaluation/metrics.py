"""
Módulo para avaliação de modelos de detecção de fraude.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score, precision_recall_curve, roc_curve,
    f1_score, precision_score, recall_score, accuracy_score
)
from typing import Dict, Any, Optional, List, Tuple
import warnings

warnings.filterwarnings('ignore')


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calcula métricas de classificação.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predições binárias
        y_proba: Probabilidades preditas (para métricas de ranking)
        
    Returns:
        Dicionário com métricas
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        metrics['average_precision'] = average_precision_score(y_true, y_proba)
    
    return metrics


def get_confusion_matrix_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Any]:
    """
    Analisa a matriz de confusão detalhadamente.
    
    Returns:
        Dicionário com análise da matriz de confusão
    """
    cm = confusion_matrix(y_true, y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    
    analysis = {
        'confusion_matrix': cm,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
    }
    
    return analysis


def print_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Imprime relatório completo de avaliação.
    
    Returns:
        Dicionário com todas as métricas
    """
    print(f"\n{'='*60}")
    print(f"RELATÓRIO DE AVALIAÇÃO: {model_name}")
    print(f"{'='*60}")
    
    # Métricas principais
    metrics = calculate_metrics(y_true, y_pred, y_proba)
    
    print(f"\nMÉTRICAS DE CLASSIFICAÇÃO:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    
    if y_proba is not None:
        print(f"\nMÉTRICAS DE RANKING:")
        print(f"  ROC-AUC:            {metrics['roc_auc']:.4f}")
        print(f"  Average Precision:  {metrics['average_precision']:.4f}")
    
    # Análise de matriz de confusão
    cm_analysis = get_confusion_matrix_analysis(y_true, y_pred)
    
    print(f"\nMATRIZ DE CONFUSÃO:")
    print(f"  True Negatives (TN):  {cm_analysis['true_negatives']:,}")
    print(f"  False Positives (FP): {cm_analysis['false_positives']:,}")
    print(f"  False Negatives (FN): {cm_analysis['false_negatives']:,}")
    print(f"  True Positives (TP):  {cm_analysis['true_positives']:,}")
    
    print(f"\nTAXAS DE ERRO:")
    print(f"  Taxa de Falsos Positivos: {cm_analysis['false_positive_rate']:.4f}")
    print(f"  Taxa de Falsos Negativos: {cm_analysis['false_negative_rate']:.4f}")
    print(f"  Especificidade:           {cm_analysis['specificity']:.4f}")
    
    # Interpretação para negócio
    total_fraud = cm_analysis['true_positives'] + cm_analysis['false_negatives']
    detected_fraud = cm_analysis['true_positives']
    missed_fraud = cm_analysis['false_negatives']
    false_alerts = cm_analysis['false_positives']
    
    print(f"\nIMPACTO NO NEGÓCIO:")
    print(f"  Fraudes no teste:     {total_fraud:,}")
    print(f"  Fraudes detectadas:   {detected_fraud:,} ({detected_fraud/total_fraud*100:.1f}%)")
    print(f"  Fraudes perdidas:     {missed_fraud:,} ({missed_fraud/total_fraud*100:.1f}%)")
    print(f"  Falsos alertas:       {false_alerts:,}")
    
    print(f"{'='*60}\n")
    
    return {**metrics, **cm_analysis}


def compare_models(
    models_results: Dict[str, Dict[str, float]],
    metrics_to_compare: List[str] = None
) -> pd.DataFrame:
    """
    Compara múltiplos modelos.
    
    Args:
        models_results: Dicionário {model_name: metrics_dict}
        metrics_to_compare: Lista de métricas para comparar
        
    Returns:
        DataFrame com comparação
    """
    if metrics_to_compare is None:
        metrics_to_compare = ['precision', 'recall', 'f1', 'roc_auc', 'average_precision']
    
    comparison = []
    
    for model_name, metrics in models_results.items():
        row = {'model': model_name}
        for metric in metrics_to_compare:
            if metric in metrics:
                row[metric] = metrics[metric]
        comparison.append(row)
    
    df = pd.DataFrame(comparison)
    df = df.set_index('model')
    
    return df


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Encontra o threshold ótimo para uma métrica específica.
    
    Args:
        y_true: Labels verdadeiros
        y_proba: Probabilidades preditas
        metric: Métrica para otimizar ('f1', 'precision', 'recall')
        
    Returns:
        Tuple (threshold_ótimo, score_ótimo)
    """
    thresholds = np.arange(0.1, 0.95, 0.05)
    best_threshold = 0.5
    best_score = 0
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold, best_score


def calculate_business_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    amounts: np.ndarray,
    cost_per_false_positive: float = 10,
    cost_per_false_negative_rate: float = 1.0
) -> Dict[str, float]:
    """
    Calcula métricas de impacto no negócio.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predições
        amounts: Valores das transações
        cost_per_false_positive: Custo operacional de investigar falso positivo
        cost_per_false_negative_rate: Percentual do valor perdido em fraude não detectada
        
    Returns:
        Dicionário com métricas de negócio
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Índices das fraudes não detectadas
    false_negatives_mask = (y_true == 1) & (y_pred == 0)
    missed_fraud_amount = amounts[false_negatives_mask].sum()
    
    # Índices das fraudes detectadas
    true_positives_mask = (y_true == 1) & (y_pred == 1)
    detected_fraud_amount = amounts[true_positives_mask].sum()
    
    # Custos
    cost_false_positives = fp * cost_per_false_positive
    cost_false_negatives = missed_fraud_amount * cost_per_false_negative_rate
    total_cost = cost_false_positives + cost_false_negatives
    
    # Valor total de fraudes
    total_fraud_amount = amounts[y_true == 1].sum()
    
    return {
        'detected_fraud_amount': detected_fraud_amount,
        'missed_fraud_amount': missed_fraud_amount,
        'total_fraud_amount': total_fraud_amount,
        'detection_rate_by_value': detected_fraud_amount / total_fraud_amount if total_fraud_amount > 0 else 0,
        'cost_false_positives': cost_false_positives,
        'cost_false_negatives': cost_false_negatives,
        'total_cost': total_cost,
        'n_false_positives': int(fp),
        'n_false_negatives': int(fn)
    }


class ModelEvaluator:
    """Classe para avaliação completa de modelos."""
    
    def __init__(self, y_true: np.ndarray, amounts: Optional[np.ndarray] = None):
        """
        Inicializa o avaliador.
        
        Args:
            y_true: Labels verdadeiros
            amounts: Valores das transações (opcional)
        """
        self.y_true = y_true
        self.amounts = amounts
        self.results = {}
    
    def evaluate_model(
        self,
        model_name: str,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Avalia um modelo e armazena resultados.
        
        Returns:
            Dicionário com métricas
        """
        result = print_evaluation_report(
            self.y_true, y_pred, y_proba, model_name
        )
        
        if self.amounts is not None:
            business = calculate_business_metrics(
                self.y_true, y_pred, self.amounts
            )
            result.update(business)
            
            print(f"MÉTRICAS DE NEGÓCIO ({model_name}):")
            print(f"  Valor de fraudes detectadas: ${business['detected_fraud_amount']:,.2f}")
            print(f"  Valor de fraudes perdidas:   ${business['missed_fraud_amount']:,.2f}")
            print(f"  Taxa de detecção por valor:  {business['detection_rate_by_value']:.2%}")
        
        self.results[model_name] = result
        return result
    
    def get_comparison_table(self) -> pd.DataFrame:
        """Retorna tabela comparativa de todos os modelos avaliados."""
        return compare_models(self.results)
    
    def get_best_model(self, metric: str = 'f1') -> str:
        """Retorna o nome do melhor modelo para a métrica especificada."""
        if not self.results:
            return None
        
        best_model = max(
            self.results.items(),
            key=lambda x: x[1].get(metric, 0)
        )
        
        return best_model[0]
