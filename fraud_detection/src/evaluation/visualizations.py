"""
Módulo para visualizações de análise e avaliação.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve, auc
)
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# Configuração global de estilo
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_class_distribution(
    y: pd.Series,
    title: str = "Distribuição das Classes",
    figsize: Tuple[int, int] = (10, 5)
) -> plt.Figure:
    """
    Plota a distribuição das classes.
    
    Args:
        y: Série com labels
        title: Título do gráfico
        figsize: Tamanho da figura
        
    Returns:
        Figura matplotlib
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Contagem
    counts = y.value_counts()
    colors = ['#2ecc71', '#e74c3c']
    labels = ['Normal', 'Fraude']
    
    # Gráfico de barras
    axes[0].bar(labels, counts.values, color=colors, edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('Quantidade')
    axes[0].set_title('Contagem por Classe')
    
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')
    
    # Gráfico de pizza (escala log para visualização)
    axes[1].pie(
        counts.values,
        labels=[f'{l}\n({v:,})' for l, v in zip(labels, counts.values)],
        colors=colors,
        autopct='%1.3f%%',
        explode=(0, 0.1),
        shadow=True
    )
    axes[1].set_title('Proporção das Classes')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_feature_distributions(
    df: pd.DataFrame,
    features: List[str],
    target_col: str = 'Class',
    n_cols: int = 4,
    figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    Plota distribuições de features por classe.
    
    Args:
        df: DataFrame com dados
        features: Lista de features para plotar
        target_col: Coluna alvo
        n_cols: Número de colunas no grid
        figsize: Tamanho da figura
        
    Returns:
        Figura matplotlib
    """
    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        # Histograma por classe
        for label, color in zip([0, 1], ['#2ecc71', '#e74c3c']):
            data = df[df[target_col] == label][feature]
            ax.hist(data, bins=50, alpha=0.6, color=color, 
                   label='Normal' if label == 0 else 'Fraude', density=True)
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Densidade')
        ax.legend(fontsize=8)
    
    # Remover eixos vazios
    for idx in range(len(features), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('Distribuição das Features por Classe', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Plota matriz de correlação.
    
    Args:
        df: DataFrame com dados
        features: Lista de features (None para todas numéricas)
        figsize: Tamanho da figura
        
    Returns:
        Figura matplotlib
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    corr_matrix = df[features].corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap='RdBu_r',
        center=0,
        annot=False,
        square=True,
        linewidths=0.5,
        ax=ax
    )
    
    ax.set_title('Matriz de Correlação', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Matriz de Confusão",
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plota matriz de confusão.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predições
        title: Título do gráfico
        figsize: Tamanho da figura
        
    Returns:
        Figura matplotlib
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Normal', 'Fraude'],
        yticklabels=['Normal', 'Fraude'],
        ax=ax
    )
    
    ax.set_xlabel('Predição')
    ax.set_ylabel('Real')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Adicionar porcentagens
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = cm[i, j] / total * 100
            current_text = ax.texts[i * 2 + j]
            current_text.set_text(f'{cm[i, j]:,}\n({percentage:.2f}%)')
    
    plt.tight_layout()
    
    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    models_proba: Dict[str, np.ndarray],
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plota curvas ROC para múltiplos modelos.
    
    Args:
        y_true: Labels verdadeiros
        models_proba: Dicionário {nome_modelo: probabilidades}
        figsize: Tamanho da figura
        
    Returns:
        Figura matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(models_proba)))
    
    for (model_name, y_proba), color in zip(models_proba.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=color, lw=2,
               label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    # Linha de referência
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.5)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taxa de Falsos Positivos', fontsize=12)
    ax.set_ylabel('Taxa de Verdadeiros Positivos', fontsize=12)
    ax.set_title('Curvas ROC - Comparação de Modelos', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def plot_precision_recall_curves(
    y_true: np.ndarray,
    models_proba: Dict[str, np.ndarray],
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plota curvas Precision-Recall para múltiplos modelos.
    
    Args:
        y_true: Labels verdadeiros
        models_proba: Dicionário {nome_modelo: probabilidades}
        figsize: Tamanho da figura
        
    Returns:
        Figura matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(models_proba)))
    
    for (model_name, y_proba), color in zip(models_proba.items(), colors):
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        
        ax.plot(recall, precision, color=color, lw=2,
               label=f'{model_name} (AP = {pr_auc:.4f})')
    
    # Linha de referência (proporção da classe positiva)
    baseline = y_true.mean()
    ax.axhline(y=baseline, color='k', linestyle='--', lw=1, 
               label=f'Baseline ({baseline:.4f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Curvas Precision-Recall - Comparação de Modelos', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def plot_feature_importance(
    importance_dict: Dict[str, float],
    top_n: int = 20,
    title: str = "Importância das Features",
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plota importância das features.
    
    Args:
        importance_dict: Dicionário {feature: importância}
        top_n: Número de features mais importantes
        title: Título do gráfico
        figsize: Tamanho da figura
        
    Returns:
        Figura matplotlib
    """
    # Ordenar e selecionar top N
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, importances = zip(*sorted_items)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(features)))[::-1]
    
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importances, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Importância')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Adicionar valores
    for i, v in enumerate(importances):
        ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plota comparação de modelos em gráfico de barras.
    
    Args:
        comparison_df: DataFrame com métricas por modelo
        metrics: Lista de métricas para plotar
        figsize: Tamanho da figura
        
    Returns:
        Figura matplotlib
    """
    if metrics is None:
        metrics = comparison_df.columns.tolist()
    
    df_plot = comparison_df[metrics].copy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(df_plot.index))
    width = 0.8 / len(metrics)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = width * (i - len(metrics) / 2 + 0.5)
        bars = ax.bar(x + offset, df_plot[metric], width, label=metric, color=color)
        
        # Adicionar valores
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, rotation=45)
    
    ax.set_ylabel('Score')
    ax.set_title('Comparação de Modelos', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot.index, rotation=45, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    return fig


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plota análise de threshold.
    
    Args:
        y_true: Labels verdadeiros
        y_proba: Probabilidades preditas
        figsize: Tamanho da figura
        
    Returns:
        Figura matplotlib
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    thresholds = np.arange(0.1, 0.95, 0.05)
    
    precisions = []
    recalls = []
    f1s = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Métricas vs Threshold
    axes[0].plot(thresholds, precisions, 'b-', label='Precision', lw=2)
    axes[0].plot(thresholds, recalls, 'r-', label='Recall', lw=2)
    axes[0].plot(thresholds, f1s, 'g-', label='F1-Score', lw=2)
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Métricas vs Threshold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Precision vs Recall
    axes[1].plot(recalls, precisions, 'purple', lw=2)
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Trade-off Precision-Recall')
    axes[1].grid(True, alpha=0.3)
    
    # Marcar threshold 0.5
    idx_05 = int(len(thresholds) * 0.4)  # Aproximadamente threshold 0.5
    axes[1].scatter([recalls[idx_05]], [precisions[idx_05]], 
                   color='red', s=100, zorder=5, label='Threshold=0.5')
    axes[1].legend()
    
    fig.suptitle('Análise de Threshold', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def save_all_figures(figures: Dict[str, plt.Figure], output_dir: str) -> None:
    """
    Salva todas as figuras em um diretório.
    
    Args:
        figures: Dicionário {nome: figura}
        output_dir: Diretório de saída
    """
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for name, fig in figures.items():
        filepath = output_path / f"{name}.png"
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Figura salva: {filepath}")
