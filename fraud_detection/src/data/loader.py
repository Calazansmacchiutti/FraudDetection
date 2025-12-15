"""
Módulo para carregamento e preparação inicial dos dados.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


def load_credit_card_data(filepath: str | Path) -> pd.DataFrame:
    """
    Carrega o dataset de fraude de cartão de crédito.
    
    Args:
        filepath: Caminho para o arquivo CSV
        
    Returns:
        DataFrame com os dados carregados
    """
    df = pd.read_csv(filepath)
    print(f"Dataset carregado: {df.shape[0]:,} transações, {df.shape[1]} colunas")
    return df


def get_data_summary(df: pd.DataFrame, target_col: str = "Class") -> dict:
    """
    Retorna resumo estatístico do dataset.
    
    Args:
        df: DataFrame com os dados
        target_col: Nome da coluna alvo
        
    Returns:
        Dicionário com estatísticas do dataset
    """
    n_fraud = df[target_col].sum()
    n_normal = len(df) - n_fraud
    
    summary = {
        "total_transactions": len(df),
        "n_features": df.shape[1] - 1,
        "n_fraud": int(n_fraud),
        "n_normal": int(n_normal),
        "fraud_percentage": round(n_fraud / len(df) * 100, 4),
        "imbalance_ratio": round(n_normal / n_fraud, 2),
        "missing_values": df.isnull().sum().sum(),
        "duplicates": df.duplicated().sum()
    }
    
    return summary


def print_data_summary(summary: dict) -> None:
    """Imprime o resumo do dataset de forma formatada."""
    print("\n" + "="*60)
    print("RESUMO DO DATASET")
    print("="*60)
    print(f"Total de transações: {summary['total_transactions']:,}")
    print(f"Número de features: {summary['n_features']}")
    print(f"\nDistribuição das classes:")
    print(f"  - Transações normais: {summary['n_normal']:,} ({100-summary['fraud_percentage']:.2f}%)")
    print(f"  - Transações fraudulentas: {summary['n_fraud']:,} ({summary['fraud_percentage']:.4f}%)")
    print(f"  - Razão de desbalanceamento: {summary['imbalance_ratio']}:1")
    print(f"\nQualidade dos dados:")
    print(f"  - Valores faltantes: {summary['missing_values']}")
    print(f"  - Registros duplicados: {summary['duplicates']}")
    print("="*60 + "\n")


def split_features_target(
    df: pd.DataFrame,
    target_col: str = "Class",
    drop_cols: Optional[list] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa features e variável alvo.
    
    Args:
        df: DataFrame com os dados
        target_col: Nome da coluna alvo
        drop_cols: Colunas adicionais para remover
        
    Returns:
        Tuple (X, y) com features e target
    """
    cols_to_drop = [target_col]
    if drop_cols:
        cols_to_drop.extend(drop_cols)
    
    X = df.drop(columns=cols_to_drop, errors='ignore')
    y = df[target_col]
    
    return X, y


def create_time_features(df: pd.DataFrame, time_col: str = "Time") -> pd.DataFrame:
    """
    Cria features derivadas da coluna Time.
    
    O dataset original tem Time em segundos desde a primeira transação.
    Criamos features cíclicas para capturar padrões temporais.
    
    Args:
        df: DataFrame com os dados
        time_col: Nome da coluna de tempo
        
    Returns:
        DataFrame com novas features temporais
    """
    df = df.copy()
    
    # Converter para horas (assumindo 2 dias de dados)
    df['Hour'] = (df[time_col] / 3600) % 24
    
    # Features cíclicas para capturar padrões circulares
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    
    # Período do dia
    df['Is_Night'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)
    
    return df


def create_amount_features(df: pd.DataFrame, amount_col: str = "Amount") -> pd.DataFrame:
    """
    Cria features derivadas do valor da transação.
    
    Args:
        df: DataFrame com os dados
        amount_col: Nome da coluna de valor
        
    Returns:
        DataFrame com novas features de valor
    """
    df = df.copy()
    
    # Log do valor (para normalizar distribuição)
    df['Amount_log'] = np.log1p(df[amount_col])
    
    # Categorias de valor
    df['Amount_category'] = pd.cut(
        df[amount_col],
        bins=[0, 10, 50, 100, 500, np.inf],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)
    
    # Indicador de valor alto (acima do percentil 95)
    p95 = df[amount_col].quantile(0.95)
    df['Is_High_Amount'] = (df[amount_col] > p95).astype(int)
    
    return df
