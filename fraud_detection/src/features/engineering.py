"""
Módulo para engenharia de features e pré-processamento.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from typing import Tuple, Optional, Literal
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Classe para engenharia de features e transformações."""
    
    def __init__(self, scaler_type: Literal["standard", "robust"] = "robust"):
        """
        Inicializa o engenheiro de features.
        
        Args:
            scaler_type: Tipo de scaler a usar ('standard' ou 'robust')
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self.feature_names = None
        
    def fit_scaler(self, X: pd.DataFrame, columns: Optional[list] = None) -> 'FeatureEngineer':
        """
        Ajusta o scaler nos dados.
        
        Args:
            X: DataFrame com features
            columns: Colunas para aplicar scaling (None = todas numéricas)
            
        Returns:
            self
        """
        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = RobustScaler()
        
        if columns is None:
            columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        self.scale_columns = columns
        self.scaler.fit(X[columns])
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica transformação de scaling.
        
        Args:
            X: DataFrame com features
            
        Returns:
            DataFrame transformado
        """
        X_transformed = X.copy()
        
        if self.scaler is not None and hasattr(self, 'scale_columns'):
            X_transformed[self.scale_columns] = self.scaler.transform(X[self.scale_columns])
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, columns: Optional[list] = None) -> pd.DataFrame:
        """Ajusta e transforma em uma única operação."""
        self.fit_scaler(X, columns)
        return self.transform(X)


def prepare_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    validation_size: Optional[float] = None,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple:
    """
    Divide os dados em conjuntos de treino, validação e teste.
    
    Args:
        X: Features
        y: Target
        test_size: Proporção do conjunto de teste
        validation_size: Proporção do conjunto de validação (do treino)
        random_state: Seed para reprodutibilidade
        stratify: Se deve estratificar pela classe alvo
        
    Returns:
        Se validation_size is None: (X_train, X_test, y_train, y_test)
        Se validation_size is not None: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    strat = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    
    if validation_size is not None:
        strat_val = y_train if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_size, 
            random_state=random_state, stratify=strat_val
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    return X_train, X_test, y_train, y_test


def handle_imbalance(
    X: pd.DataFrame,
    y: pd.Series,
    method: Literal["smote", "undersample", "smote_tomek", "none"] = "smote",
    random_state: int = 42,
    sampling_strategy: float = 0.5
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Aplica técnicas de balanceamento de classes.
    
    Args:
        X: Features
        y: Target
        method: Método de balanceamento
        random_state: Seed para reprodutibilidade
        sampling_strategy: Razão desejada entre minoritária/majoritária
        
    Returns:
        Tuple (X_resampled, y_resampled)
    """
    if method == "none":
        return X, y
    
    print(f"\nAplicando balanceamento: {method}")
    print(f"Distribuição original: {dict(y.value_counts())}")
    
    if method == "smote":
        sampler = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            n_jobs=-1
        )
    elif method == "undersample":
        sampler = RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=random_state
        )
    elif method == "smote_tomek":
        sampler = SMOTETomek(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Método desconhecido: {method}")
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    # Converter de volta para DataFrame se necessário
    if isinstance(X, pd.DataFrame):
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    
    y_resampled = pd.Series(y_resampled, name=y.name)
    
    print(f"Distribuição após balanceamento: {dict(y_resampled.value_counts())}")
    
    return X_resampled, y_resampled


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features de interação entre variáveis.
    
    Args:
        df: DataFrame com features
        
    Returns:
        DataFrame com features de interação adicionadas
    """
    df = df.copy()
    
    # Interações entre features PCA mais importantes (baseado em literatura)
    if 'V1' in df.columns and 'V2' in df.columns:
        df['V1_V2'] = df['V1'] * df['V2']
    
    if 'V3' in df.columns and 'V4' in df.columns:
        df['V3_V4'] = df['V3'] * df['V4']
    
    # Magnitude das principais componentes
    pca_cols = [c for c in df.columns if c.startswith('V')][:10]
    if pca_cols:
        df['PCA_magnitude'] = np.sqrt((df[pca_cols] ** 2).sum(axis=1))
    
    return df


def select_features_by_importance(
    X: pd.DataFrame,
    importance_scores: dict,
    threshold: float = 0.01
) -> list:
    """
    Seleciona features com importância acima do threshold.
    
    Args:
        X: DataFrame com features
        importance_scores: Dicionário {feature: importance}
        threshold: Limite mínimo de importância
        
    Returns:
        Lista de features selecionadas
    """
    selected = [
        feat for feat, imp in importance_scores.items()
        if imp >= threshold and feat in X.columns
    ]
    
    return selected


def get_feature_statistics(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Calcula estatísticas das features por classe.
    
    Args:
        X: Features
        y: Target
        
    Returns:
        DataFrame com estatísticas comparativas
    """
    stats = []
    
    for col in X.columns:
        fraud_vals = X.loc[y == 1, col]
        normal_vals = X.loc[y == 0, col]
        
        stats.append({
            'feature': col,
            'mean_normal': normal_vals.mean(),
            'mean_fraud': fraud_vals.mean(),
            'std_normal': normal_vals.std(),
            'std_fraud': fraud_vals.std(),
            'mean_diff': abs(fraud_vals.mean() - normal_vals.mean()),
            'mean_diff_ratio': abs(fraud_vals.mean() - normal_vals.mean()) / (normal_vals.std() + 1e-8)
        })
    
    stats_df = pd.DataFrame(stats)
    stats_df = stats_df.sort_values('mean_diff_ratio', ascending=False)
    
    return stats_df
