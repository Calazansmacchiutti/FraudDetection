"""
Módulo para modelos de classificação supervisionada.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from typing import Dict, Any, Optional, List, Tuple
import joblib
from pathlib import Path
import time
import warnings

warnings.filterwarnings('ignore')


class FraudClassifier:
    """Wrapper para modelos de classificação de fraude."""
    
    AVAILABLE_MODELS = {
        'logistic_regression': LogisticRegression,
        'random_forest': RandomForestClassifier,
        'gradient_boosting': GradientBoostingClassifier,
        'xgboost': XGBClassifier,
        'lightgbm': LGBMClassifier
    }
    
    def __init__(
        self,
        model_type: str,
        params: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None
    ):
        """
        Inicializa o classificador.
        
        Args:
            model_type: Tipo do modelo (ver AVAILABLE_MODELS)
            params: Parâmetros do modelo
            name: Nome customizado para identificação
        """
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(f"Modelo '{model_type}' não disponível. Opções: {list(self.AVAILABLE_MODELS.keys())}")
        
        self.model_type = model_type
        self.params = params or {}
        self.name = name or model_type
        self.model = self.AVAILABLE_MODELS[model_type](**self.params)
        self.is_fitted = False
        self.training_time = None
        self.feature_names = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FraudClassifier':
        """
        Treina o modelo.
        
        Args:
            X: Features de treino
            y: Target de treino
            
        Returns:
            self
        """
        self.feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else None
        
        start_time = time.time()
        self.model.fit(X, y)
        self.training_time = time.time() - start_time
        
        self.is_fitted = True
        print(f"Modelo {self.name} treinado em {self.training_time:.2f}s")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Retorna predições binárias."""
        if not self.is_fitted:
            raise RuntimeError("Modelo não treinado. Execute fit() primeiro.")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Retorna probabilidades de cada classe."""
        if not self.is_fitted:
            raise RuntimeError("Modelo não treinado. Execute fit() primeiro.")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Retorna importância das features.
        
        Returns:
            Dicionário {feature: importance} ou None se não disponível
        """
        if not self.is_fitted:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            return None
        
        if self.feature_names:
            return dict(zip(self.feature_names, importances))
        return dict(enumerate(importances))
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: str = 'roc_auc'
    ) -> Dict[str, float]:
        """
        Realiza validação cruzada.
        
        Args:
            X: Features
            y: Target
            cv: Número de folds
            scoring: Métrica de avaliação
            
        Returns:
            Dicionário com média e desvio padrão dos scores
        """
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        return {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores.tolist()
        }
    
    def save(self, filepath: str | Path) -> None:
        """Salva o modelo em disco."""
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'params': self.params,
            'name': self.name,
            'feature_names': self.feature_names,
            'training_time': self.training_time
        }, filepath)
        print(f"Modelo salvo em: {filepath}")
    
    @classmethod
    def load(cls, filepath: str | Path) -> 'FraudClassifier':
        """Carrega um modelo salvo."""
        data = joblib.load(filepath)
        
        instance = cls(
            model_type=data['model_type'],
            params=data['params'],
            name=data['name']
        )
        instance.model = data['model']
        instance.feature_names = data['feature_names']
        instance.training_time = data['training_time']
        instance.is_fitted = True
        
        return instance


def train_multiple_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_configs: Dict[str, Dict[str, Any]]
) -> Dict[str, FraudClassifier]:
    """
    Treina múltiplos modelos.
    
    Args:
        X_train: Features de treino
        y_train: Target de treino
        model_configs: Configurações dos modelos {nome: {type, params}}
        
    Returns:
        Dicionário com modelos treinados
    """
    models = {}
    
    for name, config in model_configs.items():
        print(f"\n{'='*50}")
        print(f"Treinando: {name}")
        print(f"{'='*50}")
        
        clf = FraudClassifier(
            model_type=config['type'],
            params=config.get('params', {}),
            name=name
        )
        clf.fit(X_train, y_train)
        models[name] = clf
    
    return models


def hyperparameter_tuning(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict[str, List],
    cv: int = 3,
    scoring: str = 'average_precision',
    n_jobs: int = -1
) -> Tuple[Dict[str, Any], float]:
    """
    Realiza busca de hiperparâmetros.
    
    Args:
        model_type: Tipo do modelo
        X_train: Features de treino
        y_train: Target de treino
        param_grid: Grid de parâmetros
        cv: Número de folds
        scoring: Métrica para otimização
        n_jobs: Número de jobs paralelos
        
    Returns:
        Tuple (melhores_params, melhor_score)
    """
    base_model = FraudClassifier.AVAILABLE_MODELS[model_type]()
    
    print(f"\nIniciando GridSearch para {model_type}")
    print(f"Parâmetros: {param_grid}")
    print(f"Scoring: {scoring}, CV: {cv}")
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1,
        refit=True
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nMelhores parâmetros: {grid_search.best_params_}")
    print(f"Melhor score ({scoring}): {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_, grid_search.best_score_


def get_model_summary(models: Dict[str, FraudClassifier]) -> pd.DataFrame:
    """
    Gera resumo dos modelos treinados.
    
    Args:
        models: Dicionário de modelos
        
    Returns:
        DataFrame com informações dos modelos
    """
    summary = []
    
    for name, model in models.items():
        summary.append({
            'model': name,
            'type': model.model_type,
            'training_time_s': round(model.training_time, 2) if model.training_time else None,
            'n_features': len(model.feature_names) if model.feature_names else None
        })
    
    return pd.DataFrame(summary)
