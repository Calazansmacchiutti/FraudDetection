"""
Módulo para detecção de anomalias não supervisionada.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional, Tuple, Literal
import joblib
from pathlib import Path
import time
import warnings

warnings.filterwarnings('ignore')


class AnomalyDetector:
    """Wrapper para modelos de detecção de anomalias."""
    
    AVAILABLE_MODELS = {
        'isolation_forest': IsolationForest,
        'lof': LocalOutlierFactor,
        'kmeans': KMeans
    }
    
    def __init__(
        self,
        model_type: str,
        params: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None
    ):
        """
        Inicializa o detector de anomalias.
        
        Args:
            model_type: Tipo do modelo
            params: Parâmetros do modelo
            name: Nome customizado
        """
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(f"Modelo '{model_type}' não disponível. Opções: {list(self.AVAILABLE_MODELS.keys())}")
        
        self.model_type = model_type
        self.params = params or {}
        self.name = name or model_type
        
        # Configurações específicas
        if model_type == 'lof':
            self.params['novelty'] = True
        
        self.model = self.AVAILABLE_MODELS[model_type](**self.params)
        self.is_fitted = False
        self.training_time = None
        self.scaler = StandardScaler()
        self.threshold = None
        
    def fit(self, X: pd.DataFrame) -> 'AnomalyDetector':
        """
        Treina o detector com dados normais.
        
        Args:
            X: Features (idealmente apenas transações legítimas)
            
        Returns:
            self
        """
        # Escalar dados
        X_scaled = self.scaler.fit_transform(X)
        
        start_time = time.time()
        
        if self.model_type == 'kmeans':
            self.model.fit(X_scaled)
            # Para K-Means, calculamos distâncias aos centroides
            distances = self.model.transform(X_scaled).min(axis=1)
            self.threshold = np.percentile(distances, 99)  # Top 1% como anomalia
        else:
            self.model.fit(X_scaled)
        
        self.training_time = time.time() - start_time
        self.is_fitted = True
        
        print(f"Detector {self.name} treinado em {self.training_time:.2f}s")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prediz se as instâncias são anomalias.
        
        Returns:
            Array com 1 para normal, -1 para anomalia
        """
        if not self.is_fitted:
            raise RuntimeError("Detector não treinado. Execute fit() primeiro.")
        
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'kmeans':
            distances = self.model.transform(X_scaled).min(axis=1)
            return np.where(distances > self.threshold, -1, 1)
        else:
            return self.model.predict(X_scaled)
    
    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        """
        Retorna scores de anomalia (quanto menor, mais anômalo).
        
        Returns:
            Array com scores de anomalia
        """
        if not self.is_fitted:
            raise RuntimeError("Detector não treinado. Execute fit() primeiro.")
        
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'isolation_forest':
            return self.model.score_samples(X_scaled)
        elif self.model_type == 'lof':
            return self.model.score_samples(X_scaled)
        elif self.model_type == 'kmeans':
            # Distância negativa (quanto mais distante, mais negativo)
            distances = self.model.transform(X_scaled).min(axis=1)
            return -distances
        
        return np.zeros(len(X))
    
    def predict_proba_anomaly(self, X: pd.DataFrame) -> np.ndarray:
        """
        Retorna probabilidade aproximada de ser anomalia.
        Normaliza os scores para [0, 1].
        
        Returns:
            Array com probabilidades de anomalia
        """
        scores = self.score_samples(X)
        
        # Normalizar para [0, 1] usando min-max
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score - min_score > 0:
            # Inverter para que anomalias tenham probabilidade alta
            proba = 1 - (scores - min_score) / (max_score - min_score)
        else:
            proba = np.zeros(len(scores))
        
        return proba
    
    def save(self, filepath: str | Path) -> None:
        """Salva o detector em disco."""
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'params': self.params,
            'name': self.name,
            'scaler': self.scaler,
            'threshold': self.threshold,
            'training_time': self.training_time
        }, filepath)
        print(f"Detector salvo em: {filepath}")
    
    @classmethod
    def load(cls, filepath: str | Path) -> 'AnomalyDetector':
        """Carrega um detector salvo."""
        data = joblib.load(filepath)
        
        instance = cls(
            model_type=data['model_type'],
            params=data['params'],
            name=data['name']
        )
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.threshold = data['threshold']
        instance.training_time = data['training_time']
        instance.is_fitted = True
        
        return instance


class HybridFraudDetector:
    """
    Combina modelo supervisionado e detector de anomalias.
    
    Estratégia: Uma transação é suspeita se:
    - O modelo supervisionado indica alta probabilidade de fraude, OU
    - O detector de anomalias identifica comportamento anômalo
    """
    
    def __init__(
        self,
        supervised_model,
        anomaly_detector: AnomalyDetector,
        supervised_weight: float = 0.7,
        anomaly_weight: float = 0.3
    ):
        """
        Inicializa o detector híbrido.
        
        Args:
            supervised_model: Modelo supervisionado treinado
            anomaly_detector: Detector de anomalias treinado
            supervised_weight: Peso do modelo supervisionado
            anomaly_weight: Peso do detector de anomalias
        """
        self.supervised_model = supervised_model
        self.anomaly_detector = anomaly_detector
        self.supervised_weight = supervised_weight
        self.anomaly_weight = anomaly_weight
        
        # Normalizar pesos
        total = supervised_weight + anomaly_weight
        self.supervised_weight /= total
        self.anomaly_weight /= total
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Retorna probabilidade combinada de fraude.
        
        Returns:
            Array com probabilidades de fraude
        """
        # Probabilidade do modelo supervisionado
        supervised_proba = self.supervised_model.predict_proba(X)[:, 1]
        
        # Probabilidade do detector de anomalias
        anomaly_proba = self.anomaly_detector.predict_proba_anomaly(X)
        
        # Combinação ponderada
        combined_proba = (
            self.supervised_weight * supervised_proba +
            self.anomaly_weight * anomaly_proba
        )
        
        return combined_proba
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Retorna predições binárias.
        
        Args:
            X: Features
            threshold: Limiar de decisão
            
        Returns:
            Array com predições (0 ou 1)
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def get_detection_breakdown(self, X: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """
        Retorna análise detalhada de cada componente.
        
        Returns:
            DataFrame com scores de cada detector
        """
        supervised_proba = self.supervised_model.predict_proba(X)[:, 1]
        anomaly_proba = self.anomaly_detector.predict_proba_anomaly(X)
        combined = self.predict_proba(X)
        
        return pd.DataFrame({
            'supervised_proba': supervised_proba,
            'anomaly_proba': anomaly_proba,
            'combined_proba': combined,
            'final_prediction': (combined >= threshold).astype(int)
        })


def train_anomaly_detector(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = 'isolation_forest',
    params: Optional[Dict[str, Any]] = None,
    train_on_normal_only: bool = True
) -> AnomalyDetector:
    """
    Treina um detector de anomalias.
    
    Args:
        X_train: Features de treino
        y_train: Labels (para filtrar normais se train_on_normal_only=True)
        model_type: Tipo do detector
        params: Parâmetros do modelo
        train_on_normal_only: Se deve treinar apenas com dados normais
        
    Returns:
        Detector treinado
    """
    detector = AnomalyDetector(model_type=model_type, params=params)
    
    if train_on_normal_only:
        # Treinar apenas com transações legítimas
        X_normal = X_train[y_train == 0]
        print(f"Treinando detector com {len(X_normal):,} transações normais")
        detector.fit(X_normal)
    else:
        detector.fit(X_train)
    
    return detector
