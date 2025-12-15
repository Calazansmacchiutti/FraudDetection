"""
Configurações centralizadas do projeto de Detecção de Fraude.
"""
from pathlib import Path

# Diretórios
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models_saved"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Criar diretórios se não existirem
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset
DATASET_FILENAME = "creditcard.csv"
DATASET_PATH = RAW_DATA_DIR / DATASET_FILENAME

# Features
TARGET_COLUMN = "Class"
TIME_COLUMN = "Time"
AMOUNT_COLUMN = "Amount"
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]  # V1 a V28

# Parâmetros de treinamento
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Thresholds
FRAUD_THRESHOLD_DEFAULT = 0.5
ANOMALY_CONTAMINATION = 0.001  # ~0.1% de anomalias esperadas

# Hiperparâmetros padrão dos modelos
MODEL_PARAMS = {
    "logistic_regression": {
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "scale_pos_weight": 577,  # ratio de classes (284315/492)
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "eval_metric": "aucpr"
    },
    "lightgbm": {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1
    },
    "isolation_forest": {
        "n_estimators": 100,
        "contamination": ANOMALY_CONTAMINATION,
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    }
}

# Grid Search (versão reduzida para demonstração)
GRID_SEARCH_PARAMS = {
    "random_forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15],
        "min_samples_split": [2, 5]
    },
    "xgboost": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2]
    },
    "lightgbm": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2]
    }
}
