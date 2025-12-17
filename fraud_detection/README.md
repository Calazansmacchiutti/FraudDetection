# 🔍 Fraud Detection System

Sistema completo de detecção de fraude em cartões de crédito utilizando Machine Learning.

##  Visão Geral

Este projeto implementa um pipeline de ML para detecção de fraudes com:

- **Modelos Supervisionados**: Logistic Regression, Random Forest, XGBoost, LightGBM
- **Detecção de Anomalias**: Isolation Forest
- **Sistema Híbrido**: Combinação de abordagens para máxima cobertura

##  Estrutura do Projeto

```
fraud_detection/
├── config/
│   └── config.py           # Configurações centralizadas
├── data/
│   ├── raw/                # Dados brutos (creditcard.csv)
│   └── processed/          # Dados processados
├── models_saved/           # Modelos treinados
├── notebooks/
│   └── 01_fraud_detection_complete.ipynb  # Notebook principal
├── reports/                # Relatórios e visualizações
├── src/
│   ├── data/              # Carregamento e preparação de dados
│   │   └── loader.py
│   ├── features/          # Engenharia de features
│   │   └── engineering.py
│   ├── models/            # Modelos de classificação e anomalia
│   │   ├── classifiers.py
│   │   └── anomaly.py
│   ├── evaluation/        # Métricas e visualizações
│   │   ├── metrics.py
│   │   └── visualizations.py
│   └── utils/             # Funções utilitárias
│       └── helpers.py
├── tests/                  # Testes unitários
├── requirements.txt        # Dependências
└── README.md
```

##  Quick Start

### Instalar Dependências

```bash
pip install -r requirements.txt
```

### Baixar Dataset

Baixe o dataset do Kaggle:
- [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

`creditcard.csv` em `data/raw/`

### Executar o Notebook

```bash
cd fraud_detection
jupyter notebook notebooks/01_fraud_detection_complete.ipynb
```

## Dataset

**Credit Card Fraud Detection Dataset**
- 284.807 transações
- 492 fraudes (0.17%)
- 30 features (V1-V28 são componentes PCA, + Time e Amount)

## Módulos

### Data (`src/data/`)
- Carregamento de dados
- Resumo estatístico
- Separação features/target

### Features (`src/features/`)
- Feature engineering (temporal, valor, interações)
- Scaling com RobustScaler
- Balanceamento de classes (SMOTE, Undersampling)

### Models (`src/models/`)
- `FraudClassifier`: Wrapper para modelos supervisionados
- `AnomalyDetector`: Detecção de anomalias
- `HybridFraudDetector`: Combinação de abordagens

### Evaluation (`src/evaluation/`)
- Métricas: Precision, Recall, F1, ROC-AUC, Average Precision
- Visualizações: ROC curves, PR curves, Confusion Matrix
- Análise de threshold

## Resultados Típicos

| Modelo | Precision | Recall | F1 | Avg Precision |
|--------|-----------|--------|-----|---------------|
| XGBoost | ~0.90 | ~0.80 | ~0.85 | ~0.85 |
| LightGBM | ~0.88 | ~0.82 | ~0.85 | ~0.84 |
| Random Forest | ~0.85 | ~0.78 | ~0.81 | ~0.80 |
| Logistic Regression | ~0.75 | ~0.70 | ~0.72 | ~0.70 |

*Resultados podem variar dependendo do random seed e hiperparâmetros*

## Uso em Produção

```python
from src.models import FraudClassifier
import joblib

# Carregar modelo
model = FraudClassifier.load('models_saved/xgboost_model.pkl')
scaler = joblib.load('models_saved/feature_engineer.pkl')

# Predição
X_scaled = scaler.transform(transaction_data)
proba = model.predict_proba(X_scaled)[:, 1]
is_fraud = proba > 0.5
```

## Métricas Importantes

Para dados desbalanceados, foque em:

1. **Average Precision (AP)**: Área sob a curva PR
2. **Recall**: Proporção de fraudes detectadas
3. **Precision**: Proporção de alertas que são realmente fraudes
4. **F1-Score**: Média harmônica de Precision e Recall

⚠️ **Evite usar apenas Accuracy** - pode ser enganosa com dados desbalanceados.

## Melhorias Futuras

- [ ] API REST para servir predições
- [ ] Dashboard de monitoramento
- [ ] Pipeline de retreinamento automático
- [ ] Deep Learning (Autoencoders)
- [ ] Feature store
- [ ] A/B testing framework

## Referências

- [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

## Licença

MIT License
