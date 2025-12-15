# Credit Card Fraud Detection System

A production-ready machine learning system for detecting fraudulent credit card transactions using ensemble methods and anomaly detection.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Model Performance](#model-performance)
- [Configuration](#configuration)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Overview

This system implements a comprehensive fraud detection pipeline that combines supervised and unsupervised machine learning approaches:

- **Supervised Models**: XGBoost, Random Forest
- **Unsupervised Models**: Isolation Forest, K-Means Clustering
- **Hybrid Approach**: Ensemble combining multiple models for improved detection

### Key Features

- Production-ready code with proper logging and error handling
- RESTful API for real-time predictions
- Batch processing capabilities for large-scale inference
- Comprehensive model evaluation and monitoring
- Configurable thresholds for precision-recall trade-offs
- Docker support for containerized deployment

### Performance Summary

| Model | Precision | Recall | F1-Score | ROC-AUC | Avg Precision |
|-------|-----------|--------|----------|---------|---------------|
| XGBoost | 0.883 | 0.847 | 0.865 | 0.970 | 0.873 |
| Random Forest | 0.762 | 0.816 | 0.788 | 0.979 | 0.816 |
| K-Means | 0.045 | 0.827 | 0.085 | 0.955 | 0.187 |
| Isolation Forest | 0.001 | 0.684 | 0.002 | 0.956 | 0.166 |

## Project Structure

```
fraud_detection_project/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── setup.py
├── Makefile
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
│
├── config/
│   ├── __init__.py
│   ├── settings.py              # Application settings
│   └── logging_config.py        # Logging configuration
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py            # Data loading utilities
│   │   ├── preprocessor.py      # Data preprocessing
│   │   └── validator.py         # Data validation
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engineer.py          # Feature engineering
│   │   └── selector.py          # Feature selection
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py              # Base model class
│   │   ├── supervised.py        # Supervised models
│   │   ├── unsupervised.py      # Unsupervised models
│   │   ├── ensemble.py          # Ensemble methods
│   │   └── registry.py          # Model registry
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py           # Evaluation metrics
│       └── visualizer.py        # Visualization utilities
│
├── api/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── predict.py           # Prediction endpoints
│   │   └── health.py            # Health check endpoints
│   └── schemas/
│       ├── __init__.py
│       └── transaction.py       # Request/Response schemas
│
├── scripts/
│   ├── train.py                 # Model training script
│   ├── evaluate.py              # Model evaluation script
│   ├── predict.py               # Batch prediction script
│   └── download_data.py         # Data download script
│
├── notebooks/
│   └── 01_fraud_detection_analysis.ipynb
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Test fixtures
│   ├── test_data/
│   ├── test_features/
│   ├── test_models/
│   └── test_api/
│
├── data/
│   ├── raw/                     # Raw data files
│   ├── processed/               # Processed data files
│   └── external/                # External data sources
│
├── models/                      # Trained model artifacts
│
├── reports/
│   ├── figures/                 # Generated figures
│   └── metrics/                 # Evaluation metrics
│
├── logs/                        # Application logs
│
└── docs/
    ├── api.md                   # API documentation
    ├── deployment.md            # Deployment guide
    └── model_card.md            # Model documentation
```

## Installation

### Prerequisites

- Python 3.9+
- pip or conda

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Docker Installation

```bash
# Build the Docker image
docker build -t fraud-detection .

# Run the container
docker run -p 8000:8000 fraud-detection
```

## Quick Start

### 1. Download the Dataset

```bash
python scripts/download_data.py
```

Or manually download from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place in `data/raw/`.

### 2. Train Models

```bash
python scripts/train.py --config config/settings.py
```

### 3. Evaluate Models

```bash
python scripts/evaluate.py --model xgboost
```

### 4. Start API Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 5. Make Predictions

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{"V1": -1.35, "V2": -0.07, ..., "Amount": 149.62}'
```

## Usage

### Python API

```python
from src.models import FraudDetector

# Initialize detector
detector = FraudDetector.load("models/xgboost_model.pkl")

# Single prediction
result = detector.predict(transaction_data)
print(f"Fraud probability: {result['probability']:.4f}")
print(f"Risk level: {result['risk_level']}")

# Batch prediction
results = detector.predict_batch(transactions_df)
```

### Command Line Interface

```bash
# Train a specific model
python scripts/train.py --model xgboost --output models/

# Evaluate model performance
python scripts/evaluate.py --model models/xgboost_model.pkl --data data/processed/test.csv

# Run batch predictions
python scripts/predict.py --input data/new_transactions.csv --output predictions.csv
```

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/v1/models` | List available models |
| POST | `/api/v1/predict` | Single transaction prediction |
| POST | `/api/v1/predict/batch` | Batch predictions |

### Example Request

```json
POST /api/v1/predict
{
    "V1": -1.359807,
    "V2": -0.072781,
    "V3": 2.536347,
    ...
    "V28": -0.021053,
    "Amount": 149.62
}
```

### Example Response

```json
{
    "transaction_id": "txn_123456",
    "is_fraud": false,
    "probability": 0.0234,
    "risk_level": "LOW",
    "model_version": "1.0.0",
    "timestamp": "2025-01-15T10:30:00Z"
}
```

## Model Performance

### Evaluation Metrics

The system uses multiple metrics appropriate for imbalanced classification:

- **Average Precision (AP)**: Primary metric for model selection
- **Precision**: Proportion of fraud alerts that are actual fraud
- **Recall**: Proportion of actual fraud that is detected
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Overall discriminative ability

### Threshold Selection

Default threshold is 0.5, but can be adjusted based on business requirements:

| Threshold | Precision | Recall | Use Case |
|-----------|-----------|--------|----------|
| 0.3 | Lower | Higher | Maximize fraud detection |
| 0.5 | Balanced | Balanced | Default setting |
| 0.7 | Higher | Lower | Minimize false alerts |

## Configuration

Configuration is managed through environment variables and `config/settings.py`:

```python
# config/settings.py
MODEL_PATH = "models/"
LOG_LEVEL = "INFO"
API_HOST = "0.0.0.0"
API_PORT = 8000
PREDICTION_THRESHOLD = 0.5
```

Environment variables (`.env`):

```bash
MODEL_PATH=models/
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test module
pytest tests/test_models/

# Run with verbose output
pytest -v
```

## Deployment

### Docker Compose

```bash
docker-compose up -d
```

### Kubernetes

Kubernetes manifests are available in `k8s/` directory.

```bash
kubectl apply -f k8s/
```

### Cloud Deployment

See [docs/deployment.md](docs/deployment.md) for detailed deployment guides for:
- AWS (ECS, Lambda)
- GCP (Cloud Run, GKE)
- Azure (Container Apps, AKS)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Open a Pull Request

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and development process.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) by Machine Learning Group - ULB
- Inspired by production ML systems at major financial institutions
