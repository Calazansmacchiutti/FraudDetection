# 🏦 KYC/KYT Loan Default Prediction

A production-ready machine learning system for predicting loan defaults using **KYC (Know Your Customer)** and **KYT (Know Your Transaction)** approaches, with hyperparameter optimization via **Optuna**.

---

## 📊 Project Overview

This system predicts loan defaults by combining:
- **KYC Features**: Loan characteristics (amount, duration, payments)
- **KYT Features**: Transaction behavior patterns (frequency, volatility, balance stability)

**Dataset**: Czech Banking Dataset (PKDD'99)
- **682 loans** (6.6% default rate)
- **1,056,320 transactions** from 4,500 accounts

---

## 🎯 Performance

| Model | Precision | Recall | F1 | ROC-AUC | Avg Precision |
|-------|-----------|--------|-----|---------|---------------|
| **Random Forest (Tuned)** | 72.7% | **88.9%** | 80.0% | **99.1%** | **91.1%** |
| XGBoost (Tuned) | 72.7% | 88.9% | 80.0% | 99.1% | 90.0% |
| Autoencoder | 35.3% | 66.7% | 46.2% | 93.5% | 47.2% |

**Key Achievement**: 14.3% improvement in Average Precision after Optuna tuning (76.8% → 91.1%)

---

## 🗂️ Project Structure

```
kyc_kyt_fraud_detection/
├── README.md
├── requirements.txt
├── setup.py
│
├── config/
│   └── settings.py              # Centralized configuration
│
├── data/
│   ├── raw/                     # loan.csv, trans.csv
│   └── processed/               # Processed features
│
├── src/
│   ├── data/
│   │   ├── loader.py            # Czech Bank data loader
│   │   ├── aggregator.py        # Transaction aggregation
│   │   └── validator.py         # Data quality checks
│   │
│   ├── features/
│   │   ├── kyc_features.py      # Customer-level features
│   │   ├── kyt_features.py      # Transaction behavior features
│   │   └── engineering.py       # Feature engineering pipeline
│   │
│   ├── models/
│   │   ├── base.py              # Base model class
│   │   ├── supervised.py        # XGBoost, Random Forest
│   │   ├── unsupervised.py      # Autoencoder anomaly detection
│   │   └── optimizer.py         # Optuna hyperparameter tuning
│   │
│   └── evaluation/
│       ├── metrics.py           # Evaluation metrics
│       └── statistical_tests.py # Cohen's d, KS test
│
├── scripts/
│   └── train.py                 # Training pipeline
│
├── models/                      # Saved models
├── reports/                     # Evaluation reports
└── notebooks/                   # Exploratory analysis
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
cd kyc_kyt_fraud_detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Prepare Data

Place the Czech Bank dataset files in `data/raw/`:
- `loan.csv`
- `trans.csv`

### Train Models

```bash
# Train all models with pre-tuned parameters (fast)
python scripts/train.py

# Train specific model
python scripts/train.py --model random_forest

# Run Optuna optimization (slow but better results)
python scripts/train.py --model random_forest --optimize --n-trials 50

# Full optimization on all models
python scripts/train.py --model all --optimize --n-trials 50
```

---

## 📈 Key Features

### 1. **KYC Features** (Loan Characteristics)
```python
- payment_to_amount          # Monthly payment burden
- amount_per_month           # Loan size per month
- overpayment_ratio          # Total payment / original amount
- implied_interest_rate      # Effective interest rate
- loan_size categories       # Short/medium/long-term
```

### 2. **KYT Features** (Transaction Behavior)
```python
- amount_cv                  # Transaction amount volatility
- balance_stability          # Inverse of balance CV
- had_negative_balance       # Risk flag
- balance_range              # Min-max spread
- tx_type_diversity          # Transaction type entropy
```

### 3. **Interaction Features** (KYC × KYT)
```python
- loan_to_balance            # Loan size vs average balance
- payment_to_avg_transaction # Payment burden vs spending
- risk_flag_count            # Combined risk indicators
```

---

## 🔬 Model Details

### Random Forest (Best Model)

**Optuna-Tuned Hyperparameters:**
```python
{
    'n_estimators': 173,
    'max_depth': 5,
    'min_samples_split': 9,
    'min_samples_leaf': 5,
    'max_features': None,
    'class_weight': 'balanced'
}
```

**Why it's best:**
- Highest Average Precision (91.1%)
- Excellent recall (88.9% of defaults detected)
- More interpretable than XGBoost
- Less prone to overfitting on small dataset

### XGBoost (Alternative)

**Optuna-Tuned Hyperparameters:**
```python
{
    'n_estimators': 358,
    'max_depth': 6,
    'learning_rate': 0.038805,
    'scale_pos_weight': 25.217325,
    # ... (see config/settings.py for full params)
}
```

---

## 📊 Feature Discrimination Analysis

Top features by Cohen's d (effect size):

| Feature | Cohen's d | Effect | Interpretation |
|---------|-----------|--------|----------------|
| `amount_min` | 1.69 | **Large** | Minimum transaction amount differs significantly |
| `balance_min` | 1.34 | **Large** | Defaults have negative balances (-4.6k avg) |
| `amount` (loan) | 0.86 | **Large** | Defaults borrow 72% more |
| `balance_median` | 0.82 | **Large** | Defaults have 26% lower median balance |
| `balance_mean` | 0.76 | **Medium** | 23% lower average balance |

---

## 🛠️ Configuration

Edit `config/settings.py` to customize:

```python
# Data paths
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Model selection
DEFAULT_MODEL = "random_forest"  # or "xgboost"

# Optuna settings
N_TRIALS = 50
CV_FOLDS = 5
METRIC = "average_precision"

# Risk thresholds
CRITICAL_RISK_THRESHOLD = 0.8
HIGH_RISK_THRESHOLD = 0.6
MEDIUM_RISK_THRESHOLD = 0.3
```

---

## 📚 Usage Examples

### Python API

```python
from pathlib import Path
from src.data.loader import CzechBankDataLoader
from src.data.aggregator import TransactionAggregator
from src.features.engineering import FeatureEngineer
from src.models.supervised import RandomForestDefaultModel
import joblib

# Load data
loader = CzechBankDataLoader(Path("data/raw"))
loan_df, trans_df = loader.load_all()

# Aggregate transactions
aggregator = TransactionAggregator()
trans_agg = aggregator.aggregate(trans_df)

# Engineer features
engineer = FeatureEngineer()
df = engineer.engineer_features(loan_df, trans_agg)
X, y, features = engineer.prepare_modeling_data(df)

# Load trained model and scaler
model = RandomForestDefaultModel.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Predict
X_scaled = scaler.transform(X)
predictions = model.predict(X_scaled)
probabilities = model.predict_proba(X_scaled)

# Get risk level
for prob in probabilities[:5]:
    risk = model.get_risk_level(prob)
    print(f"Probability: {prob:.2%}, Risk: {risk}")
```

### Hyperparameter Optimization

```python
from src.models.optimizer import OptunaOptimizer

# Initialize optimizer
optimizer = OptunaOptimizer(n_trials=50, metric='average_precision')

# Optimize Random Forest
best_params = optimizer.optimize_random_forest(X_train, y_train)

# Get optimization history
history = optimizer.get_optimization_history()
history.to_csv("rf_optimization_history.csv")
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test module
pytest tests/test_models.py
```

---

## 📈 Model Evaluation

The training script automatically generates:

1. **Model Comparison** (`reports/metrics/model_comparison.csv`)
2. **Feature Importance** (`reports/metrics/{model}_feature_importance.csv`)
3. **Optuna Study** (`reports/optuna_studies/{model}_optimization.csv`)
4. **Statistical Discrimination Analysis** (printed during training)

---

## 🔄 Workflow

```
1. Load Data (loan.csv + trans.csv)
   ↓
2. Validate Quality
   ↓
3. Aggregate Transactions by Account
   ↓
4. Engineer Features (KYC + KYT)
   ↓
5. Statistical Analysis (Cohen's d, KS test)
   ↓
6. Train/Test Split + Scaling
   ↓
7. Train Models (with/without Optuna)
   ↓
8. Evaluate & Compare
   ↓
9. Save Best Model
```

---

## 🚧 Future Enhancements

- [ ] FastAPI REST API for real-time predictions
- [ ] SHAP values for model interpretability
- [ ] Time-series features (transaction trends)
- [ ] Ensemble stacking (RF + XGBoost)
- [ ] Docker containerization
- [ ] MLflow experiment tracking
- [ ] Production monitoring dashboard

---

## 📖 Documentation

- **Model Card**: `docs/model_card.md`
- **Methodology**: `docs/methodology.md`
- **Implementation Log**: `../IMPLEMENTATION_LOG.md`

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## 📄 License

MIT License

---

## 🙏 Acknowledgments

- **Dataset**: Czech Banking Dataset (PKDD'99 Discovery Challenge)
- **Optuna**: For efficient Bayesian optimization
- **Scikit-learn, XGBoost**: Core ML libraries

---

## 📬 Contact

For questions or issues, please open an issue on GitHub.

---

**Built with ❤️ using Python, Optuna, and modern ML best practices**
