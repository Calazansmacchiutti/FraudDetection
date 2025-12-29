# 🚀 Quick Start Guide - KYC/KYT Loan Default Prediction

## ⚡ 5-Minute Setup

### 1. Install Dependencies
```bash
cd kyc_kyt_fraud_detection
pip install -r requirements.txt
```

### 2. Prepare Data
Place these files in `data/raw/`:
- `loan.csv` (682 loans)
- `trans.csv` (1M+ transactions)

### 3. Train Models
```bash
# Fast training with pre-tuned parameters (recommended)
python scripts/train.py

# Train specific model
python scripts/train.py --model random_forest

# Full Optuna optimization (slower, 5-10 minutes)
python scripts/train.py --optimize --n-trials 50
```

---

## 📊 What You Get

After training, you'll find:

**Models:**
- `models/random_forest_model.pkl` (Best: 91.1% Avg Precision)
- `models/xgboost_model.pkl` (Alternative: 90.0% Avg Precision)
- `models/autoencoder_model.pkl` (Anomaly detection: 47.2% Avg Precision)
- `models/scaler.pkl` (Data scaler)

**Reports:**
- `reports/metrics/model_comparison.csv` - Performance comparison
- `reports/metrics/random_forest_feature_importance.csv` - Top features
- `reports/optuna_studies/{model}_optimization.csv` - Optimization history

---

## 🔮 Making Predictions

```python
from src.models.supervised import RandomForestDefaultModel
import joblib
import pandas as pd

# Load model and scaler
model = RandomForestDefaultModel.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Prepare new loan data (must have same features as training)
new_loan = pd.DataFrame({...})  # Your loan data

# Scale and predict
X_scaled = scaler.transform(new_loan)
probability = model.predict_proba(X_scaled)[0]
prediction = model.predict(X_scaled)[0]
risk_level = model.get_risk_level(probability)

print(f"Default Probability: {probability:.2%}")
print(f"Prediction: {'DEFAULT' if prediction == 1 else 'GOOD'}")
print(f"Risk Level: {risk_level}")
```

---

## 🎯 Key Features

**KYC Features (Loan):**
- `payment_to_amount` - Monthly payment burden
- `loan_to_balance` - Loan size vs account balance
- `overpayment_ratio` - Total interest

**KYT Features (Transactions):**
- `balance_min` - Minimum balance (negative = risk!)
- `had_negative_balance` - Risk flag
- `amount_cv` - Transaction volatility
- `balance_stability` - Balance consistency

---

## 📈 Expected Performance

| Model | Precision | Recall | F1 | ROC-AUC | Avg Precision |
|-------|-----------|--------|-----|---------|---------------|
| Random Forest | 72.7% | **88.9%** | 80.0% | **99.1%** | **91.1%** |
| XGBoost | 72.7% | 88.9% | 80.0% | 99.1% | 90.0% |

**Translation:** Catches 9 out of 10 defaults with 73% precision

---

## 🛠️ Customization

Edit `config/settings.py`:

```python
# Change default model
DEFAULT_MODEL = "random_forest"  # or "xgboost"

# Adjust risk thresholds
CRITICAL_RISK_THRESHOLD = 0.8  # 80%+ probability
HIGH_RISK_THRESHOLD = 0.6      # 60-79%
MEDIUM_RISK_THRESHOLD = 0.3    # 30-59%

# Optuna settings
N_TRIALS = 50  # More trials = better but slower
CV_FOLDS = 5   # Cross-validation folds
```

---

## 🔍 Troubleshooting

**Missing Dependencies?**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**TensorFlow Issues? (for Autoencoder)**
```bash
# CPU-only version (lighter)
pip install tensorflow-cpu
```

**Data Not Found?**
```bash
# Check file paths
ls data/raw/
# Should show: loan.csv, trans.csv
```

**Training Errors?**
```bash
# Check Python version
python --version  # Should be 3.9+

# Run with verbose logging
python scripts/train.py --model random_forest 2>&1 | tee training.log
```

---

## 📚 Next Steps

1. **Explore Results:**
   - Check `reports/metrics/` for detailed metrics
   - Review feature importance to understand model decisions

2. **Optimize Further:**
   - Run with `--optimize` flag for better hyperparameters
   - Increase `--n-trials` for more thorough search

3. **Deploy (Optional):**
   - Add FastAPI for REST API
   - Containerize with Docker
   - Set up monitoring

---

## 📖 Full Documentation

- **Complete Guide**: `README.md`
- **Implementation Log**: `../IMPLEMENTATION_LOG.md`
- **Configuration**: `config/settings.py`

---

**Ready to predict loan defaults!** 🎉
