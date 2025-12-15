# Model Card: Credit Card Fraud Detection

## Model Details

### Model Description

This model detects fraudulent credit card transactions using machine learning. The primary model is XGBoost, with supporting models including Random Forest, Isolation Forest, and K-Means for ensemble predictions.

### Model Version

- Version: 1.0.0
- Date: 2025

### Model Type

- Primary: XGBoost Classifier (Gradient Boosting)
- Secondary: Random Forest Classifier
- Anomaly Detection: Isolation Forest, K-Means Clustering

## Intended Use

### Primary Intended Uses

- Real-time fraud detection for credit card transactions
- Batch processing of historical transactions
- Risk scoring for transaction monitoring

### Primary Intended Users

- Financial institutions
- Payment processors
- E-commerce platforms

### Out-of-Scope Uses

- Detection of other types of fraud (identity theft, account takeover)
- Credit scoring or creditworthiness assessment
- Transaction authorization decisions (should be one input among many)

## Training Data

### Dataset

- Source: Kaggle Credit Card Fraud Detection Dataset
- Origin: Transactions by European cardholders in September 2013
- Size: 284,807 transactions
- Features: 28 PCA components (V1-V28), Time, Amount
- Target: Class (0=legitimate, 1=fraud)

### Class Distribution

- Legitimate: 284,315 (99.83%)
- Fraudulent: 492 (0.17%)
- Imbalance Ratio: 577:1

### Preprocessing

- RobustScaler normalization
- Engineered features: Hour_sin, Hour_cos, Amount_log, PCA_magnitude, V14_V12

## Evaluation Results

### Performance Metrics

| Model | Precision | Recall | F1-Score | ROC-AUC | Avg Precision |
|-------|-----------|--------|----------|---------|---------------|
| XGBoost | 0.883 | 0.847 | 0.865 | 0.970 | 0.873 |
| Random Forest | 0.762 | 0.816 | 0.788 | 0.979 | 0.816 |
| K-Means | 0.045 | 0.827 | 0.085 | 0.955 | 0.187 |
| Isolation Forest | 0.001 | 0.684 | 0.002 | 0.956 | 0.166 |

### Confusion Matrix (XGBoost)

- True Positives: 83 frauds correctly detected
- False Positives: 11 legitimate transactions flagged
- True Negatives: 56,853 legitimate transactions correctly classified
- False Negatives: 15 frauds missed

### Business Impact

- Fraud Detection Rate: 84.7%
- False Alert Rate: 0.02%

## Limitations

### Known Limitations

1. **Data Age**: Training data from 2013 may not reflect current fraud patterns
2. **Geographic Scope**: European cardholders only
3. **Feature Anonymization**: PCA features limit interpretability
4. **Class Imbalance**: Performance on minority class requires careful threshold tuning

### Recommendations

1. Regularly retrain with recent data
2. Monitor for concept drift
3. Adjust threshold based on operational requirements
4. Use as one component in a multi-layered fraud prevention system

## Ethical Considerations

### Potential Biases

- Model trained on European transactions may perform differently on other populations
- Historical biases in fraud labeling may be reflected in predictions

### Mitigation Strategies

- Regular fairness audits across demographic segments
- Human review for high-risk decisions
- Continuous monitoring of prediction distributions

## Caveats and Recommendations

### Threshold Selection

| Use Case | Recommended Threshold | Trade-off |
|----------|----------------------|-----------|
| High Security | 0.3 | More fraud caught, more false alerts |
| Balanced | 0.5 | Default setting |
| Low Friction | 0.7 | Fewer alerts, some fraud may pass |

### Deployment Recommendations

1. Implement real-time monitoring of key metrics
2. Set up alerts for significant metric changes
3. Establish feedback loop for confirmed fraud cases
4. Plan for regular model retraining (monthly or quarterly)
