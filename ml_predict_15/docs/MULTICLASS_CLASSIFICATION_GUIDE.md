# Multi-Class Classification Guide (3-Class)

## Overview

The ml_predict_15 project now supports **3-class classification** for predicting cryptocurrency price movements:

- **Class -1 (Short)**: Price will decrease by ≥ target_pct%
- **Class 0 (Flat)**: Price will stay within ±target_pct%
- **Class 1 (Long)**: Price will increase by ≥ target_pct%

This is more realistic than binary classification as it accounts for sideways/flat markets.

## Changes Made

### 1. **Data Preparation** (`src/data_preparation.py`)

**Target Creation:**
```python
# Calculate percentage change
df[f'pct_change_{target_bars}'] = (
    df['Close'].shift(-target_bars) / df['Close'] - 1
) * 100

# Create 3-class target
target_up = (df[f'pct_change_{target_bars}'] >= target_pct).astype(int)
target_down = (df[f'pct_change_{target_bars}'] <= -target_pct).astype(int)
df['target'] = 0  # Default: Flat
df.loc[target_up, 'target'] = 1    # Long
df.loc[target_down, 'target'] = -1  # Short
```

**Example with target_pct=3.0:**
- If price increases by ≥3%: target = 1 (Long)
- If price decreases by ≥3%: target = -1 (Short)
- If price changes by <3% (either direction): target = 0 (Flat)

### 2. **Model Training** (`src/model_training.py`)

**Metrics Calculation:**
- Uses `average='weighted'` for multi-class metrics
- F1, Precision, Recall calculated per-class and weighted by support
- ROC AUC uses one-vs-rest (OVR) approach with weighted average

**Threshold Optimization:**
- Automatically skipped for 3-class (only works for binary)
- Displays message: "Multi-class classification detected (3 classes)"

**Classification Report:**
- Automatically detects number of classes
- Uses appropriate target names:
  - 3-class: `['Short (-1)', 'Flat (0)', 'Long (1)']`
  - 2-class: `['No Increase (0)', 'Increase (1)']`

**Code Changes:**
```python
# Detect number of classes
num_classes = len(np.unique(y_val))

# Use weighted average for multi-class
f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)

# ROC AUC for multi-class
if num_classes > 2:
    roc_auc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr', average='weighted')
```

### 3. **Model Evaluation** (`src/model_evaluation.py`)

**Confusion Matrix Summary:**
- Automatically detects matrix shape (2x2 or 3x3)
- For 3-class, displays:
  - **Pred Short/Flat/Long**: Number of predictions per class
  - **Correct Short/Flat/Long**: Correctly predicted samples per class
  - **Total Correct**: All correct predictions
  - **Total Wrong**: All incorrect predictions

**Example Output (3-Class):**
```
============================================================================
CONFUSION MATRIX SUMMARY (Validation Set)
============================================================================
           Model  Total  Pred Short  Pred Flat  Pred Long  Correct Short  Correct Flat  Correct Long  Total Correct  Total Wrong  Accuracy      F1
        Xgboost   5000        1200       2500       1300            850          1800           950           3600         1400    0.7200  0.6985
  Random Forest   5000        1150       2600       1250            820          1850           920           3590         1410    0.7180  0.6942
============================================================================
Legend (3-Class):
  Pred Short/Flat/Long = Number of predictions for each class (-1/0/1)
  Correct Short/Flat/Long = Correctly predicted samples for each class
  Total Correct = All correct predictions, Total Wrong = All incorrect predictions
============================================================================
```

## Benefits of 3-Class Classification

### 1. **More Realistic**
- Real markets have three states: up, down, and sideways
- Binary classification forces "flat" markets into up or down
- 3-class better represents actual market behavior

### 2. **Better Risk Management**
- Can choose to stay out of market (Class 0: Flat)
- Avoid false signals in ranging markets
- Only trade when strong directional movement predicted

### 3. **Improved Strategy**
- **Class 1 (Long)**: Buy signal
- **Class -1 (Short)**: Sell/Short signal
- **Class 0 (Flat)**: No trade / Hold cash

### 4. **More Informative**
- Know when model is uncertain (predicts Flat)
- Better understanding of market conditions
- Can adjust position sizing based on prediction confidence

## Usage

### Training with 3-Class Target

```python
from src.model_training import train

# Train with 3-class classification
models, scaler, results, best_model = train(
    df_train,
    target_bars=15,      # Look ahead 15 bars
    target_pct=3.0,      # ±3% threshold
    use_smote=False,     # SMOTE works with multi-class
    use_gpu=False,
    n_jobs=-1
)
```

### Understanding Results

**Classification Report:**
```
Classification Report:
              precision    recall  f1-score   support

  Short (-1)       0.71      0.68      0.69       980
   Flat (0)        0.72      0.74      0.73      2500
   Long (1)        0.73      0.71      0.72      1520

    accuracy                           0.72      5000
   macro avg       0.72      0.71      0.71      5000
weighted avg       0.72      0.72      0.72      5000
```

**Interpretation:**
- **Short (-1)**: Model correctly predicts 68% of downward movements
- **Flat (0)**: Model correctly predicts 74% of flat markets
- **Long (1)**: Model correctly predicts 71% of upward movements
- **Weighted F1**: 0.72 (overall performance across all classes)

### Backtesting with 3-Class Predictions

```python
from src.MLBacktester import MLBacktester

# Predictions will be -1, 0, or 1
predictions = model.predict(X_test_scaled)

# Strategy:
# -1: Short position
#  0: No position (flat/cash)
#  1: Long position

backtester = MLBacktester(
    df_test,
    predictions,
    initial_capital=10000,
    position_size=0.95,
    stop_loss_pct=2.0,
    take_profit_pct=5.0
)

results = backtester.run_backtest()
```

## Backward Compatibility

✅ **Fully backward compatible!**

The code automatically detects whether you're using:
- **Binary classification** (target = 0 or 1)
- **Multi-class classification** (target = -1, 0, or 1)

All metrics, visualizations, and reports adapt automatically.

### Switching Between Binary and Multi-Class

**For Binary (2-class):**
```python
# In data_preparation.py or your data prep code
df['target'] = (df[f'pct_change_{target_bars}'] >= target_pct).astype(int)
# Result: 0 (no increase) or 1 (increase)
```

**For Multi-Class (3-class):**
```python
# In data_preparation.py (already implemented)
target_up = (df[f'pct_change_{target_bars}'] >= target_pct).astype(int)
target_down = (df[f'pct_change_{target_bars}'] <= -target_pct).astype(int)
df['target'] = 0
df.loc[target_up, 'target'] = 1
df.loc[target_down, 'target'] = -1
# Result: -1 (short), 0 (flat), or 1 (long)
```

## Model Compatibility

### ✅ All Models Support Multi-Class

All sklearn models in the project support multi-class classification:
- Logistic Regression
- Ridge Classifier
- Naive Bayes
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- K-Nearest Neighbors
- SVM
- Neural Networks (LSTM, CNN, Hybrid)

### SMOTE with Multi-Class

SMOTE works with multi-class classification:
```python
# SMOTE will balance all 3 classes
models, scaler, results, best_model = train(
    df_train,
    use_smote=True  # Works with 3-class!
)
```

## Performance Considerations

### Class Imbalance

With 3 classes, imbalance is more complex:
- **Typical distribution**: Flat (50-60%), Long (20-25%), Short (20-25%)
- **SMOTE** can help balance all three classes
- **Class weights** automatically handled by models

### Evaluation Metrics

**Weighted Average (default):**
- Accounts for class imbalance
- More representative of overall performance
- Used for F1, Precision, Recall

**Macro Average:**
- Treats all classes equally
- Good for balanced datasets
- Can be misleading with imbalance

**Per-Class Metrics:**
- Most informative for multi-class
- Shows which classes model predicts well
- Available in classification report

## Best Practices

### 1. **Choose Appropriate Threshold**
```python
# Smaller threshold = more Flat predictions
target_pct = 2.0  # ±2% → More Flat (0)

# Larger threshold = fewer Flat predictions
target_pct = 5.0  # ±5% → Fewer Flat (0)
```

### 2. **Analyze Per-Class Performance**
- Check which class has lowest recall
- May need to adjust features or threshold
- Consider cost of misclassification per class

### 3. **Use Confusion Matrix**
- See which classes are confused with each other
- Common: Flat confused with Long/Short
- Adjust strategy based on confusion patterns

### 4. **Backtest Carefully**
- Account for transaction costs on each trade
- Flat (0) predictions = no trade = no cost
- Compare 3-class vs binary strategy performance

### 5. **Monitor Class Distribution**
```python
# Check class distribution
print(df['target'].value_counts())

# Expected for target_pct=3.0:
# 0 (Flat):  ~50-60%
# 1 (Long):  ~20-25%
# -1 (Short): ~20-25%
```

## Troubleshooting

### Issue: Too Many Flat Predictions

**Cause**: Threshold too small
**Solution**: Increase `target_pct`

```python
# Before: target_pct = 2.0 (too small)
# After:  target_pct = 4.0 (better)
```

### Issue: Too Few Flat Predictions

**Cause**: Threshold too large
**Solution**: Decrease `target_pct`

```python
# Before: target_pct = 5.0 (too large)
# After:  target_pct = 3.0 (better)
```

### Issue: Poor Performance on One Class

**Cause**: Class imbalance or insufficient features
**Solutions**:
1. Enable SMOTE: `use_smote=True`
2. Add more relevant features
3. Adjust class weights in model config
4. Use different model (e.g., XGBoost handles imbalance well)

### Issue: Threshold Optimization Not Working

**Expected**: Threshold optimization only works for binary classification
**Solution**: This is normal behavior for 3-class

## Summary

✅ **3-Class Classification Implemented**
- Target: -1 (Short), 0 (Flat), 1 (Long)
- All models support multi-class
- Automatic detection and handling
- Backward compatible with binary

✅ **Key Changes**
- `data_preparation.py`: 3-class target creation
- `model_training.py`: Multi-class metrics and ROC AUC
- `model_evaluation.py`: 3-class confusion matrix summary

✅ **Benefits**
- More realistic market representation
- Better risk management
- Improved trading strategy
- More informative predictions

The system seamlessly handles both binary and multi-class classification, automatically adapting all metrics, visualizations, and reports!
