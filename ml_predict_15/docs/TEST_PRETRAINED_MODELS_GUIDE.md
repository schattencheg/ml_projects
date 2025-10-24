# Test Pretrained Models Guide

This guide explains how to use the `test_pretrained_models.ipynb` notebook to evaluate pretrained models on test datasets.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Notebook Features](#notebook-features)
4. [Usage Instructions](#usage-instructions)
5. [Output Files](#output-files)
6. [Interpreting Results](#interpreting-results)

---

## Overview

The `test_pretrained_models.ipynb` notebook provides an interactive way to:
- Load pretrained models from any training session
- Test models against new datasets
- Compare training vs test performance
- Visualize results
- Save test reports

---

## Quick Start

### 1. Open the Notebook

```bash
jupyter notebook test_pretrained_models.ipynb
```

Or use VS Code, JupyterLab, or Google Colab.

### 2. Run All Cells

Click "Run All" or execute cells sequentially.

### 3. Review Results

The notebook will:
- List available training sessions
- Load the latest session by default
- Test all models
- Display results and visualizations
- Save test report

---

## Notebook Features

### 1. **Session Selection**

Lists all available training sessions with details:

```
Found 3 training session(s):

1. 2024-01-17_16-45-10
   Best Model: logistic_regression, Accuracy: 0.7234

2. 2024-01-16_09-15-22
   Best Model: random_forest, Accuracy: 0.7156

3. 2024-01-15_14-30-45
   Best Model: xgboost, Accuracy: 0.7087
```

### 2. **Model Loading**

Automatically loads:
- All trained models
- Fitted scaler
- Training configuration

```python
✓ Scaler loaded
✓ Loaded: logistic_regression
✓ Loaded: random_forest
✓ Loaded: xgboost
...
Total models loaded: 8
```

### 3. **Test Data Preparation**

Loads and prepares test dataset:
- Reads CSV file
- Generates features
- Creates target variable
- Scales features

### 4. **Model Evaluation**

Tests all models and calculates:
- Accuracy
- F1 Score
- Precision
- Recall
- ROC AUC

### 5. **Visualizations**

Creates multiple plots:
- Performance comparison (all metrics)
- Confusion matrix (best model)
- Training vs Test comparison

### 6. **Detailed Analysis**

For the best model:
- Classification report
- Confusion matrix
- Additional metrics (specificity, sensitivity, etc.)

### 7. **Results Saving**

Saves to session directory:
- `test_results.csv` - All model metrics
- `test_report.txt` - Detailed text report

---

## Usage Instructions

### Default Usage (Latest Session)

Simply run all cells. The notebook will:
1. Load the most recent training session
2. Use default test dataset (`data/btc_2023.csv`)
3. Evaluate all models
4. Display results

### Custom Session Selection

To test a specific session, modify cell 3:

```python
# Option 1: Use latest session (default)
selected_session = sessions[0]

# Option 2: Manually specify session
selected_session = '2024-01-15_14-30-45'
```

### Custom Test Dataset

To use a different test dataset, modify cell 5:

```python
# Change this to your test dataset path
test_data_path = 'data/my_custom_test_data.csv'
```

### Custom Target Parameters

If your test data requires different parameters, modify cell 6:

```python
# Use same parameters as training (adjust if needed)
target_bars = 45
target_pct = 3.0
```

---

## Output Files

### 1. test_results.csv

Located in: `models/{session}/test_results.csv`

Contains all model metrics:

```csv
Model,Accuracy,F1_Score,Precision,Recall,ROC_AUC
logistic_regression,0.7234,0.6187,0.6621,0.5834,0.7892
random_forest,0.7156,0.6089,0.6543,0.5701,0.7823
xgboost,0.7087,0.5976,0.6234,0.5567,0.7756
...
```

### 2. test_report.txt

Located in: `models/{session}/test_report.txt`

Contains detailed report:

```
TEST RESULTS REPORT
================================================================================

Training Session: 2024-01-15_14-30-45
Test Dataset: data/btc_2023.csv
Test Set Size: 50000
Target Bars: 45
Target Percentage: 3.0%

Test Results Summary:
================================================================================
           Model  Accuracy  F1_Score  Precision    Recall   ROC_AUC
logistic_regression    0.7234    0.6187     0.6621    0.5834    0.7892
...

Best Model: logistic_regression
Best F1 Score: 0.6187

Classification Report (Best Model):
================================================================================
              precision    recall  f1-score   support
...
```

---

## Interpreting Results

### 1. Performance Metrics

**Accuracy:**
- Overall correctness
- Good for balanced datasets
- Can be misleading for imbalanced data

**F1 Score:**
- Balance between precision and recall
- Best metric for imbalanced data
- Range: 0 to 1 (higher is better)

**Precision:**
- Of predicted "Increase", how many were correct?
- High precision = fewer false positives
- Important if false alarms are costly

**Recall:**
- Of actual "Increase", how many did we catch?
- High recall = fewer missed opportunities
- Important for not missing profitable trades

**ROC AUC:**
- Overall model discrimination ability
- Range: 0.5 (random) to 1.0 (perfect)
- Good for comparing models

### 2. Training vs Test Comparison

**Good Signs:**
- Test performance close to training (< 5% drop)
- Similar ranking of models
- Consistent best model

**Warning Signs:**
- Large performance drop (> 10%)
- Different best model on test set
- High training, low test (overfitting)

**Example:**

```
TRAINING vs TEST PERFORMANCE
================================================================================
           Model  Accuracy_Train  Accuracy_Test  F1_Score_Train  F1_Score_Test
logistic_regression          0.7234         0.7156          0.6187         0.6089
```

**Analysis:**
- Accuracy drop: 0.78% (good!)
- F1 drop: 1.58% (acceptable)
- Model generalizes well

### 3. Confusion Matrix

```
                Predicted
              No Inc  Increase
Actual No Inc   8500      500    ← 500 false positives
       Increase  350      650    ← 350 false negatives
```

**Interpretation:**
- True Negatives (8500): Correctly predicted "No Increase"
- False Positives (500): Predicted "Increase" but was "No Increase"
- False Negatives (350): Predicted "No Increase" but was "Increase"
- True Positives (650): Correctly predicted "Increase"

**For Trading:**
- False Positives = Bad trades (lose money)
- False Negatives = Missed opportunities (no profit)
- True Positives = Good trades (make money)

### 4. Model Selection

Choose model based on:

**Best F1 Score:**
- Balanced performance
- Good for most cases

**Best Recall:**
- Catch more opportunities
- Accept more false alarms
- Aggressive strategy

**Best Precision:**
- Fewer false alarms
- Miss some opportunities
- Conservative strategy

---

## Advanced Usage

### Testing Multiple Datasets

```python
# Test on multiple datasets
test_datasets = [
    'data/btc_2023.csv',
    'data/eth_2023.csv',
    'data/btc_2024.csv'
]

for dataset in test_datasets:
    print(f"\nTesting on: {dataset}")
    df_test = pd.read_csv(dataset)
    # ... rest of evaluation code
```

### Comparing Multiple Sessions

```python
# Compare last 3 sessions
for session in sessions[:3]:
    print(f"\nSession: {session}")
    # Load models from session
    # Evaluate on same test set
    # Compare results
```

### Custom Metrics

```python
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score

# Add custom metrics
mcc = matthews_corrcoef(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)

print(f"Matthews Correlation: {mcc:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
```

### Threshold Optimization

```python
# Find optimal threshold for best model
from sklearn.metrics import precision_recall_curve

y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

# Find threshold that maximizes F1
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"F1 Score at optimal: {f1_scores[optimal_idx]:.4f}")
```

---

## Troubleshooting

### Issue: "No training sessions found"

**Solution:**
- Check that `models/` directory exists
- Ensure you've trained models first
- Run `train_and_save_models.py`

### Issue: "Test data file not found"

**Solution:**
- Check test data path in cell 5
- Ensure CSV file exists
- Use absolute path if needed

### Issue: "Scaler/Model loading error"

**Solution:**
- Ensure session directory is complete
- Check for corrupted .joblib files
- Retrain if necessary

### Issue: "Feature mismatch error"

**Solution:**
- Use same `target_bars` and `target_pct` as training
- Check training config file for parameters
- Ensure test data has same structure

### Issue: "Poor test performance"

**Possible causes:**
- Overfitting (retrain with regularization)
- Different data distribution (retrain on more diverse data)
- Wrong parameters (check training config)
- Data leakage during training

---

## Best Practices

### 1. Always Test on Unseen Data

- Never test on training data
- Use data from different time period
- Ensure no data leakage

### 2. Compare Multiple Sessions

- Test several training sessions
- Identify most robust model
- Check consistency across sessions

### 3. Monitor Performance Drop

- Track training vs test gap
- Large drop indicates overfitting
- Retrain if drop > 10%

### 4. Save All Results

- Keep test reports for reference
- Track performance over time
- Document findings

### 5. Consider Multiple Metrics

- Don't rely on single metric
- Balance precision and recall
- Consider trading costs

---

## Summary

### Quick Reference

**Open notebook:**
```bash
jupyter notebook test_pretrained_models.ipynb
```

**Default behavior:**
- Loads latest training session
- Tests on `data/btc_2023.csv`
- Evaluates all models
- Saves results to session directory

**Output files:**
- `models/{session}/test_results.csv`
- `models/{session}/test_report.txt`

**Key metrics:**
- F1 Score (primary)
- Accuracy
- Precision
- Recall
- ROC AUC

### Benefits

✅ Interactive testing
✅ Visual results
✅ Automatic comparison
✅ Detailed reports
✅ Easy to customize

---

## Additional Resources

- [Model Training Documentation](../src/model_training.py)
- [Timestamped Saves Guide](TIMESTAMPED_SAVES_GUIDE.md)
- [Backtesting Guide](BACKTEST_GUIDE.md)

Happy testing! 🧪📊✅
