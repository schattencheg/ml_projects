# MLflow Troubleshooting Guide

## Current Setup

The MLflow server runs as a **standalone application** (separate from this project). The `ml_predict_15` project connects to it as a client.

### Architecture

```
┌─────────────────────────┐         ┌─────────────────────────┐
│  MLflow Server          │         │  ml_predict_15          │
│  (Standalone App)       │◄────────│  (Client)               │
│  Port: 5000             │  HTTP   │  src/model_training.py  │
│  http://localhost:5000  │         │                         │
└─────────────────────────┘         └─────────────────────────┘
```

## Issue: Only Some Models Being Tracked

### Symptoms
- Only one model (e.g., Naive Bayes) appears in MLflow
- Other models are trained but not logged to MLflow

### Possible Causes & Solutions

#### 1. **MLflow Run Context Lost**
**Problem**: The MLflow run might be ending prematurely or losing context between models.

**Check**: Look for these messages in training output:
```
✓ MLflow: Logged metrics for logistic_regression
✓ MLflow: Logged metrics for ridge_classifier
✗ MLflow: No active run found for decision_tree  ← Problem!
```

**Solution**: The code now checks for active run before logging each model.

#### 2. **Metric Type Issues**
**Problem**: Some metrics might not be proper Python floats.

**Solution**: Already fixed - all metrics are now explicitly converted to `float()`:
```python
mlflow.log_metric(f"{model_name}_accuracy", float(result['accuracy']))
```

#### 3. **MLflow Server Connection Issues**
**Problem**: Connection to MLflow server might be intermittent.

**Check**:
```python
# Test connection
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
print(mlflow.get_tracking_uri())  # Should print: http://localhost:5000
```

**Verify server is running**:
- Open browser: http://localhost:5000
- Should see MLflow UI

#### 4. **Disabled Models Not Tracked**
**Problem**: Disabled models in `MODEL_ENABLED_CONFIG` won't be trained or tracked.

**Check**: Review `MODEL_ENABLED_CONFIG` in `src/model_training.py`:
```python
MODEL_ENABLED_CONFIG = {
    'logistic_regression': True,   # ✓ Will be tracked
    'knn_k_neighbours': False,     # ✗ Won't be tracked (disabled)
    ...
}
```

**Currently Enabled Models** (fast models only):
- ✓ Logistic Regression
- ✓ Ridge Classifier
- ✓ Naive Bayes
- ✓ Decision Tree
- ✓ Random Forest
- ✓ XGBoost
- ✓ LightGBM

**Currently Disabled Models** (slow models):
- ✗ KNN
- ✗ Gradient Boosting
- ✗ SVM
- ✗ LSTM
- ✗ CNN
- ✗ Hybrid LSTM-CNN

## Debugging Steps

### Step 1: Check Training Output
Look for MLflow logging messages after each model:
```
Training logistic_regression... ✓ Completed in 2.34 seconds
  ✓ MLflow: Logged metrics for logistic_regression  ← Success!

Training ridge_classifier... ✓ Completed in 2.45 seconds
  ✗ MLflow: Failed to log metrics for ridge_classifier  ← Problem!
     Error: ValueError: Invalid metric value
```

### Step 2: Verify MLflow Server
```bash
# Check if server is running
curl http://localhost:5000/health

# Or open in browser
http://localhost:5000
```

### Step 3: Check MLflow Run Status
After training, check the MLflow UI:
1. Open: http://localhost:5000
2. Navigate to experiment: `ml_predict_15/classification/crypto_price_prediction`
3. Click on the latest run
4. Check "Metrics" tab - should see all enabled models

### Step 4: Manual Test
Test MLflow logging manually:
```python
import mlflow

# Connect to server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("test_experiment")

# Start run
with mlflow.start_run(run_name="test_run"):
    # Log test metrics
    mlflow.log_metric("test_metric_1", 0.95)
    mlflow.log_metric("test_metric_2", 0.87)
    print("✓ Test metrics logged successfully")

# Check in MLflow UI: http://localhost:5000
```

## Expected Behavior

### During Training
You should see these messages for **each enabled model**:
```
================================================================================
MLFLOW TRACKING ENABLED
================================================================================
Tracking URI: http://localhost:5000
Experiment: ml_predict_15/classification/crypto_price_prediction
Run: training_20251027_090745
Run ID: abc123def456...
================================================================================

Training Progress: |████████| 7/7 [00:45<00:00]

Training logistic_regression... ✓ Completed in 2.34 seconds
  ✓ MLflow: Logged metrics for logistic_regression

Training ridge_classifier... ✓ Completed in 2.45 seconds
  ✓ MLflow: Logged metrics for ridge_classifier

Training naive_bayes... ✓ Completed in 3.12 seconds
  ✓ MLflow: Logged metrics for naive_bayes

Training decision_tree... ✓ Completed in 5.67 seconds
  ✓ MLflow: Logged metrics for decision_tree

Training random_forest... ✓ Completed in 18.23 seconds
  ✓ MLflow: Logged metrics for random_forest

Training xgboost... ✓ Completed in 3.45 seconds
  ✓ MLflow: Logged metrics for xgboost

Training lightgbm... ✓ Completed in 3.21 seconds
  ✓ MLflow: Logged metrics for lightgbm

✓ Best model logged to MLflow: xgboost
✓ Artifacts logged to MLflow

================================================================================
MLFLOW TRACKING COMPLETE
================================================================================
View results at: http://localhost:5000
Run ID: abc123def456...
================================================================================
```

### In MLflow UI
For each run, you should see:

**Parameters** (12 total):
- target_bars, target_pct
- use_smote, use_gpu, n_jobs
- dataset_shape, train_size, val_size
- class_imbalance_ratio, smote_applied
- best_model_name, num_models_trained

**Metrics** (per enabled model + best model):
- {model_name}_accuracy
- {model_name}_f1_score
- {model_name}_precision
- {model_name}_recall
- {model_name}_roc_auc
- {model_name}_training_time
- best_accuracy, best_f1_score, etc.
- total_training_time, avg_training_time

**Artifacts**:
- best_model/ (sklearn model)
- results/training_results_summary.csv
- config/training_config.txt
- plots/model_comparison_training.png

## Common Issues

### Issue: "No active run found"
**Cause**: MLflow run ended prematurely
**Fix**: Already implemented - code now checks for active run

### Issue: "Connection refused"
**Cause**: MLflow server not running
**Fix**: Start MLflow server (standalone application)

### Issue: "Only last model tracked"
**Cause**: Metrics being overwritten instead of accumulated
**Fix**: Each model uses unique metric names: `{model_name}_accuracy`

### Issue: "No metrics in MLflow UI"
**Cause**: Metrics logged but not visible
**Fix**: Refresh MLflow UI, check correct experiment/run

## Configuration

### Change MLflow Server URI
Edit `train()` function call:
```python
models, scaler, results, best_model = train(
    df_train,
    mlflow_tracking_uri="http://localhost:8080"  # Custom port
)
```

### Disable MLflow Tracking
```python
models, scaler, results, best_model = train(
    df_train,
    use_mlflow=False  # Disable MLflow
)
```

### Enable/Disable Specific Models
Edit `MODEL_ENABLED_CONFIG` in `src/model_training.py`:
```python
MODEL_ENABLED_CONFIG = {
    'logistic_regression': True,   # Enable
    'random_forest': False,        # Disable
    ...
}
```

## Contact & Support

If issues persist:
1. Check training output for `✗ MLflow:` error messages
2. Verify MLflow server is accessible: http://localhost:5000
3. Check MLflow server logs for errors
4. Review `MODEL_ENABLED_CONFIG` to ensure models are enabled

## Quick Reference

**MLflow Server**: Standalone application (not started by this project)
**Server URL**: http://localhost:5000
**Experiment**: ml_predict_15/classification/crypto_price_prediction
**Client**: src/model_training.py (connects to server)
**Enabled Models**: 7 fast models (see MODEL_ENABLED_CONFIG)
**Disabled Models**: 6 slow models (KNN, GB, SVM, neural networks)
