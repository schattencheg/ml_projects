# Model Configuration System - Summary

## Overview

Added a centralized `MODEL_ENABLED_CONFIG` dictionary to control which ML models are trained, making it easy to enable/disable models without modifying code throughout the project.

## What Was Added

### 1. MODEL_ENABLED_CONFIG Dictionary

**Location**: `src/model_training.py` (line ~116)

```python
# Model enabled/disabled configuration
# Set to False to disable a model from training
# Current config: Only fast models (<10 min training time)
MODEL_ENABLED_CONFIG = {
    'logistic_regression': True,      # Fast: ~2-5 seconds
    'ridge_classifier': True,         # Fast: ~2-5 seconds
    'naive_bayes': True,              # Fast: ~3-5 seconds
    'knn_k_neighbours': False,        # SLOW: 50-120 seconds (disabled)
    'decision_tree': True,            # Fast: ~5-10 seconds
    'random_forest': True,            # Medium: ~15-30 seconds (acceptable)
    'gradient_boosting': False,       # SLOW: 60+ seconds (disabled, use XGBoost)
    'svm_support_vector_classification': False,  # VERY SLOW: 100+ seconds (disabled)
    'xgboost': True,                  # Fast: ~3-5 seconds (optimized)
    'lightgbm': True,                 # Fast: ~3-5 seconds (optimized)
    'lstm': False,                    # SLOW: Neural network (disabled)
    'cnn': False,                     # SLOW: Neural network (disabled)
    'hybrid_lstm_cnn': False,         # SLOW: Neural network (disabled)
}
```

### 2. Updated Model Definitions

All model tuples now use the configuration dictionary for their third parameter:

**Before**:
```python
"logistic_regression": (
    LogisticRegression(...),
    {...},
    True  # Hardcoded
)
```

**After**:
```python
"logistic_regression": (
    LogisticRegression(...),
    {...},
    MODEL_ENABLED_CONFIG.get('logistic_regression', True)  # From config
)
```

### 3. Improved MLflow Logging

Enhanced MLflow tracking with better error handling and debugging:

```python
# Check if MLflow run is active
active_run = mlflow.active_run()
if active_run is None:
    print(f"  ✗ MLflow: No active run found for {model_name}")
else:
    # Log model-specific metrics
    mlflow.log_metric(f"{model_name}_accuracy", float(result['accuracy']))
    mlflow.log_metric(f"{model_name}_f1_score", float(result['f1_score']))
    # ... more metrics
    print(f"  ✓ MLflow: Logged metrics for {model_name}")
```

### 4. Example Files

- **example_model_config.py** - Examples of different configurations
- **MLFLOW_TROUBLESHOOTING.md** - Complete troubleshooting guide

## Current Configuration

### ✅ Enabled Models (7 total - Fast models only)

| Model | Training Time | Notes |
|-------|--------------|-------|
| Logistic Regression | ~2-5 sec | Linear model, very fast |
| Ridge Classifier | ~2-5 sec | Regularized linear model |
| Naive Bayes | ~3-5 sec | Probabilistic model |
| Decision Tree | ~5-10 sec | Single tree, fast |
| Random Forest | ~15-30 sec | Ensemble, acceptable speed |
| XGBoost | ~3-5 sec | Optimized gradient boosting |
| LightGBM | ~3-5 sec | Fast gradient boosting |

**Expected Total Training Time**: ~40-70 seconds

### ❌ Disabled Models (6 total - Slow models)

| Model | Training Time | Reason Disabled |
|-------|--------------|-----------------|
| KNN | 50-120 sec | Too slow, memory intensive |
| Gradient Boosting | 60+ sec | Slow, use XGBoost instead |
| SVM | 100+ sec | Very slow on large datasets |
| LSTM | Variable | Neural network, slow training |
| CNN | Variable | Neural network, slow training |
| Hybrid LSTM-CNN | Variable | Neural network, very slow |

## How to Use

### Default Usage (No Changes Needed)

```python
from src.model_training import train

# Train with current configuration (only enabled models)
models, scaler, results, best_model = train(df_train)
```

### View Current Configuration

```python
from src.model_training import MODEL_ENABLED_CONFIG

print("Enabled models:")
for model_name, enabled in MODEL_ENABLED_CONFIG.items():
    if enabled:
        print(f"  ✓ {model_name}")

print("\nDisabled models:")
for model_name, enabled in MODEL_ENABLED_CONFIG.items():
    if not enabled:
        print(f"  ✗ {model_name}")
```

### Modify Configuration

**Option 1: Edit the dictionary directly** (Recommended)

Open `src/model_training.py` and edit `MODEL_ENABLED_CONFIG`:

```python
MODEL_ENABLED_CONFIG = {
    'logistic_regression': True,   # Keep enabled
    'random_forest': False,        # Disable this model
    'xgboost': True,              # Keep enabled
    # ... etc
}
```

**Option 2: Programmatic modification** (Advanced)

```python
from src import model_training

# Disable a specific model
model_training.MODEL_ENABLED_CONFIG['random_forest'] = False

# Enable a specific model
model_training.MODEL_ENABLED_CONFIG['knn_k_neighbours'] = True

# Train with modified config
models, scaler, results, best_model = model_training.train(df_train)
```

## Common Configurations

### 1. Only Fast Models (Current - Default)
```python
MODEL_ENABLED_CONFIG = {
    'logistic_regression': True,
    'ridge_classifier': True,
    'naive_bayes': True,
    'knn_k_neighbours': False,
    'decision_tree': True,
    'random_forest': True,
    'gradient_boosting': False,
    'svm_support_vector_classification': False,
    'xgboost': True,
    'lightgbm': True,
    'lstm': False,
    'cnn': False,
    'hybrid_lstm_cnn': False,
}
```
**Training Time**: ~40-70 seconds
**Models**: 7 enabled

### 2. Only Tree-Based Models
```python
MODEL_ENABLED_CONFIG = {
    'logistic_regression': False,
    'ridge_classifier': False,
    'naive_bayes': False,
    'knn_k_neighbours': False,
    'decision_tree': True,
    'random_forest': True,
    'gradient_boosting': True,
    'svm_support_vector_classification': False,
    'xgboost': True,
    'lightgbm': True,
    'lstm': False,
    'cnn': False,
    'hybrid_lstm_cnn': False,
}
```
**Training Time**: ~90-120 seconds
**Models**: 5 enabled

### 3. Only Neural Networks
```python
MODEL_ENABLED_CONFIG = {
    'logistic_regression': False,
    'ridge_classifier': False,
    'naive_bayes': False,
    'knn_k_neighbours': False,
    'decision_tree': False,
    'random_forest': False,
    'gradient_boosting': False,
    'svm_support_vector_classification': False,
    'xgboost': False,
    'lightgbm': False,
    'lstm': True,
    'cnn': True,
    'hybrid_lstm_cnn': True,
}
```
**Training Time**: Variable (depends on dataset)
**Models**: 3 enabled

### 4. Single Model (Testing)
```python
MODEL_ENABLED_CONFIG = {
    'logistic_regression': False,
    'ridge_classifier': False,
    'naive_bayes': False,
    'knn_k_neighbours': False,
    'decision_tree': False,
    'random_forest': False,
    'gradient_boosting': False,
    'svm_support_vector_classification': False,
    'xgboost': True,  # Only this one
    'lightgbm': False,
    'lstm': False,
    'cnn': False,
    'hybrid_lstm_cnn': False,
}
```
**Training Time**: ~3-5 seconds
**Models**: 1 enabled

### 5. All Models (Complete Comparison)
```python
MODEL_ENABLED_CONFIG = {
    'logistic_regression': True,
    'ridge_classifier': True,
    'naive_bayes': True,
    'knn_k_neighbours': True,
    'decision_tree': True,
    'random_forest': True,
    'gradient_boosting': True,
    'svm_support_vector_classification': True,
    'xgboost': True,
    'lightgbm': True,
    'lstm': True,
    'cnn': True,
    'hybrid_lstm_cnn': True,
}
```
**Training Time**: 5-10+ minutes (depending on dataset)
**Models**: 13 enabled

## Training Output

### With Enabled/Disabled Models

```
================================================================================
TRAINING AND EVALUATING MODELS
================================================================================
Total models available: 13
Enabled models: 7
Disabled models: knn_k_neighbours, gradient_boosting, svm_support_vector_classification, lstm, cnn, hybrid_lstm_cnn

Training Progress: |████████████████████| 7/7 [00:45<00:00]

================================================================================
Model: LOGISTIC_REGRESSION
================================================================================
Training logistic_regression... ✓ Completed in 2.34 seconds
  ✓ MLflow: Logged metrics for logistic_regression

[... more models ...]

================================================================================
TRAINING COMPLETE
================================================================================
Total training time: 45.67 seconds (0.76 minutes)
Average time per model: 6.52 seconds
```

## MLflow Integration

### What Gets Tracked

For **each enabled model**:
- `{model_name}_accuracy`
- `{model_name}_f1_score`
- `{model_name}_precision`
- `{model_name}_recall`
- `{model_name}_roc_auc`
- `{model_name}_training_time`

**Disabled models are NOT tracked** (they're not trained).

### MLflow Logging Messages

Success:
```
  ✓ MLflow: Logged metrics for logistic_regression
```

No active run:
```
  ✗ MLflow: No active run found for ridge_classifier
```

Error:
```
  ✗ MLflow: Failed to log metrics for xgboost
     Error: ConnectionError: Connection refused
```

## Benefits

### 1. Easy Model Selection
- Enable/disable models with a single boolean change
- No need to comment out code or modify multiple locations

### 2. Faster Experimentation
- Quickly test different model combinations
- Focus on fast models during development
- Enable all models for final comparison

### 3. Better Resource Management
- Disable slow models to save time
- Reduce memory usage by training fewer models
- Optimize for your hardware constraints

### 4. Clear Documentation
- Configuration is self-documenting with comments
- Easy to see which models are enabled/disabled
- Training time estimates included

### 5. MLflow Tracking
- Only enabled models are tracked in MLflow
- Cleaner experiment tracking
- Better debugging with detailed logging

## Best Practices

### 1. Start with Fast Models
Use the default configuration (fast models only) for initial development and testing.

### 2. Document Your Changes
Add comments when changing configuration:
```python
MODEL_ENABLED_CONFIG = {
    'random_forest': False,  # Disabled: too slow for current dataset
    'xgboost': True,         # Enabled: best performance
}
```

### 3. Create Configuration Profiles
Save different configurations for different purposes:
- Development: Fast models only
- Testing: Medium-speed models
- Production: All models for comparison

### 4. Monitor Training Time
Watch the training output to verify expected times:
```
Total training time: 45.67 seconds (0.76 minutes)
Average time per model: 6.52 seconds
```

### 5. Check MLflow Logs
Verify all enabled models are tracked:
```
  ✓ MLflow: Logged metrics for logistic_regression
  ✓ MLflow: Logged metrics for xgboost
```

## Troubleshooting

### Issue: Models not training
**Check**: `MODEL_ENABLED_CONFIG` - ensure models are set to `True`

### Issue: Training too slow
**Solution**: Disable slow models (KNN, SVM, Gradient Boosting, Neural Networks)

### Issue: Models not in MLflow
**Check**: 
1. MLflow server running (http://localhost:5000)
2. Look for `✓ MLflow: Logged metrics` messages
3. See `MLFLOW_TROUBLESHOOTING.md` for detailed help

### Issue: Want to train all models
**Solution**: Set all models to `True` in `MODEL_ENABLED_CONFIG`

## Files Modified

1. **src/model_training.py**
   - Added `MODEL_ENABLED_CONFIG` dictionary
   - Updated all model definitions to use config
   - Improved MLflow logging with error handling
   - Added active run checks

## Files Created

1. **example_model_config.py** - Configuration examples
2. **MLFLOW_TROUBLESHOOTING.md** - MLflow debugging guide
3. **MODEL_CONFIG_SUMMARY.md** - This file

## Summary

The `MODEL_ENABLED_CONFIG` dictionary provides a simple, centralized way to control which models are trained. The current configuration enables only fast models (<10 min training time), resulting in ~40-70 second training sessions. All enabled models are automatically tracked in MLflow with improved error handling and debugging.

**Quick Start**: The default configuration is ready to use - just run `train(df_train)` and only fast models will be trained!
