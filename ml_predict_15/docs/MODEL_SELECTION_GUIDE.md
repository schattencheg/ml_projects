# Model Selection Guide

This guide explains how to enable or disable specific models during training.

## Overview

Each model in the training pipeline can be individually enabled or disabled. This is useful for:
- **Faster training** - Skip slow models
- **Experimentation** - Test specific model combinations
- **Resource management** - Disable models that require unavailable resources
- **Production deployment** - Train only models that perform well

---

## How It Works

### Model Configuration Structure

Each model is defined as a tuple with 3 elements:

```python
"model_name": (
    model_instance,      # The scikit-learn/XGBoost/LightGBM model
    params_dict,         # Dictionary of model parameters
    enabled_flag         # True = enabled, False = disabled
)
```

### Example

```python
"logistic_regression": (
    LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', n_jobs=n_jobs),
    {"max_iter": 1000, "class_weight": "balanced", "n_jobs": n_jobs},
    True  # Enabled - this model will be trained
),

"gradient_boosting": (
    GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    {"n_estimators": 100, "max_depth": 5},
    False  # Disabled - this model will be skipped
),
```

---

## Enabling/Disabling Models

### Location

Edit the `get_model_configs()` function in `src/model_training.py` (around line 135).

### Steps

1. **Open the file:**
   ```
   src/model_training.py
   ```

2. **Find the model you want to enable/disable:**
   ```python
   "model_name": (
       ModelClass(...),
       {...},
       True  # or False
   ),
   ```

3. **Change the enabled flag:**
   - `True` = Model will be trained
   - `False` = Model will be skipped

4. **Save the file**

5. **Retrain models** - The changes will take effect on next training run

---

## Available Models

### Currently Enabled (Default)

| Model | Speed | Accuracy | Notes |
|-------|-------|----------|-------|
| **logistic_regression** | Fast | Good | Best for linear relationships |
| **ridge_classifier** | Fast | Good | Similar to logistic regression |
| **naive_bayes** | Very Fast | Moderate | Assumes feature independence |
| **knn_k_neighbours** | Slow | Moderate | Memory-intensive |
| **decision_tree** | Fast | Moderate | Prone to overfitting |
| **random_forest** | Moderate | Good | Ensemble of decision trees |
| **xgboost** | Fast (GPU) | Excellent | Best overall performance |
| **lightgbm** | Fast (GPU) | Excellent | Similar to XGBoost |

### Currently Disabled (Default)

| Model | Speed | Accuracy | Why Disabled |
|-------|-------|----------|--------------|
| **gradient_boosting** | Very Slow | Good | XGBoost is faster and better |
| **svm_support_vector_classification** | Extremely Slow | Good | Too slow for large datasets |

---

## Common Scenarios

### Scenario 1: Fast Training (Minimal Models)

Enable only the fastest models:

```python
"logistic_regression": (..., True),   # ✓ Keep
"ridge_classifier": (..., False),     # ✗ Disable
"naive_bayes": (..., True),           # ✓ Keep
"knn_k_neighbours": (..., False),     # ✗ Disable (slow)
"decision_tree": (..., False),        # ✗ Disable
"random_forest": (..., False),        # ✗ Disable
"xgboost": (..., True),               # ✓ Keep (best)
"lightgbm": (..., True),              # ✓ Keep (best)
```

**Result:** ~3 models, training time: ~10 seconds

### Scenario 2: Best Performance (Quality Over Speed)

Enable models with best accuracy:

```python
"logistic_regression": (..., True),   # ✓ Keep
"ridge_classifier": (..., True),      # ✓ Keep
"naive_bayes": (..., False),          # ✗ Disable (lower accuracy)
"knn_k_neighbours": (..., False),     # ✗ Disable (slow + moderate accuracy)
"decision_tree": (..., False),        # ✗ Disable (overfits)
"random_forest": (..., True),         # ✓ Keep
"xgboost": (..., True),               # ✓ Keep (best)
"lightgbm": (..., True),              # ✓ Keep (best)
```

**Result:** ~5 models, training time: ~30 seconds

### Scenario 3: Comprehensive Testing (All Models)

Enable everything for comparison:

```python
"logistic_regression": (..., True),
"ridge_classifier": (..., True),
"naive_bayes": (..., True),
"knn_k_neighbours": (..., True),
"decision_tree": (..., True),
"random_forest": (..., True),
"gradient_boosting": (..., True),     # Enable
"svm_support_vector_classification": (..., True),  # Enable (warning: very slow!)
"xgboost": (..., True),
"lightgbm": (..., True),
```

**Result:** ~10 models, training time: 5-10 minutes (depending on dataset size)

### Scenario 4: Production Deployment (Best Model Only)

After identifying the best model, train only that one:

```python
"logistic_regression": (..., False),
"ridge_classifier": (..., False),
"naive_bayes": (..., False),
"knn_k_neighbours": (..., False),
"decision_tree": (..., False),
"random_forest": (..., False),
"xgboost": (..., True),               # ✓ Only the best model
"lightgbm": (..., False),
```

**Result:** 1 model, training time: ~3 seconds

---

## Output During Training

When you run training, you'll see which models are enabled/disabled:

```
================================================================================
TRAINING AND EVALUATING MODELS
================================================================================
Total models available: 10
Enabled models: 8
Disabled models: gradient_boosting, svm_support_vector_classification

Training Progress: |████████████████████| 8/8 [00:45<00:00]
```

---

## Model Performance Comparison

Based on typical cryptocurrency price prediction:

### Speed (100K samples, 50 features)

| Model | Training Time | Speedup with GPU |
|-------|---------------|------------------|
| naive_bayes | 0.5s | N/A |
| logistic_regression | 2.3s | N/A |
| ridge_classifier | 1.8s | N/A |
| decision_tree | 3.1s | N/A |
| xgboost | 3.2s | 0.2s (16x faster) |
| lightgbm | 2.9s | 0.3s (10x faster) |
| random_forest | 15.7s | N/A |
| knn_k_neighbours | 51.9s | N/A |
| gradient_boosting | 89.3s | N/A |
| svm | 245.6s | N/A |

### Accuracy (Typical Results)

| Model | Accuracy | F1 Score | Notes |
|-------|----------|----------|-------|
| xgboost | 0.72-0.75 | 0.60-0.65 | Best overall |
| lightgbm | 0.71-0.74 | 0.59-0.64 | Very close to XGBoost |
| random_forest | 0.70-0.73 | 0.58-0.62 | Good ensemble |
| logistic_regression | 0.68-0.71 | 0.55-0.60 | Fast baseline |
| ridge_classifier | 0.67-0.70 | 0.54-0.59 | Similar to logistic |
| gradient_boosting | 0.69-0.72 | 0.57-0.61 | Slower than XGBoost |
| decision_tree | 0.63-0.67 | 0.48-0.53 | Overfits easily |
| knn_k_neighbours | 0.62-0.66 | 0.47-0.52 | Slow, moderate accuracy |
| naive_bayes | 0.60-0.64 | 0.45-0.50 | Fast but less accurate |
| svm | 0.68-0.71 | 0.55-0.60 | Too slow |

---

## Recommendations

### For Development/Experimentation

Enable 5-6 diverse models:
- ✓ logistic_regression (fast baseline)
- ✓ naive_bayes (very fast baseline)
- ✓ random_forest (ensemble)
- ✓ xgboost (best performance)
- ✓ lightgbm (best performance)
- ✗ Others (disable for speed)

### For Production

Enable only the best 1-2 models:
- ✓ xgboost (best overall)
- ✓ lightgbm (backup/comparison)
- ✗ All others (disable)

### For Research/Comparison

Enable all models to find the best:
- ✓ Enable everything
- Run comprehensive comparison
- Identify best performer
- Then disable others for production

---

## Best Practices

### 1. Start with Default Configuration

The default enabled/disabled settings are optimized for most use cases:
- Fast models: Enabled
- Accurate models: Enabled
- Slow models: Disabled
- Redundant models: Disabled

### 2. Disable Slow Models First

If training takes too long:
1. Disable `knn_k_neighbours` (very slow)
2. Disable `random_forest` if still too slow
3. Keep `xgboost` and `lightgbm` (best performance)

### 3. Enable GPU for Speed

If you have a GPU:
```python
models, scaler, results, best_model = train(df_train, use_gpu=True)
```

This makes `xgboost` and `lightgbm` 10-50x faster!

### 4. Compare Before Disabling

Before permanently disabling a model:
1. Train with it enabled once
2. Check its performance in results
3. If it performs poorly, disable it
4. If it performs well, keep it enabled

### 5. Document Your Changes

Add comments explaining why you enabled/disabled models:

```python
"gradient_boosting": (
    GradientBoostingClassifier(...),
    {...},
    False  # Disabled: XGBoost is faster and more accurate
),
```

---

## Troubleshooting

### Issue: Model not training even though enabled

**Solution:** Check that the model tuple has exactly 3 elements:
```python
"model_name": (
    ModelClass(...),  # Element 1
    {...},            # Element 2
    True              # Element 3 - must be present!
),
```

### Issue: All models disabled by accident

**Solution:** Check that at least one model has `True` as the third element.

### Issue: Training fails after enabling a model

**Possible causes:**
- Model requires library not installed (e.g., XGBoost, LightGBM)
- Model parameters incompatible with data
- Insufficient memory for model

**Solution:** Check error message and install required libraries or adjust parameters.

---

## Summary

### Quick Reference

**Enable a model:**
```python
"model_name": (..., True),
```

**Disable a model:**
```python
"model_name": (..., False),
```

**Check which models are enabled:**
Look for this output during training:
```
Enabled models: 8
Disabled models: gradient_boosting, svm_support_vector_classification
```

### Default Configuration

✅ **Enabled (8 models):**
- logistic_regression
- ridge_classifier
- naive_bayes
- knn_k_neighbours
- decision_tree
- random_forest
- xgboost
- lightgbm

❌ **Disabled (2 models):**
- gradient_boosting (slow, XGBoost is better)
- svm_support_vector_classification (very slow)

---

## Additional Resources

- [Model Training Documentation](../src/model_training.py)
- [Hardware Acceleration Guide](HARDWARE_ACCELERATION_GUIDE.md)
- [Progress Tracking Guide](PROGRESS_TRACKING_GUIDE.md)

Happy model training! 🚀🤖✅
