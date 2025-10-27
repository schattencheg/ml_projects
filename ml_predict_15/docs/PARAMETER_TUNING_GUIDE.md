# Parameter Tuning Guide - Improving Model Differentiation

## Problem: Models Have Similar Results

When all models show very similar accuracy, F1 scores, and other metrics, it indicates that:

1. **Target is too easy/hard**: All models converge to similar baseline performance
2. **Features lack discriminative power**: Models can't find meaningful patterns
3. **Class imbalance**: Majority class dominates predictions
4. **Overfitting/Underfitting**: All models behave similarly

## Key Parameters to Tune

### 1. `target_pct` - Target Percentage Change

**What it does**: Defines the minimum price increase to predict (e.g., 3% = predict if price goes up 3% or more)

**Impact on model differentiation**:

| target_pct | Difficulty | Model Differentiation | Use Case |
|------------|------------|----------------------|----------|
| 0.5% - 1.0% | Too Easy | ❌ Low - All models similar | Not recommended |
| 1.5% - 2.5% | Easy | ⚠️ Medium - Some differentiation | Quick moves |
| 3.0% - 5.0% | **Optimal** | ✅ **High - Clear differences** | **Recommended** |
| 5.0% - 10.0% | Hard | ✅ High - But fewer opportunities | Large moves |
| 10%+ | Very Hard | ⚠️ Medium - Too few samples | Rare events |

**Recommendation**: Start with **3.0%** for crypto, **2.0%** for stocks

**Example**:
```python
# Too easy - models will be similar
models, scaler, results, best_model = train(df_train, target_pct=0.5)

# Optimal - models will differentiate
models, scaler, results, best_model = train(df_train, target_pct=3.0)

# Too hard - may have too few positive samples
models, scaler, results, best_model = train(df_train, target_pct=15.0)
```

### 2. `target_bars` - Look-Ahead Period

**What it does**: How many bars (candles) to look ahead for the target

**Impact on model differentiation**:

| target_bars | Time Horizon | Model Differentiation | Use Case |
|-------------|--------------|----------------------|----------|
| 5-15 bars | Very Short | ⚠️ Medium - Noisy | Scalping |
| 15-30 bars | Short | ✅ Good | Day trading |
| 30-60 bars | **Medium** | ✅ **Best** | **Swing trading** |
| 60-120 bars | Long | ✅ Good | Position trading |
| 120+ bars | Very Long | ⚠️ Medium - Slow feedback | Long-term |

**Recommendation**: 
- Hourly data: **45-60 bars** (2-3 days)
- Daily data: **15-30 bars** (2-4 weeks)

**Example**:
```python
# Short-term (noisy)
models, scaler, results, best_model = train(df_train, target_bars=10)

# Medium-term (optimal)
models, scaler, results, best_model = train(df_train, target_bars=45)

# Long-term (slower)
models, scaler, results, best_model = train(df_train, target_bars=120)
```

### 3. `use_smote` - Handle Class Imbalance

**What it does**: Balances the dataset by creating synthetic samples of minority class

**Impact**: Can significantly improve model differentiation by preventing majority class bias

**When to use**:
- ✅ Use when class imbalance > 2:1 (e.g., 70% No Increase, 30% Increase)
- ❌ Don't use when classes are already balanced

**Example**:
```python
# Without SMOTE - models may all predict majority class
models, scaler, results, best_model = train(df_train, use_smote=False)

# With SMOTE - models can differentiate better
models, scaler, results, best_model = train(df_train, use_smote=True)
```

## Recommended Configurations

### Configuration 1: Optimal Differentiation (Recommended)

```python
models, scaler, results, best_model = train(
    df_train,
    target_bars=45,      # Medium-term (2-3 days for hourly)
    target_pct=3.0,      # Meaningful price change
    use_smote=True,      # Handle imbalance
    use_gpu=False,       # Hardware acceleration
    n_jobs=-1            # Use all CPU cores
)
```

**Expected Results**:
- Clear differentiation between models
- Accuracy: 60-75% (varies by model)
- F1 Score: 0.45-0.65 (varies by model)
- Best models: XGBoost, LightGBM, Random Forest

### Configuration 2: More Aggressive (Harder Target)

```python
models, scaler, results, best_model = train(
    df_train,
    target_bars=60,      # Longer horizon
    target_pct=5.0,      # Larger price change
    use_smote=True,
    use_gpu=False,
    n_jobs=-1
)
```

**Expected Results**:
- Very clear differentiation
- Accuracy: 55-70% (wider range)
- Fewer positive samples (harder problem)
- Tree-based models (XGBoost, LightGBM) perform best

### Configuration 3: Conservative (Easier Target)

```python
models, scaler, results, best_model = train(
    df_train,
    target_bars=30,      # Shorter horizon
    target_pct=2.0,      # Smaller price change
    use_smote=True,
    use_gpu=False,
    n_jobs=-1
)
```

**Expected Results**:
- Moderate differentiation
- Accuracy: 65-80% (narrower range)
- More positive samples (easier problem)
- Linear models may perform better

## Diagnosing Similar Results

### Check 1: Class Distribution

Look at the training output:
```
Target distribution:
0    8500  # No Increase
1    1500  # Increase
```

**Problem**: If ratio > 5:1, models may all predict majority class

**Solution**: 
- Enable SMOTE: `use_smote=True`
- Increase `target_pct` to balance classes
- Use class weights (already enabled by default)

### Check 2: Baseline Accuracy

If all models have accuracy ~85-90%:

**Problem**: Models are predicting majority class (e.g., always "No Increase")

**Solution**:
- Check F1 Score and Recall (should be > 0.4)
- Increase `target_pct` to 3-5%
- Enable SMOTE
- Check threshold optimization is working

### Check 3: Feature Importance

If models have similar results, features might not be discriminative:

**Check**: Look at feature importance in XGBoost/LightGBM
```python
import matplotlib.pyplot as plt
import xgboost as xgb

# After training
model = models['xgboost'][0]
xgb.plot_importance(model, max_num_features=20)
plt.show()
```

**Problem**: If all features have low importance, features aren't useful

**Solution**:
- Add more technical indicators (RSI, MACD, Bollinger Bands)
- Add price action features (higher highs, lower lows)
- Add volume features
- Add market regime indicators

### Check 4: Overfitting

If training accuracy is high but test accuracy is low:

**Problem**: Models are overfitting

**Solution**:
- Reduce model complexity (lower max_depth, n_estimators)
- Add regularization
- Use more training data
- Simplify features

## Expected Model Performance Ranges

### Good Differentiation (Target: 3% in 45 bars)

| Model | Accuracy | F1 Score | Notes |
|-------|----------|----------|-------|
| Logistic Regression | 62-68% | 0.45-0.55 | Baseline |
| Ridge Classifier | 63-69% | 0.46-0.56 | Similar to Logistic |
| Naive Bayes | 60-65% | 0.42-0.52 | Fast but less accurate |
| Decision Tree | 58-64% | 0.40-0.50 | Prone to overfitting |
| Random Forest | 68-74% | 0.52-0.62 | Good ensemble |
| XGBoost | 70-76% | 0.55-0.65 | Usually best |
| LightGBM | 69-75% | 0.54-0.64 | Fast and accurate |

**Spread**: 10-15% difference between best and worst

### Poor Differentiation (Target: 0.5% in 45 bars)

| Model | Accuracy | F1 Score | Notes |
|-------|----------|----------|-------|
| All Models | 85-87% | 0.35-0.40 | Too similar! |

**Spread**: Only 2-3% difference - **Problem!**

## Quick Fixes

### Fix 1: Increase target_pct

```python
# Before (poor differentiation)
train(df_train, target_pct=1.0)

# After (good differentiation)
train(df_train, target_pct=3.0)
```

### Fix 2: Enable SMOTE

```python
# Before (biased to majority class)
train(df_train, use_smote=False)

# After (balanced predictions)
train(df_train, use_smote=True)
```

### Fix 3: Adjust Look-Ahead Period

```python
# Before (too short, noisy)
train(df_train, target_bars=10)

# After (optimal)
train(df_train, target_bars=45)
```

### Fix 4: Check Threshold Optimization

The code automatically optimizes decision thresholds. Look for this in output:
```
Threshold Optimization:
  Default threshold (0.5): F1=0.45, Recall=0.42
  Optimal threshold (0.35): F1=0.58, Recall=0.65
  Improvement: +28.9% F1, +54.8% Recall
```

If you don't see improvement, threshold optimization might not be helping.

## Advanced Tuning

### Experiment with Different Targets

Try multiple configurations and compare:

```python
# Configuration A: Conservative
results_a = train(df_train, target_bars=30, target_pct=2.0)

# Configuration B: Balanced
results_b = train(df_train, target_bars=45, target_pct=3.0)

# Configuration C: Aggressive
results_c = train(df_train, target_bars=60, target_pct=5.0)

# Compare results
# Look for configuration with:
# 1. Good model differentiation (10-15% accuracy spread)
# 2. Reasonable F1 scores (> 0.5)
# 3. Balanced precision/recall
```

### Grid Search for Optimal Parameters

```python
import pandas as pd

results_grid = []

for target_bars in [30, 45, 60]:
    for target_pct in [2.0, 3.0, 4.0, 5.0]:
        print(f"\nTesting: bars={target_bars}, pct={target_pct}")
        
        models, scaler, results, best_model = train(
            df_train,
            target_bars=target_bars,
            target_pct=target_pct,
            use_smote=True
        )
        
        # Calculate spread (differentiation)
        accuracies = [r['accuracy'] for r in results.values()]
        spread = max(accuracies) - min(accuracies)
        
        results_grid.append({
            'target_bars': target_bars,
            'target_pct': target_pct,
            'best_accuracy': max(accuracies),
            'worst_accuracy': min(accuracies),
            'spread': spread,
            'best_model': best_model
        })

# Find configuration with best spread
df_grid = pd.DataFrame(results_grid)
df_grid = df_grid.sort_values('spread', ascending=False)
print("\nBest configurations (by model differentiation):")
print(df_grid.head())
```

## Summary

**Main Causes of Similar Results**:
1. ❌ `target_pct` too small (< 2%)
2. ❌ Class imbalance not handled
3. ❌ Features not discriminative
4. ❌ All models predicting majority class

**Quick Solutions**:
1. ✅ Increase `target_pct` to 3-5%
2. ✅ Enable `use_smote=True`
3. ✅ Use optimal `target_bars` (45-60 for hourly)
4. ✅ Check F1 score and recall, not just accuracy

**Recommended Starting Point**:
```python
models, scaler, results, best_model = train(
    df_train,
    target_bars=45,
    target_pct=3.0,
    use_smote=True,
    use_gpu=False,
    n_jobs=-1
)
```

This should give you **clear differentiation** between models with accuracy spread of 10-15% and F1 scores ranging from 0.45 to 0.65.
