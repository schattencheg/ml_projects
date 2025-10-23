# Handling Imbalanced Data - Improving "Increase" Target Detection

This guide explains the techniques implemented to improve the detection of the "Increase" target class in imbalanced datasets.

## Table of Contents

1. [Problem Overview](#problem-overview)
2. [Solutions Implemented](#solutions-implemented)
3. [How to Use](#how-to-use)
4. [Understanding the Techniques](#understanding-the-techniques)
5. [Expected Improvements](#expected-improvements)
6. [Troubleshooting](#troubleshooting)

---

## Problem Overview

### The Imbalanced Data Challenge

In financial time series prediction, the "Increase" class (when price goes up by a certain percentage) is often much rarer than the "No Increase" class. This creates an **imbalanced dataset** problem.

**Typical Class Distribution:**
```
No Increase (0): 85-95% of data
Increase (1):     5-15% of data
```

**Why This Is a Problem:**
- Models learn to predict the majority class (No Increase)
- Minority class (Increase) gets poor accuracy
- Low recall for "Increase" means missing profitable opportunities
- Overall accuracy can be high but useless for trading

**Example:**
```
Accuracy: 90% (looks good!)
But...
  No Increase: 95% recall ✓
  Increase:     20% recall ✗ (Missing 80% of opportunities!)
```

---

## Solutions Implemented

We've implemented **three complementary techniques** to improve "Increase" detection:

### 1. Class Weight Balancing ⚖️

**What it does:** Tells the model to pay more attention to the minority class during training.

**How it works:** 
- Assigns higher penalty for misclassifying "Increase" samples
- Models automatically adjust to balance the classes
- No data modification needed

**Implementation:**
```python
# Before (imbalanced)
LogisticRegression(max_iter=1000, random_state=42)

# After (balanced)
LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
```

**Models with class_weight:**
- ✅ Logistic Regression
- ✅ Ridge Classifier
- ✅ Decision Tree
- ✅ Random Forest
- ✅ SVM
- ❌ Naive Bayes (not supported)
- ❌ KNN (not supported)
- ❌ Gradient Boosting (not directly supported)

### 2. SMOTE Oversampling 🔄

**What it does:** Creates synthetic samples of the minority class to balance the dataset.

**How it works:**
- Finds minority class samples
- Creates new synthetic samples between existing ones
- Balances the training data
- Validation data remains unchanged (realistic evaluation)

**Implementation:**
```python
from imblearn.over_sampling import SMOTE

# Original data
X_train: 80,000 samples (10% Increase)
y_train: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, ...]

# After SMOTE
X_train_resampled: 144,000 samples (50% Increase)
y_train_resampled: [0, 1, 0, 1, 0, 1, 0, 1, ...]
```

**Advantages:**
- More balanced training data
- Model sees more "Increase" examples
- Better learning of minority class patterns

**When SMOTE is applied:**
- Automatically when imbalance ratio > 1.5:1
- Only on training data (not validation)
- Can be disabled with `use_smote=False`

### 3. Optimal Threshold Tuning 🎯

**What it does:** Adjusts the decision threshold to maximize F1 score and recall.

**How it works:**
- Default threshold: 0.5 (predict "Increase" if probability > 0.5)
- Tests thresholds from 0.1 to 0.9
- Finds threshold that maximizes F1 score
- Often lowers threshold to catch more "Increase" cases

**Example:**
```python
# Default threshold (0.5)
Probability: [0.45, 0.52, 0.48, 0.55]
Prediction:  [0,    1,    0,    1   ]  # Only 2 "Increase"

# Optimized threshold (0.40)
Probability: [0.45, 0.52, 0.48, 0.55]
Prediction:  [1,    1,    1,    1   ]  # All 4 "Increase"
```

**Impact:**
```
Default (0.5):  F1=0.45, Recall=0.35
Optimal (0.35): F1=0.62, Recall=0.58
Improvement:    F1=+0.17, Recall=+0.23 ✓
```

---

## How to Use

### Installation

First, install the required library:

```bash
pip install imbalanced-learn
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

### Basic Usage

The improvements are **automatically applied** when you train models:

```python
from src.model_training import train

# Train with all improvements (default)
models, scaler, results, best_model = train(
    df_train,
    target_bars=45,
    target_pct=3.0,
    use_smote=True  # Enable SMOTE (default)
)
```

### Disable SMOTE (if needed)

```python
# Train without SMOTE
models, scaler, results, best_model = train(
    df_train,
    target_bars=45,
    target_pct=3.0,
    use_smote=False  # Disable SMOTE
)
```

### Training Output

You'll see detailed information about the improvements:

```
Dataset shape: (100000, 10)
Target distribution:
0    85000
1    15000
Target balance: 
0    0.85
1    0.15
Class imbalance ratio: 5.67:1

Applying SMOTE to balance training data...
Original training size: 80000
Resampled training size: 136000
New class distribution: 
0    68000
1    68000

================================================================================
Model: LOGISTIC_REGRESSION
================================================================================

  Threshold Optimization:
    Default threshold (0.5): F1=0.4523, Recall=0.3542
    Optimal threshold (0.35): F1=0.6187, Recall=0.5834
    Improvement: F1=+0.1664, Recall=+0.2292

Validation Set Performance:
  Accuracy:  0.7234
  F1 Score:  0.6187
  Precision: 0.6621
  Recall:    0.5834
  ROC AUC:   0.7892

Classification Report:
              precision    recall  f1-score   support

 No Increase       0.75      0.82      0.78     17000
    Increase       0.66      0.58      0.62      3000

    accuracy                           0.72     20000
```

---

## Understanding the Techniques

### When to Use Each Technique

| Technique | Use When | Benefit |
|-----------|----------|---------|
| **Class Weight** | Always | Simple, no data modification |
| **SMOTE** | Imbalance > 1.5:1 | More training examples |
| **Threshold Tuning** | Model has predict_proba | Better precision-recall trade-off |

### Combining Techniques

The techniques work together:

1. **Class Weight** → Model pays attention to minority class
2. **SMOTE** → Model sees more minority examples
3. **Threshold Tuning** → Predictions optimized for F1/Recall

**Result:** Significantly better "Increase" detection!

### Metrics to Watch

Focus on these metrics for the "Increase" class:

- **Recall**: % of actual "Increase" cases detected (most important for trading)
- **F1 Score**: Balance between precision and recall
- **Precision**: % of "Increase" predictions that are correct
- **ROC AUC**: Overall model discrimination ability

**Trading Perspective:**
- High Recall = Don't miss opportunities
- High Precision = Avoid false signals
- F1 Score = Good balance

---

## Expected Improvements

### Before Improvements

```
Classification Report (Typical):
              precision    recall  f1-score   support

 No Increase       0.92      0.95      0.94     17000
    Increase       0.45      0.35      0.39      3000

    accuracy                           0.88     20000
```

**Problems:**
- Only 35% recall on "Increase" (missing 65% of opportunities!)
- Low F1 score (0.39)
- High overall accuracy is misleading

### After Improvements

```
Classification Report (Expected):
              precision    recall  f1-score   support

 No Increase       0.78      0.85      0.81     17000
    Increase       0.65      0.55      0.60      3000

    accuracy                           0.76     20000
```

**Improvements:**
- ✅ Recall increased from 35% → 55% (+20%)
- ✅ F1 score increased from 0.39 → 0.60 (+0.21)
- ✅ Precision increased from 0.45 → 0.65 (+0.20)
- ⚠️ Overall accuracy decreased slightly (88% → 76%)

**Why lower accuracy is OK:**
- We're catching more "Increase" cases (what matters for trading!)
- Trade-off: More false positives, but fewer missed opportunities
- Better balance between classes

### Real-World Impact

**Before (35% Recall):**
- 100 profitable opportunities
- Model detects: 35
- Missed: 65 ❌

**After (55% Recall):**
- 100 profitable opportunities  
- Model detects: 55
- Missed: 45 ✓

**Result:** 57% more opportunities detected!

---

## Troubleshooting

### Issue: SMOTE not available

**Error:**
```
Warning: imbalanced-learn not installed. Install with: pip install imbalanced-learn
Continuing without SMOTE...
```

**Solution:**
```bash
pip install imbalanced-learn
```

### Issue: SMOTE takes too long

**Cause:** Large dataset with many features

**Solutions:**

1. **Reduce SMOTE neighbors:**
```python
# In model_training.py, modify SMOTE call:
smote = SMOTE(random_state=42, k_neighbors=3)  # Default is 5
```

2. **Disable SMOTE:**
```python
models, scaler, results, best_model = train(
    df_train,
    use_smote=False
)
```

3. **Use smaller training set:**
```python
# Sample data before training
df_train_sample = df_train.sample(n=100000, random_state=42)
models, scaler, results, best_model = train(df_train_sample)
```

### Issue: Threshold optimization not working

**Cause:** Model doesn't support `predict_proba`

**Models with predict_proba:**
- ✅ Logistic Regression
- ✅ Random Forest
- ✅ Gradient Boosting
- ✅ SVM (with probability=True)
- ✅ Naive Bayes
- ❌ Ridge Classifier
- ❌ KNN (by default)

**Solution:** Use models that support probability predictions.

### Issue: Still poor recall

**Possible causes and solutions:**

1. **Target too strict:**
```python
# Try lower threshold
train(df_train, target_pct=2.0)  # Instead of 3.0
```

2. **Not enough data:**
```python
# Use more historical data
df_train = prepare_data('data/larger_dataset.csv')
```

3. **Features not predictive:**
```python
# Add more technical indicators
# Check FeaturesGenerator.py
```

4. **Extreme imbalance:**
```python
# Adjust SMOTE sampling strategy
smote = SMOTE(random_state=42, sampling_strategy=0.7)  # 70% minority
```

---

## Advanced Configuration

### Custom SMOTE Parameters

Edit `src/model_training.py`:

```python
# More aggressive oversampling
smote = SMOTE(
    random_state=42,
    k_neighbors=5,
    sampling_strategy=0.8  # Minority will be 80% of majority
)

# More conservative
smote = SMOTE(
    random_state=42,
    k_neighbors=3,
    sampling_strategy=0.5  # Minority will be 50% of majority
)
```

### Custom Threshold Optimization

Edit `src/model_training.py`:

```python
# Optimize for recall instead of F1
optimal_threshold, optimal_recall = find_optimal_threshold(
    model, X_val_scaled, y_val, metric='recall'
)

# Use wider threshold range
thresholds = np.arange(0.05, 0.95, 0.01)  # More granular
```

### Model-Specific Weights

Instead of 'balanced', use custom weights:

```python
from sklearn.utils.class_weight import compute_class_weight

# Compute custom weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
weight_dict = {0: class_weights[0], 1: class_weights[1] * 2}  # 2x weight on minority

# Use in model
LogisticRegression(class_weight=weight_dict)
```

---

## Best Practices

### 1. Always Check Class Distribution

```python
print(f"Target distribution:\n{y.value_counts()}")
print(f"Imbalance ratio: {y.value_counts().max() / y.value_counts().min():.2f}:1")
```

### 2. Monitor Multiple Metrics

Don't rely on accuracy alone:
- Recall (most important for trading)
- F1 Score (balance)
- Precision (avoid false signals)
- ROC AUC (overall performance)

### 3. Validate on Unseen Data

```python
# Train on training set
models, scaler, results, best_model = train(df_train)

# Test on completely separate test set
test_results = test(df_test, models, scaler)
```

### 4. Adjust Based on Trading Strategy

**Conservative (fewer trades, higher confidence):**
- Higher threshold (0.6-0.7)
- Higher precision
- Lower recall

**Aggressive (more trades, catch opportunities):**
- Lower threshold (0.3-0.4)
- Higher recall
- Lower precision

### 5. Use Walk-Forward Validation

```python
# Test on multiple time periods
for year in [2018, 2019, 2020]:
    df_test_year = df_test[df_test['year'] == year]
    results = test(df_test_year, models, scaler)
    print(f"Year {year}: Recall = {results['recall']:.2f}")
```

---

## Summary

### What We Implemented

1. ✅ **Class Weight Balancing** - Models pay attention to minority class
2. ✅ **SMOTE Oversampling** - More training examples for "Increase"
3. ✅ **Threshold Optimization** - Better precision-recall trade-off

### Expected Results

- 📈 **Recall**: +15-25% improvement
- 📈 **F1 Score**: +0.15-0.25 improvement
- 📈 **Precision**: +10-20% improvement
- ⚠️ **Accuracy**: May decrease slightly (but that's OK!)

### Key Takeaway

**Better to catch 55% of opportunities with 65% precision than to catch 35% with 45% precision!**

The improvements make your model much more useful for actual trading by detecting more "Increase" cases while maintaining reasonable precision.

---

## References

- [SMOTE Paper](https://arxiv.org/abs/1106.1813)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/)
- [Scikit-learn Class Weight](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)
- [Precision-Recall Trade-off](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)

Happy trading! 📈🎯
