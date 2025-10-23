# Model Performance Visualization Guide

This guide explains the automatic visualization features added to the model training and testing pipeline.

## Overview

The training and testing pipeline now automatically generates comprehensive visualizations comparing all models across multiple performance metrics.

## Features

### 1. Training Results Visualization

**When:** Automatically generated after training all models

**File:** `plots/model_comparison_training.png`

**What it shows:**
- Performance on validation set (20% of training data)
- 6 subplots comparing all models:
  1. **Accuracy** - Overall correctness
  2. **F1 Score** - Balance between precision and recall
  3. **Precision** - Accuracy of positive predictions
  4. **Recall** - Ability to find all positive cases
  5. **ROC AUC** - Overall discrimination ability
  6. **All Metrics** - Side-by-side comparison

**Console Output:**
- Summary table sorted by F1 Score
- Best model for each metric

### 2. Test Results Visualization

**When:** Automatically generated after testing on held-out test data

**File:** `plots/model_comparison_test.png`

**What it shows:**
- Performance on completely unseen test data
- Same 6 subplots as training visualization
- Helps identify overfitting (training vs test performance)

---

## How to Use

### Automatic Generation

Visualizations are **automatically created** when you train or test models:

```python
from src.model_training import train, test
from src.data_preparation import prepare_data

# Load data
df_train, df_test = prepare_data('data/6e_2007_2019.csv')

# Train models (automatically creates training visualization)
models, scaler, results, best_model = train(df_train)
# → Creates: plots/model_comparison_training.png

# Test models (automatically creates test visualization)
test_results = test(models, scaler, df_test)
# → Creates: plots/model_comparison_test.png
```

### Console Output

You'll see detailed summaries in the console:

```
================================================================================
TRAINING RESULTS SUMMARY (Validation Set)
================================================================================

           Model  Accuracy  F1 Score  Precision    Recall   ROC AUC
logistic_regression    0.7234    0.6187     0.6621    0.5834    0.7892
  ridge_classifier    0.7156    0.6089     0.6543    0.5701    0.7823
    decision_tree    0.6987    0.5876     0.6234    0.5567    0.7456
       naive_bayes    0.6823    0.5654     0.5987    0.5389    0.7234
               knn    0.6756    0.5543     0.5876    0.5234    0.7123

================================================================================
BEST MODELS BY METRIC (Validation Set)
================================================================================
  Best Accuracy:  logistic_regression (0.7234)
  Best F1 Score:  logistic_regression (0.6187)
  Best Precision: logistic_regression (0.6621)
  Best Recall:    logistic_regression (0.5834)
  Best ROC AUC:   logistic_regression (0.7892)

Training comparison plot saved to: plots/model_comparison_training.png
```

---

## Understanding the Visualizations

### Plot Layout

Each visualization contains **6 subplots**:

```
┌─────────────┬─────────────┬─────────────┐
│  Accuracy   │  F1 Score   │  Precision  │
├─────────────┼─────────────┼─────────────┤
│   Recall    │  ROC AUC    │ All Metrics │
└─────────────┴─────────────┴─────────────┘
```

### Individual Metric Plots (Plots 1-5)

**Features:**
- Bar chart for each model
- Values displayed on top of bars
- Y-axis: 0 to 1 (normalized scores)
- Grid lines for easy reading
- Color-coded by metric

**How to read:**
- Higher bars = Better performance
- Compare heights across models
- Look for consistent performers

### All Metrics Comparison (Plot 6)

**Features:**
- Grouped bar chart
- All 5 metrics side-by-side for each model
- Legend showing metric colors
- Easy to spot strengths/weaknesses

**How to read:**
- Each model has 5 bars (one per metric)
- Look for balanced performance (similar heights)
- Identify trade-offs (high precision but low recall, etc.)

---

## Interpreting Results

### Key Metrics Explained

#### 1. Accuracy
- **What:** Overall correctness (correct predictions / total predictions)
- **Good for:** Balanced datasets
- **Warning:** Can be misleading with imbalanced data
- **Target:** > 0.70

#### 2. F1 Score ⭐ (Most Important)
- **What:** Harmonic mean of precision and recall
- **Good for:** Imbalanced datasets (like ours!)
- **Why important:** Balances false positives and false negatives
- **Target:** > 0.60

#### 3. Precision
- **What:** Accuracy of positive predictions (true positives / predicted positives)
- **Meaning:** When model says "Increase", how often is it right?
- **High precision:** Fewer false alarms
- **Target:** > 0.65

#### 4. Recall ⭐ (Critical for Trading)
- **What:** Ability to find all positive cases (true positives / actual positives)
- **Meaning:** Of all actual "Increase" cases, how many did we catch?
- **High recall:** Don't miss opportunities
- **Target:** > 0.55

#### 5. ROC AUC
- **What:** Area under ROC curve (discrimination ability)
- **Good for:** Overall model quality
- **Range:** 0.5 (random) to 1.0 (perfect)
- **Target:** > 0.75

### What to Look For

#### ✅ Good Signs

1. **Consistent Performance**
   - Similar scores across metrics
   - No extreme trade-offs

2. **High F1 Score**
   - > 0.60 is good
   - > 0.70 is excellent

3. **Balanced Precision/Recall**
   - Both > 0.55
   - Difference < 0.15

4. **Training vs Test Similarity**
   - Test scores within 5-10% of training
   - Indicates good generalization

#### ⚠️ Warning Signs

1. **Large Training-Test Gap**
   - Training: 0.80, Test: 0.60
   - Indicates overfitting

2. **Very Low Recall**
   - < 0.40 means missing 60%+ of opportunities
   - Not useful for trading

3. **Very Low Precision**
   - < 0.50 means more false signals than true
   - Too many bad trades

4. **Extreme Imbalance**
   - Precision: 0.90, Recall: 0.20
   - Or vice versa
   - Model is too conservative or aggressive

---

## Example Analysis

### Scenario 1: Good Model

```
Model: logistic_regression
  Accuracy:  0.7234
  F1 Score:  0.6187  ✅ Good
  Precision: 0.6621  ✅ Good
  Recall:    0.5834  ✅ Good
  ROC AUC:   0.7892  ✅ Good
```

**Analysis:**
- ✅ All metrics above target
- ✅ Balanced precision/recall
- ✅ Good F1 score
- **Verdict:** Excellent for trading

### Scenario 2: Overfitted Model

```
Training:
  F1 Score:  0.8234  ← High

Test:
  F1 Score:  0.5123  ← Much lower
```

**Analysis:**
- ⚠️ Large gap (0.31 difference)
- ⚠️ Model memorized training data
- ⚠️ Poor generalization
- **Verdict:** Need regularization or more data

### Scenario 3: Too Conservative

```
Model: ridge_classifier
  Precision: 0.8234  ← Very high
  Recall:    0.3456  ← Very low
```

**Analysis:**
- ⚠️ High precision but low recall
- ⚠️ Missing 65% of opportunities
- ⚠️ Too few trades
- **Verdict:** Lower threshold or use different model

### Scenario 4: Too Aggressive

```
Model: decision_tree
  Precision: 0.4123  ← Very low
  Recall:    0.7834  ← Very high
```

**Analysis:**
- ⚠️ High recall but low precision
- ⚠️ Too many false signals
- ⚠️ Many losing trades
- **Verdict:** Raise threshold or use different model

---

## Comparing Training vs Test

### How to Compare

1. **Open both plots side-by-side**
   - `plots/model_comparison_training.png`
   - `plots/model_comparison_test.png`

2. **Look for consistency**
   - Similar relative rankings
   - Similar metric values

3. **Check for overfitting**
   - Training much better than test = overfitting
   - Test better than training = lucky (rare)

### Expected Differences

**Normal:** Test scores 2-10% lower than training
```
Training F1: 0.6187
Test F1:     0.5876  (5% lower) ✅ OK
```

**Concerning:** Test scores >15% lower than training
```
Training F1: 0.7234
Test F1:     0.5123  (29% lower) ⚠️ Overfitting!
```

---

## Customization

### Change Save Path

```python
# Custom path for training plot
plot_training_comparison(results, save_path='my_plots/training.png')

# Custom path for test plot
plot_model_comparison(test_results, save_path='my_plots/test.png')
```

### Disable Plots

If you don't want automatic plots, comment out the calls in `model_training.py`:

```python
# In train() function
# print_training_results_summary(results)
# plot_training_comparison(results)

# In test() function
# print_test_results_summary(results_test)
# plot_model_comparison(results_test)
```

### Create Custom Plots

You can create your own visualizations using the results:

```python
import matplotlib.pyplot as plt

# Get results
models, scaler, results, best_model = train(df_train)

# Extract F1 scores
f1_scores = {name: metrics['f1'] for name, metrics in results.items()}

# Custom plot
plt.figure(figsize=(10, 6))
plt.bar(f1_scores.keys(), f1_scores.values())
plt.title('F1 Scores Comparison')
plt.ylabel('F1 Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('my_custom_plot.png')
plt.show()
```

---

## Best Practices

### 1. Always Review Both Plots

- Training plot shows model capability
- Test plot shows real-world performance
- Compare them to detect overfitting

### 2. Focus on F1 and Recall

For trading:
- **F1 Score** - Overall quality
- **Recall** - Don't miss opportunities
- Precision is important but secondary

### 3. Look for Consistency

- Best model should be best across multiple metrics
- If different models win different metrics, investigate why

### 4. Save and Compare Over Time

```bash
# Organize by date
plots/
  2024-01-15_training.png
  2024-01-15_test.png
  2024-01-20_training.png
  2024-01-20_test.png
```

Track improvements as you:
- Add more features
- Tune hyperparameters
- Apply imbalanced data techniques

### 5. Document Your Findings

Keep notes on what works:
```
2024-01-15: Baseline
  Best F1: 0.45 (logistic_regression)

2024-01-16: Added SMOTE
  Best F1: 0.62 (logistic_regression) ✅ +37% improvement!

2024-01-17: Optimized threshold
  Best F1: 0.65 (logistic_regression) ✅ +5% improvement
```

---

## Troubleshooting

### Issue: Plots not appearing

**Cause:** matplotlib backend issue

**Solution:**
```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt
```

### Issue: Plots saved but not displayed

**Cause:** Running in non-interactive environment

**Solution:** Check the saved files in `plots/` directory

### Issue: Text overlapping on x-axis

**Cause:** Long model names

**Solution:** Already handled with rotation=45 and ha='right'

### Issue: Want higher resolution

**Solution:** Increase DPI in the code:
```python
plt.savefig(save_path, dpi=600, bbox_inches='tight')  # Default is 300
```

---

## Summary

### What You Get

✅ **Automatic Visualizations**
- Training performance comparison
- Test performance comparison
- 6 subplots per visualization

✅ **Console Summaries**
- Sorted tables
- Best model identification
- Easy to read format

✅ **Saved Files**
- High-resolution PNG files
- Ready for reports/presentations
- Organized in `plots/` directory

### When They're Created

- **Training:** After `train()` completes
- **Testing:** After `test()` completes
- **Automatic:** No extra code needed

### Key Benefits

1. **Quick Comparison** - See all models at a glance
2. **Identify Best Model** - Clear visual ranking
3. **Spot Issues** - Detect overfitting, imbalance
4. **Track Progress** - Compare over time
5. **Professional** - Ready for presentations

Happy visualizing! 📊📈
