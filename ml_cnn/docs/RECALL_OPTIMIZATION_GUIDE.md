# Recall Optimization Guide

## Overview

The CNN optimizer has been configured to **maximize Recall** instead of F1 score. This is ideal when you want to minimize false negatives (missed opportunities).

## What is Recall?

**Recall** (also called Sensitivity or True Positive Rate) measures:
```
Recall = True Positives / (True Positives + False Negatives)
```

**High Recall means**: The model catches most of the positive cases (fewer missed opportunities)

## When to Use Recall Optimization

✅ **Use Recall when**:
- Missing a positive case is costly (e.g., missing a profitable trade)
- You want to catch all potential opportunities
- False positives are acceptable (you can filter them later)
- You prefer to be cautious and not miss signals

❌ **Don't use Recall when**:
- False positives are very costly
- You need balanced precision and recall (use F1 instead)
- You want high confidence predictions (use precision instead)

## Trade-offs

### Maximizing Recall:
- ✅ Catches more true positives (fewer missed opportunities)
- ✅ Lower false negative rate
- ⚠️ May increase false positives (more false alarms)
- ⚠️ Lower precision (more noise in predictions)

### Example:
If optimizing for recall in trading:
- **Good**: Won't miss many profitable trades
- **Bad**: May enter some unprofitable trades (false signals)

## How It Works

The optimizer now:
1. **Objective Function**: Returns `recall` instead of `f1_score`
2. **Optimization Target**: Maximizes validation recall
3. **Model Selection**: Chooses models with highest recall
4. **Visualization**: Shows recall as primary metric

## Changes Made

### 1. Objective Function
```python
# Before (F1 optimization)
return f1

# After (Recall optimization)
return recall
```

### 2. Visualization
- Title: "CNN Hyperparameter Optimization Results (Maximizing Recall)"
- Panel 1: "Recall Progress (Optimization Target)"
- Panel 3: "Architecture Comparison (by Recall)"
- Panel 4: "Learning Rate vs Recall"
- Panel 5: "Batch Size Impact on Recall"
- Panel 6: "Top 10 Trials (by Recall)"

### 3. Output Messages
```
Best Recall: 0.8234
```

## Usage

No code changes needed! Just run the optimizer:

```python
from src.enhanced_cnn_optimizer import EnhancedCNNOptimizer

optimizer = EnhancedCNNOptimizer(X_train, y_train, X_val, y_val, input_shape)
study = optimizer.optimize(n_trials=50)
```

The optimizer will automatically maximize recall.

## Interpreting Results

### Good Recall Values:
- **0.80-0.90**: Excellent - catching most positives
- **0.70-0.80**: Good - catching many positives
- **0.60-0.70**: Moderate - missing some positives
- **< 0.60**: Poor - missing many positives

### Example Output:
```
================================================================================
OPTIMIZATION COMPLETE
================================================================================
Best trial: 23
Best Recall: 0.8234
Best architecture: deeper

Best parameters:
  filters_1: 48
  filters_2: 96
  dense_units_1: 256
  dense_units_2: 128
  ...

Top model test performance:
  Test Accuracy: 0.7456
  Test Precision: 0.6789
  Test Recall: 0.8234 ← OPTIMIZED METRIC
  Test F1: 0.7434
================================================================================
```

### What This Means:
- **Recall 0.8234**: Model catches 82.34% of all positive cases
- **Precision 0.6789**: 67.89% of predicted positives are actually positive
- **Trade-off**: High recall but moderate precision (more false positives)

## Comparison with Other Metrics

### Recall vs F1 vs Precision:

| Metric | What It Optimizes | Use Case |
|--------|------------------|----------|
| **Recall** | Catch all positives | Don't miss opportunities |
| **F1** | Balance precision & recall | Balanced performance |
| **Precision** | Avoid false positives | High confidence predictions |

### Example Scenarios:

**Scenario 1: Trading Signals**
- Optimize for: **Recall**
- Reason: Don't want to miss profitable trades
- Accept: Some false signals (can be filtered)

**Scenario 2: Medical Diagnosis**
- Optimize for: **Recall**
- Reason: Don't want to miss diseases
- Accept: Some false positives (can be retested)

**Scenario 3: Spam Detection**
- Optimize for: **Precision**
- Reason: Don't want to mark real emails as spam
- Accept: Some spam gets through

## Switching to Other Metrics

### To Optimize for F1 Score:
Edit `src/enhanced_cnn_optimizer.py`:
```python
# In objective() method, change:
return recall  # Current
# To:
return f1  # For F1 optimization
```

### To Optimize for Precision:
```python
return precision  # For precision optimization
```

### To Optimize for Accuracy:
```python
return accuracy  # For accuracy optimization
```

Then update visualization labels accordingly.

## Best Practices

1. **Understand Your Goal**:
   - What's more costly: missing positives or false alarms?
   - Choose metric accordingly

2. **Monitor All Metrics**:
   - Even when optimizing recall, check precision and F1
   - Ensure recall isn't too high at expense of precision

3. **Set Thresholds**:
   - After optimization, adjust prediction threshold
   - Lower threshold = higher recall, lower precision
   - Higher threshold = lower recall, higher precision

4. **Validate on Test Set**:
   - Training recall may be higher than test recall
   - Always check test performance

5. **Consider Business Impact**:
   - Calculate cost of false negatives vs false positives
   - Choose metric that minimizes total cost

## Example: Adjusting Prediction Threshold

After optimization, you can adjust the threshold:

```python
# Default threshold (0.5)
predictions = (probabilities > 0.5).astype(int)

# Lower threshold for higher recall
predictions = (probabilities > 0.3).astype(int)  # More positives predicted

# Higher threshold for higher precision
predictions = (probabilities > 0.7).astype(int)  # Fewer positives predicted
```

## Monitoring During Optimization

Watch for:
1. **Recall Progress**: Should increase over trials
2. **Precision Trade-off**: May decrease as recall increases
3. **F1 Score**: Balance between precision and recall
4. **Architecture Impact**: Which architecture achieves best recall

## Expected Results

### Typical Recall Optimization:
- **Baseline Recall**: 0.60-0.65
- **After Optimization**: 0.75-0.85
- **Improvement**: +15-25%

### Trade-offs:
- **Precision**: May drop 5-10%
- **Accuracy**: May stay similar or drop slightly
- **F1**: Usually improves (balanced metric)

## Summary

✅ **Recall optimization is now active**
✅ **Best for**: Minimizing missed opportunities
✅ **Trade-off**: May increase false positives
✅ **Monitor**: All metrics, not just recall
✅ **Adjust**: Prediction threshold after optimization

The optimizer will find models that catch the most positive cases while maintaining reasonable overall performance.
