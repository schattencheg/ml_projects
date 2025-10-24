# Progress Tracking Guide

This guide explains the progress tracking and time measurement features added to the ML model training pipeline.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Output Examples](#output-examples)
4. [Time Metrics](#time-metrics)
5. [Progress Bar](#progress-bar)
6. [Interpreting Results](#interpreting-results)
7. [Best Practices](#best-practices)

---

## Overview

The training pipeline now includes **comprehensive progress tracking** to help you monitor training execution and identify performance bottlenecks.

### ✅ What's Tracked

1. **Individual Model Training Time** - Time taken to train each model
2. **Total Training Time** - Cumulative time for all models
3. **Average Training Time** - Mean time per model
4. **Progress Bar** - Visual progress indicator with ETA
5. **Real-time Updates** - Live status updates during training

---

## Features

### 1. Per-Model Time Tracking

Each model's training time is measured and displayed:

```
Training logistic_regression... ✓ Completed in 2.34 seconds
Training random_forest... ✓ Completed in 15.67 seconds
Training xgboost... ✓ Completed in 3.21 seconds
```

### 2. Progress Bar

Visual progress indicator with:
- Current model being trained
- Number of models completed
- Elapsed time
- Estimated time remaining

```
Training Progress: |████████████████████| 8/8 [01:23<00:00]
```

### 3. Summary Statistics

After training completes:
- Total training time (seconds and minutes)
- Average time per model
- Training time included in results table

### 4. Results Table with Time

Training results summary includes training time:

```
           Model  Accuracy  F1 Score  Precision    Recall   ROC AUC  Train Time (s)
logistic_regression    0.7234    0.6187     0.6621    0.5834    0.7892            2.34
  random_forest        0.7156    0.6089     0.6543    0.5701    0.7823           15.67
       xgboost         0.7087    0.5976     0.6234    0.5567    0.7756            3.21
```

---

## Output Examples

### Example 1: Full Training Session

```
================================================================================
HARDWARE ACCELERATION SETTINGS
================================================================================
CPU cores to use: 15 of 16
GPU acceleration: Disabled
================================================================================

Dataset shape: (100000, 50)
Target distribution:
0    85000
1    15000
Target balance: {0: 0.85, 1: 0.15}
Class imbalance ratio: 5.67:1

Applying SMOTE to balance training data...
Original training size: 80000
Resampled training size: 136000
New class distribution: {0: 68000, 1: 68000}

================================================================================
TRAINING AND EVALUATING MODELS
================================================================================
Total models to train: 8

Training Progress: |██                  | 1/8 [00:02<00:14]

================================================================================
Model: LOGISTIC_REGRESSION
================================================================================
Training logistic_regression... ✓ Completed in 2.34 seconds

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

Training Progress: |████                | 2/8 [00:05<00:15]

================================================================================
Model: RIDGE_CLASSIFIER
================================================================================
Training ridge_classifier... ✓ Completed in 1.89 seconds
...

Training Progress: |████████████████████| 8/8 [01:23<00:00]

================================================================================
TRAINING COMPLETE
================================================================================
Total training time: 83.45 seconds (1.39 minutes)
Average time per model: 10.43 seconds

BEST MODEL: LOGISTIC_REGRESSION with accuracy: 0.7234
================================================================================

================================================================================
TRAINING RESULTS SUMMARY (Validation Set)
================================================================================

           Model  Accuracy  F1 Score  Precision    Recall   ROC AUC  Train Time (s)
logistic_regression    0.7234    0.6187     0.6621    0.5834    0.7892            2.34
  ridge_classifier    0.7156    0.6089     0.6543    0.5701    0.7823            1.89
       random_forest    0.7087    0.5976     0.6234    0.5567    0.7756           15.67
            xgboost    0.7023    0.5865     0.6123    0.5456    0.7689            3.21
           lightgbm    0.6956    0.5754     0.5987    0.5345    0.7623            2.98
     decision_tree    0.6823    0.5543     0.5876    0.5234    0.7456            4.56
       naive_bayes    0.6756    0.5432     0.5765    0.5123    0.7389            0.87
knn_k_neighbours    0.6689    0.5321     0.5654    0.5012    0.7312           51.93

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

### Example 2: With GPU Acceleration

```
================================================================================
HARDWARE ACCELERATION SETTINGS
================================================================================
CPU cores to use: 15 of 16
GPU acceleration: Enabled
================================================================================
  ✓ XGBoost: GPU acceleration enabled

...

Training xgboost... ✓ Completed in 0.18 seconds  # 18x faster with GPU!

...

Total training time: 45.23 seconds (0.75 minutes)
Average time per model: 5.65 seconds
```

---

## Time Metrics

### Individual Model Training Time

**What it measures:** Time from `model.fit()` start to completion

**Includes:**
- Model initialization
- Training iterations
- Internal cross-validation (if any)
- Hyperparameter tuning (if enabled)

**Does NOT include:**
- Data preparation
- Feature scaling
- Threshold optimization
- Metric calculation
- Visualization

### Total Training Time

**What it measures:** Sum of all individual model training times

**Formula:** `total_time = sum(model_training_times)`

**Use case:** Compare overall training efficiency across different configurations

### Average Training Time

**What it measures:** Mean training time per model

**Formula:** `avg_time = total_time / num_models`

**Use case:** Identify if certain configurations slow down all models

---

## Progress Bar

### Progress Bar Format

```
Training Progress: |████████████████████| 8/8 [01:23<00:00]
                    ^                    ^ ^  ^      ^
                    |                    | |  |      |
                    Bar                  | |  |      Remaining time
                                         | |  Elapsed time
                                         | Total models
                                         Completed models
```

### Progress Bar States

**Initial:**
```
Training Progress: |                    | 0/8 [00:00<??:??]
```

**In Progress:**
```
Training logistic_regression: |████    | 2/8 [00:05<00:15]
```

**Complete:**
```
Training Progress: |████████████████████| 8/8 [01:23<00:00]
```

### Progress Bar Features

- **Dynamic description:** Shows current model being trained
- **Accurate ETA:** Estimates remaining time based on average
- **Clean output:** Automatically clears after completion
- **Color coding:** (if terminal supports it)

---

## Interpreting Results

### Identifying Slow Models

Look at the "Train Time (s)" column:

```
           Model  Train Time (s)
       random_forest           15.67  # Slowest
knn_k_neighbours           51.93  # Very slow!
            xgboost            3.21  # Fast
logistic_regression            2.34  # Fast
```

**Analysis:**
- KNN is extremely slow (51.93s) - Consider reducing neighbors or using approximate methods
- Random Forest is slow (15.67s) - Consider reducing n_estimators or max_depth
- XGBoost and Logistic Regression are fast - Good for quick iterations

### Comparing Configurations

**Before hardware acceleration:**
```
Total training time: 245.67 seconds (4.09 minutes)
Average time per model: 30.71 seconds
```

**After multi-core (n_jobs=-1):**
```
Total training time: 83.45 seconds (1.39 minutes)  # 2.9x faster
Average time per model: 10.43 seconds
```

**After GPU (use_gpu=True):**
```
Total training time: 45.23 seconds (0.75 minutes)  # 5.4x faster
Average time per model: 5.65 seconds
```

### Identifying Bottlenecks

**Scenario 1: All models slow**
- **Cause:** Large dataset, insufficient hardware
- **Solution:** Enable hardware acceleration, reduce data size

**Scenario 2: One model very slow**
- **Cause:** Model complexity, poor hyperparameters
- **Solution:** Tune hyperparameters, consider removing model

**Scenario 3: Inconsistent times**
- **Cause:** System load, background processes
- **Solution:** Close other applications, run during off-hours

---

## Best Practices

### 1. Monitor Training Progress

Always watch the progress bar and time estimates:

```python
# The progress bar will show:
# - Which model is currently training
# - How many models are complete
# - Estimated time remaining
models, scaler, results, best_model = train(df_train)
```

### 2. Compare Training Times

After training, review the summary table:

```python
# Look for models with high training time
# Consider if the extra time is worth the performance gain
```

### 3. Optimize Slow Models

If a model is too slow:

**Option 1: Reduce complexity**
```python
# Before
RandomForestClassifier(n_estimators=100, max_depth=10)  # 15.67s

# After
RandomForestClassifier(n_estimators=50, max_depth=5)    # 4.23s
```

**Option 2: Enable acceleration**
```python
# Enable multi-core
train(df_train, n_jobs=-1)  # 2-4x faster

# Enable GPU (if available)
train(df_train, use_gpu=True)  # 10-50x faster for XGBoost/LightGBM
```

**Option 3: Remove model**
```python
# If KNN is too slow and not performing well, comment it out
# in get_model_configs()
```

### 4. Benchmark Different Configurations

```python
import time

configs = [
    {'n_jobs': 1, 'use_gpu': False},
    {'n_jobs': -1, 'use_gpu': False},
    {'n_jobs': -1, 'use_gpu': True}
]

for config in configs:
    start = time.time()
    models, scaler, results, best_model = train(df_train, **config)
    total_time = time.time() - start
    print(f"Config {config}: {total_time:.2f} seconds")
```

### 5. Track Progress Over Time

Keep a log of training times:

```python
# training_log.txt
2024-01-15: Total time: 245s, Avg: 30.7s (baseline)
2024-01-16: Total time: 83s, Avg: 10.4s (multi-core enabled)
2024-01-17: Total time: 45s, Avg: 5.7s (GPU enabled)
```

### 6. Set Realistic Expectations

**Small datasets (< 10K samples):**
- Expected: 1-5 seconds per model
- Total: 10-40 seconds

**Medium datasets (10K-100K samples):**
- Expected: 5-30 seconds per model
- Total: 40-240 seconds (0.7-4 minutes)

**Large datasets (> 100K samples):**
- Expected: 30-300 seconds per model
- Total: 4-40 minutes

### 7. Use Progress Tracking for Debugging

If training hangs:
1. Check which model is currently training (progress bar shows this)
2. Look at the training time for that model
3. If it's taking unusually long, interrupt and investigate

---

## Advanced Usage

### Custom Progress Tracking

If you want to add custom progress tracking:

```python
import time
from tqdm import tqdm

# Track data preparation
with tqdm(total=3, desc="Data Preparation") as pbar:
    # Load data
    df_train = pd.read_csv('data.csv')
    pbar.update(1)
    
    # Prepare features
    X, y = prepare_data(df_train)
    pbar.update(1)
    
    # Scale data
    scaler, X_scaled = fit_scaler_minmax(X)
    pbar.update(1)

# Train models (built-in progress tracking)
models, scaler, results, best_model = train(df_train)
```

### Logging Training Times

Save training times to a file:

```python
import json
from datetime import datetime

# Train models
models, scaler, results, best_model = train(df_train)

# Extract training times
training_times = {
    model_name: metrics['training_time']
    for model_name, metrics in results.items()
}

# Save to log file
log_entry = {
    'timestamp': datetime.now().isoformat(),
    'total_time': sum(training_times.values()),
    'model_times': training_times
}

with open('training_log.json', 'a') as f:
    f.write(json.dumps(log_entry) + '\n')
```

### Comparing Multiple Runs

```python
import matplotlib.pyplot as plt

# Run multiple training sessions
run_times = []
for i in range(5):
    models, scaler, results, best_model = train(df_train)
    total_time = sum(r['training_time'] for r in results.values())
    run_times.append(total_time)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(run_times, marker='o')
plt.xlabel('Run Number')
plt.ylabel('Total Training Time (seconds)')
plt.title('Training Time Consistency')
plt.grid(True)
plt.show()

print(f"Mean: {np.mean(run_times):.2f}s")
print(f"Std: {np.std(run_times):.2f}s")
```

---

## Summary

### Quick Reference

**View progress during training:**
- Progress bar shows current model and ETA
- Individual model times printed in real-time

**View summary after training:**
- Total training time
- Average time per model
- Per-model times in results table

**Optimize training time:**
1. Enable multi-core: `train(df_train, n_jobs=-1)`
2. Enable GPU: `train(df_train, use_gpu=True)`
3. Reduce model complexity
4. Remove slow models

### Key Metrics

- **Individual Time:** Time to train one model
- **Total Time:** Sum of all model training times
- **Average Time:** Mean time per model
- **Progress:** Visual indicator with ETA

### Benefits

✅ **Monitor progress** - Know how long training will take
✅ **Identify bottlenecks** - Find slow models
✅ **Compare configurations** - Measure acceleration impact
✅ **Track improvements** - Log training times over time
✅ **Debug issues** - Identify hanging models

---

## Additional Resources

- [Hardware Acceleration Guide](HARDWARE_ACCELERATION_GUIDE.md) - Speed up training
- [Model Training Documentation](../src/model_training.py) - Implementation details
- [tqdm Documentation](https://tqdm.github.io/) - Progress bar library

Happy training! ⏱️📊🚀
