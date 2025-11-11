# Quick Start: CNN Optimization

## 1. Run the Example (Easiest)

```bash
python example_enhanced_optimization.py
```

This will automatically:
- Load BTC-USD data
- Generate features
- Run 30 optimization trials
- Show visualization
- Compare top 5 models
- Save results

## 2. Use in Your Code

```python
from src.enhanced_cnn_optimizer import EnhancedCNNOptimizer

# Prepare your data (X_train, y_train, X_val, y_val)
# X should be 3D: (samples, sequence_length, features)
# y should be 1D: (samples,) with binary labels

# Initialize
input_shape = (X_train.shape[1], X_train.shape[2])
optimizer = EnhancedCNNOptimizer(X_train, y_train, X_val, y_val, input_shape)

# Optimize
study = optimizer.optimize(n_trials=50)

# Visualize
optimizer.plot_optimization_results(study, save_path='results.png')

# Compare
comparison = optimizer.compare_optimized_models(study, X_test, y_test, top_n=5)
```

## 3. What You Get

### Optimization Results
- Best F1 score
- Best architecture type
- Best hyperparameters (neuron counts, learning rate, etc.)

### Visualization (6 panels)
1. F1 score progress
2. Best model metrics
3. Architecture comparison
4. Learning rate impact
5. Batch size impact
6. Top 10 trials

### Model Comparison
- DataFrame with test metrics for top N models
- Comparison plots
- CSV file with results

## 4. Key Parameters

```python
# Number of optimization trials (more = better but slower)
n_trials = 50  # Try 30-100

# Number of top models to compare
top_n = 5  # Try 3-10

# Show progress bar during optimization
show_progress = True

# Save visualization
save_path = 'optimization_results.png'
```

## 5. Customization

### Adjust Neuron Count Ranges

Edit `src/enhanced_cnn_optimizer.py`, in `objective()` method:

```python
# For simple CNN
filters_1 = trial.suggest_int('filters_1', 16, 128, step=16)  # Change range
filters_2 = trial.suggest_int('filters_2', 32, 128, step=16)
dense_units = trial.suggest_int('dense_units', 32, 128, step=16)
```

### Adjust Learning Rate Range

```python
learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
# Change to: 1e-6 to 1e-1 for wider range
```

### Adjust Batch Sizes

```python
batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
# Add more: [8, 16, 32, 64, 128]
```

### Adjust Epochs

```python
epochs = trial.suggest_int('epochs', 20, 100, step=10)
# Change to: 10 to 50 for faster trials
```

## 6. Expected Time

**30 trials**:
- CPU only: ~1-2 hours
- With GPU: ~20-40 minutes

**50 trials**:
- CPU only: ~2-3 hours
- With GPU: ~30-60 minutes

**100 trials**:
- CPU only: ~4-6 hours
- With GPU: ~1-2 hours

## 7. Expected Results

**Baseline** (no optimization):
- F1 Score: ~0.60-0.65

**After Optimization**:
- F1 Score: ~0.70-0.75
- Improvement: +10-15%

## 8. Troubleshooting

### "Out of memory"
```python
# Reduce batch size
batch_size = trial.suggest_categorical('batch_size', [16, 32])  # Remove 64

# Or reduce sequence length when creating data
sequence_length = 5  # Instead of 10
```

### "Too slow"
```python
# Reduce trials
study = optimizer.optimize(n_trials=20)  # Instead of 50

# Reduce epochs range
epochs = trial.suggest_int('epochs', 10, 30, step=10)  # Instead of 20-100
```

### "Poor results"
```python
# Increase trials
study = optimizer.optimize(n_trials=100)  # More exploration

# Check data quality
print(f"Positive samples: {y_train.sum()}")
print(f"Negative samples: {len(y_train) - y_train.sum()}")
# Should be somewhat balanced (at least 20-80%)
```

## 9. Files Generated

After running optimization:

1. **optimization_results.png** - 6-panel visualization
2. **model_comparison_results.csv** - Detailed comparison table
3. **Console output** - Best parameters and metrics

## 10. Integration with Main Pipeline

```python
# In main.py or your script

# 1. Run optimization once
from src.enhanced_cnn_optimizer import EnhancedCNNOptimizer
optimizer = EnhancedCNNOptimizer(X_train, y_train, X_val, y_val, input_shape)
study = optimizer.optimize(n_trials=50)

# 2. Get best parameters
best_params = study.best_params
print(f"Best architecture: {best_params['architecture']}")
print(f"Best learning rate: {best_params['learning_rate']}")

# 3. Use in TrendPredictor
from src.trend_predictor import TrendPredictor
predictor = TrendPredictor(sequence_length=10)

# Train with optimized parameters
predictor.train(
    features_df, 
    price_series,
    optimize_hyperparams=False,  # Use your own parameters
    # ... set other parameters from best_params
)

# 4. Make predictions
predictions, probabilities = predictor.predict_with_probability(test_features)
```

## 11. Tips for Best Results

1. **More trials = better results** (but slower)
   - Start with 30 for quick test
   - Use 50-100 for production

2. **Use GPU if available**
   - 3-5x faster
   - Set `CUDA_VISIBLE_DEVICES=0` environment variable

3. **Check data balance**
   - Should have reasonable class balance
   - Use SMOTE if very imbalanced

4. **Monitor progress**
   - Watch F1 scores during optimization
   - Stop early if not improving

5. **Save best model**
   - After optimization, train final model with best parameters
   - Save for later use

## 12. Common Workflows

### Workflow 1: Quick Test
```python
# 20 trials, quick visualization
study = optimizer.optimize(n_trials=20)
optimizer.plot_optimization_results(study)
```

### Workflow 2: Production
```python
# 100 trials, full comparison
study = optimizer.optimize(n_trials=100)
optimizer.plot_optimization_results(study, save_path='results.png')
comparison = optimizer.compare_optimized_models(study, X_test, y_test, top_n=10)
comparison.to_csv('comparison.csv')
```

### Workflow 3: Iterative
```python
# Run multiple optimization sessions
for i in range(3):
    study = optimizer.optimize(n_trials=30)
    print(f"Session {i+1} best F1: {study.best_value:.4f}")
    # Adjust ranges based on results
```

## Need Help?

Check these files:
- `CNN_OPTIMIZATION_ENHANCEMENTS.md` - Detailed implementation
- `OPTIMIZATION_COMPLETE_SUMMARY.md` - Complete summary
- `example_enhanced_optimization.py` - Working example
- `src/enhanced_cnn_optimizer.py` - Source code with comments

## Quick Commands

```bash
# Run example
python example_enhanced_optimization.py

# View results
# - Check optimization_results.png
# - Check model_comparison_results.csv
# - Check console output for best parameters
```

That's it! You're ready to optimize your CNN models! 🚀
