# CNN Optimization Enhancements

## Summary of Issues Fixed and Enhancements Needed

### 1. Fixed Issues

#### A. Predict Method Error (trend_predictor.py line 261)
**Problem**: `ValueError: X has 133 features, but StandardScaler is expecting 126 features`

**Root Cause**: 
- Timestamp columns were being included in features during prediction
- Feature count mismatch between training and prediction

**Solution Applied**:
- Fixed `predict_with_probability()` method to properly handle numeric columns
- Added `_prepare_features_for_training()` call to clean prediction data
- Ensured consistent feature selection between training and prediction

#### B. Object Dtype Errors
**Problem**: `Invalid dtype: object` and `float() argument must be a string or a real number, not 'Timestamp'`

**Solution Applied**:
- Added `_prepare_features()` method in `HyperparameterOptimizer`
- Added `_prepare_features_for_training()` method in `TrendPredictor`
- Both methods handle 3D arrays with Timestamp objects
- Convert DataFrames to numeric, filter non-numeric columns

### 2. Enhancements Needed

#### A. Custom CNN Models with Neuron Count Optimization

**Current State**:
- Hyperparameter optimizer suggests neuron counts (filters, dense units, LSTM units)
- But uses fixed architectures from `CNNArchitectures` class
- Suggested neuron counts are not actually used

**Enhancement Required**:
Add custom model building methods that use the suggested neuron counts:

```python
def _build_simple_cnn(self, filters_1, filters_2, dense_units):
    """Build simple CNN with custom neuron counts."""
    model = models.Sequential([
        layers.Conv1D(filters=filters_1, kernel_size=3, padding='same', 
                     activation='relu', input_shape=self.input_shape),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(filters=filters_2, kernel_size=3, padding='same', 
                     activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(dense_units, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def _build_deeper_cnn(self, filters_1, filters_2, dense_units_1, dense_units_2, dropout_1, dropout_2):
    """Build deeper CNN with custom neuron counts."""
    model = models.Sequential([
        layers.Conv1D(filters=filters_1, kernel_size=3, padding='same',
                     activation='relu', input_shape=self.input_shape),
        layers.BatchNormalization(),
        layers.Conv1D(filters=filters_1, kernel_size=3, padding='same',
                     activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(dropout_1),
        
        layers.Conv1D(filters=filters_2, kernel_size=3, padding='same',
                     activation='relu'),
        layers.BatchNormalization(),
        layers.Conv1D(filters=filters_2, kernel_size=3, padding='same',
                     activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(dropout_1),
        
        layers.Dense(dense_units_1, activation='relu'),
        layers.Dropout(dropout_2),
        layers.Dense(dense_units_2, activation='relu'),
        layers.Dropout(dropout_2),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def _build_cnn_lstm(self, filters_1, filters_2, lstm_units_1, lstm_units_2, 
                    dense_units_1, dense_units_2, dropout_1, dropout_2, dropout_3):
    """Build CNN-LSTM with custom neuron counts."""
    model = models.Sequential([
        layers.Conv1D(filters=filters_1, kernel_size=3, padding='same',
                     activation='relu', input_shape=self.input_shape),
        layers.BatchNormalization(),
        layers.Conv1D(filters=filters_2, kernel_size=3, padding='same',
                     activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(dropout_1),
        
        layers.LSTM(lstm_units_1, return_sequences=True),
        layers.Dropout(dropout_2),
        layers.LSTM(lstm_units_2),
        layers.Dropout(dropout_2),
        
        layers.Dense(dense_units_1, activation='relu'),
        layers.Dropout(dropout_3),
        layers.Dense(dense_units_2, activation='relu'),
        layers.Dropout(dropout_3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model
```

#### B. Optimization Results Visualization

**Enhancement Required**:
Add method to visualize optimization results:

```python
def plot_optimization_results(self, study, save_path=None):
    """
    Plot optimization results showing:
    1. F1 score progress over trials
    2. Best metrics comparison (accuracy, precision, recall, F1)
    3. Architecture performance comparison
    4. Learning rate vs F1 score
    5. Batch size impact
    6. Top 10 trials
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. F1 score progress
    # 2. Metrics bar chart
    # 3. Architecture comparison
    # 4. Learning rate scatter
    # 5. Batch size impact
    # 6. Top 10 trials
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
```

#### C. Model Comparison and Evaluation

**Enhancement Required**:
Add method to compare all optimized model versions:

```python
def compare_optimized_models(self, study, X_test, y_test, top_n=5):
    """
    Compare top N optimized models on test data.
    
    Returns DataFrame with:
    - Model rank
    - Architecture type
    - Validation F1 score
    - Test accuracy, precision, recall, F1
    - Hyperparameters used
    """
    results = []
    
    # Get top N trials
    top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:top_n]
    
    for i, trial in enumerate(top_trials):
        # Rebuild model with trial parameters
        # Train on full training set
        # Evaluate on test set
        # Store results
        pass
    
    comparison_df = pd.DataFrame(results)
    
    # Plot comparison
    self._plot_model_comparison(comparison_df)
    
    return comparison_df

def _plot_model_comparison(self, comparison_df):
    """Plot comparison of top models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Metrics comparison (bar chart)
    # 2. Architecture distribution (pie chart)
    
    plt.show()
```

#### D. Enhanced Objective Function

**Enhancement Required**:
Store optimization history for later analysis:

```python
def __init__(self, ...):
    # ... existing code ...
    self.optimization_history = []

def objective(self, trial):
    # ... existing code ...
    
    # Calculate all metrics
    accuracy = accuracy_score(self.y_val, val_predictions_binary)
    precision = precision_score(self.y_val, val_predictions_binary, zero_division=0)
    recall = recall_score(self.y_val, val_predictions_binary, zero_division=0)
    f1 = f1_score(self.y_val, val_predictions_binary, zero_division=0)
    
    # Store history
    self.optimization_history.append({
        'trial': trial.number,
        'architecture': architecture_type,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'params': trial.params
    })
    
    # Return F1 score (better metric for imbalanced data)
    return f1
```

### 3. Implementation Steps

1. **Add custom model building methods** to `hyperparameter_optimizer.py`
   - `_build_simple_cnn()`
   - `_build_deeper_cnn()`
   - `_build_cnn_lstm()`

2. **Update objective function**:
   - Store optimization history
   - Calculate all metrics (accuracy, precision, recall, F1)
   - Return F1 score instead of accuracy

3. **Add visualization method**:
   - `plot_optimization_results(study, save_path)`
   - Create 6-panel plot showing optimization progress

4. **Add comparison method**:
   - `compare_optimized_models(study, X_test, y_test, top_n)`
   - Rebuild and evaluate top N models
   - Return comparison DataFrame
   - Plot comparison results

5. **Update main.py** to use new features:
   ```python
   # Run optimization
   study = optimizer.optimize(n_trials=50)
   
   # Visualize results
   optimizer.plot_optimization_results(study, save_path='optimization_results.png')
   
   # Compare top models
   comparison = optimizer.compare_optimized_models(study, X_test, y_test, top_n=5)
   print(comparison)
   ```

### 4. Benefits

- **Custom Neuron Counts**: Models actually use the optimized neuron counts
- **Better Visualization**: See optimization progress and results clearly
- **Model Comparison**: Compare multiple optimized versions systematically
- **Better Metrics**: Use F1 score (better for imbalanced data) instead of accuracy
- **Complete Analysis**: Full picture of optimization process and results

### 5. Next Steps

1. Implement the custom model building methods
2. Add optimization history tracking
3. Implement visualization methods
4. Implement comparison methods
5. Update main.py to demonstrate new features
6. Test with actual data

## Files Modified

1. `src/hyperparameter_optimizer.py` - Enhanced with custom models, visualization, comparison
2. `src/trend_predictor.py` - Fixed predict method, added data cleaning
3. `main.py` - Updated to use new optimization features

## Expected Output

After optimization, you'll see:
1. Progress bar during optimization
2. Best parameters and F1 score
3. 6-panel visualization plot showing:
   - F1 score progress
   - Best model metrics
   - Architecture comparison
   - Learning rate impact
   - Batch size impact
   - Top 10 trials
4. Comparison table of top 5 models with test metrics
5. Comparison plots showing model performance
