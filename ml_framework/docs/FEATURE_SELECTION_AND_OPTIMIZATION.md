# Feature Selection and Hyperparameter Optimization Guide

## Overview

The ML Framework now includes powerful tools for:
1. **Feature Selection** - Track importance and drop unnecessary features BEFORE training
2. **Hyperparameter Optimization** - Find optimal parameters for ML models
3. **GPU Acceleration** - Use GPU for XGBoost, CatBoost, and CNN models

## Feature Selection

### FeatureSelector

The `FeatureSelector` helps identify and select the most important features before training models.

#### Available Methods

1. **Tree-based** (`method='tree'`)
   - Uses Random Forest feature importance
   - Fast and effective
   - **Recommended for most cases**

2. **Mutual Information** (`method='mutual_info'`)
   - Measures statistical dependency
   - Works well for non-linear relationships

3. **Correlation** (`method='correlation'`)
   - Removes highly correlated features
   - Reduces multicollinearity

4. **RFE** (`method='rfe'`)
   - Recursive Feature Elimination
   - Slower but thorough

5. **L1-based** (`method='lasso'`)
   - Uses L1 regularization
   - Good for linear relationships

#### Basic Usage

```python
from src import FeatureSelector

# Create selector
feature_selector = FeatureSelector(method='tree')

# Fit on training data
feature_selector.fit(
    X=train_df[feature_cols],
    y=train_df['target'],
    n_features=50,      # Select top 50 features (None = auto)
    threshold=0.01      # Importance threshold (None = auto)
)

# Transform data
train_selected = feature_selector.transform(train_df[feature_cols])
test_selected = feature_selector.transform(test_df[feature_cols])

# Get selected features
selected_features = feature_selector.get_selected_features()
dropped_features = feature_selector.get_dropped_features()
```

#### Auto Selection

```python
# Automatic feature selection based on importance
feature_selector.fit(
    X=train_df[feature_cols],
    y=train_df['target'],
    n_features=None,    # Auto: selects features above mean importance
    threshold=None      # Auto: uses method-specific threshold
)
```

#### Feature Importance

```python
# Get feature importance scores
importance = feature_selector.get_feature_importance(top_n=20)

# Print summary
feature_selector.print_summary()

# Plot importance
feature_selector.plot_importance(top_n=20, save_path='importance.png')
```

#### Save/Load

```python
# Save selector
feature_selector.save(run_dir)

# Load selector
feature_selector = FeatureSelector.load(run_dir)
```

### Integration with Pipeline

```python
from src import DataProvider, FeaturesGenerator, FeatureSelector, TrainManager

# 1. Load data and generate features
data_provider = DataProvider()
df = data_provider.load_yahoo('BTC-USD', '2020-01-01', '2023-12-31')

features_gen = FeaturesGenerator()
df = features_gen.generate_features(df)
df = features_gen.create_target(df)

train_df, val_df, test_df = data_provider.split_data(df)
initial_features = features_gen.get_feature_names()

# 2. Select features BEFORE training
feature_selector = FeatureSelector(method='tree')
feature_selector.fit(train_df[initial_features], train_df['target'])

# Transform all datasets
train_selected = feature_selector.transform(train_df[initial_features])
val_selected = feature_selector.transform(val_df[initial_features])
test_selected = feature_selector.transform(test_df[initial_features])

# 3. Train with selected features
selected_features = feature_selector.get_selected_features()
train_manager = TrainManager()
train_output = train_manager.train(
    models=models,
    train_data=train_selected,
    feature_cols=selected_features
)
```

## Hyperparameter Optimization

### HyperparameterOptimizer

The `HyperparameterOptimizer` finds optimal parameters for your models.

#### Available Methods

1. **Random Search** (`method='random'`)
   - Samples random parameter combinations
   - Fast and effective
   - **Recommended for most cases**

2. **Grid Search** (`method='grid'`)
   - Exhaustive search
   - Slower but thorough
   - Use for small parameter spaces

3. **Bayesian Optimization** (`method='bayesian'`)
   - Smart search using Bayesian inference
   - Best for expensive models
   - Requires `scikit-optimize`

#### Basic Usage

```python
from src import HyperparameterOptimizer
from sklearn.ensemble import RandomForestClassifier

# Create optimizer
optimizer = HyperparameterOptimizer(
    method='random',
    cv=5,           # Cross-validation folds
    n_jobs=-1,      # Use all CPU cores
    verbose=1
)

# Define parameter space
param_space = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Optimize
results = optimizer.optimize(
    model_name='random_forest',
    model_class=RandomForestClassifier,
    param_space=param_space,
    X_train=X_train,
    y_train=y_train,
    scoring='f1_weighted',
    n_iter=50  # Number of iterations for random/bayesian
)

# Get best parameters
best_params = optimizer.get_best_params('random_forest')
best_score = optimizer.get_best_score('random_forest')
```

#### Optimize Multiple Models

```python
# Define multiple models
models_config = {
    'xgboost': {
        'model_class': xgb.XGBClassifier,
        'param_space': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    },
    'random_forest': {
        'model_class': RandomForestClassifier,
        'param_space': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15]
        }
    }
}

# Optimize all
results = optimizer.optimize_multiple(
    models_config=models_config,
    X_train=X_train,
    y_train=y_train,
    scoring='f1_weighted',
    n_iter=50
)
```

#### Default Parameter Spaces

```python
# Get default parameter spaces for common models
default_spaces = HyperparameterOptimizer.get_default_param_spaces()

# Available for: xgboost, catboost, random_forest, logistic_regression
xgb_space = default_spaces['xgboost']
```

#### Save Results

```python
# Save optimization results
optimizer.save_results(run_dir)

# Saved files:
# - best_params.json
# - best_scores.json
# - {model_name}/cv_results.csv
# - {model_name}/summary.json
```

#### Visualization

```python
# Plot optimization history
optimizer.plot_optimization_history(
    model_name='xgboost',
    save_path='optimization_history.png'
)
```

### Integration with Training

```python
from src import ModelManager, TrainManager, HyperparameterOptimizer

# 1. Optimize hyperparameters
optimizer = HyperparameterOptimizer(method='random')
results = optimizer.optimize(
    model_name='xgboost',
    model_class=xgb.XGBClassifier,
    param_space=param_space,
    X_train=X_train,
    y_train=y_train
)

# 2. Get best parameters
best_params = optimizer.get_best_params('xgboost')

# 3. Create model with optimized parameters
model_manager = ModelManager()
model = model_manager.create_model('xgboost', **best_params)

# 4. Train with optimized model
train_manager = TrainManager()
train_output = train_manager.train(
    models={'xgboost': model},
    train_data=train_df
)
```

## GPU Acceleration

### Supported Models

GPU acceleration is available for:
- **XGBoost** - CUDA-based tree construction
- **CatBoost** - GPU training
- **CNN Models** - TensorFlow GPU support (SimpleCNN, DeepCNN, ResidualCNN)

### Enabling GPU

#### Method 1: ModelManager

```python
from src import ModelManager

# Enable GPU for all supported models
model_manager = ModelManager(use_gpu=True)

# Create models (will use GPU if available)
models = model_manager.create_models(['xgboost', 'catboost', 'simple_cnn'])
```

#### Method 2: Direct Model Creation

```python
from src import XGBoostModel, CatBoostModel, SimpleCNN

# XGBoost with GPU
xgb_model = XGBoostModel(use_gpu=True)

# CatBoost with GPU
cat_model = CatBoostModel(use_gpu=True)

# CNN with GPU (TensorFlow auto-detects)
cnn_model = SimpleCNN(use_gpu=True)
```

### GPU Requirements

#### XGBoost
- CUDA-enabled GPU
- XGBoost compiled with GPU support
- Install: `pip install xgboost[gpu]` or build from source

#### CatBoost
- CUDA-enabled GPU
- CatBoost with GPU support
- Install: `pip install catboost` (GPU support included)

#### TensorFlow (CNNs)
- CUDA-enabled GPU
- cuDNN library
- Install: `pip install tensorflow[and-cuda]` or `tensorflow-gpu`

### Checking GPU Availability

```python
# XGBoost
import xgboost as xgb
print("XGBoost GPU available:", xgb.gpu.is_available())

# TensorFlow
import tensorflow as tf
print("TensorFlow GPUs:", tf.config.list_physical_devices('GPU'))

# CatBoost
from catboost import CatBoost
# CatBoost will print GPU info when task_type='GPU'
```

### GPU Configuration

#### XGBoost GPU Parameters

```python
xgb_model = XGBoostModel(
    use_gpu=True,
    tree_method='gpu_hist',      # GPU histogram algorithm
    gpu_id=0,                    # GPU device ID
    predictor='gpu_predictor'    # GPU predictor
)
```

#### CatBoost GPU Parameters

```python
cat_model = CatBoostModel(
    use_gpu=True,
    task_type='GPU',
    devices='0'  # GPU device ID
)
```

#### TensorFlow GPU Configuration

```python
# Automatic memory growth (prevents allocating all GPU memory)
# Configured automatically when use_gpu=True

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

## Complete Example

```python
from src import (
    DataProvider, FeaturesGenerator,
    FeatureSelector, HyperparameterOptimizer,
    ModelManager, TrainManager
)
import xgboost as xgb

# 1. Load data
data_provider = DataProvider()
df = data_provider.load_yahoo('BTC-USD', '2020-01-01', '2023-12-31')

features_gen = FeaturesGenerator()
df = features_gen.generate_features(df)
df = features_gen.create_target(df)

train_df, val_df, test_df = data_provider.split_data(df)
initial_features = features_gen.get_feature_names()

# 2. Feature Selection
feature_selector = FeatureSelector(method='tree')
feature_selector.fit(train_df[initial_features], train_df['target'])

selected_features = feature_selector.get_selected_features()
train_selected = feature_selector.transform(train_df[initial_features])
train_selected['target'] = train_df['target'].values

# 3. Hyperparameter Optimization
optimizer = HyperparameterOptimizer(method='random', cv=3)
results = optimizer.optimize(
    model_name='xgboost',
    model_class=xgb.XGBClassifier,
    param_space={
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    },
    X_train=train_selected[selected_features].values,
    y_train=train_selected['target'].values,
    n_iter=20
)

# 4. Train with optimized parameters and GPU
best_params = optimizer.get_best_params('xgboost')
model_manager = ModelManager(use_gpu=True)
model = model_manager.create_model('xgboost', **best_params)

train_manager = TrainManager()
train_output = train_manager.train(
    models={'xgboost': model},
    train_data=train_selected,
    feature_cols=selected_features
)

# 5. Test
test_selected = feature_selector.transform(test_df[initial_features])
test_selected['target'] = test_df['target'].values

test_results = train_manager.test(
    test_data=test_selected,
    feature_cols=selected_features
)

print(f"Test F1 Score: {test_results['xgboost']['f1_score']:.4f}")
```

## Best Practices

### Feature Selection

1. **Always select features BEFORE training** - Reduces overfitting and training time
2. **Use tree-based method** - Fast and effective for most cases
3. **Try different methods** - Compare results from multiple methods
4. **Keep at least 10-20 features** - Too few features may lose information
5. **Save the selector** - Reuse for consistent feature selection

### Hyperparameter Optimization

1. **Start with random search** - Faster than grid search
2. **Use cross-validation** - More robust parameter estimates
3. **Limit iterations** - 20-50 iterations usually sufficient for random search
4. **Use default spaces** - Good starting point for common models
5. **Optimize on validation set** - Avoid overfitting to test set

### GPU Acceleration

1. **Enable for large datasets** - GPU shines with large data
2. **Monitor GPU memory** - Use memory growth for TensorFlow
3. **Batch size matters** - Larger batches better utilize GPU
4. **CPU fallback** - Code works without GPU
5. **Profile performance** - Measure actual speedup

## Troubleshooting

### Feature Selection Issues

**Issue**: Too many features dropped
```python
# Solution: Adjust threshold or n_features
feature_selector.fit(X, y, n_features=50)  # Keep at least 50
```

**Issue**: Feature selector not fitted
```python
# Solution: Call fit() before transform()
feature_selector.fit(X_train, y_train)
X_transformed = feature_selector.transform(X_test)
```

### Optimization Issues

**Issue**: Optimization too slow
```python
# Solution: Use random search with fewer iterations
optimizer = HyperparameterOptimizer(method='random')
results = optimizer.optimize(..., n_iter=20)  # Reduce iterations
```

**Issue**: scikit-optimize not found
```bash
# Solution: Install for Bayesian optimization
pip install scikit-optimize
```

### GPU Issues

**Issue**: GPU not detected
```python
# Check GPU availability
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# For XGBoost
import xgboost as xgb
print(xgb.gpu.is_available())
```

**Issue**: Out of GPU memory
```python
# Solution: Enable memory growth or reduce batch size
# TensorFlow: Automatic with use_gpu=True
# XGBoost/CatBoost: Reduce n_estimators or max_depth
```

---

**Version**: 0.2.0  
**Last Updated**: 2025-11-18
