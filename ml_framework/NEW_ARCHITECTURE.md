# ML Framework - New Architecture (v0.2.0)

## Overview

The ML Framework has been restructured with a manager-based architecture that provides better organization, flexibility, and comprehensive result tracking.

## Key Improvements

### 1. **Manager-Based Architecture**
All functionality is organized into specialized managers:
- `ModelManager` - Model creation and management
- `TrainManager` - Unified training and testing
- `ScalerManager` - Feature scaling with save/load
- `MLFlowManager` - Experiment tracking
- `BacktestManager` - Multi-backend backtesting
- `ResultManager` - Result aggregation and processing
- `VisualizationManager` - HTML report generation
- `PipelineManager` - Complete workflow orchestration

### 2. **BaseModel with Automatic Target Conversion**
All models inherit from `BaseModel` which automatically handles:
- Target conversion from -1/0/1 to 0/1/2 (for DL/ML compatibility)
- Reverse conversion of predictions
- Unified interface across all model types

### 3. **Comprehensive Model Support**
- **Traditional ML**: Logistic Regression, Random Forest
- **Gradient Boosting**: XGBoost, CatBoost
- **Deep Learning**: SimpleCNN, DeepCNN, ResidualCNN
- **Linear Models**: Linear Regression

### 4. **Structured Results Directory**
```
results/
└── YYYY-MM-DD_HH-MM-SS/          # Timestamped run folder
    ├── models/                    # Trained models (joblib)
    │   ├── model1.joblib
    │   ├── model2.joblib
    │   └── ...
    ├── reports/
    │   ├── train/
    │   │   └── train_report.html  # Training visualizations
    │   ├── test/
    │   │   └── test_report.html   # Test visualizations
    │   └── backtest/
    │       └── backtest_report.html  # Backtest visualizations
    ├── scaler.joblib              # Fitted scaler
    └── metadata.joblib            # Run metadata
```

### 5. **HTML Visualization Reports**
- All charts stacked vertically (not subplots)
- Interactive Plotly charts
- Separate reports for train/test/backtest phases
- Comprehensive equity curves for all backtests

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      PipelineManager                             │
│                  (Orchestrates Everything)                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌────────────────┐ ┌────────────┐ ┌──────────────┐
│ DataProvider   │ │  Features  │ │ ModelManager │
│                │ │  Generator │ │              │
└────────────────┘ └────────────┘ └──────┬───────┘
                                          │
                                          ▼
                                  ┌───────────────┐
                                  │ TrainManager  │
                                  │ (Train & Test)│
                                  └───────┬───────┘
                                          │
                         ┌────────────────┼────────────────┐
                         │                │                │
                         ▼                ▼                ▼
                  ┌─────────────┐ ┌──────────────┐ ┌─────────────┐
                  │   Scaler    │ │   Backtest   │ │   MLFlow    │
                  │   Manager   │ │   Manager    │ │   Manager   │
                  └─────────────┘ └──────────────┘ └─────────────┘
                                          │
                         ┌────────────────┼────────────────┐
                         │                                 │
                         ▼                                 ▼
                  ┌─────────────┐                  ┌──────────────┐
                  │   Result    │                  │Visualization │
                  │   Manager   │                  │   Manager    │
                  └─────────────┘                  └──────────────┘
```

## Component Details

### ModelManager
**Purpose**: Create and manage ML models

**Features**:
- Create models from configuration
- Support for multiple model types
- Enable/disable models
- Save/load with metadata

**Example**:
```python
from src import ModelManager

model_manager = ModelManager()
model_manager.enable_model('xgboost', True)
model_manager.enable_model('catboost', True)

models = model_manager.create_models()
```

### TrainManager
**Purpose**: Unified training and testing

**Features**:
- Train multiple models
- Automatic feature scaling
- Validation support
- Test/evaluate models
- Comprehensive metrics

**Example**:
```python
from src import TrainManager

train_manager = TrainManager(use_scaler=True)

# Train
train_output = train_manager.train(
    models=models,
    train_data=train_df,
    val_data=val_df,
    target_col='target',
    feature_cols=feature_cols
)

# Test
test_results = train_manager.test(
    test_data=test_df,
    target_col='target',
    feature_cols=feature_cols
)
```

### ScalerManager
**Purpose**: Manage feature scaling

**Features**:
- Multiple scaler types (Standard, MinMax, Robust)
- Scale only float fields or selected fields
- Save/load scaler state
- DataFrame and array support

**Example**:
```python
from src import ScalerManager

scaler = ScalerManager(scaler_type='standard')
scaler.fit(train_df, only_float=True)
train_scaled = scaler.transform(train_df)
test_scaled = scaler.transform(test_df)

scaler.save(run_dir)
```

### BacktestManager
**Purpose**: Backtest trading strategies

**Features**:
- Multiple backends (NoLib, Backtrader, Backtesting.py)
- Position sizing and commissions
- Performance metrics
- Equity curve generation

**Example**:
```python
from src import BacktestManager

backtest_manager = BacktestManager(
    backend='nolib',
    initial_capital=10000,
    commission=0.001
)

results = backtest_manager.run(
    data=test_df,
    model=trained_model,
    scaler_manager=scaler,
    feature_cols=feature_cols
)
```

### MLFlowManager
**Purpose**: Track experiments with MLFlow

**Features**:
- Connect to local MLFlow server
- Log parameters, metrics, artifacts
- Track multiple runs
- Model logging

**Example**:
```python
from src import MLFlowManager

mlflow_manager = MLFlowManager(tracking_uri="http://localhost:5000")
mlflow_manager.connect()
mlflow_manager.start_run(run_name="experiment_1")

mlflow_manager.log_params({'learning_rate': 0.1})
mlflow_manager.log_metrics({'accuracy': 0.85})
mlflow_manager.log_model(model, 'my_model')

mlflow_manager.end_run()
```

### ResultManager
**Purpose**: Aggregate and process results

**Features**:
- Collect results from all managers
- Compare models
- Generate summaries
- Save structured results

**Example**:
```python
from src import ResultManager

result_manager = ResultManager()
result_manager.add_train_results(train_results)
result_manager.add_test_results(test_results)
result_manager.add_backtest_results('model1', backtest_results)

result_manager.print_summary()
result_manager.save_results(run_dir)
```

### VisualizationManager
**Purpose**: Generate HTML visualization reports

**Features**:
- Interactive Plotly charts
- Separate reports for train/test/backtest
- Charts stacked vertically
- Comprehensive equity curves

**Example**:
```python
from src import VisualizationManager

viz_manager = VisualizationManager()
viz_manager.create_train_report(train_results, run_dir)
viz_manager.create_test_report(test_results, run_dir)
viz_manager.create_backtest_report(backtest_results, run_dir)
```

### PipelineManager
**Purpose**: Orchestrate complete workflow

**Features**:
- End-to-end pipeline execution
- Automatic directory structure
- Coordinated manager execution
- One-line complete pipeline

**Example**:
```python
from src import PipelineManager

pipeline = PipelineManager(results_dir='results', use_mlflow=False)

results = pipeline.run_complete_pipeline(
    ticker='BTC-USD',
    start_date='2020-01-01',
    end_date='2023-12-31',
    model_names=['xgboost', 'random_forest'],
    feature_set='basic',
    backtest=True
)
```

## BaseModel System

All models inherit from `BaseModel` which provides:

### Automatic Target Conversion
```python
# Input targets: [-1, 0, 1] or any other values
# Automatically converted to: [0, 1, 2]
# Predictions automatically converted back to original space
```

### Unified Interface
```python
model = XGBoostModel(n_estimators=100)
model.fit(X_train, y_train)
predictions = model.predict(X_test)  # Automatically in original target space
probabilities = model.predict_proba(X_test)
```

### Supported Models
- `LogisticRegressionModel`
- `RandomForestModel`
- `XGBoostModel`
- `CatBoostModel`
- `LinearRegressionModel`
- `SimpleCNN`
- `DeepCNN`
- `ResidualCNN`

## Usage Patterns

### Pattern 1: Complete Pipeline (Easiest)
```python
from src import PipelineManager

pipeline = PipelineManager()
results = pipeline.run_complete_pipeline(
    ticker='BTC-USD',
    start_date='2020-01-01',
    end_date='2023-12-31',
    model_names=['xgboost', 'random_forest']
)
```

### Pattern 2: Step-by-Step (More Control)
```python
from src import (
    DataProvider, FeaturesGenerator, ModelManager,
    TrainManager, BacktestManager, ResultManager
)

# 1. Data
data_provider = DataProvider()
df = data_provider.load_yahoo('BTC-USD', '2020-01-01', '2023-12-31')

# 2. Features
features_gen = FeaturesGenerator()
df = features_gen.generate_features(df)
df = features_gen.create_target(df)

# 3. Models
model_manager = ModelManager()
models = model_manager.create_models(['xgboost', 'random_forest'])

# 4. Train
train_manager = TrainManager()
train_output = train_manager.train(models, train_df, val_data=val_df)

# 5. Test
test_results = train_manager.test(test_df)

# 6. Backtest
backtest_manager = BacktestManager()
# ... and so on
```

### Pattern 3: Custom Models
```python
from src import BaseModel
import numpy as np

class MyCustomModel(BaseModel):
    def __init__(self, name='CustomModel', **params):
        super().__init__(name)
        self.params = params
        
    def _fit(self, X, y, **kwargs):
        # Your training logic here
        # y is already converted to 0, 1, 2, ...
        pass
        
    def _predict(self, X, **kwargs):
        # Your prediction logic
        # Return predictions as 0, 1, 2, ...
        # Will be automatically converted back
        return predictions
```

## Migration from Old Architecture

### Old Code
```python
from src import ML_Trainer, ML_Tester, ModelManager

model_manager = ModelManager()
configs = model_manager.get_models()

trainer = ML_Trainer()
results = trainer.train(df, model_configs=configs)

tester = ML_Tester()
test_results = tester.evaluate(test_df, models, scaler)
```

### New Code
```python
from src import ModelManager, TrainManager

model_manager = ModelManager()
models = model_manager.create_models()

train_manager = TrainManager()
train_output = train_manager.train(models, train_df, val_data=val_df)
test_results = train_manager.test(test_df)
```

## Best Practices

1. **Use PipelineManager for quick experiments**
2. **Use individual managers for fine-grained control**
3. **Always save results with timestamps**
4. **Enable MLFlow for experiment tracking**
5. **Review HTML reports for comprehensive analysis**
6. **Use BaseModel for custom model implementations**

## Future Enhancements

- [ ] Full Backtrader integration
- [ ] Full Backtesting.py integration
- [ ] Hyperparameter optimization manager
- [ ] Real-time prediction manager
- [ ] Model ensemble manager
- [ ] Feature selection manager

---

**Version**: 0.2.0  
**Last Updated**: 2025-11-18
