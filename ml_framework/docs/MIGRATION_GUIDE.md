## Migration Guide: v0.1.0 → v0.2.0

This guide helps you migrate from the old architecture to the new manager-based architecture.

## What Changed?

### Old Structure
```
src/
├── data_provider.py
├── features_generator.py
├── model_manager.py      # Only configuration
├── ml_trainer.py         # Training only
├── ml_tester.py          # Testing only
└── backtester.py
```

### New Structure
```
src/
├── data_provider.py      # Same (backward compatible)
├── features_generator.py # Same (backward compatible)
├── managers/             # NEW: All managers
│   ├── model_manager.py  # Enhanced: Creates models
│   ├── train_manager.py  # NEW: Unified train/test
│   ├── scaler_manager.py # NEW: Scaling management
│   ├── mlflow_manager.py # NEW: Experiment tracking
│   ├── backtest_manager.py # Enhanced: Multi-backend
│   ├── result_manager.py # NEW: Result aggregation
│   ├── visualization_manager.py # NEW: HTML reports
│   └── pipeline_manager.py # NEW: Workflow orchestration
└── models_lib/           # NEW: Model implementations
    ├── base_model.py     # Base class with auto conversion
    ├── xgboost_model.py
    ├── catboost_model.py
    ├── linear_model.py
    └── cnn_models.py
```

## Key Changes

### 1. ModelManager
**Old**: Only stored configurations
```python
model_manager = ModelManager()
configs = model_manager.get_models()
```

**New**: Creates actual model instances
```python
model_manager = ModelManager()
models = model_manager.create_models()  # Returns model instances
```

### 2. Training & Testing
**Old**: Separate classes
```python
trainer = ML_Trainer()
results = trainer.train(df, model_configs=configs)

tester = ML_Tester()
test_results = tester.evaluate(test_df, models, scaler)
```

**New**: Unified TrainManager
```python
train_manager = TrainManager()
train_output = train_manager.train(models, train_df, val_data=val_df)
test_results = train_manager.test(test_df)
```

### 3. Scaling
**Old**: Built into trainer
```python
trainer = ML_Trainer()
results = trainer.train(df, scale_features=True)
# Scaler stored internally
```

**New**: Dedicated ScalerManager
```python
scaler_manager = ScalerManager(scaler_type='standard')
scaler_manager.fit(train_df, only_float=True)
train_scaled = scaler_manager.transform(train_df)
scaler_manager.save(run_dir)
```

### 4. Results Structure
**Old**: Flat models directory
```
models/
└── YYYY-MM-DD_HH-MM-SS/
    ├── model1.joblib
    ├── scaler.joblib
    └── metadata.joblib
```

**New**: Comprehensive results structure
```
results/
└── YYYY-MM-DD_HH-MM-SS/
    ├── models/
    │   └── *.joblib
    ├── reports/
    │   ├── train/
    │   ├── test/
    │   └── backtest/
    ├── scaler.joblib
    └── metadata.joblib
```

## Migration Examples

### Example 1: Basic Training

**Old Code**:
```python
from src import DataProvider, FeaturesGenerator, ModelManager, ML_Trainer

# Data
data_provider = DataProvider()
df = data_provider.load_yahoo('BTC-USD', '2020-01-01', '2023-12-31')

# Features
features_gen = FeaturesGenerator()
df = features_gen.generate_features(df)
df = features_gen.create_target(df)

# Split
train_df, val_df, test_df = data_provider.split_data(df)

# Models
model_manager = ModelManager()
configs = model_manager.get_models()

# Train
trainer = ML_Trainer()
results = trainer.train(
    df=train_df,
    target_col='target',
    model_configs=configs
)
```

**New Code**:
```python
from src import DataProvider, FeaturesGenerator, ModelManager, TrainManager

# Data (same)
data_provider = DataProvider()
df = data_provider.load_yahoo('BTC-USD', '2020-01-01', '2023-12-31')

# Features (same)
features_gen = FeaturesGenerator()
df = features_gen.generate_features(df)
df = features_gen.create_target(df)

# Split (same)
train_df, val_df, test_df = data_provider.split_data(df)

# Models (CHANGED)
model_manager = ModelManager()
models = model_manager.create_models()  # Creates instances

# Train (CHANGED)
train_manager = TrainManager()
train_output = train_manager.train(
    models=models,  # Pass instances, not configs
    train_data=train_df,
    val_data=val_df,
    target_col='target'
)
```

### Example 2: Complete Pipeline

**Old Code**:
```python
# Multiple steps with different classes
data_provider = DataProvider()
features_gen = FeaturesGenerator()
model_manager = ModelManager()
trainer = ML_Trainer()
tester = ML_Tester()
backtester = Backtester()

# ... many lines of code ...
```

**New Code**:
```python
from src import PipelineManager

# One manager, one call
pipeline = PipelineManager()
results = pipeline.run_complete_pipeline(
    ticker='BTC-USD',
    start_date='2020-01-01',
    end_date='2023-12-31',
    model_names=['xgboost', 'random_forest'],
    backtest=True
)
```

### Example 3: Custom Models

**Old Code**:
```python
# Had to modify ModelManager and ML_Trainer
# No unified interface
```

**New Code**:
```python
from src import BaseModel

class MyModel(BaseModel):
    def _fit(self, X, y, **kwargs):
        # Your logic
        pass
    
    def _predict(self, X, **kwargs):
        # Your logic
        return predictions

# Use like any other model
model = MyModel()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Backward Compatibility

The following components remain **fully backward compatible**:
- `DataProvider`
- `FeaturesGenerator`

You can continue using them exactly as before.

## Deprecated Components

The following are **deprecated** but still available:
- `ml_trainer.py` → Use `TrainManager`
- `ml_tester.py` → Use `TrainManager.test()`
- Old `model_manager.py` → Use new `ModelManager`

## New Features Not Available in Old Version

1. **Automatic target conversion** (BaseModel)
2. **CNN models** (SimpleCNN, DeepCNN, ResidualCNN)
3. **CatBoost support**
4. **MLFlow integration**
5. **HTML visualization reports**
6. **Unified train/test manager**
7. **Dedicated scaler management**
8. **Result aggregation**
9. **Pipeline orchestration**

## Recommended Migration Path

### Phase 1: Keep Using Old Code
- No changes needed
- Old code continues to work

### Phase 2: Try PipelineManager
- Use `PipelineManager` for new experiments
- Compare with old workflow

### Phase 3: Migrate Gradually
- Replace `ML_Trainer` with `TrainManager`
- Replace `ML_Tester` with `TrainManager.test()`
- Use new `ModelManager.create_models()`

### Phase 4: Full Migration
- Use all new managers
- Leverage new features (MLFlow, HTML reports, etc.)

## Common Issues

### Issue 1: "model_configs not recognized"
**Old**: `trainer.train(df, model_configs=configs)`  
**New**: `train_manager.train(models=models, train_data=df)`

### Issue 2: "Scaler not found"
**Old**: Scaler was internal to trainer  
**New**: Access via `train_output['scaler_manager']`

### Issue 3: "Results directory structure changed"
**Old**: `models/timestamp/`  
**New**: `results/timestamp/models/`

## Getting Help

1. Check `NEW_ARCHITECTURE.md` for detailed documentation
2. Review `examples/new_pipeline_example.py`
3. Review `examples/step_by_step_example.py`
4. Check the old `examples/basic_workflow.py` for comparison

## Quick Reference

| Old | New |
|-----|-----|
| `ML_Trainer` | `TrainManager` |
| `ML_Tester` | `TrainManager.test()` |
| `model_configs` | `models` (instances) |
| `trainer.train(df, model_configs=...)` | `train_manager.train(models=..., train_data=...)` |
| `tester.evaluate(df, models, scaler)` | `train_manager.test(test_data=...)` |
| No scaler management | `ScalerManager` |
| No visualization | `VisualizationManager` |
| No pipeline | `PipelineManager` |
| Manual workflow | `pipeline.run_complete_pipeline()` |

---

**Need help?** Open an issue or check the examples directory.
