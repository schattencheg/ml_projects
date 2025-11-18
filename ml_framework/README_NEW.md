# ML Framework v0.2.0 - Quick Start Guide

A comprehensive machine learning framework for financial data analysis with manager-based architecture.

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd ml_framework

# Install dependencies
pip install -r requirements.txt
```

### 30-Second Example

```python
from src import PipelineManager

# Run complete ML pipeline in one line
pipeline = PipelineManager()
results = pipeline.run_complete_pipeline(
    ticker='BTC-USD',
    start_date='2020-01-01',
    end_date='2023-12-31',
    model_names=['xgboost', 'random_forest'],
    backtest=True
)

# Results automatically saved to: results/YYYY-MM-DD_HH-MM-SS/
```

## 📁 Results Structure

After running, you'll get:

```
results/2025-11-18_14-30-00/
├── models/
│   ├── xgboost.joblib
│   └── random_forest.joblib
├── reports/
│   ├── train/train_report.html      ← Open in browser
│   ├── test/test_report.html        ← Open in browser
│   └── backtest/backtest_report.html ← Open in browser
├── scaler.joblib
└── metadata.joblib
```

## 🎯 Key Features

### 1. **Multiple Model Types**
- Traditional ML: Logistic Regression, Random Forest
- Gradient Boosting: XGBoost, CatBoost
- Deep Learning: SimpleCNN, DeepCNN, ResidualCNN
- All with automatic target conversion (-1/0/1 → 0/1/2)

### 2. **Comprehensive Workflow**
- Data loading (Yahoo Finance, CSV)
- Feature generation (Technical indicators)
- Model training with validation
- Testing with metrics
- Backtesting with equity curves
- HTML visualization reports

### 3. **Manager-Based Architecture**
- `ModelManager` - Model creation
- `TrainManager` - Training & testing
- `ScalerManager` - Feature scaling
- `BacktestManager` - Strategy backtesting
- `ResultManager` - Result aggregation
- `VisualizationManager` - HTML reports
- `PipelineManager` - Complete orchestration
- `MLFlowManager` - Experiment tracking

## 📚 Usage Patterns

### Pattern 1: Complete Pipeline (Easiest)

```python
from src import PipelineManager

pipeline = PipelineManager(use_mlflow=False)
results = pipeline.run_complete_pipeline(
    ticker='BTC-USD',
    start_date='2020-01-01',
    end_date='2023-12-31',
    model_names=['xgboost', 'catboost', 'random_forest'],
    feature_set='basic',
    backtest=True
)
```

### Pattern 2: Step-by-Step (More Control)

```python
from src import (
    DataProvider, FeaturesGenerator,
    ModelManager, TrainManager, BacktestManager
)

# 1. Load data
data_provider = DataProvider()
df = data_provider.load_yahoo('BTC-USD', '2020-01-01', '2023-12-31')

# 2. Generate features
features_gen = FeaturesGenerator()
df = features_gen.generate_features(df, feature_set='basic')
df = features_gen.create_target(df, future_bars=5, threshold=0.02)

# 3. Split data
train_df, val_df, test_df = data_provider.split_data(df)

# 4. Create models
model_manager = ModelManager()
models = model_manager.create_models(['xgboost', 'random_forest'])

# 5. Train
train_manager = TrainManager(use_scaler=True)
train_output = train_manager.train(
    models=models,
    train_data=train_df,
    val_data=val_df
)

# 6. Test
test_results = train_manager.test(test_data=test_df)

# 7. Backtest
backtest_manager = BacktestManager(backend='nolib')
for model_name, model in train_output['models'].items():
    results = backtest_manager.run(
        data=test_df,
        model=model,
        scaler_manager=train_output['scaler_manager']
    )
```

### Pattern 3: Custom Models

```python
from src import BaseModel
import numpy as np

class MyCustomModel(BaseModel):
    def __init__(self, name='MyModel'):
        super().__init__(name)
        self.model = None
        
    def _fit(self, X, y, **kwargs):
        # Your training logic
        # y is already converted to 0, 1, 2, ...
        self.model = YourModelClass()
        self.model.fit(X, y)
        
    def _predict(self, X, **kwargs):
        # Your prediction logic
        # Return predictions as 0, 1, 2, ...
        # Will be automatically converted back to original space
        return self.model.predict(X)

# Use it
model = MyCustomModel()
model.fit(X_train, y_train)  # Handles -1/0/1 automatically
predictions = model.predict(X_test)  # Returns in original space
```

## 🔧 Configuration

### Enable/Disable Models

```python
from src import ModelManager

model_manager = ModelManager()

# Enable specific models
model_manager.enable_model('xgboost', True)
model_manager.enable_model('catboost', True)
model_manager.enable_model('simple_cnn', True)

# Disable a model
model_manager.enable_model('logistic_regression', False)

# View configuration
model_manager.print_config()
```

### Feature Scaling

```python
from src import ScalerManager

# Create scaler
scaler = ScalerManager(scaler_type='standard')  # or 'minmax', 'robust'

# Fit on training data
scaler.fit(train_df, only_float=True)

# Transform
train_scaled = scaler.transform(train_df)
test_scaled = scaler.transform(test_df)

# Save for later use
scaler.save(run_dir)

# Load
scaler = ScalerManager.load(run_dir)
```

### MLFlow Tracking

```python
from src import PipelineManager

# Start MLFlow server first:
# mlflow server --host 0.0.0.0 --port 5000

pipeline = PipelineManager(
    use_mlflow=True,
    mlflow_uri="http://localhost:5000"
)

results = pipeline.run_complete_pipeline(
    ticker='BTC-USD',
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# View results at http://localhost:5000
```

## 📊 Available Models

| Model | Type | Key Parameters |
|-------|------|----------------|
| `logistic_regression` | Traditional ML | `max_iter`, `random_state` |
| `random_forest` | Traditional ML | `n_estimators`, `max_depth` |
| `xgboost` | Gradient Boosting | `n_estimators`, `learning_rate` |
| `catboost` | Gradient Boosting | `iterations`, `depth` |
| `linear_regression` | Linear | - |
| `simple_cnn` | Deep Learning | `num_classes` |
| `deep_cnn` | Deep Learning | `num_classes` |
| `residual_cnn` | Deep Learning | `num_classes` |

## 📈 Visualization Reports

HTML reports include:

### Training Report
- Accuracy comparison
- Training time comparison
- Validation accuracy (if available)

### Test Report
- Metrics comparison (Accuracy, Precision, Recall, F1)
- Individual model performance
- Confusion matrices

### Backtest Report
- Performance metrics comparison
- Individual equity curves
- **Comprehensive equity curves** (all models together)
- Returns distribution

All charts are **stacked vertically** for easy viewing.

## 🎓 Examples

Check the `examples/` directory:

1. **`new_pipeline_example.py`** - Using PipelineManager
2. **`step_by_step_example.py`** - Using individual managers
3. **`basic_workflow.py`** - Old architecture (still works)

Run an example:
```bash
python examples/new_pipeline_example.py
```

## 📖 Documentation

- **`NEW_ARCHITECTURE.md`** - Complete architecture documentation
- **`MIGRATION_GUIDE.md`** - Migrating from v0.1.0
- **`RESTRUCTURE_SUMMARY.md`** - Implementation summary

## 🔄 Migration from v0.1.0

Old code still works! But to use new features:

**Old**:
```python
from src import ML_Trainer, ML_Tester

trainer = ML_Trainer()
results = trainer.train(df, model_configs=configs)

tester = ML_Tester()
test_results = tester.evaluate(test_df, models, scaler)
```

**New**:
```python
from src import TrainManager

train_manager = TrainManager()
train_output = train_manager.train(models, train_df, val_data=val_df)
test_results = train_manager.test(test_df)
```

See `MIGRATION_GUIDE.md` for details.

## 🛠️ Requirements

**Core**:
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- plotly >= 5.14.0

**Optional**:
- xgboost >= 1.7.0
- catboost >= 1.2.0
- tensorflow >= 2.10.0
- mlflow >= 2.0.0
- backtrader >= 1.9.76

## 🎯 Common Use Cases

### 1. Quick Experiment
```python
from src import PipelineManager

pipeline = PipelineManager()
results = pipeline.run_complete_pipeline(
    ticker='AAPL',
    start_date='2020-01-01',
    end_date='2023-12-31',
    model_names=['xgboost']
)
```

### 2. Compare Multiple Models
```python
pipeline = PipelineManager()
results = pipeline.run_complete_pipeline(
    ticker='BTC-USD',
    start_date='2020-01-01',
    end_date='2023-12-31',
    model_names=['xgboost', 'catboost', 'random_forest', 'simple_cnn']
)
```

### 3. Production Pipeline
```python
from src import (
    ModelManager, TrainManager, ScalerManager,
    ResultManager, VisualizationManager
)
from datetime import datetime

# Setup
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
run_dir = Path('results') / timestamp

# Train
model_manager = ModelManager()
models = model_manager.create_models(['xgboost'])

train_manager = TrainManager()
train_output = train_manager.train(models, train_df, val_data=val_df)

# Save
model_manager.save_models(train_output['models'], run_dir)
train_output['scaler_manager'].save(run_dir)

# Later: Load and predict
models, metadata = model_manager.load_models(run_dir)
scaler = ScalerManager.load(run_dir)

predictions = models['xgboost'].predict(scaler.transform(new_data))
```

## 🐛 Troubleshooting

**Issue**: Import errors
```bash
# Make sure you're in the project root
cd ml_framework
python examples/new_pipeline_example.py
```

**Issue**: MLFlow connection failed
```bash
# Start MLFlow server first
mlflow server --host 0.0.0.0 --port 5000

# Or disable MLFlow
pipeline = PipelineManager(use_mlflow=False)
```

**Issue**: Model not found
```python
# Check available models
model_manager = ModelManager()
model_manager.print_config()
```

## 🤝 Contributing

Suggestions for improvements are welcome! The architecture is designed to be extensible.

## 📝 License

[Your License Here]

## 🔗 Links

- [Architecture Documentation](NEW_ARCHITECTURE.md)
- [Migration Guide](MIGRATION_GUIDE.md)
- [Implementation Summary](RESTRUCTURE_SUMMARY.md)

---

**Version**: 0.2.0  
**Last Updated**: 2025-11-18

**Happy Trading! 📈🚀**
