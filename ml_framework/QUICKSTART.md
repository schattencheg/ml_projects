# Quick Start Guide

Get started with the ML Framework in 5 minutes!

## Installation

```bash
# Navigate to project directory
cd ml_framework

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

### 1. Simple Example (Minimal Code)

```python
from src.data_provider import DataProvider
from src.features_generator import FeaturesGenerator
from src.ml_trainer import ML_Trainer

# Load data
data_provider = DataProvider()
df = data_provider.load_yahoo('BTC-USD', '2020-01-01', '2023-12-31')

# Generate features and target
features_gen = FeaturesGenerator()
df = features_gen.generate_features(df, feature_set='basic')
df = features_gen.create_target(df, future_bars=5, threshold=0.02)

# Train models
trainer = ML_Trainer()
results = trainer.train(df, feature_cols=features_gen.get_feature_names())

print(f"Best model: {results['best_model']}")
```

### 2. Complete Workflow

Run the example script:

```bash
python examples/basic_workflow.py
```

This demonstrates the complete workflow:
1. Load data from Yahoo Finance
2. Generate technical indicators
3. Create target variable
4. Split data (train/val/test)
5. Train multiple models
6. Evaluate on test data
7. Save models with versioning
8. Backtest the best model

## Core Classes Overview

### DataProvider
```python
from src.data_provider import DataProvider

provider = DataProvider()

# Load from Yahoo Finance
df = provider.load_yahoo('BTC-USD', '2020-01-01', '2023-12-31')

# Or load from CSV
df = provider.load_csv('data/my_data.csv')

# Clean and split
df = provider.clean_data(df)
train_df, val_df, test_df = provider.split_data(df)
```

### FeaturesGenerator
```python
from src.features_generator import FeaturesGenerator

gen = FeaturesGenerator()

# Generate features
df = gen.generate_features(df, feature_set='basic')  # or 'advanced', 'all'

# Create target (classification)
df = gen.create_target(df, target_type='classification', 
                       future_bars=5, threshold=0.02)

# Get feature names
features = gen.get_feature_names()
```

### ModelManager
```python
from src.model_manager import ModelManager

manager = ModelManager()

# Configure models
manager.enable_model('xgboost', True)
manager.enable_model('lightgbm', False)
manager.print_config()

# Get enabled models
configs = manager.get_models()

# Save models
manager.save_models(models, scaler, metadata)

# Load latest models
models, scaler, metadata = manager.load_models('latest')
```

### ML_Trainer
```python
from src.ml_trainer import ML_Trainer

trainer = ML_Trainer()

results = trainer.train(
    df=train_df,
    target_col='target',
    feature_cols=feature_cols,
    model_configs=configs,
    scale_features=True
)

# Access results
models = results['models']
scaler = results['scaler']
best_model = results['best_model']
```

### ML_Tester
```python
from src.ml_tester import ML_Tester

tester = ML_Tester()

test_results = tester.evaluate(
    df=test_df,
    models=models,
    scaler=scaler,
    feature_cols=feature_cols
)

# Compare models
comparison = tester.compare_models(metric='f1_score')
```

### Backtester
```python
from src.backtester import Backtester

backtester = Backtester(
    initial_capital=10000,
    position_size=1.0,
    commission=0.001
)

results = backtester.run(
    df=test_df,
    model=best_model,
    scaler=scaler,
    feature_cols=feature_cols
)

# Visualize
backtester.plot_results()
```

## Common Workflows

### Workflow 1: Quick Model Training

```python
from src.data_provider import DataProvider
from src.features_generator import FeaturesGenerator
from src.ml_trainer import ML_Trainer

# Load and prepare
provider = DataProvider()
df = provider.load_yahoo('ETH-USD', '2021-01-01', '2023-12-31')

# Features
gen = FeaturesGenerator()
df = gen.generate_features(df)
df = gen.create_target(df)

# Train
trainer = ML_Trainer()
results = trainer.train(df, feature_cols=gen.get_feature_names())
trainer.print_results()
```

### Workflow 2: Model Comparison

```python
from src.model_manager import ModelManager
from src.ml_trainer import ML_Trainer
from src.ml_tester import ML_Tester

# Enable multiple models
manager = ModelManager()
manager.enable_model('logistic_regression', True)
manager.enable_model('random_forest', True)
manager.enable_model('xgboost', True)

# Train all
trainer = ML_Trainer()
results = trainer.train(df, model_configs=manager.get_models())

# Test and compare
tester = ML_Tester()
test_results = tester.evaluate(test_df, results['models'], results['scaler'])
comparison = tester.compare_models('f1_score')
print(comparison)
```

### Workflow 3: Save and Load Models

```python
from src.model_manager import ModelManager

manager = ModelManager()

# After training
save_dir = manager.save_models(
    models=trained_models,
    scaler=scaler,
    metadata={'ticker': 'BTC-USD', 'best_model': 'xgboost'}
)

# Later, load models
models, scaler, metadata = manager.load_models('latest')
# Or load specific version
models, scaler, metadata = manager.load_models('2025-11-13_16-30-00')

# List all versions
versions = manager.list_saved_models()
print(versions)
```

### Workflow 4: Backtesting

```python
from src.backtester import Backtester

# Create backtester
backtester = Backtester(
    initial_capital=10000,
    position_size=1.0,
    commission=0.001
)

# Run backtest
results = backtester.run(
    df=test_df,
    model=best_model,
    scaler=scaler,
    feature_cols=feature_cols
)

# View metrics
metrics = results['metrics']
print(f"Total Return: {metrics['total_return']*100:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")

# Plot
backtester.plot_results()
```

## Tips & Tricks

### 1. Feature Engineering
```python
# Try different feature sets
df_basic = gen.generate_features(df, feature_set='basic')
df_advanced = gen.generate_features(df, feature_set='advanced')
df_all = gen.generate_features(df, feature_set='all')
```

### 2. Target Definition
```python
# Classification: Will price increase by 2% in 5 days?
df = gen.create_target(df, target_type='classification', 
                       future_bars=5, threshold=0.02)

# Regression: Predict actual price change
df = gen.create_target(df, target_type='regression', 
                       future_bars=5)
```

### 3. Data Cleaning
```python
# Different cleaning methods
df = provider.clean_data(df, method='drop')      # Remove NaN rows
df = provider.clean_data(df, method='ffill')     # Forward fill
df = provider.clean_data(df, method='bfill')     # Backward fill
df = provider.clean_data(df, method='interpolate')  # Interpolate
```

### 4. Model Selection
```python
# Enable only fast models
manager.enable_model('logistic_regression', True)
manager.enable_model('xgboost', True)
manager.enable_model('random_forest', False)  # Slower
```

## Troubleshooting

### Issue: "No module named 'src'"
**Solution:** Make sure you're running from the project root or add to path:
```python
import sys
sys.path.insert(0, '/path/to/ml_framework')
```

### Issue: "No data loaded"
**Solution:** Load data first:
```python
provider = DataProvider()
df = provider.load_yahoo('BTC-USD', '2020-01-01', '2023-12-31')
```

### Issue: "Missing required columns"
**Solution:** Ensure data has OHLC columns:
```python
provider.validate_data(df)
```

### Issue: Models not saving
**Solution:** Check models directory exists and has write permissions:
```python
manager = ModelManager(models_dir='models')  # Will create if needed
```

## Next Steps

1. ✅ Run `examples/basic_workflow.py`
2. ✅ Experiment with different tickers and timeframes
3. ✅ Try different feature sets and target definitions
4. ✅ Compare multiple models
5. ✅ Optimize backtest parameters
6. ✅ Add your own custom features
7. ✅ Integrate MLflow for experiment tracking

## Resources

- **STRUCTURE.md** - Detailed architecture documentation
- **README.md** - Project overview
- **examples/basic_workflow.py** - Complete working example
- **src/** - Source code with detailed docstrings

---

Happy modeling! 🚀
