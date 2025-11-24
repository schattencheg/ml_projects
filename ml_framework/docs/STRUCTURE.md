# ML Framework - Project Structure

## Overview

This is a simple and functional ML framework for financial data analysis and backtesting. The framework follows a modular design with clear separation of concerns.

## Directory Structure

```
ml_framework/
│
├── src/                          # Core framework modules
│   ├── __init__.py              # Package initialization
│   ├── data_provider.py         # Data loading and preprocessing
│   ├── features_generator.py   # Feature engineering
│   ├── model_manager.py         # Model configuration and management
│   ├── ml_trainer.py            # Model training
│   ├── ml_tester.py             # Model testing and evaluation
│   └── backtester.py            # Strategy backtesting
│
├── examples/                     # Example scripts
│   └── basic_workflow.py        # Complete workflow example
│
├── data/                         # Data directory (gitignored)
│   └── *.csv                    # CSV data files
│
├── models/                       # Saved models (gitignored)
│   └── YYYY-MM-DD_HH-MM-SS/    # Timestamped model directories
│       ├── *.joblib             # Model files
│       ├── scaler.joblib        # Scaler
│       └── metadata.joblib      # Metadata
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── STRUCTURE.md                  # This file
└── .gitignore                   # Git ignore rules
```

## Core Modules

### 1. DataProvider (`src/data_provider.py`)

**Purpose:** Handles data loading, validation, and basic preprocessing.

**Key Methods:**
- `load_csv()` - Load data from CSV file
- `load_yahoo()` - Download data from Yahoo Finance
- `validate_data()` - Validate OHLCV data format
- `clean_data()` - Handle missing values
- `split_data()` - Split into train/val/test sets
- `save_data()` - Save data to CSV

**Example:**
```python
data_provider = DataProvider()
df = data_provider.load_yahoo('BTC-USD', '2020-01-01', '2023-12-31')
df = data_provider.clean_data(df)
train_df, val_df, test_df = data_provider.split_data(df)
```

---

### 2. FeaturesGenerator (`src/features_generator.py`)

**Purpose:** Generates technical indicators and features from raw OHLC data.

**Key Methods:**
- `generate_features()` - Generate technical indicators
- `create_target()` - Create target variable for ML
- `get_feature_names()` - Get list of generated features
- `select_features()` - Select only feature columns

**Features Supported:**
- Moving Averages (SMA, EMA)
- Momentum Indicators (RSI, MACD)
- Volatility Indicators (Bollinger Bands, ATR)
- Volume Indicators
- Price Returns

**Example:**
```python
features_gen = FeaturesGenerator()
df_features = features_gen.generate_features(df, feature_set='basic')
df_features = features_gen.create_target(df_features, future_bars=5, threshold=0.02)
```

---

### 3. ModelManager (`src/model_manager.py`)

**Purpose:** Manages model configurations, saving, and loading.

**Key Methods:**
- `get_models()` - Get enabled model configurations
- `enable_model()` - Enable/disable specific models
- `save_models()` - Save models to timestamped directory
- `load_models()` - Load models from directory
- `list_saved_models()` - List all saved model versions
- `print_config()` - Print current configuration

**Features:**
- Timestamped model versioning
- Model enable/disable configuration
- Metadata tracking
- Automatic directory management

**Example:**
```python
model_manager = ModelManager()
model_manager.enable_model('xgboost', True)
model_configs = model_manager.get_models()

# Save models
save_dir = model_manager.save_models(models, scaler, metadata)

# Load models
models, scaler, metadata = model_manager.load_models('latest')
```

---

### 4. ML_Trainer (`src/ml_trainer.py`)

**Purpose:** Trains machine learning models on prepared data.

**Key Methods:**
- `train()` - Train multiple models
- `get_trained_models()` - Get trained models
- `get_results()` - Get training results
- `print_results()` - Print results summary

**Features:**
- Multiple model training
- Progress tracking with time measurement
- Automatic train/test split
- Feature scaling
- Performance metrics

**Example:**
```python
trainer = ML_Trainer()
results = trainer.train(
    df=train_df,
    target_col='target',
    feature_cols=feature_cols,
    model_configs=model_configs
)
```

---

### 5. ML_Tester (`src/ml_tester.py`)

**Purpose:** Evaluates trained models and generates performance metrics.

**Key Methods:**
- `evaluate()` - Evaluate models on test data
- `get_results()` - Get test results
- `get_predictions()` - Get predictions for specific model
- `compare_models()` - Compare models by metric
- `print_classification_report()` - Detailed classification report

**Metrics:**
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

**Example:**
```python
tester = ML_Tester()
test_results = tester.evaluate(
    df=test_df,
    models=trained_models,
    scaler=scaler,
    feature_cols=feature_cols
)
```

---

### 6. Backtester (`src/backtester.py`)

**Purpose:** Backtests trading strategies using ML predictions.

**Key Methods:**
- `run()` - Run backtest
- `plot_results()` - Visualize backtest results
- `get_results()` - Get backtest results

**Features:**
- Position sizing
- Commission modeling
- Performance metrics (Sharpe, drawdown, win rate)
- Equity curve visualization
- Buy & hold comparison

**Metrics:**
- Total return
- Annualized return
- Sharpe ratio
- Maximum drawdown
- Win rate
- Number of trades

**Example:**
```python
backtester = Backtester(initial_capital=10000, commission=0.001)
results = backtester.run(
    df=test_df,
    model=best_model,
    scaler=scaler,
    feature_cols=feature_cols
)
backtester.plot_results()
```

---

## Workflow

The typical workflow follows these steps:

```
1. Load Data (DataProvider)
   ↓
2. Generate Features (FeaturesGenerator)
   ↓
3. Create Target (FeaturesGenerator)
   ↓
4. Split Data (DataProvider)
   ↓
5. Setup Models (ModelManager)
   ↓
6. Train Models (ML_Trainer)
   ↓
7. Test Models (ML_Tester)
   ↓
8. Save Models (ModelManager)
   ↓
9. Backtest Strategy (Backtester)
```

See `examples/basic_workflow.py` for a complete working example.

---

## Design Principles

### 1. **Modularity**
Each class has a single, well-defined responsibility:
- DataProvider → Data management
- FeaturesGenerator → Feature engineering
- ModelManager → Model lifecycle
- ML_Trainer → Training
- ML_Tester → Evaluation
- Backtester → Strategy testing

### 2. **Simplicity**
- Clear, intuitive API
- Minimal configuration required
- Sensible defaults
- Easy to understand and modify

### 3. **Flexibility**
- Support for multiple data sources
- Configurable feature sets
- Multiple model support
- Customizable backtesting

### 4. **Reproducibility**
- Timestamped model versioning
- Metadata tracking
- Deterministic results (random_state)
- Complete workflow documentation

---

## Extension Points

The framework is designed to be easily extended:

### Add New Data Sources
Extend `DataProvider` with new `load_*()` methods

### Add New Features
Extend `FeaturesGenerator` with new indicator methods

### Add New Models
Update `ModelManager.model_config` and `ML_Trainer._create_model()`

### Add New Metrics
Extend `ML_Tester` with additional metric calculations

### Add New Backtest Strategies
Extend `Backtester` with custom trading logic

---

## Dependencies

Core dependencies (see `requirements.txt`):
- pandas - Data manipulation
- numpy - Numerical computing
- scikit-learn - ML models and metrics
- matplotlib - Visualization
- yfinance - Data download
- joblib - Model persistence

Optional:
- xgboost - Gradient boosting
- lightgbm - Gradient boosting
- tensorflow - Deep learning
- mlflow - Experiment tracking

---

## Best Practices

1. **Always validate data** before feature generation
2. **Clean missing values** appropriately for your use case
3. **Use train/val/test split** to avoid overfitting
4. **Scale features** for most ML models
5. **Save models with metadata** for reproducibility
6. **Test on unseen data** before backtesting
7. **Account for commissions** in backtests
8. **Compare to buy & hold** as baseline

---

## Next Steps

1. Run `examples/basic_workflow.py` to see the framework in action
2. Customize model configurations in `ModelManager`
3. Add your own features in `FeaturesGenerator`
4. Experiment with different target definitions
5. Optimize backtest parameters
6. Add MLflow tracking for experiment management

---

## Support

For issues or questions:
1. Check the example scripts
2. Review the docstrings in each module
3. Refer to this structure document
4. Consult the README.md

---

**Version:** 0.1.0  
**Last Updated:** 2025-11-13
