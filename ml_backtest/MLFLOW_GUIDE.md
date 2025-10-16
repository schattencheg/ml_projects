# MLflow Tracking Guide

## 📊 Overview

This project uses **MLflow** to track all ML experiments with a structured naming convention that keeps everything organized across multiple projects.

## 🎯 Naming Convention

### Structure
```
Project: ml_backtest, ml_forecast, ml_classification, etc.
    ↓
Experiment: {project}/{model_type}/{asset_or_timeframe}
    ↓
Run: {model_name}_v{version}_{timestamp}
```

### Examples

#### Example 1: Backtest Regression
```
Project:    ml_backtest
Experiment: ml_backtest/regression/btc_usd_daily
Run:        linear_regression_v1_20251016_120000
```

#### Example 2: Price Forecast
```
Project:    ml_forecast
Experiment: ml_forecast/timeseries/spy_hourly
Run:        lstm_v2_20251016_120500
```

#### Example 3: Classification
```
Project:    ml_classification
Experiment: ml_classification/classification/sentiment_news
Run:        random_forest_v1_20251016_121000
```

---

## 🚀 Quick Start

### 1. Start MLflow Server

```bash
# Start MLflow server on port 5000
mlflow server --host 0.0.0.0 --port 5000
```

Access UI at: `http://localhost:5000`

### 2. Basic Usage

```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'MLflow'))

from mlflow_tracker import setup_mlflow_tracker

# Initialize tracker
tracker = setup_mlflow_tracker(
    tracking_uri="http://localhost:5000",
    project_name="ml_backtest"
)

# Start a run
with tracker.start_run(
    model_type="regression",
    asset_or_timeframe="btc_usd_daily",
    model_name="linear_regression",
    version="1",
    description="Price prediction model"
):
    # Log parameters
    tracker.log_model_params({
        "learning_rate": 0.01,
        "epochs": 100
    })
    
    # Log metrics
    tracker.log_model_metrics({
        "rmse": 0.05,
        "r2": 0.85
    })
    
    # Log model
    tracker.log_sklearn_model(model, "model")
```

---

## 📁 Project Structure

### Project Names (Consistent Across All Projects)

Use these standardized project names:

| Project Name | Purpose |
|-------------|---------|
| `ml_backtest` | Trading strategy backtesting |
| `ml_forecast` | Time series forecasting |
| `ml_classification` | Classification tasks |
| `ml_clustering` | Clustering analysis |
| `ml_timeseries` | Time series analysis |
| `ml_nlp` | Natural language processing |
| `ml_cv` | Computer vision |

### Model Types

| Model Type | Use For |
|-----------|---------|
| `regression` | Regression models |
| `classification` | Classification models |
| `clustering` | Clustering models |
| `timeseries` | Time series models |
| `deep_learning` | Neural networks |
| `ensemble` | Ensemble methods |
| `strategy` | Trading strategies |

### Asset/Timeframe Naming

Format: `{asset}_{timeframe}`

Examples:
- `btc_usd_daily`
- `spy_hourly`
- `aapl_15min`
- `eurusd_1h`

---

## 📊 What to Track

### For Regression Models

**Parameters:**
- Model hyperparameters
- Train/test split ratio
- Feature engineering settings
- Data preprocessing steps

**Metrics:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score
- MAPE (Mean Absolute Percentage Error)

### For Trading Strategies

**Parameters:**
- Strategy parameters (MA periods, thresholds, etc.)
- Starting capital
- Commission rates
- Asset and timeframe

**Metrics:**
- Return %
- Sharpe Ratio
- Sortino Ratio
- Max Drawdown %
- Win Rate %
- Profit Factor
- Number of Trades

### For Classification Models

**Parameters:**
- Model hyperparameters
- Class weights
- Feature selection method

**Metrics:**
- Accuracy
- Precision
- Recall
- F1 Score
- AUC-ROC

---

## 🔧 MLflowTracker API

### Initialize Tracker

```python
from mlflow_tracker import setup_mlflow_tracker

tracker = setup_mlflow_tracker(
    tracking_uri="http://localhost:5000",
    project_name="ml_backtest"
)
```

### Start a Run

```python
with tracker.start_run(
    model_type="regression",
    asset_or_timeframe="btc_usd_daily",
    model_name="linear_regression",
    version="1",
    description="Optional description",
    tags={"custom_tag": "value"}
):
    # Your training code here
    pass
```

### Log Parameters

```python
tracker.log_model_params({
    "learning_rate": 0.01,
    "n_estimators": 100,
    "max_depth": 10
})
```

### Log Metrics

```python
tracker.log_model_metrics({
    "rmse": 0.05,
    "mae": 0.03,
    "r2": 0.85
})
```

### Log Backtest Results

```python
# Automatically logs all backtest metrics
tracker.log_backtest_results(backtest_stats)
```

### Log Strategy Parameters

```python
# Automatically extracts and logs strategy parameters
tracker.log_strategy_params(strategy_instance)
```

### Log Model

```python
# Log scikit-learn model
tracker.log_sklearn_model(model, "model_name")
```

### Log Artifacts

```python
# Log a file
tracker.log_artifact_from_file("plot.png")

# Log a dictionary as JSON
tracker.log_dict_as_json({"key": "value"}, "data.json")
```

### Find Best Run

```python
best_run = MLflowTracker.get_best_run(
    experiment_name="ml_backtest/regression/btc_usd_daily",
    metric="r2",
    ascending=False  # Higher is better
)
```

---

## 💡 Best Practices

### 1. Consistent Naming

✅ **DO:**
```python
asset = "btc_usd_daily"
model_type = "regression"
model_name = "linear_regression"
```

❌ **DON'T:**
```python
asset = "BTC-USD-Daily"  # Inconsistent
model_type = "reg"  # Abbreviated
model_name = "LinReg"  # Unclear
```

### 2. Version Control

- Increment version for significant changes
- Use v1, v2, v3, etc.
- Document what changed in description

```python
# First version
version="1", description="Initial model"

# Improved version
version="2", description="Added feature engineering"

# Production version
version="3", description="Optimized hyperparameters"
```

### 3. Meaningful Descriptions

```python
# Good
description="Linear regression with 20 features, 2-year training data"

# Better
description="Linear regression: 20 features (MA, volatility, momentum), 2yr BTC data, 80/20 split"
```

### 4. Tag Everything

```python
tags={
    "environment": "production",
    "data_source": "yfinance",
    "feature_set": "v2",
    "purpose": "baseline_model"
}
```

### 5. Log Artifacts

- Save plots
- Save feature importance
- Save model predictions
- Save configuration files

---

## 📈 Example Workflows

### Workflow 1: Train Multiple Models

```python
tracker = setup_mlflow_tracker()

models = {
    "linear_regression": LinearRegression(),
    "ridge": Ridge(alpha=1.0),
    "random_forest": RandomForestRegressor()
}

for model_name, model in models.items():
    with tracker.start_run(
        model_type="regression",
        asset_or_timeframe="btc_usd_daily",
        model_name=model_name,
        version="1"
    ):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        tracker.log_model_metrics({
            "rmse": rmse(y_test, predictions),
            "r2": r2_score(y_test, predictions)
        })
        
        tracker.log_sklearn_model(model, model_name)
```

### Workflow 2: Hyperparameter Tuning

```python
tracker = setup_mlflow_tracker()

for alpha in [0.1, 1.0, 10.0]:
    with tracker.start_run(
        model_type="regression",
        asset_or_timeframe="btc_usd_daily",
        model_name="ridge",
        version="1",
        description=f"Ridge with alpha={alpha}"
    ):
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        
        tracker.log_model_params({"alpha": alpha})
        tracker.log_model_metrics({
            "rmse": calculate_rmse(model, X_test, y_test)
        })
```

### Workflow 3: Track Backtesting

```python
tracker = setup_mlflow_tracker()

with tracker.start_run(
    model_type="strategy",
    asset_or_timeframe="btc_usd_daily",
    model_name="sma_crossover",
    version="1"
):
    # Run backtest
    bt = Backtest(df, Strategy, cash=10000)
    stats = bt.run()
    
    # Log everything automatically
    tracker.log_backtest_results(stats)
    tracker.log_strategy_params(stats._strategy)
    
    # Log plot
    plot_file = "Output/backtest_plot.html"
    bt.plot(filename=plot_file)
    tracker.log_artifact_from_file(plot_file)
```

---

## 🔍 Viewing Results

### MLflow UI

1. Start server: `mlflow server --host 0.0.0.0 --port 5000`
2. Open browser: `http://localhost:5000`
3. Navigate through:
   - **Experiments**: See all experiments
   - **Runs**: View individual runs
   - **Compare**: Compare multiple runs
   - **Models**: View registered models

### Programmatic Access

```python
import mlflow

# Search runs
runs = mlflow.search_runs(
    experiment_names=["ml_backtest/regression/btc_usd_daily"],
    filter_string="metrics.r2 > 0.8",
    order_by=["metrics.rmse ASC"]
)

# Get best run
best_run = runs.iloc[0]
print(f"Best R²: {best_run['metrics.r2']}")
print(f"Best RMSE: {best_run['metrics.rmse']}")
```

---

## 📚 Files Reference

| File | Purpose |
|------|---------|
| `src/MLflow/mlflow_tracker.py` | Main tracking utility |
| `mlflow_config.py` | Configuration and naming conventions |
| `ml_model_example.py` | Example ML model with tracking |
| `MLFLOW_GUIDE.md` | This guide |

---

## 🎯 Quick Reference

### Common Commands

```bash
# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000

# Start with specific backend
mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000

# View UI
http://localhost:5000
```

### Common Patterns

```python
# Initialize
tracker = setup_mlflow_tracker()

# Start run
with tracker.start_run(...):
    # Log params
    tracker.log_model_params({...})
    
    # Log metrics
    tracker.log_model_metrics({...})
    
    # Log model
    tracker.log_sklearn_model(model, "model")
```

---

## 🚨 Troubleshooting

### MLflow server not starting
```bash
# Check if port is in use
netstat -ano | findstr :5000

# Use different port
mlflow server --port 5001
```

### Can't connect to server
```python
# Check tracking URI
tracker = setup_mlflow_tracker(tracking_uri="http://localhost:5000")

# Verify server is running
# Open http://localhost:5000 in browser
```

### Runs not appearing
- Check experiment name is correct
- Verify tracking URI
- Ensure run completed successfully
- Check MLflow UI for errors

---

## 📖 Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Models](https://mlflow.org/docs/latest/models.html)

---

**Happy Tracking!** 📊🚀
