# MLflow Quick Start - 5 Minutes

## Step 1: Start MLflow Server (1 minute)

```bash
mlflow server --host 0.0.0.0 --port 5000
```

Open browser: `http://localhost:5000`

---

## Step 2: Run Example (2 minutes)

```bash
python ml_model_example.py
```

This will:
- Train 4 different ML models on BTC price data
- Log all parameters and metrics to MLflow
- Save models to MLflow registry

---

## Step 3: View Results (2 minutes)

1. Go to `http://localhost:5000`
2. Click on experiment: `ml_backtest/regression/btc_usd_daily`
3. See all 4 model runs
4. Compare metrics (RMSE, R²)
5. Click on a run to see details

---

## 🎯 Naming Convention

```
Experiment: ml_backtest/regression/btc_usd_daily
            └─project  └─model_type └─asset_timeframe

Run:        linear_regression_v1_20251016_120000
            └─model_name      └─v └─timestamp
```

---

## 💻 Basic Code Pattern

```python
from mlflow_tracker import setup_mlflow_tracker

# 1. Initialize
tracker = setup_mlflow_tracker()

# 2. Start run
with tracker.start_run(
    model_type="regression",
    asset_or_timeframe="btc_usd_daily",
    model_name="linear_regression",
    version="1"
):
    # 3. Train model
    model.fit(X_train, y_train)
    
    # 4. Log metrics
    tracker.log_model_metrics({
        "rmse": 0.05,
        "r2": 0.85
    })
    
    # 5. Log model
    tracker.log_sklearn_model(model, "model")
```

---

## 📊 What Gets Tracked

### Automatically Logged:
- ✅ Experiment name
- ✅ Run name with timestamp
- ✅ Project, model type, asset tags
- ✅ Run ID

### You Log:
- Parameters (hyperparameters, settings)
- Metrics (RMSE, R², accuracy, etc.)
- Models (sklearn, keras, etc.)
- Artifacts (plots, data files)

---

## 🔍 Finding Best Model

```python
from mlflow_tracker import MLflowTracker

best = MLflowTracker.get_best_run(
    experiment_name="ml_backtest/regression/btc_usd_daily",
    metric="r2",
    ascending=False  # Higher is better
)

print(f"Best R²: {best['metrics.r2']}")
```

---

## 📁 Project Organization

```
Your MLflow UI will show:

ml_backtest/
├── regression/
│   ├── btc_usd_daily/
│   │   ├── linear_regression_v1_20251016_120000
│   │   ├── ridge_regression_v1_20251016_120100
│   │   └── random_forest_v1_20251016_120200
│   └── spy_hourly/
│       └── ...
└── strategy/
    └── btc_usd_daily/
        └── sma_crossover_v1_20251016_120300
```

---

## 🚀 Next Steps

1. **Read Full Guide**: [MLFLOW_GUIDE.md](MLFLOW_GUIDE.md)
2. **Try Different Models**: Modify `ml_model_example.py`
3. **Track Backtests**: Add MLflow to your strategies
4. **Compare Results**: Use MLflow UI to compare runs

---

## 💡 Pro Tips

### Tip 1: Version Your Models
```python
version="1"  # Initial
version="2"  # Improved features
version="3"  # Optimized hyperparameters
```

### Tip 2: Use Meaningful Descriptions
```python
description="Linear regression with 20 features, 2-year BTC data"
```

### Tip 3: Tag Everything
```python
tags={
    "environment": "production",
    "data_source": "yfinance",
    "purpose": "baseline"
}
```

### Tip 4: Log Artifacts
```python
tracker.log_artifact_from_file("plot.png")
tracker.log_dict_as_json(feature_importance, "features.json")
```

---

## 🎓 Example Projects

### Project 1: Price Prediction
```
Project: ml_backtest
Experiment: ml_backtest/regression/btc_usd_daily
Models: LinearRegression, Ridge, RandomForest, LSTM
```

### Project 2: Trend Classification
```
Project: ml_classification
Experiment: ml_classification/classification/market_trend
Models: LogisticRegression, RandomForest, XGBoost
```

### Project 3: Strategy Backtesting
```
Project: ml_backtest
Experiment: ml_backtest/strategy/btc_usd_daily
Strategies: SMA_Crossover, RSI, MACD
```

---

## 🔧 Configuration

Edit `mlflow_config.py` to customize:
- Project names
- Model types
- Standard metrics
- Naming conventions

---

## 📚 Files

| File | Purpose |
|------|---------|
| `mlflow_tracker.py` | Main tracking utility |
| `mlflow_config.py` | Configuration |
| `ml_model_example.py` | Complete example |
| `MLFLOW_GUIDE.md` | Full documentation |
| `MLFLOW_QUICK_START.md` | This file |

---

**You're ready to track!** 🎉

Run `python ml_model_example.py` and see your first tracked experiments!
