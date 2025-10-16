"""
MLflow Configuration for ML Projects

This file contains configuration settings and naming conventions
for MLflow tracking across different projects.
"""

# MLflow Server Configuration
MLFLOW_TRACKING_URI = "http://localhost:5000"

# Project Names
# Use these consistent project names across all your ML projects
PROJECT_NAMES = {
    "BACKTEST": "ml_backtest",
    "FORECAST": "ml_forecast",
    "CLASSIFICATION": "ml_classification",
    "CLUSTERING": "ml_clustering",
    "TIMESERIES": "ml_timeseries",
    "NLP": "ml_nlp",
    "COMPUTER_VISION": "ml_cv"
}

# Model Types
# Categorize your models by type
MODEL_TYPES = {
    "REGRESSION": "regression",
    "CLASSIFICATION": "classification",
    "CLUSTERING": "clustering",
    "TIMESERIES": "timeseries",
    "DEEP_LEARNING": "deep_learning",
    "ENSEMBLE": "ensemble",
    "STRATEGY": "strategy"  # For trading strategies
}

# Asset/Timeframe Naming Convention
# Format: {asset}_{timeframe}
# Examples:
#   - btc_usd_daily
#   - spy_hourly
#   - aapl_15min
#   - eurusd_1h

# Experiment Naming Convention
# Format: {project_name}/{model_type}/{asset_or_timeframe}
# Examples:
#   - ml_backtest/regression/btc_usd_daily
#   - ml_forecast/timeseries/spy_hourly
#   - ml_classification/classification/sentiment_news

# Run Naming Convention
# Format: {model_name}_v{version}_{timestamp}
# Examples:
#   - linear_regression_v1_20251016_120000
#   - random_forest_v2_20251016_120500
#   - lstm_v1_20251016_121000

# Standard Metrics to Track
STANDARD_METRICS = {
    "REGRESSION": [
        "rmse",
        "mae",
        "r2",
        "mape"
    ],
    "CLASSIFICATION": [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auc_roc"
    ],
    "BACKTEST": [
        "return_pct",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown_pct",
        "win_rate_pct",
        "profit_factor"
    ]
}

# Standard Parameters to Track
STANDARD_PARAMS = {
    "DATASET": [
        "train_samples",
        "test_samples",
        "n_features",
        "asset",
        "timeframe",
        "start_date",
        "end_date"
    ],
    "MODEL": [
        "model_type",
        "version",
        "random_state"
    ]
}


def get_experiment_name(project: str, model_type: str, asset_or_timeframe: str) -> str:
    """
    Generate a standardized experiment name.
    
    Args:
        project: Project name (use PROJECT_NAMES)
        model_type: Model type (use MODEL_TYPES)
        asset_or_timeframe: Asset or timeframe identifier
    
    Returns:
        Formatted experiment name
    
    Example:
        >>> get_experiment_name("ml_backtest", "regression", "btc_usd_daily")
        'ml_backtest/regression/btc_usd_daily'
    """
    return f"{project}/{model_type}/{asset_or_timeframe}"


def format_asset_name(asset: str, timeframe: str = "daily") -> str:
    """
    Format asset name for consistency.
    
    Args:
        asset: Asset ticker (e.g., 'BTC-USD', 'SPY')
        timeframe: Timeframe (e.g., 'daily', 'hourly', '15min')
    
    Returns:
        Formatted asset name
    
    Example:
        >>> format_asset_name("BTC-USD", "daily")
        'btc_usd_daily'
    """
    asset_clean = asset.replace("-", "_").replace("/", "_").lower()
    return f"{asset_clean}_{timeframe}"


# Example Usage
if __name__ == "__main__":
    print("MLflow Configuration Examples")
    print("="*60)
    
    # Example 1: Backtest project
    project = PROJECT_NAMES["BACKTEST"]
    model_type = MODEL_TYPES["REGRESSION"]
    asset = format_asset_name("BTC-USD", "daily")
    experiment = get_experiment_name(project, model_type, asset)
    
    print(f"\nExample 1: Backtest Regression")
    print(f"  Project: {project}")
    print(f"  Model Type: {model_type}")
    print(f"  Asset: {asset}")
    print(f"  Experiment: {experiment}")
    
    # Example 2: Forecast project
    project = PROJECT_NAMES["FORECAST"]
    model_type = MODEL_TYPES["TIMESERIES"]
    asset = format_asset_name("SPY", "hourly")
    experiment = get_experiment_name(project, model_type, asset)
    
    print(f"\nExample 2: Forecast Timeseries")
    print(f"  Project: {project}")
    print(f"  Model Type: {model_type}")
    print(f"  Asset: {asset}")
    print(f"  Experiment: {experiment}")
    
    # Example 3: Classification project
    project = PROJECT_NAMES["CLASSIFICATION"]
    model_type = MODEL_TYPES["CLASSIFICATION"]
    dataset = "sentiment_news"
    experiment = get_experiment_name(project, model_type, dataset)
    
    print(f"\nExample 3: Classification")
    print(f"  Project: {project}")
    print(f"  Model Type: {model_type}")
    print(f"  Dataset: {dataset}")
    print(f"  Experiment: {experiment}")
    
    print("\n" + "="*60)
