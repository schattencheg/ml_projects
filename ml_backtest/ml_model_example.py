"""
ML Model Training Example with MLflow Tracking

This example demonstrates how to train ML models for price prediction
and track everything with MLflow using structured naming conventions.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add src directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'Data'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'Background'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'MLflow'))

from DataProvider import DataProvider
from enums import DataPeriod, DataResolution
from mlflow_tracker import setup_mlflow_tracker


def create_features(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """
    Create features for ML model.
    
    Args:
        df: DataFrame with OHLCV data
        lookback: Number of periods to look back
    
    Returns:
        DataFrame with features
    """
    data = df.copy()
    
    # Price-based features
    data['returns'] = data['close'].pct_change()
    data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
    
    # Moving averages
    for period in [5, 10, 20]:
        data[f'sma_{period}'] = data['close'].rolling(window=period).mean()
        data[f'ema_{period}'] = data['close'].ewm(span=period).mean()
    
    # Volatility
    data['volatility_5'] = data['returns'].rolling(window=5).std()
    data['volatility_20'] = data['returns'].rolling(window=20).std()
    
    # Price momentum
    data['momentum_5'] = data['close'] - data['close'].shift(5)
    data['momentum_10'] = data['close'] - data['close'].shift(10)
    
    # Volume features
    data['volume_sma_5'] = data['volume'].rolling(window=5).mean()
    data['volume_ratio'] = data['volume'] / data['volume_sma_5']
    
    # Target: Next day's return
    data['target'] = data['close'].shift(-1) / data['close'] - 1
    
    # Drop NaN values
    data = data.dropna()
    
    return data


def train_model_with_mlflow(model, model_name: str, X_train, X_test, y_train, y_test,
                            tracker, asset: str, model_params: dict = None):
    """
    Train a model and log everything to MLflow.
    
    Args:
        model: Sklearn model instance
        model_name: Name of the model
        X_train, X_test, y_train, y_test: Train/test data
        tracker: MLflowTracker instance
        asset: Asset name
        model_params: Model parameters to log
    """
    
    # Start MLflow run with structured naming
    with tracker.start_run(
        model_type="regression",
        asset_or_timeframe=f"{asset}_daily",
        model_name=model_name,
        version="1",
        description=f"Price prediction using {model_name}",
        tags={"asset": asset, "task": "price_prediction"}
    ):
        # Train model
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_metrics = {
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "train_mae": mean_absolute_error(y_train, y_pred_train),
            "train_r2": r2_score(y_train, y_pred_train)
        }
        
        test_metrics = {
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
            "test_r2": r2_score(y_test, y_pred_test)
        }
        
        # Log parameters
        if model_params:
            tracker.log_model_params(model_params)
        
        # Log dataset info
        tracker.log_model_params({
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "n_features": X_train.shape[1],
            "asset": asset
        })
        
        # Log metrics
        tracker.log_model_metrics(train_metrics)
        tracker.log_model_metrics(test_metrics)
        
        # Log model
        tracker.log_sklearn_model(model, model_name)
        
        # Log feature importance if available
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X_train.columns, model.feature_importances_))
            tracker.log_dict_as_json(feature_importance, "feature_importance.json")
        
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        print(f"Train RMSE: {train_metrics['train_rmse']:.6f}")
        print(f"Test RMSE:  {test_metrics['test_rmse']:.6f}")
        print(f"Train R²:   {train_metrics['train_r2']:.4f}")
        print(f"Test R²:    {test_metrics['test_r2']:.4f}")
        print(f"{'='*60}\n")
        
        return model, test_metrics


def main():
    """Main function to train multiple models and track with MLflow."""
    
    # Initialize MLflow tracker
    print("="*60)
    print("ML Model Training with MLflow Tracking")
    print("="*60)
    
    tracker = setup_mlflow_tracker(
        tracking_uri="http://localhost:5000",
        project_name="ml_backtest"
    )
    
    # Load data
    print("\nLoading data...")
    asset = "BTC-USD"
    dp = DataProvider(
        tickers=[asset],
        resolution=DataResolution.DAY_01,
        period=DataPeriod.YEAR_02
    )
    
    df = dp.data_request_by_ticker(asset)
    print(f"Loaded {len(df)} days of {asset} data")
    
    # Create features
    print("\nCreating features...")
    data = create_features(df)
    print(f"Created {len(data.columns)} features")
    
    # Prepare train/test split
    feature_cols = [col for col in data.columns if col not in ['target', 'date']]
    X = data[feature_cols]
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # Time series - don't shuffle
    )
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Define models to train
    models = {
        "linear_regression": (
            LinearRegression(),
            {}
        ),
        "ridge_regression": (
            Ridge(alpha=1.0),
            {"alpha": 1.0}
        ),
        "lasso_regression": (
            Lasso(alpha=0.1),
            {"alpha": 0.1}
        ),
        "random_forest": (
            RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            {"n_estimators": 100, "max_depth": 10, "random_state": 42}
        )
    }
    
    # Train all models
    print("\n" + "="*60)
    print("Training Models")
    print("="*60)
    
    results = {}
    for model_name, (model, params) in models.items():
        print(f"\nTraining {model_name}...")
        trained_model, metrics = train_model_with_mlflow(
            model=model,
            model_name=model_name,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            tracker=tracker,
            asset=asset.replace("-", "_").lower(),
            model_params=params
        )
        results[model_name] = metrics
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Asset: {asset}")
    print(f"Models trained: {len(models)}")
    print(f"\nTest RMSE Comparison:")
    for model_name, metrics in sorted(results.items(), key=lambda x: x[1]['test_rmse']):
        print(f"  {model_name:20s}: {metrics['test_rmse']:.6f}")
    
    print(f"\nTest R² Comparison:")
    for model_name, metrics in sorted(results.items(), key=lambda x: x[1]['test_r2'], reverse=True):
        print(f"  {model_name:20s}: {metrics['test_r2']:.4f}")
    
    print(f"\n✓ All results logged to MLflow")
    print(f"  View at: http://localhost:5000")
    print(f"  Experiment: ml_backtest/regression/{asset.replace('-', '_').lower()}_daily")
    print("="*60)


if __name__ == '__main__':
    main()
