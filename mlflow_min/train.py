import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from data_provider import get_data

def create_features(df):
    """Create time-series features from the date index and a target variable."""
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week.astype(int)

    # Target variable: predict the next day's closing price
    df['target'] = df['Close'].shift(-1)

    # Drop the last row with NaN target
    df.dropna(inplace=True)
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']
    target = 'target'
    
    return df, features, target

def train_and_log_model(model, model_name, params, X_train, y_train, X_test, y_test):
    """Train a model and log parameters, metrics, and the model to MLflow."""
    with mlflow.start_run(run_name=model_name) as run:
        print(f"Training {model_name}...")
        # Log hyperparameters
        mlflow.log_params(params)

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        # Log metrics
        mlflow.log_metric("rmse", rmse)
        print(f"  RMSE for {model_name}: {rmse:.4f}")

        # Log the model
        mlflow.sklearn.log_model(model, model_name)
        print(f"  Model {model_name} logged.")

        return run.info.run_id

if __name__ == "__main__":
    # Set the MLflow tracking URI to connect to the local server
    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    # Set the experiment name
    experiment_name = "Trading Price Prediction"
    mlflow.set_experiment(experiment_name)
    print(f"MLflow experiment set to '{experiment_name}'")

    # 1. Load data
    ohlc_data = get_data(num_rows=500)

    # 2. Create features and target
    featured_data, features, target = create_features(ohlc_data)

    X = featured_data[features]
    y = featured_data[target]

    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # 4. Define models and their parameters
    # Model 1: Linear Regression
    lr = LinearRegression()
    train_and_log_model(lr, "LinearRegression", {}, X_train, y_train, X_test, y_test)

    # Model 2: Random Forest with specific parameters
    rf_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    }
    rf = RandomForestRegressor(**rf_params)
    train_and_log_model(rf, "RandomForest", rf_params, X_train, y_train, X_test, y_test)

    print("\nTraining complete. Run 'run_mlflow_server.bat' and navigate to http://127.0.0.1:8080 to see the results.")
