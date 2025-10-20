# MLflow Integration Guide

This guide explains how the Data Provider Server integrates with MLflow for experiment tracking and model management.

## Overview

The Data Provider Server now automatically starts an MLflow server alongside the data server, providing a unified environment for data retrieval and ML experiment tracking.

## Features

- **Automatic MLflow Server Startup**: Checks if MLflow is already running and starts it if needed
- **Smart Detection**: Detects existing MLflow instances to avoid conflicts
- **Graceful Shutdown**: Properly terminates MLflow server when data server stops
- **Configurable**: Easy to enable/disable via configuration file
- **Shared Environment**: Both servers run in the same environment for seamless integration

## Configuration

Edit `config.py` to customize MLflow settings:

```python
# MLflow configuration
MLFLOW_ENABLED = True  # Set to False to disable MLflow server startup
MLFLOW_HOST = '127.0.0.1'
MLFLOW_PORT = 5000
MLFLOW_BACKEND_STORE_URI = os.path.join(os.path.dirname(__file__), 'mlruns')
MLFLOW_DEFAULT_ARTIFACT_ROOT = os.path.join(os.path.dirname(__file__), 'mlartifacts')
```

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `MLFLOW_ENABLED` | Enable/disable MLflow server startup | `True` |
| `MLFLOW_HOST` | MLflow server host | `'127.0.0.1'` |
| `MLFLOW_PORT` | MLflow server port | `5000` |
| `MLFLOW_BACKEND_STORE_URI` | Path to store experiment metadata | `'./mlruns'` |
| `MLFLOW_DEFAULT_ARTIFACT_ROOT` | Path to store artifacts | `'./mlartifacts'` |

## Usage

### Starting the Servers

Simply start the data server as usual:

```bash
python server.py
```

Output:
```
============================================================
Data Provider Server with MLflow Integration
============================================================

✓ MLflow server is already running on http://127.0.0.1:5000
(or)
Starting MLflow server on http://127.0.0.1:5000...
✓ MLflow server started successfully on http://127.0.0.1:5000

============================================================
Data Provider Server: http://0.0.0.0:5001
MLflow Server: http://127.0.0.1:5000
============================================================
```

### Accessing the Servers

- **Data Provider Server**: http://localhost:5001
- **MLflow UI**: http://localhost:5000

### Stopping the Servers

Press `Ctrl+C` to stop both servers gracefully:

```
Shutting down servers...
✓ MLflow server stopped
✓ Data Provider server stopped
```

## Server Behavior

### Scenario 1: No MLflow Server Running

The data server will:
1. Check if port 5000 is available
2. Start a new MLflow server
3. Wait 3 seconds for initialization
4. Verify the server is responding
5. Display success message

### Scenario 2: MLflow Server Already Running

The data server will:
1. Detect the existing MLflow server
2. Display a message confirming it's already running
3. Continue without starting a new instance

### Scenario 3: Port in Use by Another Service

The data server will:
1. Detect that port 5000 is in use
2. Attempt to verify if it's an MLflow server
3. Display a warning if it's not MLflow
4. Continue without starting MLflow

### Scenario 4: MLflow Not Installed

The data server will:
1. Attempt to start MLflow
2. Catch the FileNotFoundError
3. Display installation instructions
4. Continue without MLflow

## Integration with ML Workflows

### Example: Training with Data Provider and MLflow

```python
import mlflow
from src.Data.DataProviderLocal import DataProviderLocal
from src.Background.enums import DataResolution, DataPeriod
from sklearn.linear_model import LinearRegression
import numpy as np

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Get data from Data Provider
provider = DataProviderLocal()
df = provider.get_data_socket(
    ticker='BTC-USD',
    resolution=DataResolution.DAY_01,
    period=DataPeriod.YEAR_01,
    local_only=True
)

# Prepare features
df['Returns'] = df['Close'].pct_change()
df['SMA_20'] = df['Close'].rolling(20).mean()
df = df.dropna()

X = df[['SMA_20']].values
y = df['Returns'].values

# Train model with MLflow tracking
with mlflow.start_run(run_name="btc_linear_regression"):
    # Log parameters
    mlflow.log_param("ticker", "BTC-USD")
    mlflow.log_param("resolution", "1d")
    mlflow.log_param("period", "1y")
    mlflow.log_param("feature", "SMA_20")
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Log metrics
    score = model.score(X, y)
    mlflow.log_metric("r2_score", score)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Model R² Score: {score:.4f}")
    print(f"Run logged to MLflow: http://localhost:5000")
```

### Example: Batch Training Multiple Tickers

```python
import mlflow
from src.Data.DataProviderLocal import DataProviderLocal
from src.Background.enums import DataResolution, DataPeriod
from sklearn.ensemble import RandomForestRegressor

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("crypto_batch_training")

provider = DataProviderLocal()

# Get data for multiple tickers
results = provider.get_batch_data_socket(
    tickers=['BTC-USD', 'ETH-USD', 'SOL-USD'],
    resolution=DataResolution.DAY_01,
    period=DataPeriod.YEAR_01,
    local_only=True
)

# Train model for each ticker
for ticker, df in results.items():
    with mlflow.start_run(run_name=f"{ticker}_rf_model"):
        # Prepare data
        df['Returns'] = df['Close'].pct_change()
        df['SMA_10'] = df['Close'].rolling(10).mean()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df = df.dropna()
        
        X = df[['SMA_10', 'SMA_20']].values
        y = df['Returns'].values
        
        # Log parameters
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Log metrics
        score = model.score(X, y)
        mlflow.log_metric("r2_score", score)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"{ticker} - R² Score: {score:.4f}")
```

## Troubleshooting

### MLflow Server Won't Start

**Problem**: MLflow server fails to start

**Solutions**:
1. Check if MLflow is installed: `pip install mlflow`
2. Verify port 5000 is not in use by another application
3. Check file permissions for `mlruns` and `mlartifacts` directories
4. Review logs in `data_server.log`

### Port Already in Use

**Problem**: Port 5000 is already in use

**Solutions**:
1. Check if MLflow is already running: `http://localhost:5000`
2. Change `MLFLOW_PORT` in `config.py` to a different port
3. Stop the conflicting service
4. Use `netstat -ano | findstr :5000` (Windows) to identify the process

### MLflow UI Not Accessible

**Problem**: Cannot access MLflow UI at http://localhost:5000

**Solutions**:
1. Wait a few seconds after startup (MLflow may take time to initialize)
2. Check if the server is running: look for success message in console
3. Try accessing from `http://127.0.0.1:5000` instead
4. Check firewall settings
5. Review server logs for errors

### Data Server Starts But MLflow Doesn't

**Problem**: Data server runs but MLflow server doesn't start

**Solutions**:
1. Check if `MLFLOW_ENABLED = True` in `config.py`
2. Verify MLflow is installed: `pip list | grep mlflow`
3. Check console output for error messages
4. Try starting MLflow manually: `mlflow server --host 127.0.0.1 --port 5000`

## Disabling MLflow

To run the data server without MLflow:

1. Edit `config.py`:
   ```python
   MLFLOW_ENABLED = False
   ```

2. Restart the server:
   ```bash
   python server.py
   ```

## Manual MLflow Server Management

If you prefer to manage MLflow separately:

### Start MLflow Manually

```bash
mlflow server \
    --host 127.0.0.1 \
    --port 5000 \
    --backend-store-uri ./mlruns \
    --default-artifact-root ./mlartifacts
```

### Set MLflow Tracking URI in Code

```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
```

## Directory Structure

After running the servers, you'll have:

```
data_server/
├── server.py
├── config.py
├── data/                    # OHLC data cache
│   ├── day_01/
│   ├── hour_01/
│   └── ...
├── mlruns/                  # MLflow experiment metadata
│   ├── 0/
│   ├── 1/
│   └── ...
├── mlartifacts/            # MLflow artifacts (models, plots, etc.)
│   └── ...
└── data_server.log         # Server logs
```

## Best Practices

1. **Use Experiments**: Organize runs into experiments for better tracking
   ```python
   mlflow.set_experiment("my_experiment_name")
   ```

2. **Log Everything**: Parameters, metrics, artifacts, and models
   ```python
   mlflow.log_param("param_name", value)
   mlflow.log_metric("metric_name", value)
   mlflow.log_artifact("file_path")
   ```

3. **Version Control**: Keep `mlruns` in `.gitignore` but track experiment code

4. **Backup**: Regularly backup `mlruns` and `mlartifacts` directories

5. **Clean Up**: Periodically delete old experiments to save space

## Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking Guide](https://mlflow.org/docs/latest/tracking.html)
- [Data Provider API Guide](API_GUIDE.md)
- [WebSocket Guide](WEBSOCKET_GUIDE.md)
- [MLflow Tracker Usage](MLFLOW_GUIDE.md) (if exists in ml_backtest project)

## Support

For issues related to:
- **Data Server**: Check `data_server.log` and server console output
- **MLflow Server**: Check MLflow logs and UI at http://localhost:5000
- **Integration**: Review this guide and example code above
