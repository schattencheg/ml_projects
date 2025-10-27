# MLflow Tracking Guide

This guide explains how to use MLflow for experiment tracking in the ml_predict_15 project.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Starting MLflow Server](#starting-mlflow-server)
4. [Usage](#usage)
5. [What Gets Tracked](#what-gets-tracked)
6. [Viewing Results](#viewing-results)
7. [Advanced Usage](#advanced-usage)
8. [Troubleshooting](#troubleshooting)

---

## Overview

MLflow is an open-source platform for managing the ML lifecycle, including experimentation, reproducibility, and deployment. This project integrates MLflow to automatically track:

- **Parameters**: Training configuration (target_bars, target_pct, use_smote, etc.)
- **Metrics**: Model performance (accuracy, F1, precision, recall, ROC AUC)
- **Models**: Trained models with versioning
- **Artifacts**: Training results CSV, configuration files, plots

### Benefits

✅ **Track all experiments** - Never lose training results  
✅ **Compare runs** - Easily compare different configurations  
✅ **Reproduce results** - All parameters and artifacts saved  
✅ **Model versioning** - Track model evolution over time  
✅ **Visualization** - Built-in charts and comparisons  
✅ **Team collaboration** - Share results with team members  

---

## Installation

### 1. Install MLflow

```bash
pip install mlflow>=2.0.0
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
mlflow --version
```

Expected output: `mlflow, version 2.x.x`

---

## Starting MLflow Server

### Option 1: Start Server Locally (Recommended)

```bash
# Start MLflow tracking server on localhost:5000
mlflow server --host 127.0.0.1 --port 5000
```

**Keep this terminal running** - The server needs to be active during training.

### Option 2: Use Default File Store

If you don't start a server, MLflow will use a local file store (`mlruns/` directory).

### Option 3: Custom Port

```bash
# Use a different port
mlflow server --host 127.0.0.1 --port 8080
```

Then update your training code:

```python
models, scaler, results, best_model = train(
    df_train,
    mlflow_tracking_uri="http://localhost:8080"
)
```

---

## Usage

### Basic Usage (Default)

MLflow tracking is **enabled by default** and connects to `http://localhost:5000`:

```python
from src.model_training import train

# MLflow tracking enabled automatically
models, scaler, results, best_model = train(df_train)
```

### Disable MLflow Tracking

```python
# Disable MLflow tracking
models, scaler, results, best_model = train(df_train, use_mlflow=False)
```

### Custom MLflow Server

```python
# Use custom MLflow server
models, scaler, results, best_model = train(
    df_train,
    mlflow_tracking_uri="http://localhost:8080"
)
```

### Complete Example

```python
from src.model_training import train
from src.data_preparation import prepare_data
import pandas as pd

# Load data
df_train = pd.read_csv('data/btc_2023.csv')

# Train with MLflow tracking
models, scaler, results, best_model = train(
    df_train,
    target_bars=45,
    target_pct=3.0,
    use_smote=True,
    use_gpu=False,
    n_jobs=-1,
    use_mlflow=True,  # Enable MLflow (default)
    mlflow_tracking_uri="http://localhost:5000"  # MLflow server URL
)
```

---

## What Gets Tracked

### Parameters (Configuration)

All training configuration is automatically logged:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `target_bars` | Bars to look ahead | 45 |
| `target_pct` | Percentage threshold | 3.0 |
| `use_smote` | SMOTE enabled | True |
| `use_gpu` | GPU acceleration | False |
| `n_jobs` | CPU cores used | 15 |
| `dataset_shape` | Dataset dimensions | 420612x5 |
| `train_size` | Training set size | 336489 |
| `val_size` | Validation set size | 84123 |
| `class_imbalance_ratio` | Class imbalance | 364.65 |
| `smote_applied` | SMOTE actually applied | True |
| `best_model_name` | Best performing model | logistic_regression |
| `num_models_trained` | Total models trained | 8 |

### Metrics (Performance)

#### Per-Model Metrics

For each trained model:

- `{model_name}_accuracy` - Classification accuracy
- `{model_name}_f1_score` - F1 score
- `{model_name}_precision` - Precision
- `{model_name}_recall` - Recall
- `{model_name}_roc_auc` - ROC AUC score
- `{model_name}_training_time` - Training time (seconds)

Example:
- `logistic_regression_accuracy`: 0.7234
- `logistic_regression_f1_score`: 0.6187
- `xgboost_accuracy`: 0.7087
- `xgboost_f1_score`: 0.5976

#### Best Model Metrics

- `best_accuracy` - Best model accuracy
- `best_f1_score` - Best model F1 score
- `best_precision` - Best model precision
- `best_recall` - Best model recall
- `best_roc_auc` - Best model ROC AUC
- `total_training_time` - Total time for all models
- `avg_training_time` - Average time per model

### Models

The best performing model is logged to MLflow with:

- **Model artifact**: Serialized model file
- **Model signature**: Input/output schema
- **Registered name**: `ml_predict_15_{model_name}`
- **Version**: Automatically incremented

### Artifacts

All training artifacts are saved:

| Artifact | Path | Description |
|----------|------|-------------|
| Training Results CSV | `results/training_results_summary.csv` | All model metrics |
| Training Config | `config/training_config.txt` | Training parameters |
| Comparison Plot | `plots/model_comparison_training.png` | Visual comparison |

---

## Viewing Results

### 1. Open MLflow UI

With the MLflow server running, open your browser:

```
http://localhost:5000
```

### 2. Navigate to Experiment

Find your experiment:
- **Experiment name**: `ml_predict_15/classification/crypto_price_prediction`
- **Run name**: `training_YYYYMMDD_HHMMSS`

### 3. View Run Details

Click on a run to see:

#### Parameters Tab
- All training configuration
- Dataset characteristics
- Model settings

#### Metrics Tab
- Per-model performance metrics
- Best model metrics
- Training times
- **Charts**: Automatic visualization of metrics

#### Artifacts Tab
- Best model (downloadable)
- Training results CSV
- Training configuration
- Comparison plots

### 4. Compare Runs

Select multiple runs and click **Compare**:
- Side-by-side parameter comparison
- Metric comparison charts
- Identify best configuration

---

## Training Output

When MLflow tracking is enabled, you'll see:

```
================================================================================
MLFLOW TRACKING ENABLED
================================================================================
Tracking URI: http://localhost:5000
Experiment: ml_predict_15/classification/crypto_price_prediction
Run: training_20251024_123045
Run ID: abc123def456...
================================================================================

[... training progress ...]

✓ Best model logged to MLflow: logistic_regression
✓ Artifacts logged to MLflow

================================================================================
MLFLOW TRACKING COMPLETE
================================================================================
View results at: http://localhost:5000
Run ID: abc123def456...
================================================================================
```

---

## Advanced Usage

### 1. Custom Experiment Names

Modify `src/model_training.py` to use custom experiment names:

```python
# In train() function, change:
experiment_name = "ml_predict_15/classification/my_custom_experiment"
```

### 2. Add Custom Tags

Add tags to runs for better organization:

```python
# After mlflow.start_run()
mlflow.set_tag("dataset", "btc_2023")
mlflow.set_tag("strategy", "conservative")
mlflow.set_tag("version", "v1.0")
```

### 3. Log Additional Metrics

Log custom metrics during training:

```python
# In training loop
mlflow.log_metric("custom_metric", value)
```

### 4. Load Models from MLflow

```python
import mlflow

# Load latest version of a model
model_uri = "models:/ml_predict_15_logistic_regression/latest"
loaded_model = mlflow.sklearn.load_model(model_uri)

# Load specific version
model_uri = "models:/ml_predict_15_logistic_regression/1"
loaded_model = mlflow.sklearn.load_model(model_uri)

# Load from run ID
run_id = "abc123def456..."
model_uri = f"runs:/{run_id}/best_model"
loaded_model = mlflow.sklearn.load_model(model_uri)
```

### 5. Search Runs Programmatically

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get experiment
experiment = client.get_experiment_by_name(
    "ml_predict_15/classification/crypto_price_prediction"
)

# Search runs
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.best_f1_score > 0.6",
    order_by=["metrics.best_f1_score DESC"],
    max_results=10
)

# Print top runs
for run in runs:
    print(f"Run ID: {run.info.run_id}")
    print(f"F1 Score: {run.data.metrics['best_f1_score']}")
    print(f"Best Model: {run.data.params['best_model_name']}")
    print()
```

### 6. Export Run Data

```python
import mlflow
import pandas as pd

# Get run data
run_id = "abc123def456..."
run = mlflow.get_run(run_id)

# Export parameters
params_df = pd.DataFrame([run.data.params])
params_df.to_csv("run_parameters.csv", index=False)

# Export metrics
metrics_df = pd.DataFrame([run.data.metrics])
metrics_df.to_csv("run_metrics.csv", index=False)
```

---

## Troubleshooting

### Issue: "Connection refused" or "Cannot connect to MLflow server"

**Solution 1**: Start the MLflow server

```bash
mlflow server --host 127.0.0.1 --port 5000
```

**Solution 2**: Disable MLflow tracking

```python
models, scaler, results, best_model = train(df_train, use_mlflow=False)
```

**Solution 3**: Use file store (no server needed)

```python
models, scaler, results, best_model = train(
    df_train,
    mlflow_tracking_uri="file:./mlruns"
)
```

### Issue: "MLflow not installed"

**Solution**: Install MLflow

```bash
pip install mlflow>=2.0.0
```

### Issue: "Port 5000 already in use"

**Solution**: Use a different port

```bash
# Start server on port 8080
mlflow server --host 127.0.0.1 --port 8080
```

```python
# Update training code
models, scaler, results, best_model = train(
    df_train,
    mlflow_tracking_uri="http://localhost:8080"
)
```

### Issue: "Failed to log model to MLflow"

**Possible causes**:
- Model type not supported by MLflow
- Model too large
- Network issues

**Solution**: Check error message and ensure model is compatible

### Issue: "Artifacts not showing in UI"

**Solution**: Check that files exist before logging:

```python
# In model_training.py
if os.path.exists(csv_path):
    mlflow.log_artifact(csv_path)
```

### Issue: "Slow training with MLflow"

MLflow adds minimal overhead (~1-2% of training time). If training is slow:

1. Check network connection to MLflow server
2. Use local file store instead of remote server
3. Disable MLflow for quick experiments

---

## Best Practices

### 1. Always Run MLflow Server

Start the server before training:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

### 2. Use Descriptive Run Names

Modify run names to include important info:

```python
run_name = f"training_smote_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
```

### 3. Tag Runs for Organization

```python
mlflow.set_tag("experiment_type", "baseline")
mlflow.set_tag("dataset_version", "v2")
```

### 4. Compare Runs Regularly

Use the MLflow UI to compare runs and identify best configurations.

### 5. Clean Up Old Runs

Periodically delete old/failed runs to keep the UI clean:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.delete_run(run_id)
```

### 6. Backup MLflow Data

The `mlruns/` directory contains all tracking data. Back it up regularly:

```bash
# Backup mlruns directory
tar -czf mlruns_backup_$(date +%Y%m%d).tar.gz mlruns/
```

### 7. Document Experiments

Add notes to runs in the MLflow UI to document findings and insights.

---

## Integration with Existing Workflow

MLflow tracking integrates seamlessly with your existing workflow:

### Training

```python
# Same as before, MLflow tracking happens automatically
models, scaler, results, best_model = train(df_train)
```

### Backtesting

```python
# Load model from MLflow
import mlflow

model = mlflow.sklearn.load_model("models:/ml_predict_15_xgboost/latest")
scaler = joblib.load('models/2024-01-15_14-30-45/scaler.joblib')

# Run backtest as usual
from src.MLBacktester import MLBacktester

backtester = MLBacktester(model, scaler)
results = backtester.run_backtest(df_test)
```

### Model Deployment

```python
# Load production model from MLflow
model_uri = "models:/ml_predict_15_logistic_regression/Production"
model = mlflow.sklearn.load_model(model_uri)

# Use for predictions
predictions = model.predict(X_new)
```

---

## Summary

### Quick Reference

**Start MLflow Server:**
```bash
mlflow server --host 127.0.0.1 --port 5000
```

**Train with MLflow (default):**
```python
models, scaler, results, best_model = train(df_train)
```

**View Results:**
```
http://localhost:5000
```

**Disable MLflow:**
```python
models, scaler, results, best_model = train(df_train, use_mlflow=False)
```

### What's Tracked

✅ All training parameters  
✅ All model metrics  
✅ Best model artifact  
✅ Training results CSV  
✅ Configuration file  
✅ Comparison plots  

### Benefits

🎯 **Never lose experiments**  
📊 **Easy comparison**  
🔄 **Full reproducibility**  
📈 **Built-in visualization**  
🤝 **Team collaboration**  
🚀 **Model versioning**  

---

## Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Models](https://mlflow.org/docs/latest/models.html)
- [Model Training Guide](../src/model_training.py)

Happy tracking! 📊🚀✅
