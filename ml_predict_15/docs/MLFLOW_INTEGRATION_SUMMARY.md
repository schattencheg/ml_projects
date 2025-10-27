# MLflow Integration Summary

Comprehensive MLflow experiment tracking has been integrated into the ml_predict_15 project for automatic logging of all training experiments.

## ✅ What Was Added

### 1. **MLflow Tracking in model_training.py**

Added automatic experiment tracking that logs:
- **Parameters**: All training configuration (target_bars, target_pct, use_smote, use_gpu, n_jobs, dataset info)
- **Metrics**: Per-model and best model performance (accuracy, F1, precision, recall, ROC AUC, training time)
- **Models**: Best performing model with versioning
- **Artifacts**: Training results CSV, configuration file, comparison plots

### 2. **Files Created**

| File | Description |
|------|-------------|
| `docs/MLFLOW_TRACKING_GUIDE.md` | Comprehensive guide (500+ lines) |
| `start_mlflow.bat` | Windows batch script to start MLflow server |
| `MLFLOW_INTEGRATION_SUMMARY.md` | This summary document |

### 3. **Dependencies Added**

Updated `requirements.txt`:
```
mlflow>=2.0.0  # For experiment tracking
```

---

## 🚀 Quick Start

### Step 1: Install MLflow

```bash
pip install mlflow>=2.0.0
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

### Step 2: Start MLflow Server

**Option A: Use batch script (Windows)**

```bash
start_mlflow.bat
```

**Option B: Manual start**

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Keep this terminal running!

### Step 3: Train Models (MLflow Enabled by Default)

```python
from src.model_training import train
import pandas as pd

# Load data
df_train = pd.read_csv('data/btc_2023.csv')

# Train with automatic MLflow tracking
models, scaler, results, best_model = train(df_train)
```

### Step 4: View Results

Open your browser:
```
http://localhost:5000
```

Navigate to experiment: `ml_predict_15/classification/crypto_price_prediction`

---

## 📊 What Gets Tracked

### Parameters (12 total)

- `target_bars`, `target_pct` - Prediction configuration
- `use_smote`, `use_gpu`, `n_jobs` - Training settings
- `dataset_shape`, `train_size`, `val_size` - Dataset info
- `class_imbalance_ratio`, `smote_applied` - Data characteristics
- `best_model_name`, `num_models_trained` - Training results

### Metrics (Per Model + Best)

**For each model** (e.g., logistic_regression, xgboost):
- `{model}_accuracy`
- `{model}_f1_score`
- `{model}_precision`
- `{model}_recall`
- `{model}_roc_auc`
- `{model}_training_time`

**Best model metrics**:
- `best_accuracy`, `best_f1_score`, `best_precision`, `best_recall`, `best_roc_auc`
- `total_training_time`, `avg_training_time`

### Artifacts

- **Best Model**: Serialized sklearn model with versioning
- **Training Results**: `training_results_summary.csv`
- **Configuration**: `training_config.txt`
- **Visualization**: `model_comparison_training.png`

---

## 💡 Usage Examples

### Default (MLflow Enabled)

```python
# MLflow tracking enabled automatically
models, scaler, results, best_model = train(df_train)
```

**Output:**
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

### Disable MLflow

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

### Complete Training Example

```python
from src.model_training import train
import pandas as pd

# Load data
df_train = pd.read_csv('data/btc_2023.csv')

# Train with full configuration
models, scaler, results, best_model = train(
    df_train,
    target_bars=45,
    target_pct=3.0,
    use_smote=True,
    use_gpu=False,
    n_jobs=-1,
    use_mlflow=True,  # Enable MLflow (default)
    mlflow_tracking_uri="http://localhost:5000"
)

print(f"Best model: {best_model}")
print(f"View results at: http://localhost:5000")
```

---

## 🎯 Key Features

### 1. **Automatic Tracking**

No code changes needed - MLflow tracking happens automatically:

```python
# Just train as usual
models, scaler, results, best_model = train(df_train)
```

### 2. **Experiment Organization**

All runs organized under:
```
ml_predict_15/classification/crypto_price_prediction
```

Each run named:
```
training_YYYYMMDD_HHMMSS
```

### 3. **Model Versioning**

Best model registered as:
```
ml_predict_15_{model_name}
```

With automatic version incrementing (v1, v2, v3, ...)

### 4. **Easy Comparison**

In MLflow UI:
1. Select multiple runs
2. Click "Compare"
3. See side-by-side parameter and metric comparison

### 5. **Full Reproducibility**

Every run includes:
- All parameters
- All metrics
- Model artifact
- Training configuration
- Results CSV
- Visualization plots

### 6. **Load Models from MLflow**

```python
import mlflow

# Load latest version
model = mlflow.sklearn.load_model("models:/ml_predict_15_xgboost/latest")

# Load specific version
model = mlflow.sklearn.load_model("models:/ml_predict_15_xgboost/3")

# Load from run ID
model = mlflow.sklearn.load_model(f"runs:/{run_id}/best_model")
```

---

## 📈 Benefits

✅ **Never Lose Experiments** - All training runs automatically saved  
✅ **Easy Comparison** - Compare runs side-by-side in UI  
✅ **Full Reproducibility** - All parameters and artifacts logged  
✅ **Model Versioning** - Track model evolution over time  
✅ **Built-in Visualization** - Automatic charts and plots  
✅ **Team Collaboration** - Share results with team  
✅ **Production Ready** - Load models directly from MLflow  

---

## 🔧 Configuration

### Default Settings

```python
use_mlflow=True  # MLflow enabled by default
mlflow_tracking_uri="http://localhost:5000"  # Default server
```

### Experiment Name

```
ml_predict_15/classification/crypto_price_prediction
```

To change, edit `src/model_training.py`:

```python
experiment_name = "ml_predict_15/classification/your_custom_name"
```

### Run Naming

```
training_YYYYMMDD_HHMMSS
```

Example: `training_20251024_123045`

---

## 🛠️ Troubleshooting

### Issue: "Connection refused"

**Solution**: Start MLflow server

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Or disable MLflow:

```python
models, scaler, results, best_model = train(df_train, use_mlflow=False)
```

### Issue: "MLflow not installed"

**Solution**: Install MLflow

```bash
pip install mlflow>=2.0.0
```

### Issue: "Port 5000 already in use"

**Solution**: Use different port

```bash
mlflow server --host 127.0.0.1 --port 8080
```

```python
models, scaler, results, best_model = train(
    df_train,
    mlflow_tracking_uri="http://localhost:8080"
)
```

---

## 📚 Documentation

### Complete Guide

See `docs/MLFLOW_TRACKING_GUIDE.md` for:
- Detailed installation instructions
- Complete usage examples
- Advanced features
- Best practices
- Troubleshooting

### Quick Commands

**Start server:**
```bash
mlflow server --host 127.0.0.1 --port 5000
```

**View UI:**
```
http://localhost:5000
```

**Train with MLflow:**
```python
models, scaler, results, best_model = train(df_train)
```

**Disable MLflow:**
```python
models, scaler, results, best_model = train(df_train, use_mlflow=False)
```

---

## 🔄 Integration with Existing Workflow

MLflow integrates seamlessly - no changes to existing code needed!

### Before (Still Works)

```python
models, scaler, results, best_model = train(df_train)
```

### After (Same Code, Now with Tracking)

```python
models, scaler, results, best_model = train(df_train)
# MLflow tracking happens automatically!
```

### Backtesting

```python
# Load model from MLflow
import mlflow

model = mlflow.sklearn.load_model("models:/ml_predict_15_xgboost/latest")
scaler = joblib.load('models/2024-01-15_14-30-45/scaler.joblib')

# Backtest as usual
from src.MLBacktester import MLBacktester
backtester = MLBacktester(model, scaler)
results = backtester.run_backtest(df_test)
```

---

## 📊 Example MLflow UI

When you open `http://localhost:5000`, you'll see:

### Experiments List
```
ml_predict_15/classification/crypto_price_prediction
  ├── training_20251024_123045 (Run 1)
  ├── training_20251024_145030 (Run 2)
  └── training_20251024_163015 (Run 3)
```

### Run Details

**Parameters Tab:**
- target_bars: 45
- target_pct: 3.0
- use_smote: True
- use_gpu: False
- n_jobs: 15
- dataset_shape: 420612x5
- best_model_name: logistic_regression

**Metrics Tab:**
- best_accuracy: 0.7234
- best_f1_score: 0.6187
- logistic_regression_accuracy: 0.7234
- xgboost_accuracy: 0.7087
- [Charts showing metric trends]

**Artifacts Tab:**
- best_model/ (downloadable model)
- results/training_results_summary.csv
- config/training_config.txt
- plots/model_comparison_training.png

---

## 🎉 Summary

### What You Get

✅ **Automatic experiment tracking** - No code changes needed  
✅ **Complete run history** - Never lose results  
✅ **Easy comparison** - Compare runs in UI  
✅ **Model versioning** - Track model evolution  
✅ **Full reproducibility** - All artifacts saved  
✅ **Production ready** - Load models from MLflow  

### How to Use

1. **Start MLflow server**: `start_mlflow.bat` or `mlflow server --host 127.0.0.1 --port 5000`
2. **Train models**: `models, scaler, results, best_model = train(df_train)`
3. **View results**: Open `http://localhost:5000`

### Next Steps

1. Start MLflow server
2. Run a training session
3. Open MLflow UI to view results
4. Compare multiple runs
5. Load best model for backtesting

**Your ML experiments are now fully tracked and reproducible!** 📊🚀✅

---

## Additional Resources

- **Complete Guide**: `docs/MLFLOW_TRACKING_GUIDE.md`
- **MLflow Docs**: https://mlflow.org/docs/latest/index.html
- **Model Training**: `src/model_training.py`

Happy tracking! 🎯📈✨
