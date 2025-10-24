# Timestamped Model Saves - Summary

## 🎯 What Was Added

Models and training results are now automatically saved to **timestamped subdirectories** for better organization and experiment tracking.

---

## ✅ Features Implemented

### 1. Timestamped Directory Structure

Every training session creates a new subdirectory:

```
models/
├── 2024-01-15_14-30-45/
├── 2024-01-16_09-15-22/
└── 2024-01-17_16-45-10/
```

**Format:** `YYYY-MM-DD_HH-MM-SS`

### 2. Comprehensive File Saving

Each session directory contains:

```
models/2024-01-15_14-30-45/
├── logistic_regression.joblib          # All trained models
├── random_forest.joblib
├── xgboost.joblib
├── lightgbm.joblib
├── ...
├── scaler.joblib                       # Fitted scaler
├── logistic_regression_best.joblib     # Best model copy
├── training_results_summary.csv        # ✨ NEW: Results CSV
├── training_config.txt                 # ✨ NEW: Config file
└── model_comparison_training.png       # Training plot
```

### 3. Training Results CSV

**File:** `training_results_summary.csv`

Contains all metrics in spreadsheet format:

```csv
Model,Accuracy,F1_Score,Precision,Recall,ROC_AUC,Training_Time_Seconds
logistic_regression,0.7234,0.6187,0.6621,0.5834,0.7892,2.34
random_forest,0.7156,0.6089,0.6543,0.5701,0.7823,15.67
xgboost,0.7087,0.5976,0.6234,0.5567,0.7756,3.21
...
SUMMARY,,,,,,
Best Model: logistic_regression,0.7234,0.6187,0.6621,0.5834,0.7892,2.34
Total Training Time,,,,,,"83.45s (1.39min)"
Average Time per Model,,,,,,"10.43s"
```

### 4. Training Configuration File

**File:** `training_config.txt`

Documents all training parameters:

```
Training Configuration
================================================================================
Timestamp: 2024-01-15_14-30-45
Target Bars: 45
Target Percentage: 3.0%
SMOTE Enabled: True
GPU Enabled: False
CPU Cores: 15
Dataset Shape: (100000, 50)
Class Imbalance Ratio: 5.67:1

Training Summary
================================================================================
Total Models Trained: 8
Best Model: logistic_regression
Best Accuracy: 0.7234
Total Training Time: 83.45s (1.39min)
Average Time per Model: 10.43s
```

---

## 🚀 Usage

### Automatic (No Code Changes!)

Timestamped saving happens automatically:

```python
from src.model_training import train

# Train models
models, scaler, results, best_model = train(df_train)

# Output:
# ================================================================================
# SAVING MODELS AND RESULTS
# ================================================================================
# Save directory: models/2024-01-15_14-30-45
# Models saved to: models/2024-01-15_14-30-45/
# Best model saved to: models/2024-01-15_14-30-45/logistic_regression_best.joblib
# Training results saved to: models/2024-01-15_14-30-45/training_results_summary.csv
# Training config saved to: models/2024-01-15_14-30-45/training_config.txt
# Training plot copied to: models/2024-01-15_14-30-45/model_comparison_training.png
# ================================================================================
```

### Loading from Specific Session

```python
import joblib
import os

# Load from specific timestamp
timestamp = '2024-01-15_14-30-45'
models_dir = f'models/{timestamp}'

# Load model
model = joblib.load(os.path.join(models_dir, 'xgboost.joblib'))

# Load scaler
scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
```

### Loading Latest Session

```python
import os

# Get all sessions (sorted newest first)
sessions = sorted([d for d in os.listdir('models') 
                   if os.path.isdir(os.path.join('models', d))],
                  reverse=True)

latest_session = sessions[0]
print(f"Latest session: {latest_session}")

# Load from latest
model = joblib.load(f'models/{latest_session}/xgboost.joblib')
scaler = joblib.load(f'models/{latest_session}/scaler.joblib')
```

### Comparing Multiple Sessions

```python
import pandas as pd

# Compare last 3 sessions
sessions = sorted(os.listdir('models'), reverse=True)[:3]

for session in sessions:
    csv_path = f'models/{session}/training_results_summary.csv'
    df = pd.read_csv(csv_path)
    
    # Get best model info
    best_row = df[df['Model'].str.contains('Best Model', na=False)]
    print(f"\nSession: {session}")
    print(f"  Best Model: {best_row['Model'].values[0]}")
    print(f"  F1 Score: {best_row['F1_Score'].values[0]}")
    print(f"  Training Time: {best_row['Training_Time_Seconds'].values[0]}s")

# Output:
# Session: 2024-01-17_16-45-10
#   Best Model: Best Model: logistic_regression
#   F1 Score: 0.6523
#   Training Time: 2.12s
#
# Session: 2024-01-16_09-15-22
#   Best Model: Best Model: random_forest
#   F1 Score: 0.6187
#   Training Time: 15.67s
```

---

## 📊 Benefits

### 1. Training History

Track all your experiments:

```
models/
├── 2024-01-15_14-30-45/  # Baseline
├── 2024-01-16_09-15-22/  # After SMOTE
├── 2024-01-16_14-20-10/  # After threshold tuning
├── 2024-01-17_10-05-33/  # After hardware acceleration
└── 2024-01-17_16-45-10/  # Final optimized version ✓
```

### 2. Never Overwrite Models

Old models are preserved:
- Can always roll back
- Compare different versions
- A/B test models

### 3. Easy Comparison

Compare configurations side-by-side:

| Session | Best Model | F1 Score | Training Time |
|---------|-----------|----------|---------------|
| 2024-01-15_14-30-45 | Logistic | 0.45 | 245s |
| 2024-01-16_09-15-22 | Logistic | 0.62 | 283s |
| 2024-01-17_16-45-10 | Logistic | 0.62 | 45s ✓ |

### 4. Full Reproducibility

Each session contains everything needed:
- ✅ All models
- ✅ Scaler
- ✅ Configuration
- ✅ Results (CSV)
- ✅ Visualization

### 5. Easy Sharing

Share entire training session:

```bash
# Zip a session
zip -r session_2024-01-15.zip models/2024-01-15_14-30-45/

# Share with team
# They get: models + scaler + results + config + plot
```

### 6. CSV for Analysis

Open results in Excel/Google Sheets:
- Sort by any metric
- Create charts
- Share with stakeholders
- Import into BI tools

---

## 📁 Files Modified/Created

### Modified Files

**`src/model_training.py`**

Changes:
1. Added `from datetime import datetime` import
2. Modified `train()` function:
   - Creates timestamped directory: `models/YYYY-MM-DD_HH-MM-SS/`
   - Saves all models to timestamped directory
   - Saves best model separately
   - Creates `training_results_summary.csv` with all metrics
   - Creates `training_config.txt` with parameters
   - Copies training plot to timestamped directory
   - Prints save locations

**Total new code:** ~120 lines

### Created Files

1. **`docs/TIMESTAMPED_SAVES_GUIDE.md`** (~700 lines)
   - Complete guide to timestamped saves
   - Directory structure explanation
   - All saved files documented
   - Usage examples
   - Best practices
   - Advanced usage (experiment tracking, cleanup)

2. **`TIMESTAMPED_SAVES_SUMMARY.md`** (this file)
   - Quick reference
   - Usage examples
   - Benefits overview

---

## 💡 Use Cases

### 1. Track Optimization Progress

```
2024-01-15_14-30-45: Baseline (F1: 0.45)
2024-01-16_09-15-22: + SMOTE (F1: 0.62) ✓ +37% improvement
2024-01-17_10-05-33: + GPU (F1: 0.62, Time: 45s) ✓ 6x faster
```

### 2. A/B Testing

```python
# Train with different configurations
# Session 1: SMOTE enabled
models1, _, _, _ = train(df_train, use_smote=True)

# Session 2: SMOTE disabled
models2, _, _, _ = train(df_train, use_smote=False)

# Compare results from CSV files
```

### 3. Rollback to Previous Version

```python
# Current model not performing well in production
# Load previous best model
model = joblib.load('models/2024-01-16_09-15-22/xgboost_best.joblib')
scaler = joblib.load('models/2024-01-16_09-15-22/scaler.joblib')
```

### 4. Reproduce Results

```python
# Read config from previous session
with open('models/2024-01-15_14-30-45/training_config.txt') as f:
    config = f.read()
    print(config)

# Use same parameters to reproduce
models, scaler, results, best_model = train(
    df_train,
    target_bars=45,
    target_pct=3.0,
    use_smote=True,
    use_gpu=False,
    n_jobs=15
)
```

### 5. Share with Team

```bash
# Package a successful experiment
tar -czf best_model.tar.gz models/2024-01-17_16-45-10/

# Email to team with note:
# "Best model so far - F1: 0.62, Training time: 45s"
# "All files included: models, scaler, results, config, plot"
```

---

## 🎓 Best Practices

### 1. Add Descriptions

Rename directories with descriptions:

```python
import os

old_name = 'models/2024-01-15_14-30-45'
new_name = 'models/2024-01-15_14-30-45_baseline'
os.rename(old_name, new_name)
```

Or add README files:

```python
with open('models/2024-01-15_14-30-45/README.txt', 'w') as f:
    f.write("Baseline model - no optimizations\n")
    f.write("Dataset: BTC 2022\n")
    f.write("Purpose: Establish baseline performance\n")
```

### 2. Keep Experiment Log

```python
import pandas as pd

log = pd.DataFrame([
    {
        'Timestamp': '2024-01-15_14-30-45',
        'Description': 'Baseline',
        'Best_F1': 0.45,
        'Training_Time': 245,
        'Notes': 'Poor recall'
    },
    {
        'Timestamp': '2024-01-16_09-15-22',
        'Description': 'Added SMOTE',
        'Best_F1': 0.62,
        'Training_Time': 283,
        'Notes': 'Much better recall!'
    }
])

log.to_csv('experiments_log.csv', index=False)
```

### 3. Clean Up Old Sessions

```python
import shutil
from datetime import datetime, timedelta

# Remove sessions older than 30 days
cutoff = datetime.now() - timedelta(days=30)

for session in os.listdir('models'):
    try:
        session_date = datetime.strptime(session, '%Y-%m-%d_%H-%M-%S')
        if session_date < cutoff:
            shutil.rmtree(f'models/{session}')
            print(f"Removed old session: {session}")
    except ValueError:
        continue
```

### 4. Backup Best Sessions

```bash
# Backup best performing model
cp -r models/2024-01-17_16-45-10 backups/best_model_v1/
```

### 5. Load Latest by Default

```python
def load_latest_session():
    """Load models from most recent session."""
    sessions = sorted(os.listdir('models'), reverse=True)
    latest = sessions[0]
    
    print(f"Loading from: {latest}")
    
    model = joblib.load(f'models/{latest}/xgboost_best.joblib')
    scaler = joblib.load(f'models/{latest}/scaler.joblib')
    
    return model, scaler, latest

# Usage
model, scaler, session = load_latest_session()
```

---

## 🎉 Summary

### What You Get

✅ **Timestamped directories** - Never overwrite models
✅ **Training results CSV** - Easy analysis in Excel
✅ **Configuration file** - Full reproducibility
✅ **Training plot copy** - Visual results
✅ **Best model copy** - Quick access
✅ **Automatic saving** - No code changes needed

### Directory Structure

```
models/YYYY-MM-DD_HH-MM-SS/
├── {model_name}.joblib (all models)
├── scaler.joblib
├── {best_model}_best.joblib
├── training_results_summary.csv  ← NEW
├── training_config.txt           ← NEW
└── model_comparison_training.png
```

### Key Benefits

1. **Track history** - See all training experiments
2. **Never lose models** - Old versions preserved
3. **Easy comparison** - Compare sessions side-by-side
4. **Full reproducibility** - All info saved
5. **Easy sharing** - Package and share sessions
6. **CSV analysis** - Open results in Excel

---

## 🚀 Next Steps

1. **Retrain your models** to see timestamped saves:
   ```bash
   python train_and_save_models.py
   ```

2. **Check the new directory** in `models/`

3. **Open the CSV** in Excel to analyze results

4. **Read the guide** at `docs/TIMESTAMPED_SAVES_GUIDE.md`

---

## 📚 Documentation

- **Complete Guide:** `docs/TIMESTAMPED_SAVES_GUIDE.md`
- **This Summary:** `TIMESTAMPED_SAVES_SUMMARY.md`
- **Implementation:** `src/model_training.py`

---

**Your training sessions are now fully organized and tracked!** 📂📊🎯

Never lose a model again! 🚀
