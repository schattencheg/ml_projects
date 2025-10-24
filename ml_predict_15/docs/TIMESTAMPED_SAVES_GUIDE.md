# Timestamped Model Saves Guide

This guide explains the timestamped model saving feature that organizes training sessions into dated subfolders with comprehensive results.

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Saved Files](#saved-files)
4. [Usage](#usage)
5. [Benefits](#benefits)
6. [Best Practices](#best-practices)

---

## Overview

Every training session now automatically saves models and results to a **timestamped subdirectory** in the format:

```
models/YYYY-MM-DD_HH-MM-SS/
```

This allows you to:
- ✅ Track training history
- ✅ Compare different training runs
- ✅ Reproduce results
- ✅ Never overwrite previous models
- ✅ Organize experiments systematically

---

## Directory Structure

### Example Structure

```
ml_predict_15/
├── models/
│   ├── 2024-01-15_14-30-45/          # Training session 1
│   │   ├── logistic_regression.joblib
│   │   ├── random_forest.joblib
│   │   ├── xgboost.joblib
│   │   ├── lightgbm.joblib
│   │   ├── ...
│   │   ├── scaler.joblib
│   │   ├── logistic_regression_best.joblib
│   │   ├── training_results_summary.csv
│   │   ├── training_config.txt
│   │   └── model_comparison_training.png
│   │
│   ├── 2024-01-16_09-15-22/          # Training session 2
│   │   ├── logistic_regression.joblib
│   │   ├── ...
│   │   ├── training_results_summary.csv
│   │   ├── training_config.txt
│   │   └── model_comparison_training.png
│   │
│   └── 2024-01-17_16-45-10/          # Training session 3
│       ├── ...
│
├── plots/
│   └── model_comparison_training.png  # Latest training plot
```

### Timestamp Format

- **Format:** `YYYY-MM-DD_HH-MM-SS`
- **Example:** `2024-01-15_14-30-45`
- **Meaning:** January 15, 2024 at 2:30:45 PM

---

## Saved Files

Each timestamped directory contains:

### 1. Model Files (`.joblib`)

All trained models saved individually:

```
logistic_regression.joblib
ridge_classifier.joblib
naive_bayes.joblib
knn_k_neighbours.joblib
decision_tree.joblib
random_forest.joblib
gradient_boosting.joblib
xgboost.joblib
lightgbm.joblib
```

### 2. Scaler File

```
scaler.joblib
```

The fitted MinMaxScaler used for feature normalization.

### 3. Best Model

```
{best_model_name}_best.joblib
```

Example: `logistic_regression_best.joblib`

A copy of the best performing model for quick access.

### 4. Training Results Summary (CSV)

**File:** `training_results_summary.csv`

Contains all model metrics in a structured format:

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

**Benefits:**
- Easy to open in Excel/Google Sheets
- Can be imported into analysis tools
- Includes summary statistics
- Sorted by F1 Score

### 5. Training Configuration (TXT)

**File:** `training_config.txt`

Contains all training parameters and settings:

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

**Benefits:**
- Reproduces exact training conditions
- Documents hyperparameters
- Records dataset characteristics
- Tracks hardware settings

### 6. Training Visualization (PNG)

**File:** `model_comparison_training.png`

A copy of the training comparison plot with 6 subplots showing all metrics.

---

## Usage

### Automatic Saving

No code changes needed! Timestamped saving is automatic:

```python
from src.model_training import train

# Train models - automatically saves to timestamped directory
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

### Loading Models from Specific Session

```python
import joblib
import os

# Specify the timestamp of the session you want to load
timestamp = '2024-01-15_14-30-45'
models_dir = f'models/{timestamp}'

# Load specific model
model = joblib.load(os.path.join(models_dir, 'logistic_regression.joblib'))

# Load scaler
scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))

# Load best model
best_model = joblib.load(os.path.join(models_dir, 'logistic_regression_best.joblib'))
```

### Loading Latest Session

```python
import os
import joblib

# Get all timestamped directories
models_base = 'models'
sessions = [d for d in os.listdir(models_base) 
            if os.path.isdir(os.path.join(models_base, d))]

# Sort by timestamp (newest first)
sessions.sort(reverse=True)
latest_session = sessions[0]

print(f"Loading from latest session: {latest_session}")

# Load models from latest session
models_dir = os.path.join(models_base, latest_session)
model = joblib.load(os.path.join(models_dir, 'xgboost.joblib'))
scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
```

### Comparing Multiple Sessions

```python
import pandas as pd
import os

# Get all sessions
sessions = sorted([d for d in os.listdir('models') 
                   if os.path.isdir(os.path.join('models', d))],
                  reverse=True)

# Load results from each session
all_results = []
for session in sessions[:5]:  # Last 5 sessions
    csv_path = f'models/{session}/training_results_summary.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Get best model row
        best_row = df[df['Model'].str.contains('Best Model', na=False)]
        if not best_row.empty:
            all_results.append({
                'Session': session,
                'Best_Model': best_row['Model'].values[0].split(': ')[1],
                'Accuracy': best_row['Accuracy'].values[0],
                'F1_Score': best_row['F1_Score'].values[0],
                'Training_Time': best_row['Training_Time_Seconds'].values[0]
            })

# Create comparison dataframe
comparison_df = pd.DataFrame(all_results)
print(comparison_df)

# Example output:
#                Session        Best_Model  Accuracy  F1_Score  Training_Time
# 0  2024-01-17_16-45-10  logistic_regression    0.7456    0.6523           2.12
# 1  2024-01-16_09-15-22  random_forest          0.7234    0.6187          15.67
# 2  2024-01-15_14-30-45  xgboost                0.7087    0.5976           3.21
```

---

## Benefits

### 1. Training History

Track all your training experiments:

```
models/
├── 2024-01-15_14-30-45/  # Baseline
├── 2024-01-16_09-15-22/  # After SMOTE
├── 2024-01-16_14-20-10/  # After threshold tuning
├── 2024-01-17_10-05-33/  # After hardware acceleration
└── 2024-01-17_16-45-10/  # Final optimized version
```

### 2. Easy Comparison

Compare different configurations:

```python
# Session 1: Baseline (no SMOTE)
# F1 Score: 0.45, Training Time: 245s

# Session 2: With SMOTE
# F1 Score: 0.62, Training Time: 283s

# Session 3: With SMOTE + GPU
# F1 Score: 0.62, Training Time: 45s
```

### 3. Reproducibility

Each session contains everything needed to reproduce results:
- Exact models
- Scaler
- Configuration
- Results
- Visualization

### 4. Version Control

Never lose a good model:
- Old models are never overwritten
- Can always roll back to previous version
- Easy to A/B test different models

### 5. Experiment Tracking

Document your ML experiments:
- What changed between sessions?
- Which configuration worked best?
- How did performance evolve?

### 6. Easy Sharing

Share entire training session:
```bash
# Zip a specific session
zip -r session_2024-01-15.zip models/2024-01-15_14-30-45/

# Share with team
# They get models + results + config + visualization
```

---

## Best Practices

### 1. Naming Convention

Add notes to identify sessions:

```python
# After training, rename directory with description
import os
old_name = 'models/2024-01-15_14-30-45'
new_name = 'models/2024-01-15_14-30-45_baseline'
os.rename(old_name, new_name)

# Or add a README
with open('models/2024-01-15_14-30-45/README.txt', 'w') as f:
    f.write("Baseline model - no SMOTE, no GPU\n")
    f.write("Dataset: BTC 2022\n")
    f.write("Purpose: Establish baseline performance\n")
```

### 2. Keep Important Sessions

Delete unsuccessful experiments to save space:

```python
import shutil

# Remove a session
shutil.rmtree('models/2024-01-15_10-20-30')  # Failed experiment
```

### 3. Track Experiments

Create an experiment log:

```python
# experiments_log.csv
import pandas as pd

log = pd.DataFrame([
    {
        'Timestamp': '2024-01-15_14-30-45',
        'Description': 'Baseline - no optimizations',
        'Best_F1': 0.45,
        'Training_Time': 245,
        'Notes': 'Poor recall on minority class'
    },
    {
        'Timestamp': '2024-01-16_09-15-22',
        'Description': 'Added SMOTE',
        'Best_F1': 0.62,
        'Training_Time': 283,
        'Notes': 'Significant improvement in recall'
    },
    {
        'Timestamp': '2024-01-17_16-45-10',
        'Description': 'SMOTE + GPU acceleration',
        'Best_F1': 0.62,
        'Training_Time': 45,
        'Notes': 'Same performance, 6x faster'
    }
])

log.to_csv('experiments_log.csv', index=False)
```

### 4. Backup Important Sessions

```bash
# Backup best performing session
cp -r models/2024-01-17_16-45-10 backups/best_model_v1/

# Or use version control
git add models/2024-01-17_16-45-10/
git commit -m "Best model - F1: 0.62, Time: 45s"
```

### 5. Load Latest by Default

Create a helper function:

```python
def load_latest_models():
    """Load models from the most recent training session."""
    import os
    import joblib
    
    sessions = sorted([d for d in os.listdir('models') 
                      if os.path.isdir(os.path.join('models', d))],
                     reverse=True)
    
    if not sessions:
        raise ValueError("No training sessions found")
    
    latest = sessions[0]
    models_dir = f'models/{latest}'
    
    print(f"Loading from session: {latest}")
    
    # Load all models
    models = {}
    for file in os.listdir(models_dir):
        if file.endswith('.joblib') and file != 'scaler.joblib':
            model_name = file.replace('.joblib', '')
            models[model_name] = joblib.load(os.path.join(models_dir, file))
    
    # Load scaler
    scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
    
    return models, scaler, latest

# Usage
models, scaler, session = load_latest_models()
```

### 6. Clean Up Old Sessions

```python
import os
import shutil
from datetime import datetime, timedelta

def cleanup_old_sessions(keep_days=30):
    """Remove training sessions older than keep_days."""
    cutoff_date = datetime.now() - timedelta(days=keep_days)
    
    for session in os.listdir('models'):
        session_path = os.path.join('models', session)
        if not os.path.isdir(session_path):
            continue
        
        try:
            # Parse timestamp from directory name
            session_date = datetime.strptime(session, '%Y-%m-%d_%H-%M-%S')
            
            if session_date < cutoff_date:
                print(f"Removing old session: {session}")
                shutil.rmtree(session_path)
        except ValueError:
            # Skip directories that don't match timestamp format
            continue

# Keep only last 30 days
cleanup_old_sessions(keep_days=30)
```

---

## Advanced Usage

### Automated Experiment Tracking

```python
import pandas as pd
import os
from datetime import datetime

class ExperimentTracker:
    def __init__(self, log_file='experiments_log.csv'):
        self.log_file = log_file
        if os.path.exists(log_file):
            self.log = pd.read_csv(log_file)
        else:
            self.log = pd.DataFrame(columns=[
                'Timestamp', 'Description', 'Best_Model', 'Best_F1', 
                'Best_Accuracy', 'Training_Time', 'Dataset', 'Notes'
            ])
    
    def add_experiment(self, timestamp, description, results, 
                      best_model_name, total_time, dataset, notes=''):
        """Add experiment to log."""
        best_metrics = results[best_model_name]
        
        new_entry = {
            'Timestamp': timestamp,
            'Description': description,
            'Best_Model': best_model_name,
            'Best_F1': best_metrics['f1'],
            'Best_Accuracy': best_metrics['accuracy'],
            'Training_Time': total_time,
            'Dataset': dataset,
            'Notes': notes
        }
        
        self.log = pd.concat([self.log, pd.DataFrame([new_entry])], 
                            ignore_index=True)
        self.log.to_csv(self.log_file, index=False)
    
    def get_best_experiment(self, metric='Best_F1'):
        """Get best experiment by metric."""
        return self.log.loc[self.log[metric].idxmax()]
    
    def compare_experiments(self, timestamps):
        """Compare specific experiments."""
        return self.log[self.log['Timestamp'].isin(timestamps)]

# Usage
tracker = ExperimentTracker()

# After training
models, scaler, results, best_model = train(df_train)
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

tracker.add_experiment(
    timestamp=timestamp,
    description='SMOTE + GPU acceleration',
    results=results,
    best_model_name=best_model,
    total_time=45.23,
    dataset='BTC 2022',
    notes='Best configuration so far'
)

# View best experiment
best = tracker.get_best_experiment('Best_F1')
print(f"Best experiment: {best['Timestamp']} - F1: {best['Best_F1']}")
```

---

## Summary

### Quick Reference

**Directory format:**
```
models/YYYY-MM-DD_HH-MM-SS/
```

**Files saved:**
- All model `.joblib` files
- `scaler.joblib`
- `{best_model}_best.joblib`
- `training_results_summary.csv`
- `training_config.txt`
- `model_comparison_training.png`

**Load latest session:**
```python
sessions = sorted(os.listdir('models'), reverse=True)
latest = sessions[0]
model = joblib.load(f'models/{latest}/xgboost.joblib')
```

**Compare sessions:**
```python
for session in sessions:
    df = pd.read_csv(f'models/{session}/training_results_summary.csv')
    # Analyze results
```

### Benefits

✅ Never overwrite models
✅ Track training history
✅ Easy comparison
✅ Full reproducibility
✅ Organized experiments
✅ Easy sharing

---

## Additional Resources

- [Model Training Documentation](../src/model_training.py)
- [Progress Tracking Guide](PROGRESS_TRACKING_GUIDE.md)
- [Hardware Acceleration Guide](HARDWARE_ACCELERATION_GUIDE.md)

Happy experimenting! 🚀📊🎯
