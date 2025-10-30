# ModelsManager Timestamped Subdirectories - Summary

## What Changed

The `ModelsManager` class now saves and loads models using **timestamped subdirectories** instead of flat file structure with timestamp suffixes.

### Before (Flat Structure)
```
models/
├── logistic_regression_20251030_135527.joblib
├── xgboost_20251030_135527.joblib
├── scaler_20251030_135527.joblib
└── metadata_20251030_135527.joblib
```

### After (Timestamped Subdirectories)
```
models/
└── 2025-10-30_13-55-27/
    ├── logistic_regression.joblib
    ├── xgboost.joblib
    ├── scaler.joblib
    └── metadata.joblib
```

## Key Benefits

✅ **Better Organization** - Each training session in its own folder  
✅ **Never Overwrite** - New subdirectory for each save  
✅ **Easy Comparison** - All files from one session together  
✅ **Easy Sharing** - Zip entire subdirectory  
✅ **Easy Cleanup** - Delete old subdirectories  
✅ **Consistent** - Matches project standards (same as model_training.py pattern)

## Files Modified

### `src/ModelsManager.py`
- **`save_models()`** - Creates timestamped subdirectory, saves files without suffix
- **`load_models()`** - Loads from timestamped subdirectory
- **`_find_latest_suffix()`** - Finds latest subdirectory
- **`list_saved_models()`** - Lists all subdirectories

### Timestamp Format Change
- **Old:** `20251030_135527` (YYYYMMDD_HHMMSS)
- **New:** `2025-10-30_13-55-27` (YYYY-MM-DD_HH-MM-SS)
- More readable and sorts correctly

## Files Created

1. **`test_models_manager.py`** - Test script demonstrating new functionality
2. **`migrate_models_to_subdirs.py`** - Migration script for existing models
3. **`docs/MODELS_MANAGER_TIMESTAMPED_SUBDIRS.md`** - Complete documentation

## Usage

### No Changes Required!

The API remains the same. Your existing code will work:

```python
from src.ModelsManager import ModelsManager

manager = ModelsManager(models_dir='models')

# Save models (creates timestamped subdirectory automatically)
manager.save_models(models, scaler)

# Load latest models (from latest subdirectory)
models, scaler, metadata = manager.load_models(suffix='latest')

# List all versions
versions = manager.list_saved_models()
```

### Output Example

**Saving:**
```
======================================================================
SAVING MODELS TO: models/2025-10-30_13-55-27
======================================================================
✓ Saved logistic_regression
✓ Saved xgboost
✓ Saved scaler
✓ Saved metadata

✓ Saved 7 models successfully to models/2025-10-30_13-55-27
======================================================================
```

**Loading:**
```
======================================================================
LOADING MODELS FROM: models/2025-10-30_13-55-27
======================================================================
✓ Loaded metadata
✓ Loaded logistic_regression
✓ Loaded xgboost
✓ Loaded scaler

✓ Loaded 7 models successfully from models/2025-10-30_13-55-27
======================================================================
```

## Migration

### For Existing Models

If you have existing models in the old flat structure:

```bash
# Preview changes
python migrate_models_to_subdirs.py --dry-run

# Execute migration
python migrate_models_to_subdirs.py --execute
```

The script will:
1. Group files by timestamp
2. Create subdirectories
3. Move files and remove timestamp suffixes
4. Update metadata
5. Convert timestamp format (old → new)

### Example Migration

**Before:**
```
models/
├── xgboost_20251030_135527.joblib
├── scaler_20251030_135527.joblib
└── metadata_20251030_135527.joblib
```

**After:**
```
models/
└── 2025-10-30_13-55-27/
    ├── xgboost.joblib
    ├── scaler.joblib
    └── metadata.joblib
```

## Testing

Run the test script to verify everything works:

```bash
python test_models_manager.py
```

This will:
1. Create and train sample models
2. Save to timestamped subdirectory
3. Load from latest subdirectory
4. List all saved versions
5. Verify models work correctly

## Integration

### Already Integrated

The `run_me.py` script already expects this structure:

```python
# From run_me.py main_backtest() function
models_subfolders = [x for x in os.listdir(PATH_MODELS) 
                     if os.path.isdir(os.path.join(PATH_MODELS, x))]
last_experiment = max(models_subfolders, 
                     key=lambda x: datetime.strptime(x, '%Y-%m-%d_%H-%M-%S'))
```

### Works With All Modules

- ✅ `Trainer` - Saves models via ModelsManager
- ✅ `Tester` - Loads models via ModelsManager
- ✅ `ReportManager` - References model directories
- ✅ `HealthManager` - Tracks model versions
- ✅ `BacktestNoLib` - Loads specific model versions
- ✅ `BacktestBacktesting` - Loads specific model versions
- ✅ `BacktestBacktrader` - Loads specific model versions

## Directory Structure Examples

### Single Session
```
models/
└── 2025-10-30_13-55-27/
    ├── logistic_regression.joblib
    ├── ridge_classifier.joblib
    ├── naive_bayes.joblib
    ├── decision_tree.joblib
    ├── random_forest.joblib
    ├── xgboost.joblib
    ├── lightgbm.joblib
    ├── scaler.joblib
    └── metadata.joblib
```

### Multiple Sessions
```
models/
├── 2025-10-30_13-55-27/  # Latest (baseline)
│   ├── xgboost.joblib
│   ├── scaler.joblib
│   └── metadata.joblib
├── 2025-10-29_10-30-15/  # With SMOTE
│   ├── xgboost.joblib
│   ├── scaler.joblib
│   └── metadata.joblib
└── 2025-10-28_16-45-00/  # With GPU
    ├── xgboost.joblib
    ├── scaler.joblib
    └── metadata.joblib
```

## Best Practices

### 1. Always Use 'latest'
```python
# Good - loads latest automatically
models, scaler, metadata = manager.load_models()

# Only use specific timestamp when needed
models, scaler, metadata = manager.load_models(suffix='2025-10-30_13-55-27')
```

### 2. Keep Experiment Log
Track what each session represents:

```csv
Timestamp,Description,Best_Model,Accuracy,Notes
2025-10-30_13-55-27,Baseline,xgboost,0.85,Initial run
2025-10-29_10-30-15,With SMOTE,random_forest,0.87,Better recall
2025-10-28_16-45-00,GPU enabled,xgboost,0.86,Faster training
```

### 3. Cleanup Old Sessions
```python
# Keep only last 30 days
import os
import shutil
from datetime import datetime, timedelta

cutoff = datetime.now() - timedelta(days=30)
for item in os.listdir('models'):
    if os.path.isdir(os.path.join('models', item)):
        try:
            timestamp = datetime.strptime(item, '%Y-%m-%d_%H-%M-%S')
            if timestamp < cutoff:
                shutil.rmtree(os.path.join('models', item))
        except ValueError:
            pass
```

### 4. Backup Best Sessions
```bash
# Zip best performing session
zip -r models_best.zip models/2025-10-30_13-55-27/

# Or copy to backup location
cp -r models/2025-10-30_13-55-27/ backups/
```

## Troubleshooting

### No saved models found
- Check if models directory exists
- Run migration script if you have old flat structure
- Train and save new models

### Model directory not found
- Use `list_saved_models()` to see available timestamps
- Use `suffix='latest'` to load most recent
- Check for typos in timestamp

### Files have wrong names
- Files should be named without timestamp suffix
- Use migration script for automatic conversion
- Example: `xgboost.joblib` not `xgboost_20251030.joblib`

## Quick Reference

### Save Models
```python
manager.save_models(models, scaler)
# Creates: models/YYYY-MM-DD_HH-MM-SS/
```

### Load Latest
```python
models, scaler, metadata = manager.load_models()
# or explicitly:
models, scaler, metadata = manager.load_models(suffix='latest')
```

### Load Specific Version
```python
models, scaler, metadata = manager.load_models(suffix='2025-10-30_13-55-27')
```

### List All Versions
```python
versions = manager.list_saved_models()
for timestamp, metadata in versions:
    print(f"{timestamp}: {metadata['models']}")
```

### Migrate Old Models
```bash
python migrate_models_to_subdirs.py --dry-run  # Preview
python migrate_models_to_subdirs.py --execute  # Execute
```

### Test
```bash
python test_models_manager.py
```

## Documentation

- **Complete Guide:** `docs/MODELS_MANAGER_TIMESTAMPED_SUBDIRS.md`
- **This Summary:** `TIMESTAMPED_SUBDIRS_SUMMARY.md`
- **Test Script:** `test_models_manager.py`
- **Migration Script:** `migrate_models_to_subdirs.py`

## Summary

This change brings `ModelsManager` in line with project standards and best practices:

✅ Consistent with existing patterns (model_training.py)  
✅ Better organization and management  
✅ No API changes - existing code works  
✅ Easy migration path provided  
✅ Comprehensive documentation  
✅ Test script included  

The timestamped subdirectory structure makes model management more efficient, organized, and scalable.
