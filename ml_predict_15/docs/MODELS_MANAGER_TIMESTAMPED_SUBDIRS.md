# ModelsManager Timestamped Subdirectories

## Overview

The `ModelsManager` class now uses **timestamped subdirectories** to organize saved models, providing better organization and easier management of multiple training sessions.

## Changes Summary

### Old Structure (Flat)
```
models/
├── logistic_regression_20251030_135527.joblib
├── ridge_classifier_20251030_135527.joblib
├── xgboost_20251030_135527.joblib
├── scaler_20251030_135527.joblib
└── metadata_20251030_135527.joblib
```

### New Structure (Timestamped Subdirectories)
```
models/
└── 2025-10-30_13-55-27/
    ├── logistic_regression.joblib
    ├── ridge_classifier.joblib
    ├── xgboost.joblib
    ├── scaler.joblib
    └── metadata.joblib
```

## Benefits

### 1. Better Organization
- Each training session has its own folder
- All related files grouped together
- Easy to see what belongs to which session

### 2. Never Overwrite Models
- New subdirectory created for each save
- Old models automatically preserved
- Safe experimentation without data loss

### 3. Easy Comparison
- Compare different training sessions side-by-side
- All files from one session in one place
- Clear timestamp-based naming

### 4. Easy Sharing
- Zip entire subdirectory and share
- Team gets complete training session
- No missing dependencies

### 5. Easy Cleanup
- Delete old subdirectories to save space
- Keep only best performing sessions
- Simple maintenance

### 6. Consistent with Project Standards
- Matches the pattern used in other modules
- Follows established best practices
- Familiar structure for team members

## Usage

### Saving Models

```python
from src.ModelsManager import ModelsManager

# Initialize manager
manager = ModelsManager(models_dir='models')

# Create and train models
models = manager.create_models(enabled_only=True)
# ... train models ...

# Save models (creates timestamped subdirectory automatically)
saved_paths = manager.save_models(models, scaler)

# Output:
# ======================================================================
# SAVING MODELS TO: models/2025-10-30_13-55-27
# ======================================================================
# ✓ Saved logistic_regression
# ✓ Saved ridge_classifier
# ✓ Saved xgboost
# ✓ Saved scaler
# ✓ Saved metadata
# 
# ✓ Saved 7 models successfully to models/2025-10-30_13-55-27
# ======================================================================
```

### Loading Models

#### Load Latest Models
```python
# Load from most recent subdirectory
models, scaler, metadata = manager.load_models(suffix='latest')

# Output:
# ======================================================================
# LOADING MODELS FROM: models/2025-10-30_13-55-27
# ======================================================================
# ✓ Loaded metadata
# ✓ Loaded logistic_regression
# ✓ Loaded ridge_classifier
# ✓ Loaded xgboost
# ✓ Loaded scaler
# 
# ✓ Loaded 7 models successfully from models/2025-10-30_13-55-27
# ======================================================================
```

#### Load Specific Version
```python
# Load from specific timestamp
timestamp = '2025-10-30_13-55-27'
models, scaler, metadata = manager.load_models(suffix=timestamp)
```

### List All Saved Versions

```python
# Get all saved model versions
versions = manager.list_saved_models()

for timestamp, metadata in versions:
    print(f"Timestamp: {timestamp}")
    print(f"  Models: {', '.join(metadata['models'])}")
    print(f"  Has scaler: {metadata['has_scaler']}")
    print(f"  Save dir: {metadata['save_dir']}")
    print()
```

## API Changes

### `save_models(models, scaler=None, suffix='')`

**Changes:**
- Now creates a timestamped subdirectory: `models/{timestamp}/`
- Timestamp format changed to: `YYYY-MM-DD_HH-MM-SS` (more readable)
- Files saved without timestamp suffix (e.g., `xgboost.joblib` instead of `xgboost_20251030_135527.joblib`)
- Metadata includes `save_dir` field

**Parameters:**
- `models` (dict): Dictionary of model_name -> trained_model
- `scaler` (sklearn scaler, optional): Fitted scaler
- `suffix` (str, optional): Custom timestamp (default: auto-generated)

**Returns:**
- `dict`: Paths where models were saved

### `load_models(suffix='latest')`

**Changes:**
- Now loads from timestamped subdirectory: `models/{timestamp}/`
- Looks for files without timestamp suffix
- Better error messages with full paths

**Parameters:**
- `suffix` (str): Timestamp subdirectory name or 'latest' (default: 'latest')

**Returns:**
- `tuple`: (models_dict, scaler, metadata)

### `list_saved_models()`

**Changes:**
- Now lists timestamped subdirectories instead of files
- Validates subdirectories by checking for `metadata.joblib`
- Returns sorted list (newest first)

**Returns:**
- `list`: List of (timestamp, metadata) tuples

### `_find_latest_suffix()`

**Changes:**
- Now finds latest timestamped subdirectory
- Validates by checking for `metadata.joblib`
- Sorts lexicographically (works with YYYY-MM-DD format)

**Returns:**
- `str`: Latest timestamp or None

## Migration

### Automatic Migration

Use the provided migration script to convert existing models:

```bash
# Preview changes (dry run)
python migrate_models_to_subdirs.py --dry-run

# Execute migration
python migrate_models_to_subdirs.py --execute
```

### Manual Migration

If you prefer manual migration:

1. Create subdirectory: `models/YYYY-MM-DD_HH-MM-SS/`
2. Move model files into subdirectory
3. Remove timestamp suffix from filenames:
   - `xgboost_20251030_135527.joblib` → `xgboost.joblib`
   - `scaler_20251030_135527.joblib` → `scaler.joblib`
   - `metadata_20251030_135527.joblib` → `metadata.joblib`
4. Update metadata to include `save_dir` field

## Testing

Run the test script to verify functionality:

```bash
python test_models_manager.py
```

This will:
1. Create sample models
2. Save to timestamped subdirectory
3. Load from latest subdirectory
4. List all saved versions
5. Verify models work correctly

## Backward Compatibility

### Breaking Changes
- Old flat structure files will NOT be loaded automatically
- Must migrate existing models using migration script

### Why Not Backward Compatible?
- Clean separation between old and new structure
- Prevents confusion and errors
- Encourages migration to better structure
- Simplifies code maintenance

### Migration Path
1. Run migration script on existing models
2. Update any scripts that hardcode model paths
3. Use `load_models(suffix='latest')` for automatic loading

## Directory Structure Examples

### Single Training Session
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

### Multiple Training Sessions
```
models/
├── 2025-10-30_13-55-27/  # Latest
│   ├── logistic_regression.joblib
│   ├── xgboost.joblib
│   ├── scaler.joblib
│   └── metadata.joblib
├── 2025-10-29_10-30-15/  # Previous
│   ├── logistic_regression.joblib
│   ├── xgboost.joblib
│   ├── scaler.joblib
│   └── metadata.joblib
└── 2025-10-28_16-45-00/  # Older
    ├── logistic_regression.joblib
    ├── xgboost.joblib
    ├── scaler.joblib
    └── metadata.joblib
```

## Best Practices

### 1. Use Descriptive Timestamps
The timestamp format `YYYY-MM-DD_HH-MM-SS` is:
- Human-readable
- Sorts correctly lexicographically
- Includes date and time for uniqueness

### 2. Keep Experiment Log
Maintain a separate log file tracking what each session represents:

```csv
Timestamp,Description,Best_Model,Accuracy,Notes
2025-10-30_13-55-27,Baseline,xgboost,0.85,Initial run
2025-10-29_10-30-15,With SMOTE,random_forest,0.87,Better recall
2025-10-28_16-45-00,GPU enabled,xgboost,0.86,Faster training
```

### 3. Cleanup Old Sessions
Periodically remove old sessions to save space:

```python
import os
import shutil
from datetime import datetime, timedelta

def cleanup_old_models(models_dir='models', days_to_keep=30):
    """Remove model subdirectories older than specified days."""
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path):
            # Parse timestamp from directory name
            try:
                timestamp = datetime.strptime(item, '%Y-%m-%d_%H-%M-%S')
                if timestamp < cutoff_date:
                    shutil.rmtree(item_path)
                    print(f"Removed old session: {item}")
            except ValueError:
                pass  # Not a valid timestamp directory
```

### 4. Backup Best Sessions
Keep backups of your best performing sessions:

```bash
# Zip best session
zip -r models_best_2025-10-30.zip models/2025-10-30_13-55-27/

# Or copy to backup location
cp -r models/2025-10-30_13-55-27/ backups/
```

### 5. Load Latest by Default
Always use `suffix='latest'` unless you need a specific version:

```python
# Good - loads latest automatically
models, scaler, metadata = manager.load_models()

# Only use specific timestamp when needed
models, scaler, metadata = manager.load_models(suffix='2025-10-30_13-55-27')
```

## Troubleshooting

### Issue: No saved models found

**Cause:** No timestamped subdirectories in models folder

**Solution:**
1. Check if models directory exists
2. Run migration script if you have old flat structure
3. Train and save new models

### Issue: Model directory not found

**Cause:** Specified timestamp doesn't exist

**Solution:**
1. Use `list_saved_models()` to see available timestamps
2. Use `suffix='latest'` to load most recent
3. Check for typos in timestamp

### Issue: Files have wrong names

**Cause:** Manual migration with incorrect filenames

**Solution:**
1. Files should be named without timestamp suffix
2. Use migration script for automatic conversion
3. Check file naming: `xgboost.joblib` not `xgboost_20251030.joblib`

## Integration with Other Modules

### With Trainer Module
```python
from src.ModelsManager import ModelsManager
from src.Trainer import Trainer

manager = ModelsManager()
trainer = Trainer()

# Train models
models, scaler = trainer.train(X_train, y_train)

# Save with ModelsManager
manager.save_models(models, scaler)
```

### With Tester Module
```python
from src.ModelsManager import ModelsManager
from src.Tester import Tester

manager = ModelsManager()
tester = Tester()

# Load models
models, scaler, metadata = manager.load_models(suffix='latest')

# Test models
results = tester.test(models, scaler, X_test, y_test)
```

### With Backtest Modules
```python
from src.ModelsManager import ModelsManager
from src.BacktestNoLib import BacktestNoLib

manager = ModelsManager()

# Load specific model
models, scaler, metadata = manager.load_models(suffix='2025-10-30_13-55-27')
model = models['xgboost']

# Run backtest
backtester = BacktestNoLib(model, scaler)
results = backtester.run(data)
```

## Summary

The timestamped subdirectory structure provides:

✅ **Better Organization** - Each session in its own folder  
✅ **Never Overwrite** - Automatic preservation of old models  
✅ **Easy Comparison** - Side-by-side session comparison  
✅ **Easy Sharing** - Zip and share complete sessions  
✅ **Easy Cleanup** - Delete old subdirectories  
✅ **Consistent** - Matches project standards  
✅ **Scalable** - Handles many training sessions  

This change aligns with best practices and makes model management more efficient and reliable.
