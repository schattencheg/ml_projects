# Migration Guide - New Architecture

## Overview

The project has been refactored to use a clean, class-based architecture. This guide helps you migrate from the old structure to the new one.

## What Changed

### 1. **Main Script Updated**

**Old:** `run_me.py` (moved to `old/run_me_old.py`)
- Used functional approach
- Limited features
- Hard to extend

**New:** `run_me.py`
- Uses class-based architecture
- Comprehensive reporting
- Health monitoring
- Easy to customize

### 2. **Examples Organized**

**Old:** Example files scattered in root directory

**New:** All examples in `examples/` folder
- `example_complete_workflow.py` - Full pipeline ⭐
- `example_refactored_workflow.py` - Architecture demo
- `example_crypto_features.py` - Crypto features
- `backtest_*.py` - Backtesting examples
- See `examples/README.md` for full list

### 3. **Outdated Files Moved**

**Location:** `old/` folder

**Files moved:**
- `run_me_old.py` - Old main script
- `train_and_save_models.py` - Old training
- `test_models.py` - Old testing
- `analyze_features.py` - Old analysis
- `diagnose_data.py` - Old diagnostics
- `create_notebook.py` - Old notebook creator

See `old/README.md` for details.

---

## Quick Start with New Structure

### Run the Main Script

```bash
python run_me.py
```

This will:
1. Load data
2. Generate features
3. Create target
4. Train models
5. Test models
6. Generate reports (CSV + PNG)
7. Save models
8. Set up health monitoring

### Run Complete Example

```bash
python examples/example_complete_workflow.py
```

Same as above, with more detailed output.

---

## Migration Steps

### Step 1: Update Imports

**Old:**
```python
from src.model_training import train, test
from src.data_preparation import prepare_data
```

**New:**
```python
from src.ModelsManager import ModelsManager
from src.FeaturesGeneratorNew import FeaturesGenerator
from src.Trainer import Trainer
from src.Tester import Tester
from src.ReportManager import ReportManager
from src.HealthManager import HealthManager
```

### Step 2: Update Training Code

**Old:**
```python
# Prepare data
X, y = prepare_data(df, target_bars=15, target_pct=3.0)

# Train models
models, scaler, results, best = train(X, y)

# Test models
test_results = test(models, scaler, X_test, y_test)
```

**New:**
```python
# Generate features
fg = FeaturesGenerator()
df_features = fg.generate_features(df, method='classical')

# Create target
df_target = fg.create_target(df_features, target_bars=15, target_pct=3.0)

# Prepare data
X, y = df_target[feature_cols], df_target['target']

# Create models
models_manager = ModelsManager()
models = models_manager.create_models()

# Train models
trainer = Trainer(use_smote=True, optimize_threshold=True)
trained_models, scaler, results, best = trainer.train(models, X, y)

# Test models
tester = Tester(scaler=scaler)
test_results = tester.test(trained_models, X_test, y_test)

# Generate reports
report_manager = ReportManager()
report_manager.export_full_report(results, test_results, y_test)
```

### Step 3: Update Model Saving

**Old:**
```python
# Models saved automatically in train()
```

**New:**
```python
# Explicit model saving with versioning
models_manager.save_models(trained_models, scaler, suffix='20251030')

# Load models later
models, scaler, metadata = models_manager.load_models(suffix='latest')
```

### Step 4: Add Health Monitoring

**New feature - not available in old version:**
```python
health_manager = HealthManager(
    performance_threshold=0.05,
    time_threshold_days=30
)

# Set baseline
health_manager.set_baseline(best_model, test_metrics)

# Check health later
health_report = health_manager.check_health(best_model, current_metrics)
```

---

## Feature Comparison

| Feature | Old Version | New Version |
|---------|------------|-------------|
| Training | ✅ train() | ✅ Trainer class |
| Testing | ✅ test() | ✅ Tester class |
| Model Management | ❌ | ✅ ModelsManager |
| Feature Generation | ✅ Basic | ✅ Multiple methods |
| Reports | ❌ | ✅ ReportManager |
| Visualizations | ❌ | ✅ Comprehensive |
| Health Monitoring | ❌ | ✅ HealthManager |
| Model Versioning | ❌ | ✅ Timestamps |
| SMOTE | ✅ | ✅ Automatic |
| Threshold Optimization | ✅ | ✅ Automatic |
| Progress Tracking | ✅ | ✅ Enhanced |

---

## Backward Compatibility

The old functional API still works:

```python
from src.model_training import train, test

# Old code still works
models, scaler, results, best = train(df_train)
test_results = test(models, scaler, df_test)
```

**But we recommend migrating to the new classes for:**
- Better organization
- More features
- Easier maintenance
- Production readiness

---

## Common Migration Patterns

### Pattern 1: Simple Training

**Old:**
```python
models, scaler, results, best = train(df)
```

**New:**
```python
models_manager = ModelsManager()
trainer = Trainer()
models, scaler, results, best = trainer.train(
    models_manager.create_models(), X, y
)
```

### Pattern 2: Training + Testing

**Old:**
```python
models, scaler, results, best = train(df_train)
test_results = test(models, scaler, df_test)
```

**New:**
```python
# Train
trainer = Trainer()
models, scaler, results, best = trainer.train(models, X_train, y_train)

# Test
tester = Tester(scaler=scaler)
test_results = tester.test(models, X_test, y_test)
```

### Pattern 3: Training + Saving

**Old:**
```python
models, scaler, results, best = train(df)
# Models saved automatically
```

**New:**
```python
# Train
trainer = Trainer()
models, scaler, results, best = trainer.train(models, X_train, y_train)

# Save explicitly
models_manager = ModelsManager()
models_manager.save_models(models, scaler)
```

### Pattern 4: Complete Pipeline

**New only:**
```python
# 1. Features
fg = FeaturesGenerator()
df_features = fg.generate_features(df, method='classical')
df_target = fg.create_target(df_features, target_bars=15, target_pct=3.0)

# 2. Train
models_manager = ModelsManager()
trainer = Trainer()
models, scaler, results, best = trainer.train(
    models_manager.create_models(), X_train, y_train
)

# 3. Test
tester = Tester(scaler=scaler)
test_results = tester.test(models, X_test, y_test)

# 4. Report
report_manager = ReportManager()
report_manager.export_full_report(results, test_results, y_test)

# 5. Save
models_manager.save_models(models, scaler)

# 6. Monitor
health_manager = HealthManager()
health_manager.set_baseline(best, test_results[best]['metrics'])
```

---

## Benefits of Migration

### 1. **Better Organization**
- Clear separation of concerns
- Each class has one job
- Easy to understand

### 2. **More Features**
- Comprehensive reports with visualizations
- Health monitoring
- Model versioning
- Better diagnostics

### 3. **Easier Maintenance**
- Modify one class without affecting others
- Add new features easily
- Better testing

### 4. **Production Ready**
- Health monitoring for production
- Model versioning for rollback
- Comprehensive logging
- Performance tracking

### 5. **Better Documentation**
- Clear class interfaces
- Comprehensive examples
- Detailed guides

---

## Troubleshooting

### Issue: Import errors

**Solution:** Make sure you're using the correct imports:
```python
# New imports
from src.ModelsManager import ModelsManager
from src.FeaturesGeneratorNew import FeaturesGenerator  # Note: New
from src.Trainer import Trainer
from src.Tester import Tester
```

### Issue: Old code not working

**Solution:** Old functional API still works:
```python
from src.model_training import train, test
# This still works
```

### Issue: Can't find examples

**Solution:** Examples moved to `examples/` folder:
```bash
python examples/example_complete_workflow.py
```

### Issue: Missing reports

**Solution:** Use ReportManager:
```python
from src.ReportManager import ReportManager
report_manager = ReportManager()
report_manager.export_full_report(train_results, test_results, y_test)
```

---

## Migration Checklist

- [ ] Update imports to use new classes
- [ ] Replace `prepare_data()` with `FeaturesGenerator`
- [ ] Replace `train()` with `Trainer` class
- [ ] Replace `test()` with `Tester` class
- [ ] Add `ReportManager` for reports
- [ ] Add `HealthManager` for monitoring
- [ ] Use `ModelsManager` for saving/loading
- [ ] Move custom code to `examples/` folder
- [ ] Update documentation
- [ ] Test new code
- [ ] Remove old code references

---

## Getting Help

### Documentation

- `ARCHITECTURE_SUMMARY.md` - System overview
- `docs/REFACTORED_ARCHITECTURE.md` - Detailed architecture guide
- `examples/README.md` - Examples guide
- `old/README.md` - Old files reference

### Examples

- `examples/example_complete_workflow.py` - Full pipeline
- `examples/example_refactored_workflow.py` - Architecture demo
- See `examples/` folder for more

### Quick Start

```bash
# Run main script
python run_me.py

# Run complete example
python examples/example_complete_workflow.py

# Check reports
ls reports/
```

---

## Summary

✅ **Main script updated** - `run_me.py` uses new architecture  
✅ **Examples organized** - All in `examples/` folder  
✅ **Old files moved** - To `old/` folder for reference  
✅ **New features added** - Reports, health monitoring, versioning  
✅ **Backward compatible** - Old API still works  
✅ **Well documented** - Comprehensive guides  

**Recommendation:** Migrate to new architecture for better features and maintainability.

**Next steps:**
1. Run `python run_me.py` to see new structure
2. Review reports in `reports/` folder
3. Explore examples in `examples/` folder
4. Migrate your custom code gradually

🚀 **Happy coding with the new architecture!**
