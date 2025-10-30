# Refactored Architecture Guide

## Overview

The codebase has been refactored into a clean, object-oriented architecture with separate classes for each responsibility. This improves maintainability, testability, and reusability.

## Architecture Components

### 1. **ModelsManager** (`src/ModelsManager.py`)

Handles all model-related operations: creation, loading, and saving.

**Responsibilities:**
- Create fresh model instances with configurations
- Save trained models to disk with timestamps
- Load pretrained models
- Manage model enable/disable configuration
- List available model versions

**Key Methods:**
```python
models_manager = ModelsManager(models_dir='models')

# Create models
models = models_manager.create_models(enabled_only=True)

# Save models
models_manager.save_models(models, scaler, suffix='20251030_120000')

# Load models
models, scaler, metadata = models_manager.load_models(suffix='latest')

# Configure models
models_manager.enable_model('xgboost', enabled=True)
models_manager.print_config()
```

**Features:**
- 10 pre-configured ML models (LogisticRegression, RandomForest, XGBoost, LightGBM, etc.)
- Easy enable/disable of models
- Automatic timestamped saves
- Version management
- Metadata tracking

---

### 2. **FeaturesGenerator** (`src/FeaturesGeneratorNew.py`)

Unified feature generation and target creation.

**Responsibilities:**
- Generate features using different methods
- Create target variables
- Handle data preprocessing

**Key Methods:**
```python
fg = FeaturesGenerator()

# Generate features
df_features = fg.generate_features(df, method='classical')
df_features = fg.generate_features(df, method='crypto', price_change_threshold=0.02)

# Create target
df_with_target = fg.create_target(df, target_bars=15, target_pct=3.0, method='binary')
```

**Feature Methods:**
- `'classical'` - Traditional technical indicators (SMA, RSI, Stochastic, Bollinger)
- `'crypto'` - Comprehensive crypto features (150+ indicators)
- `'otus'` - OTUS-style features (backward compatibility)

**Target Methods:**
- `'classification'` - Three classes: 1 (up), 0 (neutral), -1 (down)
- `'binary'` - Two classes: 1 (up), 0 (not up)
- `'regression'` - Continuous target (actual percentage change)

---

### 3. **Trainer** (`src/Trainer.py`)

Handles training of multiple ML models with automatic optimizations.

**Responsibilities:**
- Train multiple models with progress tracking
- Apply SMOTE for imbalanced data
- Optimize probability thresholds
- Scale features automatically
- Track training metrics

**Key Methods:**
```python
trainer = Trainer(use_smote=True, optimize_threshold=True, use_scaler=True)

# Train models
trained_models, scaler, results, best_model_name = trainer.train(
    models=models,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val
)

# Print results
trainer.print_results()

# Get best model
best_name, best_model = trainer.get_best_model()
```

**Features:**
- Automatic SMOTE application for imbalanced data
- Threshold optimization for binary classification
- Progress bar with tqdm
- Training time tracking
- Validation set support

---

### 4. **Tester** (`src/Tester.py`)

Tests trained models on test data.

**Responsibilities:**
- Test multiple models
- Calculate comprehensive metrics
- Generate detailed reports
- Compare model performance

**Key Methods:**
```python
tester = Tester(scaler=scaler)

# Test models
test_results = tester.test(
    models=trained_models,
    X_test=X_test,
    y_test=y_test,
    optimal_thresholds=optimal_thresholds
)

# Print results
tester.print_results()

# Detailed report
tester.print_detailed_report('xgboost', y_test, target_names=['No Rise', 'Rise'])

# Compare models
comparison = tester.compare_models(metric='f1')
```

**Features:**
- Automatic feature scaling
- Optimal threshold application
- Classification reports
- Confusion matrices
- Model comparison

---

### 5. **HealthManager** (`src/HealthManager.py`)

Monitors model health and determines retraining needs.

**Responsibilities:**
- Set baseline performance metrics
- Monitor performance degradation
- Track model age
- Recommend retraining
- Prioritize models for retraining

**Key Methods:**
```python
health_manager = HealthManager(
    performance_threshold=0.05,  # 5% degradation
    time_threshold_days=30       # 30 days max age
)

# Set baseline
health_manager.set_baseline(
    model_name='xgboost',
    metrics={'accuracy': 0.85, 'f1': 0.82},
    timestamp=datetime.now()
)

# Check health
health_report = health_manager.check_health(
    model_name='xgboost',
    current_metrics={'accuracy': 0.80, 'f1': 0.78}
)

health_manager.print_health_report(health_report)

# Get retraining priority
priority = health_manager.get_retraining_priority()
```

**Features:**
- Performance degradation detection
- Time-based retraining recommendations
- Online monitoring support
- Health history tracking
- Urgency scoring
- Export health reports

---

## Complete Workflow Example

```python
from src.ModelsManager import ModelsManager
from src.FeaturesGeneratorNew import FeaturesGenerator
from src.Trainer import Trainer
from src.Tester import Tester
from src.HealthManager import HealthManager

# 1. Load data
df_train = pd.read_csv("data/hour/btc.csv")
df_test = pd.read_csv("data/hour/btc_2025.csv")

# 2. Generate features
fg = FeaturesGenerator()
df_train_features = fg.generate_features(df_train, method='classical')
df_test_features = fg.generate_features(df_test, method='classical')

# 3. Create target
df_train_target = fg.create_target(df_train_features, target_bars=15, target_pct=3.0)
df_test_target = fg.create_target(df_test_features, target_bars=15, target_pct=3.0)

# 4. Prepare data
X_train, y_train = df_train_target[feature_cols], df_train_target['target']
X_test, y_test = df_test_target[feature_cols], df_test_target['target']

# 5. Create models
models_manager = ModelsManager()
models = models_manager.create_models()

# 6. Train models
trainer = Trainer(use_smote=True, optimize_threshold=True)
trained_models, scaler, results, best_name = trainer.train(models, X_train, y_train)

# 7. Save models
models_manager.save_models(trained_models, scaler)

# 8. Test models
tester = Tester(scaler=scaler)
test_results = tester.test(trained_models, X_test, y_test)

# 9. Monitor health
health_manager = HealthManager()
health_manager.set_baseline(best_name, test_results[best_name]['metrics'])
```

---

## Benefits of Refactored Architecture

### 1. **Separation of Concerns**
Each class has a single, well-defined responsibility:
- ModelsManager → Model lifecycle
- FeaturesGenerator → Feature engineering
- Trainer → Training logic
- Tester → Testing logic
- HealthManager → Health monitoring

### 2. **Improved Testability**
Each class can be tested independently with unit tests.

### 3. **Better Reusability**
Classes can be used in different projects or contexts:
```python
# Use just the trainer in another project
from src.Trainer import Trainer
trainer = Trainer()
```

### 4. **Easier Maintenance**
Changes are isolated to specific classes:
- Add new model → Modify ModelsManager
- Add new features → Modify FeaturesGenerator
- Change training logic → Modify Trainer

### 5. **Cleaner Code**
- No more 1000+ line files
- Clear interfaces
- Self-documenting code
- Consistent patterns

### 6. **Flexibility**
Easy to swap implementations:
```python
# Use different feature methods
fg.generate_features(df, method='classical')
fg.generate_features(df, method='crypto')

# Use different target methods
fg.create_target(df, method='binary')
fg.create_target(df, method='classification')
```

---

## Migration from Old Code

### Old Way:
```python
from src.model_training import train, test
from src.data_preparation import prepare_data

X, y = prepare_data(df)
models, scaler, results, best = train(X, y)
test_results = test(models, scaler, X_test, y_test)
```

### New Way:
```python
from src.ModelsManager import ModelsManager
from src.FeaturesGeneratorNew import FeaturesGenerator
from src.Trainer import Trainer
from src.Tester import Tester

# More explicit and flexible
fg = FeaturesGenerator()
df_features = fg.generate_features(df, method='classical')
df_target = fg.create_target(df_features, target_bars=15, target_pct=3.0)

models_manager = ModelsManager()
models = models_manager.create_models()

trainer = Trainer(use_smote=True)
trained_models, scaler, results, best = trainer.train(models, X_train, y_train)

tester = Tester(scaler=scaler)
test_results = tester.test(trained_models, X_test, y_test)
```

---

## Configuration

### Model Configuration (ModelsManager)

Enable/disable models in `ModelsManager.__init__()`:

```python
self.model_config = {
    'logistic_regression': {'enabled': True, ...},
    'xgboost': {'enabled': True, ...},
    'knn': {'enabled': False, ...},  # Disabled
}
```

Or programmatically:
```python
models_manager.enable_model('knn', enabled=False)
```

### Training Configuration (Trainer)

```python
trainer = Trainer(
    use_smote=True,           # Apply SMOTE for imbalanced data
    optimize_threshold=True,  # Optimize probability threshold
    use_scaler=True          # Scale features
)
```

### Health Monitoring Configuration (HealthManager)

```python
health_manager = HealthManager(
    performance_threshold=0.05,  # 5% max degradation
    time_threshold_days=30       # Retrain after 30 days
)
```

---

## File Structure

```
ml_predict_15/
├── src/
│   ├── ModelsManager.py          # Model lifecycle management
│   ├── FeaturesGeneratorNew.py   # Feature generation (new)
│   ├── FeaturesGenerator.py      # Feature generation (old, kept for compatibility)
│   ├── Trainer.py                # Training logic
│   ├── Tester.py                 # Testing logic
│   ├── HealthManager.py          # Health monitoring
│   ├── crypto_features.py        # Crypto feature engineering
│   └── ...
├── models/                        # Saved models directory
├── example_refactored_workflow.py # Complete example
└── docs/
    └── REFACTORED_ARCHITECTURE.md # This file
```

---

## Best Practices

### 1. Always Use Version Control
```python
# Save with timestamp
models_manager.save_models(models, scaler, suffix='20251030_120000')

# Load specific version
models_manager.load_models(suffix='20251030_120000')
```

### 2. Monitor Model Health
```python
# Set baseline after training
health_manager.set_baseline(model_name, test_metrics)

# Check regularly (e.g., daily)
health_report = health_manager.check_health(model_name, current_metrics)

# Retrain if needed
if health_report['needs_retraining']:
    # Retrain model
    pass
```

### 3. Use Validation Sets
```python
# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Train with validation
trainer.train(models, X_train, y_train, X_val, y_val)
```

### 4. Test on Separate Data
```python
# Never test on training data
tester.test(models, X_test, y_test)  # Use separate test set
```

### 5. Track Experiments
```python
# Save metadata with models
models_manager.save_models(
    models, 
    scaler, 
    suffix=f"experiment_{experiment_id}"
)
```

---

## Troubleshooting

### Issue: Models not loading
**Solution:** Check that the suffix matches saved files
```python
versions = models_manager.list_saved_models()
print(versions)  # See available versions
```

### Issue: SMOTE not working
**Solution:** Install imbalanced-learn
```bash
pip install imbalanced-learn
```

### Issue: Feature mismatch
**Solution:** Ensure same feature generation method for train/test
```python
# Use same method
df_train = fg.generate_features(df_train, method='classical')
df_test = fg.generate_features(df_test, method='classical')
```

---

## Summary

The refactored architecture provides:

✅ **Clean separation of concerns**  
✅ **Easy to test and maintain**  
✅ **Flexible and reusable components**  
✅ **Model health monitoring**  
✅ **Version control for models**  
✅ **Comprehensive documentation**  

**Next Steps:**
1. Run `example_refactored_workflow.py` to see it in action
2. Integrate with your existing pipeline
3. Set up regular health monitoring
4. Enjoy cleaner, more maintainable code! 🚀
