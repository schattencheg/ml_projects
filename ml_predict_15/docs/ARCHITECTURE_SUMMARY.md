# ML Prediction System - Architecture Summary

## Overview

Professional ML prediction system with clean class-based architecture, comprehensive reporting, and health monitoring.

## Core Classes

### 1. **ModelsManager** (`src/ModelsManager.py`)
Manages ML model lifecycle.

**Features:**
- Create 10 pre-configured models
- Save/load with timestamps
- Version management
- Enable/disable models

**Usage:**
```python
models_manager = ModelsManager()
models = models_manager.create_models()
models_manager.save_models(models, scaler)
```

---

### 2. **FeaturesGenerator** (`src/FeaturesGeneratorNew.py`)
Unified feature generation and target creation.

**Features:**
- Classical features (SMA, RSI, Bollinger, Stochastic)
- Crypto features (150+ indicators)
- Flexible target creation (binary, classification, regression)

**Usage:**
```python
fg = FeaturesGenerator()
df_features = fg.generate_features(df, method='classical')
df_target = fg.create_target(df, target_bars=15, target_pct=3.0)
```

---

### 3. **Trainer** (`src/Trainer.py`)
Trains multiple ML models with optimizations.

**Features:**
- Automatic SMOTE for imbalanced data
- Threshold optimization
- Progress tracking
- Feature scaling

**Usage:**
```python
trainer = Trainer(use_smote=True, optimize_threshold=True)
models, scaler, results, best = trainer.train(models, X_train, y_train)
```

---

### 4. **Tester** (`src/Tester.py`)
Tests trained models on test data.

**Features:**
- Comprehensive metrics
- Classification reports
- Confusion matrices
- Model comparison

**Usage:**
```python
tester = Tester(scaler=scaler)
test_results = tester.test(models, X_test, y_test)
```

---

### 5. **HealthManager** (`src/HealthManager.py`)
Monitors model health and recommends retraining.

**Features:**
- Performance degradation detection
- Age tracking
- Retraining recommendations
- Health history

**Usage:**
```python
health_manager = HealthManager(performance_threshold=0.05, time_threshold_days=30)
health_manager.set_baseline(model_name, metrics)
health_report = health_manager.check_health(model_name, current_metrics)
```

---

### 6. **ReportManager** (`src/ReportManager.py`) ⭐ NEW
Creates comprehensive reports with visualizations.

**Features:**
- Training reports (CSV + PNG)
- Test reports (CSV + PNG)
- Comparison reports (CSV + PNG)
- Overfitting analysis
- Confusion matrices
- Metrics heatmaps

**Usage:**
```python
report_manager = ReportManager(output_dir='reports')
report_manager.export_full_report(train_results, test_results, y_test)
```

**Output:**
- `training_report.csv` - Training metrics table
- `training_report.png` - Training visualizations (4 subplots)
- `test_report.csv` - Test metrics table
- `test_report.png` - Test visualizations (4 subplots)
- `comparison_report.csv` - Train vs test comparison
- `comparison_report.png` - Comparison visualizations (4 subplots)

---

## Complete Workflow

```python
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
X_train, y_train = df_train_target[features], df_train_target['target']
X_test, y_test = df_test_target[features], df_test_target['target']

# 5. Create models
models_manager = ModelsManager()
models = models_manager.create_models()

# 6. Train models
trainer = Trainer(use_smote=True, optimize_threshold=True)
trained_models, scaler, train_results, best = trainer.train(models, X_train, y_train)

# 7. Test models
tester = Tester(scaler=scaler)
test_results = tester.test(trained_models, X_test, y_test)

# 8. Generate reports ⭐ NEW
report_manager = ReportManager()
report_manager.export_full_report(train_results, test_results, y_test)

# 9. Save models
models_manager.save_models(trained_models, scaler)

# 10. Monitor health
health_manager = HealthManager()
health_manager.set_baseline(best, test_results[best]['metrics'])
```

---

## Examples Directory

All examples moved to `examples/` folder. Run from project root:

```bash
python examples/example_complete_workflow.py
```

**Available Examples:**
1. `example_complete_workflow.py` - Full pipeline with reports ⭐
2. `example_refactored_workflow.py` - Class architecture demo
3. `example_crypto_features.py` - Crypto features (150+)
4. `example_model_config.py` - Model configuration
5. `backtest_quick_start.py` - Basic backtesting
6. `backtest_example.py` - Advanced backtesting
7. `backtest_backtestingpy_example.py` - backtesting.py library
8. `backtest_backtrader_example.py` - Backtrader library
9. `backtest_compare_libraries.py` - Library comparison

See `examples/README.md` for details.

---

## Project Structure

```
ml_predict_15/
├── src/
│   ├── ModelsManager.py          # Model lifecycle
│   ├── FeaturesGeneratorNew.py   # Feature generation (new)
│   ├── Trainer.py                # Training logic
│   ├── Tester.py                 # Testing logic
│   ├── HealthManager.py          # Health monitoring
│   ├── ReportManager.py          # Report generation ⭐ NEW
│   ├── crypto_features.py        # Crypto features (150+)
│   ├── model_training.py         # Original training (backward compatible)
│   ├── data_preparation.py       # Data prep
│   └── ...
├── examples/                      # All examples ⭐ NEW
│   ├── README.md                 # Examples guide
│   ├── example_complete_workflow.py
│   ├── example_refactored_workflow.py
│   ├── example_crypto_features.py
│   └── ...
├── models/                        # Saved models
├── reports/                       # Generated reports ⭐ NEW
├── data/                          # Data files
├── docs/                          # Documentation
│   ├── REFACTORED_ARCHITECTURE.md
│   └── ...
├── requirements.txt
└── ARCHITECTURE_SUMMARY.md        # This file
```

---

## Key Features

### ✅ Clean Architecture
- Separation of concerns
- Single responsibility per class
- Easy to test and maintain

### ✅ Comprehensive Reporting ⭐ NEW
- Training reports with visualizations
- Test reports with confusion matrices
- Train vs test comparison
- Overfitting detection
- CSV + PNG output

### ✅ Health Monitoring
- Performance degradation detection
- Age-based retraining
- Urgency scoring
- Health history

### ✅ Production Ready
- Model versioning
- Automatic SMOTE
- Threshold optimization
- Progress tracking

### ✅ Flexible
- Multiple feature methods
- Multiple target types
- Enable/disable models
- Configurable parameters

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Complete Example
```bash
python examples/example_complete_workflow.py
```

### 3. Check Reports
```bash
# Reports saved in reports/ folder
ls reports/
```

### 4. Review Visualizations
Open PNG files in `reports/` folder to see:
- Training metrics comparison
- Test metrics comparison
- Train vs test comparison
- Confusion matrices
- Overfitting analysis

---

## Report Visualizations

### Training Report (4 subplots)
1. **Accuracy by Model** - Bar chart
2. **F1 Score by Model** - Bar chart
3. **Training Time** - Bar chart
4. **Metrics Heatmap** - All metrics

### Test Report (4 subplots)
1. **Accuracy by Model** - Bar chart
2. **F1 Score by Model** - Bar chart
3. **Metrics Comparison** - Grouped bar chart
4. **Confusion Matrix** - Best model

### Comparison Report (4 subplots)
1. **Train vs Test Accuracy** - Grouped bar chart
2. **Train vs Test F1** - Grouped bar chart
3. **Overfitting Score** - Bar chart with threshold
4. **Generalization Analysis** - Scatter plot

---

## Benefits

### 1. **Professional Reports**
- Publication-ready visualizations
- Comprehensive metrics
- Easy to share with stakeholders

### 2. **Better Decision Making**
- Identify overfitting
- Compare models easily
- Track performance over time

### 3. **Production Monitoring**
- Health checks
- Retraining alerts
- Performance tracking

### 4. **Clean Code**
- Maintainable
- Testable
- Reusable

---

## What's New

### ReportManager Class ⭐
- Comprehensive report generation
- Multiple visualization types
- CSV + PNG output
- Overfitting detection

### Examples Organization ⭐
- All examples in `examples/` folder
- Can be run from project root
- Comprehensive README
- Easy to navigate

### Complete Workflow Example ⭐
- End-to-end demonstration
- Includes report generation
- Shows all features
- Production-ready

---

## Next Steps

1. **Run the example:**
   ```bash
   python examples/example_complete_workflow.py
   ```

2. **Review reports:**
   - Check `reports/` folder
   - Open PNG files
   - Review CSV files

3. **Customize:**
   - Modify data paths
   - Change feature methods
   - Configure models
   - Adjust parameters

4. **Deploy:**
   - Monitor health
   - Retrain when needed
   - Use best model

---

## Documentation

- `docs/REFACTORED_ARCHITECTURE.md` - Complete architecture guide
- `REFACTORING_SUMMARY.md` - Quick reference
- `examples/README.md` - Examples guide
- `ARCHITECTURE_SUMMARY.md` - This file

---

## Summary

**Total Classes:** 6 (ModelsManager, FeaturesGenerator, Trainer, Tester, HealthManager, ReportManager)  
**Total Examples:** 9 (all in examples/ folder)  
**Total Code:** ~3,000 lines  
**Total Documentation:** ~2,000 lines  

**Key Additions:**
- ✅ ReportManager class with visualizations
- ✅ Examples organized in examples/ folder
- ✅ Complete workflow example
- ✅ Comprehensive documentation

**Ready for:**
- ✅ Development
- ✅ Testing
- ✅ Production
- ✅ Monitoring

🚀 **Professional ML prediction system ready to use!**
