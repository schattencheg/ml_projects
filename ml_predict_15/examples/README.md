# Examples Directory

This directory contains example scripts demonstrating various features of the ML prediction system.

## Running Examples

All examples should be run from the **project root directory**:

```bash
# From ml_predict_15/ directory
python examples/example_complete_workflow.py
python examples/example_crypto_features.py
```

## Available Examples

### 1. **example_complete_workflow.py** ⭐ RECOMMENDED
Complete end-to-end workflow demonstrating:
- Data loading
- Feature generation
- Target creation
- Model training
- Model testing
- **Report generation with visualizations**
- Health monitoring

**Run:**
```bash
python examples/example_complete_workflow.py
```

**Output:**
- Trained models in `models/`
- Comprehensive reports in `reports/`
- CSV files with metrics
- PNG files with visualizations

---

### 2. **example_refactored_workflow.py**
Demonstrates the refactored class-based architecture:
- ModelsManager
- FeaturesGenerator
- Trainer
- Tester
- HealthManager

**Run:**
```bash
python examples/example_refactored_workflow.py
```

---

### 3. **example_crypto_features.py**
Demonstrates cryptocurrency-specific feature engineering:
- 150+ technical indicators
- Feature importance analysis
- Correlation analysis
- Model training with crypto features

**Run:**
```bash
python examples/example_crypto_features.py
```

**Note:** Uses synthetic data by default. Update to use real data.

---

### 4. **example_model_config.py**
Shows how to configure which models to train:
- Enable/disable models
- View configuration
- Different configuration presets

**Run:**
```bash
python examples/example_model_config.py
```

---

## Backtesting Examples

### 5. **backtest_quick_start.py**
Quick introduction to backtesting with ML models.

**Run:**
```bash
python examples/backtest_quick_start.py
```

---

### 6. **backtest_example.py**
Advanced backtesting examples:
- Multiple strategies
- Parameter optimization
- Performance comparison

**Run:**
```bash
python examples/backtest_example.py
```

---

### 7. **backtest_backtestingpy_example.py**
Examples using the backtesting.py library:
- Fast vectorized backtesting
- Interactive visualizations
- Parameter optimization

**Run:**
```bash
python examples/backtest_backtestingpy_example.py
```

---

### 8. **backtest_backtrader_example.py**
Examples using the Backtrader library:
- Event-driven backtesting
- Realistic order execution
- Multiple strategies

**Run:**
```bash
python examples/backtest_backtrader_example.py
```

---

### 9. **backtest_compare_libraries.py**
Compares different backtesting libraries:
- MLBacktester vs backtesting.py vs Backtrader
- Performance comparison
- Feature comparison

**Run:**
```bash
python examples/backtest_compare_libraries.py
```

---

## Output Directories

Examples create the following directories:

- **`models/`** - Saved trained models
- **`reports/`** - Generated reports (CSV + PNG)
- **`visualizations/`** - Additional visualizations
- **`logs/`** - Execution logs (if enabled)

---

## Quick Start Guide

### For Beginners:

1. Start with **example_complete_workflow.py** to see the full pipeline
2. Review generated reports in `reports/` folder
3. Try **example_model_config.py** to customize models
4. Experiment with **example_crypto_features.py** for advanced features

### For Backtesting:

1. Start with **backtest_quick_start.py**
2. Try **backtest_example.py** for advanced strategies
3. Compare libraries with **backtest_compare_libraries.py**

---

## Customization

### Modify Data Paths

Edit the data paths in examples:
```python
# Change these lines
df_train = pd.read_csv("data/hour/btc.csv")
df_test = pd.read_csv("data/hour/btc_2025.csv")
```

### Change Feature Method

```python
# Classical features (fast)
df_features = fg.generate_features(df, method='classical')

# Crypto features (comprehensive, 150+)
df_features = fg.generate_features(df, method='crypto')
```

### Configure Models

```python
# Enable/disable models
models_manager.enable_model('xgboost', enabled=True)
models_manager.enable_model('knn', enabled=False)
```

### Adjust Training Parameters

```python
trainer = Trainer(
    use_smote=True,           # SMOTE for imbalanced data
    optimize_threshold=True,  # Optimize probability threshold
    use_scaler=True          # Scale features
)
```

---

## Troubleshooting

### Import Errors

If you get import errors, make sure you're running from the project root:
```bash
# Wrong (from examples/ directory)
cd examples
python example_complete_workflow.py  # ❌ Will fail

# Correct (from project root)
cd ..
python examples/example_complete_workflow.py  # ✅ Works
```

### Missing Data Files

If examples can't find data files:
1. Check that data files exist in `data/hour/` or `data/minute/`
2. Update paths in the example scripts
3. Or use synthetic data (see example_crypto_features.py)

### Memory Issues

If you run out of memory:
1. Reduce dataset size
2. Disable slow models (KNN, SVM)
3. Use fewer features
4. Process data in chunks

---

## Example Output

### Training Report
```
================================================================================
TRAINING MODELS
================================================================================

✓ Features scaled using StandardScaler
✓ Applying SMOTE (imbalance ratio: 5.23)

Training models: 100%|████████████████| 7/7 [00:45<00:00]

✓ logistic_regression: Train Acc=0.7234, F1=0.6187, Time=2.34s
✓ random_forest: Train Acc=0.7156, F1=0.6089, Time=15.67s
✓ xgboost: Train Acc=0.7087, F1=0.5976, Time=3.21s

================================================================================
TRAINING COMPLETE
================================================================================
Total training time: 45.67 seconds (0.76 minutes)
Best model: logistic_regression
```

### Test Report
```
================================================================================
TESTING MODELS
================================================================================

✓ logistic_regression: Acc=0.6987, F1=0.5834
✓ random_forest: Acc=0.6912, F1=0.5756
✓ xgboost: Acc=0.6845, F1=0.5689

================================================================================
TESTING COMPLETE
================================================================================
Best model on test set: logistic_regression
```

### Generated Files
```
reports/
├── ml_report_20251030_105623_training.csv
├── ml_report_20251030_105623_training.png
├── ml_report_20251030_105623_test.csv
├── ml_report_20251030_105623_test.png
├── ml_report_20251030_105623_comparison.csv
└── ml_report_20251030_105623_comparison.png
```

---

## Tips

1. **Start Simple**: Run example_complete_workflow.py first
2. **Review Reports**: Check CSV and PNG files in reports/
3. **Experiment**: Modify parameters and see results
4. **Monitor Health**: Use HealthManager for production
5. **Backtest**: Test strategies before live trading

---

## Support

For more information:
- See `docs/REFACTORED_ARCHITECTURE.md` for architecture details
- See `REFACTORING_SUMMARY.md` for quick reference
- Check individual example files for detailed comments

---

## Summary

| Example | Purpose | Output | Time |
|---------|---------|--------|------|
| example_complete_workflow.py | Full pipeline + reports | Models + Reports | ~1-2 min |
| example_refactored_workflow.py | Class architecture demo | Models | ~1-2 min |
| example_crypto_features.py | Crypto features | Features analysis | ~30 sec |
| example_model_config.py | Model configuration | Config display | <1 sec |
| backtest_quick_start.py | Basic backtesting | Backtest results | ~10 sec |
| backtest_example.py | Advanced backtesting | Multiple backtests | ~30 sec |
| backtest_backtestingpy_example.py | backtesting.py lib | Interactive plots | ~20 sec |
| backtest_backtrader_example.py | Backtrader lib | Backtest results | ~30 sec |
| backtest_compare_libraries.py | Library comparison | Comparison report | ~1 min |

**Recommended order:**
1. example_complete_workflow.py
2. example_model_config.py
3. backtest_quick_start.py
4. Explore others based on your needs

Happy coding! 🚀
