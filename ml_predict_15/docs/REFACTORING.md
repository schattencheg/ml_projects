# Code Refactoring Documentation

## Overview

The `run_me.py` file has been refactored into a modular structure for better code organization, maintainability, and reusability.

## Changes Made

### 1. Created Modular Files

The monolithic `run_me.py` (948 lines) has been split into focused modules:

#### **src/data_preparation.py**
- `prepare_data()` - Prepare OHLCV data with features and targets
- `fit_scaler()` - Fit StandardScaler on training data

#### **src/neural_models.py**
- `create_sequences()` - Create sequences for LSTM/CNN models
- `create_lstm_model()` - Build LSTM architecture
- `create_cnn_model()` - Build CNN architecture
- `create_hybrid_lstm_cnn_model()` - Build hybrid LSTM-CNN architecture
- `KerasClassifierWrapper` - Wrapper class for Keras models

#### **src/model_training.py**
- `get_model_configs()` - Get all model configurations
- `add_neural_network_models()` - Add neural network models
- `train_and_evaluate_model()` - Train and evaluate a single model
- `train()` - Main training function
- `test()` - Test models on new data

#### **src/visualization.py**
- `print_model_summary()` - Print model performance summary
- `create_visualizations()` - Create and save performance plots

#### **src/model_loader.py** (Already existed)
- Model loading and saving utilities

#### **src/MLBacktester.py** (Already existed)
- Backtesting engine with trailing stop loss

### 2. New Simplified run_me.py

The new `run_me.py` is now only **68 lines** (down from 948 lines):

```python
"""
Main Training and Testing Script

This script trains ML models on historical data and tests them on future data.
All helper functions have been moved to modular files in the src/ directory.
"""

import pandas as pd
from src.model_training import train, test
from src.visualization import print_model_summary

# Data paths
PATH_TRAIN = "data/btc_2022.csv"
PATH_TEST = "data/btc_2023.csv"

def main():
    """Main execution function."""
    # Load data
    df_train = pd.read_csv(PATH_TRAIN)
    df_test = pd.read_csv(PATH_TEST)
    
    # Train models
    models, scaler, train_results, best_model_name = train(df_train)
    print_model_summary(train_results)
    
    # Test models
    test_metrics = test(models, scaler, df_test)
    
    # Print summary
    # ... (summary code)

if __name__ == "__main__":
    main()
```

### 3. Moved Documentation to docs/

All markdown documentation has been moved to the `docs/` folder:

- `README.md` → `docs/README.md` (detailed documentation)
- `README_MODELS.md` → `docs/README_MODELS.md`
- `BACKTEST_GUIDE.md` → `docs/BACKTEST_GUIDE.md`
- `CHANGES.md` → `docs/CHANGES.md`
- New: `docs/REFACTORING.md` (this file)

A new simplified `README.md` was created in the root directory for quick reference.

### 4. Updated Imports

All files that previously imported from `run_me.py` have been updated:

**Before:**
```python
from run_me import train, prepare_data
```

**After:**
```python
from src.model_training import train
from src.data_preparation import prepare_data
```

Updated files:
- `train_and_save_models.py`
- `backtest_quick_start.py`
- `backtest_example.py`

### 5. Preserved Old File

The original `run_me.py` has been renamed to `run_me_old.py` for reference.

## Benefits

### ✅ Better Organization
- Each module has a single, clear responsibility
- Related functions are grouped together
- Easier to find specific functionality

### ✅ Improved Maintainability
- Changes to one component don't affect others
- Easier to debug and test individual modules
- Clear separation of concerns

### ✅ Enhanced Reusability
- Modules can be imported independently
- Functions can be used in different contexts
- Easier to create new scripts using existing components

### ✅ Cleaner Codebase
- Main script is now simple and readable
- Documentation is organized in dedicated folder
- Reduced code duplication

### ✅ Better Testing
- Individual modules can be tested in isolation
- Easier to write unit tests
- More focused test coverage

## New Project Structure

```
ml_predict_15/
├── src/                          # Source code modules
│   ├── data_preparation.py       # Data prep functions (73 lines)
│   ├── model_training.py         # Training logic (428 lines)
│   ├── neural_models.py          # Neural networks (289 lines)
│   ├── visualization.py          # Plotting (119 lines)
│   ├── MLBacktester.py           # Backtesting (700+ lines)
│   ├── model_loader.py           # Model I/O (180+ lines)
│   └── FeaturesGenerator.py      # Features (existing)
├── docs/                         # Documentation
│   ├── README.md                 # Detailed docs
│   ├── BACKTEST_GUIDE.md         # Backtesting guide
│   ├── README_MODELS.md          # Model descriptions
│   ├── CHANGES.md                # Change log
│   └── REFACTORING.md            # This file
├── data/                         # Data files
├── models/                       # Saved models
├── plots/                        # Visualizations
├── run_me.py                     # Main script (68 lines)
├── run_me_old.py                 # Original file (948 lines)
├── train_and_save_models.py     # Training script
├── backtest_quick_start.py      # Quick backtest
└── backtest_example.py          # Advanced examples
```

## Module Responsibilities

### data_preparation.py
**Purpose:** Data preprocessing and feature engineering  
**Functions:**
- Prepare raw OHLCV data
- Generate features using FeaturesGenerator
- Create target labels
- Fit scalers

### model_training.py
**Purpose:** Model configuration, training, and evaluation  
**Functions:**
- Define model configurations
- Train models
- Evaluate performance
- Save models
- Test on new data

### neural_models.py
**Purpose:** Neural network architectures  
**Functions:**
- Create LSTM models
- Create CNN models
- Create hybrid models
- Sequence generation
- Keras wrapper for sklearn compatibility

### visualization.py
**Purpose:** Plotting and reporting  
**Functions:**
- Print performance summaries
- Create comparison charts
- Generate prediction visualizations

## Usage Examples

### Import Individual Modules

```python
# Data preparation
from src.data_preparation import prepare_data, fit_scaler

# Model training
from src.model_training import train, test, get_model_configs

# Neural networks
from src.neural_models import create_lstm_model, create_cnn_model

# Visualization
from src.visualization import print_model_summary, create_visualizations

# Model loading
from src.model_loader import load_all_models, load_scaler

# Backtesting
from src.MLBacktester import MLBacktester
```

### Create Custom Training Script

```python
from src.data_preparation import prepare_data, fit_scaler
from src.model_training import get_model_configs, train_and_evaluate_model
import pandas as pd

# Load data
df = pd.read_csv("data/btc_2022.csv")

# Prepare data
X, y = prepare_data(df)

# Get models
models = get_model_configs()

# Train specific model
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
scaler, X_train_scaled = fit_scaler(X_train)
X_val_scaled = scaler.transform(X_val)

# Train Random Forest
rf_model = models['random_forest'][0]
results = train_and_evaluate_model(
    rf_model, 'random_forest',
    X_train_scaled, y_train,
    X_val_scaled, y_val
)
```

## Migration Guide

If you have existing code that imports from `run_me.py`, update as follows:

### Old Code
```python
from run_me import (
    prepare_data,
    fit_scaler,
    train,
    test,
    create_lstm_model,
    print_model_summary
)
```

### New Code
```python
from src.data_preparation import prepare_data, fit_scaler
from src.model_training import train, test
from src.neural_models import create_lstm_model
from src.visualization import print_model_summary
```

## Backward Compatibility

The original `run_me.py` is preserved as `run_me_old.py` for reference. However, it's recommended to use the new modular structure for all new development.

## Testing

All existing functionality has been preserved. The refactoring only changes the code organization, not the behavior.

To verify:
1. Run `python run_me.py` - Should work exactly as before
2. Run `python train_and_save_models.py` - Should train and save models
3. Run `python backtest_quick_start.py` - Should load models and backtest

## Future Enhancements

The modular structure makes it easier to:
- Add new models (just update `model_training.py`)
- Add new features (just update `data_preparation.py`)
- Add new visualizations (just update `visualization.py`)
- Create new neural architectures (just update `neural_models.py`)
- Write unit tests for each module
- Create additional training scripts
- Implement cross-validation
- Add hyperparameter optimization

## Summary

**Before:**
- 1 monolithic file (948 lines)
- Hard to navigate and maintain
- Difficult to reuse components

**After:**
- 4 focused modules + existing utilities
- Clear separation of concerns
- Easy to import and reuse
- Better organized documentation
- Simpler main script (68 lines)

The refactoring improves code quality while maintaining all existing functionality.
