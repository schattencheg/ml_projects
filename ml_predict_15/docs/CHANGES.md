# Recent Changes: Model Loading for Backtesting

## Summary

Updated the backtesting workflow to load pre-trained models from the `models/` folder instead of training them every time. This makes backtesting much faster and more efficient.

## Changes Made

### New Files

1. **`src/model_loader.py`** - Utility module for loading and saving models
   - `load_model()` - Load a single model
   - `load_scaler()` - Load the fitted scaler
   - `load_all_models()` - Load all available models
   - `list_available_models()` - List available model names
   - `save_model()` - Save a single model
   - `save_scaler()` - Save the scaler
   - `save_all_models()` - Save all models and scaler

2. **`train_and_save_models.py`** - Dedicated script for training and saving models
   - Trains all 13 ML models
   - Saves all models and scaler to `models/` folder
   - Displays performance summary
   - Should be run once before backtesting

### Modified Files

1. **`run_me.py`**
   - Added import: `from src.model_loader import save_all_models`
   - Now saves all models and scaler (not just the best model)
   - Scaler is now saved for reuse in backtesting

2. **`backtest_quick_start.py`**
   - Removed training step
   - Now loads pre-trained models from `models/` folder
   - Checks if models exist before running
   - Much faster execution

3. **`backtest_example.py`**
   - Removed training step
   - Now loads all pre-trained models
   - Backtests each loaded model
   - Compares all models without retraining

4. **`README.md`**
   - Updated Quick Start section
   - Added step to train and save models first
   - Updated workflow section
   - Clarified that training is a one-time step

5. **`BACKTEST_GUIDE.md`**
   - Updated examples to use model loading
   - Added instructions to train models first
   - Updated code examples

## New Workflow

### Old Workflow (Slow)
```
1. Run backtest script
2. Script trains all models (takes time)
3. Script runs backtest
4. Repeat for each backtest run
```

### New Workflow (Fast)
```
1. Run train_and_save_models.py (once)
2. Models and scaler saved to models/
3. Run backtest scripts (loads models instantly)
4. Run as many backtests as needed (no retraining)
```

## Benefits

✅ **Faster Backtesting**: No need to retrain models every time  
✅ **Consistent Results**: Same models used across all backtests  
✅ **Easy Comparison**: Load and compare all models quickly  
✅ **Better Workflow**: Separate training from backtesting  
✅ **Disk Space Efficient**: Models saved once, reused many times  

## Usage

### First Time Setup

```bash
# Train and save all models (run once)
python train_and_save_models.py
```

This creates the `models/` folder with:
- `scaler.joblib` - Fitted StandardScaler
- `logistic_regression.joblib` - Logistic Regression model
- `random_forest.joblib` - Random Forest model
- `xgboost.joblib` - XGBoost model
- ... (all 13 models)

### Running Backtests

```bash
# Quick start example (loads models)
python backtest_quick_start.py

# Advanced examples (loads all models)
python backtest_example.py
```

### Loading Models in Your Own Scripts

```python
from src.model_loader import load_all_models, load_scaler, list_available_models

# List available models
models_list = list_available_models()
print(f"Available models: {models_list}")

# Load all models
models = load_all_models()
scaler = load_scaler()

# Use a specific model
model = models['random_forest']

# Or load a single model
from src.model_loader import load_model
model = load_model('random_forest')
```

## Model Storage

Models are stored in the `models/` folder as `.joblib` files:

```
models/
├── scaler.joblib                    # Fitted StandardScaler
├── logistic_regression.joblib       # Logistic Regression
├── ridge_classifier.joblib          # Ridge Classifier
├── naive_bayes.joblib              # Naive Bayes
├── knn.joblib                      # K-Nearest Neighbors
├── decision_tree.joblib            # Decision Tree
├── random_forest.joblib            # Random Forest
├── gradient_boosting.joblib        # Gradient Boosting
├── svm.joblib                      # Support Vector Machine
├── xgboost.joblib                  # XGBoost
├── lightgbm.joblib                 # LightGBM
├── lstm.joblib                     # LSTM (if TensorFlow available)
├── cnn.joblib                      # CNN (if TensorFlow available)
└── lstm_cnn_hybrid.joblib          # LSTM-CNN Hybrid (if TensorFlow available)
```

## Backward Compatibility

The old workflow still works:
- `run_me.py` can still be run directly
- It will train models and save them
- But for backtesting, use the new workflow for better efficiency

## Error Handling

If models are not found, the backtest scripts will:
1. Display a clear error message
2. Instruct you to run `train_and_save_models.py` first
3. Exit gracefully

Example:
```
❌ No trained models found!
Please run 'python train_and_save_models.py' first to train models.
```

## Performance Comparison

### Old Approach
- Training time: ~5-10 minutes (depending on data size)
- Backtesting time: ~10 seconds
- **Total time per backtest run: ~5-10 minutes**

### New Approach
- Training time (one-time): ~5-10 minutes
- Model loading time: ~1 second
- Backtesting time: ~10 seconds
- **Total time per backtest run: ~11 seconds**

**Speed improvement: ~30x faster for subsequent runs!**

## Notes

- Models are saved using `joblib` for efficient serialization
- The scaler must be saved and loaded to ensure consistent feature scaling
- All models are loaded into memory when using `load_all_models()`
- For large models, consider loading only the specific model you need
- Models are compatible across Python sessions and environments (with same dependencies)

## Future Enhancements

Potential future improvements:
- Model versioning (track different training runs)
- Model metadata (training date, data used, hyperparameters)
- Automatic model retraining on new data
- Model performance tracking over time
- Integration with MLflow for experiment tracking
