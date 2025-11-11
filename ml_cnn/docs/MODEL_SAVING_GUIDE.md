# Model Saving Guide

## Overview

The enhanced CNN optimizer now automatically saves the best model with all artifacts in a timestamped subfolder for easy deployment and sharing.

## Directory Structure

```
models/
└── YYYY_MM_DD__HH_MM_SS/          # Timestamped folder
    ├── btc_trend_cnn.h5            # Trained Keras model
    ├── btc_trend_cnn_architecture.json  # Model architecture
    ├── metadata.json               # Complete metadata
    ├── optimization_history.csv    # All trials and metrics
    ├── training_history.csv        # Training/validation metrics
    ├── optimization_results.png    # 6-panel optimization plot
    ├── training_history.png        # Training loss/accuracy plots
    ├── model_summary.txt           # Model architecture summary
    └── README.md                   # Complete documentation
```

## Usage

### Basic Usage

```python
from src.enhanced_cnn_optimizer import EnhancedCNNOptimizer

# Run optimization
optimizer = EnhancedCNNOptimizer(X_train, y_train, X_val, y_val, input_shape)
study = optimizer.optimize(n_trials=50)

# Save best model
save_dir = optimizer.save_best_model(study, model_name='my_cnn_model')
```

### Custom Parameters

```python
# Save with custom name and directory
save_dir = optimizer.save_best_model(
    study,
    model_name='btc_trend_predictor',
    base_dir='trained_models'
)
```

## What Gets Saved

### 1. Model Files

**`{model_name}.h5`** - Complete trained Keras model
- Ready to load and use for predictions
- Includes weights, architecture, and optimizer state

**`{model_name}_architecture.json`** - Model architecture only
- JSON format for inspection
- Can be used to rebuild model structure

### 2. Metadata

**`metadata.json`** - Complete metadata
```json
{
    "timestamp": "2025_11_10__16_52_33",
    "model_name": "btc_trend_cnn",
    "best_trial_number": 23,
    "best_recall": 0.8234,
    "architecture": "deeper",
    "hyperparameters": {
        "filters_1": 48,
        "filters_2": 96,
        "dense_units_1": 256,
        "dense_units_2": 128,
        "dropout_1": 0.35,
        "dropout_2": 0.45,
        "optimizer": "adam",
        "learning_rate": 0.0012,
        "batch_size": 32,
        "epochs": 60
    },
    "input_shape": [15, 126],
    "training_samples": 3500,
    "validation_samples": 750,
    "total_trials": 30,
    "optimization_metric": "recall"
}
```

### 3. History Files

**`optimization_history.csv`** - All optimization trials
- Columns: trial, architecture, accuracy, precision, recall, f1_score, params
- One row per trial
- Useful for analyzing optimization process

**`training_history.csv`** - Training metrics per epoch
- Columns: loss, val_loss, accuracy, val_accuracy
- One row per epoch
- Useful for analyzing training convergence

### 4. Visualizations

**`optimization_results.png`** - 6-panel optimization plot
- Panel 1: Recall progress over trials
- Panel 2: Best model metrics (bar chart)
- Panel 3: Architecture comparison
- Panel 4: Learning rate vs Recall
- Panel 5: Batch size impact
- Panel 6: Top 10 trials

**`training_history.png`** - Training plots
- Left: Training/validation loss
- Right: Training/validation accuracy

### 5. Documentation

**`model_summary.txt`** - Model architecture summary
- Layer-by-layer breakdown
- Parameter counts
- Output shapes

**`README.md`** - Complete documentation
- Model information
- Hyperparameters
- Usage examples
- Loading instructions

## Loading Saved Models

### Load Complete Model

```python
from tensorflow.keras.models import load_model

# Load model
model = load_model('models/2025_11_10__16_52_33/btc_trend_cnn.h5')

# Make predictions
predictions = model.predict(X_test)
```

### Load Metadata

```python
import json

# Load metadata
with open('models/2025_11_10__16_52_33/metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Best Recall: {metadata['best_recall']}")
print(f"Architecture: {metadata['architecture']}")
print(f"Hyperparameters: {metadata['hyperparameters']}")
```

### Load Optimization History

```python
import pandas as pd

# Load optimization history
history = pd.read_csv('models/2025_11_10__16_52_33/optimization_history.csv')

# Analyze trials
print(history.nlargest(10, 'recall'))
```

### Load Training History

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load training history
training = pd.read_csv('models/2025_11_10__16_52_33/training_history.csv')

# Plot training progress
plt.plot(training['loss'], label='Training Loss')
plt.plot(training['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
```

## Using in Another Project

### Step 1: Copy Model Directory

```bash
# Copy entire timestamped directory to another project
cp -r models/2025_11_10__16_52_33 /path/to/other/project/models/
```

### Step 2: Load and Use

```python
from tensorflow.keras.models import load_model
import json
import numpy as np

# Load model
model = load_model('models/2025_11_10__16_52_33/btc_trend_cnn.h5')

# Load metadata to check input shape
with open('models/2025_11_10__16_52_33/metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Expected input shape: {metadata['input_shape']}")
# Output: Expected input shape: [15, 126]

# Prepare your data to match input shape
# X_test should have shape: (samples, 15, 126)
X_test = prepare_your_data()  # Your data preparation function

# Make predictions
probabilities = model.predict(X_test)
predictions = (probabilities > 0.5).astype(int)

print(f"Predictions: {predictions.flatten()}")
```

### Step 3: Adjust Threshold (Optional)

```python
# The model was optimized for recall
# You can adjust the threshold to balance precision/recall

# Lower threshold = higher recall, lower precision
predictions_high_recall = (probabilities > 0.3).astype(int)

# Higher threshold = lower recall, higher precision
predictions_high_precision = (probabilities > 0.7).astype(int)
```

## Sharing Models

### Option 1: Share Entire Directory

```bash
# Zip the directory
zip -r btc_trend_cnn_2025_11_10.zip models/2025_11_10__16_52_33/

# Share the zip file
# Recipient can unzip and use immediately
```

### Option 2: Share Specific Files

Minimum files needed:
1. `{model_name}.h5` - The model
2. `metadata.json` - Metadata
3. `README.md` - Documentation

```bash
# Create minimal package
mkdir btc_trend_cnn_minimal
cp models/2025_11_10__16_52_33/btc_trend_cnn.h5 btc_trend_cnn_minimal/
cp models/2025_11_10__16_52_33/metadata.json btc_trend_cnn_minimal/
cp models/2025_11_10__16_52_33/README.md btc_trend_cnn_minimal/
zip -r btc_trend_cnn_minimal.zip btc_trend_cnn_minimal/
```

## Timestamp Format

Format: `YYYY_MM_DD__HH_MM_SS`

Examples:
- `2025_11_10__16_52_33` = November 10, 2025 at 4:52:33 PM
- `2025_12_25__09_30_00` = December 25, 2025 at 9:30:00 AM

Benefits:
- Chronological sorting
- Easy to identify when model was trained
- No overwrites (each run gets unique folder)
- Human-readable

## Best Practices

### 1. Organize by Purpose

```
models/
├── production/
│   └── 2025_11_10__16_52_33/  # Current production model
├── experiments/
│   ├── 2025_11_08__14_20_15/  # Experiment 1
│   └── 2025_11_09__10_45_30/  # Experiment 2
└── archive/
    └── 2025_10_15__12_00_00/  # Old models
```

### 2. Document Model Purpose

Add notes to README.md:
```markdown
## Purpose

This model is for BTC trend prediction on 15-minute timeframe.
Optimized for high recall to catch all potential uptrends.
Use in production for entry signal generation.
```

### 3. Version Control

Keep metadata.json in version control:
```bash
git add models/2025_11_10__16_52_33/metadata.json
git add models/2025_11_10__16_52_33/README.md
git commit -m "Add BTC trend CNN model v1.0"
```

### 4. Test Before Deployment

```python
# Load model
model = load_model('models/2025_11_10__16_52_33/btc_trend_cnn.h5')

# Test on holdout set
test_predictions = model.predict(X_test)
test_recall = recall_score(y_test, (test_predictions > 0.5).astype(int))

print(f"Test Recall: {test_recall:.4f}")

# Only deploy if test performance is acceptable
if test_recall > 0.75:
    print("✓ Model ready for deployment")
else:
    print("✗ Model needs improvement")
```

### 5. Monitor Performance

Track model performance over time:
```python
# Log predictions and actual outcomes
performance_log = {
    'date': datetime.now(),
    'model_version': '2025_11_10__16_52_33',
    'predictions': predictions.tolist(),
    'actuals': actuals.tolist(),
    'recall': recall_score(actuals, predictions)
}

# Save to database or file
with open('performance_log.json', 'a') as f:
    json.dump(performance_log, f)
    f.write('\n')
```

## Troubleshooting

### Issue: Model file too large

**Solution**: Use model compression
```python
# Save with compression
model.save('model.h5', save_format='h5', compression='gzip')
```

### Issue: Can't load model in another environment

**Solution**: Check TensorFlow version
```python
# Save TensorFlow version in metadata
metadata['tensorflow_version'] = tf.__version__

# In other environment, install matching version
# pip install tensorflow==2.x.x
```

### Issue: Input shape mismatch

**Solution**: Check metadata.json
```python
# Always check expected input shape
with open('metadata.json', 'r') as f:
    metadata = json.load(f)

expected_shape = metadata['input_shape']
print(f"Expected: (samples, {expected_shape[0]}, {expected_shape[1]})")
print(f"Got: {X_test.shape}")
```

## Summary

✅ **Automatic saving** - Just call `save_best_model()`
✅ **Complete package** - Model + metadata + plots + docs
✅ **Timestamped** - Never overwrites previous models
✅ **Portable** - Easy to share and deploy
✅ **Well-documented** - README included
✅ **Ready to use** - Load and predict immediately

The saved model directory contains everything needed to understand, use, and deploy the model in production!
