# Migration to 3-Class Classification

## Overview

The ml_predict_15 project has been migrated from **2-class binary classification** to **3-class multi-class classification** to better capture market movements.

## Classification Scheme

### Previous (2-Class Binary)
- **Class 0:** No Rise (price did not increase by target_pct)
- **Class 1:** Rise (price increased by target_pct or more)

### Current (3-Class Multi-Class)
- **Class 0:** Down (price decreased by target_pct or more)
- **Class 1:** Neutral (price stayed within ±target_pct threshold)
- **Class 2:** Up (price increased by target_pct or more)

## Benefits of 3-Class Classification

1. **Better Market Representation:** Captures sideways/neutral market movements
2. **More Nuanced Predictions:** Distinguishes between down, neutral, and up movements
3. **Improved Risk Management:** Can identify when to stay out of the market (neutral)
4. **Realistic Trading:** Reflects actual market conditions where not all movements are significant

## Files Modified

### 1. `src/FeaturesGenerator.py`
**Change:** Target label mapping in `add_target()` method

**Before:**
```python
# Two classes: 1 (up), 0 (not up)
df['target'] = (df[f'pct_change_{target_bars}'] >= target_pct).astype(int)
```

**After:**
```python
# Three classes: down, neutral, up
# Map to [0, 1, 2] for sklearn compatibility
target_up = (df[f'pct_change_{target_bars}'] >= target_pct)
target_down = (df[f'pct_change_{target_bars}'] <= -target_pct)
df['target'] = 1  # neutral (default)
df.loc[target_up, 'target'] = 2  # up
df.loc[target_down, 'target'] = 0  # down
```

**Reason:** sklearn requires labels to start from 0 and be consecutive integers.

### 2. `src/Trainer.py`
**Change:** Ensure labels remain integers after SMOTE resampling

**Added:**
```python
# After SMOTE resampling
X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
# Ensure y_train remains integer (SMOTE might convert to float)
y_train = y_train.astype(np.int32)
```

**Reason:** SMOTE can convert integer labels to float, causing issues with neural networks.

### 3. `src/NeuralNetworksManager.py`
**Multiple Changes:**

#### a) Output Layer Neurons (12 models updated)
**Before:**
```python
layers.Dense(2, activation='softmax')  # 2 classes
```

**After:**
```python
layers.Dense(3, activation='softmax')  # 3 classes: down, neutral, up
```

**Models Updated:**
- cnn_simple, cnn_deep, cnn_residual, cnn_attention, cnn_dilated
- lstm_simple, lstm_bidirectional, lstm_stacked, lstm_attention
- lstm_cnn_hybrid
- gru_simple, gru_bidirectional

#### b) Label Validation in `fit()` method
**Added:**
```python
# Ensure labels are integers
if not np.issubdtype(y.dtype, np.integer):
    y = y.astype(np.int32)

# If labels are not starting from 0, remap them
unique_labels = np.unique(y)
if not np.array_equal(unique_labels, np.arange(len(unique_labels))):
    self.label_map_ = {idx: label for idx, label in enumerate(unique_labels)}
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label_map[label] for label in y])
else:
    self.label_map_ = None

# Validate labels after sequence creation
y_seq = y_seq.astype(np.int32)
unique_y_seq = np.unique(y_seq)
num_classes = len(unique_y_seq)

if not np.array_equal(unique_y_seq, np.arange(num_classes)):
    raise ValueError(
        f"Labels must be consecutive integers starting from 0. "
        f"Got: {unique_y_seq}, expected: {np.arange(num_classes)}"
    )
```

#### c) Prediction Padding
**Added to handle sequence length mismatch:**
```python
# In __init__
self.n_samples_dropped_ = 0  # Number of samples dropped during sequence creation

# In fit()
self.n_samples_dropped_ = len(y) - len(y_seq)

# In predict()
if self.n_samples_dropped_ > 0:
    pad_value = np.bincount(predicted_indices).argmax()
    padded_predictions = np.full(len(X), pad_value, dtype=predicted_indices.dtype)
    padded_predictions[self.n_samples_dropped_:] = predicted_indices
    return padded_predictions

# In predict_proba()
if self.n_samples_dropped_ > 0:
    n_classes = probabilities.shape[1]
    uniform_proba = np.full((len(X), n_classes), 1.0 / n_classes)
    uniform_proba[self.n_samples_dropped_:] = probabilities
    return uniform_proba
```

**Reason:** Neural networks create sequences which drop the first `sequence_length` samples. Predictions must be padded to match original input length.

### 4. `run_me.py`
**Changes:** Updated target_names in 2 locations

**Before:**
```python
target_names=['No Rise', 'Rise']  # 2 classes
```

**After:**
```python
target_names=['Down', 'Neutral', 'Up']  # 3 classes
```

**Locations:**
- Line 528: `tester.print_detailed_report()`
- Line 543: `report_manager.generate_comprehensive_report()`

### 5. Example Files
**Files Updated:**
- `examples/example_refactored_workflow.py`
- `examples/example_crypto_features.py` (2 occurrences)
- `examples/example_complete_workflow.py`

**Change:** Updated all `target_names` from 2 to 3 classes

## Technical Details

### Label Requirements for sklearn/TensorFlow

1. **Must be integers:** Not floats or strings
2. **Must start from 0:** Labels must be 0, 1, 2, ... n-1
3. **Must be consecutive:** No gaps (e.g., 0, 2, 5 is invalid)
4. **Correct dtype:** np.int32 or np.int64

### Neural Network Requirements

1. **Output neurons = number of classes:** 3 neurons for 3 classes
2. **Loss function:** `sparse_categorical_crossentropy` for integer labels
3. **Activation:** `softmax` for multi-class classification
4. **Prediction shape:** Must match input shape (padding required for sequences)

## Testing the Changes

### 1. Verify Target Distribution
```python
from src.FeaturesGenerator import FeaturesGenerator

fg = FeaturesGenerator()
df = fg.add_target(df, method='classification', target_bars=15, target_pct=0.5)

# Check distribution
print(df['target'].value_counts().sort_index())
# Should show:
# 0    XXXX  (Down)
# 1    XXXX  (Neutral)
# 2    XXXX  (Up)
```

### 2. Train Models
```python
from src.Trainer import Trainer
from src.ModelsManager import ModelsManager

# Create models
manager = ModelsManager()
models = manager.create_models(enabled_only=True)

# Train
trainer = Trainer(use_smote=True, optimize_threshold=True)
trained_models, scaler, results, best_model = trainer.train(
    models=models,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val
)

# Check predictions have 3 classes
for model_name, model in trained_models.items():
    y_pred = model.predict(X_test)
    print(f"{model_name}: {np.unique(y_pred)}")  # Should show [0 1 2]
```

### 3. Verify Neural Networks
```python
# Check output shape
model = trained_models['cnn_simple']
proba = model.predict_proba(X_test)
print(f"Probability shape: {proba.shape}")  # Should be (n_samples, 3)
print(f"Predictions: {np.unique(model.predict(X_test))}")  # Should be [0 1 2]
```

## Common Issues and Solutions

### Issue 1: "Invalid classes inferred from unique values of y"
**Cause:** Labels are not [0, 1, 2]

**Solution:** Check FeaturesGenerator.add_target() - ensure it maps to [0, 1, 2]

### Issue 2: "Graph execution error: InvalidArgumentError"
**Cause:** Neural network has wrong number of output neurons

**Solution:** Check all neural network models have `Dense(3, activation='softmax')`

### Issue 3: "Inconsistent numbers of samples"
**Cause:** Neural network predictions don't match input length due to sequence creation

**Solution:** Ensure prediction padding is implemented in KerasClassifierWrapper

### Issue 4: "Number of classes does not match size of target_names"
**Cause:** Using 2 class names for 3-class classification

**Solution:** Update all `target_names` to `['Down', 'Neutral', 'Up']`

## Performance Considerations

### Class Imbalance
With 3 classes, you may have:
- **Down:** ~30-40% of samples
- **Neutral:** ~20-40% of samples  
- **Up:** ~30-40% of samples

**Recommendations:**
1. Use SMOTE to balance classes
2. Use `class_weight='balanced'` in sklearn models
3. Monitor per-class metrics (precision, recall, F1)

### Evaluation Metrics
For 3-class classification, use:
- **Weighted average:** Accounts for class imbalance
- **Macro average:** Equal weight to all classes
- **Per-class metrics:** Understand performance on each class

```python
from sklearn.metrics import classification_report

print(classification_report(
    y_test, y_pred, 
    target_names=['Down', 'Neutral', 'Up'],
    zero_division=0
))
```

## Migration Checklist

- [x] Update FeaturesGenerator target mapping to [0, 1, 2]
- [x] Add int32 conversion after SMOTE in Trainer
- [x] Update all 12 neural network models to 3 output neurons
- [x] Add label validation in NeuralNetworksManager
- [x] Implement prediction padding for neural networks
- [x] Update target_names in run_me.py (2 locations)
- [x] Update target_names in example files (4 files)
- [x] Test traditional ML models
- [x] Test neural network models
- [x] Verify visualization outputs
- [x] Update documentation

## Summary

The migration to 3-class classification provides a more realistic representation of market movements by distinguishing between down, neutral, and up price changes. All models (traditional ML and neural networks) have been updated to support this new classification scheme while maintaining backward compatibility where possible.

**Key Changes:**
- Target labels: [-1, 0, 1] → [0, 1, 2]
- Neural network outputs: 2 neurons → 3 neurons
- Target names: ['No Rise', 'Rise'] → ['Down', 'Neutral', 'Up']
- Added prediction padding for neural networks
- Enhanced label validation and type checking

The system is now ready for 3-class cryptocurrency price movement prediction! 🚀
