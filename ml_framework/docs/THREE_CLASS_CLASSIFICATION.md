# Three-Class Classification System

## Overview

The framework now supports three-class classification with automatic target transformation for different model types.

**Three Classes:**
- **-1 (Decrease)**: Price drops by more than threshold
- **0 (Neutral)**: Price change within ±threshold  
- **+1 (Increase)**: Price rises by more than threshold

## Key Features

✅ **Automatic Target Transformation**
- Classic ML models: -1/0/+1 → 0/1/2 (internally)
- Deep Learning models: -1/0/+1 → -1/0/+1 (no change)
- Predictions: Always returned as -1/0/+1

✅ **Seamless Integration**
- Works with all existing models
- No code changes required
- Backward compatible

✅ **Strategy Support**
- MLStrategy handles three-class signals
- -1 → Short entry
- 0 → No signal/Neutral
- +1 → Long entry

## Quick Start

### 1. Create Three-Class Target

```python
from src.features_generator import FeaturesGenerator

features_gen = FeaturesGenerator()

# Create three-class target
df_features = features_gen.create_target(
    df,
    target_type='classification',
    future_bars=15,
    threshold=0.02,
    num_classes=3  # Three classes: -1, 0, +1
)
```

**Output:**
```
Creating classification target (future_bars=15, num_classes=3)...
  Class distribution (3 classes):
    Decrease (-1): 245
    Neutral (0):   890
    Increase (+1): 235
✓ Target created, removed 15 rows with NaN target
```

### 2. Train Model (Automatic Transformation)

```python
from src.models_lib import RandomForestModel
from sklearn.preprocessing import StandardScaler

# Prepare data
X_train = train_df[feature_cols].values
y_train = train_df['target'].values  # Values: -1, 0, +1

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model (automatic target transformation)
model = RandomForestModel(n_estimators=100, n_jobs=-1)
model.fit(X_train_scaled, y_train)
# Internally: -1/0/+1 → 0/1/2 for sklearn compatibility

# Make predictions (automatic reverse transformation)
y_pred = model.predict(X_test_scaled)
# Returns: -1, 0, +1 (original format)
```

### 3. Use with Strategy

```python
from src.strategies import MLStrategy

# Create strategy
strategy = MLStrategy(
    name='ThreeClass_Strategy',
    holding_period=15,
    trailing_stop_pct=0.05,
    enable_trailing_stop=False
)

# Run backtest
results = strategy.backtest(
    df=test_df,
    model=model,
    scaler=scaler,
    feature_cols=feature_cols,
    initial_capital=10000,
    position_size_pct=1.0,
    commission=0.001
)

# Signals generated:
# +1 → Enter long position
#  0 → No action (neutral)
# -1 → Enter short position
```

## Target Creation

### Three-Class vs Two-Class

**Two-Class (Binary):**
```python
df = features_gen.create_target(
    df,
    target_type='classification',
    future_bars=15,
    threshold=0.02,
    num_classes=2  # 0 (no increase), 1 (increase)
)
```

**Three-Class (Ternary):**
```python
df = features_gen.create_target(
    df,
    target_type='classification',
    future_bars=15,
    threshold=0.02,
    num_classes=3  # -1 (decrease), 0 (neutral), +1 (increase)
)
```

### Target Logic

```python
future_return = (price[t+N] - price[t]) / price[t]

if future_return > threshold:
    target = +1  # Increase
elif future_return < -threshold:
    target = -1  # Decrease
else:
    target = 0   # Neutral
```

**Example with threshold=0.02:**
- Price rises 3% → +1 (Increase)
- Price drops 3% → -1 (Decrease)
- Price changes 1% → 0 (Neutral)

## Automatic Target Transformation

### How It Works

The `BaseModel` class automatically handles target transformation:

```python
class BaseModel:
    def fit(self, X, y):
        # Automatically convert targets
        # -1, 0, +1 → 0, 1, 2 (for sklearn)
        y_converted = self._convert_targets(y)
        
        # Train underlying model
        self._fit(X, y_converted)
    
    def predict(self, X):
        # Get predictions (0, 1, 2)
        y_pred = self._predict(X)
        
        # Automatically convert back
        # 0, 1, 2 → -1, 0, +1
        return self._reverse_convert_predictions(y_pred)
```

### Target Mapping

**For Classic ML Models (sklearn, XGBoost, etc.):**

| Original | Internal | Meaning |
|----------|----------|---------|
| -1 | 0 | Decrease |
| 0 | 1 | Neutral |
| +1 | 2 | Increase |

**For Deep Learning Models (TensorFlow, PyTorch):**

| Original | Internal | Meaning |
|----------|----------|---------|
| -1 | -1 | Decrease |
| 0 | 0 | Neutral |
| +1 | +1 | Increase |

### Why Transformation?

**Classic ML models (sklearn) require:**
- Sequential class labels starting from 0
- Example: 0, 1, 2, 3, ...
- Cannot handle negative labels

**Deep Learning models can handle:**
- Any numeric labels
- Including negative values
- Example: -1, 0, +1

## Target Transformer API

### Manual Transformation

```python
from src.target_transformer import TargetTransformer

transformer = TargetTransformer()

# For classic ML
y_train_ml = transformer.transform_for_classic_ml(y_train)
# -1, 0, +1 → 0, 1, 2

# For deep learning
y_train_dl = transformer.transform_for_deep_learning(y_train)
# -1, 0, +1 → -1, 0, +1 (no change)

# Inverse transformation
y_pred_original = transformer.inverse_transform_classic_ml(y_pred_ml)
# 0, 1, 2 → -1, 0, +1
```

### Automatic Transformation

```python
from src.target_transformer import transform_for_model

# Automatically detect model type and transform
y_transformed, model_type = transform_for_model(
    y_train,
    model,
    verbose=True
)

# Output:
# ============================================================
# TARGET TRANSFORMATION - CLASSIC_ML
# ============================================================
# Mapping: -1→0, 0→1, +1→2
# 
# Original distribution:
#   -1:   245 samples
#    0:   890 samples
#   +1:   235 samples
# 
# Transformed distribution:
#    0:   245 samples
#    1:   890 samples
#    2:   235 samples
# ============================================================
```

### Model Type Detection

```python
from src.target_transformer import get_target_transformer

transformer = get_target_transformer()

# Detect model type
model_type = transformer.get_model_type(model)

# Returns:
# 'classic_ml' for sklearn, XGBoost, CatBoost, etc.
# 'deep_learning' for TensorFlow, PyTorch, Keras, etc.
```

## Strategy Integration

### Signal Generation

The `MLStrategy` automatically handles three-class predictions:

```python
def generate_signals(self, df, model, scaler, feature_cols):
    # Get predictions from model
    predictions = model.predict(X_scaled)
    # Already in three-class format: -1, 0, +1
    
    # Transform to signals
    # -1 (decrease) → -1 (short signal)
    #  0 (neutral)  →  0 (no signal)
    # +1 (increase) → +1 (long signal)
    
    return signals
```

### Entry Logic

```python
# Long entry
if signal == +1 and no_open_position:
    enter_long()

# Short entry
if signal == -1 and no_open_position:
    enter_short()

# Neutral
if signal == 0:
    # No action
    pass
```

## Complete Example

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_provider import DataProvider
from src.features_generator import FeaturesGenerator
from src.models_lib import RandomForestModel
from src.strategies import MLStrategy
from sklearn.preprocessing import StandardScaler

# 1. Load data
data_provider = DataProvider(data_dir='data')
df = data_provider.load_yahoo('BTC-USD', '2020-01-01', '2024-11-24')

# 2. Generate features
features_gen = FeaturesGenerator()
df_features = features_gen.generate_features(df, feature_set='advanced')

# 3. Create three-class target
df_features = features_gen.create_target(
    df_features,
    target_type='classification',
    future_bars=15,
    threshold=0.02,
    num_classes=3  # Three classes: -1, 0, +1
)
df_features = df_features.dropna()

# 4. Split data
train_df, val_df, test_df = data_provider.split_data(df_features)
feature_cols = features_gen.get_feature_names()

# 5. Train model
X_train = train_df[feature_cols].values
y_train = train_df['target'].values  # -1, 0, +1

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = RandomForestModel(n_estimators=100, n_jobs=-1)
model.fit(X_train_scaled, y_train)
# Automatic transformation: -1/0/+1 → 0/1/2 internally

# 6. Create strategy
strategy = MLStrategy(
    name='BTC_ThreeClass',
    holding_period=15,
    trailing_stop_pct=0.05,
    enable_trailing_stop=False
)

# 7. Run backtest
results = strategy.backtest(
    df=test_df,
    model=model,
    scaler=scaler,
    feature_cols=feature_cols,
    initial_capital=10000,
    position_size_pct=1.0,
    commission=0.001
)

# 8. Analyze results
print(f"Initial Capital: ${results['initial_capital']:,.2f}")
print(f"Final Capital: ${results['final_capital']:,.2f}")
print(f"Total Return: {(results['final_capital']/results['initial_capital']-1)*100:.2f}%")

stats = results['strategy_stats']
print(f"\nTotal Trades: {stats['total_trades']}")
print(f"Win Rate: {stats['win_rate']*100:.2f}%")
```

## Benefits

### 1. More Granular Predictions

**Two-Class:**
- 0: Don't buy
- 1: Buy

**Three-Class:**
- -1: Short (expect decrease)
- 0: Hold (neutral)
- +1: Long (expect increase)

### 2. Better Risk Management

- Neutral class avoids false signals
- Can stay out of uncertain markets
- More conservative trading

### 3. Short Selling Support

- Native support for short positions
- -1 signal triggers short entry
- Balanced long/short strategy

### 4. Improved Performance

- Reduces false positives
- Better signal quality
- Higher win rate potential

## Testing

Run the test script:

```bash
python test_three_class.py
```

**Expected Output:**
```
✅ ALL TESTS PASSED!

Three-Class Classification System:
  ✓ Target creation with num_classes=3
  ✓ Automatic target transformation
  ✓ Classic ML: -1/0/+1 → 0/1/2 (internal)
  ✓ Deep Learning: -1/0/+1 → -1/0/+1 (no change)
  ✓ Predictions: always returned as -1/0/+1
  ✓ Strategy: handles three-class signals

Target Meanings:
  -1: Decrease (price drops > threshold)
   0: Neutral (price change within ±threshold)
  +1: Increase (price rises > threshold)

Signal Meanings:
  -1: Short entry signal
   0: No signal / Neutral
  +1: Long entry signal
```

## Best Practices

### 1. Choose Appropriate Threshold

```python
# Conservative (more neutral predictions)
threshold = 0.03  # 3%

# Balanced
threshold = 0.02  # 2%

# Aggressive (fewer neutral predictions)
threshold = 0.01  # 1%
```

### 2. Handle Class Imbalance

Three-class targets often have imbalanced distribution:
- Neutral class: ~60-70% (most common)
- Increase class: ~15-20%
- Decrease class: ~15-20%

Use SMOTE or class weights to handle imbalance.

### 3. Monitor Neutral Predictions

```python
# Check neutral prediction rate
neutral_rate = (y_pred == 0).mean()
print(f"Neutral predictions: {neutral_rate*100:.1f}%")

# If too high (>70%), consider:
# - Reducing threshold
# - Adding more features
# - Using different model
```

### 4. Validate on Multiple Assets

Test on different assets to ensure generalization:
- BTC-USD (crypto)
- ETH-USD (crypto)
- SPY (stocks)
- ES (futures)

## Troubleshooting

### Issue: All predictions are neutral (0)

**Cause:** Threshold too high or model not confident

**Solution:**
```python
# Lower threshold
threshold = 0.01  # Instead of 0.02

# Check model confidence
proba = model.predict_proba(X_test)
print(f"Max probability: {proba.max(axis=1).mean():.2f}")
```

### Issue: No short signals generated

**Cause:** Model predicts mostly increase/neutral

**Solution:**
```python
# Check class distribution in predictions
unique, counts = np.unique(y_pred, return_counts=True)
for val, count in zip(unique, counts):
    print(f"Class {val:+2d}: {count} ({count/len(y_pred)*100:.1f}%)")

# Consider:
# - Balancing training data
# - Using different features
# - Adjusting threshold
```

### Issue: Target transformation errors

**Cause:** Model not properly inheriting from BaseModel

**Solution:**
```python
# Ensure model inherits from BaseModel
from src.models_lib.base_model import BaseModel

class MyModel(BaseModel):
    def _fit(self, X, y, **kwargs):
        # y is already transformed (0, 1, 2)
        pass
    
    def _predict(self, X, **kwargs):
        # Return predictions as (0, 1, 2)
        # BaseModel will convert back to (-1, 0, +1)
        pass
```

## Summary

✅ **Three-Class System Implemented**
- Target creation with `num_classes=3`
- Automatic transformation for all models
- Strategy support for three-class signals

✅ **Seamless Integration**
- Works with existing models
- No code changes required
- Backward compatible

✅ **Complete Testing**
- Test script provided
- All features verified
- Ready to use

---

**Status:** ✅ Complete and Ready to Use  
**Last Updated:** 2024-11-24  
**Version:** 1.0.0
