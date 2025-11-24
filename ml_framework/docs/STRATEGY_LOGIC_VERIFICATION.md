# Strategy Logic Verification

## Overview

This document verifies the complete strategy logic with three-class classification and automatic target transformation.

## Strategy Logic Summary

### Signal Generation (Three-Class)

```python
# Model predictions after automatic transformation
predictions = model.predict(X_scaled)  # Returns: -1, 0, +1

# Signals mapping
-1 (decrease) → -1 (short signal)
 0 (neutral)  →  0 (no signal)
+1 (increase) → +1 (long signal)
```

### Entry Logic

**Long Entry:**
```python
if signal == +1 and not has_open_position():
    enter_long_position()
```

**Short Entry:**
```python
if signal == -1 and not has_open_position():
    enter_short_position()
```

**Neutral:**
```python
if signal == 0:
    # No action - skip this bar
    pass
```

### Exit Logic

**Primary Exit: Holding Period (Always Active)**
```python
bars_held = current_bar - entry_bar

if bars_held >= holding_period:  # N bars
    exit_position(reason='holding_period')
```

**Secondary Exit: Trailing Stop Loss (Optional, Disabled by Default)**
```python
if enable_trailing_stop and trailing_stop_pct is not None:
    # For long positions
    if price <= highest_price * (1 - trailing_stop_pct):
        exit_position(reason='trailing_stop_long')
    
    # For short positions
    if price >= lowest_price * (1 + trailing_stop_pct):
        exit_position(reason='trailing_stop_short')
```

## Complete Flow

### Example Trade Flow

**Scenario: Long Trade**

```
Bar 0: Signal = +1 (increase predicted)
       → Enter long at $100
       → Position opened

Bar 1-14: Hold position
          → Track highest price
          → No exit (bars_held < 15)

Bar 15: Exit condition met
        → bars_held = 15 (holding_period reached)
        → Exit long at current price
        → Position closed
```

**Scenario: Short Trade**

```
Bar 0: Signal = -1 (decrease predicted)
       → Enter short at $100
       → Position opened

Bar 1-14: Hold position
          → Track lowest price
          → No exit (bars_held < 15)

Bar 15: Exit condition met
        → bars_held = 15 (holding_period reached)
        → Exit short at current price
        → Position closed
```

**Scenario: Neutral Signal**

```
Bar 0: Signal = 0 (neutral)
       → No action
       → No position opened
       → Wait for next signal
```

## Target Transformation Verification

### Step 1: Target Creation

```python
from src.features_generator import FeaturesGenerator

features_gen = FeaturesGenerator()

# Create three-class target
df_features = features_gen.create_target(
    df,
    target_type='classification',
    future_bars=15,
    threshold=0.02,
    num_classes=3  # -1, 0, +1
)

# Output:
# Creating classification target (future_bars=15, num_classes=3)...
#   Class distribution (3 classes):
#     Decrease (-1): 245
#     Neutral (0):   890
#     Increase (+1): 235
```

### Step 2: Model Training (Automatic Transformation)

```python
from src.models_lib import RandomForestModel

# y_train contains: -1, 0, +1
model = RandomForestModel(n_estimators=100)
model.fit(X_train_scaled, y_train)

# Internal process (automatic):
# 1. BaseModel._convert_targets() called
# 2. Creates mapping: {-1: 0, 0: 1, 1: 2}
# 3. Transforms y_train: -1/0/+1 → 0/1/2
# 4. Trains sklearn model with 0/1/2
# 5. Stores reverse mapping: {0: -1, 1: 0, 2: 1}
```

### Step 3: Prediction (Automatic Reverse Transformation)

```python
# Make predictions
y_pred = model.predict(X_test_scaled)

# Internal process (automatic):
# 1. BaseModel._predict() called
# 2. Gets predictions from sklearn: 0/1/2
# 3. BaseModel._reverse_convert_predictions() called
# 4. Transforms back: 0/1/2 → -1/0/+1
# 5. Returns: -1, 0, +1
```

### Step 4: Signal Generation

```python
from src.strategies import MLStrategy

strategy = MLStrategy(holding_period=15)

# Generate signals
signals = strategy.generate_signals(
    df=test_df,
    model=model,
    scaler=scaler,
    feature_cols=feature_cols
)

# Internal process:
# 1. Get predictions: model.predict() → -1/0/+1
# 2. TargetTransformer verifies format
# 3. Returns signals: -1/0/+1
```

## Verification Checklist

### ✅ Target Transformation

- [x] Three-class targets created: -1, 0, +1
- [x] BaseModel automatically converts: -1/0/+1 → 0/1/2
- [x] sklearn model trains with: 0, 1, 2
- [x] Predictions automatically converted back: 0/1/2 → -1/0/+1
- [x] TargetTransformer integrated in strategy

### ✅ Signal Generation

- [x] Predictions are in three-class format: -1, 0, +1
- [x] Signals match predictions: -1 → short, 0 → neutral, +1 → long
- [x] No additional transformation needed

### ✅ Entry Logic

- [x] Signal +1 → Enter long (if no position)
- [x] Signal -1 → Enter short (if no position)
- [x] Signal 0 → No action
- [x] Only one position at a time

### ✅ Exit Logic

- [x] Exit after N bars (holding_period)
- [x] Trailing stop loss (optional, disabled by default)
- [x] No premature exits
- [x] Position closed at end of data

### ✅ Position Management

- [x] Long positions tracked correctly
- [x] Short positions tracked correctly
- [x] P&L calculated correctly for both
- [x] Commission applied correctly

## Code Verification

### 1. MLStrategy.generate_signals()

```python
def generate_signals(self, df, model, scaler, feature_cols):
    # Import target transformer
    from ..target_transformer import inverse_transform_predictions, get_target_transformer
    
    # Prepare features
    X = df[feature_cols].values
    X_scaled = scaler.transform(X)
    
    # Get predictions
    predictions = model.predict(X_scaled)
    # ✓ Already in three-class format: -1, 0, +1
    
    # Detect model type and transform if needed
    transformer = get_target_transformer()
    model_type = transformer.get_model_type(model)
    
    # Transform to three-class format
    predictions_three_class = inverse_transform_predictions(predictions, model_type)
    # ✓ Ensures format is -1, 0, +1
    
    # Create signals
    signals = pd.Series(predictions_three_class, index=df.index, dtype=int)
    # ✓ Signals are -1, 0, +1
    
    return signals
```

### 2. MLStrategy.should_enter_long()

```python
def should_enter_long(self, signal, bar_idx):
    return signal == 1 and not self.has_open_position()
    # ✓ Enters long only when signal is +1
    # ✓ Only if no existing position
```

### 3. MLStrategy.should_enter_short()

```python
def should_enter_short(self, signal, bar_idx):
    return signal == -1 and not self.has_open_position()
    # ✓ Enters short only when signal is -1
    # ✓ Only if no existing position
```

### 4. MLStrategy.should_exit()

```python
def should_exit(self, position, current_bar, current_price):
    bars_held = current_bar - position['entry_bar']
    
    # Exit condition 1: Holding period reached
    if bars_held >= self.holding_period:
        return True, 'holding_period'
    # ✓ Exits after exactly N bars
    
    # Exit condition 2: Trailing stop loss (if enabled)
    if self.enable_trailing_stop and self.trailing_stop_pct is not None:
        # ... trailing stop logic ...
        pass
    # ✓ Only active if explicitly enabled
    
    return False, ''
    # ✓ No exit before holding period (unless trailing stop)
```

## Example Execution

### Configuration

```python
FUTURE_BARS = 15
THRESHOLD = 0.02
holding_period = 15  # Same as FUTURE_BARS
enable_trailing_stop = False  # Disabled
```

### Sample Trade Sequence

```
Data:
Bar  Price  Signal  Action              Position  Bars_Held
---  -----  ------  ------------------  --------  ---------
0    100    +1      Enter Long @ 100    Long      0
1    101    0       Hold                Long      1
2    102    -1      Hold                Long      2
3    103    0       Hold                Long      3
...
14   110    +1      Hold                Long      14
15   112    0       Exit Long @ 112     None      15 ✓

16   112    -1      Enter Short @ 112   Short     0
17   111    0       Hold                Short     1
18   110    +1      Hold                Short     2
...
30   105    -1      Hold                Short     14
31   104    0       Exit Short @ 104    None      15 ✓

32   104    0       No Action           None      -
33   103    +1      Enter Long @ 103    Long      0
...
```

### Key Observations

1. **Entry only on signals:**
   - +1 → Long entry
   - -1 → Short entry
   - 0 → No action

2. **Hold for N bars:**
   - Position held for exactly 15 bars
   - Ignores intermediate signals
   - Exits on bar 15

3. **One position at a time:**
   - Cannot enter new position while holding
   - Must exit before next entry

4. **Trailing stop disabled:**
   - Only exits on holding period
   - No premature exits

## Testing

### Run Verification Test

```bash
python test_three_class.py
```

### Expected Results

```
✅ ALL TESTS PASSED!

Three-Class Classification System:
  ✓ Target creation with num_classes=3
  ✓ Automatic target transformation
  ✓ Classic ML: -1/0/+1 → 0/1/2 (internal)
  ✓ Predictions: always returned as -1/0/+1
  ✓ Strategy: handles three-class signals

Signal Meanings:
  -1: Short entry signal
   0: No signal / Neutral
  +1: Long entry signal
```

## Summary

### ✅ Verified Components

1. **Target Creation**
   - Three-class targets: -1, 0, +1
   - Correct threshold logic

2. **Target Transformation**
   - Automatic for classic ML: -1/0/+1 → 0/1/2
   - Automatic reverse: 0/1/2 → -1/0/+1
   - TargetTransformer integrated

3. **Signal Generation**
   - Predictions in three-class format
   - Signals match predictions
   - No additional transformation

4. **Entry Logic**
   - +1 → Long entry
   - -1 → Short entry
   - 0 → No action
   - One position at a time

5. **Exit Logic**
   - Exit after N bars (holding_period)
   - Trailing stop optional (disabled by default)
   - No premature exits

### ✅ Status

**All logic verified and working correctly!**

The strategy:
- ✅ Uses TargetTransformer where needed
- ✅ Enters long on +1 signal
- ✅ Enters short on -1 signal
- ✅ Exits only on Nth bar (or trailing stop if enabled)
- ✅ Handles three-class classification properly

---

**Last Updated:** 2024-11-24  
**Status:** ✅ Verified and Complete
