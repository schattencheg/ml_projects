# Backtest No Trades - Troubleshooting Guide

## Problem: Backtest Shows No Trades

You see:
```
Buy signals (1): 292
Sell signals (0): 6891
```

But:
```
Total Trades: 0
```

## Root Cause

The model is generating buy signals (292), but **none of them meet your probability threshold**.

## New Debug Output

After the latest fix, you'll now see:

```
ML Signal Statistics:
  Total rows: 7183
  Buy signals (1): 292
  Sell signals (0): 6891

Probability Statistics:
  Avg probability: 0.487
  Max probability: 0.589
  Min probability: 0.012

Buy Signals Meeting Probability Thresholds:
  >= 0.50:  124 signals ( 42.5%)
  >= 0.55:   45 signals ( 15.4%)
  >= 0.60:    0 signals (  0.0%)  ← YOUR THRESHOLD
  >= 0.65:    0 signals (  0.0%)
  >= 0.70:    0 signals (  0.0%)
  >= 0.75:    0 signals (  0.0%)
  >= 0.80:    0 signals (  0.0%)

  ⚠ If all counts are 0, your probability threshold is too high!
  ⚠ Consider lowering probability_threshold parameter.
```

## Solution

### Option 1: Lower the Probability Threshold (Recommended)

If max probability is 0.589 but your threshold is 0.6, **no trades will ever happen**.

```python
# In run_me.py, change:
results, trades = backtester.run_backtest(
    df=df_test_features,
    model=model,
    scaler=scaler,
    X_columns=top_features,
    probability_threshold=0.55,  # ← Lower from 0.6 to 0.55
    trailing_stop_pct=2.0,
)
```

### Option 2: Use Decision Function Instead of Probability

Some models (like SVM) have weak probability calibration. The probabilities might not reach high values even for strong signals.

**Check your model type:**
```python
print(f"Model type: {type(model).__name__}")
```

**If it's SVM or similar:**
- The `predict_proba()` might give poor probabilities
- Consider using a different model (XGBoost, Random Forest, Logistic Regression)
- Or calibrate probabilities with `CalibratedClassifierCV`

### Option 3: Calibrate Model Probabilities

```python
from sklearn.calibration import CalibratedClassifierCV

# After training your model:
model_calibrated = CalibratedClassifierCV(model, cv=5, method='sigmoid')
model_calibrated.fit(X_train, y_train)

# Use calibrated model for backtesting
results, trades = backtester.run_backtest(
    model=model_calibrated,  # ← Use calibrated model
    probability_threshold=0.6,
    ...
)
```

## Understanding the Output

### Scenario 1: Max Probability Below Threshold
```
Max probability: 0.589
probability_threshold: 0.6

Buy Signals Meeting Probability Thresholds:
  >= 0.60:    0 signals (  0.0%)  ← No signals meet threshold
```

**Solution:** Lower threshold to 0.55 or below

### Scenario 2: Some Signals Meet Threshold
```
Max probability: 0.782
probability_threshold: 0.6

Buy Signals Meeting Probability Thresholds:
  >= 0.60:   45 signals ( 15.4%)  ← 45 signals should generate trades
```

**Expected:** You should see ~45 trades (may be less due to position management)

### Scenario 3: All Probabilities Very Low
```
Max probability: 0.523
Avg probability: 0.487

Buy Signals Meeting Probability Thresholds:
  >= 0.50:  124 signals ( 42.5%)
  >= 0.55:    0 signals (  0.0%)
```

**Problem:** Model is not confident in its predictions
**Solutions:**
1. Retrain model with better features
2. Use a different model type
3. Calibrate probabilities
4. Use lower threshold (0.5)

## Recommended Probability Thresholds by Model Type

| Model Type | Recommended Threshold | Notes |
|------------|----------------------|-------|
| Logistic Regression | 0.55 - 0.65 | Well-calibrated probabilities |
| Random Forest | 0.55 - 0.70 | Good probability estimates |
| XGBoost | 0.55 - 0.70 | Reliable probabilities |
| LightGBM | 0.55 - 0.70 | Similar to XGBoost |
| SVM | 0.50 - 0.55 | Weak probability calibration |
| KNN | 0.50 - 0.60 | Probabilities can be noisy |
| Naive Bayes | 0.60 - 0.75 | Often overconfident |

## Quick Fix Commands

### Check Current Settings
```python
# In run_me.py, find this line:
probability_threshold = 0.6  # Current setting
```

### Try Different Thresholds
```python
# Conservative (fewer trades, higher quality)
probability_threshold = 0.65

# Balanced (recommended starting point)
probability_threshold = 0.55

# Aggressive (more trades, lower quality)
probability_threshold = 0.50
```

## Testing Different Thresholds

Run a quick test to find optimal threshold:

```python
# Test multiple thresholds
for threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
    print(f"\n{'='*80}")
    print(f"Testing threshold: {threshold}")
    print(f"{'='*80}")
    
    results, trades = backtester.run_backtest(
        df=df_test_features,
        model=model,
        scaler=scaler,
        X_columns=top_features,
        probability_threshold=threshold,
        trailing_stop_pct=2.0,
        plot=False  # Disable plots for speed
    )
    
    print(f"Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.1f}%")
    print(f"Total Return: {results['total_return_pct']:.2f}%")
```

## Expected Behavior After Fix

After lowering the threshold appropriately, you should see:

```
ML Signal Statistics:
  Buy signals (1): 292

Buy Signals Meeting Probability Thresholds:
  >= 0.55:   45 signals ( 15.4%)  ← Your new threshold

Strategy Debug Info:
  Total buy signals (1): 292
  Entry attempts (signal + prob >= threshold): 45
  Actual trades executed: 42  ← Trades are happening!
  Probability threshold: 0.55

BACKTEST RESULTS
================================================================================
Trades:
  Total Trades:           42  ← Success!
  Won Trades:             26
  Lost Trades:            16
  Win Rate:               61.90%
```

## Still No Trades?

If you've lowered the threshold and still see no trades:

1. **Check broker cash:**
   ```python
   backtester = BacktestBacktraderML(
       initial_cash=100000.0,  # Increase if needed
       commission=0.001
   )
   ```

2. **Check position sizing:**
   ```python
   results, trades = backtester.run_backtest(
       ...,
       position_size_pct=0.95,  # Use 95% of cash
   )
   ```

3. **Enable detailed logging:**
   ```python
   results, trades = backtester.run_backtest(
       ...,
       printlog=True  # See every bar's decision
   )
   ```

4. **Check data quality:**
   - Ensure OHLCV data is valid
   - Check for NaN/inf values
   - Verify timestamps are sequential

## Summary

**Most Common Issue:** Probability threshold too high for model's output

**Quick Fix:** Lower `probability_threshold` from 0.6 to 0.55 or 0.5

**Verify Fix:** Check "Buy Signals Meeting Probability Thresholds" output

**Expected Result:** Trades should be generated if signals meet the new threshold
