# Backtrader No Trades - Debug Checklist

## Current Status
- ✅ Buy signals exist: 292
- ✅ Entry attempts: 1 (at least one signal meets threshold)
- ❌ Actual trades: 0 (order not executing)
- ✅ Position size increased to 90%
- ✅ Detailed logging enabled

## What to Check in Log Output

With `printlog=True`, you should see output like this for EVERY bar:

### Expected Log Output

```
2024-01-15 Position Sizing: Cash=$10000.00, Use=90.0%, Amount=$9000.00, Price=$50000.00, Size=0.180000
2024-01-15 BUY CREATE, Price: 50000.00, Size: 0.18, Prob: 0.65
2024-01-15 BUY EXECUTED, Price: 50000.05, Cost: 9000.09, Comm: 9.00
```

### If Order Fails

```
2024-01-15 Position Sizing: Cash=$10000.00, Use=90.0%, Amount=$9000.00, Price=$50000.00, Size=0.180000
2024-01-15 BUY CREATE, Price: 50000.00, Size: 0.18, Prob: 0.65
2024-01-15 Order Canceled/Margin/Rejected  ← THIS IS THE PROBLEM
```

## Common Reasons for Order Rejection

### 1. Insufficient Cash
**Symptom:** "Order Canceled/Margin/Rejected" immediately after BUY CREATE

**Check:**
```python
# In the log, look for:
Position Sizing: Cash=$100.00, Use=90.0%, Amount=$90.00, Price=$50000.00, Size=0.001800
```

If Cash is very low, you don't have enough to buy.

**Fix:** Increase initial_cash
```python
backtester = BacktestBacktraderML(
    initial_cash=100000.0,  # Increase from 10000
    commission=0.001
)
```

### 2. Commission Too High
**Symptom:** Order creates but immediately cancels

**Check:** Commission might consume all available cash

**Fix:** Lower commission
```python
backtester = BacktestBacktraderML(
    initial_cash=10000.0,
    commission=0.0001  # Lower from 0.001
)
```

### 3. Volume Column Missing/Wrong
**Symptom:** No logs at all, or error about missing data

**Status:** ✅ FIXED - Volume column now normalized to lowercase

### 4. Data Feed Not Loading
**Symptom:** Strategy never calls `next()`, no logs

**Check:** Look for these messages:
```
ML Signal Statistics:
  Total rows: 7183
  Buy signals (1): 292
```

If you don't see this, data isn't being prepared correctly.

### 5. Backtrader Minimum Size
**Symptom:** Small position sizes get rejected

**Check:** If Size < 0.01, might be too small

**Fix:** Already handled with position_size_pct=0.9

## Debugging Steps

### Step 1: Check What You See in Console

Run your backtest and look for:

1. **Data Preparation Output:**
   ```
   ML Signal Statistics:
     Total rows: 7183
     Buy signals (1): 292
   
   Buy Signals Meeting Probability Thresholds:
     >= 0.60:    1 signals (  0.3%)
   ```

2. **Strategy Logs:**
   - Do you see ANY date-stamped lines?
   - Do you see "BUY CREATE"?
   - Do you see "Position Sizing"?
   - Do you see "Order Canceled/Margin/Rejected"?

3. **Strategy Debug Info:**
   ```
   Strategy Debug Info:
     Total buy signals (1): 292
     Entry attempts: 1
     Actual trades: 0
   ```

### Step 2: Share Log Output

Please share the COMPLETE output from your backtest run, especially:
- Everything between "BACKTEST 1/1: LOGISTIC_REGRESSION" and "BACKTEST RESULTS"
- Any lines with dates (2024-01-15, etc.)
- Any error messages

### Step 3: Try Simplified Test

Create a test script to isolate the issue:

```python
# test_backtrader_simple.py
import pandas as pd
import numpy as np
from src.BacktestBacktrader import BacktestBacktraderML

# Create simple test data
dates = pd.date_range('2024-01-01', periods=100, freq='1H')
df_test = pd.DataFrame({
    'Timestamp': dates,
    'open': 50000 + np.random.randn(100) * 100,
    'high': 50100 + np.random.randn(100) * 100,
    'low': 49900 + np.random.randn(100) * 100,
    'close': 50000 + np.random.randn(100) * 100,
    'volume': 1000 + np.random.randn(100) * 100,
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
})

# Create dummy model that always predicts 1
class DummyModel:
    def predict(self, X):
        return np.ones(len(X))
    
    def predict_proba(self, X):
        # Return high probability for all
        return np.column_stack([np.zeros(len(X)), np.ones(len(X)) * 0.9])

# Create dummy scaler
class DummyScaler:
    def transform(self, X):
        return X

model = DummyModel()
scaler = DummyScaler()

# Run backtest
backtester = BacktestBacktraderML(
    initial_cash=100000.0,
    commission=0.001
)

results, trades = backtester.run_backtest(
    df=df_test,
    model=model,
    scaler=scaler,
    X_columns=['feature1', 'feature2'],
    probability_threshold=0.6,
    trailing_stop_pct=2.0,
    position_size_pct=0.9,
    plot=False,
    printlog=True
)

print(f"\nResults: {results['total_trades']} trades")
```

Run this and see if trades execute. If they do, the issue is with your data or model.

## Quick Fixes to Try

### Fix 1: Increase Initial Cash
```python
backtester = BacktestBacktraderML(
    initial_cash=1000000.0,  # 1 million
    commission=0.001
)
```

### Fix 2: Lower Probability Threshold
```python
probability_threshold=0.5,  # Lower from 0.6
```

### Fix 3: Disable Commission Temporarily
```python
backtester = BacktestBacktraderML(
    initial_cash=10000.0,
    commission=0.0  # No commission for testing
)
```

### Fix 4: Check Data Columns
```python
# Before running backtest, print columns:
print(f"DataFrame columns: {df_test_features.columns.tolist()}")
print(f"DataFrame shape: {df_test_features.shape}")
print(f"First row:\n{df_test_features.iloc[0]}")
```

## Expected Behavior After Fixes

You should see:

```
ML Signal Statistics:
  Total rows: 7183
  Buy signals (1): 292

Buy Signals Meeting Probability Thresholds:
  >= 0.60:    1 signals (  0.3%)

[... backtest runs with logs ...]

2024-01-15 Position Sizing: Cash=$10000.00, Use=90.0%, Amount=$9000.00, Price=$50000.00, Size=0.180000
2024-01-15 BUY CREATE, Price: 50000.00, Size: 0.18, Prob: 0.65
2024-01-15 BUY EXECUTED, Price: 50000.05, Cost: 9000.09, Comm: 9.00

[... more trading ...]

2024-01-20 SELL EXECUTED, Price: 51000.00, Cost: 9180.00, Comm: 9.18
2024-01-20 TRADE PROFIT, GROSS: 1000.00, NET: 981.82

Strategy Debug Info:
  Total buy signals (1): 292
  Entry attempts: 1
  Actual trades: 1  ← SUCCESS!
  Probability threshold: 0.6

BACKTEST RESULTS
================================================================================
Trades:
  Total Trades:           1
  Won Trades:             1
  Lost Trades:            0
  Win Rate:               100.00%
```

## Next Steps

1. **Run your backtest** with current fixes
2. **Copy the COMPLETE console output** 
3. **Share it** so we can see exactly what's happening
4. Look for:
   - "BUY CREATE" messages
   - "Order Canceled/Margin/Rejected" messages
   - "Position Sizing" messages
   - Any error messages

The log output will tell us exactly why orders aren't executing!
