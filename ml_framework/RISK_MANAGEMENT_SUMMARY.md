# Risk Management System - Quick Reference

## Overview

Created a modular risk management framework for backtesting strategies with a base class and fixed bars count implementation.

---

## Files Created

1. **`src/risk_management/__init__.py`** - Module exports
2. **`src/risk_management/base_risk_manager.py`** (~160 lines) - Abstract base class
3. **`src/risk_management/fixed_bars_risk_manager.py`** (~200 lines) - Fixed bars implementation

---

## Quick Start

### Import

```python
from src.risk_management import FixedBarsCountRiskManager
```

### Create Risk Manager

```python
# Exit after 5 bars (no stop loss, no take profit)
risk_manager = FixedBarsCountRiskManager(bars_to_hold=5)
```

### Use in Backtest

```python
# Enter position
position = {
    'entry_bar': 10,
    'entry_price': 45000.0,
    'shares': 0.5,
    'entry_idx': 10
}
risk_manager.on_entry('pos_1', 10, 45000.0, 0.5, 10)

# Check if should exit
should_exit, exit_reason = risk_manager.should_exit(
    position=position,
    current_bar=15,
    current_price=47000.0,
    df=df
)

if should_exit:
    pnl = risk_manager.calculate_pnl(45000.0, 47000.0, 0.5)
    risk_manager.on_exit('pos_1')
```

---

## BaseRiskManager API

### Abstract Methods

**`should_exit(position, current_bar, current_price, df) -> (bool, str)`**
- Returns: `(should_exit, exit_reason)`
- Must be implemented by subclasses

### Utility Methods

**`on_entry(position_id, entry_bar, entry_price, shares, entry_idx)`**
- Track new position

**`on_exit(position_id)`**
- Clean up closed position

**`get_position_size(capital, price, position_size_pct=1.0) -> float`**
- Calculate position size

**`calculate_pnl(entry_price, exit_price, shares) -> float`**
- Calculate profit/loss

**`reset()`**
- Reset risk manager state

**`get_info() -> dict`**
- Get risk manager information

---

## FixedBarsCountRiskManager

### Features

- ✅ Exits after N bars
- ✅ No stop loss
- ✅ No take profit
- ✅ Simple and predictable
- ✅ Perfect for ML models

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bars_to_hold` | int | 5 | Number of bars to hold |
| `name` | str | 'FixedBarsCount' | Risk manager name |

### Methods

**`should_exit(position, current_bar, current_price, df) -> (bool, str)`**
- Returns `(True, 'fixed_bars')` after N bars
- Returns `(True, 'end_of_data')` at end of data

**`get_bars_held(position, current_bar) -> int`**
- Get number of bars held

**`get_bars_remaining(position, current_bar) -> int`**
- Get bars remaining before exit

---

## Exit Reasons

| Reason | Description |
|--------|-------------|
| `'fixed_bars'` | N bars reached |
| `'end_of_data'` | End of backtest data |
| `'signal'` | Strategy exit signal |
| `'stop_loss'` | Stop loss hit |
| `'take_profit'` | Take profit hit |
| `'trailing_stop'` | Trailing stop hit |

---

## Use Cases

### 1. ML Model Validation

```python
# Model predicts 10-bar returns
risk_manager = FixedBarsCountRiskManager(bars_to_hold=10)
```

### 2. Holding Period Optimization

```python
for bars in [3, 5, 10, 20]:
    risk_manager = FixedBarsCountRiskManager(bars_to_hold=bars)
    # Run backtest and compare results
```

### 3. Eliminate SL/TP Complexity

```python
# No parameters to optimize
risk_manager = FixedBarsCountRiskManager(bars_to_hold=5)
```

---

## Integration Example

```python
from src.backtesting import BacktestNoLib
from src.risk_management import FixedBarsCountRiskManager

# Create backtest
backtest = BacktestNoLib(initial_capital=10000)

# Create risk manager
risk_manager = FixedBarsCountRiskManager(bars_to_hold=5)

# In backtest loop
for i in range(len(df)):
    if prediction == 1 and not in_position:
        # Enter
        position = {...}
        risk_manager.on_entry('pos_1', i, price, shares, i)
        in_position = True
    
    elif in_position:
        # Check exit
        should_exit, reason = risk_manager.should_exit(
            position, i, df.iloc[i]['close'], df
        )
        
        if should_exit:
            # Exit and record trade
            pnl = risk_manager.calculate_pnl(...)
            trades.append({..., 'exit_reason': reason})
            risk_manager.on_exit('pos_1')
            in_position = False
```

---

## Testing

### Run Example

```bash
python -m src.risk_management.fixed_bars_risk_manager
```

### Output

```
======================================================================
FIXED BARS COUNT RISK MANAGER EXAMPLE
======================================================================

Risk Manager: FixedBarsCountRiskManager(bars_to_hold=5, name='FixedBarsCount')

Position entered at bar 10, price $100.00
Holding for 5 bars

Bar | Bars Held | Bars Remaining | Should Exit | Exit Reason
----------------------------------------------------------------------
 10 |         0 |              5 | No          | 
 11 |         1 |              4 | No          | 
 12 |         2 |              3 | No          | 
 13 |         3 |              2 | No          | 
 14 |         4 |              1 | No          | 
 15 |         5 |              0 | Yes         | fixed_bars

✓ Position exited at bar 15, price $102.13
  PnL: $0.56
  Return: 0.55%
```

---

## Benefits

### Simplicity
- No complex parameters
- Easy to understand
- Predictable behavior

### ML Alignment
- Matches prediction horizon
- Tests actual predictions
- No SL/TP interference

### Fair Comparison
- Same holding period
- Eliminates timing luck
- Pure signal quality

### Reduced Overfitting
- Fewer parameters
- More robust
- Simpler strategy

---

## Future Extensions

### Planned Risk Managers

1. **StopLossRiskManager** - Fixed % stop loss
2. **TakeProfitRiskManager** - Fixed % take profit
3. **TrailingStopRiskManager** - Trailing stop
4. **ATRStopRiskManager** - ATR-based stops
5. **CompositeRiskManager** - Combine multiple strategies

### Custom Implementation

```python
from src.risk_management import BaseRiskManager

class MyRiskManager(BaseRiskManager):
    def should_exit(self, position, current_bar, current_price, df):
        # Your logic here
        return should_exit, exit_reason
```

---

## Summary

**Created:**
- ✅ `BaseRiskManager` - Abstract base class
- ✅ `FixedBarsCountRiskManager` - Fixed bars implementation
- ✅ Complete documentation and examples

**Features:**
- ✅ Modular framework
- ✅ Position tracking
- ✅ PnL calculation
- ✅ Easy integration

**Usage:**
```python
risk_manager = FixedBarsCountRiskManager(bars_to_hold=5)
should_exit, reason = risk_manager.should_exit(position, bar, price, df)
```

**Perfect for:**
- ML model validation
- Holding period optimization
- Eliminating SL/TP complexity
- Fair strategy comparison

---

**Status: ✅ READY TO USE**

Risk management system tested and ready for integration with all backtest backends!
