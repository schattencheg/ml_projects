# MLStrategy + FixedBarsCountRiskManager Integration Guide

## Overview

This guide shows how to integrate `FixedBarsCountRiskManager` into `MLStrategy` to replace the existing trailing stop logic with a clean, modular risk management system.

---

## Changes Required

### 1. Add Import

**Location:** Top of `src/strategies/ml_strategy.py`

**Before:**
```python
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from src.strategies.base_strategy import BaseStrategy
```

**After:**
```python
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from src.strategies.base_strategy import BaseStrategy
from src.risk_management import FixedBarsCountRiskManager
```

---

### 2. Update `__init__` Method

**Location:** `MLStrategy.__init__` (lines ~26-43)

**Before:**
```python
def __init__(self,
             name: str = "MLStrategy",
             holding_period: int = 15,
             trailing_stop_pct: Optional[float] = None,
             enable_trailing_stop: bool = False):
    """
    Initialize ML strategy.
    
    Args:
        name: Strategy name
        holding_period: Number of bars to hold position (same as FUTURE_BARS)
        trailing_stop_pct: Trailing stop loss percentage (e.g., 0.05 = 5%)
        enable_trailing_stop: Enable/disable trailing stop loss
    """
    super().__init__(name)
    self.holding_period = holding_period
    self.trailing_stop_pct = trailing_stop_pct
    self.enable_trailing_stop = enable_trailing_stop
```

**After:**
```python
def __init__(self,
             name: str = "MLStrategy",
             holding_period: int = 15):
    """
    Initialize ML strategy with FixedBarsCountRiskManager.
    
    Args:
        name: Strategy name
        holding_period: Number of bars to hold position (same as FUTURE_BARS)
    """
    super().__init__(name)
    self.holding_period = holding_period
    
    # Initialize risk manager (replaces trailing stop logic)
    self.risk_manager = FixedBarsCountRiskManager(
        bars_to_hold=holding_period,
        name=f"{name}_RiskManager"
    )
```

---

### 3. Update `should_exit` Method

**Location:** `MLStrategy.should_exit` (lines ~124-164)

**Before:**
```python
def should_exit(self, position: Dict[str, Any], current_bar: int,
               current_price: float, **kwargs) -> tuple[bool, str]:
    """
    Determine if should exit position.
    
    Exit Conditions:
    1. Holding period reached (Nth bar)
    2. Trailing stop loss hit (if enabled)
    
    Args:
        position: Position dictionary
        current_bar: Current bar index
        current_price: Current price
        **kwargs: Additional parameters
        
    Returns:
        Tuple of (should_exit: bool, reason: str)
    """
    bars_held = current_bar - position['entry_bar']
    
    # Exit condition 1: Holding period reached
    if bars_held >= self.holding_period:
        return True, 'holding_period'
    
    # Exit condition 2: Trailing stop loss (if enabled)
    if self.enable_trailing_stop and self.trailing_stop_pct is not None:
        if position['type'] == 'long':
            # Long position: exit if price drops from highest by trailing_stop_pct
            if position['highest_price'] is not None:
                stop_price = position['highest_price'] * (1 - self.trailing_stop_pct)
                if current_price <= stop_price:
                    return True, 'trailing_stop_long'
        
        else:  # short position
            # Short position: exit if price rises from lowest by trailing_stop_pct
            if position['lowest_price'] is not None:
                stop_price = position['lowest_price'] * (1 + self.trailing_stop_pct)
                if current_price >= stop_price:
                    return True, 'trailing_stop_short'
    
    return False, ''
```

**After:**
```python
def should_exit(self, position: Dict[str, Any], current_bar: int,
               current_price: float, **kwargs) -> tuple[bool, str]:
    """
    Determine if should exit position using FixedBarsCountRiskManager.
    
    Exit Conditions:
    1. Holding period reached (Nth bar) - handled by risk manager
    2. End of data - handled by risk manager
    
    Args:
        position: Position dictionary
        current_bar: Current bar index
        current_price: Current price
        **kwargs: Additional parameters (must include 'df')
        
    Returns:
        Tuple of (should_exit: bool, reason: str)
    """
    # Get DataFrame from kwargs
    df = kwargs.get('df')
    if df is None:
        raise ValueError("DataFrame 'df' must be provided in kwargs for risk manager")
    
    # Use risk manager to determine exit
    should_exit, exit_reason = self.risk_manager.should_exit(
        position=position,
        current_bar=current_bar,
        current_price=current_price,
        df=df
    )
    
    return should_exit, exit_reason
```

---

### 4. Update `backtest` Method - Exit Check

**Location:** `MLStrategy.backtest` (lines ~210-215)

**Before:**
```python
# Check exit conditions for open positions
positions_to_close = []
for position in self.positions:
    should_exit, reason = self.should_exit(position, i, current_price)
    if should_exit:
        positions_to_close.append((position, reason))
```

**After:**
```python
# Check exit conditions for open positions
positions_to_close = []
for position in self.positions:
    should_exit, reason = self.should_exit(position, i, current_price, df=df)
    if should_exit:
        positions_to_close.append((position, reason))
```

---

### 5. Update `backtest` Method - Long Entry

**Location:** `MLStrategy.backtest` (lines ~241-243)

**Before:**
```python
if shares > 0:
    self.open_position('long', i, current_price, shares)
    capital -= position_value
```

**After:**
```python
if shares > 0:
    self.open_position('long', i, current_price, shares)
    # Notify risk manager of new position
    self.risk_manager.on_entry(
        position_id=f'pos_{i}',
        entry_bar=i,
        entry_price=current_price,
        shares=shares,
        entry_idx=i
    )
    capital -= position_value
```

---

### 6. Update `backtest` Method - Short Entry

**Location:** `MLStrategy.backtest` (lines ~251-254)

**Before:**
```python
if shares > 0:
    self.open_position('short', i, current_price, shares)
    # For short: we receive cash from selling borrowed shares
    capital += position_value - commission_cost
```

**After:**
```python
if shares > 0:
    self.open_position('short', i, current_price, shares)
    # Notify risk manager of new position (negative shares for short)
    self.risk_manager.on_entry(
        position_id=f'pos_{i}',
        entry_bar=i,
        entry_price=current_price,
        shares=-shares,  # Negative for short positions
        entry_idx=i
    )
    # For short: we receive cash from selling borrowed shares
    capital += position_value - commission_cost
```

---

### 7. Update `backtest` Method - Position Close

**Location:** `MLStrategy.backtest` (lines ~218-221)

**Before:**
```python
# Close positions
for position, reason in positions_to_close:
    closed_pos = self.close_position(position, i, current_price, reason)
    
    # Update capital
```

**After:**
```python
# Close positions
for position, reason in positions_to_close:
    closed_pos = self.close_position(position, i, current_price, reason)
    
    # Notify risk manager of position exit
    self.risk_manager.on_exit(f'pos_{closed_pos["entry_bar"]}')
    
    # Update capital
```

---

### 8. Update `get_config` Method

**Location:** `MLStrategy.get_config` (lines ~295-307)

**Before:**
```python
def get_config(self) -> Dict[str, Any]:
    """
    Get strategy configuration.
    
    Returns:
        Dictionary with configuration
    """
    return {
        'name': self.name,
        'holding_period': self.holding_period,
        'trailing_stop_pct': self.trailing_stop_pct,
        'enable_trailing_stop': self.enable_trailing_stop
    }
```

**After:**
```python
def get_config(self) -> Dict[str, Any]:
    """
    Get strategy configuration.
    
    Returns:
        Dictionary with configuration
    """
    return {
        'name': self.name,
        'holding_period': self.holding_period,
        'risk_manager': self.risk_manager.get_info()
    }
```

---

### 9. Update `__repr__` Method

**Location:** `MLStrategy.__repr__` (lines ~309-312)

**Before:**
```python
def __repr__(self):
    return (f"MLStrategy(name='{self.name}', "
            f"holding_period={self.holding_period}, "
            f"trailing_stop={'enabled' if self.enable_trailing_stop else 'disabled'})")
```

**After:**
```python
def __repr__(self):
    return (f"MLStrategy(name='{self.name}', "
            f"holding_period={self.holding_period}, "
            f"risk_manager={self.risk_manager})")
```

---

## Benefits of Integration

### 1. **Cleaner Code**
- Removed ~40 lines of trailing stop logic
- Single responsibility: strategy focuses on signals, risk manager handles exits
- Easier to understand and maintain

### 2. **Modular Design**
- Risk management logic separated from strategy logic
- Easy to swap risk managers (e.g., add StopLossRiskManager later)
- Testable independently

### 3. **Consistent Exit Reasons**
- Exit reasons now standardized: `'fixed_bars'`, `'end_of_data'`
- Better for visualization and analysis
- Matches backtest visualization expectations

### 4. **Position Tracking**
- Risk manager tracks all active positions
- Can query bars held, bars remaining
- Better monitoring and debugging

---

## Usage Example

### Before Integration

```python
from src.strategies import MLStrategy

# Old way with trailing stop
strategy = MLStrategy(
    name='ML_Strategy',
    holding_period=10,
    trailing_stop_pct=0.05,
    enable_trailing_stop=True
)
```

### After Integration

```python
from src.strategies import MLStrategy

# New way with risk manager
strategy = MLStrategy(
    name='ML_Strategy',
    holding_period=10
)

# Risk manager automatically created
print(strategy.risk_manager)
# Output: FixedBarsCountRiskManager(bars_to_hold=10, name='ML_Strategy_RiskManager')
```

---

## Testing

### Test the Integration

```python
from src.strategies import MLStrategy
import pandas as pd
import numpy as np

# Create strategy
strategy = MLStrategy(name='Test', holding_period=5)

# Verify risk manager
assert strategy.risk_manager is not None
assert strategy.risk_manager.bars_to_hold == 5

# Test should_exit
position = {
    'entry_bar': 10,
    'entry_price': 100.0,
    'shares': 1.0,
    'entry_idx': 10,
    'type': 'long'
}

df = pd.DataFrame({'close': np.random.randn(100)})

# Should not exit at bar 12 (2 bars held)
should_exit, reason = strategy.should_exit(position, 12, 105.0, df=df)
assert should_exit == False

# Should exit at bar 15 (5 bars held)
should_exit, reason = strategy.should_exit(position, 15, 105.0, df=df)
assert should_exit == True
assert reason == 'fixed_bars'

print("✓ All tests passed!")
```

---

## Migration Notes

### Breaking Changes

1. **Constructor signature changed**
   - Removed: `trailing_stop_pct`, `enable_trailing_stop`
   - Simplified to just `name` and `holding_period`

2. **Exit reasons changed**
   - Old: `'holding_period'`, `'trailing_stop_long'`, `'trailing_stop_short'`
   - New: `'fixed_bars'`, `'end_of_data'`

3. **`should_exit` requires `df` in kwargs**
   - Must pass `df=df` when calling `should_exit()`
   - Risk manager needs DataFrame to check end of data

### Non-Breaking Changes

- All other methods remain the same
- Backtest interface unchanged
- Position tracking still works
- Visualization still works

---

## Future Enhancements

### Add More Risk Managers

```python
from src.risk_management import (
    FixedBarsCountRiskManager,
    StopLossRiskManager,  # Future
    TakeProfitRiskManager,  # Future
    TrailingStopRiskManager  # Future
)

# Allow user to choose risk manager
strategy = MLStrategy(
    name='ML_Strategy',
    holding_period=10,
    risk_manager=TrailingStopRiskManager(trailing_pct=0.05)
)
```

### Composite Risk Manager

```python
# Combine multiple risk managers
risk_manager = CompositeRiskManager([
    FixedBarsCountRiskManager(bars_to_hold=10),
    StopLossRiskManager(stop_loss_pct=0.02),
    TakeProfitRiskManager(take_profit_pct=0.05)
])

strategy = MLStrategy(
    name='ML_Strategy',
    risk_manager=risk_manager
)
```

---

## Summary

**Changes:**
- ✅ Import `FixedBarsCountRiskManager`
- ✅ Simplify `__init__` (remove trailing stop params)
- ✅ Replace `should_exit` logic with risk manager call
- ✅ Pass `df=df` to `should_exit` calls
- ✅ Notify risk manager on entry/exit
- ✅ Update `get_config` and `__repr__`

**Benefits:**
- ✅ Cleaner, more modular code
- ✅ Easier to test and maintain
- ✅ Consistent exit reasons
- ✅ Better position tracking

**Result:**
- 🎯 Strategy focuses on signals
- 🎯 Risk manager handles exits
- 🎯 Clean separation of concerns

---

**Status: ✅ INTEGRATION GUIDE COMPLETE**

Follow these steps to integrate `FixedBarsCountRiskManager` into `MLStrategy`!
