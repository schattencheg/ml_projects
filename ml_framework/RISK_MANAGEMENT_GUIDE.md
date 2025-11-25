## Risk Management System

Comprehensive risk management framework for backtesting strategies with support for various exit strategies.

---

## Overview

The risk management system provides a flexible framework for controlling when and how positions are exited in backtesting. It separates risk management logic from strategy logic, making it easy to test different exit strategies.

### Key Features

- **Modular Design**: Base class with pluggable implementations
- **Multiple Strategies**: Fixed bars, stop loss, take profit, trailing stops
- **Position Tracking**: Automatic tracking of active positions
- **PnL Calculation**: Built-in profit/loss calculations
- **Easy Integration**: Works with all backtest backends

---

## Architecture

```
BaseRiskManager (Abstract)
    ├─ should_exit()        # Determine if position should be closed
    ├─ on_entry()           # Track new position
    ├─ on_exit()            # Clean up closed position
    ├─ get_position_size()  # Calculate position sizing
    └─ calculate_pnl()      # Calculate profit/loss

FixedBarsCountRiskManager
    └─ Exits after N bars (no SL/TP)
```

---

## BaseRiskManager

Abstract base class for all risk managers.

### Methods

#### `should_exit(position, current_bar, current_price, df) -> (bool, str)`

Determine if a position should be exited.

**Parameters:**
- `position` (dict): Position information
  - `entry_bar` (int): Bar index at entry
  - `entry_price` (float): Entry price
  - `shares` (float): Number of shares
  - `entry_idx` (int): DataFrame index at entry
- `current_bar` (int): Current bar index
- `current_price` (float): Current price
- `df` (DataFrame): Market data

**Returns:**
- `should_exit` (bool): True if position should be closed
- `exit_reason` (str): Reason for exit

**Exit Reasons:**
- `'signal'` - Strategy signal to exit
- `'stop_loss'` - Stop loss hit
- `'take_profit'` - Take profit hit
- `'fixed_bars'` - Fixed bar count reached
- `'trailing_stop'` - Trailing stop hit
- `'end_of_data'` - End of backtest data

#### `on_entry(position_id, entry_bar, entry_price, shares, entry_idx)`

Called when a new position is entered.

#### `on_exit(position_id)`

Called when a position is exited.

#### `get_position_size(capital, price, position_size_pct=1.0) -> float`

Calculate position size based on available capital.

#### `calculate_pnl(entry_price, exit_price, shares) -> float`

Calculate profit/loss for a position.

---

## FixedBarsCountRiskManager

Exits positions after a fixed number of bars. No stop loss or take profit.

### Features

✅ **Simple and Predictable** - Always holds for N bars  
✅ **No Optimization Complexity** - No SL/TP parameters to tune  
✅ **Perfect for ML Models** - Matches prediction horizon  
✅ **Eliminates Noise** - No premature exits from volatility  

### Usage

```python
from src.risk_management import FixedBarsCountRiskManager

# Create risk manager
risk_manager = FixedBarsCountRiskManager(
    bars_to_hold=5,  # Hold for 5 bars
    name='5-Bar Hold'
)

# Check if should exit
position = {
    'entry_bar': 10,
    'entry_price': 45000.0,
    'shares': 0.5,
    'entry_idx': 10
}

should_exit, exit_reason = risk_manager.should_exit(
    position=position,
    current_bar=15,  # 5 bars later
    current_price=47000.0,
    df=df
)

print(f"Should exit: {should_exit}")  # True
print(f"Reason: {exit_reason}")       # 'fixed_bars'
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bars_to_hold` | int | 5 | Number of bars to hold position |
| `name` | str | 'FixedBarsCount' | Name of risk manager |

### Methods

#### `should_exit(position, current_bar, current_price, df) -> (bool, str)`

Returns `(True, 'fixed_bars')` when N bars have passed.

#### `get_bars_held(position, current_bar) -> int`

Get number of bars position has been held.

#### `get_bars_remaining(position, current_bar) -> int`

Get number of bars remaining before exit.

---

## Integration with Backtests

### Example: BacktestNoLib Integration

```python
from src.backtesting import BacktestNoLib
from src.risk_management import FixedBarsCountRiskManager

# Create backtest with risk manager
backtest = BacktestNoLib(
    initial_capital=10000,
    commission=0.001,
    position_size=0.95
)

# Create risk manager
risk_manager = FixedBarsCountRiskManager(bars_to_hold=10)

# In backtest loop
for i in range(len(df)):
    # ... get prediction ...
    
    if prediction == 1 and not in_position:
        # Enter position
        entry_bar = i
        entry_price = df.iloc[i]['close']
        shares = capital * 0.95 / entry_price
        
        position = {
            'entry_bar': entry_bar,
            'entry_price': entry_price,
            'shares': shares,
            'entry_idx': i
        }
        
        risk_manager.on_entry('pos_1', entry_bar, entry_price, shares, i)
        in_position = True
    
    elif in_position:
        # Check if should exit
        current_price = df.iloc[i]['close']
        should_exit, exit_reason = risk_manager.should_exit(
            position, i, current_price, df
        )
        
        if should_exit:
            # Exit position
            pnl = risk_manager.calculate_pnl(
                position['entry_price'],
                current_price,
                position['shares']
            )
            
            # Record trade
            trades.append({
                'entry_idx': position['entry_idx'],
                'exit_idx': i,
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'shares': position['shares'],
                'pnl': pnl,
                'exit_reason': exit_reason
            })
            
            risk_manager.on_exit('pos_1')
            in_position = False
```

---

## Use Cases

### 1. ML Model Validation

Match holding period to model's prediction horizon:

```python
# Model predicts 5-bar future returns
risk_manager = FixedBarsCountRiskManager(bars_to_hold=5)
```

### 2. Holding Period Optimization

Test different holding periods:

```python
results = {}

for bars in [3, 5, 10, 20]:
    risk_manager = FixedBarsCountRiskManager(bars_to_hold=bars)
    # Run backtest...
    results[bars] = sharpe_ratio

# Find optimal holding period
best_bars = max(results, key=results.get)
```

### 3. Eliminate SL/TP Complexity

Focus on entry signals without SL/TP optimization:

```python
# No stop loss or take profit to tune
risk_manager = FixedBarsCountRiskManager(bars_to_hold=10)
```

---

## Comparison with Other Strategies

| Strategy | Pros | Cons | Best For |
|----------|------|------|----------|
| **Fixed Bars** | Simple, predictable, matches ML horizon | May miss better exits | ML model validation |
| **Stop Loss** | Limits losses | May exit prematurely | Risk control |
| **Take Profit** | Locks in gains | May exit too early | Profit taking |
| **Trailing Stop** | Captures trends | Complex to optimize | Trend following |

---

## Advanced Usage

### Custom Risk Manager

Create your own risk manager:

```python
from src.risk_management import BaseRiskManager

class CustomRiskManager(BaseRiskManager):
    def __init__(self, stop_loss_pct=0.02, take_profit_pct=0.05):
        super().__init__(name='Custom')
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
    
    def should_exit(self, position, current_bar, current_price, df):
        entry_price = position['entry_price']
        shares = position['shares']
        
        # Calculate return
        if shares > 0:  # Long
            return_pct = (current_price - entry_price) / entry_price
        else:  # Short
            return_pct = (entry_price - current_price) / entry_price
        
        # Check stop loss
        if return_pct <= -self.stop_loss_pct:
            return True, 'stop_loss'
        
        # Check take profit
        if return_pct >= self.take_profit_pct:
            return True, 'take_profit'
        
        # Check end of data
        if current_bar >= len(df) - 1:
            return True, 'end_of_data'
        
        return False, ''
```

### Multiple Position Tracking

Track multiple positions simultaneously:

```python
risk_manager = FixedBarsCountRiskManager(bars_to_hold=5)

# Enter multiple positions
risk_manager.on_entry('pos_1', 10, 45000, 0.5, 10)
risk_manager.on_entry('pos_2', 15, 46000, 0.3, 15)

# Check each position
for pos_id, position in risk_manager.active_positions.items():
    should_exit, reason = risk_manager.should_exit(
        position, current_bar, current_price, df
    )
    if should_exit:
        risk_manager.on_exit(pos_id)
```

---

## Testing

### Run Example

```bash
python -m src.risk_management.fixed_bars_risk_manager
```

### Expected Output

```
======================================================================
FIXED BARS COUNT RISK MANAGER EXAMPLE
======================================================================

Risk Manager: FixedBarsCountRiskManager(bars_to_hold=5, name='FixedBarsCount')
Info: {'name': 'FixedBarsCount', 'type': 'FixedBarsCountRiskManager', 
       'active_positions': 0, 'bars_to_hold': 5, 'strategy': 'Fixed bars count exit', 
       'stop_loss': 'None', 'take_profit': 'None'}

======================================================================
SIMULATING POSITION
======================================================================

Position entered at bar 10, price $100.00
Holding for 5 bars

----------------------------------------------------------------------
Bar | Bars Held | Bars Remaining | Should Exit | Exit Reason
----------------------------------------------------------------------
 10 |         0 |              5 | No          | 
 11 |         1 |              4 | No          | 
 12 |         2 |              3 | No          | 
 13 |         3 |              2 | No          | 
 14 |         4 |              1 | No          | 
 15 |         5 |              0 | Yes         | fixed_bars

✓ Position exited at bar 15, price $102.50
  PnL: $2.50
  Return: 2.50%

======================================================================
EXAMPLE COMPLETE
======================================================================
```

---

## Benefits

### 1. **Simplicity**
✅ No complex parameters to optimize  
✅ Easy to understand and explain  
✅ Predictable behavior  

### 2. **ML Model Alignment**
✅ Matches prediction horizon exactly  
✅ Tests model's actual predictions  
✅ No interference from SL/TP  

### 3. **Fair Comparison**
✅ Same holding period for all trades  
✅ Eliminates timing luck  
✅ Pure signal quality assessment  

### 4. **Reduced Overfitting**
✅ Fewer parameters to overfit  
✅ More robust out-of-sample  
✅ Simpler strategy  

---

## Future Risk Managers

### Planned Implementations

1. **StopLossRiskManager** - Fixed stop loss percentage
2. **TakeProfitRiskManager** - Fixed take profit percentage
3. **TrailingStopRiskManager** - Trailing stop loss
4. **ATRStopRiskManager** - ATR-based stops
5. **TimeBasedRiskManager** - Exit at specific times
6. **VolatilityAdjustedRiskManager** - Adjust based on volatility

---

## Summary

**Created:**
- `src/risk_management/base_risk_manager.py` - Abstract base class
- `src/risk_management/fixed_bars_risk_manager.py` - Fixed bars implementation
- `src/risk_management/__init__.py` - Module exports

**Features:**
- ✅ Modular risk management framework
- ✅ Fixed bars count strategy (no SL/TP)
- ✅ Position tracking
- ✅ PnL calculation
- ✅ Easy integration with backtests

**Usage:**
```python
from src.risk_management import FixedBarsCountRiskManager

risk_manager = FixedBarsCountRiskManager(bars_to_hold=5)
should_exit, reason = risk_manager.should_exit(position, bar, price, df)
```

**Perfect for:**
- ML model validation
- Holding period optimization
- Eliminating SL/TP complexity
- Fair strategy comparison

---

**Status: ✅ COMPLETE**

Risk management system ready to use with all backtest backends!
