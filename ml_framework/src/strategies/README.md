# Strategies Module

Trading strategies for backtesting with ML models.

## Overview

This module provides a base strategy class and concrete implementations for ML-based trading strategies with support for long/short positions and trailing stop loss.

## Classes

### BaseStrategy (Abstract)

Abstract base class that all strategies must inherit from.

**Key Methods:**
- `generate_signals()` - Generate trading signals from data
- `should_enter_long()` - Determine if should enter long position
- `should_enter_short()` - Determine if should enter short position
- `should_exit()` - Determine if should exit position
- `open_position()` - Open a new position
- `close_position()` - Close an existing position
- `get_statistics()` - Calculate strategy statistics

### MLStrategy

ML-based trading strategy with long/short positions.

**Strategy Logic:**
- Signal +1: Enter Long position
- Signal -1: Enter Short position  
- Exit on Nth bar (holding_period) OR trailing stop loss

**Features:**
- Long and short positions
- Fixed holding period exit
- Optional trailing stop loss
- Position tracking

## Quick Start

```python
from src.strategies import MLStrategy
from src.models_lib import RandomForestModel
from sklearn.preprocessing import StandardScaler

# Initialize strategy
strategy = MLStrategy(
    name='ML_Strategy',
    holding_period=15,  # Hold for 15 bars
    trailing_stop_pct=0.05,  # 5% trailing stop
    enable_trailing_stop=False  # Disabled for now
)

# Run backtest
results = strategy.backtest(
    df=test_df,
    model=trained_model,
    scaler=fitted_scaler,
    feature_cols=feature_columns,
    initial_capital=10000,
    position_size_pct=1.0,
    commission=0.001
)

# View results
print(f"Final Capital: ${results['final_capital']:,.2f}")
print(f"Total Trades: {len(results['trades'])}")
print(f"Win Rate: {results['strategy_stats']['win_rate']*100:.2f}%")
```

## MLStrategy Parameters

### Constructor Parameters

```python
MLStrategy(
    name='MLStrategy',
    holding_period=15,
    trailing_stop_pct=None,
    enable_trailing_stop=False
)
```

- **name** (str): Strategy name
- **holding_period** (int): Number of bars to hold position (same as FUTURE_BARS)
- **trailing_stop_pct** (float, optional): Trailing stop loss percentage (e.g., 0.05 = 5%)
- **enable_trailing_stop** (bool): Enable/disable trailing stop loss

### Backtest Parameters

```python
strategy.backtest(
    df,
    model,
    scaler,
    feature_cols,
    initial_capital=10000.0,
    position_size_pct=1.0,
    commission=0.001,
    price_col='close'
)
```

- **df** (DataFrame): OHLCV data with features
- **model**: Trained ML model with predict() method
- **scaler**: Fitted scaler with transform() method
- **feature_cols** (list): Feature column names
- **initial_capital** (float): Starting capital
- **position_size_pct** (float): Position size as percentage of capital (0.0-1.0)
- **commission** (float): Commission rate (e.g., 0.001 = 0.1%)
- **price_col** (str): Price column name

## Complete Example

```python
import pandas as pd
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
df_features = features_gen.create_target(
    df_features,
    target_type='classification',
    future_bars=15,
    threshold=0.02
)
df_features = df_features.dropna()

# 3. Split data
train_df, val_df, test_df = data_provider.split_data(df_features)
feature_cols = features_gen.get_feature_names()

# 4. Train model
X_train = train_df[feature_cols].values
y_train = train_df['target'].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = RandomForestModel(n_estimators=100, n_jobs=-1)
model.fit(X_train_scaled, y_train)

# 5. Create strategy
strategy = MLStrategy(
    name='BTC_ML_Strategy',
    holding_period=15,
    trailing_stop_pct=0.05,
    enable_trailing_stop=False  # Disabled for now
)

# 6. Run backtest
results = strategy.backtest(
    df=test_df,
    model=model,
    scaler=scaler,
    feature_cols=feature_cols,
    initial_capital=10000,
    position_size_pct=1.0,
    commission=0.001
)

# 7. Analyze results
print("\n" + "="*60)
print("BACKTEST RESULTS")
print("="*60)
print(f"Initial Capital: ${results['initial_capital']:,.2f}")
print(f"Final Capital: ${results['final_capital']:,.2f}")
print(f"Total Return: {(results['final_capital']/results['initial_capital']-1)*100:.2f}%")
print(f"\nTotal Trades: {len(results['trades'])}")

stats = results['strategy_stats']
print(f"Winning Trades: {stats['winning_trades']}")
print(f"Losing Trades: {stats['losing_trades']}")
print(f"Win Rate: {stats['win_rate']*100:.2f}%")
print(f"Avg P&L: ${stats['avg_pnl']:.2f}")
print(f"Avg P&L %: {stats['avg_pnl_pct']*100:.2f}%")
print(f"Avg Bars Held: {stats['avg_bars_held']:.1f}")
```

## Strategy Logic Details

### Signal Generation

The strategy converts ML model predictions to trading signals:

```python
# Model predictions: 0 or 1
predictions = model.predict(X_scaled)

# Convert to signals:
# 1 (buy prediction) -> +1 (long signal)
# 0 (sell prediction) -> -1 (short signal)
```

### Entry Logic

**Long Entry:**
- Signal = +1
- No open position
- Enter long at current price

**Short Entry:**
- Signal = -1
- No open position
- Enter short at current price

### Exit Logic

**Exit Condition 1: Holding Period**
- Exit after N bars (holding_period)
- Ensures position doesn't stay open indefinitely
- Default: 15 bars (same as FUTURE_BARS)

**Exit Condition 2: Trailing Stop Loss (Optional)**

For Long Positions:
- Track highest price since entry
- Exit if price drops by trailing_stop_pct from highest
- Example: 5% trailing stop, highest = $100, exit at $95

For Short Positions:
- Track lowest price since entry
- Exit if price rises by trailing_stop_pct from lowest
- Example: 5% trailing stop, lowest = $100, exit at $105

## Position Tracking

Each position tracks:
- **type**: 'long' or 'short'
- **entry_bar**: Bar index of entry
- **entry_price**: Entry price
- **size**: Position size (shares)
- **highest_price**: Highest price since entry (for long trailing stop)
- **lowest_price**: Lowest price since entry (for short trailing stop)
- **exit_bar**: Bar index of exit
- **exit_price**: Exit price
- **exit_reason**: Reason for exit ('holding_period', 'trailing_stop_long', 'trailing_stop_short', 'end_of_data')
- **bars_held**: Number of bars held
- **pnl**: Profit/loss in dollars
- **pnl_pct**: Profit/loss as percentage

## Statistics

The strategy calculates comprehensive statistics:

```python
stats = strategy.get_statistics()

# Available metrics:
stats['total_trades']        # Total number of trades
stats['winning_trades']      # Number of winning trades
stats['losing_trades']       # Number of losing trades
stats['win_rate']           # Win rate (0.0 to 1.0)
stats['avg_pnl']            # Average P&L per trade ($)
stats['avg_pnl_pct']        # Average P&L per trade (%)
stats['total_pnl']          # Total P&L ($)
stats['avg_bars_held']      # Average holding period
```

## Trailing Stop Loss

### How It Works

**For Long Positions:**
1. Track highest price since entry
2. Calculate stop price = highest_price * (1 - trailing_stop_pct)
3. Exit if current_price <= stop_price

**For Short Positions:**
1. Track lowest price since entry
2. Calculate stop price = lowest_price * (1 + trailing_stop_pct)
3. Exit if current_price >= stop_price

### Example

Long position with 5% trailing stop:
```
Entry: $100
Price moves to $110 (highest = $110, stop = $104.50)
Price moves to $120 (highest = $120, stop = $114.00)
Price drops to $114 -> EXIT (trailing stop hit)
```

### Enable/Disable

```python
# Disabled (default)
strategy = MLStrategy(
    holding_period=15,
    trailing_stop_pct=0.05,
    enable_trailing_stop=False
)

# Enabled
strategy = MLStrategy(
    holding_period=15,
    trailing_stop_pct=0.05,
    enable_trailing_stop=True
)
```

## Creating Custom Strategies

Inherit from `BaseStrategy` and implement required methods:

```python
from src.strategies import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    def __init__(self, **params):
        super().__init__(name='MyStrategy')
        # Your initialization
    
    def generate_signals(self, df, **kwargs):
        # Generate signals from data
        # Return Series with +1 (long), -1 (short), 0 (no signal)
        pass
    
    def should_enter_long(self, signal, bar_idx, **kwargs):
        # Return True if should enter long
        pass
    
    def should_enter_short(self, signal, bar_idx, **kwargs):
        # Return True if should enter short
        pass
    
    def should_exit(self, position, current_bar, current_price, **kwargs):
        # Return (should_exit: bool, reason: str)
        pass
```

## Best Practices

1. **Set Appropriate Holding Period**
   - Match FUTURE_BARS used in target creation
   - Typical: 5-20 bars

2. **Use Trailing Stop Carefully**
   - Start with it disabled
   - Test with different percentages (3-10%)
   - Monitor impact on win rate and returns

3. **Position Sizing**
   - Start with 100% (position_size_pct=1.0)
   - Consider 50% for more conservative approach
   - Never exceed 100%

4. **Commission Rates**
   - Use realistic rates (0.001 = 0.1%)
   - Higher for crypto, lower for stocks
   - Include slippage if needed

5. **Backtest on Unseen Data**
   - Always use test set
   - Never backtest on training data
   - Validate on multiple time periods

## Troubleshooting

### No trades executed
- Check model predictions (should be 0 or 1)
- Verify features are scaled
- Ensure enough data in test set
- Check position_size_pct > 0

### Too many trades
- Increase holding_period
- Adjust model threshold
- Filter signals

### Low win rate
- Review model performance
- Adjust trailing stop percentage
- Check commission impact
- Validate signal quality

## See Also

- [BaseStrategy](base_strategy.py) - Base class implementation
- [MLStrategy](ml_strategy.py) - ML strategy implementation
- [Backtesting Module](../backtesting/README.md) - Backtesting backends

---

**Last Updated:** 2024-11-24  
**Version:** 1.0.0
