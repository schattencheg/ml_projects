# ML Backtesting Guide with Trailing Stop Loss

This guide explains how to use the ML backtesting framework to test trading strategies based on machine learning model predictions with trailing stop loss functionality.

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [MLBacktester Class](#mlbacktester-class)
4. [Features](#features)
5. [Examples](#examples)
6. [Best Practices](#best-practices)

---

## Overview

The `MLBacktester` class allows you to backtest trading strategies that use ML model predictions as entry/exit signals. It includes:

- **Trailing Stop Loss**: Dynamically adjusts stop loss as price moves in your favor
- **Take Profit**: Optional fixed take profit target
- **Position Sizing**: Control how much capital to use per trade
- **Probability Threshold**: Only enter trades when model confidence is high enough
- **Commission & Slippage**: Realistic trading costs
- **Max Holding Period**: Optional maximum time to hold a position

---

## Quick Start

### 1. Basic Backtest

```python
from src.MLBacktester import MLBacktester
from run_me import train, prepare_data
import pandas as pd

# Load and prepare data
df_train = pd.read_csv("data/btc_2022.csv")
df_test = pd.read_csv("data/btc_2023.csv")

# Train models
models, scaler, train_results, best_model_name = train(df_train)
best_model = models[best_model_name][0]

# Prepare test data with features
from src.FeaturesGenerator import FeaturesGenerator
fg = FeaturesGenerator(df_test)
df_test_with_features = fg.generate_all_features()

# Get feature columns
X_test, y_test = prepare_data(df_test, target_bars=45, target_pct=3.0)
X_columns = X_test.columns.tolist()

# Initialize backtester
backtester = MLBacktester(
    initial_capital=10000.0,
    position_size=1.0,
    trailing_stop_pct=2.0,
    take_profit_pct=5.0,
    probability_threshold=0.6
)

# Run backtest
results = backtester.run_backtest(
    df=df_test_with_features,
    model=best_model,
    scaler=scaler,
    X_columns=X_columns
)

# Print and plot results
backtester.print_results(results)
backtester.plot_results(results, df_test_with_features, save_path='plots/backtest.png')
```

### 2. Run Complete Example

```bash
python backtest_example.py
```

This will:
- Train multiple ML models
- Backtest the best model
- Compare different trailing stop percentages
- Compare all models
- Test conservative vs aggressive strategies

---

## MLBacktester Class

### Initialization Parameters

```python
MLBacktester(
    initial_capital=10000.0,        # Starting capital
    position_size=1.0,              # Fraction of capital per trade (0.0-1.0)
    trailing_stop_pct=2.0,          # Trailing stop loss percentage
    take_profit_pct=None,           # Take profit percentage (optional)
    commission=0.001,               # Commission per trade (0.1%)
    slippage=0.0005,                # Slippage per trade (0.05%)
    use_probability_threshold=True, # Use probability threshold
    probability_threshold=0.6,      # Minimum probability to enter (0.0-1.0)
    max_holding_bars=None           # Max bars to hold position (optional)
)
```

### Key Methods

#### `run_backtest(df, model, scaler, X_columns, close_column='Close', timestamp_column='Timestamp')`
Run the backtest on provided data.

**Parameters:**
- `df`: DataFrame with OHLCV data and features
- `model`: Trained ML model with `predict()` method
- `scaler`: Fitted scaler for features
- `X_columns`: List of feature column names
- `close_column`: Name of close price column
- `timestamp_column`: Name of timestamp column

**Returns:**
- Dictionary with backtest results and metrics

#### `print_results(results)`
Print formatted backtest results.

#### `plot_results(results, df, close_column='Close', timestamp_column='Timestamp', save_path=None)`
Create visualization plots:
- Price chart with entry/exit signals
- Equity curve
- Drawdown chart

---

## Features

### 1. Trailing Stop Loss

The trailing stop loss dynamically adjusts as the price moves in your favor:

- **Entry**: Stop loss is set at `entry_price * (1 - trailing_stop_pct/100)`
- **Price Increases**: Stop loss moves up to `highest_price * (1 - trailing_stop_pct/100)`
- **Price Decreases**: Stop loss stays at highest level (doesn't move down)
- **Exit**: Position closes when price hits trailing stop

**Example:**
```python
# 2% trailing stop
backtester = MLBacktester(trailing_stop_pct=2.0)

# Entry at $100
# Initial stop: $98 (100 * 0.98)

# Price rises to $110
# Stop moves to: $107.80 (110 * 0.98)

# Price rises to $120
# Stop moves to: $117.60 (120 * 0.98)

# Price falls to $116
# Stop stays at: $117.60
# Position exits at $117.60 with profit
```

### 2. Take Profit

Optional fixed take profit target:

```python
backtester = MLBacktester(
    trailing_stop_pct=2.0,
    take_profit_pct=5.0  # Exit at 5% profit
)
```

### 3. Position Sizing

Control how much capital to use per trade:

```python
# Use 50% of capital per trade (more conservative)
backtester = MLBacktester(position_size=0.5)

# Use 100% of capital per trade (more aggressive)
backtester = MLBacktester(position_size=1.0)
```

### 4. Probability Threshold

Only enter trades when model confidence is high:

```python
# Only enter when model predicts >70% probability
backtester = MLBacktester(
    use_probability_threshold=True,
    probability_threshold=0.7
)
```

### 5. Max Holding Period

Limit how long to hold a position:

```python
# Exit after 100 bars regardless of profit/loss
backtester = MLBacktester(max_holding_bars=100)
```

### 6. Trading Costs

Realistic commission and slippage:

```python
backtester = MLBacktester(
    commission=0.001,   # 0.1% commission
    slippage=0.0005     # 0.05% slippage
)
```

---

## Examples

### Example 1: Conservative Strategy

```python
backtester = MLBacktester(
    initial_capital=10000.0,
    position_size=0.5,              # Use 50% of capital
    trailing_stop_pct=1.5,          # Tight stop loss
    take_profit_pct=4.0,            # Conservative profit target
    probability_threshold=0.7       # High confidence only
)
```

**Characteristics:**
- Lower risk
- Fewer trades
- Smaller drawdowns
- More stable returns

### Example 2: Aggressive Strategy

```python
backtester = MLBacktester(
    initial_capital=10000.0,
    position_size=1.0,              # Use 100% of capital
    trailing_stop_pct=3.0,          # Wider stop loss
    take_profit_pct=8.0,            # Higher profit target
    probability_threshold=0.55      # Lower confidence threshold
)
```

**Characteristics:**
- Higher risk
- More trades
- Larger potential drawdowns
- Higher potential returns

### Example 3: Compare Trailing Stops

```python
trailing_stops = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
results = []

for stop_pct in trailing_stops:
    backtester = MLBacktester(trailing_stop_pct=stop_pct)
    result = backtester.run_backtest(df, model, scaler, X_columns)
    results.append({
        'stop_pct': stop_pct,
        'return_pct': result['total_return_pct'],
        'max_drawdown': result['max_drawdown']
    })

# Find optimal trailing stop
comparison_df = pd.DataFrame(results)
print(comparison_df)
```

### Example 4: Compare Models

```python
# Train multiple models
models, scaler, train_results, best_model_name = train(df_train)

# Backtest each model
for model_name, model_data in models.items():
    model = model_data[0]
    backtester = MLBacktester(trailing_stop_pct=2.0)
    results = backtester.run_backtest(df, model, scaler, X_columns)
    print(f"{model_name}: {results['total_return_pct']:.2f}%")
```

---

## Best Practices

### 1. Walk-Forward Testing

Test on out-of-sample data:

```python
# Train on 2022 data
models, scaler, _, _ = train(df_2022)

# Test on 2023 data (never seen by model)
results = backtester.run_backtest(df_2023, model, scaler, X_columns)
```

### 2. Parameter Optimization

Test multiple parameter combinations:

```python
best_return = -float('inf')
best_params = {}

for stop in [1.0, 2.0, 3.0]:
    for threshold in [0.55, 0.6, 0.65, 0.7]:
        backtester = MLBacktester(
            trailing_stop_pct=stop,
            probability_threshold=threshold
        )
        results = backtester.run_backtest(df, model, scaler, X_columns)
        
        if results['total_return_pct'] > best_return:
            best_return = results['total_return_pct']
            best_params = {'stop': stop, 'threshold': threshold}

print(f"Best params: {best_params}")
```

### 3. Risk Management

Always consider risk-adjusted returns:

```python
# Don't just look at total return
# Also consider:
# - Max drawdown
# - Sharpe ratio
# - Win rate
# - Profit factor

if results['sharpe_ratio'] > 1.0 and results['max_drawdown'] > -20:
    print("Good risk-adjusted returns!")
```

### 4. Realistic Assumptions

Use realistic trading costs:

```python
# Crypto exchanges: 0.1-0.2% commission
backtester = MLBacktester(commission=0.001, slippage=0.001)

# Stock brokers: 0.0-0.1% commission
backtester = MLBacktester(commission=0.0005, slippage=0.0005)
```

### 5. Model Selection

Choose models based on backtest performance:

```python
# Don't just use training accuracy
# Test multiple models in backtest
# Choose based on:
# - Backtest returns
# - Risk metrics
# - Consistency across periods
```

---

## Performance Metrics

The backtester calculates comprehensive metrics:

### Return Metrics
- **Total Return**: Absolute profit/loss in dollars
- **Total Return %**: Percentage return on initial capital
- **Buy & Hold Return %**: Comparison to passive strategy

### Trade Metrics
- **Total Trades**: Number of completed trades
- **Winning Trades**: Number of profitable trades
- **Losing Trades**: Number of losing trades
- **Win Rate**: Percentage of winning trades
- **Avg Bars Held**: Average holding period

### Risk Metrics
- **Max Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return (annualized)
- **Profit Factor**: Gross profit / Gross loss
- **Avg Win**: Average profit per winning trade
- **Avg Loss**: Average loss per losing trade

---

## Troubleshooting

### Issue: No trades executed

**Possible causes:**
1. Probability threshold too high
2. Model never predicts positive class
3. Not enough data

**Solution:**
```python
# Lower probability threshold
backtester = MLBacktester(probability_threshold=0.5)

# Or disable threshold
backtester = MLBacktester(use_probability_threshold=False)
```

### Issue: Too many trades

**Possible causes:**
1. Probability threshold too low
2. Model is too aggressive

**Solution:**
```python
# Raise probability threshold
backtester = MLBacktester(probability_threshold=0.7)

# Or use position sizing
backtester = MLBacktester(position_size=0.5)
```

### Issue: Large drawdowns

**Possible causes:**
1. Trailing stop too wide
2. No take profit
3. Position size too large

**Solution:**
```python
# Tighter stops and smaller positions
backtester = MLBacktester(
    trailing_stop_pct=1.5,
    take_profit_pct=4.0,
    position_size=0.5
)
```

---

## Advanced Usage

### Custom Exit Logic

You can extend the `MLBacktester` class for custom exit logic:

```python
class CustomBacktester(MLBacktester):
    def check_exit_conditions(self, current_price, timestamp):
        # Call parent method first
        if super().check_exit_conditions(current_price, timestamp):
            return True
        
        # Add custom exit logic
        # Example: Exit if position is profitable and RSI > 70
        if self.position > 0:
            pnl_pct = (current_price / self.entry_price - 1) * 100
            if pnl_pct > 2.0 and self.custom_indicator > 70:
                self.exit_long(current_price, timestamp, reason='CUSTOM')
                return True
        
        return False
```

### Multiple Timeframes

Test on different timeframes:

```python
timeframes = ['1h', '4h', '1d']
results = {}

for tf in timeframes:
    df = load_data(timeframe=tf)
    backtester = MLBacktester()
    results[tf] = backtester.run_backtest(df, model, scaler, X_columns)
```

---

## Integration with MLflow

Track backtest results in MLflow:

```python
from src.MLflow.mlflow_tracker import MLflowTracker

tracker = MLflowTracker(
    experiment_name="ml_predict_15/backtest/btc_usd_daily",
    run_name="random_forest_trailing_stop_2pct"
)

# Run backtest
results = backtester.run_backtest(df, model, scaler, X_columns)

# Log to MLflow
tracker.log_params({
    'trailing_stop_pct': backtester.trailing_stop_pct,
    'position_size': backtester.position_size,
    'probability_threshold': backtester.probability_threshold
})

tracker.log_metrics({
    'total_return_pct': results['total_return_pct'],
    'win_rate': results['win_rate'],
    'max_drawdown': results['max_drawdown'],
    'sharpe_ratio': results['sharpe_ratio']
})

tracker.end_run()
```

---

## Summary

The ML backtesting framework provides:

✅ **Realistic Trading Simulation**: Commission, slippage, and position sizing  
✅ **Risk Management**: Trailing stop loss and take profit  
✅ **Model Integration**: Use any sklearn-compatible model  
✅ **Comprehensive Metrics**: Returns, risk, and trade statistics  
✅ **Visualization**: Charts for analysis  
✅ **Flexibility**: Customizable parameters and strategies  

Start with the `backtest_example.py` to see it in action!
