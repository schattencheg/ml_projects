# Backtesting Modules - Complete Reference

This document provides a complete reference for the three backtesting modules available in this project.

## Table of Contents

1. [Overview](#overview)
2. [Module Comparison](#module-comparison)
3. [MLBacktester (Custom)](#mlbacktester-custom)
4. [BacktestingPyStrategy](#backtestingpystrategy)
5. [BacktraderStrategy](#backtraderstrategy)
6. [Quick Start Guide](#quick-start-guide)
7. [Examples](#examples)

---

## Overview

This project includes **three backtesting solutions**, each with different strengths:

### 1. MLBacktester (Custom Implementation)
- **File**: `src/MLBacktester.py`
- **Type**: Custom pandas-based backtester
- **Best for**: Learning, customization, simple strategies
- **Example**: `backtest_example.py`

### 2. BacktestingPyStrategy (backtesting.py Library)
- **File**: `src/BacktestingPyStrategy.py`
- **Type**: Vectorized backtesting library
- **Best for**: Fast optimization, parameter tuning
- **Example**: `backtest_backtestingpy_example.py`

### 3. BacktraderStrategy (Backtrader Library)
- **File**: `src/BacktraderStrategy.py`
- **Type**: Event-driven backtesting library
- **Best for**: Realistic simulation, live trading preparation
- **Example**: `backtest_backtrader_example.py`

---

## Module Comparison

| Feature | MLBacktester | backtesting.py | Backtrader |
|---------|--------------|----------------|------------|
| **Implementation** | Custom | Library | Library |
| **Speed** | ⚡⚡ Fast | ⚡⚡⚡ Very Fast | ⚡⚡ Moderate |
| **Ease of Use** | ⭐⭐⭐ Easy | ⭐⭐⭐ Easy | ⭐⭐ Moderate |
| **Customization** | ⭐⭐⭐ Full | ⭐⭐ Limited | ⭐⭐⭐ Full |
| **Optimization** | ❌ Manual | ✅ Built-in | ✅ Built-in |
| **Visualization** | ⭐⭐ Matplotlib | ⭐⭐⭐ Interactive | ⭐⭐ Matplotlib |
| **Live Trading** | ❌ No | ❌ No | ✅ Yes |
| **Learning Curve** | ⭐⭐⭐ Easy | ⭐⭐⭐ Easy | ⭐⭐ Moderate |
| **Dependencies** | Minimal | backtesting | backtrader |

### Recommendation

- **Start with**: MLBacktester (understand the basics)
- **Develop with**: backtesting.py (fast iterations)
- **Validate with**: Backtrader (realistic results)
- **Deploy with**: Backtrader (live trading ready)

---

## MLBacktester (Custom)

### Overview

Custom pandas-based backtester with full transparency and easy customization.

### Key Features

✅ **Simple and Transparent**
- Pure Python implementation
- Easy to understand and modify
- No external dependencies

✅ **ML Integration**
- Direct model prediction integration
- Probability-based entry signals
- Configurable thresholds

✅ **Risk Management**
- Trailing stop loss
- Take profit targets
- Position sizing
- Commission and slippage

### Class: `MLBacktester`

```python
from src.MLBacktester import MLBacktester

backtester = MLBacktester(
    initial_capital=10000.0,
    position_size=1.0,              # 100% of capital
    trailing_stop_pct=2.0,          # 2% trailing stop
    take_profit_pct=5.0,            # 5% take profit
    commission=0.001,               # 0.1% commission
    slippage=0.0005,                # 0.05% slippage
    use_probability_threshold=True,
    probability_threshold=0.6,      # 60% confidence minimum
    max_holding_bars=None           # No time limit
)
```

### Main Methods

#### `run_backtest()`
```python
results = backtester.run_backtest(
    df=df_test,
    model=model,
    scaler=scaler,
    X_columns=feature_columns,
    close_column='Close',
    timestamp_column='Timestamp'
)
```

Returns dictionary with:
- `equity_curve`: DataFrame with equity over time
- `trades`: List of all trades
- `signals`: List of all signals
- `total_return`: Total return in dollars
- `total_return_pct`: Total return percentage
- `sharpe_ratio`: Sharpe ratio
- `max_drawdown`: Maximum drawdown
- `win_rate`: Win rate percentage
- `total_trades`: Number of trades

#### `print_results()`
```python
backtester.print_results(results)
```

Prints formatted results to console.

#### `plot_results()`
```python
backtester.plot_results(
    results=results,
    df=df_test,
    save_path='plots/backtest_results.png'
)
```

Creates comprehensive visualization with:
- Price chart with entry/exit signals
- Equity curve
- Drawdown chart

### Example Usage

```python
from src.MLBacktester import MLBacktester
from src.model_loader import load_all_models, load_scaler

# Initialize
backtester = MLBacktester(
    initial_capital=10000.0,
    trailing_stop_pct=2.0
)

# Load model
models = load_all_models()
scaler = load_scaler()

# Run backtest
results = backtester.run_backtest(
    df=df_test,
    model=models['logistic_regression'],
    scaler=scaler,
    X_columns=feature_columns
)

# Display results
backtester.print_results(results)
backtester.plot_results(results, df_test)
```

### Advantages

- ✅ Easy to understand
- ✅ Easy to customize
- ✅ No external dependencies
- ✅ Full control over logic
- ✅ Great for learning

### Limitations

- ❌ No built-in optimization
- ❌ Manual parameter tuning
- ❌ Basic visualizations
- ❌ Not suitable for live trading

---

## BacktestingPyStrategy

### Overview

Integration with the popular `backtesting.py` library for fast vectorized backtesting.

### Key Features

✅ **Blazing Fast**
- Vectorized operations
- Process entire dataset at once
- Ideal for optimization

✅ **Built-in Optimization**
- Grid search
- Heatmap visualization
- Constraint support

✅ **Interactive Plots**
- Bokeh-based visualizations
- Zoom, pan, explore
- Professional appearance

### Class: `MLBacktesterPy`

```python
from src.BacktestingPyStrategy import MLBacktesterPy

backtester = MLBacktesterPy(
    initial_cash=10000.0,
    commission=0.001,
    margin=1.0,
    trade_on_close=False,
    hedging=False,
    exclusive_orders=True
)
```

### Main Methods

#### `run_backtest()`
```python
stats, trades = backtester.run_backtest(
    df=df_test,
    model=model,
    scaler=scaler,
    X_columns=feature_columns,
    probability_threshold=0.6,
    trailing_stop_pct=2.0,
    take_profit_pct=5.0,
    position_size_pct=1.0,
    plot=True
)
```

Returns:
- `stats`: pd.Series with comprehensive statistics
- `trades`: pd.DataFrame with trade history

#### `optimize()`
```python
best_stats, heatmap = backtester.optimize(
    df=df_test,
    model=model,
    scaler=scaler,
    X_columns=feature_columns,
    probability_threshold_range=(0.5, 0.8, 0.05),
    trailing_stop_range=(1.0, 5.0, 0.5),
    maximize='Return [%]',
    return_heatmap=True
)
```

Automatically finds best parameters.

#### `print_results()`
```python
backtester.print_results(stats)
```

### Strategy Class: `MLStrategy`

The underlying strategy class that implements the trading logic:

```python
from backtesting import Backtest
from src.BacktestingPyStrategy import MLStrategy

# Set parameters
MLStrategy.probability_threshold = 0.6
MLStrategy.trailing_stop_pct = 2.0
MLStrategy.take_profit_pct = 5.0
MLStrategy.position_size_pct = 1.0

# Create and run backtest
bt = Backtest(df, MLStrategy, cash=10000, commission=0.001)
stats = bt.run()
```

### Example Usage

```python
from src.BacktestingPyStrategy import MLBacktesterPy

# Initialize
backtester = MLBacktesterPy(initial_cash=10000.0)

# Run backtest
stats, trades = backtester.run_backtest(
    df=df_test,
    model=model,
    scaler=scaler,
    X_columns=feature_columns,
    plot=True
)

# Optimize
best_stats, heatmap = backtester.optimize(
    df=df_test,
    model=model,
    scaler=scaler,
    X_columns=feature_columns,
    probability_threshold_range=(0.5, 0.8, 0.05),
    trailing_stop_range=(1.0, 5.0, 0.5)
)

print(f"Best threshold: {best_stats._strategy.probability_threshold}")
print(f"Best stop: {best_stats._strategy.trailing_stop_pct}%")
```

### Advantages

- ✅ Very fast execution
- ✅ Built-in optimization
- ✅ Interactive visualizations
- ✅ Comprehensive statistics
- ✅ Easy to use

### Limitations

- ❌ Less realistic execution
- ❌ No live trading support
- ❌ Limited customization
- ❌ Vectorized (not event-driven)

---

## BacktraderStrategy

### Overview

Integration with Backtrader for realistic event-driven backtesting and live trading preparation.

### Key Features

✅ **Realistic Execution**
- Bar-by-bar processing
- Accurate order simulation
- Slippage and commission

✅ **Live Trading Ready**
- Same code for backtest and live
- Multiple broker integrations
- Real-time data feeds

✅ **Flexible Architecture**
- Multiple timeframes
- Custom indicators
- Complex strategies

### Class: `MLBacktesterBT`

```python
from src.BacktraderStrategy import MLBacktesterBT

backtester = MLBacktesterBT(
    initial_cash=10000.0,
    commission=0.001,
    slippage_perc=0.0,
    slippage_fixed=0.0
)
```

### Main Methods

#### `run_backtest()`
```python
results, trades = backtester.run_backtest(
    df=df_test,
    model=model,
    scaler=scaler,
    X_columns=feature_columns,
    probability_threshold=0.6,
    trailing_stop_pct=2.0,
    take_profit_pct=5.0,
    position_size_pct=1.0,
    plot=True,
    printlog=False
)
```

Returns:
- `results`: Dict with performance metrics
- `trades`: pd.DataFrame with trade history

#### `optimize()`
```python
optimization_results = backtester.optimize(
    df=df_test,
    model=model,
    scaler=scaler,
    X_columns=feature_columns,
    probability_threshold_range=(0.5, 0.8, 0.1),
    trailing_stop_range=(1.0, 5.0, 1.0)
)

# Get best parameters
best = optimization_results[0]
print(f"Best threshold: {best['probability_threshold']}")
print(f"Best stop: {best['trailing_stop_pct']}%")
```

#### `print_results()`
```python
backtester.print_results(results)
```

### Strategy Class: `MLStrategy`

The underlying Backtrader strategy:

```python
import backtrader as bt
from src.BacktraderStrategy import MLStrategy

cerebro = bt.Cerebro()
cerebro.addstrategy(
    MLStrategy,
    probability_threshold=0.6,
    trailing_stop_pct=2.0,
    take_profit_pct=5.0,
    position_size_pct=1.0,
    printlog=True
)
```

### Example Usage

```python
from src.BacktraderStrategy import MLBacktesterBT

# Initialize
backtester = MLBacktesterBT(initial_cash=10000.0)

# Run backtest
results, trades = backtester.run_backtest(
    df=df_test,
    model=model,
    scaler=scaler,
    X_columns=feature_columns,
    plot=True
)

# Print results
backtester.print_results(results)

# Optimize
opt_results = backtester.optimize(
    df=df_test,
    model=model,
    scaler=scaler,
    X_columns=feature_columns
)

best = opt_results[0]
print(f"Best parameters: {best}")
```

### Advantages

- ✅ Realistic execution
- ✅ Live trading support
- ✅ Highly flexible
- ✅ Extensive features
- ✅ Active community

### Limitations

- ❌ Slower than vectorized
- ❌ Steeper learning curve
- ❌ More complex setup
- ❌ Verbose code

---

## Quick Start Guide

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Train Models (One-Time)

```bash
python train_and_save_models.py
```

### Run Examples

```bash
# Custom MLBacktester
python backtest_example.py

# backtesting.py library
python backtest_backtestingpy_example.py

# Backtrader library
python backtest_backtrader_example.py

# Compare all three
python backtest_compare_libraries.py
```

---

## Examples

### Example 1: Basic Backtest (All Three)

```python
# MLBacktester
from src.MLBacktester import MLBacktester
backtester = MLBacktester(initial_capital=10000.0)
results = backtester.run_backtest(df, model, scaler, X_columns)

# backtesting.py
from src.BacktestingPyStrategy import MLBacktesterPy
backtester = MLBacktesterPy(initial_cash=10000.0)
stats, trades = backtester.run_backtest(df, model, scaler, X_columns)

# Backtrader
from src.BacktraderStrategy import MLBacktesterBT
backtester = MLBacktesterBT(initial_cash=10000.0)
results, trades = backtester.run_backtest(df, model, scaler, X_columns)
```

### Example 2: Parameter Optimization

```python
# backtesting.py (fastest)
best_stats, heatmap = backtester.optimize(
    df=df_test,
    model=model,
    scaler=scaler,
    X_columns=X_columns,
    probability_threshold_range=(0.5, 0.8, 0.05),
    trailing_stop_range=(1.0, 5.0, 0.5)
)

# Backtrader (more realistic)
opt_results = backtester.optimize(
    df=df_test,
    model=model,
    scaler=scaler,
    X_columns=X_columns,
    probability_threshold_range=(0.5, 0.8, 0.1),
    trailing_stop_range=(1.0, 5.0, 1.0)
)
```

### Example 3: Compare Models

```python
for model_name, model in models.items():
    results = backtester.run_backtest(
        df=df_test,
        model=model,
        scaler=scaler,
        X_columns=X_columns,
        plot=False
    )
    print(f"{model_name}: {results['total_return_pct']:.2f}%")
```

---

## Best Practices

### 1. Development Workflow

```
1. Prototype with MLBacktester (understand logic)
   ↓
2. Optimize with backtesting.py (find best parameters)
   ↓
3. Validate with Backtrader (realistic simulation)
   ↓
4. Deploy with Backtrader (live trading)
```

### 2. Always Include Costs

```python
commission = 0.001  # 0.1%
slippage = 0.0005   # 0.05%
```

### 3. Avoid Overfitting

- Use walk-forward analysis
- Validate on out-of-sample data
- Don't over-optimize

### 4. Monitor Risk

- Track maximum drawdown
- Set position size limits
- Use stop losses

---

## Troubleshooting

### Issue: Different results between libraries

**Cause**: Execution model differences (vectorized vs event-driven)

**Solution**: This is normal. Use both for validation.

### Issue: No trades executed

**Cause**: Probability threshold too high or model not predicting

**Solution**:
```python
# Lower threshold
probability_threshold = 0.5

# Check predictions
print(df['ML_Signal'].value_counts())
```

### Issue: Optimization too slow

**Solution**:
```python
# Use fewer parameter combinations
probability_threshold_range=(0.55, 0.70, 0.05)  # Fewer steps
trailing_stop_range=(1.5, 3.0, 0.5)

# Or use smaller dataset
df_subset = df.iloc[-50000:]
```

---

## Additional Resources

- [Backtesting Libraries Guide](BACKTEST_LIBRARIES_GUIDE.md)
- [Backtest Guide](BACKTEST_GUIDE.md)
- Example scripts in project root

---

## Summary

Choose your backtesting module based on your needs:

| Need | Use |
|------|-----|
| Learning | MLBacktester |
| Fast optimization | backtesting.py |
| Realistic simulation | Backtrader |
| Live trading | Backtrader |
| Custom logic | MLBacktester |
| Quick results | backtesting.py |

**Best approach**: Use all three! Each provides unique insights.

Happy backtesting! 🚀📈
