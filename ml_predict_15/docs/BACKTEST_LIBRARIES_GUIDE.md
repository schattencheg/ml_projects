# Backtesting Libraries Guide

This guide explains how to use the two professional backtesting libraries integrated with your ML models: **backtesting.py** and **Backtrader**.

## Table of Contents

1. [Overview](#overview)
2. [backtesting.py Library](#backtestingpy-library)
3. [Backtrader Library](#backtrader-library)
4. [Comparison](#comparison)
5. [Quick Start](#quick-start)
6. [Advanced Usage](#advanced-usage)

---

## Overview

We've integrated two popular Python backtesting libraries with your ML prediction models:

### backtesting.py
- **Type**: Vectorized backtesting
- **Speed**: Very fast (processes entire dataset at once)
- **Best for**: Quick iterations, parameter optimization, statistical analysis
- **Visualization**: Interactive Bokeh plots
- **Learning curve**: Easy

### Backtrader
- **Type**: Event-driven backtesting
- **Speed**: Moderate (processes bar-by-bar)
- **Best for**: Realistic simulations, complex strategies, live trading preparation
- **Visualization**: Matplotlib plots
- **Learning curve**: Moderate

---

## backtesting.py Library

### Features

✅ **Fast Vectorized Backtesting**
- Processes entire dataset at once
- Ideal for rapid testing and optimization

✅ **Built-in Parameter Optimization**
- Grid search optimization
- Heatmap visualization
- Constraint support

✅ **Interactive Plots**
- Bokeh-based interactive charts
- Zoom, pan, and explore results
- Professional-looking visualizations

✅ **Comprehensive Statistics**
- Sharpe ratio, Sortino ratio, Calmar ratio
- Drawdown analysis
- Trade statistics

### Basic Usage

```python
from src.BacktestingPyStrategy import MLBacktesterPy
from src.model_loader import load_all_models, load_scaler

# Initialize backtester
backtester = MLBacktesterPy(
    initial_cash=10000.0,
    commission=0.001,  # 0.1%
    margin=1.0,
    trade_on_close=False
)

# Load model and scaler
models = load_all_models()
scaler = load_scaler()
model = models['logistic_regression']

# Run backtest
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

# Print results
backtester.print_results(stats)
```

### Parameter Optimization

```python
# Optimize parameters
best_stats, heatmap = backtester.optimize(
    df=df_test,
    model=model,
    scaler=scaler,
    X_columns=feature_columns,
    probability_threshold_range=(0.5, 0.8, 0.05),  # start, end, step
    trailing_stop_range=(1.0, 5.0, 0.5),
    maximize='Return [%]',
    return_heatmap=True
)

print(f"Best Probability Threshold: {best_stats._strategy.probability_threshold}")
print(f"Best Trailing Stop: {best_stats._strategy.trailing_stop_pct}%")
print(f"Best Return: {best_stats['Return [%]']:.2f}%")
```

### Available Statistics

- `Return [%]` - Total return percentage
- `Buy & Hold Return [%]` - Buy and hold benchmark
- `# Trades` - Total number of trades
- `Win Rate [%]` - Percentage of winning trades
- `Best Trade [%]` - Best single trade
- `Worst Trade [%]` - Worst single trade
- `Avg. Trade [%]` - Average trade return
- `Max. Drawdown [%]` - Maximum drawdown
- `Sharpe Ratio` - Risk-adjusted return
- `Sortino Ratio` - Downside risk-adjusted return
- `Calmar Ratio` - Return to max drawdown ratio

### Example Script

Run the complete example:
```bash
python backtest_backtestingpy_example.py
```

---

## Backtrader Library

### Features

✅ **Event-Driven Backtesting**
- Bar-by-bar processing (realistic)
- Accurate order execution simulation
- Slippage and commission modeling

✅ **Flexible Strategy Development**
- Full control over trading logic
- Multiple timeframes support
- Custom indicators

✅ **Built-in Analyzers**
- Trade analyzer
- Sharpe ratio
- Drawdown analysis
- Returns analysis

✅ **Live Trading Ready**
- Same code for backtesting and live trading
- Multiple broker integrations
- Real-time data feed support

### Basic Usage

```python
from src.BacktraderStrategy import MLBacktesterBT
from src.model_loader import load_all_models, load_scaler

# Initialize backtester
backtester = MLBacktesterBT(
    initial_cash=10000.0,
    commission=0.001,  # 0.1%
    slippage_perc=0.0,
    slippage_fixed=0.0
)

# Load model and scaler
models = load_all_models()
scaler = load_scaler()
model = models['logistic_regression']

# Run backtest
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

# Print results
backtester.print_results(results)
```

### Parameter Optimization

```python
# Optimize parameters
optimization_results = backtester.optimize(
    df=df_test,
    model=model,
    scaler=scaler,
    X_columns=feature_columns,
    probability_threshold_range=(0.5, 0.8, 0.1),
    trailing_stop_range=(1.0, 5.0, 1.0)
)

# Get best parameters
best_params = optimization_results[0]
print(f"Best Probability Threshold: {best_params['probability_threshold']}")
print(f"Best Trailing Stop: {best_params['trailing_stop_pct']}%")
print(f"Final Value: ${best_params['final_value']:,.2f}")
```

### Available Metrics

- `initial_capital` - Starting capital
- `final_value` - Final portfolio value
- `total_return` - Total return in dollars
- `total_return_pct` - Total return percentage
- `sharpe_ratio` - Sharpe ratio
- `max_drawdown` - Maximum drawdown percentage
- `total_trades` - Total number of trades
- `won_trades` - Number of winning trades
- `lost_trades` - Number of losing trades
- `win_rate` - Win rate percentage
- `avg_win` - Average winning trade
- `avg_loss` - Average losing trade
- `best_trade` - Best single trade
- `worst_trade` - Worst single trade

### Example Script

Run the complete example:
```bash
python backtest_backtrader_example.py
```

---

## Comparison

| Feature | backtesting.py | Backtrader |
|---------|---------------|------------|
| **Speed** | ⚡⚡⚡ Very Fast | ⚡⚡ Moderate |
| **Realism** | ⭐⭐ Good | ⭐⭐⭐ Excellent |
| **Ease of Use** | ⭐⭐⭐ Easy | ⭐⭐ Moderate |
| **Optimization** | ⭐⭐⭐ Built-in grid search | ⭐⭐ Manual implementation |
| **Visualization** | ⭐⭐⭐ Interactive Bokeh | ⭐⭐ Static Matplotlib |
| **Live Trading** | ❌ Not supported | ✅ Supported |
| **Multiple Timeframes** | ❌ Limited | ✅ Full support |
| **Custom Indicators** | ⭐⭐ Limited | ⭐⭐⭐ Extensive |
| **Documentation** | ⭐⭐⭐ Excellent | ⭐⭐⭐ Extensive |

### When to Use Each

**Use backtesting.py when:**
- You need fast iterations
- You want to optimize parameters quickly
- You prefer interactive visualizations
- You're doing statistical analysis
- You want simple, clean code

**Use Backtrader when:**
- You need realistic order execution
- You plan to move to live trading
- You need complex multi-timeframe strategies
- You want detailed control over execution
- You need custom indicators

---

## Quick Start

### 1. Install Dependencies

```bash
pip install backtesting backtrader matplotlib
```

### 2. Train Models (One-Time)

```bash
python train_and_save_models.py
```

### 3. Run Backtests

**Option A: backtesting.py**
```bash
python backtest_backtestingpy_example.py
```

**Option B: Backtrader**
```bash
python backtest_backtrader_example.py
```

### 4. Compare Results

Both scripts will:
- Load your trained ML models
- Run backtests with different parameters
- Optimize strategy parameters
- Compare multiple models
- Generate visualizations
- Save plots to `plots/` directory

---

## Advanced Usage

### Custom Strategy Parameters

Both libraries support customizable parameters:

```python
# Strategy parameters
probability_threshold = 0.6    # Min probability to enter (0.0-1.0)
trailing_stop_pct = 2.0       # Trailing stop loss percentage
take_profit_pct = 5.0         # Take profit percentage (None = disabled)
position_size_pct = 1.0       # Fraction of capital per trade (0.0-1.0)

# Execution parameters
commission = 0.001            # Commission per trade (0.1%)
slippage = 0.0005            # Slippage per trade (0.05%)
```

### Conservative vs Aggressive Strategies

**Conservative Strategy:**
```python
# Lower risk, higher probability threshold
probability_threshold = 0.7   # High confidence only
trailing_stop_pct = 1.5      # Tight stop loss
position_size_pct = 0.5      # Use 50% of capital
take_profit_pct = 4.0        # Quick profit taking
```

**Aggressive Strategy:**
```python
# Higher risk, lower probability threshold
probability_threshold = 0.55  # More trades
trailing_stop_pct = 3.5      # Wider stop loss
position_size_pct = 1.0      # Use 100% of capital
take_profit_pct = 8.0        # Larger profit targets
```

### Multi-Model Comparison

Both example scripts include code to compare all your trained models:

```python
# Compare all models
for model_name, model in models.items():
    results, trades = backtester.run_backtest(
        df=df_test,
        model=model,
        scaler=scaler,
        X_columns=X_columns,
        plot=False  # Don't plot each one
    )
    # Store results for comparison
```

### Optimization Constraints

Add constraints to optimization (backtesting.py):

```python
def constraint(params):
    # Only accept if Sharpe > 1.0 and Drawdown < 20%
    return params['Sharpe Ratio'] > 1.0 and params['Max. Drawdown [%]'] < 20

best_stats = backtester.optimize(
    df=df_test,
    model=model,
    scaler=scaler,
    X_columns=X_columns,
    constraint=constraint,
    maximize='Return [%]'
)
```

---

## Tips and Best Practices

### 1. Start Simple
- Begin with default parameters
- Understand the baseline performance
- Then optimize incrementally

### 2. Avoid Overfitting
- Don't over-optimize on test data
- Use walk-forward analysis
- Validate on out-of-sample data

### 3. Consider Transaction Costs
- Always include commission and slippage
- Real trading has costs
- Higher frequency = higher costs

### 4. Monitor Drawdown
- Max drawdown is critical
- Can you handle the losses?
- Risk management is key

### 5. Use Both Libraries
- backtesting.py for quick testing
- Backtrader for final validation
- Compare results between both

### 6. Document Your Findings
- Keep track of what works
- Note parameter sensitivities
- Build a knowledge base

---

## Troubleshooting

### Issue: No trades executed

**Possible causes:**
- Probability threshold too high
- ML model not making predictions
- Data preparation issues

**Solutions:**
```python
# Lower probability threshold
probability_threshold = 0.5

# Check ML predictions
print(df['ML_Signal'].value_counts())
print(df['ML_Probability'].describe())
```

### Issue: Poor performance

**Possible causes:**
- Model not trained properly
- Features not aligned
- Overfitting to training data

**Solutions:**
- Retrain models
- Check feature engineering
- Use walk-forward validation

### Issue: Optimization takes too long

**Solutions:**
```python
# Reduce parameter ranges
probability_threshold_range=(0.55, 0.70, 0.05)  # Fewer steps
trailing_stop_range=(1.5, 3.0, 0.5)

# Use smaller dataset
df_subset = df_test.iloc[-50000:]  # Last 50k rows
```

---

## Next Steps

1. **Run the examples** - Start with the provided example scripts
2. **Experiment with parameters** - Try different settings
3. **Compare models** - See which ML model performs best
4. **Optimize strategies** - Find optimal parameters
5. **Validate results** - Use walk-forward analysis
6. **Deploy** - Move to paper trading or live trading

---

## Resources

### backtesting.py
- Documentation: https://kernc.github.io/backtesting.py/
- GitHub: https://github.com/kernc/backtesting.py
- Examples: https://kernc.github.io/backtesting.py/doc/examples/

### Backtrader
- Documentation: https://www.backtrader.com/docu/
- GitHub: https://github.com/mementum/backtrader
- Community: https://community.backtrader.com/

### Additional Reading
- "Advances in Financial Machine Learning" by Marcos López de Prado
- "Quantitative Trading" by Ernest Chan
- "Algorithmic Trading" by Ernie Chan

---

## Support

For issues or questions:
1. Check the example scripts
2. Review this guide
3. Check library documentation
4. Test with simple strategies first

Happy backtesting! 🚀📈
