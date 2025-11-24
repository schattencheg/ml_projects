# Backtesting Module

Comprehensive backtesting system with three different backends sharing a common base class.

## Overview

This module provides three backtesting implementations:

1. **BacktestNoLib** - Custom implementation without external dependencies
2. **BacktestBacktrader** - Integration with Backtrader library
3. **BacktestBacktestingPy** - Integration with backtesting.py library

All implementations inherit from `BaseBacktest` and provide a consistent API.

## Installation

### Required
```bash
pip install pandas numpy scikit-learn
```

### Optional (for additional backends)
```bash
# For Backtrader
pip install backtrader

# For backtesting.py
pip install backtesting
```

## Quick Start

```python
from src.backtesting import BacktestNoLib, BacktestBacktrader, BacktestBacktestingPy

# Initialize backtest
backtest = BacktestNoLib(
    initial_capital=10000.0,
    commission=0.001,
    position_size=1.0
)

# Run backtest
results = backtest.run(
    df=test_df,
    model=trained_model,
    scaler=fitted_scaler,
    feature_cols=feature_columns,
    price_col='close'
)

# Print results
backtest.print_results()

# Get metrics
metrics = backtest.get_metrics()
```

## Backends Comparison

### BacktestNoLib (Custom)

**Pros:**
- ✅ No external dependencies
- ✅ Simple and transparent logic
- ✅ Easy to customize
- ✅ Includes stop loss and take profit
- ✅ Fast execution

**Cons:**
- ❌ Manual implementation
- ❌ Limited built-in features

**Best for:**
- Quick prototyping
- Learning backtesting concepts
- Custom strategy logic

**Example:**
```python
from src.backtesting import BacktestNoLib

backtest = BacktestNoLib(
    initial_capital=10000.0,
    commission=0.001,
    position_size=1.0,
    stop_loss=0.05,      # 5% stop loss
    take_profit=0.10     # 10% take profit
)

results = backtest.run(df, model, scaler, feature_cols)
backtest.print_results()
```

### BacktestBacktrader

**Pros:**
- ✅ Event-driven backtesting
- ✅ Realistic order execution
- ✅ Live trading ready
- ✅ Extensive built-in features
- ✅ Large community

**Cons:**
- ❌ Requires backtrader library
- ❌ Steeper learning curve
- ❌ Slower than vectorized approaches

**Best for:**
- Realistic simulation
- Production strategies
- Live trading preparation
- Complex order types

**Example:**
```python
from src.backtesting import BacktestBacktrader

backtest = BacktestBacktrader(
    initial_capital=10000.0,
    commission=0.001,
    position_size=1.0
)

results = backtest.run(df, model, scaler, feature_cols)
backtest.print_results()
```

### BacktestBacktestingPy

**Pros:**
- ✅ Fast vectorized backtesting
- ✅ Built-in parameter optimization
- ✅ Interactive Bokeh visualizations
- ✅ Easy to use API
- ✅ 10-50x faster than event-driven

**Cons:**
- ❌ Requires backtesting library
- ❌ Less realistic than event-driven
- ❌ Limited order types

**Best for:**
- Parameter optimization
- Quick iterations
- Strategy comparison
- Visualization

**Example:**
```python
from src.backtesting import BacktestBacktestingPy

backtest = BacktestBacktestingPy(
    initial_capital=10000.0,
    commission=0.001,
    position_size=1.0
)

results = backtest.run(df, model, scaler, feature_cols, plot=True)
backtest.print_results()
```

## API Reference

### BaseBacktest (Abstract Base Class)

All backtest implementations inherit from this class.

#### Constructor Parameters

```python
__init__(
    initial_capital: float = 10000.0,
    commission: float = 0.001,
    position_size: float = 1.0
)
```

- `initial_capital`: Starting capital in dollars
- `commission`: Commission rate (e.g., 0.001 = 0.1%)
- `position_size`: Position size as fraction of capital (0.0 to 1.0)

#### Methods

**run(df, model, scaler, feature_cols, price_col='close', **kwargs)**

Run the backtest.

Parameters:
- `df`: DataFrame with OHLCV data and features
- `model`: Trained ML model with predict() method
- `scaler`: Fitted scaler with transform() method
- `feature_cols`: List of feature column names
- `price_col`: Name of price column (default: 'close')
- `**kwargs`: Backend-specific parameters

Returns:
- Dictionary with backtest results

**calculate_metrics()**

Calculate performance metrics.

Returns:
- Dictionary with metrics:
  - `initial_capital`: Starting capital
  - `final_capital`: Ending capital
  - `total_return`: Total return as decimal
  - `sharpe_ratio`: Sharpe ratio
  - `max_drawdown`: Maximum drawdown as decimal
  - `total_trades`: Number of trades
  - `win_rate`: Win rate as decimal

**print_results()**

Print formatted backtest results to console.

**get_results()**

Get raw backtest results.

Returns:
- Dictionary with results or None

**get_metrics()**

Get performance metrics.

Returns:
- Dictionary with metrics or None

**get_trades()**

Get list of trades.

Returns:
- List of trade dictionaries

**save_results(filepath)**

Save backtest results to JSON file.

Parameters:
- `filepath`: Path to save results

## Complete Example

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_provider import DataProvider
from src.features_generator import FeaturesGenerator
from src.models_lib import RandomForestModel
from src.backtesting import BacktestNoLib, BacktestBacktrader, BacktestBacktestingPy
from sklearn.preprocessing import StandardScaler

# Load data
data_provider = DataProvider(data_dir='data')
df = data_provider.load_yahoo('BTC-USD', '2020-01-01', '2024-11-24')

# Generate features
features_gen = FeaturesGenerator()
df_features = features_gen.generate_features(df, feature_set='advanced')
df_features = features_gen.create_target(
    df_features,
    target_type='classification',
    future_bars=5,
    threshold=0.02
)
df_features = df_features.dropna()

# Split data
train_df, val_df, test_df = data_provider.split_data(df_features)
feature_cols = features_gen.get_feature_names()

# Train model
X_train = train_df[feature_cols].values
y_train = train_df['target'].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = RandomForestModel(n_estimators=100, n_jobs=-1)
model.fit(X_train_scaled, y_train)

# Run backtests
backtests = {
    'NoLib': BacktestNoLib(initial_capital=10000, commission=0.001),
    'Backtrader': BacktestBacktrader(initial_capital=10000, commission=0.001),
    'BacktestingPy': BacktestBacktestingPy(initial_capital=10000, commission=0.001)
}

for name, backtest in backtests.items():
    print(f"\n{'='*80}")
    print(f"{name} Backtest")
    print(f"{'='*80}")
    
    try:
        results = backtest.run(
            df=test_df,
            model=model,
            scaler=scaler,
            feature_cols=feature_cols
        )
        backtest.print_results()
    except Exception as e:
        print(f"Error: {e}")
```

## Performance Comparison

Based on typical usage with 1000 bars:

| Backend | Execution Time | Speedup | Realism |
|---------|---------------|---------|---------|
| NoLib | ~0.5s | 1x | Medium |
| Backtrader | ~2.0s | 0.25x | High |
| BacktestingPy | ~0.1s | 5x | Medium |

## Metrics Explained

### Total Return
Percentage gain/loss from initial to final capital.

Formula: `(final_capital - initial_capital) / initial_capital`

### Sharpe Ratio
Risk-adjusted return metric. Higher is better.

Formula: `sqrt(252) * mean(returns) / std(returns)`

- < 1.0: Poor
- 1.0 - 2.0: Good
- > 2.0: Excellent

### Maximum Drawdown
Largest peak-to-trough decline. Lower is better.

Formula: `min((equity - cummax(equity)) / cummax(equity))`

### Win Rate
Percentage of profitable trades.

Formula: `winning_trades / total_trades`

## Troubleshooting

### ImportError: No module named 'backtrader'

Install Backtrader:
```bash
pip install backtrader
```

Or use BacktestNoLib which has no dependencies.

### ImportError: No module named 'backtesting'

Install backtesting.py:
```bash
pip install backtesting
```

Or use BacktestNoLib which has no dependencies.

### ValueError: DataFrame missing required columns

Ensure your DataFrame has these columns:
- `open`, `high`, `low`, `close`, `volume`
- Feature columns specified in `feature_cols`

### No trades executed

Check:
1. Model predictions are correct (0 or 1)
2. Features are properly scaled
3. DataFrame has enough data
4. Position size is > 0

## Best Practices

1. **Always use a scaler** - Scale features before backtesting
2. **Test on unseen data** - Use proper train/test split
3. **Include commission** - Realistic commission rates (0.001 = 0.1%)
4. **Compare backends** - Run multiple backends to validate results
5. **Check for overfitting** - Compare train vs test performance
6. **Use stop loss** - Protect against large losses
7. **Monitor drawdown** - Keep drawdown < 20%

## Advanced Usage

### Custom Stop Loss and Take Profit (NoLib only)

```python
backtest = BacktestNoLib(
    initial_capital=10000,
    commission=0.001,
    position_size=0.5,  # Use 50% of capital per trade
    stop_loss=0.03,     # 3% stop loss
    take_profit=0.08    # 8% take profit
)
```

### Visualization (BacktestingPy only)

```python
backtest = BacktestBacktestingPy(initial_capital=10000, commission=0.001)
results = backtest.run(df, model, scaler, feature_cols, plot=True)
# Opens interactive Bokeh plot in browser
```

### Save Results

```python
backtest.run(df, model, scaler, feature_cols)
backtest.save_results('results/backtest_results.json')
```

## See Also

- [btcusdt_backtest_comparison.py](../../btcusdt_backtest_comparison.py) - Complete comparison example
- [BaseBacktest](base_backtest.py) - Base class implementation
- [BacktestNoLib](backtest_nolib.py) - Custom implementation
- [BacktestBacktrader](backtest_backtrader.py) - Backtrader integration
- [BacktestBacktestingPy](backtest_backtesting_py.py) - backtesting.py integration

---

**Last Updated:** 2024-11-24  
**Version:** 1.0.0
