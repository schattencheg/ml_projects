# Backtest Results & ML Training Guide

## Overview

The `run_me.py` script now includes functionality to:
1. **Save backtest results** to disk for reuse
2. **Load existing results** instead of re-running backtests
3. **Prepare ML training data** from trade results
4. **Separate profitable and unprofitable trades** for analysis

## How It Works

### 1. Results Storage Structure

All results are saved in the `Results/{strategy_name}/` directory:

```
Results/
└── SmaCrossStrategy/
    ├── trades.csv                  # All trades with full details
    ├── trades_profitable.csv       # Only profitable trades
    ├── trades_unprofitable.csv     # Only unprofitable trades
    ├── ml_training_data.csv        # Features + labels for ML
    └── metrics.json                # Performance metrics
```

### 2. Workflow

#### First Run (No Saved Results)
1. Loads data from DataProvider
2. Runs backtest with the strategy
3. Collects all trade data during execution
4. Saves results to `Results/{strategy_name}/`
5. Prepares ML training data
6. Displays statistics

#### Subsequent Runs (Results Exist)
1. Checks if results exist for the strategy
2. Loads saved results from disk
3. Skips backtest execution
4. Prepares ML training data from loaded results
5. Displays statistics

### 3. Trade Data Captured

Each trade includes:
- `entry_date` - When the trade was opened
- `exit_date` - When the trade was closed
- `entry_price` - Entry price
- `exit_price` - Exit price
- `size` - Trade size (0.001 BTC by default)
- `pnl` - Profit/loss (gross)
- `pnlcomm` - Profit/loss (net of commission)
- `is_profitable` - Boolean flag
- `sma_fast_entry` - Fast SMA value at entry
- `sma_slow_entry` - Slow SMA value at entry

### 4. ML Training Data

The `ml_training_data.csv` file contains:
- **Features**: `entry_price`, `size`, `sma_fast_entry`, `sma_slow_entry`
- **Label**: `1` for profitable trades, `0` for unprofitable trades

This data can be used to train ML models to predict trade profitability.

## Usage

### Basic Usage

```python
from run_me import MyMLBactester

# Create backtester instance
ml_backtester = MyMLBactester()

# Load data
ml_backtester.initialize()

# Run backtest (or load existing results)
ml_backtester.run('SmaCrossStrategy')
```

### Adding New Strategies

1. Define your strategy class (inheriting from `bt.Strategy`)
2. Add it to the `strategies` dictionary in `MyMLBactester.__init__`:

```python
self.strategies = {
    'SmaCrossStrategy': SmaCrossStrategy,
    'YourNewStrategy': YourNewStrategy,
}
```

3. Make sure your strategy tracks trade data:

```python
def __init__(self):
    # ... your indicators ...
    self.trade_data = []  # Important!

def notify_trade(self, trade):
    if not trade.isclosed:
        return
    
    trade_info = {
        'entry_date': trade.dtopen,
        'exit_date': trade.dtclose,
        'entry_price': trade.price,
        'exit_price': trade.price + trade.pnl / trade.size,
        'size': trade.size,
        'pnl': trade.pnl,
        'pnlcomm': trade.pnlcomm,
        'is_profitable': trade.pnl > 0,
        # Add your custom features here
    }
    self.trade_data.append(trade_info)
```

### Customizing Trade Size

The default trade size is 0.001 BTC. To change it:

```python
class SmaCrossStrategy(bt.Strategy):
    params = (
        ('fast_period', 10),
        ('slow_period', 30),
        ('trade_size', 0.01),  # Change to 0.01 BTC
    )
```

### Force Re-run Backtest

To force a re-run even if results exist, delete the strategy's results folder:

```python
import shutil
shutil.rmtree('Results/SmaCrossStrategy')
```

Or manually delete the folder.

## ML Training Example

Here's a simple example of training an ML model using the prepared data:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load ML training data
ml_data = pd.read_csv('Results/SmaCrossStrategy/ml_training_data.csv')

# Separate features and labels
X = ml_data[['entry_price', 'size', 'sma_fast_entry', 'sma_slow_entry']]
y = ml_data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance)
```

## Performance Metrics

The `metrics.json` file contains:
- `final_value` - Final portfolio value
- `profit_loss` - Total profit/loss
- `return_pct` - Return percentage
- `sharpe_ratio` - Sharpe ratio
- `max_drawdown` - Maximum drawdown percentage
- `total_return` - Total return
- `avg_return` - Average return
- `total_trades` - Total number of trades
- `won_trades` - Number of winning trades
- `lost_trades` - Number of losing trades
- `avg_win` - Average winning trade P&L
- `avg_loss` - Average losing trade P&L

## Benefits

1. **Faster Iteration**: No need to re-run backtests when experimenting with ML models
2. **Reproducibility**: Same results every time for the same strategy
3. **ML Ready**: Data is pre-formatted for ML training
4. **Analysis**: Separate files for profitable/unprofitable trades make analysis easier
5. **Organized**: All artifacts for each strategy are in their own folder

## Next Steps

1. Run your first backtest to generate results
2. Analyze the profitable vs unprofitable trades
3. Train ML models to predict trade profitability
4. Use ML predictions to filter trades in your strategy
5. Iterate and improve!
