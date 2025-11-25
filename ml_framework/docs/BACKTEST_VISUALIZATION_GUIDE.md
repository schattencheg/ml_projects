# Backtest Visualization Guide

Complete guide to visualizing backtest results with equity curves comparison and OHLC charts with trade markers.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Quick Start](#quick-start)
4. [Visualization Types](#visualization-types)
5. [Usage Examples](#usage-examples)
6. [API Reference](#api-reference)
7. [Customization](#customization)
8. [Best Practices](#best-practices)

---

## Overview

The backtest visualization system provides comprehensive visual analysis of trading strategy performance:

- **Equity Curves Comparison**: Compare performance across multiple backtests on a single chart
- **OHLC Charts with Trades**: Visualize entry/exit points on price charts with candlesticks
- **Performance Metrics**: Side-by-side comparison of key metrics
- **Returns Distribution**: Analyze return patterns and risk

All visualizations are generated as interactive HTML reports using Plotly.

---

## Features

### 1. Equity Curves Comparison

**What it shows:**
- All backtest equity curves on one chart for easy comparison
- Initial capital reference line
- Hover information with final capital, return, Sharpe ratio, max drawdown

**Benefits:**
- Quickly identify best performing strategy
- Compare risk-adjusted returns visually
- Spot divergence points between strategies

### 2. OHLC Charts with Trade Markers

**What it shows:**
- Candlestick chart (or line chart if OHLC not available)
- **RED arrow UP** (▲) for long entry points
- **Colored CROSS** (✕) for exit points:
  - **GREEN** for profitable trades
  - **RED** for losing trades
- Hover information with trade details (price, PnL, return %, exit reason)

**Benefits:**
- Visualize exact entry/exit timing
- Identify patterns in winning/losing trades
- Validate strategy logic against price action

### 3. Performance Metrics Comparison

**Metrics displayed:**
- Total Return (%)
- Sharpe Ratio
- Maximum Drawdown (%)
- Win Rate (%)
- Total Trades

### 4. Individual Equity Curves

- Separate equity curve for each backtest
- Buy & Hold comparison (if available)
- Time-series view of capital growth

### 5. Returns Distribution

- Histogram of trade returns
- Identify return patterns
- Assess risk distribution

---

## Quick Start

### Basic Usage

```python
from src.managers.result_manager import ResultManager
from src.managers.visualization_manager import VisualizationManager

# Initialize managers
results_manager = ResultManager()
viz_manager = VisualizationManager()

# Run backtests and add results
for name, backtest in backtests.items():
    results = backtest.run(df, model, scaler, feature_cols)
    
    results_manager.add_backtest_results(
        model_name=name,
        results={
            'status': 'success',
            'equity_curve': results['equity_curve'],
            'trades': backtest.get_trades(),
            'metrics': backtest.calculate_metrics(),
            'initial_capital': 10000
        }
    )

# Prepare visualization data
viz_data = results_manager.prepare_backtest_visualization_data(df)

# Generate report with OHLC and trades
report_path = viz_manager.create_backtest_report(
    backtest_results=viz_data,
    save_dir=Path('results'),
    df=df  # Pass OHLC data for trade visualization
)

print(f"Report saved to: {report_path}")
```

---

## Visualization Types

### 1. Equity Curves Comparison Chart

**Location in report:** Near the top (after metrics comparison)

**Features:**
- Multiple equity curves overlaid
- Color-coded by backtest/model
- Initial capital reference line (gray dashed)
- Unified hover mode (shows all values at same time point)

**Interpretation:**
- **Higher curve** = Better performance
- **Steeper slope** = Faster capital growth
- **Smooth curve** = Consistent returns
- **Volatile curve** = Higher risk/variability

### 2. OHLC Chart with Trade Markers

**Location in report:** One chart per backtest (after individual equity curve)

**Trade Markers:**

**Entry Markers (Long Positions):**
- Symbol: Red triangle pointing UP (▲)
- Color: Red
- Hover info: Trade number, entry price, shares

**Exit Markers:**
- Symbol: Cross (✕)
- Color: Green (profit) or Red (loss)
- Hover info: Trade number, exit price, PnL, return %, exit reason

**Exit Reasons:**
- `signal`: Exit triggered by model prediction
- `stop_loss`: Stop loss hit
- `take_profit`: Take profit hit
- `end_of_data`: Position closed at end of backtest

**Interpretation:**
- **Cluster of green exits** = Successful trading period
- **Red exits after short holding** = Poor entry timing
- **Exits at highs/lows** = Good stop loss/take profit placement

### 3. Performance Metrics Bar Chart

**Metrics shown:**
- Total Return (%)
- Sharpe Ratio
- Max Drawdown (%)
- Win Rate (%)

**Interpretation:**
- Compare metrics side-by-side
- Identify trade-offs (e.g., high return but high drawdown)
- Select best risk-adjusted strategy

### 4. Individual Equity Curves

**Features:**
- Single equity curve per backtest
- Buy & Hold comparison (if available)
- Detailed time-series view

**Interpretation:**
- Assess consistency of returns over time
- Identify periods of drawdown
- Compare to buy & hold benchmark

### 5. Returns Distribution Histogram

**Features:**
- Histogram of individual trade returns
- 50 bins for detailed distribution

**Interpretation:**
- **Normal distribution** = Predictable returns
- **Right-skewed** = More large wins than large losses
- **Left-skewed** = More large losses (risky)
- **Fat tails** = Extreme outcomes possible

---

## Usage Examples

### Example 1: Basic Backtest Visualization

```python
import sys
from pathlib import Path
from src.data_provider import DataProvider
from src.features_generator import FeaturesGenerator
from src.models_lib import RandomForestModel
from src.backtesting import BacktestNoLib
from src.managers.result_manager import ResultManager
from src.managers.visualization_manager import VisualizationManager
from sklearn.preprocessing import StandardScaler

# Load data
data_provider = DataProvider(data_dir='data')
df = data_provider.load_yahoo('BTC-USD', '2023-01-01', '2024-11-25', '1d')

# Generate features
features_gen = FeaturesGenerator()
df_features = features_gen.generate_features(df)
df_features = features_gen.create_target(df_features, future_bars=15, threshold=0.02)

# Split and train
train_df, val_df, test_df = data_provider.split_data(df_features)
feature_cols = features_gen.get_feature_names()

scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols])
X_test = scaler.transform(test_df[feature_cols])

model = RandomForestModel()
model.fit(X_train, train_df['target'])

# Run backtest
backtest = BacktestNoLib(initial_capital=10000, commission=0.001)
results = backtest.run(test_df, model, scaler, feature_cols)

# Visualize
results_manager = ResultManager()
results_manager.add_backtest_results(
    'RandomForest',
    {
        'status': 'success',
        'equity_curve': results['equity_curve'],
        'trades': backtest.get_trades(),
        'metrics': backtest.calculate_metrics(),
        'initial_capital': 10000
    }
)

viz_manager = VisualizationManager()
viz_data = results_manager.prepare_backtest_visualization_data(test_df)
report_path = viz_manager.create_backtest_report(viz_data, Path('results'), test_df)

print(f"Report: {report_path}")
```

### Example 2: Compare Multiple Backtests

```python
# Run multiple backtests
backtests = {
    'NoLib': BacktestNoLib(initial_capital=10000, commission=0.001),
    'Backtrader': BacktestBacktrader(initial_capital=10000, commission=0.001),
    'BacktestingPy': BacktestBacktestingPy(initial_capital=10000, commission=0.001)
}

results_manager = ResultManager()

for name, backtest in backtests.items():
    results = backtest.run(test_df, model, scaler, feature_cols)
    
    results_manager.add_backtest_results(
        name,
        {
            'status': 'success',
            'equity_curve': results['equity_curve'],
            'trades': backtest.get_trades(),
            'metrics': backtest.calculate_metrics(),
            'initial_capital': 10000
        }
    )

# Generate comparison report
viz_manager = VisualizationManager()
viz_data = results_manager.prepare_backtest_visualization_data(test_df)
report_path = viz_manager.create_backtest_report(viz_data, Path('results'), test_df)

# All three backtests will be compared on the equity curves chart
# Each will have its own OHLC chart with trade markers
```

### Example 3: Without OHLC Data

```python
# If OHLC columns not available, visualization falls back to line chart
viz_data = results_manager.prepare_backtest_visualization_data(test_df)

# Pass df=None to skip OHLC charts entirely
report_path = viz_manager.create_backtest_report(viz_data, Path('results'), df=None)

# Or pass df with only 'close' column - will create line chart instead of candlesticks
```

---

## API Reference

### VisualizationManager

#### `create_equity_curves_comparison(backtest_results: Dict[str, Any]) -> go.Figure`

Create comparison plot of equity curves for all backtests.

**Parameters:**
- `backtest_results`: Dictionary with backtest results for each model/backend

**Returns:**
- Plotly figure with all equity curves

**Example:**
```python
fig = viz_manager.create_equity_curves_comparison(backtest_results)
fig.show()  # Display in browser
```

#### `create_ohlc_with_trades(df: pd.DataFrame, trades: List[Dict], model_name: str, price_col: str = 'close') -> go.Figure`

Create OHLC candlestick chart with trade entry/exit markers.

**Parameters:**
- `df`: DataFrame with OHLC data (must have 'open', 'high', 'low', 'close' columns)
- `trades`: List of trade dictionaries with 'entry_idx', 'exit_idx', 'entry_price', 'exit_price', 'pnl'
- `model_name`: Name of the model/backend
- `price_col`: Name of the price column (default: 'close')

**Returns:**
- Plotly figure with OHLC chart and trade markers

**Trade Dictionary Format:**
```python
{
    'entry_idx': 10,           # Index in DataFrame
    'exit_idx': 25,            # Index in DataFrame
    'entry_price': 45000.0,    # Entry price
    'exit_price': 47000.0,     # Exit price
    'shares': 0.5,             # Number of shares
    'pnl': 1000.0,             # Profit/Loss in dollars
    'exit_reason': 'signal'    # Reason for exit
}
```

**Example:**
```python
trades = backtest.get_trades()
fig = viz_manager.create_ohlc_with_trades(test_df, trades, 'RandomForest')
fig.show()
```

#### `create_backtest_report(backtest_results: Dict[str, Any], save_dir: Path, df: Optional[pd.DataFrame] = None) -> str`

Create comprehensive HTML report for backtest results.

**Parameters:**
- `backtest_results`: Backtest results dictionary
- `save_dir`: Directory to save report
- `df`: Optional DataFrame with OHLC data for trade visualization

**Returns:**
- Path to saved HTML file

**Example:**
```python
report_path = viz_manager.create_backtest_report(
    backtest_results=viz_data,
    save_dir=Path('results'),
    df=test_df
)
```

### ResultManager

#### `prepare_backtest_visualization_data(df: pd.DataFrame) -> Dict[str, Any]`

Prepare backtest results with additional data for visualization.

**Parameters:**
- `df`: DataFrame with OHLC data used in backtesting

**Returns:**
- Dictionary with backtest results enhanced with visualization data

**What it does:**
- Ensures trades list is included
- Adds OHLC data availability flag
- Ensures equity curve is array-like
- Calculates initial capital if missing

**Example:**
```python
viz_data = results_manager.prepare_backtest_visualization_data(test_df)
```

---

## Customization

### Customize Chart Appearance

```python
# Create figure
fig = viz_manager.create_equity_curves_comparison(backtest_results)

# Customize layout
fig.update_layout(
    title='My Custom Title',
    height=800,
    template='plotly_dark',  # Dark theme
    font=dict(size=14)
)

# Save as image
fig.write_image('equity_curves.png')

# Save as HTML
fig.write_html('equity_curves.html')
```

### Filter Trades by Criteria

```python
# Only show profitable trades
profitable_trades = [t for t in trades if t['pnl'] > 0]

fig = viz_manager.create_ohlc_with_trades(
    df=test_df,
    trades=profitable_trades,
    model_name='RandomForest - Profitable Only'
)
```

### Custom Trade Markers

```python
# Modify the create_ohlc_with_trades method or create custom markers
fig = go.Figure()

# Add candlestick
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close']
))

# Add custom entry markers
entry_indices = [trade['entry_idx'] for trade in trades]
entry_prices = [trade['entry_price'] for trade in trades]

fig.add_trace(go.Scatter(
    x=[df.index[i] for i in entry_indices],
    y=entry_prices,
    mode='markers',
    marker=dict(symbol='star', size=15, color='gold'),
    name='Entries'
))
```

---

## Best Practices

### 1. Data Preparation

✅ **DO:**
- Ensure OHLC columns are present for candlestick charts
- Use consistent date/time index across all DataFrames
- Include all necessary trade information (entry_idx, exit_idx, prices, PnL)

❌ **DON'T:**
- Mix different time periods across backtests
- Use DataFrames with missing OHLC data without fallback

### 2. Visualization

✅ **DO:**
- Generate equity curves comparison first for overview
- Include OHLC charts for detailed trade analysis
- Use consistent initial capital across backtests for fair comparison
- Save reports with descriptive names (include date, strategy name)

❌ **DON'T:**
- Compare backtests with different initial capital without normalization
- Overcrowd charts with too many backtests (max 5-7 recommended)

### 3. Interpretation

✅ **DO:**
- Look at multiple metrics (return, Sharpe, drawdown, win rate)
- Analyze trade distribution on OHLC charts
- Check for consistency across different time periods
- Validate strategy logic against price action

❌ **DON'T:**
- Focus only on total return (ignore risk)
- Ignore drawdown periods
- Overlook trade clustering patterns

### 4. Performance

✅ **DO:**
- Use `prepare_backtest_visualization_data()` to ensure data consistency
- Generate reports after all backtests complete
- Save reports to disk for later reference

❌ **DON'T:**
- Generate visualizations for each individual trade (too slow)
- Create reports with extremely large datasets (>10,000 trades)

### 5. Workflow

**Recommended workflow:**

1. **Run backtests** with multiple strategies/backends
2. **Add results** to ResultManager
3. **Prepare visualization data** with `prepare_backtest_visualization_data()`
4. **Generate report** with `create_backtest_report()`
5. **Analyze** equity curves comparison first
6. **Drill down** into OHLC charts for specific backtests
7. **Compare** metrics and returns distribution
8. **Iterate** on strategy based on insights

---

## Troubleshooting

### Issue: OHLC chart not showing

**Solution:**
- Ensure DataFrame has 'open', 'high', 'low', 'close' columns
- Check that df parameter is passed to `create_backtest_report()`
- Verify trades list is not empty

### Issue: Trade markers not appearing

**Solution:**
- Verify trades list contains 'entry_idx', 'exit_idx', 'entry_price', 'exit_price'
- Check that indices are within DataFrame bounds
- Ensure trades list is not empty

### Issue: Equity curves not aligned

**Solution:**
- Use same test dataset for all backtests
- Ensure all backtests start with same initial capital
- Check that equity_curve length matches DataFrame length

### Issue: Report generation fails

**Solution:**
- Check that all required data is present in backtest results
- Verify save_dir path exists or can be created
- Ensure backtest_results dictionary has correct structure

---

## Examples Output

### Equity Curves Comparison

```
Chart shows:
- NoLib: Green line, final capital $12,500 (+25%)
- Backtrader: Blue line, final capital $11,800 (+18%)
- BacktestingPy: Red line, final capital $12,200 (+22%)
- Initial Capital: Gray dashed line at $10,000

Hover shows:
- Final Capital: $12,500
- Total Return: 25.00%
- Sharpe Ratio: 1.85
- Max Drawdown: -8.50%
```

### OHLC Chart with Trades

```
Chart shows:
- Candlestick chart with green (up) and red (down) candles
- Red triangles (▲) at entry points
- Green crosses (✕) at profitable exits
- Red crosses (✕) at losing exits

Hover on entry shows:
Trade #1 - ENTRY
Price: $45,000.00
Shares: 0.50

Hover on exit shows:
Trade #1 - EXIT
Price: $47,000.00
PnL: $1,000.00
Return: 4.44%
Reason: take_profit
```

---

## Summary

The backtest visualization system provides:

1. **Comprehensive comparison** - See all backtests at once
2. **Detailed trade analysis** - Visualize every entry/exit
3. **Interactive exploration** - Hover for details, zoom, pan
4. **Professional reports** - HTML format for sharing
5. **Easy integration** - Works with all backtest backends

**Key Benefits:**
- ✅ Quickly identify best performing strategies
- ✅ Validate strategy logic visually
- ✅ Spot patterns in winning/losing trades
- ✅ Compare risk-adjusted returns
- ✅ Generate professional reports for stakeholders

**Next Steps:**
1. Run `example_backtest_visualization.py` to see it in action
2. Integrate into your workflow
3. Customize visualizations for your needs
4. Share reports with your team

---

**Happy Backtesting! 📊📈**
