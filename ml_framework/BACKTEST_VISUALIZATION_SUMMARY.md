# Backtest Visualization Features - Summary

## Overview

Added comprehensive visualization capabilities for backtest results, including equity curves comparison and OHLC charts with trade entry/exit markers.

---

## What Was Added

### 1. New Visualization Methods

**VisualizationManager** (`src/managers/visualization_manager.py`):

#### `create_equity_curves_comparison(backtest_results)`
- Compares all backtest equity curves on a single chart
- Shows initial capital reference line
- Interactive hover with metrics (final capital, return, Sharpe, drawdown)
- Color-coded by backtest/model
- Height: 600px, professional styling

#### `create_ohlc_with_trades(df, trades, model_name, price_col)`
- OHLC candlestick chart (or line chart fallback)
- **RED arrow UP (▲)** for long entry points
- **Colored CROSS (✕)** for exit points:
  - GREEN for profitable trades
  - RED for losing trades
- Hover info: trade number, prices, PnL, return %, exit reason
- Height: 600px, rangeslider disabled

#### Updated `create_backtest_report(backtest_results, save_dir, df)`
- Added optional `df` parameter for OHLC data
- Includes equity curves comparison chart
- Generates OHLC chart with trades for each backtest
- Maintains all existing functionality

### 2. Data Preparation Method

**ResultManager** (`src/managers/result_manager.py`):

#### `prepare_backtest_visualization_data(df)`
- Ensures trades list is included
- Checks OHLC data availability
- Validates equity curve format
- Calculates initial capital if missing
- Returns enhanced results dictionary

### 3. Example Script

**`example_backtest_visualization.py`**:
- Complete end-to-end example
- Loads BTC-USD data
- Generates features
- Trains Random Forest model
- Runs 3 backtests (NoLib, Backtrader, BacktestingPy)
- Generates comprehensive visualization report
- ~280 lines with detailed comments

### 4. Documentation

**`docs/BACKTEST_VISUALIZATION_GUIDE.md`** (~600 lines):
- Complete usage guide
- Visualization types explained
- API reference
- Usage examples
- Customization options
- Best practices
- Troubleshooting

---

## Key Features

### Equity Curves Comparison

**What it shows:**
```
- All backtest equity curves overlaid
- Initial capital reference line
- Color-coded by strategy
- Hover: Final capital, return, Sharpe, drawdown
```

**Benefits:**
- Quick performance comparison
- Visual risk assessment
- Identify divergence points
- Spot best risk-adjusted strategy

### OHLC Charts with Trade Markers

**Trade Markers:**
- **Entry**: Red triangle UP (▲)
- **Exit (Profit)**: Green cross (✕)
- **Exit (Loss)**: Red cross (✕)

**Hover Information:**
- Entry: Trade #, price, shares
- Exit: Trade #, price, PnL, return %, exit reason

**Exit Reasons:**
- `signal`: Model prediction
- `stop_loss`: Stop loss hit
- `take_profit`: Take profit hit
- `end_of_data`: End of backtest

**Benefits:**
- Visualize exact entry/exit timing
- Identify winning/losing patterns
- Validate strategy logic
- Spot clustering of good/bad trades

---

## Usage

### Basic Usage

```python
from src.managers.result_manager import ResultManager
from src.managers.visualization_manager import VisualizationManager

# Initialize
results_manager = ResultManager()
viz_manager = VisualizationManager()

# Run backtests and add results
for name, backtest in backtests.items():
    results = backtest.run(df, model, scaler, feature_cols)
    
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

# Prepare and visualize
viz_data = results_manager.prepare_backtest_visualization_data(df)
report_path = viz_manager.create_backtest_report(viz_data, Path('results'), df)
```

### Run Example

```bash
python example_backtest_visualization.py
```

**Output:**
- Loads BTC-USD data (2023-2024)
- Trains Random Forest model
- Runs 3 backtests
- Generates HTML report with:
  - Equity curves comparison
  - 3 OHLC charts with trades
  - Performance metrics
  - Returns distributions

---

## File Changes

### Modified Files

1. **src/managers/visualization_manager.py** (+210 lines)
   - Added `create_equity_curves_comparison()` method
   - Added `create_ohlc_with_trades()` method
   - Updated `create_backtest_report()` to include new charts

2. **src/managers/result_manager.py** (+44 lines)
   - Added `prepare_backtest_visualization_data()` method

### New Files

1. **example_backtest_visualization.py** (~280 lines)
   - Complete working example
   - Demonstrates all features
   - Well-commented code

2. **docs/BACKTEST_VISUALIZATION_GUIDE.md** (~600 lines)
   - Comprehensive documentation
   - API reference
   - Usage examples
   - Best practices

3. **BACKTEST_VISUALIZATION_SUMMARY.md** (this file)
   - Quick reference
   - Feature overview
   - Usage summary

---

## Integration

### Works With

✅ All backtest backends:
- BacktestNoLib (custom)
- BacktestBacktrader (Backtrader library)
- BacktestBacktestingPy (backtesting.py library)

✅ All ML models:
- LogisticRegression
- RandomForest
- XGBoost
- LightGBM
- Neural networks
- Any custom model

✅ Existing workflow:
- ResultManager
- VisualizationManager
- DataProvider
- FeaturesGenerator

### Backward Compatible

✅ No breaking changes
✅ Optional `df` parameter (defaults to None)
✅ Existing reports still work
✅ Graceful fallback if OHLC data unavailable

---

## Visualization Output

### Report Structure

```
backtest_report.html
├── 1. Performance Metrics Comparison (bar chart)
├── 2. Equity Curves Comparison (line chart - ALL backtests)
├── 3. Individual Results (for each backtest):
│   ├── Individual Equity Curve
│   ├── OHLC Chart with Trades (NEW)
│   └── Returns Distribution
```

### Chart Specifications

**Equity Curves Comparison:**
- Type: Line chart
- Height: 600px
- Features: Hover, zoom, pan, legend
- Colors: Auto-assigned per backtest
- Reference line: Initial capital (gray dashed)

**OHLC with Trades:**
- Type: Candlestick (or line fallback)
- Height: 600px
- Entry markers: Red triangle UP, size 12
- Exit markers: Colored cross, size 12
- Features: Hover, zoom, pan, no rangeslider
- Colors: Green (profit), Red (loss)

---

## Benefits

### For Traders

✅ **Visual validation** - See if strategy makes sense
✅ **Pattern recognition** - Spot winning/losing patterns
✅ **Risk assessment** - Compare drawdowns visually
✅ **Quick comparison** - All strategies on one chart

### For Developers

✅ **Easy integration** - 3 lines of code
✅ **Flexible** - Works with any backtest backend
✅ **Customizable** - Plotly figures can be modified
✅ **Professional** - Publication-ready charts

### For Teams

✅ **Shareable reports** - HTML format
✅ **Interactive** - Hover, zoom, explore
✅ **Comprehensive** - All metrics in one place
✅ **Professional** - Impress stakeholders

---

## Examples

### Example 1: Single Backtest

```python
# Run backtest
backtest = BacktestNoLib(initial_capital=10000)
results = backtest.run(test_df, model, scaler, feature_cols)

# Visualize
results_manager.add_backtest_results('Strategy1', {
    'status': 'success',
    'equity_curve': results['equity_curve'],
    'trades': backtest.get_trades(),
    'metrics': backtest.calculate_metrics(),
    'initial_capital': 10000
})

viz_data = results_manager.prepare_backtest_visualization_data(test_df)
report = viz_manager.create_backtest_report(viz_data, Path('results'), test_df)
```

### Example 2: Compare Multiple Backtests

```python
# Run multiple backtests
backtests = {
    'NoLib': BacktestNoLib(...),
    'Backtrader': BacktestBacktrader(...),
    'BacktestingPy': BacktestBacktestingPy(...)
}

for name, backtest in backtests.items():
    results = backtest.run(test_df, model, scaler, feature_cols)
    results_manager.add_backtest_results(name, {...})

# Generate comparison report
viz_data = results_manager.prepare_backtest_visualization_data(test_df)
report = viz_manager.create_backtest_report(viz_data, Path('results'), test_df)

# Report includes:
# - Equity curves comparison (all 3 on one chart)
# - 3 separate OHLC charts with trades
# - Metrics comparison
# - Returns distributions
```

### Example 3: Without OHLC Data

```python
# If OHLC not available, pass df=None
report = viz_manager.create_backtest_report(viz_data, Path('results'), df=None)

# Or use DataFrame with only 'close' column
# Will create line chart instead of candlesticks
```

---

## API Quick Reference

### VisualizationManager

```python
# Create equity curves comparison
fig = viz_manager.create_equity_curves_comparison(backtest_results)

# Create OHLC with trades
fig = viz_manager.create_ohlc_with_trades(df, trades, 'ModelName')

# Create complete report
report_path = viz_manager.create_backtest_report(
    backtest_results=viz_data,
    save_dir=Path('results'),
    df=test_df  # Optional, for OHLC charts
)
```

### ResultManager

```python
# Prepare visualization data
viz_data = results_manager.prepare_backtest_visualization_data(df)

# Add backtest results
results_manager.add_backtest_results(
    model_name='Strategy1',
    results={
        'status': 'success',
        'equity_curve': [...],
        'trades': [...],
        'metrics': {...},
        'initial_capital': 10000
    }
)
```

---

## Trade Dictionary Format

```python
{
    'entry_idx': 10,           # Index in DataFrame
    'exit_idx': 25,            # Index in DataFrame
    'entry_price': 45000.0,    # Entry price
    'exit_price': 47000.0,     # Exit price
    'shares': 0.5,             # Number of shares
    'pnl': 1000.0,             # Profit/Loss ($)
    'exit_reason': 'signal'    # 'signal', 'stop_loss', 'take_profit', 'end_of_data'
}
```

---

## Best Practices

### Data Preparation

✅ Ensure OHLC columns present for candlestick charts
✅ Use consistent date/time index
✅ Include all trade information (indices, prices, PnL)

### Visualization

✅ Generate equity curves comparison first
✅ Use consistent initial capital for fair comparison
✅ Limit to 5-7 backtests per comparison chart
✅ Save reports with descriptive names

### Interpretation

✅ Look at multiple metrics (return, Sharpe, drawdown)
✅ Analyze trade distribution on OHLC charts
✅ Check for consistency across time periods
✅ Validate strategy logic against price action

---

## Troubleshooting

**OHLC chart not showing:**
- Ensure df has 'open', 'high', 'low', 'close' columns
- Pass df parameter to create_backtest_report()

**Trade markers not appearing:**
- Verify trades list has required fields
- Check indices are within DataFrame bounds

**Equity curves not aligned:**
- Use same test dataset for all backtests
- Ensure same initial capital

---

## Next Steps

1. **Try the example:**
   ```bash
   python example_backtest_visualization.py
   ```

2. **Read the guide:**
   - `docs/BACKTEST_VISUALIZATION_GUIDE.md`

3. **Integrate into your workflow:**
   - Add to existing backtest scripts
   - Customize visualizations
   - Share reports with team

4. **Explore customization:**
   - Modify chart appearance
   - Add custom markers
   - Create custom reports

---

## Summary

**Total Code Added:** ~254 lines
**Total Documentation:** ~600 lines
**New Files:** 3 (example + docs + summary)
**Modified Files:** 2 (VisualizationManager + ResultManager)

**Key Features:**
- ✅ Equity curves comparison (all backtests on one chart)
- ✅ OHLC charts with trade markers (entry/exit visualization)
- ✅ Interactive HTML reports (hover, zoom, pan)
- ✅ Professional styling (publication-ready)
- ✅ Easy integration (3 lines of code)
- ✅ Backward compatible (no breaking changes)

**Result:** Comprehensive backtest visualization system that makes it easy to compare strategies, validate logic, and generate professional reports.

---

**Happy Backtesting! 📊📈**
