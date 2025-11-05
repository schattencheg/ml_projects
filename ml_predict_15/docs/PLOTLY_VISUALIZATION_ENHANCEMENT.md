# Plotly Visualization Enhancement Plan

## Overview

This document outlines the enhancement of the backtesting visualization system to include ALL requested plots in separate, organized Plotly HTML files.

## Current Status

The `BacktestBase.py` currently has:
- A single comprehensive Plotly visualization with 12 plots in one HTML file
- Matplotlib fallback methods for individual plot categories
- `_plot_trade_distribution()` matplotlib method

## Requested Enhancements

Create separate Plotly HTML files for each category with ALL plots included:

### 1. Performance Overview (`_plot_performance_overview_plotly`)
- ✅ Equity curve
- ✅ Drawdown (underwater curve)
- ✅ Returns distribution
- ✅ Performance metrics summary table

### 2. Trade Analysis (`_plot_trade_analysis_plotly`)
- ✅ Trade P&L over time (cumulative)
- ✅ Win/Loss distribution (histogram)
- ✅ Trade duration analysis (box plot)
- ✅ Monthly performance (bar chart)

### 3. Risk Analysis (`_plot_risk_analysis_plotly`)
- ✅ Rolling Sharpe ratio
- ✅ Volatility analysis (rolling volatility)
- ✅ Underwater curve (drawdown over time)
- ✅ Risk-Return scatter (trades plotted by duration vs P&L)

### 4. Monthly Heatmap (`_plot_monthly_heatmap_plotly`)
- ✅ Monthly returns heatmap (year x month grid)

### 5. Trade Distribution (`_plot_trade_distribution_plotly`)
- ✅ P&L distribution (histogram)
- ✅ Box plot of wins vs losses
- ✅ Cumulative distribution function (CDF)
- ✅ Q-Q plot (normal distribution test)

## Implementation Strategy

### Step 1: Remove Old Code
Remove lines 490-866 in `BacktestBase.py` which contain the old single-file Plotly implementation.

### Step 2: Add New Methods
Add 5 new methods after the `create_comprehensive_visualizations` method:

1. `_plot_performance_overview_plotly()` - Lines ~490-650
2. `_plot_trade_analysis_plotly()` - Lines ~650-850
3. `_plot_risk_analysis_plotly()` - Lines ~850-1050
4. `_plot_monthly_heatmap_plotly()` - Lines ~1050-1150
5. `_plot_trade_distribution_plotly()` - Lines ~1150-1350

### Step 3: Update Main Method
The `create_comprehensive_visualizations()` method (lines 419-488) now calls all 5 methods:

```python
def create_comprehensive_visualizations(self, results, df=None, save_dir=None, show_plots=True, model_name=None):
    # ... setup code ...
    
    saved_files.update(self._plot_performance_overview_plotly(results, df, save_dir, show_plots, model_name))
    saved_files.update(self._plot_trade_analysis_plotly(results, save_dir, show_plots, model_name))
    saved_files.update(self._plot_risk_analysis_plotly(results, save_dir, show_plots, model_name))
    saved_files.update(self._plot_monthly_heatmap_plotly(results, save_dir, show_plots, model_name))
    saved_files.update(self._plot_trade_distribution_plotly(results, save_dir, show_plots, model_name))
    
    return saved_files
```

## Benefits

1. **Better Organization**: Each analysis category in its own HTML file
2. **Faster Loading**: Smaller files load faster in browser
3. **Easier Navigation**: Users can focus on specific analysis areas
4. **Better Performance**: Plotly handles smaller plots better
5. **Complete Coverage**: ALL requested plots are included

## Output Files

After running backtesting with visualizations, you'll get 5 HTML files:

1. `performance_overview_{model}_{timestamp}.html` - 4 plots (2x2 grid)
2. `trade_analysis_{model}_{timestamp}.html` - 4 plots (2x2 grid)
3. `risk_analysis_{model}_{timestamp}.html` - 4 plots (2x2 grid)
4. `monthly_heatmap_{model}_{timestamp}.html` - 1 plot (heatmap)
5. `trade_distribution_{model}_{timestamp}.html` - 4 plots (2x2 grid)

**Total: 17 interactive Plotly visualizations across 5 HTML files**

## Usage

```python
from src.BacktestNoLib import MLBacktester

# Run backtest
backtest = MLBacktester(...)
results, trades_df = backtest.run_backtest(df, model, scaler)

# Create all visualizations
saved_files = backtest.create_comprehensive_visualizations(
    results=results,
    df=df,
    save_dir='backtest_results',
    show_plots=True,  # Opens all 5 files in browser
    model_name='xgboost'
)

# Access individual files
print(saved_files['performance_overview'])
print(saved_files['trade_analysis'])
print(saved_files['risk_analysis'])
print(saved_files['monthly_heatmap'])
print(saved_files['trade_distribution'])
```

## Next Steps

Due to the large size of the implementation (1000+ lines of code), I recommend:

1. **Option A**: I can create a complete new version of `BacktestBase.py` with all methods implemented
2. **Option B**: I can create a separate file `BacktestPlotly.py` with all Plotly methods that can be mixed into the base class
3. **Option C**: I can provide the complete code for each method individually and you can copy-paste them

Which approach would you prefer?
