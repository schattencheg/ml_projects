# ML Backtest - Simple Trading Strategy Example

This project demonstrates simple trading strategies using the `backtesting.py` module with data provided by the custom `DataProvider` module.

> 🚀 **Quick Start:** [**GETTING_STARTED.md**](GETTING_STARTED.md) - Get your first backtest running in 5 minutes!
> 
> 📖 **New to backtesting?** Check out the [**EXAMPLES_GUIDE.md**](EXAMPLES_GUIDE.md) for a detailed walkthrough of all examples!
>
> 📊 **MLflow Tracking:** [**MLFLOW_GUIDE.md**](MLFLOW_GUIDE.md) - Track all your ML experiments with structured naming!

## Strategy Overview

The example implements a **Moving Average Crossover Strategy**:
- **Buy Signal**: When the fast moving average (10-period) crosses above the slow moving average (30-period)
- **Sell Signal**: When the fast moving average crosses below the slow moving average

## Project Structure

```
ml_backtest/
├── src/
│   ├── Data/
│   │   └── DataProvider.py    # Data fetching and management
│   └── Background/
│       └── enums.py            # Data resolution and period enums
├── Output/                     # Generated HTML reports and plots
├── data/                       # Downloaded market data (auto-created)
├── run_me.py                   # Main strategy example
├── simple_strategy_example.py  # Simple beginner example
├── optimize_strategy_example.py # Parameter optimization example
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Simple Example (Recommended for Beginners)
Run the simple example with SPY ETF (lower capital requirements):
```bash
python simple_strategy_example.py
```

This example:
- Uses SPY (S&P 500 ETF) - much more affordable than BTC
- Requires only $10,000 starting capital
- Uses 10/20 day moving average crossover
- Provides clear, formatted output

### Option 2: Bitcoin Example
Run the BTC example (requires higher capital):
```bash
python run_me.py
```

This example:
1. Loads 2 years of BTC-USD daily data using DataProvider
2. Runs the Moving Average Crossover strategy (10/30 periods)
3. Displays detailed backtest results
4. Generates an interactive plot in your browser

## Customization

You can easily modify the strategy parameters in `run_me.py`:

```python
class SmaCrossStrategy(Strategy):
    fast_period = 10  # Change fast MA period
    slow_period = 30  # Change slow MA period
```

Or change the data source:
```python
data_provider = DataProvider(
    tickers=['ETH-USD'],           # Different ticker
    resolution=DataResolution.HOUR, # Different timeframe
    period=DataPeriod.YEAR_01       # Different period
)
```

## Key Features

- **Simple and Clean**: Minimal code, easy to understand
- **Modular Design**: Uses DataProvider for data management
- **Visual Results**: Interactive plots with Bokeh
- **Extensible**: Easy to add more indicators or modify strategy logic

## MLflow Experiment Tracking

This project includes comprehensive MLflow integration for tracking all your ML experiments:

### Features
- **Structured Naming**: Consistent experiment and run naming across projects
- **Automatic Logging**: Track parameters, metrics, and models automatically
- **Backtest Integration**: Log backtest results directly to MLflow
- **Multi-Project Support**: Organize experiments across different ML projects

### Quick Example
```python
from mlflow_tracker import setup_mlflow_tracker

tracker = setup_mlflow_tracker()

with tracker.start_run(
    model_type="regression",
    asset_or_timeframe="btc_usd_daily",
    model_name="linear_regression"
):
    tracker.log_model_metrics({"rmse": 0.05, "r2": 0.85})
    tracker.log_sklearn_model(model, "model")
```

See [**MLFLOW_GUIDE.md**](MLFLOW_GUIDE.md) for complete documentation.

## Next Steps

- Try different moving average periods
- Add stop-loss and take-profit levels
- Implement other strategies (RSI, MACD, Bollinger Bands)
- Optimize parameters using `bt.optimize()`
- Test on different assets and timeframes
- Track experiments with MLflow
