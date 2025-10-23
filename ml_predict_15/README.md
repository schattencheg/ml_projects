# ML Price Prediction with Backtesting

A comprehensive machine learning framework for cryptocurrency price prediction with integrated backtesting and trailing stop loss functionality.

## 🚀 Features

- **13 ML Models**: From Logistic Regression to LSTM/CNN neural networks
- **Advanced Backtesting**: Test strategies with trained ML models
- **Trailing Stop Loss**: Dynamic risk management
- **Feature Engineering**: 50+ technical indicators
- **Walk-Forward Testing**: Train on historical data, test on future data
- **Comprehensive Metrics**: Returns, win rate, Sharpe ratio, max drawdown, and more
- **Visualization**: Beautiful charts for analysis

## 📋 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train ML Models

```bash
python run_me.py
```

This will train 13 different ML models on your data and save the best performers.

### 3. Run Backtesting

```bash
python backtest_quick_start.py
```

This will:
- Load trained models
- Run backtest with trailing stop loss
- Display comprehensive results
- Generate visualization plots

## 🎯 Backtesting with Trailing Stop Loss

The backtesting module uses ML model predictions as trading signals and implements trailing stop loss for risk management.

### Basic Example

```python
from src.MLBacktester import MLBacktester

# Initialize backtester
backtester = MLBacktester(
    initial_capital=10000.0,
    position_size=1.0,              # Use 100% of capital
    trailing_stop_pct=2.0,          # 2% trailing stop
    take_profit_pct=5.0,            # 5% take profit
    probability_threshold=0.6       # 60% confidence threshold
)

# Run backtest
results = backtester.run_backtest(
    df=df_test,
    model=trained_model,
    scaler=fitted_scaler,
    X_columns=feature_columns
)

# Display results
backtester.print_results(results)
backtester.plot_results(results, df_test, save_path='plots/backtest.png')
```

### How Trailing Stop Loss Works

1. **Entry**: When model predicts price increase with high confidence
2. **Initial Stop**: Set at entry_price × (1 - trailing_stop_pct/100)
3. **Price Rises**: Stop loss moves up to highest_price × (1 - trailing_stop_pct/100)
4. **Price Falls**: Stop loss stays at highest level (never moves down)
5. **Exit**: Position closes when price hits trailing stop or take profit

**Example with 2% trailing stop:**
- Entry at $100 → Stop at $98
- Price rises to $110 → Stop moves to $107.80
- Price rises to $120 → Stop moves to $117.60
- Price falls to $116 → Stop stays at $117.60
- Position exits at $117.60 with profit

## 📊 Available Models

### Traditional ML Models
1. Logistic Regression
2. Ridge Classifier
3. Naive Bayes
4. K-Nearest Neighbors
5. Decision Tree
6. Random Forest
7. Gradient Boosting
8. Support Vector Machine

### Advanced ML Models
9. XGBoost
10. LightGBM

### Neural Networks
11. LSTM (Long Short-Term Memory)
12. CNN (Convolutional Neural Network)
13. LSTM-CNN Hybrid

## 📁 Project Structure

```
ml_predict_15/
├── src/
│   ├── FeaturesGenerator.py      # Technical indicators
│   └── MLBacktester.py            # Backtesting engine
├── data/
│   ├── btc_2022.csv              # Training data
│   └── btc_2023.csv              # Test data
├── plots/                         # Generated visualizations
├── models/                        # Saved models
├── run_me.py                      # Train ML models
├── backtest_quick_start.py       # Quick backtesting example
├── backtest_example.py           # Advanced backtesting examples
├── BACKTEST_GUIDE.md             # Detailed backtesting guide
└── README_MODELS.md              # Model descriptions

```

## 🎓 Examples

### Example 1: Quick Start

```bash
python backtest_quick_start.py
```

Simple example that trains models and runs a basic backtest.

### Example 2: Advanced Backtesting

```bash
python backtest_example.py
```

Comprehensive example that:
- Compares different trailing stop percentages
- Tests all models
- Compares conservative vs aggressive strategies
- Generates multiple visualization plots

### Example 3: Custom Strategy

```python
# Conservative strategy
backtester = MLBacktester(
    initial_capital=10000.0,
    position_size=0.5,              # Use 50% of capital
    trailing_stop_pct=1.5,          # Tight stop
    take_profit_pct=4.0,            # Conservative target
    probability_threshold=0.7       # High confidence only
)

# Aggressive strategy
backtester = MLBacktester(
    initial_capital=10000.0,
    position_size=1.0,              # Use 100% of capital
    trailing_stop_pct=3.0,          # Wider stop
    take_profit_pct=8.0,            # Higher target
    probability_threshold=0.55      # Lower confidence threshold
)
```

## 📈 Performance Metrics

The backtester calculates:

- **Return Metrics**: Total return, return %, buy & hold comparison
- **Trade Metrics**: Total trades, win rate, avg bars held
- **Risk Metrics**: Max drawdown, Sharpe ratio, profit factor
- **Trade Analysis**: Average win, average loss, winning/losing trades

## 🎨 Visualizations

The backtester generates three plots:

1. **Price Chart**: Shows entry/exit signals on price data
2. **Equity Curve**: Portfolio value over time
3. **Drawdown Chart**: Visualizes risk exposure

## 🔧 Configuration

### Backtester Parameters

```python
MLBacktester(
    initial_capital=10000.0,        # Starting capital
    position_size=1.0,              # Fraction of capital per trade (0-1)
    trailing_stop_pct=2.0,          # Trailing stop percentage
    take_profit_pct=None,           # Take profit percentage (optional)
    commission=0.001,               # Commission per trade (0.1%)
    slippage=0.0005,                # Slippage per trade (0.05%)
    use_probability_threshold=True, # Use probability threshold
    probability_threshold=0.6,      # Min probability to enter (0-1)
    max_holding_bars=None           # Max bars to hold (optional)
)
```

## 📚 Documentation

- **[BACKTEST_GUIDE.md](BACKTEST_GUIDE.md)**: Comprehensive backtesting guide
- **[README_MODELS.md](README_MODELS.md)**: Detailed model descriptions

## 🛠️ Requirements

```
pandas
numpy
scikit-learn
matplotlib
tensorflow>=2.10.0
xgboost>=1.6.0
lightgbm>=3.3.0
joblib
```

## 💡 Best Practices

1. **Walk-Forward Testing**: Always test on out-of-sample data
2. **Parameter Optimization**: Test multiple parameter combinations
3. **Risk Management**: Consider risk-adjusted returns, not just total return
4. **Realistic Costs**: Use realistic commission and slippage values
5. **Model Selection**: Choose models based on backtest performance

## 🎯 Typical Workflow

1. **Prepare Data**: Load and clean OHLCV data
2. **Generate Features**: Create technical indicators
3. **Train Models**: Train multiple ML models
4. **Select Best Model**: Choose based on training metrics
5. **Backtest**: Test on out-of-sample data with trailing stop
6. **Optimize**: Adjust parameters for better performance
7. **Validate**: Test on additional time periods

## 📊 Example Results

```
BACKTEST RESULTS
================================================================================

Capital:
  Initial Capital:        $10,000.00
  Final Capital:          $12,450.00
  Total Return:           $2,450.00
  Total Return %:         24.50%
  Buy & Hold Return %:    18.30%

Trades:
  Total Trades:           45
  Winning Trades:         28
  Losing Trades:          17
  Win Rate:               62.22%

Profit/Loss:
  Average Win:            $150.25
  Average Loss:           -$85.40
  Profit Factor:          2.15

Risk Metrics:
  Max Drawdown:           -8.50%
  Sharpe Ratio:           1.85
```

## 🚀 Advanced Features

### Compare Trailing Stops

```python
from backtest_example import compare_trailing_stops

comparison = compare_trailing_stops(
    model_name='random_forest',
    model=trained_model,
    scaler=scaler,
    df_test=df_test,
    X_columns=feature_columns,
    trailing_stops=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
)
```

### Compare Models

```python
from backtest_example import compare_models

comparison = compare_models(
    models=trained_models,
    scaler=scaler,
    df_test=df_test,
    X_columns=feature_columns,
    trailing_stop_pct=2.0
)
```

## 🤝 Contributing

This is a personal project for ML-based trading strategy development. Feel free to fork and adapt for your own use.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Past performance does not guarantee future results. Always do your own research and never risk more than you can afford to lose.

## 📝 License

This project is for personal use and educational purposes.

---

**Happy Trading! 📈**
