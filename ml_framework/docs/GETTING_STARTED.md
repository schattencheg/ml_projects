# Getting Started with ML Framework

## 🎯 What You Can Do

This framework provides:
1. ✅ **Smart Data Loading** - Automatic caching, 20x faster
2. ✅ **Feature Generation** - 45+ technical indicators
3. ✅ **ML Models** - LogisticRegression, RandomForest, XGBoost, etc.
4. ✅ **Backtesting** - 3 different backends (NoLib, Backtrader, backtesting.py)
5. ✅ **Model Persistence** - Timestamped saves

## 🚀 Quick Start (5 Minutes)

### Step 1: Install Dependencies

```bash
pip install pandas numpy scikit-learn yfinance joblib
```

### Step 2: Run Simple Example

```bash
python btcusdt_simple_example.py
```

This will:
- Download BTC-USD data
- Generate features
- Train LogisticRegression and RandomForest
- Show results

### Step 3: Run Backtest Comparison

```bash
python btcusdt_backtest_comparison.py
```

This will:
- Load data with caching
- Train Random Forest model
- Run 3 backtesting backends
- Compare results

## 📚 Examples Overview

### 1. Simple Example (Recommended First)
**File:** `btcusdt_simple_example.py`

Standalone example with minimal dependencies:
- Direct yfinance download
- Feature generation
- Model training
- No complex imports

**Run it:**
```bash
python btcusdt_simple_example.py
```

### 2. Framework Example
**File:** `btcusdt_framework_example.py`

Uses full framework features:
- Smart data caching
- DataProvider class
- Model persistence
- Timestamped saves

**Run it:**
```bash
python btcusdt_framework_example.py
```

### 3. Backtest Comparison (Most Comprehensive)
**File:** `btcusdt_backtest_comparison.py`

Complete workflow with backtesting:
- All framework features
- 3 backtesting backends
- Side-by-side comparison
- Performance metrics

**Run it:**
```bash
python btcusdt_backtest_comparison.py
```

## 🎓 Learning Path

### Beginner
1. ✅ Run `btcusdt_simple_example.py`
2. ✅ Understand data loading and feature generation
3. ✅ See model training and evaluation

### Intermediate
1. ✅ Run `btcusdt_framework_example.py`
2. ✅ Learn about smart caching
3. ✅ Explore model persistence

### Advanced
1. ✅ Run `btcusdt_backtest_comparison.py`
2. ✅ Compare backtesting backends
3. ✅ Customize strategies

## 📖 Key Concepts

### 1. Smart Data Caching

The framework automatically caches downloaded data:

```python
from src.data_provider import DataProvider

data_provider = DataProvider(data_dir='data')

# First run: Downloads and caches
df = data_provider.load_yahoo('BTC-USD', '2020-01-01', '2024-11-24')

# Second run: Loads from cache (20x faster!)
df = data_provider.load_yahoo('BTC-USD', '2020-01-01', '2024-11-24')
```

**Cache location:** `data/BTC_USD_1d.csv`

### 2. Feature Generation

Generate 45+ technical indicators automatically:

```python
from src.features_generator import FeaturesGenerator

features_gen = FeaturesGenerator()

# Generate features
df_features = features_gen.generate_features(df, feature_set='advanced')

# Create target variable
df_features = features_gen.create_target(
    df_features,
    target_type='classification',
    future_bars=5,
    threshold=0.02
)
```

**Features include:**
- Moving averages (SMA, EMA)
- Momentum (RSI, MACD)
- Volatility (Bollinger Bands, ATR)
- Volume indicators

### 3. Model Training

Train multiple models with consistent interface:

```python
from src.models_lib import LogisticRegressionModel, RandomForestModel
from sklearn.preprocessing import StandardScaler

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train models
lr_model = LogisticRegressionModel(n_jobs=-1)
lr_model.fit(X_train_scaled, y_train)

rf_model = RandomForestModel(n_estimators=100, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
```

### 4. Backtesting

Test your strategy with 3 different backends:

```python
from src.backtesting import BacktestNoLib, BacktestBacktrader, BacktestBacktestingPy

# Choose your backend
backtest = BacktestNoLib(
    initial_capital=10000,
    commission=0.001,
    position_size=1.0,
    stop_loss=0.05,
    take_profit=0.10
)

# Run backtest
results = backtest.run(df, model, scaler, feature_cols)
backtest.print_results()
```

## 🔧 Configuration

### Data Configuration

```python
TICKER = 'BTC-USD'
START_DATE = '2020-01-01'
END_DATE = '2024-11-24'
INTERVAL = '1d'  # Options: 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo
```

### Target Configuration

```python
FUTURE_BARS = 5      # Predict 5 bars ahead
THRESHOLD = 0.02     # 2% price change threshold
```

### Backtest Configuration

```python
INITIAL_CAPITAL = 10000.0
COMMISSION = 0.001      # 0.1%
POSITION_SIZE = 1.0     # 100% of capital
STOP_LOSS = 0.05        # 5% stop loss
TAKE_PROFIT = 0.10      # 10% take profit
```

## 📊 Understanding Results

### Model Metrics

```
✓ Model trained
  Accuracy: 0.6832
  F1 Score: 0.6234
```

- **Accuracy:** Overall correctness (68.32%)
- **F1 Score:** Balance between precision and recall (62.34%)

### Backtest Metrics

```
Capital:
  Initial Capital:  $   10,000.00
  Final Capital:    $   11,031.00
  Total Return:            10.31%

Risk Metrics:
  Sharpe Ratio:             1.45
  Max Drawdown:            -8.23%

Trading:
  Total Trades:               28
  Win Rate:               53.57%
```

- **Total Return:** Profit/loss percentage
- **Sharpe Ratio:** Risk-adjusted return (>1.0 is good)
- **Max Drawdown:** Largest peak-to-trough decline
- **Win Rate:** Percentage of profitable trades

## 🎯 Common Tasks

### Task 1: Change Ticker

Edit the example file:
```python
TICKER = 'ETH-USD'  # Change from BTC-USD
```

### Task 2: Change Date Range

```python
START_DATE = '2021-01-01'  # More recent data
END_DATE = '2024-11-24'
```

### Task 3: Change Prediction Target

```python
FUTURE_BARS = 10     # Predict 10 days ahead
THRESHOLD = 0.03     # 3% threshold
```

### Task 4: Use Different Model

```python
from src.models_lib import XGBoostModel

model = XGBoostModel(n_estimators=100, n_jobs=-1)
model.fit(X_train_scaled, y_train)
```

### Task 5: Adjust Backtest Parameters

```python
backtest = BacktestNoLib(
    initial_capital=50000,   # More capital
    commission=0.0005,       # Lower commission
    position_size=0.5,       # Use 50% per trade
    stop_loss=0.03,          # Tighter stop loss
    take_profit=0.15         # Higher take profit
)
```

## 📁 Project Structure

```
ml_framework/
├── data/                          # Cached data files
│   └── BTC_USD_1d.csv
├── models/                        # Saved models (timestamped)
│   └── 2024-11-24_15-30-00/
│       ├── randomforest.joblib
│       └── metadata.joblib
├── src/
│   ├── backtesting/              # Backtesting backends
│   │   ├── base_backtest.py
│   │   ├── backtest_nolib.py
│   │   ├── backtest_backtrader.py
│   │   └── backtest_backtesting_py.py
│   ├── models_lib/               # Model classes
│   │   ├── linear_model.py
│   │   └── ...
│   ├── data_provider.py          # Data loading with caching
│   └── features_generator.py     # Feature generation
├── btcusdt_simple_example.py     # Simple example
├── btcusdt_framework_example.py  # Framework example
└── btcusdt_backtest_comparison.py # Backtest comparison
```

## 🐛 Troubleshooting

### Issue: "No module named 'yfinance'"
```bash
pip install yfinance
```

### Issue: "No module named 'sklearn'"
```bash
pip install scikit-learn
```

### Issue: "No module named 'backtrader'"
```bash
pip install backtrader
# Or use BacktestNoLib which has no dependencies
```

### Issue: Cache file corrupted
```bash
# Delete cache and re-download
rm data/BTC_USD_1d.csv
python btcusdt_framework_example.py
```

### Issue: Low model performance
- Try different models (RandomForest, XGBoost)
- Adjust FUTURE_BARS and THRESHOLD
- Use more training data (longer date range)
- Add more features

### Issue: No trades in backtest
- Check model predictions (should be 0 or 1)
- Verify features are scaled
- Ensure enough test data
- Check position_size > 0

## 💡 Tips & Best Practices

### Tip 1: Start Simple
Begin with `btcusdt_simple_example.py` to understand the basics.

### Tip 2: Use Caching
Always use `use_cache=True` to speed up data loading.

### Tip 3: Scale Features
Always scale features before training:
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Tip 4: Compare Backends
Run multiple backtesting backends to validate results.

### Tip 5: Monitor Drawdown
Keep max drawdown < 20% for safer strategies.

### Tip 6: Use Stop Loss
Always use stop loss to protect capital:
```python
backtest = BacktestNoLib(stop_loss=0.05)  # 5% stop loss
```

## 🎯 Next Steps

1. ✅ Run all three examples
2. ✅ Modify configuration parameters
3. ✅ Try different tickers (ETH-USD, SPY, etc.)
4. ✅ Experiment with different models
5. ✅ Compare backtesting backends
6. ✅ Read detailed documentation in `src/backtesting/README.md`

## 📚 Additional Resources

- **[src/backtesting/README.md](src/backtesting/README.md)** - Complete backtesting API
- **[BACKTESTING_IMPLEMENTATION.md](BACKTESTING_IMPLEMENTATION.md)** - Implementation details
- **[test_backtesting.py](test_backtesting.py)** - Test script

## ✅ Checklist

Before you start:
- [ ] Python 3.7+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Run `test_backtesting.py` to verify setup

Your first workflow:
- [ ] Run `btcusdt_simple_example.py`
- [ ] Check `data/` folder for cached data
- [ ] Run again to see caching benefit
- [ ] Run `btcusdt_backtest_comparison.py`
- [ ] Compare results from different backends

## 🎉 You're Ready!

You now have everything you need to:
- Load and cache financial data
- Generate technical features
- Train ML models
- Backtest strategies
- Compare results

Start with the simple example and work your way up!

```bash
python btcusdt_simple_example.py
```

---

**Last Updated:** 2024-11-24  
**Version:** 1.0.0  
**Status:** ✅ Ready to Use
