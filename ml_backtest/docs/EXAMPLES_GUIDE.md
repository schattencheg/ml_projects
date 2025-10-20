# Trading Strategy Examples Guide

This guide explains the three example files included in this project, from simplest to most advanced.

## 📁 Example Files Overview

| File | Difficulty | Purpose | Capital Required |
|------|-----------|---------|------------------|
| `simple_strategy_example.py` | ⭐ Beginner | Basic strategy with SPY | $10,000 |
| `run_me.py` | ⭐⭐ Intermediate | BTC strategy with detailed output | $100,000 |
| `optimize_strategy_example.py` | ⭐⭐⭐ Advanced | Parameter optimization | $10,000 |

---

## 1️⃣ Simple Strategy Example (START HERE!)

**File:** `simple_strategy_example.py`

### What it does:
- Implements a basic 10/20 day Moving Average Crossover strategy
- Uses SPY (S&P 500 ETF) which is affordable
- Provides clean, easy-to-read output
- Perfect for learning the basics

### Key Concepts:
```python
class SimpleSmaStrategy(Strategy):
    def init(self):
        # Create indicators once at the start
        self.ma_fast = self.I(SMA, self.data.Close, 10)
        self.ma_slow = self.I(SMA, self.data.Close, 20)
    
    def next(self):
        # Trading logic executed on each bar
        if crossover(self.ma_fast, self.ma_slow):
            self.buy()  # Fast MA crosses above slow MA
        elif crossover(self.ma_slow, self.ma_fast):
            self.position.close()  # Fast MA crosses below slow MA
```

### Run it:
```bash
python simple_strategy_example.py
```

### Expected Output:
- Data loading confirmation
- Price range information
- Key metrics: Return, Win Rate, Max Drawdown
- Interactive chart in browser

---

## 2️⃣ Bitcoin Strategy Example

**File:** `run_me.py`

### What it does:
- Demonstrates trading with high-value assets (Bitcoin)
- Uses 10/30 period moving averages
- Shows comprehensive backtest statistics
- Handles fractional position sizing

### Key Features:
```python
# Position sizing to use 95% of available cash
def next(self):
    if crossover(self.sma_fast, self.sma_slow):
        self.buy(size=0.95)  # Use 95% of cash
    elif crossover(self.sma_slow, self.sma_fast):
        self.position.close()
```

### Run it:
```bash
python run_me.py
```

### What you'll learn:
- Working with expensive assets
- Position sizing strategies
- Interpreting detailed performance metrics
- Understanding commission impact

---

## 3️⃣ Strategy Optimization Example

**File:** `optimize_strategy_example.py`

### What it does:
- Finds the best parameters for your strategy
- Tests multiple combinations automatically
- Compares default vs optimized performance
- Maximizes Sharpe Ratio (risk-adjusted returns)

### The Optimization Process:
```python
optimized_stats = bt.optimize(
    fast_period=range(5, 30, 5),      # Test 5, 10, 15, 20, 25
    slow_period=range(20, 100, 10),   # Test 20, 30, 40, ..., 90
    maximize='Sharpe Ratio',          # What to optimize for
    constraint=lambda p: p.fast_period < p.slow_period
)
```

### Run it:
```bash
python optimize_strategy_example.py
```

### What you'll learn:
- Parameter optimization techniques
- Avoiding overfitting (use constraints!)
- Comparing strategy variations
- Finding optimal risk-adjusted returns

### ⚠️ Important Notes:
- Optimization takes longer (tests many combinations)
- Don't over-optimize on limited data
- Always test optimized parameters on out-of-sample data
- More parameters = more risk of overfitting

---

## 🎯 Learning Path

### Step 1: Understand the Basics
1. Run `simple_strategy_example.py`
2. Modify the MA periods (try 5/15, 20/50)
3. Change the ticker to 'AAPL' or 'MSFT'
4. Adjust starting capital

### Step 2: Explore Advanced Features
1. Run `run_me.py`
2. Try different cryptocurrencies
3. Experiment with commission rates
4. Modify position sizing

### Step 3: Optimize Your Strategy
1. Run `optimize_strategy_example.py`
2. Add more parameters to optimize
3. Try different optimization metrics
4. Add constraints to prevent overfitting

---

## 🔧 Common Modifications

### Change the Asset:
```python
dp = DataProvider(
    tickers=['AAPL'],  # Apple stock
    # or ['ETH-USD']   # Ethereum
    # or ['GLD']       # Gold ETF
)
```

### Change the Timeframe:
```python
dp = DataProvider(
    resolution=DataResolution.HOUR,     # Hourly data
    period=DataPeriod.MONTH_03          # Last 3 months
)
```

### Add Stop Loss:
```python
def next(self):
    if crossover(self.ma_fast, self.ma_slow):
        self.buy(sl=0.95 * self.data.Close[-1])  # 5% stop loss
```

### Add Take Profit:
```python
def next(self):
    if crossover(self.ma_fast, self.ma_slow):
        self.buy(
            sl=0.95 * self.data.Close[-1],  # Stop loss
            tp=1.10 * self.data.Close[-1]   # Take profit
        )
```

---

## 📊 Understanding the Metrics

| Metric | What it means | Good value |
|--------|---------------|------------|
| **Return [%]** | Total profit/loss | Positive |
| **Sharpe Ratio** | Risk-adjusted return | > 1.0 |
| **Max Drawdown [%]** | Largest peak-to-trough decline | < -20% |
| **Win Rate [%]** | Percentage of profitable trades | > 50% |
| **# Trades** | Total number of trades | Not too few, not too many |
| **Profit Factor** | Gross profit / Gross loss | > 1.5 |

---

## 🚀 Next Steps

1. **Combine Indicators**: Add RSI, MACD, or Bollinger Bands
2. **Multiple Timeframes**: Use different timeframes for signals
3. **Risk Management**: Implement position sizing based on volatility
4. **Walk-Forward Testing**: Test on rolling windows
5. **Machine Learning**: Use ML models for signal generation

---

## 💡 Tips for Success

✅ **DO:**
- Start simple and add complexity gradually
- Test on multiple assets and timeframes
- Use proper position sizing
- Consider transaction costs
- Validate on out-of-sample data

❌ **DON'T:**
- Over-optimize on historical data
- Ignore transaction costs and slippage
- Use all your capital on one trade
- Assume past performance = future results
- Trade without understanding the strategy

---

## 📚 Additional Resources

- [Backtesting.py Documentation](https://kernc.github.io/backtesting.py/)
- [DataProvider Module](src/Data/DataProvider.py)
- [Strategy Examples](https://kernc.github.io/backtesting.py/doc/examples/)

Happy Trading! 🎉
