# Quick Reference - Backtesting Cheat Sheet

## 🚀 Quick Start

```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA

# 1. Define Strategy
class MyStrategy(Strategy):
    def init(self):
        self.sma = self.I(SMA, self.data.Close, 20)
    
    def next(self):
        if self.data.Close[-1] > self.sma[-1]:
            self.buy()
        elif self.data.Close[-1] < self.sma[-1]:
            self.position.close()

# 2. Load Data (using DataProvider)
from DataProvider import DataProvider
from enums import DataPeriod, DataResolution

dp = DataProvider(tickers=['SPY'], resolution=DataResolution.DAY_01, period=DataPeriod.YEAR_01)
df = dp.data_request_by_ticker('SPY')
df.columns = [col.capitalize() for col in df.columns]

# 3. Run Backtest
bt = Backtest(df, MyStrategy, cash=10000, commission=0.002)
stats = bt.run()
bt.plot()
```

---

## 📊 DataProvider Quick Reference

### Load Different Assets
```python
# Stocks
dp = DataProvider(tickers=['AAPL'])
dp = DataProvider(tickers=['MSFT', 'GOOGL', 'AMZN'])  # Multiple

# Crypto
dp = DataProvider(tickers=['BTC-USD'])
dp = DataProvider(tickers=['ETH-USD'])

# ETFs
dp = DataProvider(tickers=['SPY'])   # S&P 500
dp = DataProvider(tickers=['QQQ'])   # Nasdaq
dp = DataProvider(tickers=['GLD'])   # Gold
```

### Different Timeframes
```python
# Resolution options
DataResolution.MINUTE_01  # 1 minute
DataResolution.MINUTE_05  # 5 minutes
DataResolution.MINUTE_15  # 15 minutes
DataResolution.HOUR       # 1 hour
DataResolution.DAY_01     # 1 day (most common)
DataResolution.WEEK       # 1 week
DataResolution.MONTH_01   # 1 month

# Period options
DataPeriod.DAY_01         # 1 day
DataPeriod.MONTH_01       # 1 month
DataPeriod.MONTH_03       # 3 months
DataPeriod.YEAR_01        # 1 year
DataPeriod.YEAR_02        # 2 years
DataPeriod.YEAR_05        # 5 years
DataPeriod.YEAR_MAX       # Maximum available

# Example
dp = DataProvider(
    tickers=['AAPL'],
    resolution=DataResolution.HOUR,
    period=DataPeriod.MONTH_03
)
```

---

## 🎯 Strategy Components

### Accessing Data
```python
def next(self):
    # Current bar
    self.data.Close[-1]    # Current close
    self.data.Open[-1]     # Current open
    self.data.High[-1]     # Current high
    self.data.Low[-1]      # Current low
    self.data.Volume[-1]   # Current volume
    
    # Previous bars
    self.data.Close[-2]    # Previous close
    self.data.Close[-3]    # 2 bars ago
```

### Creating Indicators
```python
from backtesting.test import SMA, STOCH

def init(self):
    # Simple Moving Average
    self.sma = self.I(SMA, self.data.Close, 20)
    
    # Multiple indicators
    self.sma_fast = self.I(SMA, self.data.Close, 10)
    self.sma_slow = self.I(SMA, self.data.Close, 30)
    
    # Custom indicator
    self.custom = self.I(lambda x: x * 2, self.data.Close)
```

### Trading Actions
```python
def next(self):
    # Buy
    self.buy()                          # Buy with all available cash
    self.buy(size=0.5)                  # Buy with 50% of cash
    self.buy(size=10)                   # Buy 10 units
    
    # Buy with stop-loss and take-profit
    self.buy(
        sl=0.95 * self.data.Close[-1],  # Stop loss at 5% below
        tp=1.10 * self.data.Close[-1]   # Take profit at 10% above
    )
    
    # Sell
    self.sell()                         # Sell (short)
    
    # Close position
    self.position.close()               # Close current position
    
    # Check position
    if self.position:                   # If we have a position
        if self.position.is_long:       # If position is long
            pass
        if self.position.is_short:      # If position is short
            pass
```

---

## 🔧 Common Patterns

### Moving Average Crossover
```python
from backtesting.lib import crossover

class MACross(Strategy):
    fast = 10
    slow = 30
    
    def init(self):
        self.ma_fast = self.I(SMA, self.data.Close, self.fast)
        self.ma_slow = self.I(SMA, self.data.Close, self.slow)
    
    def next(self):
        if crossover(self.ma_fast, self.ma_slow):
            self.buy()
        elif crossover(self.ma_slow, self.ma_fast):
            self.position.close()
```

### RSI Strategy
```python
def init(self):
    self.rsi = self.I(lambda x: talib.RSI(x, 14), self.data.Close)

def next(self):
    if self.rsi[-1] < 30:  # Oversold
        self.buy()
    elif self.rsi[-1] > 70:  # Overbought
        self.position.close()
```

### Bollinger Bands
```python
def init(self):
    close = self.data.Close
    self.sma = self.I(SMA, close, 20)
    self.upper = self.I(lambda x: x + 2 * pd.Series(x).rolling(20).std(), close)
    self.lower = self.I(lambda x: x - 2 * pd.Series(x).rolling(20).std(), close)

def next(self):
    if self.data.Close[-1] < self.lower[-1]:
        self.buy()
    elif self.data.Close[-1] > self.upper[-1]:
        self.position.close()
```

### Multiple Conditions
```python
def next(self):
    # Buy only if ALL conditions are met
    if (self.data.Close[-1] > self.sma[-1] and
        self.data.Volume[-1] > self.avg_volume[-1] and
        not self.position):
        self.buy()
```

---

## 📈 Optimization

### Basic Optimization
```python
stats = bt.optimize(
    fast_period=range(5, 30, 5),
    slow_period=range(20, 100, 10),
    maximize='Sharpe Ratio'
)
```

### With Constraints
```python
stats = bt.optimize(
    fast_period=range(5, 30, 5),
    slow_period=range(20, 100, 10),
    maximize='Return [%]',
    constraint=lambda p: p.fast_period < p.slow_period
)
```

### Optimization Metrics
```python
maximize='Return [%]'           # Total return
maximize='Sharpe Ratio'         # Risk-adjusted return (recommended)
maximize='Sortino Ratio'        # Downside risk-adjusted return
maximize='Calmar Ratio'         # Return / Max Drawdown
maximize='Win Rate [%]'         # Percentage of winning trades
maximize='Profit Factor'        # Gross profit / Gross loss
```

---

## 📊 Backtest Configuration

```python
bt = Backtest(
    data=df,                    # DataFrame with OHLCV data
    strategy=MyStrategy,        # Strategy class
    cash=10000,                 # Starting capital
    commission=0.002,           # 0.2% per trade
    margin=1.0,                 # 1.0 = no leverage, 0.5 = 2x leverage
    trade_on_close=False,       # Execute trades on next open
    hedging=False,              # Allow multiple positions
    exclusive_orders=True       # Cancel pending orders on new signal
)
```

---

## 📉 Analyzing Results

```python
stats = bt.run()

# Key metrics
print(f"Return: {stats['Return [%]']:.2f}%")
print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
print(f"# Trades: {stats['# Trades']}")

# Access all stats
print(stats)

# Get trades DataFrame
trades = stats._trades
print(trades.head())

# Plot
bt.plot()
```

---

## 🎨 Advanced Features

### Position Sizing Based on Volatility
```python
def next(self):
    if crossover(self.ma_fast, self.ma_slow):
        volatility = pd.Series(self.data.Close).pct_change().std()
        size = 0.02 / volatility  # Risk 2% per trade
        self.buy(size=size)
```

### Trailing Stop Loss
```python
def next(self):
    if self.position:
        # Trail stop loss at 5% below highest price since entry
        trailing_stop = 0.95 * max(self.data.Close[-i] for i in range(1, len(self.position) + 1))
        if self.data.Close[-1] < trailing_stop:
            self.position.close()
```

### Time-based Exits
```python
def next(self):
    if self.position:
        # Exit after 10 bars
        if len(self.data) - self.position.entry_bar >= 10:
            self.position.close()
```

---

## 🐛 Common Issues & Solutions

### Issue: "Some prices are larger than initial cash"
**Solution:** Increase cash or use fractional sizing
```python
self.buy(size=0.95)  # Use 95% of available cash
```

### Issue: "No trades executed"
**Solution:** Check your logic, print debug info
```python
def next(self):
    print(f"Close: {self.data.Close[-1]}, SMA: {self.sma[-1]}")
```

### Issue: "Optimization takes too long"
**Solution:** Reduce parameter ranges
```python
# Instead of
fast_period=range(5, 100, 1)  # 95 values!

# Use
fast_period=range(5, 100, 5)  # 19 values
```

---

## 💡 Best Practices

1. **Start Simple**: Begin with basic strategies
2. **Test Thoroughly**: Use multiple assets and timeframes
3. **Avoid Overfitting**: Don't optimize too many parameters
4. **Consider Costs**: Always include realistic commissions
5. **Risk Management**: Use stop losses and position sizing
6. **Out-of-Sample Testing**: Validate on unseen data
7. **Document Everything**: Keep notes on what works

---

## 🔗 Useful Links

- [Backtesting.py Docs](https://kernc.github.io/backtesting.py/)
- [Examples Guide](EXAMPLES_GUIDE.md)
- [Main README](README.md)
