# 🚀 Getting Started - 5 Minutes to Your First Backtest

## Step 1: Install Dependencies (1 minute)

Open your terminal in this directory and run:

```bash
pip install -r requirements.txt
```

This installs:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `yfinance` - Market data
- `backtesting` - Backtesting framework
- `bokeh` - Interactive charts

---

## Step 2: Run Your First Strategy (30 seconds)

```bash
python simple_strategy_example.py
```

**What happens:**
1. Downloads 1 year of SPY (S&P 500 ETF) data
2. Runs a simple Moving Average strategy
3. Shows results in terminal
4. Opens interactive chart in browser

**Expected output:**
```
Loading SPY data...
Loaded 250 days of data
Price range: $493.65 - $673.11

Running backtest...

==================================================
RESULTS
==================================================
Starting Capital: $10,000.00
Final Equity: $10,207.60
Return: 2.08%
Number of Trades: 2
Win Rate: 0.00%
Max Drawdown: -16.08%

Saving plot to: Output\simple_strategy_results.html
```

**Results Location:**
All output files are saved to the `Output/` folder:
- Interactive HTML charts
- Can be opened in any web browser
- Automatically opens after generation

---

## Step 3: Understand What You Just Did (2 minutes)

Open `simple_strategy_example.py` and look at the strategy:

```python
class SimpleSmaStrategy(Strategy):
    def init(self):
        # Create two moving averages
        self.ma_fast = self.I(SMA, self.data.Close, 10)  # 10-day MA
        self.ma_slow = self.I(SMA, self.data.Close, 20)  # 20-day MA
    
    def next(self):
        # Buy when fast MA crosses above slow MA
        if crossover(self.ma_fast, self.ma_slow):
            self.buy()
        
        # Sell when fast MA crosses below slow MA
        elif crossover(self.ma_slow, self.ma_fast):
            self.position.close()
```

**What this does:**
- **init()**: Creates indicators once at the start
- **next()**: Runs on each bar of data
- **Buy signal**: Fast MA crosses above slow MA (bullish)
- **Sell signal**: Fast MA crosses below slow MA (bearish)

---

## Step 4: Make Your First Modification (1 minute)

Let's try different moving average periods!

**Edit line 29-30 in `simple_strategy_example.py`:**

```python
# Change from:
self.ma_fast = self.I(SMA, self.data.Close, 10)
self.ma_slow = self.I(SMA, self.data.Close, 20)

# To:
self.ma_fast = self.I(SMA, self.data.Close, 5)   # Faster!
self.ma_slow = self.I(SMA, self.data.Close, 50)  # Slower!
```

**Run again:**
```bash
python simple_strategy_example.py
```

**Compare the results!** Did it improve or get worse?

---

## Step 5: Try a Different Asset (30 seconds)

**Edit line 47 in `simple_strategy_example.py`:**

```python
# Change from:
dp = DataProvider(tickers=['SPY'], ...)

# To one of these:
dp = DataProvider(tickers=['AAPL'], ...)  # Apple stock
dp = DataProvider(tickers=['MSFT'], ...)  # Microsoft
dp = DataProvider(tickers=['QQQ'], ...)   # Nasdaq ETF
dp = DataProvider(tickers=['GLD'], ...)   # Gold ETF
```

**Run again and see how the strategy performs on different assets!**

---

## 🎉 Congratulations!

You've just:
- ✅ Run your first backtest
- ✅ Understood a simple strategy
- ✅ Modified parameters
- ✅ Tested different assets

## 🎯 What's Next?

### Option A: Learn More (Recommended)
Read the comprehensive guides:
1. **[EXAMPLES_GUIDE.md](EXAMPLES_GUIDE.md)** - Detailed walkthrough of all examples
2. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Code snippets and patterns

### Option B: Try More Examples
```bash
# Bitcoin trading (needs $100k capital)
python run_me.py

# Parameter optimization
python optimize_strategy_example.py
```

### Option C: Build Your Own Strategy
Use the template below:

```python
import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'Data'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'Background'))

from backtesting import Backtest, Strategy
from backtesting.test import SMA
from DataProvider import DataProvider
from enums import DataPeriod, DataResolution


class MyCustomStrategy(Strategy):
    """Your strategy description here"""
    
    def init(self):
        # Create your indicators
        pass
    
    def next(self):
        # Your trading logic
        pass


if __name__ == '__main__':
    # Load data
    dp = DataProvider(
        tickers=['YOUR_TICKER'],
        resolution=DataResolution.DAY_01,
        period=DataPeriod.YEAR_01
    )
    df = dp.data_request_by_ticker('YOUR_TICKER')
    df.columns = [col.capitalize() for col in df.columns]
    df.index = pd.to_datetime(df.index)
    
    # Run backtest
    bt = Backtest(df, MyCustomStrategy, cash=10000, commission=0.001)
    stats = bt.run()
    print(stats)
    bt.plot()
```

---

## 💡 Quick Tips

### Tip 1: Start Simple
Don't try to build a complex strategy right away. Master the basics first.

### Tip 2: Test Multiple Assets
A good strategy should work on different assets, not just one.

### Tip 3: Consider Costs
Always include realistic commission rates (0.1% - 0.2% is typical).

### Tip 4: Watch for Overfitting
If your strategy has 10+ parameters, you're probably overfitting.

### Tip 5: Use Stop Losses
Real trading needs risk management. Add stop losses to your strategies.

---

## 🆘 Troubleshooting

### "Module not found"
**Solution:** Make sure you installed requirements:
```bash
pip install -r requirements.txt
```

### "No trades executed"
**Solution:** Your strategy conditions might be too strict. Add debug prints:
```python
def next(self):
    print(f"Close: {self.data.Close[-1]}, MA: {self.ma[-1]}")
```

### "Insufficient margin"
**Solution:** Increase starting cash or use position sizing:
```python
self.buy(size=0.5)  # Use only 50% of cash
```

### Data download is slow
**Solution:** This is normal for the first run. Data is cached for future use.

---

## 📚 Documentation Map

```
Start Here
    ↓
GETTING_STARTED.md (you are here)
    ↓
EXAMPLES_GUIDE.md (detailed explanations)
    ↓
QUICK_REFERENCE.md (code snippets)
    ↓
Build your own strategy!
```

---

## 🎓 Learning Resources

- **Backtesting.py Tutorial**: https://kernc.github.io/backtesting.py/doc/examples/
- **Trading Strategies**: https://www.investopedia.com/trading-strategies-4689645
- **Technical Indicators**: https://www.investopedia.com/technical-analysis-4689657

---

## ✨ Remember

> "The goal of backtesting is not to find the perfect strategy, but to understand how your strategy behaves under different market conditions."

**Good luck with your trading strategies!** 🚀📈

---

**Questions or issues?** Check the other documentation files or review the example code!
