"""Debug backtesting.py issue."""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Create sample data similar to BTC
np.random.seed(42)
n = 200
dates = pd.date_range('2024-01-01', periods=n, freq='D')
close = 90000 + np.cumsum(np.random.randn(n) * 1000)  # BTC-like prices ~$90k
df = pd.DataFrame({
    'open': close + np.random.randn(n) * 100,
    'high': close + np.abs(np.random.randn(n) * 200),
    'low': close - np.abs(np.random.randn(n) * 200),
    'close': close,
    'volume': np.random.randint(1000, 10000, n),
    'feature1': np.random.randn(n),
    'feature2': np.random.randn(n),
}, index=dates)

print(f"Price range: ${df['close'].min():.0f} - ${df['close'].max():.0f}")
print(f"Initial capital: $10,000")
print(f"Position size: 2% = ${10000 * 0.02:.0f}")

class MockModel:
    def predict(self, X):
        # Generate buy signals every 20 bars
        return np.array([1 if i % 20 == 0 else 0 for i in range(len(X))])

model = MockModel()
scaler = StandardScaler()
feature_cols = ['feature1', 'feature2']
scaler.fit(df[feature_cols])

print("\n" + "="*60)
print("Testing backtesting.py directly")
print("="*60)

from backtesting import Backtest, Strategy
from backtesting.lib import FractionalBacktest

# Prepare data
X = df[feature_cols].values
X_scaled = scaler.transform(X)
predictions = model.predict(X_scaled)

df_bt = df.copy()
df_bt['Prediction'] = predictions

# Capitalize columns
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    if col.lower() in df_bt.columns:
        df_bt[col] = df_bt[col.lower()]

position_size_pct = 0.02
bars_to_hold = 15

class MLStrategy(Strategy):
    entry_bar = None
    last_exit_bar = None
    
    def init(self):
        pass
    
    def next(self):
        current_bar = len(self.data) - 1
        prediction = self.data.Prediction[-1]
        
        # Check exit condition
        if self.position and MLStrategy.entry_bar is not None:
            bars_held = current_bar - MLStrategy.entry_bar
            if bars_held >= bars_to_hold:
                print(f"  EXIT at bar {current_bar}, held {bars_held} bars")
                MLStrategy.last_exit_bar = current_bar
                MLStrategy.entry_bar = None
                self.position.close()
                return
        
        # Check entry condition
        if prediction == 1 and not self.position:
            can_open = (MLStrategy.last_exit_bar is None or 
                       current_bar > MLStrategy.last_exit_bar)
            
            if can_open:
                print(f"  ENTRY at bar {current_bar}, price=${self.data.Close[-1]:.0f}, equity=${self.equity:.0f}")
                MLStrategy.entry_bar = current_bar
                self.buy(size=position_size_pct)

print("\nRunning FractionalBacktest...")
bt = FractionalBacktest(
    df_bt,
    MLStrategy,
    cash=10000,
    commission=0.001,
    exclusive_orders=True,
    trade_on_close=True,
    hedging=False
)

stats = bt.run()

print(f"\nResults:")
print(f"  Final Equity: ${stats['Equity Final [$]']:.2f}")
print(f"  Total Return: {stats['Return [%]']:.2f}%")
print(f"  # Trades: {stats['# Trades']}")

if hasattr(stats, '_trades') and len(stats._trades) > 0:
    print(f"\nTrades:")
    print(stats._trades)
