"""
Even Simpler Trading Strategy Example
Uses a lower-priced asset (SPY ETF) for easier demonstration
"""

import sys
import os
import pandas as pd

# Add src directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'Data'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'Background'))

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
from DataProvider import DataProvider
from enums import DataPeriod, DataResolution


class SimpleSmaStrategy(Strategy):
    """
    Ultra-simple Moving Average Crossover Strategy
    - Fast MA: 10 days
    - Slow MA: 20 days
    """
    
    def init(self):
        # Create two simple moving averages
        close = self.data.Close
        self.ma_fast = self.I(SMA, close, 10)
        self.ma_slow = self.I(SMA, close, 20)
    
    def next(self):
        # Buy when fast MA crosses above slow MA
        if crossover(self.ma_fast, self.ma_slow):
            self.buy()
        
        # Sell when fast MA crosses below slow MA
        elif crossover(self.ma_slow, self.ma_fast):
            self.position.close()


if __name__ == '__main__':
    # Load SPY (S&P 500 ETF) data - much cheaper than BTC
    print("Loading SPY data...")
    dp = DataProvider(
        tickers=['SPY'],
        resolution=DataResolution.DAY_01,
        period=DataPeriod.YEAR_01
    )
    
    df = dp.data_request_by_ticker('SPY')
    df.columns = [col.capitalize() for col in df.columns]
    df.index = pd.to_datetime(df.index)
    
    print(f"Loaded {len(df)} days of data")
    print(f"Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    
    # Run backtest with modest capital
    bt = Backtest(
        df,
        SimpleSmaStrategy,
        cash=10000,
        commission=0.001
    )
    
    print("\nRunning backtest...")
    results = bt.run()
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Starting Capital: ${10000:,.2f}")
    print(f"Final Equity: ${results['Equity Final [$]']:,.2f}")
    print(f"Return: {results['Return [%]']:.2f}%")
    print(f"Number of Trades: {results['# Trades']}")
    print(f"Win Rate: {results['Win Rate [%]']:.2f}%")
    print(f"Max Drawdown: {results['Max. Drawdown [%]']:.2f}%")
    
    # Save plot to Output folder
    output_file = os.path.join('Output', 'simple_strategy_results.html')
    print(f"\nSaving plot to: {output_file}")
    bt.plot(filename=output_file, open_browser=True)
