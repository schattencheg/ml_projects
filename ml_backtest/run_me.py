"""
Simple Trading Strategy Example using Backtesting.py
This example demonstrates a Moving Average Crossover strategy
"""

import sys
import os

# Add src directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'Data'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'Background'))

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
from src.Background.enums import DataPeriod, DataResolution
from src.Data.DataProvider import DataProvider
from src.Data.DataProviderLocal import DataProviderLocal


class SmaCrossStrategy(Strategy):
    """
    Simple Moving Average Crossover Strategy
    Buy when fast SMA crosses above slow SMA
    Sell when fast SMA crosses below slow SMA
    """
    # Define strategy parameters
    fast_period = 10  # Fast moving average period
    slow_period = 30  # Slow moving average period
    
    def init(self):
        """Initialize indicators"""
        # Calculate moving averages
        close = self.data.Close
        self.sma_fast = self.I(SMA, close, self.fast_period)
        self.sma_slow = self.I(SMA, close, self.slow_period)
    
    def next(self):
        """Execute trading logic on each bar"""
        # If fast MA crosses above slow MA, buy with 95% of available cash
        if crossover(self.sma_fast, self.sma_slow):
            self.buy(size=0.95)
        
        # If fast MA crosses below slow MA, sell all
        elif crossover(self.sma_slow, self.sma_fast):
            self.position.close()


def main():
    """Main function to run the backtest"""
    
    # Step 1: Load data using DataProvider
    print("Loading data...")
    data_provider = DataProvider(
        tickers=['BTC-USD'],
        resolution=DataResolution.DAY_01,
        period=DataPeriod.YEAR_02  # Last 2 years of data
    )
    
    # Request data for BTC-USD
    df = data_provider.data_request_by_ticker('BTC-USD')
    
    # Prepare data for backtesting (requires specific column names)
    # Backtesting.py expects: Open, High, Low, Close, Volume
    df.columns = [col.capitalize() for col in df.columns]
    df.index = pd.to_datetime(df.index)
    
    print(f"Data loaded: {len(df)} rows from {df.index[0]} to {df.index[-1]}")
    print(f"\nData preview:\n{df.head()}")
    
    # Step 2: Initialize backtest
    print("\nInitializing backtest...")
    bt = Backtest(
        df,
        SmaCrossStrategy,
        cash=100000,  # Starting capital ($100k for BTC trading)
        commission=0.002,  # 0.2% commission per trade
        exclusive_orders=True
    )
    
    # Step 3: Run backtest
    print("\nRunning backtest...")
    stats = bt.run()
    
    # Step 4: Display results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(stats)
    
    # Step 5: Save plot to Output folder
    output_file = os.path.join('Output', 'btc_strategy_results.html')
    print(f"\nSaving plot to: {output_file}")
    bt.plot(filename=output_file, open_browser=True)
    print('\nDone!')


if __name__ == '__main__':
    import pandas as pd
    main()
