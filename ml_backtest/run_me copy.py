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
from backtesting.test import GOOG
from src.Background.enums import DataPeriod, DataResolution
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
            self.buy()
        
        # If fast MA crosses below slow MA, sell all
        elif crossover(self.sma_slow, self.sma_fast):
            self.position.close()


def main():
    """Main function to run the backtest"""
    
    # Step 1: Load data using DataProvider
    print("Loading data...")
    # Local data provider
    data_provider_local = DataProviderLocal()
    # Dataset downloaded from https://storage.googleapis.com/kagglesdsdata/datasets/1346/13409152/btcusd_1-min_data.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20251017%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20251017T090350Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=00f7e0a1e7752ad26cbd57a9c3a9e4f17ae1fc557ed3bca00421ce55ff3fa8798d42c319b488d2ca0e0d9a4ca6c2c76e6793ba493e67846b7bf947cf892d07a3db4be5c75cec15f92381e8f90c2979bd8092f156e7404f4524e54acfdd43fe02e163ab2f91147bf8692be1afe9aff16f4d29a24eb6ebe03fec04811e57fb1b47d3bf2543fd0b6147ee2d0d49f26860d08464500856c79db377e0c5d13b39bb1b887c425a30392ebe4279560736d1b26b2bbae6d2bb74e2cd4bf3bbcdd6b8ddc5891ced3f88d800bfd2540cc781186b6ddf6cb8b5685b28d79733f929fdd654339ba895801ac3389bfd99ef2742ccb4a21dd9ebf68fbf0660a90f1f6c3d4b050a
    df = data_provider_local.get_data_socket('BTC_2024', DataResolution.MINUTE_01, DataPeriod.YEAR_01)
    
    if df is None:
        return

    # Prepare data for backtesting (requires specific column names)
    # Backtesting.py expects: Open, High, Low, Close, Volume
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)
    df.columns = [col.capitalize() for col in df.columns]
    
    print(f"Data loaded: {len(df)} rows from {df.index[0]} to {df.index[-1]}")
    print(f"\nData preview:\n{df.head()}")
    
    df_ = GOOG
    df_ = df
    bt = Backtest(
        df_,
        SmaCrossStrategy,
        cash=10000000,  # Starting capital ($100k for BTC trading)
        commission=0.002,  # 0.2% commission per trade
        exclusive_orders=True
    )
    stats = bt.run()
    print(stats)
    bt.plot()
    return

    # Step 2: Initialize backtest
    print("\nInitializing backtest...")
    bt = Backtest(
        df,
        SmaCrossStrategy,
        cash=10000000,  # Starting capital ($100k for BTC trading)
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
