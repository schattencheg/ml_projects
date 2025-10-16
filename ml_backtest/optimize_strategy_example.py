"""
Strategy Optimization Example
Demonstrates how to find the best parameters for your strategy
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


class OptimizableSmaStrategy(Strategy):
    """
    Moving Average Strategy with optimizable parameters
    """
    # These will be optimized
    fast_period = 10
    slow_period = 30
    
    def init(self):
        close = self.data.Close
        self.ma_fast = self.I(SMA, close, self.fast_period)
        self.ma_slow = self.I(SMA, close, self.slow_period)
    
    def next(self):
        if crossover(self.ma_fast, self.ma_slow):
            self.buy()
        elif crossover(self.ma_slow, self.ma_fast):
            self.position.close()


if __name__ == '__main__':
    # Load data
    print("Loading SPY data...")
    dp = DataProvider(
        tickers=['SPY'],
        resolution=DataResolution.DAY_01,
        period=DataPeriod.YEAR_02
    )
    
    df = dp.data_request_by_ticker('SPY')
    df.columns = [col.capitalize() for col in df.columns]
    df.index = pd.to_datetime(df.index)
    
    print(f"Loaded {len(df)} days of data\n")
    
    # Create backtest
    bt = Backtest(
        df,
        OptimizableSmaStrategy,
        cash=10000,
        commission=0.001
    )
    
    # First, run with default parameters
    print("="*60)
    print("RUNNING WITH DEFAULT PARAMETERS (fast=10, slow=30)")
    print("="*60)
    default_stats = bt.run()
    print(f"Return: {default_stats['Return [%]']:.2f}%")
    print(f"Sharpe Ratio: {default_stats['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown: {default_stats['Max. Drawdown [%]']:.2f}%")
    print(f"Number of Trades: {default_stats['# Trades']}")
    
    # Now optimize to find best parameters
    print("\n" + "="*60)
    print("OPTIMIZING PARAMETERS...")
    print("="*60)
    print("Testing different combinations of fast and slow periods...")
    print("This may take a minute...\n")
    
    optimized_stats = bt.optimize(
        fast_period=range(5, 30, 5),      # Test: 5, 10, 15, 20, 25
        slow_period=range(20, 100, 10),   # Test: 20, 30, 40, 50, 60, 70, 80, 90
        maximize='Sharpe Ratio',          # Optimize for risk-adjusted returns
        constraint=lambda p: p.fast_period < p.slow_period  # Fast must be < slow
    )
    
    print("="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Best Fast Period: {optimized_stats._strategy.fast_period}")
    print(f"Best Slow Period: {optimized_stats._strategy.slow_period}")
    print(f"\nOptimized Return: {optimized_stats['Return [%]']:.2f}%")
    print(f"Optimized Sharpe Ratio: {optimized_stats['Sharpe Ratio']:.2f}")
    print(f"Optimized Max Drawdown: {optimized_stats['Max. Drawdown [%]']:.2f}%")
    print(f"Number of Trades: {optimized_stats['# Trades']}")
    
    # Compare improvement
    print("\n" + "="*60)
    print("IMPROVEMENT")
    print("="*60)
    return_improvement = optimized_stats['Return [%]'] - default_stats['Return [%]']
    print(f"Return Improvement: {return_improvement:+.2f}%")
    print(f"Sharpe Improvement: {optimized_stats['Sharpe Ratio'] - default_stats['Sharpe Ratio']:+.2f}")
    
    # Save plot to Output folder
    output_file = os.path.join('Output', 'optimized_strategy_results.html')
    print(f"\nSaving plot to: {output_file}")
    bt.plot(filename=output_file, open_browser=True)
