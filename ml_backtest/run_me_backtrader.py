"""
Simple Trading Strategy Example using Backtrader
This example demonstrates a Moving Average Crossover strategy
"""

import sys
import os
import pandas as pd
import backtrader as bt

# Add src directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'Data'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'Background'))

from src.Background.enums import DataPeriod, DataResolution
from src.Data.DataProviderLocal import DataProviderLocal


class SmaCrossStrategy(bt.Strategy):
    """
    Simple Moving Average Crossover Strategy
    Buy when fast SMA crosses above slow SMA
    Sell when fast SMA crosses below slow SMA
    """
    # Define strategy parameters
    params = (
        ('fast_period', 10),  # Fast moving average period
        ('slow_period', 30),  # Slow moving average period
    )
    
    def __init__(self):
        """Initialize indicators"""
        # Calculate moving averages
        self.sma_fast = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.fast_period
        )
        self.sma_slow = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.slow_period
        )
        
        # Create crossover indicator
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)
        
        # Track orders
        self.order = None
    
    def log(self, txt, dt=None):
        """Logging function for this strategy"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')
    
    def notify_order(self, order):
        """Receive notification of order status"""
        if order.status in [order.Submitted, order.Accepted]:
            # Order submitted/accepted - no action required
            return
        
        # Check if an order has been completed
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        
        # Reset order
        self.order = None
    
    def notify_trade(self, trade):
        """Receive notification of trade status"""
        if not trade.isclosed:
            return
        
        self.log(f'TRADE PROFIT, GROSS: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}')
    
    def next(self):
        """Execute trading logic on each bar"""
        # Check if we have an order pending
        if self.order:
            return
        
        # Check if we are in the market
        if not self.position:
            # Not in the market, check if fast MA crosses above slow MA
            if self.crossover > 0:
                # Buy signal - use all available cash
                self.log(f'BUY CREATE, Price: {self.data.close[0]:.2f}')
                self.order = self.buy()
        else:
            # In the market, check if fast MA crosses below slow MA
            if self.crossover < 0:
                # Sell signal - close position
                self.log(f'SELL CREATE, Price: {self.data.close[0]:.2f}')
                self.order = self.sell()


def main():
    """Main function to run the backtest"""
    
    # Step 1: Load data using DataProvider
    print("Loading data...")
    # Local data provider
    data_provider_local = DataProviderLocal()
    df = data_provider_local.get_data_socket('BTC_2024', DataResolution.MINUTE_01, DataPeriod.YEAR_01)
    
    if df is None:
        print("Failed to load data")
        return

    # Prepare data for backtrader
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)
    
    # Ensure column names are lowercase for backtrader
    df.columns = [col.lower() for col in df.columns]
    
    # Backtrader expects: open, high, low, close, volume, openinterest (optional)
    if 'openinterest' not in df.columns:
        df['openinterest'] = 0
    
    print(f"Data loaded: {len(df)} rows from {df.index[0]} to {df.index[-1]}")
    print(f"\nData preview:\n{df.head()}")
    
    # Step 2: Create a Cerebro instance
    print("\nInitializing Cerebro engine...")
    cerebro = bt.Cerebro()
    
    # Step 3: Add data feed to Cerebro
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    
    # Step 4: Add strategy
    cerebro.addstrategy(SmaCrossStrategy)
    
    # Step 5: Set broker parameters
    cerebro.broker.setcash(10000000.0)  # Starting capital ($10M for BTC trading)
    cerebro.broker.setcommission(commission=0.002)  # 0.2% commission per trade
    
    # Step 6: Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # Print starting conditions
    print("\n" + "="*60)
    print("STARTING BACKTEST")
    print("="*60)
    print(f'Starting Portfolio Value: ${cerebro.broker.getvalue():,.2f}')
    
    # Step 7: Run backtest
    print("\nRunning backtest...")
    results = cerebro.run()
    strat = results[0]
    
    # Step 8: Display results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f'Final Portfolio Value: ${cerebro.broker.getvalue():,.2f}')
    print(f'Profit/Loss: ${cerebro.broker.getvalue() - 10000000.0:,.2f}')
    print(f'Return: {((cerebro.broker.getvalue() / 10000000.0) - 1) * 100:.2f}%')
    
    # Print analyzer results
    print("\n" + "="*60)
    print("PERFORMANCE METRICS")
    print("="*60)
    
    # Sharpe Ratio
    sharpe = strat.analyzers.sharpe.get_analysis()
    print(f"Sharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")
    
    # Drawdown
    drawdown = strat.analyzers.drawdown.get_analysis()
    print(f"Max Drawdown: {drawdown.get('max', {}).get('drawdown', 'N/A'):.2f}%")
    
    # Returns
    returns = strat.analyzers.returns.get_analysis()
    print(f"Total Return: {returns.get('rtot', 'N/A') * 100:.2f}%")
    print(f"Average Return: {returns.get('ravg', 'N/A') * 100:.2f}%")
    
    # Trade Analysis
    trades = strat.analyzers.trades.get_analysis()
    print(f"\nTotal Trades: {trades.get('total', {}).get('total', 0)}")
    print(f"Won Trades: {trades.get('won', {}).get('total', 0)}")
    print(f"Lost Trades: {trades.get('lost', {}).get('total', 0)}")
    
    if trades.get('won', {}).get('total', 0) > 0:
        print(f"Win Rate: {(trades.get('won', {}).get('total', 0) / trades.get('total', {}).get('total', 1)) * 100:.2f}%")
        print(f"Average Win: ${trades.get('won', {}).get('pnl', {}).get('average', 0):,.2f}")
    
    if trades.get('lost', {}).get('total', 0) > 0:
        print(f"Average Loss: ${trades.get('lost', {}).get('pnl', {}).get('average', 0):,.2f}")
    
    # Step 9: Plot results
    print("\n" + "="*60)
    print("GENERATING PLOT")
    print("="*60)
    cerebro.plot(style='candlestick', barup='green', bardown='red')
    
    print('\nDone!')


if __name__ == '__main__':
    main()
