import sys
import os
import pandas as pd
import backtrader as bt
import pickle
import json
from datetime import datetime
from tqdm import tqdm

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
        ('trade_size', 0.001),  # Trade size in BTC
        ('verbose', False),  # Enable/disable verbose logging
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
        
        # Track orders and trades for ML
        self.order = None
        self.trade_data = []  # Store trade data for ML training
        self.entry_price = None
        self.entry_date = None
        self.trade_count = 0  # Track number of trades
        self.bar_count = 0  # Track processed bars
        
        # Get progress bar from __main__ if available
        import __main__
        self.pbar = getattr(__main__, '_backtest_pbar', None)
        
        if self.params.verbose:
            print(f"Strategy initialized with fast_period={self.params.fast_period}, slow_period={self.params.slow_period}, trade_size={self.params.trade_size}")
    
    def log(self, txt, dt=None):
        """Logging function for this strategy"""
        if self.params.verbose:
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
        
        self.trade_count += 1
        profit_status = "PROFIT" if trade.pnl > 0 else "LOSS"
        
        # Update progress bar with trade info
        if self.pbar is not None:
            self.pbar.set_postfix({'Trades': self.trade_count, 'Last': profit_status, 'PnL': f'{trade.pnlcomm:.2f}'})
        
        if self.params.verbose:
            self.log(f'TRADE #{self.trade_count} CLOSED - {profit_status}, GROSS: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}')
        
        # Store trade data for ML training
        # Calculate exit price safely
        if trade.size != 0:
            exit_price = trade.price + (trade.pnl / trade.size)
        else:
            exit_price = trade.price
        
        trade_info = {
            'entry_date': str(trade.dtopen),
            'exit_date': str(trade.dtclose),
            'entry_price': trade.price,
            'exit_price': exit_price,
            'size': abs(trade.size) if trade.size != 0 else self.params.trade_size,
            'pnl': trade.pnl,
            'pnlcomm': trade.pnlcomm,
            'is_profitable': trade.pnl > 0,
            'sma_fast_entry': self.sma_fast[0],
            'sma_slow_entry': self.sma_slow[0],
        }
        self.trade_data.append(trade_info)
    
    def next(self):
        """Execute trading logic on each bar"""
        # Update progress bar
        self.bar_count += 1
        if self.pbar is not None and self.bar_count % 1000 == 0:
            self.pbar.update(1000)
        
        # Check if we have an order pending
        if self.order:
            return
        
        # Check if we are in the market
        if not self.position:
            # Not in the market, check if fast MA crosses above slow MA
            if self.crossover > 0:
                # Buy signal - buy specified size
                self.log(f'BUY CREATE, Price: {self.data.close[0]:.2f}, Size: {self.params.trade_size}')
                self.order = self.buy(size=self.params.trade_size)
        else:
            # In the market, check if fast MA crosses below slow MA
            if self.crossover < 0:
                # Sell signal - sell specified size
                self.log(f'SELL CREATE, Price: {self.data.close[0]:.2f}, Size: {self.params.trade_size}')
                self.order = self.sell(size=self.params.trade_size)


class MyMLBactester:
    def __init__(self):
        # Create strategies dict
        self.strategies = {
            'SmaCrossStrategy': SmaCrossStrategy,
        }
        # Create dataprovider
        self.data_provider_local = DataProviderLocal()
        self.df = None
        # Create cerebro instance
        self.cerebro = bt.Cerebro()

    def initialize(self):
        print("\n" + "="*60)
        print("INITIALIZING ML BACKTESTER")
        print("="*60)
        print("Loading data...")
        self.df = self.load_data('BTC_2024_1', DataResolution.MINUTE_01, DataPeriod.YEAR_01)
        print(f"Data loaded successfully: {len(self.df)} rows")
        print(f"Date range: {self.df.index[0]} to {self.df.index[-1]}")

    def load_data(self, symbol, resolution, period):
        """Load data using DataProvider"""
        df = self.data_provider_local.get_data_socket(symbol, resolution, period)

        # Prepare data for backtrader
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df.set_index('Timestamp', inplace=True)
        
        # Ensure column names are lowercase for backtrader
        df.columns = [col.lower() for col in df.columns]
        
        # Backtrader expects: open, high, low, close, volume, openinterest (optional)
        if 'openinterest' not in df.columns:
            df['openinterest'] = 0

        return df

    def get_results_path(self, strategy_name):
        """Get the results directory path for a strategy"""
        results_dir = os.path.join('Results', strategy_name)
        os.makedirs(results_dir, exist_ok=True)
        return results_dir
    
    def results_exist(self, strategy_name):
        """Check if backtest results exist for a strategy"""
        results_dir = self.get_results_path(strategy_name)
        trades_file = os.path.join(results_dir, 'trades.csv')
        metrics_file = os.path.join(results_dir, 'metrics.json')
        return os.path.exists(trades_file) and os.path.exists(metrics_file)
    
    def save_results(self, strategy_name, trades_df, metrics, strat):
        """Save backtest results to disk"""
        results_dir = self.get_results_path(strategy_name)
        
        # Save trades data
        trades_file = os.path.join(results_dir, 'trades.csv')
        trades_df.to_csv(trades_file, index=False)
        print(f"Saved trades to: {trades_file}")
        
        # Save metrics
        metrics_file = os.path.join(results_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Saved metrics to: {metrics_file}")
        
        # Save plot
        plot_file = os.path.join(results_dir, 'backtest_plot.html')
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            self.cerebro.plot(style='candlestick', barup='green', bardown='red')
            print(f"Plot saved to: {results_dir}")
        except Exception as e:
            print(f"Could not save plot: {e}")
    
    def load_results(self, strategy_name):
        """Load backtest results from disk"""
        results_dir = self.get_results_path(strategy_name)
        
        # Load trades data
        trades_file = os.path.join(results_dir, 'trades.csv')
        trades_df = pd.read_csv(trades_file)
        print(f"Loaded trades from: {trades_file}")
        
        # Load metrics
        metrics_file = os.path.join(results_dir, 'metrics.json')
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        print(f"Loaded metrics from: {metrics_file}")
        
        return trades_df, metrics
    
    def prepare_ml_data(self, trades_df, strategy_name):
        """Prepare ML training data from trades"""
        results_dir = self.get_results_path(strategy_name)
        
        if len(trades_df) == 0:
            print("No trades found. Cannot prepare ML data.")
            return
        
        # Separate profitable and unprofitable trades
        profitable_trades = trades_df[trades_df['is_profitable'] == True]
        unprofitable_trades = trades_df[trades_df['is_profitable'] == False]
        
        print(f"Total trades: {len(trades_df)}")
        print(f"Profitable trades: {len(profitable_trades)} ({len(profitable_trades)/len(trades_df)*100:.2f}%)")
        print(f"Unprofitable trades: {len(unprofitable_trades)} ({len(unprofitable_trades)/len(trades_df)*100:.2f}%)")
        
        # Save profitable and unprofitable trades separately
        profitable_file = os.path.join(results_dir, 'trades_profitable.csv')
        unprofitable_file = os.path.join(results_dir, 'trades_unprofitable.csv')
        
        profitable_trades.to_csv(profitable_file, index=False)
        unprofitable_trades.to_csv(unprofitable_file, index=False)
        
        print(f"\nSaved profitable trades to: {profitable_file}")
        print(f"Saved unprofitable trades to: {unprofitable_file}")
        
        # Prepare ML training dataset
        # Add label column: 1 for profitable, 0 for unprofitable
        ml_data = trades_df.copy()
        ml_data['label'] = ml_data['is_profitable'].astype(int)
        
        # Select features for ML training
        feature_columns = ['entry_price', 'size', 'sma_fast_entry', 'sma_slow_entry']
        ml_features = ml_data[feature_columns + ['label']]
        
        # Save ML training data
        ml_file = os.path.join(results_dir, 'ml_training_data.csv')
        ml_features.to_csv(ml_file, index=False)
        print(f"Saved ML training data to: {ml_file}")
        
        # Print feature statistics
        print("\n" + "="*60)
        print("ML TRAINING DATA STATISTICS")
        print("="*60)
        print(ml_features.describe())
        
        return ml_features

    def run(self, strategy_name):
        """Run backtest or load existing results, then prepare ML training data"""
        
        # Check if results already exist
        if self.results_exist(strategy_name):
            print(f"\n{'='*60}")
            print(f"LOADING EXISTING RESULTS FOR: {strategy_name}")
            print(f"{'='*60}")
            trades_df, metrics = self.load_results(strategy_name)
            
            # Display loaded metrics
            print("\n" + "="*60)
            print("LOADED BACKTEST RESULTS")
            print("="*60)
            for key, value in metrics.items():
                print(f"{key}: {value}")
        else:
            print(f"\n{'='*60}")
            print(f"NO EXISTING RESULTS FOUND - RUNNING BACKTEST: {strategy_name}")
            print(f"{'='*60}")
            
            # Add data feed to Cerebro
            data = bt.feeds.PandasData(dataname=self.df)
            self.cerebro.adddata(data)
        
            # Add strategy
            self.cerebro.addstrategy(self.strategies[strategy_name])
        
            # Set broker parameters
            self.cerebro.broker.setcash(10000000.0)  # Starting capital ($10M)
            self.cerebro.broker.setcommission(commission=0.002)  # 0.2% commission
        
            # Add analyzers
            self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
            # Print starting conditions
            print("\n" + "="*60)
            print("STARTING BACKTEST")
            print("="*60)
            print(f'Starting Portfolio Value: ${self.cerebro.broker.getvalue():,.2f}')
        
            # Run backtest with progress bar
            print("\nRunning backtest...")
            total_bars = len(self.df)
            
            # Create progress bar
            pbar = tqdm(total=total_bars, desc="Processing bars", unit="bars")
            
            # Store progress bar globally so strategy can access it
            # This is a workaround since we can't pass it directly to the strategy
            import __main__
            __main__._backtest_pbar = pbar
            
            try:
                results = self.cerebro.run()
                strat = results[0]
                
                # Update to completion
                remaining = total_bars - strat.bar_count
                if remaining > 0:
                    pbar.update(remaining)
            finally:
                pbar.close()
                if hasattr(__main__, '_backtest_pbar'):
                    delattr(__main__, '_backtest_pbar')
        
            # Display results
            print("\n" + "="*60)
            print("BACKTEST RESULTS")
            print("="*60)
            final_value = self.cerebro.broker.getvalue()
            print(f'Final Portfolio Value: ${final_value:,.2f}')
            print(f'Profit/Loss: ${final_value - 10000000.0:,.2f}')
            print(f'Return: {((final_value / 10000000.0) - 1) * 100:.2f}%')
        
            # Get analyzer results
            print("\n" + "="*60)
            print("PERFORMANCE METRICS")
            print("="*60)
        
            sharpe = strat.analyzers.sharpe.get_analysis()
            drawdown = strat.analyzers.drawdown.get_analysis()
            returns = strat.analyzers.returns.get_analysis()
            trades = strat.analyzers.trades.get_analysis()
            
            print(f"Sharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")
            print(f"Max Drawdown: {drawdown.get('max', {}).get('drawdown', 'N/A'):.2f}%")
            print(f"Total Return: {returns.get('rtot', 'N/A') * 100:.2f}%")
            print(f"Average Return: {returns.get('ravg', 'N/A') * 100:.2f}%")
            print(f"\nTotal Trades: {trades.get('total', {}).get('total', 0)}")
            print(f"Won Trades: {trades.get('won', {}).get('total', 0)}")
            print(f"Lost Trades: {trades.get('lost', {}).get('total', 0)}")
        
            if trades.get('won', {}).get('total', 0) > 0:
                print(f"Win Rate: {(trades.get('won', {}).get('total', 0) / trades.get('total', {}).get('total', 1)) * 100:.2f}%")
                print(f"Average Win: ${trades.get('won', {}).get('pnl', {}).get('average', 0):,.2f}")
        
            if trades.get('lost', {}).get('total', 0) > 0:
                print(f"Average Loss: ${trades.get('lost', {}).get('pnl', {}).get('average', 0):,.2f}")
            
            # Convert trade_data to DataFrame
            trades_df = pd.DataFrame(strat.trade_data)
            
            # Prepare metrics dictionary
            metrics = {
                'final_value': final_value,
                'profit_loss': final_value - 10000000.0,
                'return_pct': ((final_value / 10000000.0) - 1) * 100,
                'sharpe_ratio': sharpe.get('sharperatio', None),
                'max_drawdown': drawdown.get('max', {}).get('drawdown', None),
                'total_return': returns.get('rtot', None),
                'avg_return': returns.get('ravg', None),
                'total_trades': trades.get('total', {}).get('total', 0),
                'won_trades': trades.get('won', {}).get('total', 0),
                'lost_trades': trades.get('lost', {}).get('total', 0),
                'avg_win': trades.get('won', {}).get('pnl', {}).get('average', 0),
                'avg_loss': trades.get('lost', {}).get('pnl', {}).get('average', 0),
            }
            
            # Save results
            print("\n" + "="*60)
            print("SAVING RESULTS")
            print("="*60)
            self.save_results(strategy_name, trades_df, metrics, strat)
        
        # Prepare ML training data
        print("\n" + "="*60)
        print("PREPARING ML TRAINING DATA")
        print("="*60)
        self.prepare_ml_data(trades_df, strategy_name)
        
        print('\nDone!')


if __name__ == '__main__':
    ml_backtester = MyMLBactester()
    ml_backtester.initialize()
    ml_backtester.run('SmaCrossStrategy')
