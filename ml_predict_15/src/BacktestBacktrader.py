"""
ML-Based Backtesting using Backtrader Library

This module provides a backtesting strategy class that integrates with the backtrader
library to use trained ML models for generating trading signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import backtrader as bt
from datetime import datetime
from src.BacktestBase import BacktestBase


class BacktestBacktrader(bt.Strategy):
    """
    Backtrader strategy that uses ML model predictions as signals.
    
    This strategy:
    1. Uses ML model predictions to generate buy/sell signals
    2. Implements trailing stop loss
    3. Supports probability thresholds for entry
    4. Allows position sizing based on signal strength
    """
    
    params = (
        ('probability_threshold', 0.6),
        ('trailing_stop_pct', 2.0),
        ('take_profit_pct', None),
        ('position_size_pct', 1.0),
        ('printlog', False),
    )
    
    def __init__(self):
        """
        Initialize the strategy.
        Called once at the start of backtesting.
        """
        # Keep references to data lines
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open
        
        # ML predictions (should be added to data feed)
        self.ml_signal = self.datas[0].ml_signal
        self.ml_probability = self.datas[0].ml_probability
        
        # Track orders and positions
        self.order = None
        self.entry_price = None
        self.highest_price = None
        self.entry_bar = None
        
        # Track trades
        self.trades_list = []
        
        # Debug counters
        self.signal_count = 0
        self.entry_attempt_count = 0
        self.bar_count = 0
        
    def log(self, txt, dt=None):
        """Logging function for strategy."""
        if self.params.printlog:
            dt = dt or bt.num2date(self.datas[0].datetime[0]).date()
            print(f'{dt.isoformat()} {txt}')
    
    def notify_order(self, order):
        """
        Called when an order is completed.
        """
        # Debug: Log ALL order status changes
        status_names = {
            order.Created: 'Created',
            order.Submitted: 'Submitted',
            order.Accepted: 'Accepted',
            order.Partial: 'Partial',
            order.Completed: 'Completed',
            order.Canceled: 'Canceled',
            order.Expired: 'Expired',
            order.Margin: 'Margin',
            order.Rejected: 'Rejected'
        }
        self.log(f'ORDER STATUS: {status_names.get(order.status, "Unknown")} - '
                f'Type: {"BUY" if order.isbuy() else "SELL"}')
        
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, '
                        f'Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.entry_price = order.executed.price
                self.highest_price = order.executed.price
                self.entry_bar = len(self)
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, '
                        f'Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected - Status: {status_names.get(order.status, "Unknown")}')
            if order.status == order.Margin:
                self.log(f'  MARGIN ISSUE: Not enough cash! Cash: ${self.broker.getcash():.2f}')
            elif order.status == order.Rejected:
                self.log(f'  REJECTED: Order was rejected by broker')
        
        self.order = None
    
    def notify_trade(self, trade):
        """
        Called when a trade is closed.
        """
        if not trade.isclosed:
            return
        
        self.log(f'TRADE PROFIT, GROSS: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}')
        
        # Store trade information
        self.trades_list.append({
            'entry_bar': self.entry_bar,
            'exit_bar': len(self),
            'entry_price': self.entry_price,
            'exit_price': self.dataclose[0],
            'pnl': trade.pnl,
            'pnl_net': trade.pnlcomm,
            'pnl_pct': (trade.pnl / self.entry_price) * 100 if self.entry_price else 0
        })
        
        # Reset tracking
        self.entry_price = None
        self.highest_price = None
        self.entry_bar = None
    
    def next(self):
        """
        Called for each bar in the backtest.
        Implements the trading logic.
        """
        # Debug: Count bars
        self.bar_count += 1
        if self.bar_count <= 5 or self.bar_count % 100 == 0:
            self.log(f'Processing bar {self.bar_count}')
        
        # Check if an order is pending
        if self.order:
            self.log(f'Order pending, waiting...')
            return
        
        # Get current values
        current_price = self.dataclose[0]
        signal = self.ml_signal[0]
        probability = self.ml_probability[0]
        
        # If we're in a position, check exit conditions
        if self.position:
            self._check_exit_conditions(current_price, signal)
        
        # If not in position, check entry conditions
        elif signal == 1 and probability >= self.params.probability_threshold:
            # Calculate position size
            size = self._calculate_position_size(current_price)
            
            # Enter long position
            self.entry_attempt_count += 1
            self.log(f'BUY CREATE, Price: {current_price:.2f}, Size: {size:.2f}, Prob: {probability:.2f}')
            self.order = self.buy(size=size)
        
        # Debug: Count signals
        if signal == 1:
            self.signal_count += 1
    
    def _check_exit_conditions(self, current_price: float, signal: float):
        """
        Check if any exit conditions are met.
        
        Parameters:
        -----------
        current_price : float
            Current market price
        signal : float
            Current ML signal
        """
        if not self.position or self.entry_price is None:
            return
        
        # Update highest price for trailing stop
        if current_price > self.highest_price:
            self.highest_price = current_price
        
        # Calculate trailing stop price
        trailing_stop_price = self.highest_price * (1 - self.params.trailing_stop_pct / 100)
        
        # Check trailing stop
        if current_price <= trailing_stop_price:
            self.log(f'TRAILING STOP HIT, Price: {current_price:.2f}, Stop: {trailing_stop_price:.2f}')
            self.order = self.close()
            return
        
        # Check take profit if enabled
        if self.params.take_profit_pct is not None:
            take_profit_price = self.entry_price * (1 + self.params.take_profit_pct / 100)
            if current_price >= take_profit_price:
                self.log(f'TAKE PROFIT HIT, Price: {current_price:.2f}, Target: {take_profit_price:.2f}')
                self.order = self.close()
                return
        
        # Check for exit signal from ML model
        if signal == 0:
            self.log(f'EXIT SIGNAL, Price: {current_price:.2f}')
            self.order = self.close()
    
    def _calculate_position_size(self, price: float) -> float:
        """
        Calculate position size based on available cash.
        
        Parameters:
        -----------
        price : float
            Current price
            
        Returns:
        --------
        size : float
            Number of units to buy
        """
        available_cash = self.broker.getcash()
        cash_to_use = available_cash * self.params.position_size_pct
        size = cash_to_use / price
        
        # Debug logging
        self.log(f'Position Sizing: Cash=${available_cash:.2f}, Use={self.params.position_size_pct*100:.1f}%, '
                f'Amount=${cash_to_use:.2f}, Price=${price:.2f}, Size={size:.6f}')
        
        if size < 0.001:
            self.log(f'WARNING: Position size {size:.6f} is very small! May fail to execute.')
        
        return size
    
    def stop(self):
        """Called at the end of the backtest."""
        self.log(f'Final Portfolio Value: {self.broker.getvalue():.2f}', dt=self.datas[0].datetime.date(0))
        print(f"\nStrategy Debug Info:")
        print(f"  Total bars processed: {self.bar_count}")
        print(f"  Total buy signals (1): {self.signal_count}")
        print(f"  Entry attempts (signal + prob >= threshold): {self.entry_attempt_count}")
        print(f"  Actual trades executed: {len(self.trades_list)}")
        print(f"  Probability threshold: {self.params.probability_threshold}")


class MLPandasData(bt.feeds.PandasData):
    """
    Extended PandasData feed that includes ML predictions.
    """
    
    lines = ('ml_signal', 'ml_probability',)
    
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),  # Changed from 'Volume' to 'volume'
        ('openinterest', None),
        ('ml_signal', 'ML_Signal'),
        ('ml_probability', 'ML_Probability'),
    )


class BacktestBacktraderML(BacktestBase):
    """
    Wrapper class for backtrader that handles ML model integration.
    
    This class:
    1. Prepares data with ML predictions
    2. Runs backtests using the backtrader library
    3. Provides easy-to-use interface for ML-based backtesting
    """
    
    def __init__(
        self,
        initial_cash: float = 10000.0,
        commission: float = 0.001,
        slippage_perc: float = 0.0,
        slippage_fixed: float = 0.0,
    ):
        """
        Initialize the ML Backtester for backtrader.
        
        Parameters:
        -----------
        initial_cash : float
            Starting capital
        commission : float
            Commission per trade as a fraction (e.g., 0.001 for 0.1%)
        slippage_perc : float
            Percentage slippage
        slippage_fixed : float
            Fixed slippage amount
        """
        # Initialize base class
        super().__init__(initial_capital=initial_cash, commission=commission, slippage=slippage_perc)
        
        self.initial_cash = initial_cash
        self.slippage_perc = slippage_perc
        self.slippage_fixed = slippage_fixed
        
    def prepare_data(
        self,
        df: pd.DataFrame,
        model,
        scaler,
        X_columns: List[str],
        close_column: str = 'close',
        timestamp_column: str = 'Timestamp'
    ) -> pd.DataFrame:
        """
        Prepare data with ML predictions for backtesting.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data and features
        model : sklearn model
            Trained ML model
        scaler : sklearn scaler
            Fitted scaler
        X_columns : List[str]
            List of feature column names
        close_column : str
            Name of close price column
        timestamp_column : str
            Name of timestamp column
            
        Returns:
        --------
        df_prepared : pd.DataFrame
            DataFrame with ML predictions added
        """
        # Make a copy
        df_prepared = df.copy()
        
        # Ensure we have the required columns (case-insensitive check)
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # Normalize column names to lowercase for checking
        df_cols_lower = {col.lower(): col for col in df_prepared.columns}
        
        # Check and rename if needed
        for req_col in required_cols:
            if req_col not in df_prepared.columns:
                # Try to find case-insensitive match
                if req_col.lower() in df_cols_lower:
                    actual_col = df_cols_lower[req_col.lower()]
                    if actual_col != req_col:
                        print(f"Renaming column '{actual_col}' to '{req_col}'")
                        df_prepared.rename(columns={actual_col: req_col}, inplace=True)
                else:
                    raise ValueError(f"Missing required column: {req_col} (case-insensitive)")
        
        # Get features (keep as DataFrame to preserve column names for scaler)
        X = df_prepared[X_columns]
        
        # Scale features (pass DataFrame to avoid feature name warning)
        X_scaled = scaler.transform(X)
        
        # Get predictions
        predictions = model.predict(X_scaled)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_scaled)
            # Get probability of positive class
            ml_probability = probabilities[:, 1]
        else:
            # Use decision function or just the prediction
            if hasattr(model, 'decision_function'):
                decision = model.decision_function(X_scaled)
                # Normalize to 0-1 range
                ml_probability = (decision - decision.min()) / (decision.max() - decision.min())
            else:
                ml_probability = predictions.astype(float)
        
        # Add predictions to dataframe
        df_prepared['ML_Signal'] = predictions
        df_prepared['ML_Probability'] = ml_probability
        
        # Set timestamp as index if it exists
        if timestamp_column in df_prepared.columns:
            df_prepared.set_index(timestamp_column, inplace=True)
        
        # Ensure index is datetime and remove nanoseconds (Backtrader doesn't support them)
        if not isinstance(df_prepared.index, pd.DatetimeIndex):
            df_prepared.index = pd.to_datetime(df_prepared.index)
        
        # Remove nanoseconds to avoid Backtrader warning
        df_prepared.index = df_prepared.index.floor('s')
        
        # Validate data - check for NaN/inf values in OHLCV columns
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_cols:
            if df_prepared[col].isna().any():
                print(f"Warning: Found NaN values in {col}, filling with forward fill")
                df_prepared[col].fillna(method='ffill', inplace=True)
                df_prepared[col].fillna(method='bfill', inplace=True)
            
            if np.isinf(df_prepared[col]).any():
                print(f"Warning: Found inf values in {col}, replacing with median")
                median_val = df_prepared[col].replace([np.inf, -np.inf], np.nan).median()
                df_prepared[col].replace([np.inf, -np.inf], median_val, inplace=True)
        
        # Drop any remaining rows with NaN in OHLCV
        initial_len = len(df_prepared)
        df_prepared = df_prepared.dropna(subset=ohlcv_cols)
        if len(df_prepared) < initial_len:
            print(f"Warning: Dropped {initial_len - len(df_prepared)} rows with NaN values")
        
        # Debug: Print ML signal statistics
        print(f"\nML Signal Statistics:")
        print(f"  Total rows: {len(df_prepared)}")
        print(f"  Buy signals (1): {(df_prepared['ML_Signal'] == 1).sum()}")
        print(f"  Sell signals (0): {(df_prepared['ML_Signal'] == 0).sum()}")
        print(f"\nProbability Statistics:")
        print(f"  Avg probability: {df_prepared['ML_Probability'].mean():.3f}")
        print(f"  Max probability: {df_prepared['ML_Probability'].max():.3f}")
        print(f"  Min probability: {df_prepared['ML_Probability'].min():.3f}")
        
        # Show how many buy signals meet different probability thresholds
        buy_signals = df_prepared[df_prepared['ML_Signal'] == 1]
        if len(buy_signals) > 0:
            print(f"\nBuy Signals Meeting Probability Thresholds:")
            for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
                count = (buy_signals['ML_Probability'] >= threshold).sum()
                pct = (count / len(buy_signals)) * 100
                print(f"  >= {threshold:.2f}: {count:4d} signals ({pct:5.1f}%)")
            print(f"\n  ⚠ If all counts are 0, your probability threshold is too high!")
            print(f"  ⚠ Consider lowering probability_threshold parameter.")
        
        return df_prepared
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        model,
        scaler,
        X_columns: List[str],
        probability_threshold: float = 0.6,
        trailing_stop_pct: float = 2.0,
        take_profit_pct: Optional[float] = None,
        position_size_pct: float = 1.0,
        plot: bool = True,
        printlog: bool = False,
        **kwargs
    ) -> Tuple[Dict, pd.DataFrame]:
        """
        Run backtest using ML model predictions.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data and features
        model : sklearn model
            Trained ML model
        scaler : sklearn scaler
            Fitted scaler
        X_columns : List[str]
            List of feature column names
        probability_threshold : float
            Minimum probability to enter trade
        trailing_stop_pct : float
            Trailing stop loss percentage
        take_profit_pct : float, optional
            Take profit percentage
        position_size_pct : float
            Fraction of equity to use per trade
        plot : bool
            Whether to plot results
        printlog : bool
            Whether to print logs
        **kwargs : dict
            Additional arguments
            
        Returns:
        --------
        results : Dict
            Backtest results and statistics
        trades : pd.DataFrame
            Trade history
        """
        # Prepare data with ML predictions
        df_prepared = self.prepare_data(
            df=df,
            model=model,
            scaler=scaler,
            X_columns=X_columns
        )
        
        # Create a Cerebro engine
        cerebro = bt.Cerebro()
        
        # Add strategy
        cerebro.addstrategy(
            BacktestBacktrader,
            probability_threshold=probability_threshold,
            trailing_stop_pct=trailing_stop_pct,
            take_profit_pct=take_profit_pct,
            position_size_pct=position_size_pct,
            printlog=printlog
        )
        
        # Create data feed
        data = MLPandasData(dataname=df_prepared)
        
        # Add data to Cerebro
        cerebro.adddata(data)
        
        # Set initial cash
        cerebro.broker.setcash(self.initial_cash)
        
        # Set commission
        cerebro.broker.setcommission(commission=self.commission)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        # Print starting conditions
        print(f'\nStarting Portfolio Value: ${cerebro.broker.getvalue():,.2f}')
        
        # Run backtest
        results = cerebro.run()
        strat = results[0]
        
        # Print ending conditions
        final_value = cerebro.broker.getvalue()
        print(f'Final Portfolio Value: ${final_value:,.2f}')
        
        # Extract results
        sharpe_ratio = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
        if sharpe_ratio is None:
            sharpe_ratio = 0
        
        drawdown_info = strat.analyzers.drawdown.get_analysis()
        returns_info = strat.analyzers.returns.get_analysis()
        trades_info = strat.analyzers.trades.get_analysis()
        
        # Calculate metrics
        total_return = final_value - self.initial_cash
        total_return_pct = (total_return / self.initial_cash) * 100
        
        # Get trade statistics
        total_trades = trades_info.get('total', {}).get('total', 0)
        won_trades = trades_info.get('won', {}).get('total', 0)
        lost_trades = trades_info.get('lost', {}).get('total', 0)
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Compile results
        results_dict = {
            'initial_capital': self.initial_cash,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': drawdown_info.get('max', {}).get('drawdown', 0),
            'total_trades': total_trades,
            'won_trades': won_trades,
            'lost_trades': lost_trades,
            'win_rate': win_rate,
            'avg_win': trades_info.get('won', {}).get('pnl', {}).get('average', 0),
            'avg_loss': trades_info.get('lost', {}).get('pnl', {}).get('average', 0),
            'best_trade': trades_info.get('won', {}).get('pnl', {}).get('max', 0),
            'worst_trade': trades_info.get('lost', {}).get('pnl', {}).get('max', 0),
        }
        
        # Get trades from strategy
        trades_df = pd.DataFrame(strat.trades_list) if strat.trades_list else pd.DataFrame()
        
        # Plot if requested
        if plot:
            try:
                print("\nGenerating backtest plot...")
                cerebro.plot(style='candlestick', iplot=False)
                print("✓ Plot generated successfully")
            except Exception as e:
                print(f"⚠ Warning: Could not generate plot: {e}")
                print("  Continuing without plot...")
        
        return results_dict, trades_df
    
    def optimize(
        self,
        df: pd.DataFrame,
        model,
        scaler,
        X_columns: List[str],
        probability_threshold_range: Tuple[float, float, float] = (0.5, 0.8, 0.05),
        trailing_stop_range: Tuple[float, float, float] = (1.0, 5.0, 0.5),
        **kwargs
    ) -> List[Dict]:
        """
        Optimize strategy parameters.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data and features
        model : sklearn model
            Trained ML model
        scaler : sklearn scaler
            Fitted scaler
        X_columns : List[str]
            List of feature column names
        probability_threshold_range : Tuple[float, float, float]
            (start, end, step) for probability threshold
        trailing_stop_range : Tuple[float, float, float]
            (start, end, step) for trailing stop percentage
        **kwargs : dict
            Additional arguments
            
        Returns:
        --------
        optimization_results : List[Dict]
            List of optimization results
        """
        # Prepare data with ML predictions
        df_prepared = self.prepare_data(
            df=df,
            model=model,
            scaler=scaler,
            X_columns=X_columns
        )
        
        # Create a Cerebro engine
        cerebro = bt.Cerebro()
        
        # Add strategy with optimization parameters
        prob_start, prob_end, prob_step = probability_threshold_range
        stop_start, stop_end, stop_step = trailing_stop_range
        
        cerebro.optstrategy(
            BacktestBacktrader,
            probability_threshold=np.arange(prob_start, prob_end, prob_step),
            trailing_stop_pct=np.arange(stop_start, stop_end, stop_step)
        )
        
        # Create data feed
        data = MLPandasData(dataname=df_prepared)
        
        # Add data to Cerebro
        cerebro.adddata(data)
        
        # Set initial cash
        cerebro.broker.setcash(self.initial_cash)
        
        # Set commission
        cerebro.broker.setcommission(commission=self.commission)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        print('\nRunning optimization...')
        
        # Run optimization
        opt_results = cerebro.run()
        
        # Extract results
        optimization_results = []
        for result in opt_results:
            strat = result[0]
            sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
            if sharpe is None:
                sharpe = 0
            
            optimization_results.append({
                'probability_threshold': strat.params.probability_threshold,
                'trailing_stop_pct': strat.params.trailing_stop_pct,
                'final_value': cerebro.broker.getvalue(),
                'sharpe_ratio': sharpe,
                'max_drawdown': strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)
            })
        
        # Sort by final value
        optimization_results.sort(key=lambda x: x['final_value'], reverse=True)
        
        print(f'\nOptimization complete. Tested {len(optimization_results)} combinations.')
        
        return optimization_results
    
    def print_results(self, results: Dict):
        """
        Print formatted backtest results.
        
        Parameters:
        -----------
        results : Dict
            Backtest results dictionary
        """
        print("\n" + "="*80)
        print("BACKTEST RESULTS (Backtrader)")
        print("="*80)
        
        print("\nCapital:")
        print(f"  Initial Capital:        ${results['initial_capital']:,.2f}")
        print(f"  Final Value:            ${results['final_value']:,.2f}")
        print(f"  Total Return:           ${results['total_return']:,.2f}")
        print(f"  Total Return %:         {results['total_return_pct']:.2f}%")
        
        print("\nTrades:")
        print(f"  Total Trades:           {results['total_trades']}")
        print(f"  Won Trades:             {results['won_trades']}")
        print(f"  Lost Trades:            {results['lost_trades']}")
        print(f"  Win Rate:               {results['win_rate']:.2f}%")
        print(f"  Average Win:            ${results['avg_win']:.2f}")
        print(f"  Average Loss:           ${results['avg_loss']:.2f}")
        print(f"  Best Trade:             ${results['best_trade']:.2f}")
        print(f"  Worst Trade:            ${results['worst_trade']:.2f}")
        
        print("\nRisk Metrics:")
        print(f"  Max Drawdown:           {results['max_drawdown']:.2f}%")
        print(f"  Sharpe Ratio:           {results['sharpe_ratio']:.2f}")
        
        print("="*80)
