"""
BacktestBacktrader - Backtesting using Backtrader library.

Uses RiskManager (from base class) for:
- Position sizing (default 2% of current capital)
- Exit after exactly N bars
- Cooldown: new positions only on the bar after previous closes
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from src.backtesting.base_backtest import BaseBacktest


class BacktestBacktrader(BaseBacktest):
    """
    Backtrader-based backtesting implementation.
    
    Features:
    - Event-driven backtesting
    - Realistic order execution
    - Built-in indicators
    - Live trading ready
    - RiskManager controls position sizing and exit timing (inherited from BaseBacktest)
    """
    
    def __init__(self,
                 initial_capital: float = 10000.0,
                 commission: float = 0.001,
                 position_size: float = 0.02,
                 bars_to_hold: int = 15,
                 risk_manager=None):
        """
        Initialize Backtrader backtest.
        
        Args:
            initial_capital: Starting capital
            commission: Commission rate
            position_size: Position size as fraction of capital (default 0.02 = 2%)
            bars_to_hold: Number of bars to hold position before exiting (default 15)
            risk_manager: Optional custom RiskManager instance (passed to base class)
        """
        super().__init__(
            initial_capital=initial_capital,
            commission=commission,
            position_size=position_size,
            bars_to_hold=bars_to_hold,
            risk_manager=risk_manager
        )
        
    def run(self,
            df: pd.DataFrame,
            model: Any,
            scaler: Any,
            feature_cols: list,
            price_col: str = 'close',
            **kwargs) -> Dict[str, Any]:
        """
        Run backtest using Backtrader.
        
        Args:
            df: DataFrame with OHLCV data and features
            model: Trained ML model
            scaler: Fitted scaler
            feature_cols: Feature column names
            price_col: Price column name
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with backtest results
        """
        try:
            import backtrader as bt
        except ImportError:
            print("❌ Backtrader not installed. Install with: pip install backtrader")
            # Return empty results structure
            self.trades = []
            self.results = {
                'equity_curve': pd.Series([self.initial_capital] * len(df), index=df.index),
                'trades': [],
                'final_capital': self.initial_capital
            }
            self.metrics = self.calculate_metrics()
            return self.results
        
        print(f"\nRunning {self.__class__.__name__} backtest...")
        print(f"  RiskManager: {self.risk_manager}")
        
        # Reset risk manager state
        self.risk_manager.reset()
        
        # Prepare predictions
        X = df[feature_cols].values
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        
        # Add predictions to dataframe
        df_bt = df.copy()
        df_bt['prediction'] = predictions
        
        # Pass RiskManager parameters to strategy
        risk_manager = self.risk_manager
        commission = self.commission

        # Define Backtrader strategy with RiskManager
        class MLStrategy(bt.Strategy):
            params = (
                ('bars_to_hold', risk_manager.bars_to_hold),
            )
            
            def __init__(strategy_self):
                strategy_self.order = None
                strategy_self.trades_list = []
                strategy_self.entry_bar = None
                strategy_self.entry_price = None
                strategy_self.entry_size = None
                strategy_self.position_type = None  # 'long' or 'short'
                strategy_self.equity_curve = []  # Track equity at each bar
                strategy_self.last_exit_bar = None  # For cooldown
                
            def next(strategy_self):
                current_bar = len(strategy_self) - 1
                
                # Track equity at each bar
                strategy_self.equity_curve.append(strategy_self.broker.getvalue())
                
                if strategy_self.order:
                    return
                
                prediction = strategy_self.data.prediction[0]
                
                # Check exit condition (fixed bars)
                if strategy_self.position and strategy_self.entry_bar is not None:
                    bars_held = current_bar - strategy_self.entry_bar
                    if bars_held >= strategy_self.params.bars_to_hold:
                        # Exit after N bars - close position
                        strategy_self.order = strategy_self.close()
                        return
                
                # Check entry condition: +1 = long, -1 = short, 0 = no trade
                if not strategy_self.position and prediction != 0:
                    # Check cooldown: can only open on bar AFTER last exit
                    can_open = (strategy_self.last_exit_bar is None or 
                               current_bar > strategy_self.last_exit_bar)
                    
                    if can_open:
                        # Position sizing: 2% of current capital
                        current_capital = strategy_self.broker.getvalue()
                        shares = risk_manager.get_position_size(
                            capital=current_capital,
                            price=strategy_self.data.close[0],
                            commission=commission
                        )
                        # For high-priced assets like BTC, allow fractional shares
                        if shares > 0.0001:  # Minimum viable position
                            if prediction == 1:  # Long
                                strategy_self.position_type = 'long'
                                strategy_self.order = strategy_self.buy(size=shares)
                            else:  # prediction == -1, Short
                                strategy_self.position_type = 'short'
                                strategy_self.order = strategy_self.sell(size=shares)
            
            def notify_order(strategy_self, order):
                if order.status in [order.Completed]:
                    # Entry: when we have no position and order completes
                    if strategy_self.entry_bar is None:
                        strategy_self.entry_bar = len(strategy_self) - 1
                        strategy_self.entry_price = order.executed.price
                        strategy_self.entry_size = order.executed.size
                    else:
                        # Exit: update last exit bar for cooldown
                        strategy_self.last_exit_bar = len(strategy_self) - 1
                
                strategy_self.order = None
            
            def notify_trade(strategy_self, trade):
                if trade.isclosed:
                    # Get exit bar index (current bar)
                    exit_bar = len(strategy_self) - 1
                    
                    # Calculate exit price from PnL
                    if strategy_self.position_type == 'long':
                        exit_price = strategy_self.entry_price + (trade.pnl / strategy_self.entry_size) if strategy_self.entry_size else 0
                    else:  # short
                        exit_price = strategy_self.entry_price - (trade.pnl / strategy_self.entry_size) if strategy_self.entry_size else 0
                    
                    # Determine exit reason
                    bars_held = exit_bar - strategy_self.entry_bar if strategy_self.entry_bar else 0
                    exit_reason = 'fixed_bars' if bars_held >= strategy_self.params.bars_to_hold else 'end_of_data'
                    
                    strategy_self.trades_list.append({
                        'entry_idx': strategy_self.entry_bar,
                        'exit_idx': exit_bar,
                        'entry_price': strategy_self.entry_price,
                        'exit_price': exit_price,
                        'shares': strategy_self.entry_size,
                        'position_type': strategy_self.position_type,
                        'pnl': trade.pnl,
                        'exit_reason': exit_reason
                    })
                    
                    # Reset entry tracking
                    strategy_self.entry_bar = None
                    strategy_self.entry_price = None
                    strategy_self.entry_size = None
                    strategy_self.position_type = None
        
        # Create Backtrader data feed
        class PandasData(bt.feeds.PandasData):
            lines = ('prediction',)
            params = (
                ('prediction', -1),
            )
        
        # Initialize Cerebro
        cerebro = bt.Cerebro()
        
        # Add strategy with bars_to_hold parameter
        cerebro.addstrategy(MLStrategy, bars_to_hold=self.bars_to_hold)
        
        # Prepare data
        df_bt.index = pd.to_datetime(df_bt.index)
        data = PandasData(dataname=df_bt)
        cerebro.adddata(data)
        
        # Set broker parameters
        cerebro.broker.setcash(self.initial_capital)
        cerebro.broker.setcommission(commission=self.commission)
        
        # Run backtest
        strategies = cerebro.run()
        strategy = strategies[0]
        
        # Get final value
        final_value = cerebro.broker.getvalue()
        
        # Extract trades
        self.trades = strategy.trades_list
        
        # Get actual equity curve from strategy
        equity_values = strategy.equity_curve
        
        # Ensure equity curve has same length as df
        if len(equity_values) < len(df):
            # Pad with initial capital if needed
            equity_values = [self.initial_capital] * (len(df) - len(equity_values)) + equity_values
        elif len(equity_values) > len(df):
            # Trim if too long
            equity_values = equity_values[:len(df)]
        
        equity_curve = pd.Series(equity_values, index=df.index)
        
        self.results = {
            'equity_curve': equity_curve,
            'trades': self.trades,
            'final_capital': final_value
        }
        
        # Calculate metrics
        self.metrics = self.calculate_metrics()
        
        print(f"✓ Backtest complete: {len(self.trades)} trades")
        
        return self.results
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Returns:
            Dictionary with metrics
        """
        if self.results is None:
            return {}
        
        equity_curve = self.results['equity_curve']
        trades = self.results['trades']
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        metrics = {
            'initial_capital': self.initial_capital,
            'final_capital': self.results['final_capital'],
            'total_return': self._calculate_returns(equity_curve),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(equity_curve),
            'total_trades': len(trades),
            'win_rate': self._calculate_win_rate(trades) if trades else 0.0,
        }
        
        return metrics
