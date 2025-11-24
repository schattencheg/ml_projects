"""
BacktestBacktrader - Backtesting using Backtrader library.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base_backtest import BaseBacktest


class BacktestBacktrader(BaseBacktest):
    """
    Backtrader-based backtesting implementation.
    
    Features:
    - Event-driven backtesting
    - Realistic order execution
    - Built-in indicators
    - Live trading ready
    """
    
    def __init__(self,
                 initial_capital: float = 10000.0,
                 commission: float = 0.001,
                 position_size: float = 1.0):
        """
        Initialize Backtrader backtest.
        
        Args:
            initial_capital: Starting capital
            commission: Commission rate
            position_size: Position size as fraction of capital
        """
        super().__init__(initial_capital, commission, position_size)
        
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
            return {}
        
        print(f"\nRunning {self.__class__.__name__} backtest...")
        
        # Prepare predictions
        X = df[feature_cols].values
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        
        # Add predictions to dataframe
        df_bt = df.copy()
        df_bt['prediction'] = predictions
        
        # Define Backtrader strategy
        class MLStrategy(bt.Strategy):
            params = (
                ('position_size', position_size),
            )
            
            def __init__(strategy_self):
                strategy_self.order = None
                strategy_self.trades_list = []
                
            def next(strategy_self):
                if strategy_self.order:
                    return
                
                prediction = strategy_self.data.prediction[0]
                
                if not strategy_self.position:
                    if prediction == 1:
                        # Buy signal
                        size = int((strategy_self.broker.getcash() * strategy_self.params.position_size) / strategy_self.data.close[0])
                        if size > 0:
                            strategy_self.order = strategy_self.buy(size=size)
                else:
                    if prediction == 0:
                        # Sell signal
                        strategy_self.order = strategy_self.sell(size=strategy_self.position.size)
            
            def notify_order(strategy_self, order):
                if order.status in [order.Completed]:
                    if order.isbuy():
                        pass
                    elif order.issell():
                        pass
                
                strategy_self.order = None
            
            def notify_trade(strategy_self, trade):
                if trade.isclosed:
                    strategy_self.trades_list.append({
                        'pnl': trade.pnl,
                        'pnlcomm': trade.pnlcomm
                    })
        
        # Create Backtrader data feed
        class PandasData(bt.feeds.PandasData):
            lines = ('prediction',)
            params = (
                ('prediction', -1),
            )
        
        # Initialize Cerebro
        cerebro = bt.Cerebro()
        
        # Add strategy
        cerebro.addstrategy(MLStrategy, position_size=self.position_size)
        
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
        
        # Create equity curve (approximation)
        equity_curve = pd.Series(
            np.linspace(self.initial_capital, final_value, len(df)),
            index=df.index
        )
        
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
