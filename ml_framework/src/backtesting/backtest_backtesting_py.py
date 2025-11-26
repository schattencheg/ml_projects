"""
BacktestBacktestingPy - Backtesting using backtesting.py library.

Uses RiskManager (from base class) for:
- Position sizing (default 2% of current capital)
- Exit after exactly N bars
- Cooldown: new positions only on the bar after previous closes
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from src.backtesting.base_backtest import BaseBacktest


class BacktestBacktestingPy(BaseBacktest):
    """
    Backtesting.py-based backtesting implementation.
    
    Features:
    - Fast vectorized backtesting
    - Built-in optimization
    - Interactive Bokeh visualizations
    - Easy to use API
    - RiskManager controls position sizing and exit timing (inherited from BaseBacktest)
    """
    
    def __init__(self,
                 initial_capital: float = 10000.0,
                 commission: float = 0.001,
                 position_size: float = 0.02,
                 bars_to_hold: int = 15,
                 risk_manager=None):
        """
        Initialize Backtesting.py backtest.
        
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
        Run backtest using backtesting.py.
        
        Args:
            df: DataFrame with OHLCV data and features
            model: Trained ML model
            scaler: Fitted scaler
            feature_cols: Feature column names
            price_col: Price column name
            **kwargs: Additional parameters (plot=True to show plot)
            
        Returns:
            Dictionary with backtest results
        """
        try:
            from backtesting import Backtest, Strategy
            from backtesting.lib import FractionalBacktest
        except ImportError:
            print("❌ backtesting.py not installed. Install with: pip install backtesting")
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
        
        # Prepare data for backtesting.py (needs OHLC columns capitalized)
        df_bt = df.copy()
        df_bt['Prediction'] = predictions
        
        # Ensure required columns exist and are capitalized
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col.lower() in df_bt.columns:
                df_bt[col] = df_bt[col.lower()]
        
        # Pass RiskManager parameters to strategy
        position_size_pct = self.risk_manager.position_size_pct
        bars_to_hold_val = self.bars_to_hold
        
        class MLStrategy(Strategy):
            def init(self):
                # Reset state for each run (instance variables, not class variables)
                self._entry_bar = None
                self._last_exit_bar = None
            
            def next(self):
                current_bar = len(self.data) - 1
                prediction = self.data.Prediction[-1]
                
                # Check exit condition (fixed bars)
                if self.position and self._entry_bar is not None:
                    bars_held = current_bar - self._entry_bar
                    if bars_held >= bars_to_hold_val:
                        # Exit after N bars
                        self._last_exit_bar = current_bar
                        self._entry_bar = None  # Reset entry bar
                        self.position.close()
                        return
                
                # Check entry condition
                if prediction == 1 and not self.position:
                    # Check cooldown: can only open on bar AFTER last exit
                    can_open = (self._last_exit_bar is None or 
                               current_bar > self._last_exit_bar)
                    
                    if can_open:
                        # Position sizing: use position_size_pct of equity
                        # backtesting.py size parameter is fraction of equity
                        self._entry_bar = current_bar
                        self.buy(size=position_size_pct)
        
        # Run backtest with FractionalBacktest for high-priced assets like BTC
        # FractionalBacktest allows trading fractional units
        bt = FractionalBacktest(
            df_bt,
            MLStrategy,
            cash=self.initial_capital,
            commission=self.commission,
            exclusive_orders=True,
            trade_on_close=True,  # Execute at close price
            hedging=False
        )
        
        stats = bt.run()
        
        # Extract trades and convert to our format
        trades_df = stats._trades if hasattr(stats, '_trades') else pd.DataFrame()
        self.trades = []
        if len(trades_df) > 0:
            for _, trade in trades_df.iterrows():
                self.trades.append({
                    'entry_idx': int(trade.get('EntryBar', 0)),
                    'exit_idx': int(trade.get('ExitBar', 0)),
                    'entry_price': float(trade.get('EntryPrice', 0)),
                    'exit_price': float(trade.get('ExitPrice', 0)),
                    'shares': float(trade.get('Size', 0)),
                    'pnl': float(trade.get('PnL', 0)),
                    'exit_reason': 'fixed_bars'
                })
        
        # Create equity curve - ensure it's a proper Series with df index
        if hasattr(stats, '_equity_curve') and stats._equity_curve is not None:
            equity_curve = stats._equity_curve['Equity']
            # Reindex to match original df if needed
            if len(equity_curve) != len(df):
                equity_curve = pd.Series(equity_curve.values, index=df.index[:len(equity_curve)])
        else:
            equity_curve = pd.Series([self.initial_capital] * len(df), index=df.index)
        
        self.results = {
            'equity_curve': equity_curve,
            'trades': self.trades,
            'final_capital': float(stats['Equity Final [$]']) if 'Equity Final [$]' in stats else self.initial_capital,
            'stats': stats
        }
        
        # Calculate metrics
        self.metrics = self.calculate_metrics()
        
        # Plot if requested
        if kwargs.get('plot', False):
            try:
                bt.plot()
            except Exception as e:
                print(f"⚠ Could not create plot: {e}")
        
        print(f"✓ Backtest complete: {stats['# Trades']} trades")
        
        return self.results
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Returns:
            Dictionary with metrics
        """
        if self.results is None:
            return {}
        
        stats = self.results.get('stats')
        equity_curve = self.results['equity_curve']
        
        if stats is not None:
            # Use stats from backtesting.py
            metrics = {
                'initial_capital': self.initial_capital,
                'final_capital': float(stats['Equity Final [$]']),
                'total_return': float(stats['Return [%]']) / 100.0,
                'sharpe_ratio': float(stats['Sharpe Ratio']) if not pd.isna(stats['Sharpe Ratio']) else 0.0,
                'max_drawdown': float(stats['Max. Drawdown [%]']) / 100.0,
                'total_trades': int(stats['# Trades']),
                'win_rate': float(stats['Win Rate [%]']) / 100.0 if 'Win Rate [%]' in stats else 0.0,
            }
        else:
            # Fallback to manual calculation
            returns = equity_curve.pct_change().dropna()
            
            metrics = {
                'initial_capital': self.initial_capital,
                'final_capital': self.results['final_capital'],
                'total_return': self._calculate_returns(equity_curve),
                'sharpe_ratio': self._calculate_sharpe_ratio(returns),
                'max_drawdown': self._calculate_max_drawdown(equity_curve),
                'total_trades': len(self.trades),
                'win_rate': 0.0,
            }
        
        return metrics
