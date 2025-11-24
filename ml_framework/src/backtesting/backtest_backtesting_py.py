"""
BacktestBacktestingPy - Backtesting using backtesting.py library.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base_backtest import BaseBacktest


class BacktestBacktestingPy(BaseBacktest):
    """
    Backtesting.py-based backtesting implementation.
    
    Features:
    - Fast vectorized backtesting
    - Built-in optimization
    - Interactive Bokeh visualizations
    - Easy to use API
    """
    
    def __init__(self,
                 initial_capital: float = 10000.0,
                 commission: float = 0.001,
                 position_size: float = 1.0):
        """
        Initialize Backtesting.py backtest.
        
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
        except ImportError:
            print("❌ backtesting.py not installed. Install with: pip install backtesting")
            return {}
        
        print(f"\nRunning {self.__class__.__name__} backtest...")
        
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
        
        # Define strategy
        position_size_val = self.position_size
        
        class MLStrategy(Strategy):
            def init(self):
                pass
            
            def next(self):
                prediction = self.data.Prediction[-1]
                
                if prediction == 1 and not self.position:
                    # Buy signal
                    self.buy(size=position_size_val)
                elif prediction == 0 and self.position:
                    # Sell signal
                    self.position.close()
        
        # Run backtest
        bt = Backtest(
            df_bt,
            MLStrategy,
            cash=self.initial_capital,
            commission=self.commission,
            exclusive_orders=True
        )
        
        stats = bt.run()
        
        # Extract results
        self.trades = stats._trades if hasattr(stats, '_trades') else []
        
        # Create equity curve
        equity_curve = stats._equity_curve['Equity'] if hasattr(stats, '_equity_curve') else pd.Series([self.initial_capital])
        
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
