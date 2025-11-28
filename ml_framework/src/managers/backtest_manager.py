"""
BacktestManager - Manages backtesting with multiple backends.
Supports: NoLib (custom), Backtrader, Backtesting.py
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Literal
from pathlib import Path
import warnings

from src.models_lib import BaseModel
from src.managers.scaler_manager import ScalerManager


class BacktestManager:
    """
    Manages backtesting operations with multiple backends.
    
    Features:
    - Multiple backends: NoLib, Backtrader, Backtesting.py
    - Position sizing and commission modeling
    - Performance metrics calculation
    - Results storage
    
    Example:
        >>> bt_mgr = BacktestManager(backend='nolib', verbose=False)
        >>> results = bt_mgr.run(data, model, scaler_manager, feature_cols)
        >>> equity = bt_mgr.get_equity_curve()
    """
    
    def __init__(self, 
                 backend: Literal['nolib', 'backtrader', 'backtesting'] = 'nolib',
                 initial_capital: float = 10000,
                 commission: float = 0.001,
                 verbose: bool = True):
        """
        Initialize BacktestManager.
        
        Args:
            backend: Backtesting backend to use
            initial_capital: Initial capital for backtesting
            commission: Commission rate (e.g., 0.001 = 0.1%)
            verbose: If True, print status messages (default: True)
        """
        self.backend = backend
        self.initial_capital = initial_capital
        self.commission = commission
        self.results = {}
        self.verbose = verbose
        
    def run(self,
           data: pd.DataFrame,
           model: BaseModel,
           scaler_manager: Optional[ScalerManager] = None,
           feature_cols: Optional[List[str]] = None,
           price_col: str = 'Close',
           **kwargs) -> Dict[str, Any]:
        """
        Run backtest using the selected backend.
        
        Args:
            data: DataFrame with OHLCV data and features
            model: Trained model for predictions
            scaler_manager: ScalerManager for feature scaling
            feature_cols: List of feature columns
            price_col: Column name for price data
            **kwargs: Additional backend-specific parameters
            
        Returns:
            Dictionary with backtest results
        """
        if self.backend == 'nolib':
            return self._run_nolib(data, model, scaler_manager, feature_cols, price_col, **kwargs)
        elif self.backend == 'backtrader':
            return self._run_backtrader(data, model, scaler_manager, feature_cols, price_col, **kwargs)
        elif self.backend == 'backtesting':
            return self._run_backtesting_py(data, model, scaler_manager, feature_cols, price_col, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _run_nolib(self,
                  data: pd.DataFrame,
                  model: BaseModel,
                  scaler_manager: Optional[ScalerManager],
                  feature_cols: Optional[List[str]],
                  price_col: str,
                  **kwargs) -> Dict[str, Any]:
        """
        Run backtest using custom implementation (no external library).
        
        This is a simple vectorized backtest implementation.
        """
        if self.verbose:
            print("\n" + "="*70)
            print("RUNNING BACKTEST (NoLib Backend)")
            print("="*70)
        
        # Prepare features
        if feature_cols is None:
            feature_cols = [col for col in data.columns 
                          if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'target']]
        
        X = data[feature_cols].values
        
        # Scale features if scaler provided
        if scaler_manager is not None:
            X = scaler_manager.transform(data[feature_cols]).values
        
        # Get predictions
        predictions = model.predict(X)
        
        # Convert predictions to signals (-1, 0, 1)
        # Assuming predictions are class labels that need mapping
        unique_preds = np.unique(predictions)
        if len(unique_preds) == 3:
            # Map to -1, 0, 1
            signal_map = {unique_preds[0]: -1, unique_preds[1]: 0, unique_preds[2]: 1}
        elif len(unique_preds) == 2:
            # Map to -1, 1 (no neutral)
            signal_map = {unique_preds[0]: -1, unique_preds[1]: 1}
        else:
            # Default: use predictions as is
            signal_map = {val: val for val in unique_preds}
        
        signals = np.array([signal_map.get(p, 0) for p in predictions])
        
        # Calculate returns
        prices = data[price_col].values
        returns = np.diff(prices) / prices[:-1]
        
        # Align signals with returns (shift signals by 1)
        signals = signals[:-1]
        
        # Calculate strategy returns
        strategy_returns = signals * returns
        
        # Apply commission
        position_changes = np.abs(np.diff(np.concatenate([[0], signals])))
        commission_costs = position_changes * self.commission
        strategy_returns = strategy_returns - commission_costs
        
        # Calculate equity curve
        equity = self.initial_capital * np.cumprod(1 + strategy_returns)
        equity = np.concatenate([[self.initial_capital], equity])
        
        # Calculate metrics
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital
        
        # Annualized return (assuming daily data)
        n_days = len(equity)
        years = n_days / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Sharpe ratio
        if len(strategy_returns) > 0 and np.std(strategy_returns) > 0:
            sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        cummax = np.maximum.accumulate(equity)
        drawdown = (equity - cummax) / cummax
        max_drawdown = np.min(drawdown)
        
        # Win rate
        winning_trades = np.sum(strategy_returns > 0)
        total_trades = np.sum(position_changes > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Buy & Hold comparison
        buy_hold_return = (prices[-1] - prices[0]) / prices[0]
        
        results = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': int(total_trades),
            'buy_hold_return': buy_hold_return,
            'equity_curve': equity,
            'signals': signals,
            'strategy_returns': strategy_returns,
            'backend': 'nolib'
        }
        
        self.results = results
        if self.verbose:
            self._print_results(results)
        
        return results
    
    def _run_backtrader(self,
                       data: pd.DataFrame,
                       model: BaseModel,
                       scaler_manager: Optional[ScalerManager],
                       feature_cols: Optional[List[str]],
                       price_col: str,
                       **kwargs) -> Dict[str, Any]:
        """
        Run backtest using Backtrader library.
        """
        try:
            import backtrader as bt
        except ImportError:
            raise ImportError("Backtrader is not installed. Install with: pip install backtrader")
        
        if self.verbose:
            print("\n" + "="*70)
            print("RUNNING BACKTEST (Backtrader Backend)")
            print("="*70)
            print("Note: Backtrader integration is a placeholder. Full implementation needed.")
            print("="*70 + "\n")
        
        # Placeholder - full implementation would create a Backtrader strategy
        warnings.warn("Backtrader backend not fully implemented. Using NoLib instead.")
        return self._run_nolib(data, model, scaler_manager, feature_cols, price_col, **kwargs)
    
    def _run_backtesting_py(self,
                           data: pd.DataFrame,
                           model: BaseModel,
                           scaler_manager: Optional[ScalerManager],
                           feature_cols: Optional[List[str]],
                           price_col: str,
                           **kwargs) -> Dict[str, Any]:
        """
        Run backtest using Backtesting.py library.
        """
        try:
            from backtesting import Backtest, Strategy
        except ImportError:
            raise ImportError("Backtesting.py is not installed. Install with: pip install backtesting")
        
        if self.verbose:
            print("\n" + "="*70)
            print("RUNNING BACKTEST (Backtesting.py Backend)")
            print("="*70)
            print("Note: Backtesting.py integration is a placeholder. Full implementation needed.")
            print("="*70 + "\n")
        
        # Placeholder - full implementation would create a Backtesting.py strategy
        warnings.warn("Backtesting.py backend not fully implemented. Using NoLib instead.")
        return self._run_nolib(data, model, scaler_manager, feature_cols, price_col, **kwargs)
    
    def _print_results(self, results: Dict[str, Any]):
        """Print backtest results."""
        print("\n" + "="*70)
        print("BACKTEST RESULTS")
        print("="*70)
        
        print(f"\nPerformance Metrics:")
        print(f"  Total Return:       {results['total_return']*100:>8.2f}%")
        print(f"  Annualized Return:  {results['annualized_return']*100:>8.2f}%")
        print(f"  Sharpe Ratio:       {results['sharpe_ratio']:>8.2f}")
        print(f"  Max Drawdown:       {results['max_drawdown']*100:>8.2f}%")
        print(f"  Win Rate:           {results['win_rate']*100:>8.2f}%")
        print(f"  Number of Trades:   {results['num_trades']:>8}")
        
        print(f"\nBenchmark:")
        print(f"  Buy & Hold Return:  {results['buy_hold_return']*100:>8.2f}%")
        
        outperformance = results['total_return'] - results['buy_hold_return']
        print(f"  Outperformance:     {outperformance*100:>8.2f}%")
        
        print("="*70 + "\n")
    
    def get_results(self) -> Dict[str, Any]:
        """Get backtest results."""
        return self.results
    
    def get_equity_curve(self) -> np.ndarray:
        """Get equity curve."""
        return self.results.get('equity_curve', np.array([]))
    
    def get_signals(self) -> np.ndarray:
        """Get trading signals."""
        return self.results.get('signals', np.array([]))
