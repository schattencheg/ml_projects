"""
BaseBacktest - Abstract base class for all backtesting implementations.

Uses RiskManager for:
- Position sizing (default 2% of current capital)
- Exit after exactly N bars
- Cooldown: new positions only on the bar after previous closes
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from src.risk_management.fixed_bars_risk_manager import FixedBarsCountRiskManager


class BaseBacktest(ABC):
    """
    Abstract base class for backtesting implementations.
    
    All backtest implementations must inherit from this class and implement
    the required abstract methods.
    
    RiskManager controls:
    - Position sizing (% of current capital)
    - Exit timing (after N bars)
    - Cooldown between trades
    """
    
    def __init__(self,
                 initial_capital: float = 10000.0,
                 commission: float = 0.001,
                 position_size: float = 0.02,
                 bars_to_hold: int = 15,
                 risk_manager: Optional['FixedBarsCountRiskManager'] = None):
        """
        Initialize base backtest.
        
        Args:
            initial_capital: Starting capital
            commission: Commission rate (e.g., 0.001 = 0.1%)
            position_size: Position size as fraction of capital (default 0.02 = 2%)
            bars_to_hold: Number of bars to hold position before exiting (default 15)
            risk_manager: Optional custom RiskManager. If None, creates FixedBarsCountRiskManager
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.position_size = position_size
        self.bars_to_hold = bars_to_hold
        
        # Create or use provided RiskManager
        if risk_manager is not None:
            self.risk_manager = risk_manager
        else:
            from src.risk_management.fixed_bars_risk_manager import FixedBarsCountRiskManager
            self.risk_manager = FixedBarsCountRiskManager(
                bars_to_hold=bars_to_hold,
                position_size_pct=position_size
            )
        
        # Results storage
        self.results = None
        self.metrics = None
        self.trades = []
        
    @abstractmethod
    def run(self,
            df: pd.DataFrame,
            model: Any,
            scaler: Any,
            feature_cols: list,
            price_col: str = 'close',
            **kwargs) -> Dict[str, Any]:
        """
        Run backtest on data.
        
        Args:
            df: DataFrame with OHLCV data and features
            model: Trained ML model
            scaler: Fitted scaler for features
            feature_cols: List of feature column names
            price_col: Name of price column to use
            **kwargs: Additional backend-specific parameters
            
        Returns:
            Dictionary with backtest results
        """
        pass
    
    @abstractmethod
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Returns:
            Dictionary with metrics (returns, sharpe, drawdown, etc.)
        """
        pass
    
    def get_results(self) -> Optional[Dict[str, Any]]:
        """
        Get backtest results.
        
        Returns:
            Dictionary with results or None if not run yet
        """
        return self.results
    
    def get_metrics(self) -> Optional[Dict[str, float]]:
        """
        Get performance metrics.
        
        Returns:
            Dictionary with metrics or None if not calculated yet
        """
        return self.metrics
    
    def get_trades(self) -> list:
        """
        Get list of trades.
        
        Returns:
            List of trade dictionaries
        """
        return self.trades
    
    def _calculate_returns(self, equity_curve: pd.Series) -> float:
        """
        Calculate total return.
        
        Args:
            equity_curve: Series of equity values
            
        Returns:
            Total return as decimal
        """
        if len(equity_curve) == 0:
            return 0.0
        return (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        excess_returns = returns - risk_free_rate
        return np.sqrt(252) * (excess_returns.mean() / returns.std())
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            equity_curve: Series of equity values
            
        Returns:
            Maximum drawdown as decimal
        """
        if len(equity_curve) == 0:
            return 0.0
        
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        return drawdown.min()
    
    def _calculate_win_rate(self, trades: list) -> float:
        """
        Calculate win rate.
        
        Args:
            trades: List of trade dictionaries with 'pnl' key
            
        Returns:
            Win rate as decimal
        """
        if len(trades) == 0:
            return 0.0
        
        winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
        return winning_trades / len(trades)
    
    def print_results(self):
        """Print backtest results in a formatted way."""
        if self.metrics is None:
            print("No results available. Run backtest first.")
            return
        
        print("\n" + "="*80)
        print(f"BACKTEST RESULTS - {self.__class__.__name__}")
        print("="*80)
        
        print(f"\nCapital:")
        print(f"  Initial Capital:  ${self.initial_capital:>12,.2f}")
        print(f"  Final Capital:    ${self.metrics.get('final_capital', 0):>12,.2f}")
        print(f"  Total Return:     {self.metrics.get('total_return', 0)*100:>12.2f}%")
        
        print(f"\nRisk Metrics:")
        print(f"  Sharpe Ratio:     {self.metrics.get('sharpe_ratio', 0):>12.2f}")
        print(f"  Max Drawdown:     {self.metrics.get('max_drawdown', 0)*100:>12.2f}%")
        
        print(f"\nTrading:")
        print(f"  Total Trades:     {self.metrics.get('total_trades', 0):>12d}")
        print(f"  Win Rate:         {self.metrics.get('win_rate', 0)*100:>12.2f}%")
        print(f"  Commission:       {self.commission*100:>12.3f}%")
        
        print("="*80 + "\n")
    
    def save_results(self, filepath: Union[str, Path]):
        """
        Save backtest results to file.
        
        Args:
            filepath: Path to save results
        """
        import json
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        results_to_save = {
            'backend': self.__class__.__name__,
            'parameters': {
                'initial_capital': self.initial_capital,
                'commission': self.commission,
                'position_size': self.position_size,
                'bars_to_hold': self.bars_to_hold
            },
            'risk_manager': self.risk_manager.get_info(),
            'metrics': self.metrics,
            'num_trades': len(self.trades)
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"✓ Results saved to {filepath}")
