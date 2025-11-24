"""
BaseStrategy - Abstract base class for trading strategies.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Literal


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    All strategy implementations must inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, name: str = "BaseStrategy"):
        """
        Initialize base strategy.
        
        Args:
            name: Strategy name
        """
        self.name = name
        self.positions = []  # List of open positions
        self.closed_positions = []  # List of closed positions
        self.current_bar = 0
        
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Generate trading signals from data.
        
        Args:
            df: DataFrame with OHLCV data and features
            **kwargs: Additional parameters
            
        Returns:
            Series with signals:
                1 = Long entry
                -1 = Short entry
                0 = No signal / Exit
        """
        pass
    
    @abstractmethod
    def should_enter_long(self, signal: int, bar_idx: int, **kwargs) -> bool:
        """
        Determine if should enter long position.
        
        Args:
            signal: Current signal value
            bar_idx: Current bar index
            **kwargs: Additional parameters
            
        Returns:
            True if should enter long, False otherwise
        """
        pass
    
    @abstractmethod
    def should_enter_short(self, signal: int, bar_idx: int, **kwargs) -> bool:
        """
        Determine if should enter short position.
        
        Args:
            signal: Current signal value
            bar_idx: Current bar index
            **kwargs: Additional parameters
            
        Returns:
            True if should enter short, False otherwise
        """
        pass
    
    @abstractmethod
    def should_exit(self, position: Dict[str, Any], current_bar: int, 
                   current_price: float, **kwargs) -> tuple[bool, str]:
        """
        Determine if should exit position.
        
        Args:
            position: Position dictionary with entry details
            current_bar: Current bar index
            current_price: Current price
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (should_exit: bool, reason: str)
        """
        pass
    
    def open_position(self, 
                     position_type: Literal['long', 'short'],
                     entry_bar: int,
                     entry_price: float,
                     size: float,
                     **kwargs) -> Dict[str, Any]:
        """
        Open a new position.
        
        Args:
            position_type: 'long' or 'short'
            entry_bar: Bar index of entry
            entry_price: Entry price
            size: Position size
            **kwargs: Additional position parameters
            
        Returns:
            Position dictionary
        """
        position = {
            'type': position_type,
            'entry_bar': entry_bar,
            'entry_price': entry_price,
            'size': size,
            'highest_price': entry_price if position_type == 'long' else None,
            'lowest_price': entry_price if position_type == 'short' else None,
            **kwargs
        }
        self.positions.append(position)
        return position
    
    def close_position(self,
                      position: Dict[str, Any],
                      exit_bar: int,
                      exit_price: float,
                      reason: str = 'signal') -> Dict[str, Any]:
        """
        Close an existing position.
        
        Args:
            position: Position to close
            exit_bar: Bar index of exit
            exit_price: Exit price
            reason: Exit reason
            
        Returns:
            Closed position dictionary
        """
        position['exit_bar'] = exit_bar
        position['exit_price'] = exit_price
        position['exit_reason'] = reason
        position['bars_held'] = exit_bar - position['entry_bar']
        
        # Calculate P&L
        if position['type'] == 'long':
            position['pnl'] = (exit_price - position['entry_price']) * position['size']
            position['pnl_pct'] = (exit_price - position['entry_price']) / position['entry_price']
        else:  # short
            position['pnl'] = (position['entry_price'] - exit_price) * position['size']
            position['pnl_pct'] = (position['entry_price'] - exit_price) / position['entry_price']
        
        self.closed_positions.append(position)
        self.positions.remove(position)
        
        return position
    
    def update_position_tracking(self, position: Dict[str, Any], current_price: float):
        """
        Update position tracking for trailing stops.
        
        Args:
            position: Position to update
            current_price: Current price
        """
        if position['type'] == 'long':
            if position['highest_price'] is None or current_price > position['highest_price']:
                position['highest_price'] = current_price
        else:  # short
            if position['lowest_price'] is None or current_price < position['lowest_price']:
                position['lowest_price'] = current_price
    
    def has_open_position(self) -> bool:
        """Check if there are any open positions."""
        return len(self.positions) > 0
    
    def get_open_positions(self) -> list:
        """Get list of open positions."""
        return self.positions.copy()
    
    def get_closed_positions(self) -> list:
        """Get list of closed positions."""
        return self.closed_positions.copy()
    
    def reset(self):
        """Reset strategy state."""
        self.positions = []
        self.closed_positions = []
        self.current_bar = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Calculate strategy statistics.
        
        Returns:
            Dictionary with statistics
        """
        if not self.closed_positions:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'avg_pnl_pct': 0.0,
                'total_pnl': 0.0
            }
        
        winning_trades = [p for p in self.closed_positions if p['pnl'] > 0]
        losing_trades = [p for p in self.closed_positions if p['pnl'] <= 0]
        
        return {
            'total_trades': len(self.closed_positions),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.closed_positions) if self.closed_positions else 0.0,
            'avg_pnl': np.mean([p['pnl'] for p in self.closed_positions]),
            'avg_pnl_pct': np.mean([p['pnl_pct'] for p in self.closed_positions]),
            'total_pnl': sum([p['pnl'] for p in self.closed_positions]),
            'avg_bars_held': np.mean([p['bars_held'] for p in self.closed_positions])
        }
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
