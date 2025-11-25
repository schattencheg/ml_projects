"""
Base Risk Manager for Backtesting Strategies

This module provides the abstract base class for risk management in trading strategies.
Risk managers control position sizing, stop losses, take profits, and exit conditions.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import pandas as pd


class BaseRiskManager(ABC):
    """
    Abstract base class for risk management strategies.
    
    Risk managers determine:
    - When to exit positions
    - Position sizing
    - Stop loss levels
    - Take profit levels
    - Risk per trade
    
    Subclasses must implement the should_exit() method.
    """
    
    def __init__(self, name: str = "BaseRiskManager"):
        """
        Initialize the risk manager.
        
        Args:
            name: Name of the risk manager
        """
        self.name = name
        self.active_positions = {}  # Track active positions
        
    @abstractmethod
    def should_exit(
        self,
        position: Dict[str, Any],
        current_bar: int,
        current_price: float,
        df: pd.DataFrame
    ) -> Tuple[bool, str]:
        """
        Determine if a position should be exited.
        
        Args:
            position: Dictionary containing position information:
                - entry_bar: Bar index when position was entered
                - entry_price: Price at entry
                - shares: Number of shares (positive for long, negative for short)
                - entry_idx: Index in DataFrame
            current_bar: Current bar index
            current_price: Current price
            df: DataFrame with market data
            
        Returns:
            Tuple of (should_exit: bool, exit_reason: str)
            - should_exit: True if position should be closed
            - exit_reason: Reason for exit ('signal', 'stop_loss', 'take_profit', 
                          'fixed_bars', 'trailing_stop', 'end_of_data', etc.)
        """
        pass
    
    def on_entry(
        self,
        position_id: str,
        entry_bar: int,
        entry_price: float,
        shares: float,
        entry_idx: int
    ) -> None:
        """
        Called when a new position is entered.
        
        Args:
            position_id: Unique identifier for the position
            entry_bar: Bar index when position was entered
            entry_price: Price at entry
            shares: Number of shares (positive for long, negative for short)
            entry_idx: Index in DataFrame
        """
        self.active_positions[position_id] = {
            'entry_bar': entry_bar,
            'entry_price': entry_price,
            'shares': shares,
            'entry_idx': entry_idx
        }
    
    def on_exit(self, position_id: str) -> None:
        """
        Called when a position is exited.
        
        Args:
            position_id: Unique identifier for the position
        """
        if position_id in self.active_positions:
            del self.active_positions[position_id]
    
    def get_position_size(
        self,
        capital: float,
        price: float,
        position_size_pct: float = 1.0
    ) -> float:
        """
        Calculate position size based on available capital.
        
        Args:
            capital: Available capital
            price: Current price
            position_size_pct: Percentage of capital to use (0.0 to 1.0)
            
        Returns:
            Number of shares to trade
        """
        if price <= 0:
            return 0.0
        
        position_value = capital * position_size_pct
        shares = position_value / price
        return shares
    
    def calculate_pnl(
        self,
        entry_price: float,
        exit_price: float,
        shares: float
    ) -> float:
        """
        Calculate profit/loss for a position.
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            shares: Number of shares (positive for long, negative for short)
            
        Returns:
            Profit/loss in currency units
        """
        if shares > 0:
            # Long position
            pnl = (exit_price - entry_price) * shares
        else:
            # Short position
            pnl = (entry_price - exit_price) * abs(shares)
        
        return pnl
    
    def reset(self) -> None:
        """Reset the risk manager state."""
        self.active_positions = {}
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the risk manager.
        
        Returns:
            Dictionary with risk manager information
        """
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'active_positions': len(self.active_positions)
        }
    
    def __repr__(self) -> str:
        """String representation of the risk manager."""
        return f"{self.__class__.__name__}(name='{self.name}')"
