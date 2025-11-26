"""
Base Risk Manager - Abstract base class for position exit and sizing logic.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import pandas as pd


class BaseRiskManager(ABC):
    """
    Abstract base for risk management. Subclasses must implement should_exit().
    
    Handles: position sizing, exit conditions, P&L calculation.
    """
    
    def __init__(self, name: str = "BaseRiskManager"):
        self.name = name
        self.active_positions = {}
    
    @abstractmethod
    def should_exit(self, position: Dict[str, Any], current_bar: int,
                    current_price: float, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check if position should exit. Returns (should_exit, reason)."""
        pass
    
    def on_entry(self, position_id: str, entry_bar: int, entry_price: float,
                 shares: float, entry_idx: int) -> None:
        """Record position entry."""
        self.active_positions[position_id] = {
            'entry_bar': entry_bar, 'entry_price': entry_price,
            'shares': shares, 'entry_idx': entry_idx
        }
    
    def on_exit(self, position_id: str) -> None:
        """Record position exit."""
        self.active_positions.pop(position_id, None)
    
    def get_position_size(self, capital: float, price: float,
                          position_size_pct: float = 1.0) -> float:
        """Calculate shares to trade based on capital and position size %."""
        return (capital * position_size_pct / price) if price > 0 else 0.0
    
    def calculate_pnl(self, entry_price: float, exit_price: float, shares: float) -> float:
        """Calculate P&L. Positive shares = long, negative = short."""
        return (exit_price - entry_price) * shares if shares > 0 else (entry_price - exit_price) * abs(shares)
    
    def reset(self) -> None:
        """Reset state."""
        self.active_positions = {}
    
    def get_info(self) -> Dict[str, Any]:
        """Get risk manager info."""
        return {'name': self.name, 'type': self.__class__.__name__,
                'active_positions': len(self.active_positions)}
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
