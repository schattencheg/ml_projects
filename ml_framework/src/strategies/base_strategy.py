"""
BaseStrategy - Abstract base class for trading strategies.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Literal, Tuple, List


class BaseStrategy(ABC):
    """
    Abstract base for trading strategies.
    
    Subclasses must implement: generate_signals(), should_enter_long(), 
    should_enter_short(), should_exit()
    """
    
    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self.positions: List[Dict] = []
        self.closed_positions: List[Dict] = []
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate signals: 1=long, -1=short, 0=neutral."""
        pass
    
    @abstractmethod
    def should_enter_long(self, signal: int, bar_idx: int, **kwargs) -> bool:
        pass
    
    @abstractmethod
    def should_enter_short(self, signal: int, bar_idx: int, **kwargs) -> bool:
        pass
    
    @abstractmethod
    def should_exit(self, position: Dict, current_bar: int, 
                    current_price: float, **kwargs) -> Tuple[bool, str]:
        """Returns (should_exit, reason)."""
        pass
    
    def open_position(self, position_type: Literal['long', 'short'],
                      entry_bar: int, entry_price: float, size: float, **kwargs) -> Dict:
        """Open a new position."""
        position = {
            'type': position_type, 'entry_bar': entry_bar,
            'entry_price': entry_price, 'size': size,
            'highest_price': entry_price if position_type == 'long' else None,
            'lowest_price': entry_price if position_type == 'short' else None,
            **kwargs
        }
        self.positions.append(position)
        return position
    
    def close_position(self, position: Dict, exit_bar: int, 
                       exit_price: float, reason: str = 'signal') -> Dict:
        """Close position and calculate P&L."""
        position.update({
            'exit_bar': exit_bar, 'exit_price': exit_price, 'exit_reason': reason,
            'bars_held': exit_bar - position['entry_bar']
        })
        
        # P&L calculation
        price_diff = exit_price - position['entry_price']
        if position['type'] == 'short':
            price_diff = -price_diff
        position['pnl'] = price_diff * position['size']
        position['pnl_pct'] = price_diff / position['entry_price']
        
        self.closed_positions.append(position)
        self.positions.remove(position)
        return position
    
    def update_position_tracking(self, position: Dict, current_price: float):
        """Update high/low tracking for trailing stops."""
        if position['type'] == 'long':
            if position['highest_price'] is None or current_price > position['highest_price']:
                position['highest_price'] = current_price
        else:
            if position['lowest_price'] is None or current_price < position['lowest_price']:
                position['lowest_price'] = current_price
    
    def has_open_position(self) -> bool:
        return len(self.positions) > 0
    
    def reset(self):
        self.positions = []
        self.closed_positions = []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate trading statistics."""
        if not self.closed_positions:
            return {'total_trades': 0, 'win_rate': 0.0, 'total_pnl': 0.0}
        
        wins = sum(1 for p in self.closed_positions if p['pnl'] > 0)
        pnls = [p['pnl'] for p in self.closed_positions]
        
        return {
            'total_trades': len(self.closed_positions),
            'winning_trades': wins,
            'win_rate': wins / len(self.closed_positions),
            'avg_pnl': np.mean(pnls),
            'total_pnl': sum(pnls)
        }
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
