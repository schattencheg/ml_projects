"""
Fixed Bars Count Risk Manager - Exit after N bars, no stop loss/take profit.
"""

from typing import Dict, Any, Tuple, Optional
import pandas as pd
from .base_risk_manager import BaseRiskManager


class FixedBarsCountRiskManager(BaseRiskManager):
    """
    Exits positions after exactly N bars. No stop loss or take profit.
    
    Features:
    - Position sizing: % of current capital (default 2%)
    - Fixed holding period exit
    - Cooldown: new positions only after previous closes
    """
    
    def __init__(self, bars_to_hold: int = 5, position_size_pct: float = 0.02,
                 name: str = "FixedBarsCount"):
        if bars_to_hold < 1:
            raise ValueError("bars_to_hold must be >= 1")
        if not 0 < position_size_pct <= 1.0:
            raise ValueError("position_size_pct must be in (0, 1]")
        
        super().__init__(name)
        self.bars_to_hold = bars_to_hold
        self.position_size_pct = position_size_pct
        self.last_exit_bar: Optional[int] = None
    
    def should_exit(self, position: Dict[str, Any], current_bar: int,
                    current_price: float, df: pd.DataFrame) -> Tuple[bool, str]:
        """Exit after N bars or at end of data."""
        bars_held = current_bar - position.get('entry_bar', 0)
        
        if bars_held >= self.bars_to_hold:
            return True, 'fixed_bars'
        if current_bar >= len(df) - 1:
            return True, 'end_of_data'
        return False, ''
    
    def can_open_position(self, current_bar: int) -> bool:
        """Check if new position allowed (cooldown: bar after last exit)."""
        return self.last_exit_bar is None or current_bar > self.last_exit_bar
    
    def get_position_size(self, capital: float, price: float,
                          commission: float = 0.0) -> float:
        """Calculate shares: (capital * pct - commission) / price."""
        if price <= 0 or capital <= 0:
            return 0.0
        position_value = capital * self.position_size_pct
        return max(0.0, (position_value - position_value * commission) / price)
    
    def on_exit(self, position_id: str, exit_bar: int) -> None:
        """Record exit and update cooldown."""
        super().on_exit(position_id)
        self.last_exit_bar = exit_bar
    
    def reset(self) -> None:
        super().reset()
        self.last_exit_bar = None
    
    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update({
            'bars_to_hold': self.bars_to_hold,
            'position_size_pct': self.position_size_pct,
            'strategy': 'Fixed bars exit'
        })
        return info
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bars={self.bars_to_hold}, size={self.position_size_pct})"
