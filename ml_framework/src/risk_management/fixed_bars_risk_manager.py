"""
Fixed Bars Count Risk Manager

This module provides a risk manager that exits positions after a fixed number of bars.
No stop loss or take profit - positions are held for exactly N bars.
"""

from typing import Dict, Any, Tuple
import pandas as pd
from .base_risk_manager import BaseRiskManager


class FixedBarsCountRiskManager(BaseRiskManager):
    """
    Risk manager that exits positions after a fixed number of bars.
    
    This strategy:
    - Holds positions for exactly N bars
    - No stop loss
    - No take profit
    - Exits only based on bar count or end of data
    
    Useful for:
    - Testing predictive models with fixed horizons
    - Comparing performance across different holding periods
    - Eliminating stop loss/take profit optimization complexity
    """
    
    def __init__(self, bars_to_hold: int = 5, name: str = "FixedBarsCount"):
        """
        Initialize the fixed bars count risk manager.
        
        Args:
            bars_to_hold: Number of bars to hold position before exiting
            name: Name of the risk manager
        """
        super().__init__(name=name)
        
        if bars_to_hold < 1:
            raise ValueError("bars_to_hold must be at least 1")
        
        self.bars_to_hold = bars_to_hold
    
    def should_exit(
        self,
        position: Dict[str, Any],
        current_bar: int,
        current_price: float,
        df: pd.DataFrame
    ) -> Tuple[bool, str]:
        """
        Determine if position should be exited based on bar count.
        
        Args:
            position: Dictionary containing position information:
                - entry_bar: Bar index when position was entered
                - entry_price: Price at entry
                - shares: Number of shares
                - entry_idx: Index in DataFrame
            current_bar: Current bar index
            current_price: Current price (not used in this strategy)
            df: DataFrame with market data
            
        Returns:
            Tuple of (should_exit: bool, exit_reason: str)
            - should_exit: True if N bars have passed or end of data
            - exit_reason: 'fixed_bars' or 'end_of_data'
        """
        entry_bar = position.get('entry_bar', 0)
        
        # Calculate bars held
        bars_held = current_bar - entry_bar
        
        # Check if we've reached the target bar count
        if bars_held >= self.bars_to_hold:
            return True, 'fixed_bars'
        
        # Check if we're at the end of data
        if current_bar >= len(df) - 1:
            return True, 'end_of_data'
        
        # Continue holding
        return False, ''
    
    def get_bars_held(self, position: Dict[str, Any], current_bar: int) -> int:
        """
        Get the number of bars a position has been held.
        
        Args:
            position: Position dictionary
            current_bar: Current bar index
            
        Returns:
            Number of bars held
        """
        entry_bar = position.get('entry_bar', 0)
        return current_bar - entry_bar
    
    def get_bars_remaining(self, position: Dict[str, Any], current_bar: int) -> int:
        """
        Get the number of bars remaining before exit.
        
        Args:
            position: Position dictionary
            current_bar: Current bar index
            
        Returns:
            Number of bars remaining (0 if should exit)
        """
        bars_held = self.get_bars_held(position, current_bar)
        bars_remaining = max(0, self.bars_to_hold - bars_held)
        return bars_remaining
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the risk manager.
        
        Returns:
            Dictionary with risk manager information
        """
        info = super().get_info()
        info.update({
            'bars_to_hold': self.bars_to_hold,
            'strategy': 'Fixed bars count exit',
            'stop_loss': 'None',
            'take_profit': 'None'
        })
        return info
    
    def __repr__(self) -> str:
        """String representation of the risk manager."""
        return f"{self.__class__.__name__}(bars_to_hold={self.bars_to_hold}, name='{self.name}')"


# Example usage
if __name__ == "__main__":
    import numpy as np
    
    # Create sample data
    df = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100,
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98
    })
    
    # Create risk manager
    risk_manager = FixedBarsCountRiskManager(bars_to_hold=5)
    
    print("="*70)
    print("FIXED BARS COUNT RISK MANAGER EXAMPLE")
    print("="*70)
    print(f"\nRisk Manager: {risk_manager}")
    print(f"Info: {risk_manager.get_info()}")
    
    # Simulate a position
    print("\n" + "="*70)
    print("SIMULATING POSITION")
    print("="*70)
    
    entry_bar = 10
    entry_price = df.iloc[entry_bar]['close']
    shares = 1.0
    
    position = {
        'entry_bar': entry_bar,
        'entry_price': entry_price,
        'shares': shares,
        'entry_idx': entry_bar
    }
    
    risk_manager.on_entry('pos_1', entry_bar, entry_price, shares, entry_bar)
    
    print(f"\nPosition entered at bar {entry_bar}, price ${entry_price:.2f}")
    print(f"Holding for {risk_manager.bars_to_hold} bars")
    
    # Check exit conditions over time
    print("\n" + "-"*70)
    print("Bar | Bars Held | Bars Remaining | Should Exit | Exit Reason")
    print("-"*70)
    
    for current_bar in range(entry_bar, min(entry_bar + 10, len(df))):
        current_price = df.iloc[current_bar]['close']
        should_exit, exit_reason = risk_manager.should_exit(
            position, current_bar, current_price, df
        )
        
        bars_held = risk_manager.get_bars_held(position, current_bar)
        bars_remaining = risk_manager.get_bars_remaining(position, current_bar)
        
        print(f"{current_bar:3d} | {bars_held:9d} | {bars_remaining:14d} | "
              f"{'Yes' if should_exit else 'No':11s} | {exit_reason}")
        
        if should_exit:
            pnl = risk_manager.calculate_pnl(entry_price, current_price, shares)
            print(f"\n✓ Position exited at bar {current_bar}, price ${current_price:.2f}")
            print(f"  PnL: ${pnl:.2f}")
            print(f"  Return: {(current_price/entry_price - 1)*100:.2f}%")
            risk_manager.on_exit('pos_1')
            break
    
    print("\n" + "="*70)
    print("EXAMPLE COMPLETE")
    print("="*70)
