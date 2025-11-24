"""
MLStrategy - ML-based trading strategy with long/short positions and trailing stop loss.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from src.strategies.base_strategy import BaseStrategy


class MLStrategy(BaseStrategy):
    """
    ML-based trading strategy.
    
    Strategy Logic:
    - Signal +1: Enter Long position
    - Signal -1: Enter Short position
    - Exit on Nth bar (holding_period) OR trailing stop loss
    
    Features:
    - Long and short positions
    - Fixed holding period exit
    - Optional trailing stop loss
    - Position tracking
    """
    
    def __init__(self,
                 name: str = "MLStrategy",
                 holding_period: int = 15,
                 trailing_stop_pct: Optional[float] = None,
                 enable_trailing_stop: bool = False):
        """
        Initialize ML strategy.
        
        Args:
            name: Strategy name
            holding_period: Number of bars to hold position (same as FUTURE_BARS)
            trailing_stop_pct: Trailing stop loss percentage (e.g., 0.05 = 5%)
            enable_trailing_stop: Enable/disable trailing stop loss
        """
        super().__init__(name)
        self.holding_period = holding_period
        self.trailing_stop_pct = trailing_stop_pct
        self.enable_trailing_stop = enable_trailing_stop
        
    def generate_signals(self, df: pd.DataFrame, 
                        model: Any = None,
                        scaler: Any = None,
                        feature_cols: list = None,
                        **kwargs) -> pd.Series:
        """
        Generate trading signals from ML model predictions.
        
        Supports three-class predictions:
        -1 (decrease) → -1 (short signal)
         0 (neutral)  →  0 (no signal)
        +1 (increase) → +1 (long signal)
        
        Args:
            df: DataFrame with OHLCV data and features
            model: Trained ML model
            scaler: Fitted scaler
            feature_cols: List of feature column names
            **kwargs: Additional parameters
            
        Returns:
            Series with signals:
                +1 = Long entry
                -1 = Short entry
                 0 = No signal / Neutral
        """
        if model is None or scaler is None or feature_cols is None:
            raise ValueError("model, scaler, and feature_cols are required")
        
        # Import target transformer
        from ..target_transformer import inverse_transform_predictions, get_target_transformer
        
        # Prepare features
        X = df[feature_cols].values
        X_scaled = scaler.transform(X)
        
        # Get predictions
        predictions = model.predict(X_scaled)
        
        # Detect model type and transform predictions back to three-class format
        transformer = get_target_transformer()
        model_type = transformer.get_model_type(model)
        
        # Transform predictions to three-class format (-1, 0, +1)
        predictions_three_class = inverse_transform_predictions(predictions, model_type)
        
        # Create signals (predictions are already in correct format)
        signals = pd.Series(predictions_three_class, index=df.index, dtype=int)
        
        return signals
    
    def should_enter_long(self, signal: int, bar_idx: int, **kwargs) -> bool:
        """
        Determine if should enter long position.
        
        Args:
            signal: Current signal value
            bar_idx: Current bar index
            **kwargs: Additional parameters
            
        Returns:
            True if should enter long (signal == 1 and no open position)
        """
        return signal == 1 and not self.has_open_position()
    
    def should_enter_short(self, signal: int, bar_idx: int, **kwargs) -> bool:
        """
        Determine if should enter short position.
        
        Args:
            signal: Current signal value
            bar_idx: Current bar index
            **kwargs: Additional parameters
            
        Returns:
            True if should enter short (signal == -1 and no open position)
        """
        return signal == -1 and not self.has_open_position()
    
    def should_exit(self, position: Dict[str, Any], current_bar: int,
                   current_price: float, **kwargs) -> tuple[bool, str]:
        """
        Determine if should exit position.
        
        Exit Conditions:
        1. Holding period reached (Nth bar)
        2. Trailing stop loss hit (if enabled)
        
        Args:
            position: Position dictionary
            current_bar: Current bar index
            current_price: Current price
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (should_exit: bool, reason: str)
        """
        bars_held = current_bar - position['entry_bar']
        
        # Exit condition 1: Holding period reached
        if bars_held >= self.holding_period:
            return True, 'holding_period'
        
        # Exit condition 2: Trailing stop loss (if enabled)
        if self.enable_trailing_stop and self.trailing_stop_pct is not None:
            if position['type'] == 'long':
                # Long position: exit if price drops from highest by trailing_stop_pct
                if position['highest_price'] is not None:
                    stop_price = position['highest_price'] * (1 - self.trailing_stop_pct)
                    if current_price <= stop_price:
                        return True, 'trailing_stop_long'
            
            else:  # short position
                # Short position: exit if price rises from lowest by trailing_stop_pct
                if position['lowest_price'] is not None:
                    stop_price = position['lowest_price'] * (1 + self.trailing_stop_pct)
                    if current_price >= stop_price:
                        return True, 'trailing_stop_short'
        
        return False, ''
    
    def backtest(self,
                df: pd.DataFrame,
                model: Any,
                scaler: Any,
                feature_cols: list,
                initial_capital: float = 10000.0,
                position_size_pct: float = 1.0,
                commission: float = 0.001,
                price_col: str = 'close') -> Dict[str, Any]:
        """
        Run backtest with this strategy.
        
        Args:
            df: DataFrame with OHLCV data and features
            model: Trained ML model
            scaler: Fitted scaler
            feature_cols: Feature column names
            initial_capital: Starting capital
            position_size_pct: Position size as percentage of capital
            commission: Commission rate
            price_col: Price column name
            
        Returns:
            Dictionary with backtest results
        """
        # Reset strategy state
        self.reset()
        
        # Generate signals
        signals = self.generate_signals(df, model, scaler, feature_cols)
        
        # Initialize tracking
        capital = initial_capital
        equity_curve = []
        
        # Simulate trading
        for i in range(len(df)):
            current_price = df[price_col].iloc[i]
            signal = signals.iloc[i]
            
            # Update position tracking for trailing stops
            for position in self.positions:
                self.update_position_tracking(position, current_price)
            
            # Check exit conditions for open positions
            positions_to_close = []
            for position in self.positions:
                should_exit, reason = self.should_exit(position, i, current_price)
                if should_exit:
                    positions_to_close.append((position, reason))
            
            # Close positions
            for position, reason in positions_to_close:
                closed_pos = self.close_position(position, i, current_price, reason)
                
                # Update capital
                if closed_pos['type'] == 'long':
                    sell_value = closed_pos['size'] * current_price
                    commission_cost = sell_value * commission
                    capital += sell_value - commission_cost
                else:  # short
                    # For short: we borrowed and sold at entry, now buy back at exit
                    buy_value = closed_pos['size'] * current_price
                    commission_cost = buy_value * commission
                    # Profit = (entry_price - exit_price) * size
                    capital += closed_pos['pnl'] - commission_cost
            
            # Check entry conditions (only if no open position)
            if not self.has_open_position():
                if self.should_enter_long(signal, i):
                    # Enter long
                    position_value = capital * position_size_pct
                    commission_cost = position_value * commission
                    shares = (position_value - commission_cost) / current_price
                    
                    if shares > 0:
                        self.open_position('long', i, current_price, shares)
                        capital -= position_value
                
                elif self.should_enter_short(signal, i):
                    # Enter short
                    position_value = capital * position_size_pct
                    commission_cost = position_value * commission
                    shares = (position_value - commission_cost) / current_price
                    
                    if shares > 0:
                        self.open_position('short', i, current_price, shares)
                        # For short: we receive cash from selling borrowed shares
                        capital += position_value - commission_cost
            
            # Calculate equity
            equity = capital
            for position in self.positions:
                if position['type'] == 'long':
                    equity += position['size'] * current_price
                else:  # short
                    # For short: equity includes unrealized P&L
                    unrealized_pnl = (position['entry_price'] - current_price) * position['size']
                    equity += unrealized_pnl
            
            equity_curve.append(equity)
        
        # Close any remaining open positions at the end
        final_price = df[price_col].iloc[-1]
        for position in self.positions.copy():
            closed_pos = self.close_position(position, len(df) - 1, final_price, 'end_of_data')
            
            if closed_pos['type'] == 'long':
                sell_value = closed_pos['size'] * final_price
                commission_cost = sell_value * commission
                capital += sell_value - commission_cost
            else:  # short
                buy_value = closed_pos['size'] * final_price
                commission_cost = buy_value * commission
                capital += closed_pos['pnl'] - commission_cost
        
        # Create results
        equity_series = pd.Series(equity_curve, index=df.index)
        
        results = {
            'equity_curve': equity_series,
            'trades': self.get_closed_positions(),
            'final_capital': capital,
            'initial_capital': initial_capital,
            'strategy_stats': self.get_statistics()
        }
        
        return results
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get strategy configuration.
        
        Returns:
            Dictionary with configuration
        """
        return {
            'name': self.name,
            'holding_period': self.holding_period,
            'trailing_stop_pct': self.trailing_stop_pct,
            'enable_trailing_stop': self.enable_trailing_stop
        }
    
    def __repr__(self):
        return (f"MLStrategy(name='{self.name}', "
                f"holding_period={self.holding_period}, "
                f"trailing_stop={'enabled' if self.enable_trailing_stop else 'disabled'})")
