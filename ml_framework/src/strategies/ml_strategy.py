"""
MLStrategy - ML-based trading strategy with holding period and optional trailing stop.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from src.strategies.base_strategy import BaseStrategy


class MLStrategy(BaseStrategy):
    """
    ML-based strategy: +1=long, -1=short, exit after N bars or trailing stop.
    """
    
    def __init__(self, name: str = "MLStrategy", holding_period: int = 15,
                 trailing_stop_pct: Optional[float] = None):
        super().__init__(name)
        self.holding_period = holding_period
        self.trailing_stop_pct = trailing_stop_pct
    
    def generate_signals(self, df: pd.DataFrame, model=None, scaler=None,
                         feature_cols: list = None, **kwargs) -> pd.Series:
        """Generate signals from ML model predictions."""
        if model is None or scaler is None or feature_cols is None:
            raise ValueError("model, scaler, and feature_cols required")
        
        from ..target_transformer import inverse_transform_predictions, get_target_transformer
        
        X_scaled = scaler.transform(df[feature_cols].values)
        predictions = model.predict(X_scaled)
        
        transformer = get_target_transformer()
        model_type = transformer.get_model_type(model)
        predictions = inverse_transform_predictions(predictions, model_type)
        
        return pd.Series(predictions, index=df.index, dtype=int)
    
    def should_enter_long(self, signal: int, bar_idx: int, **kwargs) -> bool:
        return signal == 1 and not self.has_open_position()
    
    def should_enter_short(self, signal: int, bar_idx: int, **kwargs) -> bool:
        return signal == -1 and not self.has_open_position()
    
    def should_exit(self, position: Dict, current_bar: int,
                    current_price: float, **kwargs) -> Tuple[bool, str]:
        """Exit after holding_period bars or trailing stop hit."""
        bars_held = current_bar - position['entry_bar']
        
        # Holding period exit
        if bars_held >= self.holding_period:
            return True, 'holding_period'
        
        # Trailing stop (if enabled)
        if self.trailing_stop_pct:
            if position['type'] == 'long' and position['highest_price']:
                if current_price <= position['highest_price'] * (1 - self.trailing_stop_pct):
                    return True, 'trailing_stop'
            elif position['type'] == 'short' and position['lowest_price']:
                if current_price >= position['lowest_price'] * (1 + self.trailing_stop_pct):
                    return True, 'trailing_stop'
        
        return False, ''
    
    def __repr__(self):
        return f"MLStrategy(holding={self.holding_period}, trailing_stop={self.trailing_stop_pct})"
