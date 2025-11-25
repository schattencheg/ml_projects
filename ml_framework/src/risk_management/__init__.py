"""
Risk Management Module

This module provides risk management strategies for backtesting.
"""

from .base_risk_manager import BaseRiskManager
from .fixed_bars_risk_manager import FixedBarsCountRiskManager

__all__ = [
    'BaseRiskManager',
    'FixedBarsCountRiskManager'
]
