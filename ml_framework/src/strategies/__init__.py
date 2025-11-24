"""
Strategies module - Trading strategies for backtesting.
"""

from src.strategies.base_strategy import BaseStrategy
from src.strategies.ml_strategy import MLStrategy

__all__ = [
    'BaseStrategy',
    'MLStrategy'
]
