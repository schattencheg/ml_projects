"""
ML Framework - A simple and functional machine learning framework for financial data analysis.
"""

__version__ = "0.1.0"

from .data_provider import DataProvider
from .features_generator import FeaturesGenerator
from .model_manager import ModelManager
from .ml_trainer import ML_Trainer
from .ml_tester import ML_Tester
from .backtester import Backtester

__all__ = [
    'DataProvider',
    'FeaturesGenerator',
    'ModelManager',
    'ML_Trainer',
    'ML_Tester',
    'Backtester',
]
