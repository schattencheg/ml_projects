"""
Models package initialization
"""

from .linear_model import SklearnLinearModel
from .random_forest_model import RandomForestModel

__all__ = ['SklearnLinearModel', 'RandomForestModel']