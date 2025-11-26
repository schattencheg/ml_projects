"""
Models library - ML model wrappers with automatic target conversion.
"""

from src.models_lib.base_model import BaseModel
from src.models_lib.xgboost_model import XGBoostModel
from src.models_lib.catboost_model import CatBoostModel
from src.models_lib.linear_model import LinearRegressionModel, LogisticRegressionModel, RandomForestModel
from src.models_lib.cnn_models import BaseCNN, SimpleCNN, DeepCNN, ResidualCNN

__all__ = [
    'BaseModel',
    'XGBoostModel',
    'CatBoostModel',
    'LinearRegressionModel',
    'LogisticRegressionModel',
    'RandomForestModel',
    'BaseCNN',
    'SimpleCNN',
    'DeepCNN',
    'ResidualCNN'
]
