"""
ML Framework - A comprehensive ML framework for financial data analysis.
"""

# Legacy components (for backward compatibility)
from .data_provider import DataProvider
from .features_generator import FeaturesGenerator

# New manager-based architecture
from .managers import (
    ModelManager,
    TrainManager,
    ScalerManager,
    MLFlowManager,
    BacktestManager,
    ResultManager,
    VisualizationManager,
    PipelineManager
)

# Model library
from .models_lib import (
    BaseModel,
    XGBoostModel,
    CatBoostModel,
    LinearRegressionModel,
    LogisticRegressionModel,
    RandomForestModel,
    SimpleCNN,
    DeepCNN,
    ResidualCNN
)

__version__ = '0.2.0'

__all__ = [
    # Data components
    'DataProvider',
    'FeaturesGenerator',
    
    # Managers
    'ModelManager',
    'TrainManager',
    'ScalerManager',
    'MLFlowManager',
    'BacktestManager',
    'ResultManager',
    'VisualizationManager',
    'PipelineManager',
    'FeatureSelector',
    'HyperparameterOptimizer',
    
    # Models
    'BaseModel',
    'XGBoostModel',
    'CatBoostModel',
    'LinearRegressionModel',
    'LogisticRegressionModel',
    'RandomForestModel',
    'SimpleCNN',
    'DeepCNN',
    'ResidualCNN',
]
