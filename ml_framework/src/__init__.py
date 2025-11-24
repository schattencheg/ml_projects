"""
ML Framework - A comprehensive ML framework for financial data analysis.
"""

# Legacy components (for backward compatibility)
from src.data_provider import DataProvider
from src.features_generator import FeaturesGenerator

# New manager-based architecture
from src.managers import (
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
from src.models_lib import (
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

# Backtesting module
from src.backtesting import (
    BaseBacktest,
    BacktestNoLib,
    BacktestBacktrader,
    BacktestBacktestingPy
)

# Strategies module
from src.strategies import (
    BaseStrategy,
    MLStrategy
)

# Target transformer
from src.target_transformer import (
    TargetTransformer,
    get_target_transformer,
    transform_for_model,
    inverse_transform_predictions
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
    
    # Backtesting
    'BaseBacktest',
    'BacktestNoLib',
    'BacktestBacktrader',
    'BacktestBacktestingPy',
    
    # Strategies
    'BaseStrategy',
    'MLStrategy',
    
    # Target Transformer
    'TargetTransformer',
    'get_target_transformer',
    'transform_for_model',
    'inverse_transform_predictions',
]
