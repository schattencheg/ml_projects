"""
ML Framework - A modular ML framework for financial data analysis and backtesting.
"""

__version__ = '0.3.0'

# Core data components
from src.data_provider import DataProvider
from src.features_generator import FeaturesGenerator
from src.target_transformer import TargetTransformer, get_target_transformer, transform_for_model, inverse_transform_predictions

# Managers
from src.managers import (
    ModelManager, TrainManager, ScalerManager, MLFlowManager,
    BacktestManager, ResultManager, VisualizationManager, PipelineManager,
    FeatureSelector, HyperparameterOptimizer, RunManager
)

# Models
from src.models_lib import (
    BaseModel, XGBoostModel, CatBoostModel,
    LinearRegressionModel, LogisticRegressionModel, RandomForestModel,
    BaseCNN, SimpleCNN, DeepCNN, ResidualCNN
)

# Backtesting
from src.backtesting import BaseBacktest, BacktestNoLib, BacktestBacktrader, BacktestBacktestingPy

# Strategies
from src.strategies import BaseStrategy, MLStrategy

# Risk Management
from src.risk_management import BaseRiskManager, FixedBarsCountRiskManager

__all__ = [
    # Data
    'DataProvider', 'FeaturesGenerator', 'TargetTransformer',
    'get_target_transformer', 'transform_for_model', 'inverse_transform_predictions',
    
    # Managers
    'ModelManager', 'TrainManager', 'ScalerManager', 'MLFlowManager',
    'BacktestManager', 'ResultManager', 'VisualizationManager', 'PipelineManager',
    'FeatureSelector', 'HyperparameterOptimizer', 'RunManager',
    
    # Models
    'BaseModel', 'XGBoostModel', 'CatBoostModel',
    'LinearRegressionModel', 'LogisticRegressionModel', 'RandomForestModel',
    'BaseCNN', 'SimpleCNN', 'DeepCNN', 'ResidualCNN',
    
    # Backtesting
    'BaseBacktest', 'BacktestNoLib', 'BacktestBacktrader', 'BacktestBacktestingPy',
    
    # Strategies
    'BaseStrategy', 'MLStrategy',
    
    # Risk Management
    'BaseRiskManager', 'FixedBarsCountRiskManager',
]
