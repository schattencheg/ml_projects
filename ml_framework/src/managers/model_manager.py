"""
ModelManager - Create or load models for ML.
Supports XGBoost, CatBoost, CNN, Linear Regression, and other models.
"""

import joblib
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import numpy as np

from src.models_lib import (
    BaseModel, XGBoostModel, CatBoostModel, 
    LinearRegressionModel, LogisticRegressionModel, RandomForestModel,
    SimpleCNN, DeepCNN, ResidualCNN
)


class ModelManager:
    """
    Manages ML model creation, configuration, saving, and loading.
    
    Features:
    - Create models: XGBoost, CatBoost, CNN variants, Linear/Logistic Regression, Random Forest
    - All models inherit from BaseModel with automatic target conversion
    - Save/load models with timestamps
    - Model versioning and metadata tracking
    """
    
    def __init__(self, results_dir: str = 'results', use_gpu: bool = False):
        """
        Initialize ModelManager.
        
        Args:
            results_dir: Base directory for results (models will be in timestamped subdirs)
            use_gpu: Whether to enable GPU acceleration for supported models
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.use_gpu = use_gpu
        
        # Model configuration
        self.model_config = {
            'logistic_regression': {
                'enabled': True,
                'type': 'sklearn',
                'params': {'max_iter': 1000, 'random_state': 42}
            },
            'random_forest': {
                'enabled': True,
                'type': 'sklearn',
                'params': {'n_estimators': 100, 'random_state': 42, 'max_depth': 10}
            },
            'xgboost': {
                'enabled': True,
                'type': 'boosting',
                'params': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42}
            },
            'catboost': {
                'enabled': False,
                'type': 'boosting',
                'params': {'iterations': 100, 'depth': 6, 'learning_rate': 0.1, 'random_state': 42, 'verbose': False}
            },
            'linear_regression': {
                'enabled': False,
                'type': 'linear',
                'params': {}
            },
            'simple_cnn': {
                'enabled': False,
                'type': 'deep_learning',
                'params': {'num_classes': 3}
            },
            'deep_cnn': {
                'enabled': False,
                'type': 'deep_learning',
                'params': {'num_classes': 3}
            },
            'residual_cnn': {
                'enabled': False,
                'type': 'deep_learning',
                'params': {'num_classes': 3}
            }
        }
    
    def create_model(self, model_name: str, **override_params) -> BaseModel:
        """
        Create a model instance.
        
        Args:
            model_name: Name of the model to create
            **override_params: Parameters to override default config
            
        Returns:
            Model instance (inherits from BaseModel)
        """
        if model_name not in self.model_config:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.model_config.keys())}")
        
        config = self.model_config[model_name].copy()
        params = config.get('params', {}).copy()
        params.update(override_params)
        
        # Create model based on type
        if model_name == 'logistic_regression':
            return LogisticRegressionModel(name=model_name, **params)
        elif model_name == 'random_forest':
            return RandomForestModel(name=model_name, **params)
        elif model_name == 'xgboost':
            return XGBoostModel(name=model_name, use_gpu=self.use_gpu, **params)
        elif model_name == 'catboost':
            return CatBoostModel(name=model_name, use_gpu=self.use_gpu, **params)
        elif model_name == 'linear_regression':
            return LinearRegressionModel(name=model_name, **params)
        elif model_name == 'simple_cnn':
            return SimpleCNN(name=model_name, use_gpu=self.use_gpu, **params)
        elif model_name == 'deep_cnn':
            return DeepCNN(name=model_name, use_gpu=self.use_gpu, **params)
        elif model_name == 'residual_cnn':
            return ResidualCNN(name=model_name, use_gpu=self.use_gpu, **params)
        else:
            raise ValueError(f"Model creation not implemented for: {model_name}")
    
    def create_models(self, model_names: Optional[List[str]] = None) -> Dict[str, BaseModel]:
        """
        Create multiple model instances.
        
        Args:
            model_names: List of model names to create (None = all enabled models)
            
        Returns:
            Dictionary of model instances
        """
        if model_names is None:
            model_names = self.get_enabled_models()
        
        models = {}
        for name in model_names:
            try:
                models[name] = self.create_model(name)
                print(f"✓ Created model: {name}")
            except Exception as e:
                print(f"✗ Failed to create {name}: {e}")
        
        return models
    
    def get_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get enabled model configurations.
        
        Returns:
            Dictionary of enabled models with their parameters
        """
        enabled_models = {
            name: config for name, config in self.model_config.items()
            if config['enabled']
        }
        
        print(f"✓ Loaded {len(enabled_models)} enabled model configs")
        return enabled_models
    
    def enable_model(self, model_name: str, enabled: bool = True):
        """
        Enable or disable a model.
        
        Args:
            model_name: Name of the model
            enabled: True to enable, False to disable
        """
        if model_name not in self.model_config:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model_config[model_name]['enabled'] = enabled
        status = "enabled" if enabled else "disabled"
        print(f"✓ Model '{model_name}' {status}")
    
    def get_enabled_models(self) -> List[str]:
        """Get list of enabled model names."""
        return [name for name, config in self.model_config.items() 
                if config['enabled']]
    
    def save_models(self, 
                   models: Dict[str, BaseModel],
                   save_dir: Union[str, Path],
                   metadata: Optional[Dict] = None) -> str:
        """
        Save trained models to directory.
        
        Args:
            models: Dictionary of trained models
            save_dir: Directory to save models (should be timestamped run folder)
            metadata: Additional metadata to save (optional)
            
        Returns:
            Path to saved directory
        """
        save_dir = Path(save_dir)
        models_dir = save_dir / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"SAVING MODELS TO: {models_dir}")
        print(f"{'='*70}")
        
        # Save each model
        for model_name, model in models.items():
            model_path = models_dir / f"{model_name}.joblib"
            joblib.dump(model, model_path)
            print(f"✓ Saved {model_name}")
        
        # Save metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'num_models': len(models),
            'model_names': list(models.keys()),
            'saved_at': datetime.now().isoformat()
        })
        
        metadata_path = save_dir / "metadata.joblib"
        joblib.dump(metadata, metadata_path)
        print(f"✓ Saved metadata")
        
        print(f"\n✓ Saved {len(models)} models successfully")
        print(f"{'='*70}\n")
        
        return str(models_dir)
    
    def load_models(self, load_dir: Union[str, Path]) -> tuple:
        """
        Load models from directory.
        
        Args:
            load_dir: Directory containing models (timestamped run folder)
            
        Returns:
            Tuple of (models_dict, metadata)
        """
        load_dir = Path(load_dir)
        models_dir = load_dir / 'models'
        
        if not models_dir.exists():
            raise ValueError(f"Models directory not found: {models_dir}")
        
        print(f"\n{'='*70}")
        print(f"LOADING MODELS FROM: {models_dir}")
        print(f"{'='*70}")
        
        # Load metadata
        metadata_path = load_dir / "metadata.joblib"
        metadata = joblib.load(metadata_path) if metadata_path.exists() else {}
        
        # Load models
        models = {}
        for model_file in models_dir.glob("*.joblib"):
            model_name = model_file.stem
            models[model_name] = joblib.load(model_file)
            print(f"✓ Loaded {model_name}")
        
        print(f"\n✓ Loaded {len(models)} models successfully")
        print(f"{'='*70}\n")
        
        return models, metadata
    
    def print_config(self):
        """Print current model configuration."""
        print("\n" + "="*70)
        print("MODEL CONFIGURATION")
        print("="*70)
        
        enabled = []
        disabled = []
        
        for name, config in self.model_config.items():
            if config['enabled']:
                enabled.append((name, config['type']))
            else:
                disabled.append((name, config['type']))
        
        print(f"\nEnabled models ({len(enabled)}):")
        for name, model_type in enabled:
            print(f"  ✓ {name:<25} [{model_type}]")
        
        if disabled:
            print(f"\nDisabled models ({len(disabled)}):")
            for name, model_type in disabled:
                print(f"  ✗ {name:<25} [{model_type}]")
        
        print("="*70 + "\n")
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model configuration dictionary
        """
        if model_name not in self.model_config:
            raise ValueError(f"Unknown model: {model_name}")
        
        return self.model_config[model_name].copy()
