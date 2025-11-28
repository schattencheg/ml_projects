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
    SimpleCNN, DeepCNN, ResidualCNN,
    LSTMModelWrapper, HybridModelWrapper
)

try:
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class ModelManager:
    """
    Manages ML model creation, configuration, saving, and loading.
    
    Features:
    - Create models: XGBoost, CatBoost, CNN variants, Linear/Logistic Regression, Random Forest
    - All models inherit from BaseModel with automatic target conversion
    - Save/load models with timestamps
    - Model versioning and metadata tracking
    
    Example:
        >>> model_manager = ModelManager(verbose=False)
        >>> model_manager.add_model('logistic_regression')
        >>> model_manager.add_model('xgboost', n_estimators=200)
        >>> model_manager.add_model('random_forest')
        >>> models = model_manager.get_models()  # Returns dict of model instances
    """
    
    def __init__(self, results_dir: str = 'results', use_gpu: bool = False, verbose: bool = True):
        """
        Initialize ModelManager.
        
        Args:
            results_dir: Base directory for results (models will be in timestamped subdirs)
            use_gpu: Whether to enable GPU acceleration for supported models
            verbose: If True, print status messages (default: True)
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.use_gpu = use_gpu
        self.verbose = verbose
        
        # Store created model instances
        self._models: Dict[str, BaseModel] = {}
        
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
        
        # Predefined CNN/LSTM architectures
        self.predefined_architectures = {
            'simple_cnn_small': {
                'type': 'cnn',
                'filters': [32, 64],
                'kernel_size': 3,
                'dense_units': [64],
                'dropout_rate': 0.3,
                'description': 'Small CNN with 2 conv layers'
            },
            'simple_cnn_medium': {
                'type': 'cnn',
                'filters': [64, 128, 256],
                'kernel_size': 3,
                'dense_units': [128, 64],
                'dropout_rate': 0.4,
                'description': 'Medium CNN with 3 conv layers'
            },
            'simple_cnn_large': {
                'type': 'cnn',
                'filters': [128, 256, 512, 512],
                'kernel_size': 3,
                'dense_units': [256, 128],
                'dropout_rate': 0.5,
                'description': 'Large CNN with 4 conv layers'
            },
            'lstm_small': {
                'type': 'lstm',
                'lstm_units': [64],
                'dense_units': [32],
                'dropout_rate': 0.3,
                'description': 'Small LSTM with 1 layer'
            },
            'lstm_medium': {
                'type': 'lstm',
                'lstm_units': [128, 64],
                'dense_units': [64, 32],
                'dropout_rate': 0.4,
                'description': 'Medium LSTM with 2 layers'
            },
            'lstm_large': {
                'type': 'lstm',
                'lstm_units': [256, 128, 64],
                'dense_units': [128, 64],
                'dropout_rate': 0.5,
                'description': 'Large LSTM with 3 layers'
            },
            'hybrid_cnn_lstm': {
                'type': 'hybrid',
                'filters': [64, 128],
                'kernel_size': 3,
                'lstm_units': [64],
                'dense_units': [64, 32],
                'dropout_rate': 0.4,
                'description': 'Hybrid CNN-LSTM architecture'
            }
        }
    
    def add_model(self, model_name: str, custom_name: Optional[str] = None, **override_params) -> 'ModelManager':
        """
        Add a model to the manager. Creates the model instance and stores it.
        
        Args:
            model_name: Name of the model type to create (e.g., 'xgboost', 'random_forest')
            custom_name: Optional custom name for the model (default: use model_name)
            **override_params: Parameters to override default config
            
        Returns:
            Self for method chaining
            
        Example:
            >>> mgr = ModelManager()
            >>> mgr.add_model('xgboost').add_model('random_forest')  # Chaining
            >>> mgr.add_model('xgboost', custom_name='XGB_v2', n_estimators=200)
        """
        if model_name not in self.model_config:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.model_config.keys())}")
        
        # Use custom name or default to model_name
        name = custom_name or model_name
        
        try:
            model = self._create_model_instance(model_name, name, **override_params)
            self._models[name] = model
            if self.verbose:
                print(f"  ✓ Added model: {name}")
        except Exception as e:
            if self.verbose:
                print(f"  ✗ Failed to add {name}: {e}")
            raise
        
        return self
    
    def _create_model_instance(self, model_name: str, name: str, **override_params) -> BaseModel:
        """
        Internal method to create a model instance.
        
        Args:
            model_name: Name of the model type
            name: Name to assign to the model instance
            **override_params: Parameters to override default config
            
        Returns:
            Model instance (inherits from BaseModel)
        """
        config = self.model_config[model_name].copy()
        params = config.get('params', {}).copy()
        params.update(override_params)
        
        # Create model based on type
        if model_name == 'logistic_regression':
            return LogisticRegressionModel(name=name, **params)
        elif model_name == 'random_forest':
            return RandomForestModel(name=name, **params)
        elif model_name == 'xgboost':
            return XGBoostModel(name=name, use_gpu=self.use_gpu, **params)
        elif model_name == 'catboost':
            return CatBoostModel(name=name, use_gpu=self.use_gpu, **params)
        elif model_name == 'linear_regression':
            return LinearRegressionModel(name=name, **params)
        elif model_name == 'simple_cnn':
            return SimpleCNN(name=name, use_gpu=self.use_gpu, **params)
        elif model_name == 'deep_cnn':
            return DeepCNN(name=name, use_gpu=self.use_gpu, **params)
        elif model_name == 'residual_cnn':
            return ResidualCNN(name=name, use_gpu=self.use_gpu, **params)
        else:
            raise ValueError(f"Model creation not implemented for: {model_name}")
    
    def create_model(self, model_name: str, **override_params) -> BaseModel:
        """
        Create a model instance without storing it (legacy method).
        
        Args:
            model_name: Name of the model to create
            **override_params: Parameters to override default config
            
        Returns:
            Model instance (inherits from BaseModel)
        """
        return self._create_model_instance(model_name, model_name, **override_params)
    
    def get_models(self) -> Dict[str, BaseModel]:
        """
        Get all added model instances.
        
        Returns:
            Dictionary of model instances {name: model}
            
        Example:
            >>> mgr = ModelManager()
            >>> mgr.add_model('xgboost').add_model('random_forest')
            >>> models = mgr.get_models()
            >>> for name, model in models.items():
            ...     model.fit(X_train, y_train)
        """
        if self.verbose and self._models:
            print(f"✓ Retrieved {len(self._models)} models: {list(self._models.keys())}")
        return self._models.copy()
    
    def get_model(self, name: str) -> BaseModel:
        """
        Get a specific model by name.
        
        Args:
            name: Name of the model
            
        Returns:
            Model instance
        """
        if name not in self._models:
            raise ValueError(f"Model '{name}' not found. Available: {list(self._models.keys())}")
        return self._models[name]
    
    def remove_model(self, name: str) -> 'ModelManager':
        """
        Remove a model from the manager.
        
        Args:
            name: Name of the model to remove
            
        Returns:
            Self for method chaining
        """
        if name in self._models:
            del self._models[name]
            if self.verbose:
                print(f"  ✓ Removed model: {name}")
        return self
    
    def clear_models(self) -> 'ModelManager':
        """
        Remove all models from the manager.
        
        Returns:
            Self for method chaining
        """
        self._models.clear()
        if self.verbose:
            print("  ✓ Cleared all models")
        return self
    
    def create_models(self, model_names: Optional[List[str]] = None) -> Dict[str, BaseModel]:
        """
        Create multiple model instances (legacy method, prefer add_model).
        
        Args:
            model_names: List of model names to create (None = all enabled models)
            
        Returns:
            Dictionary of model instances
        """
        if model_names is None:
            model_names = self.get_enabled_models()
        
        for name in model_names:
            try:
                self.add_model(name)
            except Exception as e:
                if self.verbose:
                    print(f"✗ Failed to create {name}: {e}")
        
        return self.get_models()
    
    def get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get enabled model configurations (not instances).
        
        Returns:
            Dictionary of enabled models with their parameters
        """
        enabled_models = {
            name: config for name, config in self.model_config.items()
            if config['enabled']
        }
        
        if self.verbose:
            print(f"✓ Found {len(enabled_models)} enabled model configs")
        return enabled_models
    
    def enable_model(self, model_name: str, enabled: bool = True) -> 'ModelManager':
        """
        Enable or disable a model in config.
        
        Args:
            model_name: Name of the model
            enabled: True to enable, False to disable
            
        Returns:
            Self for method chaining
        """
        if model_name not in self.model_config:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model_config[model_name]['enabled'] = enabled
        if self.verbose:
            status = "enabled" if enabled else "disabled"
            print(f"✓ Model '{model_name}' {status}")
        return self
    
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
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"SAVING MODELS TO: {models_dir}")
            print(f"{'='*70}")
        
        # Save each model
        for model_name, model in models.items():
            model_path = models_dir / f"{model_name}.joblib"
            joblib.dump(model, model_path)
            if self.verbose:
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
        if self.verbose:
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
        
        if self.verbose:
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
            if self.verbose:
                print(f"✓ Loaded {model_name}")
        
        if self.verbose:
            print(f"\n✓ Loaded {len(models)} models successfully")
            print(f"{'='*70}\n")
        
        return models, metadata
    
    def print_config(self):
        """Print current model configuration and added models."""
        print("\n" + "="*70)
        print("MODEL MANAGER STATUS")
        print("="*70)
        
        # Show added models
        if self._models:
            print(f"\nAdded models ({len(self._models)}):")
            for name in self._models.keys():
                print(f"  ✓ {name}")
        else:
            print("\nNo models added yet.")
        
        print("\n" + "-"*70)
        print("AVAILABLE MODEL TYPES")
        print("-"*70)
        
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
    
    def create_custom_cnn(self, 
                          name: str,
                          input_shape: tuple,
                          num_classes: int,
                          filters: List[int],
                          kernel_size: int = 3,
                          dense_units: List[int] = [64],
                          dropout_rate: float = 0.3,
                          learning_rate: float = 0.001) -> BaseModel:
        """
        Create a custom CNN model.
        
        Args:
            name: Model name
            input_shape: Input shape (features, 1) for 1D CNN
            num_classes: Number of output classes
            filters: List of filter counts for each conv layer
            kernel_size: Kernel size for conv layers
            dense_units: List of units for dense layers
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            
        Returns:
            SimpleCNN model instance
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for CNN models")
        
        return SimpleCNN(
            name=name,
            input_shape=input_shape,
            num_classes=num_classes,
            filters=filters,
            kernel_size=kernel_size,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )
    
    def create_custom_lstm(self,
                           name: str,
                           input_shape: tuple,
                           num_classes: int,
                           lstm_units: List[int],
                           dense_units: List[int] = [32],
                           dropout_rate: float = 0.3,
                           learning_rate: float = 0.001) -> BaseModel:
        """
        Create a custom LSTM model.
        
        Args:
            name: Model name
            input_shape: Input shape (timesteps, features)
            num_classes: Number of output classes
            lstm_units: List of units for each LSTM layer
            dense_units: List of units for dense layers
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            
        Returns:
            Custom LSTM model instance
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models")
        
        from src.models_lib import BaseModel
        
        # Build LSTM model
        model = keras.Sequential(name=name)
        
        # Add LSTM layers
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            model.add(keras.layers.LSTM(
                units, 
                return_sequences=return_sequences,
                input_shape=input_shape if i == 0 else None
            ))
            model.add(keras.layers.Dropout(dropout_rate))
        
        # Add dense layers
        for units in dense_units:
            model.add(keras.layers.Dense(units, activation='relu'))
            model.add(keras.layers.Dropout(dropout_rate))
        
        # Output layer
        model.add(keras.layers.Dense(num_classes, activation='softmax'))
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Wrap in BaseModel using module-level wrapper
        return LSTMModelWrapper(model, name)
    
    def add_predefined_model(self,
                             architecture_name: str,
                             name: str,
                             input_shape: tuple,
                             num_classes: int,
                             learning_rate: float = 0.001) -> 'ModelManager':
        """
        Add a predefined architecture model to the manager.
        
        Args:
            architecture_name: Name of predefined architecture (e.g., 'simple_cnn_small', 'lstm_medium')
            name: Custom name for the model
            input_shape: Input shape for the model
            num_classes: Number of output classes
            learning_rate: Learning rate
            
        Returns:
            Self for method chaining
            
        Example:
            >>> mgr = ModelManager()
            >>> mgr.add_predefined_model('simple_cnn_small', 'CNN_Small', (100, 1), 3)
            >>> mgr.add_predefined_model('lstm_medium', 'LSTM_Med', (100, 1), 3)
            >>> models = mgr.get_models()
        """
        try:
            model = self.create_from_predefined(architecture_name, name, input_shape, num_classes, learning_rate)
            self._models[name] = model
            if self.verbose:
                print(f"  ✓ Added model: {name} ({architecture_name})")
        except Exception as e:
            if self.verbose:
                print(f"  ✗ Failed to add {name}: {e}")
            raise
        return self
    
    def create_from_predefined(self,
                               architecture_name: str,
                               name: str,
                               input_shape: tuple,
                               num_classes: int,
                               learning_rate: float = 0.001) -> BaseModel:
        """
        Create a model from predefined architecture without storing it.
        
        Args:
            architecture_name: Name of predefined architecture
            name: Model name
            input_shape: Input shape
            num_classes: Number of output classes
            learning_rate: Learning rate
            
        Returns:
            Model instance
        """
        if architecture_name not in self.predefined_architectures:
            raise ValueError(f"Unknown architecture: {architecture_name}. "
                           f"Available: {list(self.predefined_architectures.keys())}")
        
        arch = self.predefined_architectures[architecture_name]
        arch_type = arch['type']
        
        if arch_type == 'cnn':
            return self.create_custom_cnn(
                name=name,
                input_shape=input_shape,
                num_classes=num_classes,
                filters=arch['filters'],
                kernel_size=arch['kernel_size'],
                dense_units=arch['dense_units'],
                dropout_rate=arch['dropout_rate'],
                learning_rate=learning_rate
            )
        elif arch_type == 'lstm':
            return self.create_custom_lstm(
                name=name,
                input_shape=input_shape,
                num_classes=num_classes,
                lstm_units=arch['lstm_units'],
                dense_units=arch['dense_units'],
                dropout_rate=arch['dropout_rate'],
                learning_rate=learning_rate
            )
        elif arch_type == 'hybrid':
            # Create hybrid CNN-LSTM
            if not TENSORFLOW_AVAILABLE:
                raise ImportError("TensorFlow is required for hybrid models")
            
            from src.models_lib import BaseModel
            
            model = keras.Sequential(name=name)
            
            # CNN layers
            for i, filters in enumerate(arch['filters']):
                model.add(keras.layers.Conv1D(
                    filters, 
                    arch['kernel_size'], 
                    activation='relu',
                    input_shape=input_shape if i == 0 else None
                ))
                model.add(keras.layers.MaxPooling1D(2))
                model.add(keras.layers.Dropout(arch['dropout_rate']))
            
            # LSTM layers
            for i, units in enumerate(arch['lstm_units']):
                return_sequences = i < len(arch['lstm_units']) - 1
                model.add(keras.layers.LSTM(units, return_sequences=return_sequences))
                model.add(keras.layers.Dropout(arch['dropout_rate']))
            
            # Dense layers
            for units in arch['dense_units']:
                model.add(keras.layers.Dense(units, activation='relu'))
                model.add(keras.layers.Dropout(arch['dropout_rate']))
            
            # Output
            model.add(keras.layers.Dense(num_classes, activation='softmax'))
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Wrap in BaseModel using module-level wrapper
            return HybridModelWrapper(model, name)
        else:
            raise ValueError(f"Unknown architecture type: {arch_type}")
    
    def list_available_models(self) -> List[str]:
        """
        Get list of all available model types that can be added.
        
        Returns:
            List of model type names
        """
        return list(self.model_config.keys())
    
    def __len__(self) -> int:
        """Return number of added models."""
        return len(self._models)
    
    def __contains__(self, name: str) -> bool:
        """Check if a model with given name exists."""
        return name in self._models
    
    def list_predefined_architectures(self):
        """
        Print all predefined architectures.
        """
        print("\n" + "="*70)
        print("PREDEFINED CNN/LSTM ARCHITECTURES")
        print("="*70)
        
        for name, arch in self.predefined_architectures.items():
            print(f"\n{name}:")
            print(f"  Type: {arch['type']}")
            print(f"  Description: {arch['description']}")
            if 'filters' in arch:
                print(f"  Filters: {arch['filters']}")
            if 'lstm_units' in arch:
                print(f"  LSTM Units: {arch['lstm_units']}")
            print(f"  Dense Units: {arch['dense_units']}")
            print(f"  Dropout: {arch['dropout_rate']}")
        
        print("\n" + "="*70 + "\n")
    
    def get_predefined_architectures(self) -> List[str]:
        """
        Get list of predefined architecture names.
        
        Returns:
            List of architecture names
        """
        return list(self.predefined_architectures.keys())
