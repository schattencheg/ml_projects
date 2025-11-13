"""
ModelManager - Manages model configurations, saving, and loading.
"""

import joblib
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path


class ModelManager:
    """
    Manages ML model configurations, saving, and loading.
    
    Features:
    - Model configuration management
    - Enable/disable models
    - Save/load models with timestamps
    - Model versioning
    """
    
    def __init__(self, models_dir: str = 'models'):
        """
        Initialize ModelManager.
        
        Args:
            models_dir: Directory for saving/loading models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configuration
        self.model_config = {
            'logistic_regression': {
                'enabled': True,
                'params': {'max_iter': 1000, 'random_state': 42}
            },
            'random_forest': {
                'enabled': True,
                'params': {'n_estimators': 100, 'random_state': 42}
            },
            'xgboost': {
                'enabled': True,
                'params': {'n_estimators': 100, 'random_state': 42}
            },
            'lightgbm': {
                'enabled': False,
                'params': {'n_estimators': 100, 'random_state': 42}
            },
        }
    
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
        
        print(f"✓ Loaded {len(enabled_models)} enabled models")
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
    
    def save_models(self, models: Dict[str, Any], 
                   scaler: Optional[Any] = None,
                   metadata: Optional[Dict] = None,
                   timestamp: Optional[str] = None) -> str:
        """
        Save trained models to timestamped directory.
        
        Args:
            models: Dictionary of trained models
            scaler: Fitted scaler (optional)
            metadata: Additional metadata to save (optional)
            timestamp: Custom timestamp (auto-generated if None)
            
        Returns:
            Path to saved directory
        """
        # Generate timestamp
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # Create timestamped directory
        save_dir = self.models_dir / timestamp
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"SAVING MODELS TO: {save_dir}")
        print(f"{'='*70}")
        
        # Save each model
        for model_name, model in models.items():
            model_path = save_dir / f"{model_name}.joblib"
            joblib.dump(model, model_path)
            print(f"✓ Saved {model_name}")
        
        # Save scaler
        if scaler is not None:
            scaler_path = save_dir / "scaler.joblib"
            joblib.dump(scaler, scaler_path)
            print(f"✓ Saved scaler")
        
        # Save metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'timestamp': timestamp,
            'num_models': len(models),
            'model_names': list(models.keys()),
        })
        
        metadata_path = save_dir / "metadata.joblib"
        joblib.dump(metadata, metadata_path)
        print(f"✓ Saved metadata")
        
        print(f"\n✓ Saved {len(models)} models successfully to {save_dir}")
        print(f"{'='*70}\n")
        
        return str(save_dir)
    
    def load_models(self, timestamp: str = 'latest') -> tuple:
        """
        Load models from timestamped directory.
        
        Args:
            timestamp: Timestamp of directory to load ('latest' for most recent)
            
        Returns:
            Tuple of (models_dict, scaler, metadata)
        """
        # Find directory
        if timestamp == 'latest':
            timestamp = self._find_latest_timestamp()
        
        load_dir = self.models_dir / timestamp
        
        if not load_dir.exists():
            raise ValueError(f"Directory not found: {load_dir}")
        
        print(f"\n{'='*70}")
        print(f"LOADING MODELS FROM: {load_dir}")
        print(f"{'='*70}")
        
        # Load metadata
        metadata_path = load_dir / "metadata.joblib"
        metadata = joblib.load(metadata_path) if metadata_path.exists() else {}
        
        # Load models
        models = {}
        for model_file in load_dir.glob("*.joblib"):
            if model_file.name in ['scaler.joblib', 'metadata.joblib']:
                continue
            
            model_name = model_file.stem
            models[model_name] = joblib.load(model_file)
            print(f"✓ Loaded {model_name}")
        
        # Load scaler
        scaler_path = load_dir / "scaler.joblib"
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        if scaler is not None:
            print(f"✓ Loaded scaler")
        
        print(f"\n✓ Loaded {len(models)} models successfully")
        print(f"{'='*70}\n")
        
        return models, scaler, metadata
    
    def _find_latest_timestamp(self) -> str:
        """Find the most recent timestamped directory."""
        subdirs = [d for d in self.models_dir.iterdir() if d.is_dir()]
        
        if not subdirs:
            raise ValueError(f"No model directories found in {self.models_dir}")
        
        # Sort by name (timestamp format ensures chronological order)
        latest = sorted(subdirs, reverse=True)[0]
        return latest.name
    
    def list_saved_models(self) -> List[str]:
        """
        List all saved model timestamps.
        
        Returns:
            List of timestamp strings (sorted newest first)
        """
        subdirs = [d.name for d in self.models_dir.iterdir() if d.is_dir()]
        return sorted(subdirs, reverse=True)
    
    def print_config(self):
        """Print current model configuration."""
        print("\n" + "="*70)
        print("MODEL CONFIGURATION")
        print("="*70)
        
        enabled = []
        disabled = []
        
        for name, config in self.model_config.items():
            if config['enabled']:
                enabled.append(name)
            else:
                disabled.append(name)
        
        print(f"\nEnabled models ({len(enabled)}):")
        for name in enabled:
            print(f"  ✓ {name}")
        
        if disabled:
            print(f"\nDisabled models ({len(disabled)}):")
            for name in disabled:
                print(f"  ✗ {name}")
        
        print("="*70 + "\n")
