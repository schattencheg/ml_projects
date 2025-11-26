"""
RunManager - Manages experiment runs with timestamped artifact storage.
Saves models, scalers, datasets, feature selectors, and metadata to organized folders.
"""

import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from pathlib import Path


class RunManager:
    """
    Manages experiment runs with organized artifact storage.
    
    Directory structure:
        models/YYYYMMDD_HHMMSS/
        ├── models/           # Trained models (.joblib)
        ├── scalers/          # Fitted scalers (.joblib)
        ├── features/         # Feature selector and importance (.joblib, .csv)
        ├── datasets/         # Train/Val/Test splits (.parquet)
        ├── reports/          # HTML visualization reports
        └── metadata.json     # Run configuration and metrics
    """
    
    def __init__(self, base_dir: str = 'models', run_name: Optional[str] = None):
        """
        Initialize RunManager.
        
        Args:
            base_dir: Base directory for all runs (default: 'models')
            run_name: Optional custom run name (default: timestamp)
        """
        self.base_dir = Path(base_dir)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_name = run_name or self.timestamp
        self.run_dir = self.base_dir / self.run_name
        
        # Subdirectories
        self.models_dir = self.run_dir / 'models'
        self.scalers_dir = self.run_dir / 'scalers'
        self.features_dir = self.run_dir / 'features'
        self.datasets_dir = self.run_dir / 'datasets'
        self.reports_dir = self.run_dir / 'reports'
        
        # Metadata storage
        self.metadata = {
            'run_name': self.run_name,
            'timestamp': self.timestamp,
            'created_at': datetime.now().isoformat(),
            'artifacts': {}
        }
        
        self._initialized = False
    
    def initialize(self) -> 'RunManager':
        """Create directory structure for the run."""
        if self._initialized:
            return self
            
        print(f"\n{'='*70}")
        print(f"INITIALIZING RUN: {self.run_name}")
        print(f"{'='*70}")
        
        # Create all directories
        for dir_path in [self.models_dir, self.scalers_dir, self.features_dir, 
                         self.datasets_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Run directory: {self.run_dir}")
        self._initialized = True
        return self
    
    def save_models(self, models: Dict[str, Any], metrics: Optional[Dict[str, Dict]] = None) -> Path:
        """
        Save trained models.
        
        Args:
            models: Dictionary of {model_name: model_instance}
            metrics: Optional dictionary of {model_name: {metric: value}}
            
        Returns:
            Path to models directory
        """
        self.initialize()
        
        print(f"\nSaving {len(models)} models...")
        
        saved_models = []
        for name, model in models.items():
            model_path = self.models_dir / f"{name}.joblib"
            joblib.dump(model, model_path)
            saved_models.append(name)
            print(f"  ✓ {name}")
        
        # Update metadata
        self.metadata['artifacts']['models'] = {
            'count': len(models),
            'names': saved_models,
            'metrics': metrics or {}
        }
        
        self._save_metadata()
        return self.models_dir
    
    def save_scaler(self, scaler, scaler_name: str = 'scaler') -> Path:
        """
        Save fitted scaler.
        
        Args:
            scaler: Fitted scaler instance (ScalerManager or sklearn scaler)
            scaler_name: Name for the scaler file
            
        Returns:
            Path to saved scaler
        """
        self.initialize()
        
        scaler_path = self.scalers_dir / f"{scaler_name}.joblib"
        
        # Handle ScalerManager or raw sklearn scaler
        if hasattr(scaler, 'scaler'):
            # ScalerManager
            scaler_data = {
                'scaler': scaler.scaler,
                'scaler_type': getattr(scaler, 'scaler_type', 'unknown'),
                'feature_names': getattr(scaler, 'feature_names', None),
                'is_fitted': getattr(scaler, 'is_fitted', True)
            }
        else:
            # Raw sklearn scaler
            scaler_data = {'scaler': scaler, 'scaler_type': type(scaler).__name__}
        
        joblib.dump(scaler_data, scaler_path)
        print(f"✓ Saved scaler: {scaler_path.name}")
        
        self.metadata['artifacts']['scaler'] = {
            'name': scaler_name,
            'type': scaler_data.get('scaler_type', 'unknown'),
            'feature_count': len(scaler_data.get('feature_names', []) or [])
        }
        
        self._save_metadata()
        return scaler_path
    
    def save_feature_importance(self, 
                                 feature_importance: pd.Series,
                                 selected_features: Optional[List[str]] = None,
                                 dropped_features: Optional[List[str]] = None,
                                 method: str = 'unknown',
                                 selector: Any = None) -> Path:
        """
        Save feature importance analysis results.
        
        Args:
            feature_importance: Series with feature names as index and importance as values
            selected_features: List of selected feature names
            dropped_features: List of dropped feature names
            method: Feature selection method used
            selector: Optional fitted selector object
            
        Returns:
            Path to features directory
        """
        self.initialize()
        
        # Save importance as CSV for easy viewing
        importance_df = feature_importance.reset_index()
        importance_df.columns = ['feature', 'importance']
        importance_df.to_csv(self.features_dir / 'feature_importance.csv', index=False)
        print(f"✓ Saved feature importance CSV")
        
        # Save full selector data as joblib
        selector_data = {
            'method': method,
            'feature_importance': feature_importance,
            'selected_features': selected_features or feature_importance.index.tolist(),
            'dropped_features': dropped_features or [],
            'selector': selector
        }
        joblib.dump(selector_data, self.features_dir / 'feature_selector.joblib')
        print(f"✓ Saved feature selector")
        
        # Update metadata
        self.metadata['artifacts']['features'] = {
            'method': method,
            'total_features': len(feature_importance),
            'selected_count': len(selected_features) if selected_features else len(feature_importance),
            'dropped_count': len(dropped_features) if dropped_features else 0,
            'top_5_features': feature_importance.head(5).to_dict()
        }
        
        self._save_metadata()
        return self.features_dir
    
    def save_datasets(self,
                      X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
                      X_test: Optional[pd.DataFrame] = None, y_test: Optional[pd.Series] = None,
                      df_full: Optional[pd.DataFrame] = None) -> Path:
        """
        Save train/val/test datasets.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data (optional)
            X_test, y_test: Test data (optional)
            df_full: Full original DataFrame (optional)
            
        Returns:
            Path to datasets directory
        """
        self.initialize()
        
        print(f"\nSaving datasets...")
        
        dataset_info = {}
        
        # Save training data
        train_df = X_train.copy()
        train_df['target'] = y_train.values
        train_df.to_parquet(self.datasets_dir / 'train.parquet')
        dataset_info['train'] = {'rows': len(train_df), 'features': len(X_train.columns)}
        print(f"  ✓ train.parquet ({len(train_df)} rows)")
        
        # Save validation data
        if X_val is not None and y_val is not None:
            val_df = X_val.copy()
            val_df['target'] = y_val.values
            val_df.to_parquet(self.datasets_dir / 'val.parquet')
            dataset_info['val'] = {'rows': len(val_df), 'features': len(X_val.columns)}
            print(f"  ✓ val.parquet ({len(val_df)} rows)")
        
        # Save test data
        if X_test is not None and y_test is not None:
            test_df = X_test.copy()
            test_df['target'] = y_test.values
            test_df.to_parquet(self.datasets_dir / 'test.parquet')
            dataset_info['test'] = {'rows': len(test_df), 'features': len(X_test.columns)}
            print(f"  ✓ test.parquet ({len(test_df)} rows)")
        
        # Save full DataFrame if provided
        if df_full is not None:
            df_full.to_parquet(self.datasets_dir / 'full_data.parquet')
            dataset_info['full'] = {'rows': len(df_full), 'columns': len(df_full.columns)}
            print(f"  ✓ full_data.parquet ({len(df_full)} rows)")
        
        # Save feature names
        feature_names = X_train.columns.tolist()
        joblib.dump(feature_names, self.datasets_dir / 'feature_names.joblib')
        
        self.metadata['artifacts']['datasets'] = dataset_info
        self.metadata['feature_names'] = feature_names
        
        self._save_metadata()
        return self.datasets_dir
    
    def save_config(self, config: Dict[str, Any]) -> Path:
        """
        Save run configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Path to config file
        """
        self.initialize()
        
        self.metadata['config'] = config
        self._save_metadata()
        
        # Also save as separate JSON for easy viewing
        config_path = self.run_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        print(f"✓ Saved configuration")
        return config_path
    
    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Save evaluation metrics.
        
        Args:
            metrics: Metrics dictionary
        """
        self.initialize()
        self.metadata['metrics'] = metrics
        self._save_metadata()
        print(f"✓ Saved metrics")
    
    def _save_metadata(self) -> None:
        """Save metadata to JSON file."""
        metadata_path = self.run_dir / 'metadata.json'
        
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(i) for i in obj]
            return obj
        
        with open(metadata_path, 'w') as f:
            json.dump(convert_types(self.metadata), f, indent=2, default=str)
    
    def get_run_dir(self) -> Path:
        """Get the run directory path."""
        self.initialize()
        return self.run_dir
    
    def get_reports_dir(self) -> Path:
        """Get the reports directory path."""
        self.initialize()
        return self.reports_dir
    
    @classmethod
    def load(cls, run_dir: Union[str, Path]) -> 'RunManager':
        """
        Load an existing run.
        
        Args:
            run_dir: Path to run directory
            
        Returns:
            RunManager instance
        """
        run_dir = Path(run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        
        # Extract run name from path
        run_name = run_dir.name
        base_dir = run_dir.parent
        
        instance = cls(base_dir=str(base_dir), run_name=run_name)
        instance._initialized = True
        
        # Load metadata
        metadata_path = run_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                instance.metadata = json.load(f)
        
        print(f"✓ Loaded run: {run_name}")
        return instance
    
    def load_models(self) -> Dict[str, Any]:
        """Load all saved models."""
        models = {}
        for model_file in self.models_dir.glob('*.joblib'):
            name = model_file.stem
            models[name] = joblib.load(model_file)
        return models
    
    def load_scaler(self, scaler_name: str = 'scaler'):
        """
        Load saved scaler.
        
        Returns:
            The sklearn scaler object (not the wrapper dict)
        """
        scaler_path = self.scalers_dir / f"{scaler_name}.joblib"
        scaler_data = joblib.load(scaler_path)
        
        # Handle both dict format and raw scaler
        if isinstance(scaler_data, dict) and 'scaler' in scaler_data:
            return scaler_data['scaler']
        return scaler_data
    
    def load_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all saved datasets."""
        datasets = {}
        for dataset_file in self.datasets_dir.glob('*.parquet'):
            name = dataset_file.stem
            datasets[name] = pd.read_parquet(dataset_file)
        return datasets
    
    def load_feature_importance(self) -> Dict[str, Any]:
        """Load feature importance data."""
        selector_path = self.features_dir / 'feature_selector.joblib'
        if selector_path.exists():
            return joblib.load(selector_path)
        return {}
    
    def print_summary(self) -> None:
        """Print run summary."""
        print(f"\n{'='*70}")
        print(f"RUN SUMMARY: {self.run_name}")
        print(f"{'='*70}")
        print(f"Directory: {self.run_dir}")
        print(f"Created: {self.metadata.get('created_at', 'unknown')}")
        
        artifacts = self.metadata.get('artifacts', {})
        
        if 'models' in artifacts:
            print(f"\nModels: {artifacts['models'].get('count', 0)}")
            for name in artifacts['models'].get('names', []):
                print(f"  - {name}")
        
        if 'datasets' in artifacts:
            print(f"\nDatasets:")
            for name, info in artifacts['datasets'].items():
                print(f"  - {name}: {info.get('rows', 0)} rows")
        
        if 'features' in artifacts:
            feat = artifacts['features']
            print(f"\nFeatures: {feat.get('selected_count', 0)} selected, {feat.get('dropped_count', 0)} dropped")
        
        if 'config' in self.metadata:
            print(f"\nConfiguration:")
            for key, value in self.metadata['config'].items():
                print(f"  {key}: {value}")
        
        print(f"{'='*70}\n")
    
    def __repr__(self) -> str:
        return f"RunManager(run='{self.run_name}', dir='{self.run_dir}')"
