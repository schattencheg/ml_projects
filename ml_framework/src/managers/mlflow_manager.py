"""
MLFlowManager - Connects to local MLFlow and tracks experiments.
"""

import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    import mlflow.catboost
    import mlflow.tensorflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class MLFlowManager:
    """
    Manages MLFlow experiment tracking.
    
    Features:
    - Connect to local MLFlow server
    - Track experiments and runs
    - Log parameters, metrics, and artifacts
    - Log models
    """
    
    def __init__(self, 
                 tracking_uri: str = "http://localhost:5000",
                 experiment_name: str = "ml_framework"):
        """
        Initialize MLFlowManager.
        
        Args:
            tracking_uri: MLFlow tracking server URI
            experiment_name: Name of the experiment
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLFlow is not installed. Install with: pip install mlflow")
        
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.experiment_id = None
        self.active_run = None
        
    def connect(self):
        """Connect to MLFlow tracking server."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
                print(f"✓ Created new experiment: {self.experiment_name}")
            else:
                self.experiment_id = experiment.experiment_id
                print(f"✓ Connected to existing experiment: {self.experiment_name}")
            
            mlflow.set_experiment(self.experiment_name)
            print(f"✓ Connected to MLFlow at {self.tracking_uri}")
            
        except Exception as e:
            print(f"✗ Failed to connect to MLFlow: {e}")
            print(f"  Make sure MLFlow server is running at {self.tracking_uri}")
            print(f"  Start with: mlflow server --host 0.0.0.0 --port 5000")
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict] = None):
        """
        Start a new MLFlow run.
        
        Args:
            run_name: Name for the run
            tags: Dictionary of tags
        """
        self.active_run = mlflow.start_run(run_name=run_name)
        
        if tags:
            mlflow.set_tags(tags)
        
        print(f"✓ Started MLFlow run: {run_name or 'unnamed'}")
        return self.active_run
    
    def end_run(self):
        """End the active MLFlow run."""
        if self.active_run:
            mlflow.end_run()
            print("✓ Ended MLFlow run")
            self.active_run = None
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to MLFlow.
        
        Args:
            params: Dictionary of parameters
        """
        try:
            # Flatten nested dictionaries
            flat_params = self._flatten_dict(params)
            mlflow.log_params(flat_params)
            print(f"✓ Logged {len(flat_params)} parameters")
        except Exception as e:
            print(f"✗ Failed to log parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to MLFlow.
        
        Args:
            metrics: Dictionary of metrics
            step: Step number (for tracking over time)
        """
        try:
            mlflow.log_metrics(metrics, step=step)
            print(f"✓ Logged {len(metrics)} metrics")
        except Exception as e:
            print(f"✗ Failed to log metrics: {e}")
    
    def log_model(self, model: Any, model_name: str, model_type: str = 'sklearn'):
        """
        Log a model to MLFlow.
        
        Args:
            model: Model instance
            model_name: Name for the model
            model_type: Type of model ('sklearn', 'xgboost', 'catboost', 'tensorflow')
        """
        try:
            if model_type == 'sklearn':
                mlflow.sklearn.log_model(model, model_name)
            elif model_type == 'xgboost':
                mlflow.xgboost.log_model(model, model_name)
            elif model_type == 'catboost':
                mlflow.catboost.log_model(model, model_name)
            elif model_type == 'tensorflow':
                mlflow.tensorflow.log_model(model, model_name)
            else:
                # Generic logging
                mlflow.sklearn.log_model(model, model_name)
            
            print(f"✓ Logged model: {model_name}")
        except Exception as e:
            print(f"✗ Failed to log model {model_name}: {e}")
    
    def log_artifact(self, artifact_path: str, artifact_name: Optional[str] = None):
        """
        Log an artifact (file) to MLFlow.
        
        Args:
            artifact_path: Path to the artifact file
            artifact_name: Name for the artifact (optional)
        """
        try:
            mlflow.log_artifact(artifact_path, artifact_name)
            print(f"✓ Logged artifact: {artifact_path}")
        except Exception as e:
            print(f"✗ Failed to log artifact: {e}")
    
    def log_artifacts(self, artifacts_dir: str):
        """
        Log all artifacts in a directory to MLFlow.
        
        Args:
            artifacts_dir: Directory containing artifacts
        """
        try:
            mlflow.log_artifacts(artifacts_dir)
            print(f"✓ Logged artifacts from: {artifacts_dir}")
        except Exception as e:
            print(f"✗ Failed to log artifacts: {e}")
    
    def log_training_results(self, 
                            train_results: Dict[str, Dict],
                            model_configs: Optional[Dict] = None):
        """
        Log training results for multiple models.
        
        Args:
            train_results: Dictionary of training results
            model_configs: Dictionary of model configurations
        """
        for model_name, results in train_results.items():
            if results.get('status') != 'success':
                continue
            
            # Log metrics with model name prefix
            metrics = {
                f"{model_name}_train_accuracy": results.get('train_accuracy', 0),
                f"{model_name}_training_time": results.get('training_time', 0)
            }
            
            if results.get('val_accuracy') is not None:
                metrics[f"{model_name}_val_accuracy"] = results['val_accuracy']
            
            self.log_metrics(metrics)
            
            # Log model config if available
            if model_configs and model_name in model_configs:
                config = model_configs[model_name]
                params = {f"{model_name}_{k}": v for k, v in config.get('params', {}).items()}
                self.log_params(params)
    
    def log_test_results(self, test_results: Dict[str, Dict]):
        """
        Log test results for multiple models.
        
        Args:
            test_results: Dictionary of test results
        """
        for model_name, results in test_results.items():
            if results.get('status') != 'success':
                continue
            
            # Log metrics with model name prefix
            metrics = {
                f"{model_name}_test_accuracy": results.get('accuracy', 0),
                f"{model_name}_test_precision": results.get('precision', 0),
                f"{model_name}_test_recall": results.get('recall', 0),
                f"{model_name}_test_f1": results.get('f1_score', 0)
            }
            
            self.log_metrics(metrics)
    
    def log_backtest_results(self, backtest_results: Dict[str, Any], model_name: str):
        """
        Log backtest results.
        
        Args:
            backtest_results: Dictionary of backtest results
            model_name: Name of the model
        """
        metrics = {
            f"{model_name}_total_return": backtest_results.get('total_return', 0),
            f"{model_name}_sharpe_ratio": backtest_results.get('sharpe_ratio', 0),
            f"{model_name}_max_drawdown": backtest_results.get('max_drawdown', 0),
            f"{model_name}_win_rate": backtest_results.get('win_rate', 0),
            f"{model_name}_num_trades": backtest_results.get('num_trades', 0)
        }
        
        self.log_metrics(metrics)
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """
        Flatten nested dictionary.
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key for recursion
            sep: Separator for nested keys
            
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (list, tuple, np.ndarray)):
                # Convert to string for lists/arrays
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        
        return dict(items)
    
    def set_tags(self, tags: Dict[str, str]):
        """
        Set tags for the current run.
        
        Args:
            tags: Dictionary of tags
        """
        try:
            mlflow.set_tags(tags)
            print(f"✓ Set {len(tags)} tags")
        except Exception as e:
            print(f"✗ Failed to set tags: {e}")
    
    def get_experiment_id(self) -> Optional[str]:
        """Get the current experiment ID."""
        return self.experiment_id
    
    def get_run_id(self) -> Optional[str]:
        """Get the current run ID."""
        if self.active_run:
            return self.active_run.info.run_id
        return None
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.end_run()
