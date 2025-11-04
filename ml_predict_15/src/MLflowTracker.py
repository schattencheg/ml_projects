"""
MLflow Tracking Integration

Provides comprehensive experiment tracking for all models including:
- Traditional ML models (sklearn, XGBoost, LightGBM)
- Neural networks (TensorFlow/Keras models)
- Model parameters, metrics, and artifacts
- Automatic server detection and fallback
"""

import os
import json
import pickle
import tempfile
import warnings
from datetime import datetime
from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np

# MLflow imports with error handling
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.keras
    import mlflow.tensorflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow not available. Install with: pip install mlflow")

# Suppress MLflow warnings
warnings.filterwarnings('ignore', category=UserWarning, module='mlflow')


class MLflowTracker:
    """
    Comprehensive MLflow tracking for ML experiments.
    
    Features:
    - Automatic server detection
    - Support for all model types (sklearn, keras, etc.)
    - Comprehensive logging (parameters, metrics, artifacts)
    - Graceful fallback when MLflow unavailable
    - Neural network specific tracking
    """
    
    def __init__(self, 
                 experiment_name: str = "ml_predict_15/crypto_prediction",
                 tracking_uri: str = "http://localhost:5000",
                 auto_detect_server: bool = True,
                 enable_tracking: bool = True):
        """
        Initialize MLflow tracker.
        
        Parameters:
        -----------
        experiment_name : str
            Name of the MLflow experiment
        tracking_uri : str
            MLflow tracking server URI
        auto_detect_server : bool
            Whether to auto-detect running MLflow server
        enable_tracking : bool
            Whether to enable MLflow tracking
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.auto_detect_server = auto_detect_server
        self.enable_tracking = enable_tracking and MLFLOW_AVAILABLE
        
        self.client = None
        self.experiment_id = None
        self.run_id = None
        self.server_available = False
        
        if self.enable_tracking:
            self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow connection and experiment."""
        try:
            # Auto-detect server if enabled
            if self.auto_detect_server:
                self._detect_mlflow_server()
            
            # Set tracking URI
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Test connection
            self.client = MlflowClient(self.tracking_uri)
            self.client.list_experiments()  # Test connection
            
            # Create or get experiment
            try:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
            except Exception:
                # Experiment already exists
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                self.experiment_id = experiment.experiment_id
            
            mlflow.set_experiment(self.experiment_name)
            self.server_available = True
            
            print(f"✓ MLflow tracking enabled")
            print(f"  Server: {self.tracking_uri}")
            print(f"  Experiment: {self.experiment_name}")
            
        except Exception as e:
            print(f"⚠ MLflow server not available: {e}")
            print(f"  Tracking disabled. To enable:")
            print(f"  1. Start MLflow server: mlflow server --host 127.0.0.1 --port 5000")
            print(f"  2. Or disable tracking: use_mlflow=False")
            self.server_available = False
            self.enable_tracking = False
    
    def _detect_mlflow_server(self):
        """Auto-detect running MLflow server on common ports."""
        common_ports = [5000, 8080, 8000, 5001]
        
        for port in common_ports:
            test_uri = f"http://localhost:{port}"
            try:
                test_client = MlflowClient(test_uri)
                test_client.list_experiments()
                self.tracking_uri = test_uri
                print(f"✓ Auto-detected MLflow server at {test_uri}")
                return
            except Exception:
                continue
        
        print(f"⚠ No MLflow server auto-detected, using default: {self.tracking_uri}")
    
    def start_run(self, run_name: Optional[str] = None) -> Optional[str]:
        """
        Start a new MLflow run.
        
        Parameters:
        -----------
        run_name : str, optional
            Name for the run. If None, auto-generated.
            
        Returns:
        --------
        str : Run ID if successful, None otherwise
        """
        if not self.enable_tracking or not self.server_available:
            return None
        
        try:
            if run_name is None:
                run_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            run = mlflow.start_run(run_name=run_name)
            self.run_id = run.info.run_id
            
            print(f"\n{'='*80}")
            print(f"MLFLOW TRACKING STARTED")
            print(f"{'='*80}")
            print(f"Run: {run_name}")
            print(f"Run ID: {self.run_id}")
            print(f"View at: {self.tracking_uri}")
            print(f"{'='*80}")
            
            return self.run_id
            
        except Exception as e:
            print(f"✗ Failed to start MLflow run: {e}")
            return None
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        if not self.enable_tracking or not self.server_available:
            return
        
        try:
            # Convert complex objects to strings
            clean_params = {}
            for key, value in params.items():
                if isinstance(value, (dict, list, tuple)):
                    clean_params[key] = str(value)
                elif isinstance(value, np.ndarray):
                    clean_params[key] = f"array_shape_{value.shape}"
                else:
                    clean_params[key] = value
            
            mlflow.log_params(clean_params)
            
        except Exception as e:
            print(f"⚠ Failed to log parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow."""
        if not self.enable_tracking or not self.server_available:
            return
        
        try:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    mlflow.log_metric(key, value, step=step)
                    
        except Exception as e:
            print(f"⚠ Failed to log metrics: {e}")
    
    def log_model(self, model, model_name: str, model_type: str = "auto"):
        """
        Log model to MLflow with automatic type detection.
        
        Parameters:
        -----------
        model : object
            Trained model to log
        model_name : str
            Name for the model
        model_type : str
            Type of model ('sklearn', 'keras', 'auto')
        """
        if not self.enable_tracking or not self.server_available:
            return
        
        try:
            # Auto-detect model type
            if model_type == "auto":
                model_type = self._detect_model_type(model)
            
            # Log based on model type
            if model_type == "sklearn":
                mlflow.sklearn.log_model(
                    model, 
                    model_name,
                    registered_model_name=f"ml_predict_15_{model_name}"
                )
            elif model_type == "keras":
                mlflow.keras.log_model(
                    model.model if hasattr(model, 'model') else model,
                    model_name,
                    registered_model_name=f"ml_predict_15_{model_name}"
                )
            elif model_type == "tensorflow":
                mlflow.tensorflow.log_model(
                    model.model if hasattr(model, 'model') else model,
                    model_name,
                    registered_model_name=f"ml_predict_15_{model_name}"
                )
            else:
                # Fallback: save as pickle
                with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                    pickle.dump(model, f)
                    mlflow.log_artifact(f.name, f"models/{model_name}.pkl")
                    os.unlink(f.name)
            
            print(f"✓ Model logged: {model_name} ({model_type})")
            
        except Exception as e:
            print(f"⚠ Failed to log model {model_name}: {e}")
    
    def _detect_model_type(self, model) -> str:
        """Detect the type of model for appropriate logging."""
        model_class = str(type(model))
        
        # Check for neural network wrappers
        if hasattr(model, 'model') and 'keras' in str(type(model.model)):
            return "keras"
        
        # Check for TensorFlow/Keras models
        if 'keras' in model_class or 'tensorflow' in model_class:
            return "keras"
        
        # Check for sklearn-compatible models
        if hasattr(model, 'fit') and hasattr(model, 'predict'):
            return "sklearn"
        
        return "unknown"
    
    def log_artifacts(self, artifacts: Dict[str, Any]):
        """
        Log various artifacts to MLflow.
        
        Parameters:
        -----------
        artifacts : dict
            Dictionary of artifact_name -> artifact_data
        """
        if not self.enable_tracking or not self.server_available:
            return
        
        try:
            for name, data in artifacts.items():
                if isinstance(data, pd.DataFrame):
                    # Save DataFrame as CSV
                    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
                        data.to_csv(f.name, index=False)
                        mlflow.log_artifact(f.name, f"results/{name}.csv")
                        os.unlink(f.name)
                
                elif isinstance(data, dict):
                    # Save dict as JSON
                    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
                        json.dump(data, f, indent=2, default=str)
                        mlflow.log_artifact(f.name, f"config/{name}.json")
                        os.unlink(f.name)
                
                elif isinstance(data, str) and os.path.exists(data):
                    # Log file path
                    mlflow.log_artifact(data, f"files/{name}")
                
                else:
                    # Save as text
                    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w') as f:
                        f.write(str(data))
                        mlflow.log_artifact(f.name, f"misc/{name}.txt")
                        os.unlink(f.name)
            
            print(f"✓ Artifacts logged: {list(artifacts.keys())}")
            
        except Exception as e:
            print(f"⚠ Failed to log artifacts: {e}")
    
    def log_training_results(self, 
                           models: Dict[str, Any],
                           results: Dict[str, Dict[str, float]],
                           best_model_name: str,
                           training_config: Dict[str, Any]):
        """
        Log comprehensive training results.
        
        Parameters:
        -----------
        models : dict
            Dictionary of trained models
        results : dict
            Training results for each model
        best_model_name : str
            Name of the best performing model
        training_config : dict
            Training configuration parameters
        """
        if not self.enable_tracking or not self.server_available:
            return
        
        try:
            # Log training configuration
            self.log_params(training_config)
            
            # Log model-specific metrics
            all_metrics = {}
            for model_name, model_results in results.items():
                for metric_name, value in model_results.items():
                    if isinstance(value, (int, float)):
                        all_metrics[f"{model_name}_{metric_name}"] = value
            
            # Log best model metrics
            if best_model_name in results:
                best_results = results[best_model_name]
                for metric_name, value in best_results.items():
                    if isinstance(value, (int, float)):
                        all_metrics[f"best_{metric_name}"] = value
            
            # Add summary metrics
            all_metrics["num_models_trained"] = len(models)
            all_metrics["best_model_name"] = best_model_name
            
            self.log_metrics(all_metrics)
            
            # Log best model
            if best_model_name in models:
                self.log_model(models[best_model_name], best_model_name)
            
            # Log results as artifacts
            results_df = pd.DataFrame(results).T
            artifacts = {
                "training_results": results_df,
                "training_config": training_config,
                "model_summary": {
                    "best_model": best_model_name,
                    "num_models": len(models),
                    "model_names": list(models.keys())
                }
            }
            self.log_artifacts(artifacts)
            
            print(f"✓ Training results logged to MLflow")
            
        except Exception as e:
            print(f"⚠ Failed to log training results: {e}")
    
    def log_neural_network_info(self, model, model_name: str):
        """
        Log neural network specific information.
        
        Parameters:
        -----------
        model : KerasClassifierWrapper or similar
            Neural network model
        model_name : str
            Name of the model
        """
        if not self.enable_tracking or not self.server_available:
            return
        
        try:
            nn_params = {}
            
            # Extract neural network parameters
            if hasattr(model, 'sequence_length'):
                nn_params[f"{model_name}_sequence_length"] = model.sequence_length
            if hasattr(model, 'epochs'):
                nn_params[f"{model_name}_epochs"] = model.epochs
            if hasattr(model, 'batch_size'):
                nn_params[f"{model_name}_batch_size"] = model.batch_size
            
            # Extract model architecture info
            if hasattr(model, 'model') and model.model is not None:
                keras_model = model.model
                nn_params[f"{model_name}_total_params"] = keras_model.count_params()
                nn_params[f"{model_name}_layers"] = len(keras_model.layers)
                
                # Get model summary
                try:
                    import io
                    summary_buffer = io.StringIO()
                    keras_model.summary(print_fn=lambda x: summary_buffer.write(x + '\n'))
                    model_summary = summary_buffer.getvalue()
                    
                    # Log model architecture as artifact
                    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w') as f:
                        f.write(model_summary)
                        mlflow.log_artifact(f.name, f"model_architectures/{model_name}_summary.txt")
                        os.unlink(f.name)
                        
                except Exception:
                    pass
            
            self.log_params(nn_params)
            print(f"✓ Neural network info logged: {model_name}")
            
        except Exception as e:
            print(f"⚠ Failed to log neural network info: {e}")
    
    def end_run(self):
        """End the current MLflow run."""
        if not self.enable_tracking or not self.server_available:
            return
        
        try:
            mlflow.end_run()
            
            print(f"\n{'='*80}")
            print(f"MLFLOW TRACKING COMPLETED")
            print(f"{'='*80}")
            print(f"Run ID: {self.run_id}")
            print(f"View results at: {self.tracking_uri}")
            print(f"{'='*80}\n")
            
        except Exception as e:
            print(f"⚠ Failed to end MLflow run: {e}")
    
    def is_available(self) -> bool:
        """Check if MLflow tracking is available and working."""
        return self.enable_tracking and self.server_available
    
    @staticmethod
    def load_model_from_mlflow(model_name: str, 
                              version: str = "latest",
                              tracking_uri: str = "http://localhost:5000"):
        """
        Load a model from MLflow model registry.
        
        Parameters:
        -----------
        model_name : str
            Name of the registered model
        version : str
            Version to load ("latest", "1", "2", etc.)
        tracking_uri : str
            MLflow tracking server URI
            
        Returns:
        --------
        object : Loaded model
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow not available. Install with: pip install mlflow")
        
        try:
            mlflow.set_tracking_uri(tracking_uri)
            model_uri = f"models:/{model_name}/{version}"
            
            # Try different loading methods
            try:
                return mlflow.sklearn.load_model(model_uri)
            except Exception:
                try:
                    return mlflow.keras.load_model(model_uri)
                except Exception:
                    return mlflow.pyfunc.load_model(model_uri)
                    
        except Exception as e:
            print(f"✗ Failed to load model {model_name}: {e}")
            return None


# Convenience functions
def create_tracker(experiment_name: str = "ml_predict_15/crypto_prediction",
                  tracking_uri: str = "http://localhost:5000",
                  enable_tracking: bool = True) -> MLflowTracker:
    """Create and return an MLflow tracker instance."""
    return MLflowTracker(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        enable_tracking=enable_tracking
    )


def log_experiment(models: Dict[str, Any],
                  results: Dict[str, Dict[str, float]],
                  best_model_name: str,
                  training_config: Dict[str, Any],
                  experiment_name: str = "ml_predict_15/crypto_prediction",
                  tracking_uri: str = "http://localhost:5000") -> Optional[str]:
    """
    Convenience function to log a complete experiment.
    
    Returns:
    --------
    str : Run ID if successful, None otherwise
    """
    tracker = create_tracker(experiment_name, tracking_uri)
    
    if not tracker.is_available():
        return None
    
    run_id = tracker.start_run()
    if run_id:
        tracker.log_training_results(models, results, best_model_name, training_config)
        
        # Log neural network specific info
        for model_name, model in models.items():
            if hasattr(model, 'model') or 'neural' in model_name.lower():
                tracker.log_neural_network_info(model, model_name)
        
        tracker.end_run()
    
    return run_id


# Example usage
if __name__ == "__main__":
    print("Testing MLflow Tracker...")
    
    # Create tracker
    tracker = create_tracker()
    
    if tracker.is_available():
        print("✓ MLflow tracker available")
        
        # Test logging
        run_id = tracker.start_run("test_run")
        if run_id:
            tracker.log_params({"test_param": "test_value"})
            tracker.log_metrics({"test_metric": 0.95})
            tracker.end_run()
            print("✓ Test logging completed")
    else:
        print("⚠ MLflow tracker not available")
        print("Start MLflow server: mlflow server --host 127.0.0.1 --port 5000")
