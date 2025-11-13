"""
MLflow Tracker for CNN Optimizer

Standalone class to handle all MLflow tracking operations including:
- Experiment setup and configuration
- Parent and nested run management
- Parameter and metric logging
- Model and artifact logging
- Error handling and graceful fallback
"""

import mlflow
import mlflow.keras
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


class MLflowTracker:
    """
    Handles all MLflow tracking operations for CNN optimization.
    """
    
    def __init__(self, tracking_uri='http://localhost:5000', experiment_name='cnn_ml', enabled=True):
        """
        Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the MLflow experiment
            enabled: Whether MLflow tracking is enabled
        """
        self.enabled = enabled
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.parent_run = None
        
        if self.enabled:
            try:
                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(experiment_name)
                print(f"✓ MLflow tracking enabled: {tracking_uri}")
                print(f"✓ Experiment: {experiment_name}")
            except Exception as e:
                print(f"✗ MLflow initialization failed: {str(e)}")
                print("  Continuing without MLflow tracking")
                self.enabled = False
    
    def start_optimization_run(self, n_trials, training_samples, validation_samples, 
                               sequence_length, num_features):
        """
        Start parent MLflow run for optimization.
        
        Args:
            n_trials: Number of optimization trials
            training_samples: Number of training samples
            validation_samples: Number of validation samples
            sequence_length: Input sequence length
            num_features: Number of features
            
        Returns:
            bool: True if run started successfully
        """
        if not self.enabled:
            return False
        
        try:
            run_name = f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.parent_run = mlflow.start_run(run_name=run_name)
            
            # Log parameters
            mlflow.log_param('n_trials', n_trials)
            mlflow.log_param('training_samples', training_samples)
            mlflow.log_param('validation_samples', validation_samples)
            mlflow.log_param('sequence_length', sequence_length)
            mlflow.log_param('num_features', num_features)
            
            # Set tags
            mlflow.set_tag('optimization_type', 'cnn_hyperparameter')
            mlflow.set_tag('optimization_metric', 'recall')
            
            return True
        except Exception as e:
            print(f"✗ Failed to start optimization run: {str(e)}")
            self.enabled = False
            return False
    
    def log_trial(self, trial_number, architecture, params, metrics):
        """
        Log a single trial as a nested run.
        
        Args:
            trial_number: Trial number
            architecture: Architecture type
            params: Dictionary of hyperparameters
            metrics: Dictionary of metrics (accuracy, precision, recall, f1_score)
            
        Returns:
            bool: True if logged successfully
        """
        if not self.enabled:
            return False
        
        try:
            with mlflow.start_run(run_name=f"trial_{trial_number}", nested=True):
                # Log parameters
                mlflow.log_param('trial_number', trial_number)
                mlflow.log_param('architecture', architecture)
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
                
                # Log metrics
                mlflow.log_metric('accuracy', float(metrics.get('accuracy', 0)))
                mlflow.log_metric('precision', float(metrics.get('precision', 0)))
                mlflow.log_metric('recall', float(metrics.get('recall', 0)))
                mlflow.log_metric('f1_score', float(metrics.get('f1_score', 0)))
                
                # Set tags
                mlflow.set_tag('optimization_metric', 'recall')
                mlflow.set_tag('trial_status', 'completed')
            
            return True
        except Exception as e:
            print(f"✗ Failed to log trial {trial_number}: {str(e)}")
            return False
    
    def log_best_results(self, study):
        """
        Log best results to parent run.
        
        Args:
            study: Optuna study object
            
        Returns:
            bool: True if logged successfully
        """
        if not self.enabled or self.parent_run is None:
            return False
        
        try:
            mlflow.log_metric('best_recall', float(study.best_value))
            mlflow.log_param('best_trial', study.best_trial.number)
            mlflow.log_param('best_architecture', study.best_params['architecture'])
            
            # Log all best parameters
            for key, value in study.best_params.items():
                mlflow.log_param(f'best_{key}', value)
            
            return True
        except Exception as e:
            print(f"✗ Failed to log best results: {str(e)}")
            return False
    
    def end_optimization_run(self):
        """
        End parent optimization run.
        """
        if self.enabled and self.parent_run is not None:
            try:
                mlflow.end_run()
                self.parent_run = None
            except Exception as e:
                print(f"✗ Failed to end optimization run: {str(e)}")
    
    def log_best_model(self, model, model_name, timestamp, architecture, best_trial, 
                       total_trials, sequence_length, num_features, best_params,
                       X_val, y_val, artifact_paths):
        """
        Log best model with all artifacts to MLflow.
        
        Args:
            model: Trained Keras model
            model_name: Model name
            timestamp: Timestamp string
            architecture: Architecture type
            best_trial: Best trial number
            total_trials: Total number of trials
            sequence_length: Input sequence length
            num_features: Number of features
            best_params: Dictionary of best hyperparameters
            X_val: Validation features
            y_val: Validation labels
            artifact_paths: Dictionary of artifact file paths
            
        Returns:
            tuple: (success: bool, run_id: str or None)
        """
        if not self.enabled:
            return False, None
        
        try:
            run_name = f"best_model_{timestamp}"
            with mlflow.start_run(run_name=run_name):
                # Log parameters
                mlflow.log_param('model_name', model_name)
                mlflow.log_param('timestamp', timestamp)
                mlflow.log_param('architecture', architecture)
                mlflow.log_param('best_trial', best_trial)
                mlflow.log_param('total_trials', total_trials)
                mlflow.log_param('sequence_length', sequence_length)
                mlflow.log_param('num_features', num_features)
                
                # Log all hyperparameters
                for key, value in best_params.items():
                    mlflow.log_param(key, value)
                
                # Calculate and log validation metrics
                val_pred = model.predict(X_val, verbose=0)
                val_pred_binary = (val_pred > 0.5).astype(int).flatten()
                
                val_accuracy = accuracy_score(y_val, val_pred_binary)
                val_precision = precision_score(y_val, val_pred_binary, zero_division=0)
                val_recall = recall_score(y_val, val_pred_binary, zero_division=0)
                val_f1 = f1_score(y_val, val_pred_binary, zero_division=0)
                
                mlflow.log_metric('val_accuracy', float(val_accuracy))
                mlflow.log_metric('val_precision', float(val_precision))
                mlflow.log_metric('val_recall', float(val_recall))
                mlflow.log_metric('val_f1_score', float(val_f1))
                
                # Log best recall from optimization
                if 'best_recall' in best_params:
                    mlflow.log_metric('best_recall', float(best_params['best_recall']))
                
                # Log model to registry
                registry_name = f"cnn_ml_{model_name}"
                mlflow.keras.log_model(model, "model", registered_model_name=registry_name)
                print(f"✓ Model logged to MLflow registry: {registry_name}")
                
                # Log artifacts
                if 'model_path' in artifact_paths:
                    mlflow.log_artifact(artifact_paths['model_path'], "model_files")
                if 'arch_path' in artifact_paths:
                    mlflow.log_artifact(artifact_paths['arch_path'], "model_files")
                if 'metadata_path' in artifact_paths:
                    mlflow.log_artifact(artifact_paths['metadata_path'], "metadata")
                if 'history_path' in artifact_paths:
                    mlflow.log_artifact(artifact_paths['history_path'], "data")
                if 'training_history_path' in artifact_paths:
                    mlflow.log_artifact(artifact_paths['training_history_path'], "data")
                if 'opt_plot_path' in artifact_paths:
                    mlflow.log_artifact(artifact_paths['opt_plot_path'], "plots")
                if 'training_plot_path' in artifact_paths:
                    mlflow.log_artifact(artifact_paths['training_plot_path'], "plots")
                if 'summary_path' in artifact_paths:
                    mlflow.log_artifact(artifact_paths['summary_path'], "documentation")
                if 'readme_path' in artifact_paths:
                    mlflow.log_artifact(artifact_paths['readme_path'], "documentation")
                
                print("✓ All artifacts logged to MLflow")
                
                # Set tags
                mlflow.set_tag('optimization_metric', 'recall')
                mlflow.set_tag('model_type', 'cnn')
                mlflow.set_tag('framework', 'tensorflow')
                if 'save_directory' in artifact_paths:
                    mlflow.set_tag('save_directory', artifact_paths['save_directory'])
                
                run_id = mlflow.active_run().info.run_id
                print(f"✓ MLflow Run ID: {run_id}")
                
                return True, run_id
                
        except Exception as e:
            print(f"✗ MLflow logging failed: {str(e)}")
            print("  Model still saved locally")
            return False, None
    
    def get_tracking_uri(self):
        """
        Get MLflow tracking URI.
        
        Returns:
            str: Tracking URI
        """
        if self.enabled:
            try:
                return mlflow.get_tracking_uri()
            except:
                return self.tracking_uri
        return None
    
    def is_enabled(self):
        """
        Check if MLflow tracking is enabled.
        
        Returns:
            bool: True if enabled
        """
        return self.enabled
