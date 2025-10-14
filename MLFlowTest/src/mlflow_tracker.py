"""MLflow experiment tracking for OHLC prediction models."""

import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Union
import tempfile
import os
from pathlib import Path

from src.utils import config, get_logger, EnvConfig
from src.models.base_model import BaseModel

logger = get_logger(__name__)

class MLflowTracker:
    """MLflow experiment tracking manager."""
    
    def __init__(self, experiment_name: Optional[str] = None):
        """Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
        """
        self.experiment_name = experiment_name or EnvConfig.MLFLOW_EXPERIMENT_NAME
        self.tracking_uri = EnvConfig.MLFLOW_TRACKING_URI
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Set or create experiment
        try:
            self.experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if self.experiment is None:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created new MLflow experiment: {self.experiment_name}")
            else:
                self.experiment_id = self.experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {self.experiment_name}")
        except Exception as e:
            logger.error(f"Error setting up MLflow experiment: {str(e)}")
            raise
        
        mlflow.set_experiment(self.experiment_name)
        
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> str:
        """Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            tags: Optional tags for the run
            
        Returns:
            Run ID
        """
        run = mlflow.start_run(run_name=run_name, tags=tags)
        logger.info(f"Started MLflow run: {run.info.run_id}")
        return run.info.run_id
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameters to log
        """
        try:
            for key, value in params.items():
                # Convert complex types to strings
                if isinstance(value, (list, dict, tuple)):
                    value = str(value)
                mlflow.log_param(key, value)
        except Exception as e:
            logger.error(f"Error logging parameters: {str(e)}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        try:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    mlflow.log_metric(key, value, step=step)
        except Exception as e:
            logger.error(f"Error logging metrics: {str(e)}")
    
    def log_model(self, model: BaseModel, model_name: str = "model") -> None:
        """Log model to MLflow.
        
        Args:
            model: Trained model to log
            model_name: Name for the logged model
        """
        try:
            # Create temporary file to save model
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
                model.save_model(tmp_file.name)
                
                # Log as artifact
                mlflow.log_artifact(tmp_file.name, f"models/{model_name}.pkl")
                
                # Clean up temporary file
                os.unlink(tmp_file.name)
            
            # Log model info as parameters
            model_info = model.get_model_info()
            self.log_params({f"model_{k}": v for k, v in model_info.items()})
            
            logger.info(f"Model {model_name} logged to MLflow")
            
        except Exception as e:
            logger.error(f"Error logging model: {str(e)}")
    
    def log_sklearn_model(self, model, model_name: str = "sklearn_model") -> None:
        """Log scikit-learn model to MLflow.
        
        Args:
            model: Scikit-learn model
            model_name: Name for the logged model
        """
        try:
            mlflow.sklearn.log_model(model, model_name)
            logger.info(f"Sklearn model {model_name} logged to MLflow")
        except Exception as e:
            logger.error(f"Error logging sklearn model: {str(e)}")
    
    def log_tensorflow_model(self, model, model_name: str = "tensorflow_model") -> None:
        """Log TensorFlow model to MLflow.
        
        Args:
            model: TensorFlow/Keras model
            model_name: Name for the logged model
        """
        try:
            mlflow.tensorflow.log_model(model, model_name)
            logger.info(f"TensorFlow model {model_name} logged to MLflow")
        except Exception as e:
            logger.error(f"Error logging tensorflow model: {str(e)}")
    
    def log_predictions_plot(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        title: str = "Predictions vs Actual",
        target_name: str = "Close"
    ) -> None:
        """Log predictions vs actual plot.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            target_name: Name of target variable
        """
        try:
            plt.figure(figsize=(10, 6))
            
            # Handle multi-output case
            if len(y_true.shape) > 1 and y_true.shape[1] > 1:
                # Plot first target only
                y_true_plot = y_true[:, 0]
                y_pred_plot = y_pred[:, 0]
            else:
                y_true_plot = y_true.flatten()
                y_pred_plot = y_pred.flatten()
            
            # Scatter plot
            plt.scatter(y_true_plot, y_pred_plot, alpha=0.6)
            
            # Perfect prediction line
            min_val = min(y_true_plot.min(), y_pred_plot.min())
            max_val = max(y_true_plot.max(), y_pred_plot.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            plt.xlabel(f'Actual {target_name}')
            plt.ylabel(f'Predicted {target_name}')
            plt.title(title)
            plt.grid(True, alpha=0.3)
            
            # Save and log plot
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                plt.savefig(tmp_file.name, dpi=150, bbox_inches='tight')
                mlflow.log_artifact(tmp_file.name, "plots/predictions_vs_actual.png")
                os.unlink(tmp_file.name)
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error logging predictions plot: {str(e)}")
    
    def log_residuals_plot(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        title: str = "Residuals Plot"
    ) -> None:
        """Log residuals plot.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
        """
        try:
            plt.figure(figsize=(10, 6))
            
            # Handle multi-output case
            if len(y_true.shape) > 1 and y_true.shape[1] > 1:
                y_true_plot = y_true[:, 0]
                y_pred_plot = y_pred[:, 0]
            else:
                y_true_plot = y_true.flatten()
                y_pred_plot = y_pred.flatten()
            
            residuals = y_true_plot - y_pred_plot
            
            plt.scatter(y_pred_plot, residuals, alpha=0.6)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title(title)
            plt.grid(True, alpha=0.3)
            
            # Save and log plot
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                plt.savefig(tmp_file.name, dpi=150, bbox_inches='tight')
                mlflow.log_artifact(tmp_file.name, "plots/residuals.png")
                os.unlink(tmp_file.name)
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error logging residuals plot: {str(e)}")
    
    def log_feature_importance_plot(
        self, 
        feature_importance: Dict[str, float], 
        top_n: int = 20,
        title: str = "Feature Importance"
    ) -> None:
        """Log feature importance plot.
        
        Args:
            feature_importance: Dictionary of feature importances
            top_n: Number of top features to plot
            title: Plot title
        """
        try:
            if not feature_importance:
                return
            
            # Get top N features
            top_features = dict(list(feature_importance.items())[:top_n])
            
            plt.figure(figsize=(12, 8))
            features = list(top_features.keys())
            importances = list(top_features.values())
            
            # Create horizontal bar plot
            y_pos = np.arange(len(features))
            plt.barh(y_pos, importances)
            plt.yticks(y_pos, features)
            plt.xlabel('Importance')
            plt.title(title)
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            # Save and log plot
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                plt.savefig(tmp_file.name, dpi=150, bbox_inches='tight')
                mlflow.log_artifact(tmp_file.name, "plots/feature_importance.png")
                os.unlink(tmp_file.name)
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error logging feature importance plot: {str(e)}")
    
    def log_training_history(self, history: Dict[str, List[float]]) -> None:
        """Log training history metrics.
        
        Args:
            history: Training history dictionary
        """
        try:
            for metric_name, values in history.items():
                for step, value in enumerate(values):
                    mlflow.log_metric(metric_name, value, step=step)
        except Exception as e:
            logger.error(f"Error logging training history: {str(e)}")
    
    def log_time_series_plot(
        self, 
        dates: pd.DatetimeIndex, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        title: str = "Time Series Predictions",
        target_name: str = "Close"
    ) -> None:
        """Log time series predictions plot.
        
        Args:
            dates: Date index
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            target_name: Name of target variable
        """
        try:
            plt.figure(figsize=(15, 8))
            
            # Handle multi-output case
            if len(y_true.shape) > 1 and y_true.shape[1] > 1:
                y_true_plot = y_true[:, 0]
                y_pred_plot = y_pred[:, 0]
            else:
                y_true_plot = y_true.flatten()
                y_pred_plot = y_pred.flatten()
            
            plt.plot(dates, y_true_plot, label='Actual', alpha=0.8)
            plt.plot(dates, y_pred_plot, label='Predicted', alpha=0.8)
            
            plt.xlabel('Date')
            plt.ylabel(target_name)
            plt.title(title)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save and log plot
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                plt.savefig(tmp_file.name, dpi=150, bbox_inches='tight')
                mlflow.log_artifact(tmp_file.name, "plots/time_series_predictions.png")
                os.unlink(tmp_file.name)
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error logging time series plot: {str(e)}")
    
    def log_data_info(self, data: pd.DataFrame, prefix: str = "data") -> None:
        """Log dataset information.
        
        Args:
            data: Dataset
            prefix: Prefix for parameter names
        """
        try:
            info = {
                f"{prefix}_shape": str(data.shape),
                f"{prefix}_columns": str(list(data.columns)),
                f"{prefix}_dtypes": str(data.dtypes.to_dict()),
                f"{prefix}_missing_values": data.isnull().sum().sum(),
                f"{prefix}_start_date": str(data.index.min()) if isinstance(data.index, pd.DatetimeIndex) else "N/A",
                f"{prefix}_end_date": str(data.index.max()) if isinstance(data.index, pd.DatetimeIndex) else "N/A"
            }
            self.log_params(info)
        except Exception as e:
            logger.error(f"Error logging data info: {str(e)}")
    
    def log_artifact_from_file(self, filepath: str, artifact_path: Optional[str] = None) -> None:
        """Log file as artifact.
        
        Args:
            filepath: Path to file to log
            artifact_path: Optional path within artifacts directory
        """
        try:
            mlflow.log_artifact(filepath, artifact_path)
            logger.info(f"Logged artifact: {filepath}")
        except Exception as e:
            logger.error(f"Error logging artifact: {str(e)}")
    
    def log_text(self, text: str, filename: str) -> None:
        """Log text content as artifact.
        
        Args:
            text: Text content to log
            filename: Name of the file
        """
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
                tmp_file.write(text)
                tmp_file.flush()
                mlflow.log_artifact(tmp_file.name, f"text/{filename}")
                os.unlink(tmp_file.name)
        except Exception as e:
            logger.error(f"Error logging text: {str(e)}")
    
    def end_run(self) -> None:
        """End the current MLflow run."""
        try:
            mlflow.end_run()
            logger.info("MLflow run ended")
        except Exception as e:
            logger.error(f"Error ending MLflow run: {str(e)}")
    
    def get_experiment_runs(self) -> pd.DataFrame:
        """Get all runs from the current experiment.
        
        Returns:
            DataFrame with run information
        """
        try:
            runs = mlflow.search_runs(experiment_ids=[self.experiment_id])
            return runs
        except Exception as e:
            logger.error(f"Error getting experiment runs: {str(e)}")
            return pd.DataFrame()
    
    def get_best_run(self, metric_name: str, ascending: bool = False) -> Optional[Dict[str, Any]]:
        """Get the best run based on a metric.
        
        Args:
            metric_name: Name of metric to optimize
            ascending: Whether to sort in ascending order
            
        Returns:
            Best run information or None
        """
        try:
            runs = self.get_experiment_runs()
            if runs.empty:
                return None
            
            # Sort by metric
            metric_col = f"metrics.{metric_name}"
            if metric_col in runs.columns:
                best_run = runs.sort_values(metric_col, ascending=ascending).iloc[0]
                return best_run.to_dict()
            else:
                logger.warning(f"Metric {metric_name} not found in runs")
                return None
                
        except Exception as e:
            logger.error(f"Error getting best run: {str(e)}")
            return None
    
    def compare_runs(self, run_ids: List[str], metrics: List[str]) -> pd.DataFrame:
        """Compare multiple runs on specified metrics.
        
        Args:
            run_ids: List of run IDs to compare
            metrics: List of metrics to compare
            
        Returns:
            DataFrame with comparison results
        """
        try:
            runs = self.get_experiment_runs()
            if runs.empty:
                return pd.DataFrame()
            
            # Filter runs
            comparison_runs = runs[runs['run_id'].isin(run_ids)]
            
            # Select relevant columns
            columns = ['run_id', 'start_time'] + [f'metrics.{m}' for m in metrics]
            available_columns = [col for col in columns if col in comparison_runs.columns]
            
            return comparison_runs[available_columns]
            
        except Exception as e:
            logger.error(f"Error comparing runs: {str(e)}")
            return pd.DataFrame()

def track_experiment(
    model: BaseModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None
) -> str:
    """Convenience function to track a complete experiment.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        run_name: Optional run name
        tags: Optional tags
        
    Returns:
        Run ID
    """
    tracker = MLflowTracker()
    
    try:
        # Start run
        run_id = tracker.start_run(run_name=run_name, tags=tags)
        
        # Log model info
        tracker.log_params(model.get_model_info())
        
        # Evaluate and log metrics
        metrics = model.evaluate(X_test, y_test)
        tracker.log_metrics(metrics)
        
        # Make predictions for plots
        y_pred = model.predict(X_test)
        
        # Log plots
        tracker.log_predictions_plot(y_test, y_pred)
        tracker.log_residuals_plot(y_test, y_pred)
        
        # Log feature importance if available
        if hasattr(model, 'get_feature_importance'):
            try:
                importance = model.get_feature_importance()
                if importance:
                    tracker.log_feature_importance_plot(importance)
            except:
                pass
        
        # Log model
        tracker.log_model(model)
        
        return run_id
        
    finally:
        tracker.end_run()
