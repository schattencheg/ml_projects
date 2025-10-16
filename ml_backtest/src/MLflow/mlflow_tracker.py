"""
MLflow Tracking Utility for ML Backtesting Projects

This module provides structured MLflow tracking with organized naming conventions
for different projects, models, and experiments.

Naming Convention:
    Project: {project_name}
    Experiment: {project_name}/{model_type}/{timeframe}
    Run: {model_name}_{version}_{timestamp}
"""

import mlflow
import mlflow.sklearn
from datetime import datetime
from typing import Dict, Any, Optional
import os
import json


class MLflowTracker:
    """
    Centralized MLflow tracking with structured naming conventions.
    
    Naming Structure:
        - Project: ml_backtest, ml_forecast, ml_classification, etc.
        - Experiment: {project}/{model_type}/{asset_or_timeframe}
        - Run: {model_name}_v{version}_{timestamp}
    
    Example:
        Project: ml_backtest
        Experiment: ml_backtest/regression/btc_daily
        Run: linear_regression_v1_20251016_115430
    """
    
    def __init__(self, 
                 tracking_uri: str = "http://localhost:5000",
                 project_name: str = "ml_backtest"):
        """
        Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow server URI
            project_name: Name of the project (e.g., 'ml_backtest', 'ml_forecast')
        """
        self.tracking_uri = tracking_uri
        self.project_name = project_name
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        print(f"✓ MLflow tracking initialized")
        print(f"  Server: {tracking_uri}")
        print(f"  Project: {project_name}")
    
    def create_experiment_name(self, 
                              model_type: str,
                              asset_or_timeframe: str,
                              additional_tags: Optional[str] = None) -> str:
        """
        Create a structured experiment name.
        
        Args:
            model_type: Type of model (e.g., 'regression', 'classification', 'lstm')
            asset_or_timeframe: Asset name or timeframe (e.g., 'btc_daily', 'spy_hourly')
            additional_tags: Optional additional tags
        
        Returns:
            Structured experiment name
        
        Example:
            >>> tracker.create_experiment_name('regression', 'btc_daily')
            'ml_backtest/regression/btc_daily'
        """
        parts = [self.project_name, model_type, asset_or_timeframe]
        if additional_tags:
            parts.append(additional_tags)
        return "/".join(parts)
    
    def create_run_name(self,
                       model_name: str,
                       version: str = "1",
                       include_timestamp: bool = True) -> str:
        """
        Create a structured run name.
        
        Args:
            model_name: Name of the model (e.g., 'linear_regression', 'random_forest')
            version: Version number
            include_timestamp: Whether to include timestamp
        
        Returns:
            Structured run name
        
        Example:
            >>> tracker.create_run_name('linear_regression', '1')
            'linear_regression_v1_20251016_115430'
        """
        parts = [model_name, f"v{version}"]
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            parts.append(timestamp)
        return "_".join(parts)
    
    def start_run(self,
                  model_type: str,
                  asset_or_timeframe: str,
                  model_name: str,
                  version: str = "1",
                  description: Optional[str] = None,
                  tags: Optional[Dict[str, str]] = None) -> mlflow.ActiveRun:
        """
        Start a new MLflow run with structured naming.
        
        Args:
            model_type: Type of model
            asset_or_timeframe: Asset or timeframe
            model_name: Model name
            version: Version number
            description: Run description
            tags: Additional tags
        
        Returns:
            Active MLflow run
        
        Example:
            >>> with tracker.start_run('regression', 'btc_daily', 'linear_regression'):
            ...     mlflow.log_param('alpha', 0.1)
        """
        # Create experiment name
        experiment_name = self.create_experiment_name(model_type, asset_or_timeframe)
        
        # Set or create experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(experiment_name)
        
        # Create run name
        run_name = self.create_run_name(model_name, version)
        
        # Prepare tags
        run_tags = {
            "project": self.project_name,
            "model_type": model_type,
            "asset_or_timeframe": asset_or_timeframe,
            "model_name": model_name,
            "version": version
        }
        
        if description:
            run_tags["description"] = description
        
        if tags:
            run_tags.update(tags)
        
        # Start run
        run = mlflow.start_run(run_name=run_name, tags=run_tags)
        
        print(f"\n{'='*60}")
        print(f"MLflow Run Started")
        print(f"{'='*60}")
        print(f"Experiment: {experiment_name}")
        print(f"Run Name: {run_name}")
        print(f"Run ID: {run.info.run_id}")
        print(f"{'='*60}\n")
        
        return run
    
    def log_model_params(self, params: Dict[str, Any]):
        """Log model parameters."""
        for key, value in params.items():
            mlflow.log_param(key, value)
        print(f"✓ Logged {len(params)} parameters")
    
    def log_model_metrics(self, metrics: Dict[str, float]):
        """Log model metrics."""
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        print(f"✓ Logged {len(metrics)} metrics")
    
    def log_backtest_results(self, stats: Any):
        """
        Log backtesting results to MLflow.
        
        Args:
            stats: Backtest statistics object from backtesting.py
        """
        # Extract key metrics
        metrics = {
            "return_pct": float(stats['Return [%]']),
            "sharpe_ratio": float(stats['Sharpe Ratio']) if stats['Sharpe Ratio'] == stats['Sharpe Ratio'] else 0.0,
            "sortino_ratio": float(stats['Sortino Ratio']) if stats['Sortino Ratio'] == stats['Sortino Ratio'] else 0.0,
            "max_drawdown_pct": float(stats['Max. Drawdown [%]']),
            "win_rate_pct": float(stats['Win Rate [%]']) if stats['Win Rate [%]'] == stats['Win Rate [%]'] else 0.0,
            "num_trades": int(stats['# Trades']),
            "profit_factor": float(stats['Profit Factor']) if stats['Profit Factor'] == stats['Profit Factor'] else 0.0,
            "exposure_time_pct": float(stats['Exposure Time [%]']),
            "equity_final": float(stats['Equity Final [$]']),
        }
        
        self.log_model_metrics(metrics)
        
        # Log additional stats as params
        params = {
            "duration_days": str(stats['Duration']),
            "start_date": str(stats['Start']),
            "end_date": str(stats['End']),
        }
        
        self.log_model_params(params)
    
    def log_strategy_params(self, strategy_instance: Any):
        """
        Log strategy parameters from a backtesting strategy instance.
        
        Args:
            strategy_instance: Instance of backtesting Strategy class
        """
        params = {}
        for attr in dir(strategy_instance):
            if not attr.startswith('_') and not callable(getattr(strategy_instance, attr)):
                value = getattr(strategy_instance, attr)
                if isinstance(value, (int, float, str, bool)):
                    params[attr] = value
        
        self.log_model_params(params)
    
    def log_artifact_from_file(self, file_path: str, artifact_path: Optional[str] = None):
        """
        Log a file as an artifact.
        
        Args:
            file_path: Path to the file
            artifact_path: Optional subdirectory in artifact store
        """
        mlflow.log_artifact(file_path, artifact_path)
        print(f"✓ Logged artifact: {file_path}")
    
    def log_dict_as_json(self, data: Dict, filename: str):
        """
        Log a dictionary as a JSON artifact.
        
        Args:
            data: Dictionary to log
            filename: Name of the JSON file
        """
        temp_file = f"temp_{filename}"
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        mlflow.log_artifact(temp_file)
        os.remove(temp_file)
        print(f"✓ Logged JSON artifact: {filename}")
    
    def log_sklearn_model(self, model: Any, model_name: str = "model"):
        """
        Log a scikit-learn model.
        
        Args:
            model: Trained scikit-learn model
            model_name: Name for the model artifact
        """
        mlflow.sklearn.log_model(model, model_name)
        print(f"✓ Logged sklearn model: {model_name}")
    
    def end_run(self, status: str = "FINISHED"):
        """
        End the current MLflow run.
        
        Args:
            status: Run status ('FINISHED', 'FAILED', 'KILLED')
        """
        mlflow.end_run(status=status)
        print(f"\n✓ MLflow run ended with status: {status}\n")
    
    @staticmethod
    def get_best_run(experiment_name: str, metric: str = "return_pct", ascending: bool = False):
        """
        Get the best run from an experiment based on a metric.
        
        Args:
            experiment_name: Name of the experiment
            metric: Metric to optimize
            ascending: If True, lower is better
        
        Returns:
            Best run information
        """
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found")
            return None
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"]
        )
        
        if len(runs) == 0:
            print(f"No runs found in experiment '{experiment_name}'")
            return None
        
        best_run = runs.iloc[0]
        print(f"\nBest run for metric '{metric}':")
        print(f"  Run ID: {best_run['run_id']}")
        print(f"  {metric}: {best_run[f'metrics.{metric}']}")
        
        return best_run


# Convenience function for quick setup
def setup_mlflow_tracker(tracking_uri: str = "http://localhost:5000",
                        project_name: str = "ml_backtest") -> MLflowTracker:
    """
    Quick setup for MLflow tracker.
    
    Args:
        tracking_uri: MLflow server URI
        project_name: Project name
    
    Returns:
        Configured MLflowTracker instance
    """
    return MLflowTracker(tracking_uri, project_name)


if __name__ == "__main__":
    # Example usage
    tracker = setup_mlflow_tracker()
    
    # Example: Start a run
    with tracker.start_run(
        model_type="regression",
        asset_or_timeframe="btc_daily",
        model_name="linear_regression",
        version="1",
        description="Testing MLflow integration"
    ):
        # Log some example metrics
        tracker.log_model_metrics({
            "rmse": 0.05,
            "mae": 0.03,
            "r2": 0.85
        })
        
        # Log some parameters
        tracker.log_model_params({
            "learning_rate": 0.01,
            "epochs": 100
        })
    
    print("Example run completed!")
