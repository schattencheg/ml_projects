"""Setup script for MLflow server and initial configuration."""

import os
import sys
import subprocess
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import EnvConfig, get_logger

logger = get_logger(__name__)

def setup_mlflow_directories():
    """Create necessary directories for MLflow."""
    directories = [
        "mlruns",
        "models", 
        "data",
        "logs",
        "artifacts"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")

def start_mlflow_server():
    """Start MLflow tracking server."""
    try:
        # Check if MLflow server is already running
        import requests
        response = requests.get(EnvConfig.MLFLOW_TRACKING_URI, timeout=5)
        if response.status_code == 200:
            logger.info("MLflow server is already running")
            return True
    except:
        pass
    
    # Start MLflow server
    cmd = [
        "mlflow", "server",
        "--host", "127.0.0.1",
        "--port", "5000",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--default-artifact-root", "./mlruns"
    ]
    
    logger.info("Starting MLflow server...")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        # Start server in background
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path.cwd()
        )
        
        # Wait a bit for server to start
        time.sleep(5)
        
        # Check if server is running
        import requests
        response = requests.get(EnvConfig.MLFLOW_TRACKING_URI, timeout=10)
        if response.status_code == 200:
            logger.info(f"MLflow server started successfully at {EnvConfig.MLFLOW_TRACKING_URI}")
            return True
        else:
            logger.error("MLflow server failed to start properly")
            return False
            
    except Exception as e:
        logger.error(f"Error starting MLflow server: {str(e)}")
        return False

def create_sample_experiment():
    """Create a sample experiment in MLflow."""
    try:
        import mlflow
        
        mlflow.set_tracking_uri(EnvConfig.MLFLOW_TRACKING_URI)
        
        # Create experiment
        experiment_name = "ohlc_prediction_demo"
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created sample experiment: {experiment_name}")
        except:
            # Experiment might already exist
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {experiment_name}")
        
        # Log a sample run
        with mlflow.start_run(experiment_id=experiment_id):
            mlflow.log_param("model_type", "demo")
            mlflow.log_param("symbol", "AAPL")
            mlflow.log_metric("accuracy", 0.85)
            mlflow.log_metric("mse", 0.15)
            
        logger.info("Sample experiment run logged successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error creating sample experiment: {str(e)}")
        return False

def main():
    """Main setup function."""
    logger.info("Setting up MLflow environment...")
    
    # Create directories
    setup_mlflow_directories()
    
    # Start MLflow server
    if start_mlflow_server():
        logger.info("MLflow server setup completed")
        
        # Create sample experiment
        time.sleep(2)  # Wait for server to be fully ready
        create_sample_experiment()
        
        print("\n" + "="*50)
        print("MLFLOW SETUP COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"MLflow UI: {EnvConfig.MLFLOW_TRACKING_URI}")
        print("You can now:")
        print("1. Open the MLflow UI in your browser")
        print("2. Run training scripts to log experiments")
        print("3. Compare model performance")
        print("="*50)
        
    else:
        logger.error("MLflow setup failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
