"""Configuration management utilities."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration manager for the OHLC prediction project."""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'data.sources.yahoo_finance')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    @property
    def data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.get('data', {})
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.get('models', {})
    
    @property
    def training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.get('training', {})
    
    @property
    def mlflow_config(self) -> Dict[str, Any]:
        """Get MLflow configuration."""
        return self.get('mlflow', {})
    
    @property
    def feature_config(self) -> Dict[str, Any]:
        """Get feature engineering configuration."""
        return self.get('features', {})

# Environment variables
class EnvConfig:
    """Environment variable configuration."""
    
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')
    MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'ohlc_prediction')
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///mlflow.db')
    DEFAULT_SYMBOL = os.getenv('DEFAULT_SYMBOL', 'AAPL')
    DEFAULT_PERIOD = os.getenv('DEFAULT_PERIOD', '2y')
    PREDICTION_HORIZON = int(os.getenv('PREDICTION_HORIZON', '5'))
    API_HOST = os.getenv('API_HOST', '127.0.0.1')
    API_PORT = int(os.getenv('API_PORT', '8000'))

# Global configuration instance
config = Config()
