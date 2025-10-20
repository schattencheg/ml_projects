"""
Configuration file for the Data Provider Server
"""
import os

# Server configuration
HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 5001  # Port for the data server (5000 is used by MLflow)

# Data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Default settings
DEFAULT_RESOLUTION = '1d'
DEFAULT_PERIOD = 'max'

# Supported instruments
CRYPTO_INSTRUMENTS = [
    'BTC-USD', 'BTC-USDT', 'ETH-USD', 'ETH-USDT',
    'DOGE-USDT', 'ADA-USD', 'SOL-USD', 'XRP-USD'
]

REGULAR_MARKET_INSTRUMENTS = [
    'SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN',
    'TSLA', 'NVDA', 'META', 'NFLX', '6B', '6E'
]

# Cache settings
CACHE_ENABLED = True
CACHE_EXPIRY_MINUTES = 60

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = 'data_server.log'

# MLflow configuration
MLFLOW_ENABLED = True  # Set to False to disable MLflow server startup
MLFLOW_HOST = '127.0.0.1'
MLFLOW_PORT = 5000

# Convert paths to file:// URIs for MLflow compatibility
_mlruns_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'mlruns'))
_mlartifacts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'mlartifacts'))

# Use file:// URI format (required by MLflow)
MLFLOW_BACKEND_STORE_URI = f"file:///{_mlruns_path.replace(os.sep, '/')}"
MLFLOW_DEFAULT_ARTIFACT_ROOT = f"file:///{_mlartifacts_path.replace(os.sep, '/')}"
