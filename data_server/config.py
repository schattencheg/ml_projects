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
