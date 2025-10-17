"""
Data Provider Server - REST API for OHLC data
Provides crypto and regular market data for different instruments and timeframes
"""
import os
import sys
if 'real_prefix' not in sys.__dict__:
    #sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    #sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'venv', 'Lib', 'site-packages'))
    sys.real_prefix = sys.prefix    

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.Data.DataProvider import DataProvider
from src.Data.enums import DataResolution, DataPeriod
import config

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*")  # Enable WebSocket with CORS

# In-memory log storage for web dashboard
class InMemoryLogHandler(logging.Handler):
    def __init__(self, max_logs=100):
        super().__init__()
        self.logs = []
        self.max_logs = max_logs
    
    def emit(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'message': self.format(record)
        }
        self.logs.append(log_entry)
        # Keep only last max_logs entries
        if len(self.logs) > self.max_logs:
            self.logs.pop(0)
    
    def get_logs(self):
        return self.logs
    
    def clear_logs(self):
        self.logs.clear()

# Create in-memory log handler
memory_handler = InMemoryLogHandler(max_logs=200)
memory_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler(),
        memory_handler
    ]
)
logger = logging.getLogger(__name__)

# Cache for data providers
data_providers: Dict[str, DataProvider] = {}


def get_resolution_enum(resolution_str: str) -> DataResolution:
    """Convert resolution string to DataResolution enum"""
    resolution_map = {
        '1m': DataResolution.MINUTE_01,
        '2m': DataResolution.MINUTE_02,
        '5m': DataResolution.MINUTE_05,
        '15m': DataResolution.MINUTE_15,
        '30m': DataResolution.MINUTE_30,
        '60m': DataResolution.MINUTE_60,
        '90m': DataResolution.MINUTE_90,
        '1h': DataResolution.HOUR_01,
        '1d': DataResolution.DAY_01,
        '5d': DataResolution.DAY_05,
        '1wk': DataResolution.WEEK,
        '1mo': DataResolution.MONTH_01,
        '3mo': DataResolution.MONTH_03,
    }
    return resolution_map.get(resolution_str, DataResolution.DAY_01)


def get_period_enum(period_str: str) -> DataPeriod:
    """Convert period string to DataPeriod enum"""
    period_map = {
        '1d': DataPeriod.DAY_01,
        '5d': DataPeriod.DAY_05,
        '1mo': DataPeriod.MONTH_01,
        '3mo': DataPeriod.MONTH_03,
        '6mo': DataPeriod.MONTH_06,
        '1y': DataPeriod.YEAR_01,
        '2y': DataPeriod.YEAR_02,
        '5y': DataPeriod.YEAR_05,
        '10y': DataPeriod.YEAR_10,
        'ytd': DataPeriod.YEAR_YTD,
        'max': DataPeriod.YEAR_MAX,
    }
    return period_map.get(period_str, DataPeriod.YEAR_MAX)


def get_or_create_provider(ticker: str, resolution: str, period: str) -> DataProvider:
    """Get or create a DataProvider instance"""
    key = f"{ticker}_{resolution}_{period}"
    
    if key not in data_providers:
        resolution_enum = get_resolution_enum(resolution)
        period_enum = get_period_enum(period)
        
        logger.info(f"Creating new DataProvider for {ticker} with resolution={resolution}, period={period}")
        data_providers[key] = DataProvider(
            tickers=[ticker],
            resolution=resolution_enum,
            period=period_enum
        )
    
    return data_providers[key]


@app.route('/')
def index():
    """Serve the dashboard HTML"""
    return send_file('static/index.html')


@app.route('/api')
def api_info():
    """API documentation endpoint"""
    return jsonify({
        'name': 'Data Provider Server',
        'version': '1.0.0',
        'description': 'REST API for OHLC data - crypto and regular market instruments',
        'endpoints': {
            '/api/health': 'Health check',
            '/api/logs': 'Get recent log messages',
            '/api/logs/clear': 'Clear log messages',
            '/api/instruments': 'List all available instruments',
            '/api/local-instruments': 'List locally cached instruments',
            '/api/local-instruments-detailed': 'Get detailed info about all local instruments (type, resolutions, time ranges)',
            '/api/local-data/<ticker>': 'Get info about locally cached data for a ticker',
            '/api/resolutions': 'List all available resolutions',
            '/api/periods': 'List all available periods',
            '/api/data/<ticker>': 'Get OHLC data for a specific ticker',
            '/api/data/<ticker>/csv': 'Download OHLC data as CSV',
            '/api/refresh/<ticker>': 'Refresh data for a specific ticker',
            '/api/cache/clear': 'Clear the data cache',
        },
        'examples': {
            'Get BTC daily data': '/api/data/BTC-USD?resolution=1d&period=max',
            'Get ETH 1h data': '/api/data/ETH-USD?resolution=1h&period=1y',
            'Get SPY data': '/api/data/SPY?resolution=1d&period=5y',
            'Get local data only': '/api/data/BTC-USD?resolution=1d&local_only=true',
        }
    })


@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'data_dir': config.DATA_DIR,
        'cached_providers': len(data_providers)
    })


@app.route('/api/logs')
def get_logs():
    """Get recent log messages"""
    limit = request.args.get('limit', type=int, default=50)
    logs = memory_handler.get_logs()
    
    # Return last 'limit' logs
    return jsonify({
        'logs': logs[-limit:] if limit > 0 else logs,
        'total': len(logs)
    })


@app.route('/api/logs/clear', methods=['POST'])
def clear_logs():
    """Clear log messages"""
    memory_handler.clear_logs()
    logger.info("Logs cleared")
    return jsonify({
        'status': 'success',
        'message': 'Logs cleared'
    })


@app.route('/api/instruments')
def list_instruments():
    """List all available instruments"""
    return jsonify({
        'crypto': config.CRYPTO_INSTRUMENTS,
        'regular_market': config.REGULAR_MARKET_INSTRUMENTS,
        'total': len(config.CRYPTO_INSTRUMENTS) + len(config.REGULAR_MARKET_INSTRUMENTS)
    })


@app.route('/api/resolutions')
def list_resolutions():
    """List all available resolutions"""
    return jsonify({
        'resolutions': [r.value for r in DataResolution],
        'description': {
            '1m': '1 minute',
            '5m': '5 minutes',
            '15m': '15 minutes',
            '30m': '30 minutes',
            '1h': '1 hour',
            '1d': '1 day',
            '1wk': '1 week',
            '1mo': '1 month',
        }
    })


@app.route('/api/periods')
def list_periods():
    """List all available periods"""
    return jsonify({
        'periods': [p.value for p in DataPeriod],
        'description': {
            '1d': '1 day',
            '5d': '5 days',
            '1mo': '1 month',
            '1y': '1 year',
            '5y': '5 years',
            'max': 'Maximum available',
        }
    })


@app.route('/api/local-instruments')
def list_local_instruments():
    """List all locally available instruments from the data folder"""
    try:
        # Use DataProvider static methods
        local_instruments = DataProvider.get_local_instruments(config.DATA_DIR)
        all_tickers = DataProvider.get_all_local_tickers(config.DATA_DIR)
        
        return jsonify({
            'by_resolution': local_instruments,
            'all_tickers': all_tickers,
            'total': len(all_tickers),
            'resolutions': list(local_instruments.keys())
        })
        
    except Exception as e:
        logger.error(f"Error listing local instruments: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'by_resolution': {},
            'all_tickers': [],
            'total': 0
        }), 500


@app.route('/api/local-data/<ticker>')
def get_local_data_info(ticker: str):
    """
    Get detailed information about locally stored data for a ticker
    
    Query parameters:
    - resolution: Data resolution (default: 1d)
    """
    try:
        resolution = request.args.get('resolution', config.DEFAULT_RESOLUTION)
        ticker = ticker.upper()
        
        # Get resolution enum
        resolution_enum = get_resolution_enum(resolution)
        
        # Get data info using DataProvider static method
        data_info = DataProvider.get_local_data_info(ticker, resolution_enum, config.DATA_DIR)
        
        if data_info:
            return jsonify({
                'ticker': ticker,
                'resolution': resolution,
                'exists': True,
                **data_info
            })
        else:
            return jsonify({
                'ticker': ticker,
                'resolution': resolution,
                'exists': False,
                'message': f'No local data found for {ticker} with resolution {resolution}'
            }), 404
            
    except Exception as e:
        logger.error(f"Error getting local data info for {ticker}: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'ticker': ticker
        }), 500


@app.route('/api/local-instruments-detailed')
def list_local_instruments_detailed():
    """
    Get comprehensive information about all local instruments including:
    - Instrument type (crypto or stock)
    - Available resolutions for each instrument
    - Time range for each resolution
    - Row counts and file sizes
    """
    try:
        logger.info("Fetching detailed local instruments information...")
        
        # Use DataProvider static method
        detailed_info = DataProvider.get_local_instruments_detailed(config.DATA_DIR)
        
        logger.info(f"Found {detailed_info['summary']['total_instruments']} instruments: "
                   f"{detailed_info['summary']['crypto_count']} crypto, "
                   f"{detailed_info['summary']['stock_count']} stocks")
        
        return jsonify(detailed_info)
        
    except Exception as e:
        logger.error(f"Error getting detailed local instruments: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'instruments': [],
            'summary': {
                'total_instruments': 0,
                'crypto_count': 0,
                'stock_count': 0,
                'resolutions': []
            }
        }), 500


@app.route('/api/data/<ticker>')
def get_data(ticker: str):
    """
    Get OHLC data for a specific ticker
    
    Query parameters:
    - resolution: Data resolution (default: 1d)
    - period: Data period (default: max)
    - start_date: Start date (YYYY-MM-DD, optional)
    - end_date: End date (YYYY-MM-DD, optional)
    - limit: Limit number of rows (optional)
    - local_only: Only use local data, don't download (default: false)
    """
    try:
        # Get query parameters
        resolution = request.args.get('resolution', config.DEFAULT_RESOLUTION)
        period = request.args.get('period', config.DEFAULT_PERIOD)
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        limit = request.args.get('limit', type=int)
        local_only = request.args.get('local_only', 'false').lower() == 'true'
        
        # Normalize ticker
        ticker = ticker.upper()
        
        logger.info(f"Data request: ticker={ticker}, resolution={resolution}, period={period}, local_only={local_only}")
        
        # Get resolution and period enums
        resolution_enum = get_resolution_enum(resolution)
        period_enum = get_period_enum(period)
        
        # Check if data exists locally
        data_exists = DataProvider.check_local_data_exists(ticker, resolution_enum, config.DATA_DIR)
        
        if data_exists:
            # Check if data needs updating (max age: 7 days)
            needs_update, reason = DataProvider.check_data_needs_update(
                ticker, resolution_enum, config.DATA_DIR, max_age_days=7
            )
            
            if needs_update and not local_only:
                logger.info(f"{ticker}: {reason}. Updating data...")
                # Update local data with missing records
                df, update_message = DataProvider.update_local_data(
                    ticker, resolution_enum, period_enum, config.DATA_DIR
                )
                logger.info(update_message)
            else:
                if needs_update and local_only:
                    logger.warning(f"{ticker}: {reason}, but local-only mode is enabled")
                else:
                    logger.info(f"{ticker}: Data is up to date")
                
                # Load existing data
                provider = get_or_create_provider(ticker, resolution, period)
                df = provider.data_load_by_ticker(ticker)
        else:
            # No local data exists
            if local_only:
                logger.warning(f"Local-only mode: No data found for {ticker}")
                return jsonify({
                    'error': f'No local data found for {ticker} with resolution {resolution}',
                    'ticker': ticker,
                    'resolution': resolution,
                    'local_only': True,
                    'suggestion': 'Disable local-only mode to download data'
                }), 404
            else:
                logger.info(f"No local data found for {ticker}. Downloading...")
                # Download fresh data
                df, download_message = DataProvider.update_local_data(
                    ticker, resolution_enum, period_enum, config.DATA_DIR
                )
                logger.info(download_message)
        
        # Filter by date range if provided
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        # Apply limit if provided
        if limit:
            df = df.tail(limit)
        
        # Convert to JSON format
        result = {
            'ticker': ticker,
            'resolution': resolution,
            'period': period,
            'rows': len(df),
            'start_date': str(df.index[0]) if len(df) > 0 else None,
            'end_date': str(df.index[-1]) if len(df) > 0 else None,
            'columns': df.columns.tolist(),
            'data': df.reset_index().to_dict(orient='records')
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error getting data for {ticker}: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'ticker': ticker
        }), 500


@app.route('/api/data/<ticker>/csv')
def get_data_csv(ticker: str):
    """
    Download OHLC data as CSV file
    
    Query parameters:
    - resolution: Data resolution (default: 1d)
    - period: Data period (default: max)
    """
    try:
        # Get query parameters
        resolution = request.args.get('resolution', config.DEFAULT_RESOLUTION)
        period = request.args.get('period', config.DEFAULT_PERIOD)
        
        # Normalize ticker
        ticker = ticker.upper()
        
        logger.info(f"CSV download request: ticker={ticker}, resolution={resolution}, period={period}")
        
        # Get or create provider
        provider = get_or_create_provider(ticker, resolution, period)
        
        # Check if data exists locally
        data_file = os.path.join(provider.dir_data, f"{ticker}.csv")
        
        if os.path.exists(data_file):
            logger.info(f"Sending CSV file: {data_file}")
            return send_file(
                data_file,
                mimetype='text/csv',
                as_attachment=True,
                download_name=f"{ticker}_{resolution}_{period}.csv"
            )
        else:
            logger.info(f"Downloading new data for {ticker}")
            df = provider.data_request_by_ticker(ticker)
            provider.data[ticker] = df
            provider.data_save_by_ticker(ticker)
            
            return send_file(
                data_file,
                mimetype='text/csv',
                as_attachment=True,
                download_name=f"{ticker}_{resolution}_{period}.csv"
            )
        
    except Exception as e:
        logger.error(f"Error getting CSV for {ticker}: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'ticker': ticker
        }), 500


@app.route('/api/refresh/<ticker>', methods=['POST'])
def refresh_data(ticker: str):
    """
    Refresh data for a specific ticker
    
    Query parameters:
    - resolution: Data resolution (default: 1d)
    - period: Data period (default: max)
    """
    try:
        # Get query parameters
        resolution = request.args.get('resolution', config.DEFAULT_RESOLUTION)
        period = request.args.get('period', config.DEFAULT_PERIOD)
        
        # Normalize ticker
        ticker = ticker.upper()
        
        logger.info(f"Refresh request: ticker={ticker}, resolution={resolution}, period={period}")
        
        # Get or create provider
        provider = get_or_create_provider(ticker, resolution, period)
        
        # Download fresh data
        df = provider.data_request_by_ticker(ticker)
        provider.data[ticker] = df
        provider.data_save_by_ticker(ticker)
        
        return jsonify({
            'status': 'success',
            'ticker': ticker,
            'resolution': resolution,
            'period': period,
            'rows': len(df),
            'start_date': str(df.index[0]) if len(df) > 0 else None,
            'end_date': str(df.index[-1]) if len(df) > 0 else None,
            'message': f'Data refreshed successfully for {ticker}'
        })
        
    except Exception as e:
        logger.error(f"Error refreshing data for {ticker}: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'ticker': ticker
        }), 500


@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear the data provider cache"""
    try:
        count = len(data_providers)
        data_providers.clear()
        logger.info(f"Cache cleared: {count} providers removed")
        
        return jsonify({
            'status': 'success',
            'message': f'Cache cleared: {count} providers removed'
        })
        
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/api/batch', methods=['POST'])
def get_batch_data():
    """
    Get data for multiple tickers at once
    
    Request body (JSON):
    {
        "tickers": ["BTC-USD", "ETH-USD", "SPY"],
        "resolution": "1d",
        "period": "1y",
        "limit": 100
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'tickers' not in data:
            return jsonify({
                'error': 'Missing required field: tickers'
            }), 400
        
        tickers = data.get('tickers', [])
        resolution = data.get('resolution', config.DEFAULT_RESOLUTION)
        period = data.get('period', config.DEFAULT_PERIOD)
        limit = data.get('limit')
        
        results = {}
        errors = {}
        
        for ticker in tickers:
            try:
                ticker = ticker.upper()
                provider = get_or_create_provider(ticker, resolution, period)
                
                # Check if data exists locally
                data_file = os.path.join(provider.dir_data, f"{ticker}.csv")
                
                if os.path.exists(data_file):
                    df = provider.data_load_by_ticker(ticker)
                else:
                    df = provider.data_request_by_ticker(ticker)
                    provider.data[ticker] = df
                    provider.data_save_by_ticker(ticker)
                
                # Apply limit if provided
                if limit:
                    df = df.tail(limit)
                
                results[ticker] = {
                    'rows': len(df),
                    'start_date': str(df.index[0]) if len(df) > 0 else None,
                    'end_date': str(df.index[-1]) if len(df) > 0 else None,
                    'data': df.reset_index().to_dict(orient='records')
                }
                
            except Exception as e:
                logger.error(f"Error getting data for {ticker}: {str(e)}")
                errors[ticker] = str(e)
        
        return jsonify({
            'resolution': resolution,
            'period': period,
            'results': results,
            'errors': errors if errors else None,
            'success_count': len(results),
            'error_count': len(errors)
        })
        
    except Exception as e:
        logger.error(f"Error in batch request: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e)
        }), 500


# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connection_response', {'status': 'connected', 'message': 'Connected to Data Provider Server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on('get_data')
def handle_get_data(data):
    """
    Handle real-time data request via WebSocket
    
    Expected data format:
    {
        'ticker': 'BTC-USD',
        'resolution': '1d',
        'period': 'max',
        'limit': 100,
        'local_only': True
    }
    """
    try:
        ticker = data.get('ticker', '').upper()
        resolution = data.get('resolution', config.DEFAULT_RESOLUTION)
        period = data.get('period', config.DEFAULT_PERIOD)
        limit = data.get('limit')
        local_only = data.get('local_only', False)
        
        if not ticker:
            emit('data_error', {'error': 'Ticker is required'})
            return
        
        logger.info(f"WebSocket request for {ticker} (resolution={resolution}, period={period}, local_only={local_only})")
        
        provider = get_or_create_provider(ticker, resolution, period)
        data_file = os.path.join(provider.dir_data, f"{ticker}.csv")
        
        # Check if data exists locally
        if os.path.exists(data_file):
            df = provider.data_load_by_ticker(ticker)
        elif local_only:
            emit('data_error', {
                'error': f'No local data available for {ticker}',
                'ticker': ticker
            })
            return
        else:
            # Download data if not in local_only mode
            emit('data_status', {'status': 'downloading', 'ticker': ticker})
            df = provider.data_request_by_ticker(ticker)
            provider.data[ticker] = df
            provider.data_save_by_ticker(ticker)
        
        # Apply limit if provided
        if limit:
            df = df.tail(limit)
        
        # Convert datetime index to string
        df.index = df.index.strftime('%Y-%m-%d %H:%M')

        # Send data back to client
        response_data = {
            'ticker': ticker,
            'resolution': resolution,
            'period': period,
            'rows': len(df),
            'start_date': str(df.index[0]) if len(df) > 0 else None,
            'end_date': str(df.index[-1]) if len(df) > 0 else None,
            'columns': df.reset_index().columns.tolist(),
            'data': df.reset_index().to_dict(orient='records')
        }
        
        emit('data_response', response_data)
        logger.info(f"WebSocket response sent for {ticker}: {len(df)} rows")
        print(f"WebSocket response sent for {ticker}: {len(df)} rows")
        
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)
        emit('data_error', {'error': str(e), 'ticker': data.get('ticker', 'unknown')})


@socketio.on('get_batch_data')
def handle_get_batch_data(data):
    """
    Handle batch data request via WebSocket
    
    Expected data format:
    {
        'tickers': ['BTC-USD', 'ETH-USD', 'SPY'],
        'resolution': '1d',
        'period': '1y',
        'limit': 100,
        'local_only': False
    }
    """
    try:
        tickers = data.get('tickers', [])
        resolution = data.get('resolution', config.DEFAULT_RESOLUTION)
        period = data.get('period', config.DEFAULT_PERIOD)
        limit = data.get('limit')
        local_only = data.get('local_only', False)
        
        if not tickers:
            emit('batch_error', {'error': 'Tickers list is required'})
            return
        
        logger.info(f"WebSocket batch request for {len(tickers)} tickers")
        
        results = {}
        errors = {}
        
        for ticker in tickers:
            try:
                ticker = ticker.upper()
                emit('batch_progress', {
                    'status': 'processing',
                    'ticker': ticker,
                    'progress': f"{len(results) + len(errors) + 1}/{len(tickers)}"
                })
                
                provider = get_or_create_provider(ticker, resolution, period)
                data_file = os.path.join(provider.dir_data, f"{ticker}.csv")
                
                if os.path.exists(data_file):
                    df = provider.data_load_by_ticker(ticker)
                elif local_only:
                    errors[ticker] = f'No local data available for {ticker}'
                    continue
                else:
                    df = provider.data_request_by_ticker(ticker)
                    provider.data[ticker] = df
                    provider.data_save_by_ticker(ticker)
                
                if limit:
                    df = df.tail(limit)
                
                results[ticker] = {
                    'rows': len(df),
                    'start_date': str(df.index[0]) if len(df) > 0 else None,
                    'end_date': str(df.index[-1]) if len(df) > 0 else None,
                    'data': df.reset_index().to_dict(orient='records')
                }
                
            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
                errors[ticker] = str(e)
        
        # Send final batch response
        emit('batch_response', {
            'resolution': resolution,
            'period': period,
            'results': results,
            'errors': errors if errors else None,
            'success_count': len(results),
            'error_count': len(errors)
        })
        
        logger.info(f"WebSocket batch response sent: {len(results)} successful, {len(errors)} errors")
        
    except Exception as e:
        logger.error(f"WebSocket batch error: {str(e)}", exc_info=True)
        emit('batch_error', {'error': str(e)})


if __name__ == '__main__':
    logger.info(f"Starting Data Provider Server on {config.HOST}:{config.PORT}")
    logger.info(f"Data directory: {config.DATA_DIR}")
    logger.info("WebSocket support enabled")
    
    # Create data directory if it doesn't exist
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    # Use socketio.run instead of app.run for WebSocket support
    socketio.run(
        app,
        host=config.HOST,
        port=config.PORT,
        debug=True,
        allow_unsafe_werkzeug=True
    )
