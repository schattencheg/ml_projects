import requests
import pandas as pd
import datetime as dt
import socketio
import time
from typing import Optional, List, Dict, Callable
from src.Background.enums import DataPeriod, DataResolution

BASE_URL = "http://localhost:5001"


'''
===========================================
EXAMPLE USAGE - REST API (HTTP)
===========================================

# Basic usage with REST API
provider = DataProviderLocal()

# Get single ticker data
df = provider.get_data(
    ticker='BTC-USD',
    resolution=DataResolution.DAY_01,
    period=DataPeriod.YEAR_01
)
print(df.head())

# Get local instruments
instruments = provider.get_instruments_local()
print(instruments)

# Get detailed local instruments info
detailed = provider.get_instruments_local_detailed()
print(detailed)


===========================================
EXAMPLE USAGE - WEBSOCKET (Real-time)
===========================================

# Basic WebSocket usage
provider = DataProviderLocal()

# Example 1: Simple synchronous call
df = provider.get_data_socket(
    ticker='BTC-USD',
    resolution=DataResolution.DAY_01,
    period=DataPeriod.YEAR_MAX,
    local_only=True
)
if df is not None:
    print(f"Received {len(df)} rows")
    print(df.tail())

# Example 2: With callback for real-time updates
def on_data_received(data):
    print(f"Data received for {data['ticker']}")
    print(f"Rows: {data['rows']}, Date range: {data['start_date']} to {data['end_date']}")

def on_error(error):
    print(f"Error: {error}")

df = provider.get_data_socket(
    ticker='ETH-USD',
    resolution=DataResolution.HOUR_01,
    period=DataPeriod.MONTH_01,
    callback=on_data_received,
    error_callback=on_error
)

# Example 3: Batch request via WebSocket
results = provider.get_batch_data_socket(
    tickers=['BTC-USD', 'ETH-USD', 'SPY', 'AAPL'],
    resolution=DataResolution.DAY_01,
    period=DataPeriod.YEAR_01,
    limit=100
)

for ticker, data in results.items():
    print(f"{ticker}: {len(data)} rows")

# Example 4: Batch with progress callback
def on_progress(progress_info):
    print(f"Processing {progress_info['ticker']} - {progress_info['progress']}")

results = provider.get_batch_data_socket(
    tickers=['BTC-USD', 'ETH-USD', 'SOL-USD'],
    resolution=DataResolution.DAY_01,
    period=DataPeriod.MONTH_06,
    progress_callback=on_progress
)

'''


class DataProviderLocal:
    def __init__(self):
        self.sio = None
        self._socket_connected = False
        self._last_response = None
        self._last_error = None
        self._batch_results = {}
        self._batch_errors = {}
    
    def get_data(self, ticker: str, resolution: DataResolution, period: DataPeriod, 
                    time_start: dt.date = None, time_end: dt.date = None):
        """Get OHLC data via REST API (HTTP)"""
        response = None
        try:
            # Get Bitcoin daily data
            response = requests.get(f"{BASE_URL}/api/data/{ticker}", params={
                'resolution': resolution.value,
                'period': period.value,
                'local_only': True
            })
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return None
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data['data'], columns=data['columns'])
            return df
        else:
            return None

    def get_instruments_local(self):
        """Get list of locally available instruments"""
        response = requests.get(f"{BASE_URL}/api/local-instruments")
        if response.status_code == 200:
            return response.json()
        else:
            return None

    def get_instruments_local_detailed(self):
        """Get detailed information about locally available instruments"""
        response = requests.get(f"{BASE_URL}/api/local-instruments-detailed")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    
    def _connect_socket(self) -> bool:
        """Establish WebSocket connection to the server"""
        if self._socket_connected and self.sio is not None:
            return True
        
        try:
            self.sio = socketio.Client()
            
            @self.sio.on('connect')
            def on_connect():
                self._socket_connected = True
                print("WebSocket connected to server")
            
            @self.sio.on('disconnect')
            def on_disconnect():
                self._socket_connected = False
                print("WebSocket disconnected from server")
            
            @self.sio.on('connection_response')
            def on_connection_response(data):
                print(f"Server: {data.get('message', 'Connected')}")
            
            @self.sio.on('data_response')
            def on_data_response(data):
                self._last_response = data
            
            @self.sio.on('data_error')
            def on_data_error(data):
                self._last_error = data.get('error', 'Unknown error')
            
            @self.sio.on('data_status')
            def on_data_status(data):
                print(f"Status: {data.get('status', '')} - {data.get('ticker', '')}")
            
            @self.sio.on('batch_response')
            def on_batch_response(data):
                self._last_response = data
            
            @self.sio.on('batch_error')
            def on_batch_error(data):
                self._last_error = data.get('error', 'Unknown batch error')
            
            @self.sio.on('batch_progress')
            def on_batch_progress(data):
                # Store progress info for potential callback
                if hasattr(self, '_progress_callback') and self._progress_callback:
                    self._progress_callback(data)
            
            # Connect to server
            self.sio.connect(BASE_URL)
            
            # Wait for connection
            timeout = 5
            start_time = time.time()
            while not self._socket_connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            return self._socket_connected
            
        except Exception as e:
            print(f"Failed to connect WebSocket: {str(e)}")
            self._socket_connected = False
            return False
    
    def _disconnect_socket(self):
        """Disconnect WebSocket connection"""
        if self.sio is not None and self._socket_connected:
            try:
                self.sio.disconnect()
                self._socket_connected = False
            except Exception as e:
                print(f"Error disconnecting WebSocket: {str(e)}")
    
    def get_data_socket(
        self,
        ticker: str,
        resolution: DataResolution,
        period: DataPeriod,
        time_start: Optional[dt.date] = None,
        time_end: Optional[dt.date] = None,
        limit: Optional[int] = None,
        local_only: bool = True,
        callback: Optional[Callable] = None,
        error_callback: Optional[Callable] = None,
        auto_disconnect: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Get OHLC data via WebSocket connection (real-time)
        
        Args:
            ticker: Instrument ticker (e.g., 'BTC-USD', 'SPY')
            resolution: Data resolution (DataResolution enum)
            period: Data period (DataPeriod enum)
            time_start: Optional start date filter
            time_end: Optional end date filter
            limit: Optional limit on number of rows
            local_only: If True, only use cached data (no downloads)
            callback: Optional callback function for data response
            error_callback: Optional callback function for errors
            auto_disconnect: If True, disconnect after receiving data
        
        Returns:
            DataFrame with OHLC data or None if error
        
        Example:
            provider = DataProviderLocal()
            df = provider.get_data_socket(
                ticker='BTC-USD',
                resolution=DataResolution.DAY_01,
                period=DataPeriod.YEAR_01,
                limit=100
            )
        """
        # Reset response tracking
        self._last_response = None
        self._last_error = None
        
        # Connect to WebSocket
        if not self._connect_socket():
            if error_callback:
                error_callback("Failed to connect to WebSocket server")
            return None
        
        try:
            # Prepare request data
            request_data = {
                'ticker': ticker,
                'resolution': resolution.value,
                'period': period.value,
                'local_only': local_only
            }
            
            if limit:
                request_data['limit'] = limit
            
            # Send request
            self.sio.emit('get_data', request_data)
            
            # Wait for response
            timeout = 30  # 30 seconds timeout
            start_time = time.time()
            
            while self._last_response is None and self._last_error is None:
                if (time.time() - start_time) > timeout:
                    if error_callback:
                        error_callback("Request timeout")
                    return None
                time.sleep(0.1)
            
            # Check for errors
            if self._last_error:
                if error_callback:
                    error_callback(self._last_error)
                else:
                    print(f"Error: {self._last_error}")
                return None
            
            # Process response
            if self._last_response:
                if callback:
                    callback(self._last_response)
                
                # Convert to DataFrame
                data = self._last_response.get('data', [])
                if data:
                    df = pd.DataFrame(data)
                    
                    # Set index if 'Date' or 'Datetime' column exists
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)
                    elif 'Datetime' in df.columns:
                        df['Datetime'] = pd.to_datetime(df['Datetime'])
                        df.set_index('Datetime', inplace=True)
                    
                    # Apply date filters if provided
                    if time_start:
                        df = df[df.index >= pd.to_datetime(time_start)]
                    if time_end:
                        df = df[df.index <= pd.to_datetime(time_end)]
                    
                    return df
            
            return None
            
        except Exception as e:
            error_msg = f"WebSocket request error: {str(e)}"
            if error_callback:
                error_callback(error_msg)
            else:
                print(error_msg)
            return None
        
        finally:
            if auto_disconnect:
                self._disconnect_socket()
    
    def get_batch_data_socket(
        self,
        tickers: List[str],
        resolution: DataResolution,
        period: DataPeriod,
        limit: Optional[int] = None,
        local_only: bool = True,
        progress_callback: Optional[Callable] = None,
        error_callback: Optional[Callable] = None,
        auto_disconnect: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Get OHLC data for multiple tickers via WebSocket (batch request)
        
        Args:
            tickers: List of instrument tickers
            resolution: Data resolution (DataResolution enum)
            period: Data period (DataPeriod enum)
            limit: Optional limit on number of rows per ticker
            local_only: If True, only use cached data (no downloads)
            progress_callback: Optional callback for progress updates
            error_callback: Optional callback function for errors
            auto_disconnect: If True, disconnect after receiving data
        
        Returns:
            Dictionary mapping ticker to DataFrame
        
        Example:
            provider = DataProviderLocal()
            results = provider.get_batch_data_socket(
                tickers=['BTC-USD', 'ETH-USD', 'SPY'],
                resolution=DataResolution.DAY_01,
                period=DataPeriod.YEAR_01
            )
        """
        # Reset response tracking
        self._last_response = None
        self._last_error = None
        self._progress_callback = progress_callback
        
        # Connect to WebSocket
        if not self._connect_socket():
            if error_callback:
                error_callback("Failed to connect to WebSocket server")
            return {}
        
        try:
            # Prepare request data
            request_data = {
                'tickers': tickers,
                'resolution': resolution.value,
                'period': period.value,
                'local_only': local_only
            }
            
            if limit:
                request_data['limit'] = limit
            
            # Send batch request
            self.sio.emit('get_batch_data', request_data)
            
            # Wait for response
            timeout = 60  # 60 seconds timeout for batch
            start_time = time.time()
            
            while self._last_response is None and self._last_error is None:
                if (time.time() - start_time) > timeout:
                    if error_callback:
                        error_callback("Batch request timeout")
                    return {}
                time.sleep(0.1)
            
            # Check for errors
            if self._last_error:
                if error_callback:
                    error_callback(self._last_error)
                else:
                    print(f"Batch Error: {self._last_error}")
                return {}
            
            # Process batch response
            results = {}
            if self._last_response:
                batch_results = self._last_response.get('results', {})
                batch_errors = self._last_response.get('errors', {})
                
                # Convert each result to DataFrame
                for ticker, ticker_data in batch_results.items():
                    data = ticker_data.get('data', [])
                    if data:
                        df = pd.DataFrame(data)
                        
                        # Set index
                        if 'Date' in df.columns:
                            df['Date'] = pd.to_datetime(df['Date'])
                            df.set_index('Date', inplace=True)
                        elif 'Datetime' in df.columns:
                            df['Datetime'] = pd.to_datetime(df['Datetime'])
                            df.set_index('Datetime', inplace=True)
                        
                        results[ticker] = df
                
                # Report errors if any
                if batch_errors and error_callback:
                    error_callback(f"Errors for some tickers: {batch_errors}")
                elif batch_errors:
                    print(f"Batch errors: {batch_errors}")
            
            return results
            
        except Exception as e:
            error_msg = f"WebSocket batch request error: {str(e)}"
            if error_callback:
                error_callback(error_msg)
            else:
                print(error_msg)
            return {}
        
        finally:
            self._progress_callback = None
            if auto_disconnect:
                self._disconnect_socket()
