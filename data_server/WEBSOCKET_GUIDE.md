# WebSocket Guide for Data Provider

This guide explains how to use the WebSocket functionality in the Data Provider Server for real-time data streaming.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

The Data Provider Server now supports WebSocket connections for real-time data streaming. WebSocket provides:

- **Real-time updates**: Receive data as soon as it's available
- **Bidirectional communication**: Server can push updates to clients
- **Lower latency**: Persistent connection reduces overhead
- **Progress tracking**: Monitor batch operations in real-time

### When to Use WebSocket vs REST API

**Use WebSocket when:**
- You need real-time data updates
- You're making multiple sequential requests
- You want progress updates for batch operations
- You need lower latency

**Use REST API when:**
- You need a simple one-time data fetch
- You're working in environments without WebSocket support
- You prefer stateless communication

## Installation

Install the required dependencies:

```bash
pip install flask-socketio==5.3.5 python-socketio==5.10.0
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Start the Server

```bash
python server.py
```

The server will start on `http://localhost:5001` with WebSocket support enabled.

### 2. Basic Usage

```python
from src.Data.DataProviderLocal import DataProviderLocal
from src.Background.enums import DataResolution, DataPeriod

# Create provider instance
provider = DataProviderLocal()

# Get data via WebSocket
df = provider.get_data_socket(
    ticker='BTC-USD',
    resolution=DataResolution.DAY_01,
    period=DataPeriod.YEAR_01,
    local_only=True
)

print(df.head())
```

## API Reference

### `get_data_socket()`

Get OHLC data for a single ticker via WebSocket.

**Parameters:**
- `ticker` (str): Instrument ticker (e.g., 'BTC-USD', 'SPY')
- `resolution` (DataResolution): Data resolution enum
- `period` (DataPeriod): Data period enum
- `time_start` (Optional[date]): Start date filter
- `time_end` (Optional[date]): End date filter
- `limit` (Optional[int]): Limit number of rows
- `local_only` (bool): Only use cached data (default: True)
- `callback` (Optional[Callable]): Callback for data response
- `error_callback` (Optional[Callable]): Callback for errors
- `auto_disconnect` (bool): Disconnect after request (default: True)

**Returns:**
- `pd.DataFrame` or `None`: OHLC data with datetime index

**Example:**
```python
df = provider.get_data_socket(
    ticker='ETH-USD',
    resolution=DataResolution.HOUR_01,
    period=DataPeriod.MONTH_01,
    limit=100
)
```

### `get_batch_data_socket()`

Get OHLC data for multiple tickers via WebSocket.

**Parameters:**
- `tickers` (List[str]): List of instrument tickers
- `resolution` (DataResolution): Data resolution enum
- `period` (DataPeriod): Data period enum
- `limit` (Optional[int]): Limit rows per ticker
- `local_only` (bool): Only use cached data (default: True)
- `progress_callback` (Optional[Callable]): Progress updates callback
- `error_callback` (Optional[Callable]): Error callback
- `auto_disconnect` (bool): Disconnect after request (default: True)

**Returns:**
- `Dict[str, pd.DataFrame]`: Dictionary mapping ticker to DataFrame

**Example:**
```python
results = provider.get_batch_data_socket(
    tickers=['BTC-USD', 'ETH-USD', 'SPY'],
    resolution=DataResolution.DAY_01,
    period=DataPeriod.YEAR_01
)

for ticker, df in results.items():
    print(f"{ticker}: {len(df)} rows")
```

## Examples

### Example 1: Simple Data Fetch

```python
from src.Data.DataProviderLocal import DataProviderLocal
from src.Background.enums import DataResolution, DataPeriod

provider = DataProviderLocal()

df = provider.get_data_socket(
    ticker='BTC-USD',
    resolution=DataResolution.DAY_01,
    period=DataPeriod.YEAR_MAX,
    local_only=True
)

if df is not None:
    print(f"Received {len(df)} rows")
    print(df.tail())
```

### Example 2: With Callbacks

```python
def on_data_received(data):
    print(f"Data received for {data['ticker']}")
    print(f"Rows: {data['rows']}")
    print(f"Date range: {data['start_date']} to {data['end_date']}")

def on_error(error):
    print(f"Error: {error}")

df = provider.get_data_socket(
    ticker='ETH-USD',
    resolution=DataResolution.HOUR_01,
    period=DataPeriod.MONTH_01,
    callback=on_data_received,
    error_callback=on_error
)
```

### Example 3: Batch Request

```python
results = provider.get_batch_data_socket(
    tickers=['BTC-USD', 'ETH-USD', 'SPY', 'AAPL'],
    resolution=DataResolution.DAY_01,
    period=DataPeriod.YEAR_01,
    limit=100
)

for ticker, df in results.items():
    print(f"{ticker}: {len(df)} rows")
    print(f"Last Close: ${df['Close'].iloc[-1]:.2f}")
```

### Example 4: Progress Tracking

```python
def on_progress(progress_info):
    ticker = progress_info['ticker']
    progress = progress_info['progress']
    print(f"Processing {ticker} ({progress})")

results = provider.get_batch_data_socket(
    tickers=['BTC-USD', 'ETH-USD', 'SOL-USD'],
    resolution=DataResolution.DAY_01,
    period=DataPeriod.MONTH_06,
    progress_callback=on_progress
)
```

### Example 5: Persistent Connection

```python
provider = DataProviderLocal()

# Connect once
provider._connect_socket()

# Make multiple requests
for ticker in ['BTC-USD', 'ETH-USD', 'SPY']:
    df = provider.get_data_socket(
        ticker=ticker,
        resolution=DataResolution.DAY_01,
        period=DataPeriod.MONTH_01,
        auto_disconnect=False  # Keep connection alive
    )
    print(f"{ticker}: {len(df)} rows")

# Disconnect manually
provider._disconnect_socket()
```

### Example 6: Date Filtering

```python
import datetime as dt

df = provider.get_data_socket(
    ticker='SPY',
    resolution=DataResolution.DAY_01,
    period=DataPeriod.YEAR_MAX,
    time_start=dt.date(2023, 1, 1),
    time_end=dt.date(2023, 12, 31),
    local_only=True
)

print(f"SPY data for 2023: {len(df)} rows")
```

## Best Practices

### 1. Connection Management

**Good:**
```python
# Auto-disconnect after single request (default)
df = provider.get_data_socket(ticker='BTC-USD', ...)
```

**Better for multiple requests:**
```python
# Reuse connection for multiple requests
provider._connect_socket()
for ticker in tickers:
    df = provider.get_data_socket(ticker=ticker, auto_disconnect=False)
provider._disconnect_socket()
```

### 2. Error Handling

Always use error callbacks for production code:

```python
def on_error(error):
    logger.error(f"WebSocket error: {error}")
    # Implement retry logic or fallback to REST API

df = provider.get_data_socket(
    ticker='BTC-USD',
    error_callback=on_error,
    ...
)
```

### 3. Batch Operations

For multiple tickers, use batch requests instead of individual calls:

**Bad:**
```python
for ticker in tickers:
    df = provider.get_data_socket(ticker=ticker, ...)
```

**Good:**
```python
results = provider.get_batch_data_socket(tickers=tickers, ...)
```

### 4. Local-Only Mode

Use `local_only=True` to avoid unnecessary downloads:

```python
df = provider.get_data_socket(
    ticker='BTC-USD',
    local_only=True  # Only use cached data
)
```

### 5. Limit Data Size

Use the `limit` parameter to reduce data transfer:

```python
df = provider.get_data_socket(
    ticker='BTC-USD',
    limit=100  # Only get last 100 rows
)
```

## Troubleshooting

### Connection Failed

**Problem:** `Failed to connect to WebSocket server`

**Solutions:**
1. Ensure the server is running: `python server.py`
2. Check if port 5001 is available
3. Verify firewall settings
4. Check server logs for errors

### Request Timeout

**Problem:** `Request timeout` after 30 seconds

**Solutions:**
1. Check network connectivity
2. Verify the ticker exists locally (if using `local_only=True`)
3. Increase timeout in the code (modify `timeout` variable)
4. Check server performance and logs

### No Data Returned

**Problem:** `get_data_socket()` returns `None`

**Solutions:**
1. Check if data exists locally: `provider.get_instruments_local()`
2. Set `local_only=False` to download data
3. Verify ticker symbol is correct
4. Check error callback for detailed error message

### Batch Request Errors

**Problem:** Some tickers fail in batch request

**Solutions:**
1. Use `error_callback` to see which tickers failed
2. Check if failed tickers exist locally
3. Verify ticker symbols are correct
4. Try individual requests for failed tickers

## WebSocket Events

The server emits the following WebSocket events:

### Client → Server

- `get_data`: Request data for a single ticker
- `get_batch_data`: Request data for multiple tickers

### Server → Client

- `connection_response`: Connection confirmation
- `data_response`: Data response for single ticker
- `data_error`: Error for single ticker request
- `data_status`: Status update (e.g., downloading)
- `batch_response`: Batch data response
- `batch_error`: Batch request error
- `batch_progress`: Progress update for batch request

## Performance Tips

1. **Reuse connections**: Keep connection alive for multiple requests
2. **Use batch requests**: More efficient than individual requests
3. **Limit data**: Use `limit` parameter to reduce transfer size
4. **Local-only mode**: Avoid downloads when possible
5. **Async operations**: Consider using async/await for concurrent requests

## Additional Resources

- [Server Documentation](README.md)
- [API Guide](API_GUIDE.md)
- [Example Usage](example_websocket_usage.py)
- [Flask-SocketIO Documentation](https://flask-socketio.readthedocs.io/)
- [Python-SocketIO Documentation](https://python-socketio.readthedocs.io/)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review server logs: `data_server.log`
3. Run example file: `python example_websocket_usage.py`
4. Check if REST API works: `python test_client.py`
