# Data Provider Server - API Usage Guide

Complete guide for using the Data Provider Server API to fetch OHLC data for crypto and market instruments.

## Table of Contents

1. [Getting Started](#getting-started)
2. [API Endpoints](#api-endpoints)
3. [Python Examples](#python-examples)
4. [JavaScript Examples](#javascript-examples)
5. [Integration with DataProvider](#integration-with-dataprovider)
6. [Error Handling](#error-handling)
7. [Best Practices](#best-practices)

## Getting Started

### Starting the Server

**Option 1: Using the batch file (Windows)**
```bash
start_server.bat
```

**Option 2: Manual start**
```bash
# Install dependencies
pip install -r requirements.txt

# Start server
python server.py
```

The server will be available at: `http://localhost:5001`

### Quick Test

Open your browser and navigate to:
- Dashboard: `http://localhost:5001`
- API Info: `http://localhost:5001/api`
- Health Check: `http://localhost:5001/api/health`

## API Endpoints

### 1. Health Check

Check if the server is running and get status information.

**Endpoint:** `GET /api/health`

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-16T13:16:08",
  "data_dir": "d:\\Work\\Repos\\ml_projects\\data_server\\data",
  "cached_providers": 3
}
```

### 2. List Instruments

Get all available crypto and regular market instruments.

**Endpoint:** `GET /api/instruments`

**Response:**
```json
{
  "crypto": ["BTC-USD", "BTC-USDT", "ETH-USD", "ETH-USDT", ...],
  "regular_market": ["SPY", "QQQ", "AAPL", "MSFT", ...],
  "total": 16
}
```

### 3. List Resolutions

Get all available data resolutions (timeframes).

**Endpoint:** `GET /api/resolutions`

**Response:**
```json
{
  "resolutions": ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"],
  "description": {
    "1m": "1 minute",
    "1h": "1 hour",
    "1d": "1 day"
  }
}
```

### 4. List Periods

Get all available data periods.

**Endpoint:** `GET /api/periods`

**Response:**
```json
{
  "periods": ["1d", "5d", "1mo", "1y", "5y", "max"],
  "description": {
    "1d": "1 day",
    "1y": "1 year",
    "max": "Maximum available"
  }
}
```

### 5. Get OHLC Data

Get OHLC data for a specific ticker.

**Endpoint:** `GET /api/data/<ticker>`

**Query Parameters:**
- `resolution` (optional): Data resolution - `1m`, `5m`, `15m`, `30m`, `1h`, `1d`, `1wk`, `1mo` (default: `1d`)
- `period` (optional): Data period - `1d`, `5d`, `1mo`, `1y`, `5y`, `max` (default: `max`)
- `start_date` (optional): Start date in YYYY-MM-DD format
- `end_date` (optional): End date in YYYY-MM-DD format
- `limit` (optional): Limit number of rows returned

**Example Request:**
```
GET /api/data/BTC-USD?resolution=1d&period=max&limit=100
```

**Response:**
```json
{
  "ticker": "BTC-USD",
  "resolution": "1d",
  "period": "max",
  "rows": 100,
  "start_date": "2024-07-08",
  "end_date": "2025-10-16",
  "columns": ["timestamp", "open", "high", "low", "close", "volume"],
  "data": [
    {
      "timestamp": "2024-07-08",
      "open": 57123.45,
      "high": 58234.56,
      "low": 56789.12,
      "close": 57890.23,
      "volume": 12345678
    },
    ...
  ]
}
```

### 6. Download CSV

Download OHLC data as a CSV file.

**Endpoint:** `GET /api/data/<ticker>/csv`

**Query Parameters:**
- `resolution` (optional): Data resolution (default: `1d`)
- `period` (optional): Data period (default: `max`)

**Example:**
```bash
curl "http://localhost:5001/api/data/BTC-USD/csv?resolution=1d&period=max" -o btc_data.csv
```

### 7. Refresh Data

Force refresh data for a specific ticker (downloads fresh data).

**Endpoint:** `POST /api/refresh/<ticker>`

**Query Parameters:**
- `resolution` (optional): Data resolution (default: `1d`)
- `period` (optional): Data period (default: `max`)

**Response:**
```json
{
  "status": "success",
  "ticker": "BTC-USD",
  "resolution": "1d",
  "period": "max",
  "rows": 2847,
  "start_date": "2014-09-17",
  "end_date": "2025-10-16",
  "message": "Data refreshed successfully for BTC-USD"
}
```

### 8. Batch Request

Get data for multiple tickers in a single request.

**Endpoint:** `POST /api/batch`

**Request Body:**
```json
{
  "tickers": ["BTC-USD", "ETH-USD", "SPY"],
  "resolution": "1d",
  "period": "1y",
  "limit": 100
}
```

**Response:**
```json
{
  "resolution": "1d",
  "period": "1y",
  "results": {
    "BTC-USD": {
      "rows": 100,
      "start_date": "2024-07-08",
      "end_date": "2025-10-16",
      "data": [...]
    },
    "ETH-USD": {...},
    "SPY": {...}
  },
  "errors": null,
  "success_count": 3,
  "error_count": 0
}
```

### 9. Clear Cache

Clear the in-memory data provider cache.

**Endpoint:** `POST /api/cache/clear`

**Response:**
```json
{
  "status": "success",
  "message": "Cache cleared: 5 providers removed"
}
```

## Python Examples

### Example 1: Basic Data Retrieval

```python
import requests
import pandas as pd

BASE_URL = "http://localhost:5001"

# Get Bitcoin daily data
response = requests.get(f"{BASE_URL}/api/data/BTC-USD", params={
    'resolution': '1d',
    'period': 'max',
    'limit': 100
})

data = response.json()
df = pd.DataFrame(data['data'])
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

print(f"Ticker: {data['ticker']}")
print(f"Rows: {data['rows']}")
print(f"Date range: {data['start_date']} to {data['end_date']}")
print("\nData:")
print(df.head())
```

### Example 2: Multiple Timeframes

```python
import requests
import pandas as pd

BASE_URL = "http://localhost:5001"

def get_ohlc_data(ticker, resolution='1d', period='max', limit=None):
    """Get OHLC data for a ticker"""
    params = {
        'resolution': resolution,
        'period': period
    }
    if limit:
        params['limit'] = limit
    
    response = requests.get(f"{BASE_URL}/api/data/{ticker}", params=params)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data['data'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df
    else:
        raise Exception(f"Error: {response.json()}")

# Get different timeframes
btc_daily = get_ohlc_data('BTC-USD', resolution='1d', period='1y')
btc_hourly = get_ohlc_data('BTC-USD', resolution='1h', period='1mo')
btc_15min = get_ohlc_data('BTC-USD', resolution='15m', period='5d')

print(f"Daily data: {len(btc_daily)} rows")
print(f"Hourly data: {len(btc_hourly)} rows")
print(f"15-min data: {len(btc_15min)} rows")
```

### Example 3: Batch Request

```python
import requests
import pandas as pd

BASE_URL = "http://localhost:5001"

# Request multiple tickers at once
response = requests.post(f"{BASE_URL}/api/batch", json={
    'tickers': ['BTC-USD', 'ETH-USD', 'SPY', 'AAPL'],
    'resolution': '1d',
    'period': '1y',
    'limit': 50
})

batch_data = response.json()

print(f"Success: {batch_data['success_count']}")
print(f"Errors: {batch_data['error_count']}")

# Process each ticker
dataframes = {}
for ticker, info in batch_data['results'].items():
    df = pd.DataFrame(info['data'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    dataframes[ticker] = df
    print(f"{ticker}: {len(df)} rows")

# Access individual dataframes
btc_df = dataframes['BTC-USD']
eth_df = dataframes['ETH-USD']
```

### Example 4: Integration with Trading Strategy

```python
import requests
import pandas as pd
import numpy as np

BASE_URL = "http://localhost:5001"

def get_ohlc_data(ticker, resolution='1d', period='max'):
    """Get OHLC data"""
    response = requests.get(f"{BASE_URL}/api/data/{ticker}", params={
        'resolution': resolution,
        'period': period
    })
    data = response.json()
    df = pd.DataFrame(data['data'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

# Get data
df = get_ohlc_data('BTC-USD', resolution='1d', period='1y')

# Calculate indicators
df['SMA_20'] = df['close'].rolling(window=20).mean()
df['SMA_50'] = df['close'].rolling(window=50).mean()

# Generate signals
df['signal'] = 0
df.loc[df['SMA_20'] > df['SMA_50'], 'signal'] = 1  # Buy
df.loc[df['SMA_20'] < df['SMA_50'], 'signal'] = -1  # Sell

# Calculate returns
df['returns'] = df['close'].pct_change()
df['strategy_returns'] = df['signal'].shift(1) * df['returns']

# Performance metrics
total_return = (1 + df['strategy_returns']).cumprod().iloc[-1] - 1
print(f"Total Return: {total_return:.2%}")
```

### Example 5: Download and Save CSV

```python
import requests

BASE_URL = "http://localhost:5001"

def download_csv(ticker, resolution='1d', period='max', filename=None):
    """Download OHLC data as CSV"""
    response = requests.get(f"{BASE_URL}/api/data/{ticker}/csv", params={
        'resolution': resolution,
        'period': period
    })
    
    if response.status_code == 200:
        if filename is None:
            filename = f"{ticker}_{resolution}_{period}.csv"
        
        with open(filename, 'wb') as f:
            f.write(response.content)
        
        print(f"Downloaded: {filename}")
        return filename
    else:
        raise Exception(f"Error downloading CSV: {response.status_code}")

# Download multiple files
download_csv('BTC-USD', resolution='1d', period='max')
download_csv('ETH-USD', resolution='1h', period='1mo')
download_csv('SPY', resolution='1d', period='5y')
```

## JavaScript Examples

### Example 1: Fetch Data

```javascript
const BASE_URL = 'http://localhost:5001';

async function getOHLCData(ticker, resolution = '1d', period = 'max', limit = null) {
    const params = new URLSearchParams({
        resolution,
        period
    });
    
    if (limit) {
        params.append('limit', limit);
    }
    
    const response = await fetch(`${BASE_URL}/api/data/${ticker}?${params}`);
    const data = await response.json();
    
    return data;
}

// Usage
getOHLCData('BTC-USD', '1d', 'max', 100)
    .then(data => {
        console.log(`Ticker: ${data.ticker}`);
        console.log(`Rows: ${data.rows}`);
        console.log(`Date range: ${data.start_date} to ${data.end_date}`);
        console.log('Data:', data.data);
    })
    .catch(error => console.error('Error:', error));
```

### Example 2: Batch Request

```javascript
const BASE_URL = 'http://localhost:5001';

async function batchRequest(tickers, resolution = '1d', period = '1y', limit = 100) {
    const response = await fetch(`${BASE_URL}/api/batch`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            tickers,
            resolution,
            period,
            limit
        })
    });
    
    const data = await response.json();
    return data;
}

// Usage
batchRequest(['BTC-USD', 'ETH-USD', 'SPY'], '1d', '1y', 50)
    .then(data => {
        console.log(`Success: ${data.success_count}`);
        console.log(`Errors: ${data.error_count}`);
        
        for (const [ticker, info] of Object.entries(data.results)) {
            console.log(`${ticker}: ${info.rows} rows`);
        }
    })
    .catch(error => console.error('Error:', error));
```

## Integration with DataProvider

The server uses the existing `DataProvider` class. You can also use it directly in your Python scripts:

```python
import sys
sys.path.insert(0, 'src')

from src.Data.DataProvider import DataProvider
from src.Data.enums import DataResolution, DataPeriod

# Create a provider
provider = DataProvider(
    tickers=['BTC-USD', 'ETH-USD'],
    resolution=DataResolution.DAY_01,
    period=DataPeriod.YEAR_MAX
)

# Request data (downloads if not available locally)
btc_df = provider.data_request_by_ticker('BTC-USD')

# Load from local cache
eth_df = provider.data_load_by_ticker('ETH-USD')

# Save to local cache
provider.data_save_by_ticker('BTC-USD')

print(f"BTC data: {len(btc_df)} rows")
print(f"ETH data: {len(eth_df)} rows")
```

## Error Handling

### HTTP Status Codes

- `200 OK` - Success
- `400 Bad Request` - Invalid parameters
- `500 Internal Server Error` - Server error

### Error Response Format

```json
{
  "error": "Error message description",
  "ticker": "BTC-USD"
}
```

### Python Error Handling

```python
import requests

BASE_URL = "http://localhost:5001"

try:
    response = requests.get(f"{BASE_URL}/api/data/INVALID-TICKER")
    response.raise_for_status()  # Raise exception for bad status codes
    data = response.json()
except requests.exceptions.HTTPError as e:
    print(f"HTTP Error: {e}")
    print(f"Response: {response.json()}")
except requests.exceptions.ConnectionError:
    print("Error: Could not connect to server")
except Exception as e:
    print(f"Error: {e}")
```

## Best Practices

### 1. Use Batch Requests for Multiple Tickers

Instead of making multiple individual requests, use the batch endpoint:

```python
# ❌ Bad - Multiple requests
btc = get_data('BTC-USD')
eth = get_data('ETH-USD')
spy = get_data('SPY')

# ✅ Good - Single batch request
batch = batch_request(['BTC-USD', 'ETH-USD', 'SPY'])
```

### 2. Cache Data Locally

Download data once and save it locally to avoid repeated API calls:

```python
# Download and save
df = get_ohlc_data('BTC-USD', resolution='1d', period='max')
df.to_csv('btc_data.csv')

# Load from local file
df = pd.read_csv('btc_data.csv', index_col=0, parse_dates=True)
```

### 3. Use Appropriate Timeframes

Choose the right resolution for your use case:
- **Backtesting**: Use `1d` or `1h` for faster processing
- **Live trading**: Use `1m` or `5m` for real-time data
- **Analysis**: Use `1d` or `1wk` for long-term trends

### 4. Limit Data When Testing

Use the `limit` parameter to reduce data size during development:

```python
# Get only last 100 rows for testing
df = get_ohlc_data('BTC-USD', limit=100)
```

### 5. Handle Connection Errors

Always wrap API calls in try-except blocks:

```python
try:
    data = get_ohlc_data('BTC-USD')
except requests.exceptions.ConnectionError:
    print("Server is not running. Start it with: python server.py")
except Exception as e:
    print(f"Error: {e}")
```

## Performance Tips

1. **Use the cache**: The server caches DataProvider instances to avoid re-downloading data
2. **Batch requests**: Fetch multiple tickers in one request
3. **Local storage**: Data is stored in `data/` directory and reused
4. **Limit rows**: Use `limit` parameter to reduce response size
5. **Clear cache**: Use `/api/cache/clear` if you need to free memory

## Troubleshooting

### Server won't start

```bash
# Check if port 5001 is already in use
netstat -ano | findstr :5001

# Kill the process if needed
taskkill /PID <process_id> /F
```

### Data not updating

```bash
# Force refresh data
curl -X POST "http://localhost:5001/api/refresh/BTC-USD?resolution=1d&period=max"
```

### Connection refused

Make sure the server is running:
```bash
python server.py
```

Check the logs in `data_server.log` for errors.

---

For more information, see the [README.md](README.md) file.
