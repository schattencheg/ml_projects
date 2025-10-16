# Data Provider Server

A localhost REST API server that provides OHLC (Open, High, Low, Close) data for crypto and regular market instruments across different timeframes.

## Features

- 🚀 **REST API** - Easy-to-use HTTP endpoints
- 💰 **Crypto & Stocks** - Support for cryptocurrencies and traditional market instruments
- ⏰ **Multiple Timeframes** - From 1-minute to monthly data
- 💾 **Local Caching** - Data stored locally in CSV format
- 🔄 **Auto-refresh** - Automatic data downloading if not available locally
- 📊 **Batch Requests** - Get data for multiple instruments at once
- 📁 **CSV Export** - Download data as CSV files

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Server

```bash
python server.py
```

The server will start on `http://localhost:5001`

### 3. Test the API

Open your browser and navigate to:
```
http://localhost:5001
```

You'll see the API documentation with all available endpoints.

## API Endpoints

### Health Check
```
GET /api/health
```
Check if the server is running.

### List Instruments
```
GET /api/instruments
```
Get all available crypto and regular market instruments.

**Response:**
```json
{
  "crypto": ["BTC-USD", "ETH-USD", "DOGE-USDT", ...],
  "regular_market": ["SPY", "QQQ", "AAPL", ...],
  "total": 16
}
```

### List Resolutions
```
GET /api/resolutions
```
Get all available data resolutions (timeframes).

### List Periods
```
GET /api/periods
```
Get all available data periods.

### Get OHLC Data
```
GET /api/data/<ticker>?resolution=1d&period=max&limit=100
```

**Parameters:**
- `ticker` (required): Instrument ticker (e.g., BTC-USD, SPY)
- `resolution` (optional): Data resolution - `1m`, `5m`, `15m`, `30m`, `1h`, `1d`, `1wk`, `1mo` (default: `1d`)
- `period` (optional): Data period - `1d`, `5d`, `1mo`, `1y`, `5y`, `max` (default: `max`)
- `start_date` (optional): Start date in YYYY-MM-DD format
- `end_date` (optional): End date in YYYY-MM-DD format
- `limit` (optional): Limit number of rows returned

**Example:**
```bash
# Get Bitcoin daily data for the last 100 days
curl "http://localhost:5001/api/data/BTC-USD?resolution=1d&period=max&limit=100"

# Get Ethereum hourly data for 1 year
curl "http://localhost:5001/api/data/ETH-USD?resolution=1h&period=1y"

# Get SPY daily data
curl "http://localhost:5001/api/data/SPY?resolution=1d&period=5y"
```

**Response:**
```json
{
  "ticker": "BTC-USD",
  "resolution": "1d",
  "period": "max",
  "rows": 100,
  "start_date": "2024-07-08",
  "end_date": "2024-10-16",
  "columns": ["date", "open", "high", "low", "close", "volume"],
  "data": [
    {
      "date": "2024-07-08",
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

### Download CSV
```
GET /api/data/<ticker>/csv?resolution=1d&period=max
```

Download OHLC data as a CSV file.

**Example:**
```bash
curl "http://localhost:5001/api/data/BTC-USD/csv?resolution=1d&period=max" -o btc_data.csv
```

### Refresh Data
```
POST /api/refresh/<ticker>?resolution=1d&period=max
```

Force refresh data for a specific ticker (downloads fresh data from source).

**Example:**
```bash
curl -X POST "http://localhost:5001/api/refresh/BTC-USD?resolution=1d&period=max"
```

### Batch Request
```
POST /api/batch
Content-Type: application/json

{
  "tickers": ["BTC-USD", "ETH-USD", "SPY"],
  "resolution": "1d",
  "period": "1y",
  "limit": 100
}
```

Get data for multiple tickers in a single request.

**Example:**
```bash
curl -X POST "http://localhost:5001/api/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["BTC-USD", "ETH-USD", "SPY"],
    "resolution": "1d",
    "period": "1y",
    "limit": 100
  }'
```

### Clear Cache
```
POST /api/cache/clear
```

Clear the in-memory data provider cache.

## Supported Instruments

### Cryptocurrencies
- BTC-USD, BTC-USDT
- ETH-USD, ETH-USDT
- DOGE-USDT
- ADA-USD
- SOL-USD
- XRP-USD

### Regular Market
- SPY, QQQ (ETFs)
- AAPL, MSFT, GOOGL, AMZN (Tech stocks)
- TSLA, NVDA, META, NFLX
- 6B, 6E (Futures)

## Supported Resolutions

| Resolution | Description |
|------------|-------------|
| 1m         | 1 minute    |
| 2m         | 2 minutes   |
| 5m         | 5 minutes   |
| 15m        | 15 minutes  |
| 30m        | 30 minutes  |
| 60m        | 60 minutes  |
| 90m        | 90 minutes  |
| 1h         | 1 hour      |
| 1d         | 1 day       |
| 5d         | 5 days      |
| 1wk        | 1 week      |
| 1mo        | 1 month     |
| 3mo        | 3 months    |

## Supported Periods

| Period | Description        |
|--------|-------------------|
| 1d     | 1 day             |
| 5d     | 5 days            |
| 1mo    | 1 month           |
| 3mo    | 3 months          |
| 6mo    | 6 months          |
| 1y     | 1 year            |
| 2y     | 2 years           |
| 5y     | 5 years           |
| 10y    | 10 years          |
| ytd    | Year to date      |
| max    | Maximum available |

## Data Storage

Data is stored locally in the `data/` directory, organized by resolution:

```
data/
├── day_01/
│   ├── BTC-USD.csv
│   ├── ETH-USD.csv
│   └── SPY.csv
├── minute_01/
│   └── BTC-USDT.csv
└── hour_01/
    └── ETH-USD.csv
```

## Configuration

Edit `config.py` to customize:

- Server host and port
- Default resolution and period
- Supported instruments
- Cache settings
- Logging configuration

## Python Client Example

```python
import requests
import pandas as pd

# Base URL
BASE_URL = "http://localhost:5001"

# Get Bitcoin daily data
response = requests.get(f"{BASE_URL}/api/data/BTC-USD", params={
    'resolution': '1d',
    'period': 'max',
    'limit': 100
})

data = response.json()
df = pd.DataFrame(data['data'])
print(df.head())

# Batch request for multiple tickers
response = requests.post(f"{BASE_URL}/api/batch", json={
    'tickers': ['BTC-USD', 'ETH-USD', 'SPY'],
    'resolution': '1d',
    'period': '1y',
    'limit': 50
})

batch_data = response.json()
for ticker, info in batch_data['results'].items():
    print(f"{ticker}: {info['rows']} rows")
```

## Integration with DataProvider

The server uses the existing `DataProvider` class for data management:

```python
from src.Data.DataProvider import DataProvider
from src.Data.enums import DataResolution, DataPeriod

# Create a provider
provider = DataProvider(
    tickers=['BTC-USD'],
    resolution=DataResolution.DAY_01,
    period=DataPeriod.YEAR_MAX
)

# Request data (downloads if not available)
df = provider.data_request_by_ticker('BTC-USD')

# Load from local cache
df = provider.data_load_by_ticker('BTC-USD')

# Save to local cache
provider.data_save_by_ticker('BTC-USD')
```

## Logging

Logs are written to both console and `data_server.log` file.

## Error Handling

The API returns appropriate HTTP status codes:
- `200 OK` - Success
- `400 Bad Request` - Invalid parameters
- `500 Internal Server Error` - Server error

Error responses include details:
```json
{
  "error": "Error message",
  "ticker": "BTC-USD"
}
```

## Development

### Running in Development Mode

```bash
python server.py
```

The server runs with Flask's debug mode enabled for development.

### Adding New Instruments

Edit `config.py` and add instruments to `CRYPTO_INSTRUMENTS` or `REGULAR_MARKET_INSTRUMENTS`:

```python
CRYPTO_INSTRUMENTS = [
    'BTC-USD', 'ETH-USD', 'YOUR-NEW-CRYPTO'
]
```

## License

MIT License

## Support

For issues or questions, please check the logs in `data_server.log`.
