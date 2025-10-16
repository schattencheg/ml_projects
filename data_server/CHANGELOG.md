# Changelog

## [1.2.0] - 2025-10-16

### Refactored
- **Major Code Refactoring**: Moved all data-related logic from `server.py` into `DataProvider` class
- Added 4 new static methods to `DataProvider`:
  - `get_local_instruments()` - List all locally available instruments
  - `get_all_local_tickers()` - Get unique list of all tickers
  - `check_local_data_exists()` - Check if data exists locally
  - `get_local_data_info()` - Get detailed info about local data
- Reduced `server.py` complexity by ~40 lines
- Improved separation of concerns (server handles HTTP, DataProvider handles data)
- Enhanced code reusability and testability

### Added
- **New API Endpoint**: `/api/local-data/<ticker>`
  - Get detailed information about locally cached data
  - Returns file size, row count, date range, and columns
  - Useful for checking data availability before loading

### Changed
- `/api/local-instruments` now uses `DataProvider.get_local_instruments()`
- `/api/data/<ticker>` now uses `DataProvider.check_local_data_exists()`
- Improved logging messages for better debugging

---

## [1.1.0] - 2025-10-16

### Added
- **New API Endpoint**: `/api/local-instruments`
  - Lists all locally cached instruments from the data folder
  - Returns instruments organized by resolution
  - Provides a complete list of all available tickers across all timeframes
  
- **Local-Only Mode**: New `local_only` parameter for `/api/data/<ticker>` endpoint
  - When set to `true`, prevents downloading new data
  - Only uses locally cached data
  - Returns 404 error with helpful message if data not found locally
  
- **Enhanced Web Dashboard**:
  - Dropdown list automatically populated with locally available instruments
  - Checkbox to enable "local-only mode" (prevents downloads)
  - Ticker selection from dropdown auto-fills the ticker input field
  - Real-time updates of local instruments list (refreshes every 60 seconds)
  - Improved UI with better form layout and helper text

### Changed
- Updated API documentation to include new endpoint and local_only parameter
- Enhanced form submission to include local_only flag
- Improved JavaScript to handle local instrument selection

### Technical Details

#### New Endpoint Response Format
```json
{
  "by_resolution": {
    "1d": ["BTC-USD", "ETH-USD", "SPY"],
    "1h": ["BTC-USD", "ETH-USD"],
    "1m": ["BTC-USDT"]
  },
  "all_tickers": ["BTC-USD", "BTC-USDT", "ETH-USD", "SPY"],
  "total": 4,
  "resolutions": ["1d", "1h", "1m"]
}
```

#### Local-Only Mode Usage
```bash
# API call with local_only=true
GET /api/data/BTC-USD?resolution=1d&period=max&local_only=true

# Error response if data not found
{
  "error": "No local data found for BTC-USD with resolution 1d",
  "ticker": "BTC-USD",
  "resolution": "1d",
  "local_only": true,
  "suggestion": "Disable local-only mode to download data"
}
```

#### Dashboard Features
- **Dropdown Selection**: Choose from existing local instruments
- **Local-Only Checkbox**: Prevent automatic downloads
- **Auto-refresh**: Local instruments list updates every 60 seconds
- **Visual Feedback**: Helper text explains each feature

### Use Cases

1. **Offline Mode**: Work with cached data when internet is unavailable
2. **Performance**: Avoid download delays when data already exists
3. **Data Management**: See what data is already cached locally
4. **Testing**: Use local data for faster development/testing cycles

---

## [1.0.0] - 2025-10-16

### Initial Release
- Flask REST API server for OHLC data
- Support for crypto and regular market instruments
- Multiple timeframes and periods
- Local data caching
- CSV download capability
- Batch requests
- Web dashboard
- Comprehensive documentation
