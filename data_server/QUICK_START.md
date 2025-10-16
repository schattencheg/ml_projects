# Data Provider Server - Quick Start Guide

## 🚀 Start the Server

### Option 1: Using Batch File (Easiest)
```bash
start_server.bat
```

### Option 2: Manual Start
```bash
pip install -r requirements.txt
python server.py
```

The server will start on **http://localhost:5001**

## 🌐 Access the Dashboard

Open your browser and go to:
```
http://localhost:5001
```

You'll see a beautiful web dashboard where you can:
- Check server status
- Request OHLC data for any ticker
- View data in JSON format
- See all available API endpoints

## 📊 Quick API Examples

### 1. Get Bitcoin Daily Data
```bash
http://localhost:5001/api/data/BTC-USD?resolution=1d&period=max&limit=100
```

### 2. Get Ethereum Hourly Data
```bash
http://localhost:5001/api/data/ETH-USD?resolution=1h&period=1mo&limit=50
```

### 3. Get SPY Daily Data
```bash
http://localhost:5001/api/data/SPY?resolution=1d&period=5y
```

### 4. Download CSV
```bash
http://localhost:5001/api/data/BTC-USD/csv?resolution=1d&period=max
```

## 🐍 Python Client Example

```python
import requests
import pandas as pd

BASE_URL = "http://localhost:5001"

# Get Bitcoin data
response = requests.get(f"{BASE_URL}/api/data/BTC-USD", params={
    'resolution': '1d',
    'period': 'max',
    'limit': 100
})

data = response.json()
df = pd.DataFrame(data['data'])
print(df.head())
```

## 🧪 Test the Server

Run the test client:
```bash
python test_client.py
```

This will test all API endpoints and show you example outputs.

## 📚 Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web dashboard |
| `/api` | GET | API documentation |
| `/api/health` | GET | Health check |
| `/api/instruments` | GET | List all instruments |
| `/api/data/<ticker>` | GET | Get OHLC data |
| `/api/data/<ticker>/csv` | GET | Download CSV |
| `/api/refresh/<ticker>` | POST | Refresh data |
| `/api/batch` | POST | Batch request |
| `/api/cache/clear` | POST | Clear cache |

## 🎯 Supported Instruments

### Cryptocurrencies
- BTC-USD, BTC-USDT
- ETH-USD, ETH-USDT
- DOGE-USDT
- ADA-USD, SOL-USD, XRP-USD

### Regular Market
- SPY, QQQ (ETFs)
- AAPL, MSFT, GOOGL, AMZN
- TSLA, NVDA, META, NFLX
- 6B, 6E (Futures)

## ⏰ Supported Timeframes

| Resolution | Description |
|------------|-------------|
| 1m | 1 minute |
| 5m | 5 minutes |
| 15m | 15 minutes |
| 30m | 30 minutes |
| 1h | 1 hour |
| 1d | 1 day |
| 1wk | 1 week |
| 1mo | 1 month |

## 📅 Supported Periods

| Period | Description |
|--------|-------------|
| 1d | 1 day |
| 5d | 5 days |
| 1mo | 1 month |
| 1y | 1 year |
| 5y | 5 years |
| max | Maximum available |

## 💾 Data Storage

Data is automatically stored in the `data/` directory:

```
data/
├── day_01/          # Daily data
│   ├── BTC-USD.csv
│   ├── ETH-USD.csv
│   └── SPY.csv
├── minute_01/       # 1-minute data
│   └── BTC-USDT.csv
└── hour_01/         # Hourly data
    └── ETH-USD.csv
```

## 🔧 Configuration

Edit `config.py` to customize:
- Server host and port
- Default resolution and period
- Supported instruments
- Cache settings

## 📖 Full Documentation

- **README.md** - Complete project documentation
- **API_GUIDE.md** - Detailed API usage guide with examples
- **test_client.py** - Test client with examples

## 🆘 Troubleshooting

### Server won't start
```bash
# Check if port 5001 is in use
netstat -ano | findstr :5001
```

### Dependencies missing
```bash
pip install -r requirements.txt
```

### Data not updating
```bash
# Force refresh via API
curl -X POST "http://localhost:5001/api/refresh/BTC-USD"
```

## 🎉 You're Ready!

The server is now running and ready to provide OHLC data for your trading strategies, backtesting, or analysis!

**Dashboard:** http://localhost:5001  
**API Info:** http://localhost:5001/api  
**Health Check:** http://localhost:5001/api/health

Happy trading! 📈
