# Detailed Instruments API Documentation

## Overview

The `/api/local-instruments-detailed` endpoint provides comprehensive information about all locally cached instruments, including their type (crypto or stock), available resolutions, time ranges, and file statistics.

## Endpoint

**GET** `/api/local-instruments-detailed`

Returns detailed information about all instruments stored locally.

## Response Format

```json
{
  "instruments": [
    {
      "ticker": "BTC-USD",
      "type": "crypto",
      "resolutions": {
        "1d": {
          "rows": 2847,
          "start_date": "2014-09-17",
          "end_date": "2025-10-16",
          "file_size_bytes": 123456,
          "file_size_mb": 0.12,
          "columns": ["Open", "High", "Low", "Close", "Volume"]
        },
        "1h": {
          "rows": 68328,
          "start_date": "2024-10-16",
          "end_date": "2025-10-16",
          "file_size_bytes": 2345678,
          "file_size_mb": 2.24,
          "columns": ["Open", "High", "Low", "Close", "Volume"]
        }
      }
    },
    {
      "ticker": "SPY",
      "type": "stock",
      "resolutions": {
        "1d": {
          "rows": 1258,
          "start_date": "2020-10-16",
          "end_date": "2025-10-16",
          "file_size_bytes": 54321,
          "file_size_mb": 0.05,
          "columns": ["Open", "High", "Low", "Close", "Volume"]
        }
      }
    }
  ],
  "summary": {
    "total_instruments": 2,
    "crypto_count": 1,
    "stock_count": 1,
    "resolutions": ["1d", "1h"]
  }
}
```

## Response Fields

### Instruments Array

Each instrument object contains:

| Field | Type | Description |
|-------|------|-------------|
| `ticker` | string | Ticker symbol (e.g., "BTC-USD", "SPY") |
| `type` | string | Instrument type: "crypto" or "stock" |
| `resolutions` | object | Dictionary of available resolutions |

### Resolution Object

For each resolution, the following information is provided:

| Field | Type | Description |
|-------|------|-------------|
| `rows` | integer | Number of data rows |
| `start_date` | string | First date in the dataset |
| `end_date` | string | Last date in the dataset |
| `file_size_bytes` | integer | File size in bytes |
| `file_size_mb` | float | File size in megabytes (rounded to 2 decimals) |
| `columns` | array | List of column names in the CSV file |

### Summary Object

| Field | Type | Description |
|-------|------|-------------|
| `total_instruments` | integer | Total number of instruments |
| `crypto_count` | integer | Number of cryptocurrency instruments |
| `stock_count` | integer | Number of stock/ETF instruments |
| `resolutions` | array | List of all unique resolutions available |

## Instrument Type Detection

The API automatically detects instrument types based on ticker patterns:

### Crypto Patterns
Tickers containing these patterns are classified as "crypto":
- BTC, ETH, DOGE, ADA, SOL, XRP, LTC, DOT, LINK, UNI
- USDT, USDC, BNB, MATIC, AVAX, SHIB, TRX, ATOM, XLM

### Stock
All other tickers are classified as "stock" (includes stocks, ETFs, futures, etc.)

## Use Cases

### 1. Dashboard Display

Show all available instruments with their data coverage:

```javascript
fetch('http://localhost:5001/api/local-instruments-detailed')
  .then(response => response.json())
  .then(data => {
    data.instruments.forEach(inst => {
      console.log(`${inst.ticker} (${inst.type})`);
      Object.entries(inst.resolutions).forEach(([res, info]) => {
        console.log(`  ${res}: ${info.rows} rows, ${info.start_date} to ${info.end_date}`);
      });
    });
  });
```

### 2. Data Coverage Report

Generate a report of data availability:

```python
import requests
import pandas as pd

response = requests.get('http://localhost:5001/api/local-instruments-detailed')
data = response.json()

# Create DataFrame for analysis
rows = []
for inst in data['instruments']:
    for resolution, info in inst['resolutions'].items():
        rows.append({
            'ticker': inst['ticker'],
            'type': inst['type'],
            'resolution': resolution,
            'rows': info['rows'],
            'start_date': info['start_date'],
            'end_date': info['end_date'],
            'size_mb': info['file_size_mb']
        })

df = pd.DataFrame(rows)
print(df.to_string())

# Summary statistics
print(f"\nTotal instruments: {data['summary']['total_instruments']}")
print(f"Crypto: {data['summary']['crypto_count']}")
print(f"Stocks: {data['summary']['stock_count']}")
print(f"Total size: {df['size_mb'].sum():.2f} MB")
```

### 3. Filter by Type

Get only crypto or stock instruments:

```python
import requests

response = requests.get('http://localhost:5001/api/local-instruments-detailed')
data = response.json()

# Filter crypto only
crypto_instruments = [inst for inst in data['instruments'] if inst['type'] == 'crypto']
print(f"Found {len(crypto_instruments)} crypto instruments")

# Filter stocks only
stock_instruments = [inst for inst in data['instruments'] if inst['type'] == 'stock']
print(f"Found {len(stock_instruments)} stock instruments")
```

### 4. Check Data Completeness

Find instruments with incomplete data:

```python
import requests
from datetime import datetime, timedelta

response = requests.get('http://localhost:5001/api/local-instruments-detailed')
data = response.json()

# Check for outdated data (>7 days old)
outdated = []
for inst in data['instruments']:
    for resolution, info in inst['resolutions'].items():
        end_date = datetime.fromisoformat(info['end_date'])
        age_days = (datetime.now() - end_date).days
        
        if age_days > 7:
            outdated.append({
                'ticker': inst['ticker'],
                'resolution': resolution,
                'age_days': age_days,
                'end_date': info['end_date']
            })

if outdated:
    print("Outdated data found:")
    for item in outdated:
        print(f"  {item['ticker']} ({item['resolution']}): {item['age_days']} days old")
```

### 5. Storage Analysis

Analyze storage usage by instrument type:

```python
import requests

response = requests.get('http://localhost:5001/api/local-instruments-detailed')
data = response.json()

crypto_size = 0
stock_size = 0

for inst in data['instruments']:
    total_size = sum(info['file_size_mb'] for info in inst['resolutions'].values())
    
    if inst['type'] == 'crypto':
        crypto_size += total_size
    else:
        stock_size += total_size

print(f"Crypto data: {crypto_size:.2f} MB")
print(f"Stock data: {stock_size:.2f} MB")
print(f"Total: {crypto_size + stock_size:.2f} MB")
```

## Example Response

Here's a complete example response with multiple instruments:

```json
{
  "instruments": [
    {
      "ticker": "BTC-USD",
      "type": "crypto",
      "resolutions": {
        "1d": {
          "rows": 2847,
          "start_date": "2014-09-17",
          "end_date": "2025-10-16",
          "file_size_bytes": 123456,
          "file_size_mb": 0.12,
          "columns": ["Open", "High", "Low", "Close", "Volume"]
        },
        "1h": {
          "rows": 68328,
          "start_date": "2024-10-16",
          "end_date": "2025-10-16",
          "file_size_bytes": 2345678,
          "file_size_mb": 2.24,
          "columns": ["Open", "High", "Low", "Close", "Volume"]
        }
      }
    },
    {
      "ticker": "ETH-USD",
      "type": "crypto",
      "resolutions": {
        "1d": {
          "rows": 1856,
          "start_date": "2017-11-09",
          "end_date": "2025-10-16",
          "file_size_bytes": 98765,
          "file_size_mb": 0.09,
          "columns": ["Open", "High", "Low", "Close", "Volume"]
        }
      }
    },
    {
      "ticker": "SPY",
      "type": "stock",
      "resolutions": {
        "1d": {
          "rows": 1258,
          "start_date": "2020-10-16",
          "end_date": "2025-10-16",
          "file_size_bytes": 54321,
          "file_size_mb": 0.05,
          "columns": ["Open", "High", "Low", "Close", "Volume"]
        }
      }
    },
    {
      "ticker": "AAPL",
      "type": "stock",
      "resolutions": {
        "1d": {
          "rows": 1258,
          "start_date": "2020-10-16",
          "end_date": "2025-10-16",
          "file_size_bytes": 54987,
          "file_size_mb": 0.05,
          "columns": ["Open", "High", "Low", "Close", "Volume"]
        }
      }
    }
  ],
  "summary": {
    "total_instruments": 4,
    "crypto_count": 2,
    "stock_count": 2,
    "resolutions": ["1d", "1h"]
  }
}
```

## Error Handling

### Error Response Format

```json
{
  "error": "Error message description",
  "instruments": [],
  "summary": {
    "total_instruments": 0,
    "crypto_count": 0,
    "stock_count": 0,
    "resolutions": []
  }
}
```

### Common Errors

1. **Data directory not found**: Returns empty instruments list
2. **File read error**: Individual resolution will have `"error"` field instead of data
3. **Permission denied**: HTTP 500 with error message

## Performance Considerations

- **Response Time**: Depends on number of instruments and files
  - ~100ms for 10 instruments
  - ~500ms for 50 instruments
  - ~1s for 100+ instruments
  
- **Caching**: Consider caching the response if calling frequently

- **File I/O**: Reads CSV headers and row counts (not full data)

## Integration Examples

### React Component

```javascript
import React, { useEffect, useState } from 'react';

function InstrumentsList() {
  const [data, setData] = useState(null);
  
  useEffect(() => {
    fetch('http://localhost:5001/api/local-instruments-detailed')
      .then(res => res.json())
      .then(setData);
  }, []);
  
  if (!data) return <div>Loading...</div>;
  
  return (
    <div>
      <h2>Local Instruments ({data.summary.total_instruments})</h2>
      <p>Crypto: {data.summary.crypto_count}, Stocks: {data.summary.stock_count}</p>
      
      {data.instruments.map(inst => (
        <div key={inst.ticker}>
          <h3>{inst.ticker} ({inst.type})</h3>
          <ul>
            {Object.entries(inst.resolutions).map(([res, info]) => (
              <li key={res}>
                {res}: {info.rows} rows ({info.start_date} to {info.end_date})
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}
```

### Python Data Analysis

```python
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Fetch data
response = requests.get('http://localhost:5001/api/local-instruments-detailed')
data = response.json()

# Create DataFrame
records = []
for inst in data['instruments']:
    for resolution, info in inst['resolutions'].items():
        records.append({
            'ticker': inst['ticker'],
            'type': inst['type'],
            'resolution': resolution,
            'rows': info['rows'],
            'size_mb': info['file_size_mb'],
            'days': (pd.to_datetime(info['end_date']) - 
                    pd.to_datetime(info['start_date'])).days
        })

df = pd.DataFrame(records)

# Plot storage by type
df.groupby('type')['size_mb'].sum().plot(kind='bar', title='Storage by Type')
plt.ylabel('Size (MB)')
plt.show()

# Plot data coverage
df.groupby('ticker')['days'].max().sort_values(ascending=False).plot(
    kind='barh', title='Data Coverage (Days)'
)
plt.xlabel('Days')
plt.show()
```

## Comparison with Other Endpoints

| Endpoint | Purpose | Response Size | Speed |
|----------|---------|---------------|-------|
| `/api/instruments` | List all supported tickers | Small | Fast |
| `/api/local-instruments` | List locally cached tickers | Small | Fast |
| `/api/local-instruments-detailed` | Detailed info with time ranges | Large | Moderate |
| `/api/local-data/<ticker>` | Info for specific ticker | Small | Fast |

## Best Practices

1. **Cache Results**: Response doesn't change frequently, cache for 5-10 minutes
2. **Filter Client-Side**: Get all data once, filter in your application
3. **Monitor Size**: Watch for large response sizes with many instruments
4. **Use for Dashboards**: Perfect for overview/monitoring dashboards
5. **Combine with Other APIs**: Use with `/api/data/<ticker>` for complete workflow

---

**Summary**: The detailed instruments API provides a comprehensive view of all locally cached data, making it easy to build dashboards, reports, and monitoring tools for your data provider server.
