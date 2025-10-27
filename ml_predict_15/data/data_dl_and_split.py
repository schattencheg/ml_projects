import pandas as pd
import kagglehub
import os


def from_kaggle():
    os.makedirs('data', exist_ok=True)

    path_to_data = 'BTC-USD_data_1min.csv'
    #path_to_data = kagglehub.dataset_download("aklimarimi/bitcoin-historical-data-1min-interval", force_download=True)

    df = pd.read_csv(os.path.join('data', path_to_data))
    if 'Date' in df.columns:
        df.columns = [x if x != 'Date' else 'Timestamp' for x in df.columns]
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna()
    min_year = df.index.min().year
    max_year = df.index.max().year
    for year in range(min_year, max_year + 1):
        print(f'Processing year {year}')
        df_year = df[df.index.year == year]
        df_year.to_csv(os.path.join('data', f'btc_{year}.csv'))

def from_ccxt():
    import ccxt
    import datetime
    import time
    
    os.makedirs('data', exist_ok=True)
    
    exchange = ccxt.binance()
    symbol = 'BTC/USDT'
    timeframe = '1h'
    
    # BTC/USDT trading started on Binance around August 2017
    # Start from the earliest possible date
    start_date = datetime.datetime(2017, 8, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    start_timestamp = int(start_date.timestamp() * 1000)
    
    # End at current time
    end_timestamp = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
    
    print(f"Downloading {symbol} hourly data from {start_date} to now...")
    print(f"This may take several minutes due to API rate limits.\n")
    
    all_data = []
    current_timestamp = start_timestamp
    batch_count = 0
    
    while current_timestamp < end_timestamp:
        try:
            # Fetch OHLCV data (Binance typically returns 500-1000 candles per request)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_timestamp, limit=1000)
            
            if not ohlcv:
                print("No more data available.")
                break
            
            # Add to our collection
            all_data.extend(ohlcv)
            batch_count += 1
            
            # Update timestamp to the last candle + 1 hour
            last_timestamp = ohlcv[-1][0]
            current_timestamp = last_timestamp + 3600000  # Add 1 hour in milliseconds
            
            # Print progress
            last_date = datetime.datetime.fromtimestamp(last_timestamp / 1000, tz=datetime.timezone.utc)
            print(f"Batch {batch_count}: Downloaded {len(ohlcv)} candles. Last date: {last_date}")
            
            # Check if we've reached the end
            if last_timestamp >= end_timestamp:
                print("Reached current time.")
                break
            
            # Sleep to respect rate limits (Binance allows ~1200 requests per minute)
            time.sleep(0.1)  # 100ms delay between requests
            
        except ccxt.RateLimitExceeded:
            print("Rate limit exceeded. Waiting 60 seconds...")
            time.sleep(60)
        except Exception as e:
            print(f"Error: {e}")
            print("Waiting 5 seconds before retry...")
            time.sleep(5)
    
    # Convert to DataFrame
    print(f"\nTotal candles downloaded: {len(all_data)}")
    df = pd.DataFrame(all_data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)
    
    # Remove duplicates (in case of overlapping data)
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    
    # Save complete dataset
    output_file = os.path.join('data', 'btc_usdt_hourly_complete.csv')
    df.to_csv(output_file)
    print(f"\nData saved to: {output_file}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Total rows: {len(df)}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nLast few rows:")
    print(df.tail())
    
    return df

if __name__ == "__main__":
    from_ccxt()
