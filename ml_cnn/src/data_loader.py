import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import Optional, Tuple
import yfinance as yf


def get_cache_path() -> str:
    """
    Get the path for the data cache directory.
    
    Returns:
        Path to the data cache directory
    """
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_cache_filename(ticker: str, start_date: str, end_date: str) -> str:
    """
    Generate a cache filename based on ticker and date range.
    
    Args:
        ticker: The ticker symbol
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        Filename for cached data
    """
    # Create a safe filename by replacing special characters
    safe_ticker = ticker.replace('-', '_').replace('.', '_').replace('/', '_')
    filename = f"{safe_ticker}.csv"
    return filename


def save_cached_data(data: pd.DataFrame, ticker: str, start_date: str, end_date: str):
    """
    Save downloaded data to cache.
    
    Args:
        data: DataFrame to save
        ticker: The ticker symbol
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
    """
    cache_dir = get_cache_path()
    filename = get_cache_filename(ticker, start_date, end_date)
    filepath = os.path.join(cache_dir, filename)
    data.to_csv(filepath)


def load_cached_data(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Load data from cache if it exists and is up to date.
    
    Args:
        ticker: The ticker symbol
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        Cached DataFrame if exists and is up to date, None otherwise
    """
    cache_dir = get_cache_path()
    filename = get_cache_filename(ticker, start_date, end_date)
    filepath = os.path.join(cache_dir, filename)
    
    if os.path.exists(filepath):
        print(f"Loading {ticker} data from cache...")
        cached_data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        cached_data.columns = cached_data.columns.get_level_values(0)
        cached_data['Date'] = pd.to_datetime(cached_data['Date'])
        return cached_data
    else:
        print(f"No cached data found for {ticker}, downloading...")
        return None


def download_ticker_data_batch(ticker: str, start_date: str, end_date: str, max_period: str = "2y") -> pd.DataFrame:
    """
    Download data for a ticker in batches to handle large time ranges.
    
    Args:
        ticker: The ticker symbol to download
        start_date: Start date for data in 'YYYY-MM-DD' format
        end_date: End date for data in 'YYYY-MM-DD' format
        max_period: Maximum period for each batch download (default: "2y")
        
    Returns:
        DataFrame containing the combined data
    """
    # First, check if data is already cached
    print(f"Downloading {ticker} data...")
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()

    # DataFrames to store the result for this ticker
    all_data = []

    cached_data = load_cached_data(ticker, start_date, end_date)
    if cached_data is not None:
        start = cached_data['Date'].max()
        all_data.append(cached_data)
    
    # Calculate batch size based on max_period
    if max_period == "2y":
        batch_size = pd.DateOffset(years=2)
    elif max_period == "1y":
        batch_size = pd.DateOffset(years=1)
    elif max_period == "6mo":
        batch_size = pd.DateOffset(months=6)
    elif max_period == "3mo":
        batch_size = pd.DateOffset(months=3)
    else:
        batch_size = pd.DateOffset(years=2)  # Default to 2 years
        
    current_start = start
    while current_start < end:
        current_end = min(current_start + batch_size, end)
        
        # Download data for this batch
        print(f"Downloading {ticker} data for {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}...")
        try:
            batch_data = yf.download(ticker, start=current_start.strftime('%Y-%m-%d'), end=current_end.strftime('%Y-%m-%d'), auto_adjust=True, progress=False)
            
            if not batch_data.empty:
                # Handle multi-level columns if present
                if isinstance(batch_data.columns, pd.MultiIndex):
                    batch_data.columns = batch_data.columns.get_level_values(0)
                
                # Reset index to make Date a column
                batch_data = batch_data.reset_index()
                all_data.append(batch_data)
                print(f"✓ Downloaded {len(batch_data)} rows for {ticker} ({current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')})")
            else:
                print(f"⚠ No data available for {ticker} in this period")
        except Exception as e:
            print(f"✗ Error downloading {ticker} data: {str(e)}")
        
        # Move to next batch, adding one day to avoid duplicate dates
        current_start = current_end + pd.DateOffset(days=1)
    
    # Combine all the batches
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicate dates in case of slight overlaps
        if 'Date' in combined_data.columns:
            combined_data = combined_data.drop_duplicates(subset=['Date'], keep='first')
            combined_data = combined_data.sort_values('Date').reset_index(drop=True)
        
        print(f"✓ Total {len(combined_data)} rows downloaded for {ticker}")
        
        # Save the data to cache for future use
        save_cached_data(combined_data, ticker, start_date, end_date)
        combined_data.columns = [x.lower() for x in combined_data.columns]
        if 'date' in combined_data.columns: 
            combined_data.rename(columns={'date': 'timestamp'}, inplace=True)
        return combined_data
    else:
        print(f"✗ No data downloaded for {ticker}")
        return pd.DataFrame()


def load_data(ticker: str, start_date: str = "2000-01-01", end_date: str = None):
    """
    Load data from Yahoo Finance with batch downloading.

    Args:
        ticker: Ticker symbol for data (default: "BTC-USD")
        start_date: Start date for data in 'YYYY-MM-DD' format
        end_date: End date for data in 'YYYY-MM-DD' format. If None, uses current date.

    Returns:
        tuple: (es_data, btc_data) as DataFrames containing OHLC data
    """
    # Download data from Yahoo Finance using batch processing
    data = download_ticker_data_batch(ticker, start_date, end_date)

    # Ensure OHLC columns exist
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if not data.empty and col not in data.columns:
            raise ValueError(f"Data missing required column: {col}")

    # Remove any rows with missing values
    if not data.empty:
        data = data.dropna()

    return data


def load_single_ticker_data(ticker: str, start_date: str = "2000-01-01", end_date: str = None) -> pd.DataFrame:
    """
    Load data for a single ticker from Yahoo Finance with batch downloading.

    Args:
        ticker: Ticker symbol to download
        start_date: Start date for data in 'YYYY-MM-DD' format
        end_date: End date for data in 'YYYY-MM-DD' format. If None, uses current date.

    Returns:
        DataFrame containing OHLC data
    """
    data = download_ticker_data_batch(ticker, start_date, end_date)
    
    # Ensure OHLC columns exist
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if not data.empty and col not in data.columns:
            raise ValueError(f"Data for {ticker} missing required column: {col}")

    # Remove any rows with missing values
    if not data.empty:
        data = data.dropna()

    return data


def align_data_on_dates(df: pd.DataFrame):
    """
    Align EC and BTC data on common dates.
    
    Args:
        df: DataFrame containing data
        
    Returns:
        df: DataFrame with common date indices
    """
    # Find common dates between the two datasets
    common_dates = df.index.intersection(df.index)
    
    # Align both datasets to common dates
    aligned_df = df.loc[common_dates]
    
    return aligned_df
