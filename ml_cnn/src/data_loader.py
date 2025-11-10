import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple


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
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()
    
    # DataFrames to store the result for this ticker
    all_data = []
    
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
        batch_data = yf.download(ticker, start=current_start.strftime('%Y-%m-%d'), 
                                 end=current_end.strftime('%Y-%m-%d'), auto_adjust=True, multi_level_index=False)
        
        if not batch_data.empty:
            all_data.append(batch_data)
        
        # Move to next batch, adding one day to avoid duplicate dates
        current_start = current_end + pd.DateOffset(days=1)
    
    # Combine all the batches
    if all_data:
        combined_data = pd.concat(all_data)
        # Remove duplicate indices in case of slight overlaps
        combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
        return combined_data.sort_index()
    else:
        return pd.DataFrame()


def load_es_and_btc_data(start_date: str = "2000-01-01", end_date: str = None, 
                         es_ticker: str = "ES", btc_ticker: str = "BTC-USD"):
    """
    Load EC (Ethereum Classic) and BTC (Bitcoin) data from Yahoo Finance with batch downloading.

    Args:
        start_date: Start date for data in 'YYYY-MM-DD' format
        end_date: End date for data in 'YYYY-MM-DD' format. If None, uses current date.
        ec_ticker: Ticker symbol for Ethereum Classic (default: "ETC-USD")
        btc_ticker: Ticker symbol for Bitcoin (default: "BTC-USD")

    Returns:
        tuple: (es_data, btc_data) as DataFrames containing OHLC data
    """
    # Download data from Yahoo Finance using batch processing
    es_data = download_ticker_data_batch(es_ticker, start_date, end_date)
    btc_data = download_ticker_data_batch(btc_ticker, start_date, end_date)

    # Ensure OHLC columns exist
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if not es_data.empty and col not in es_data.columns:
            raise ValueError(f"EC data missing required column: {col}")
        if not btc_data.empty and col not in btc_data.columns:
            raise ValueError(f"BTC data missing required column: {col}")

    # Remove any rows with missing values
    if not es_data.empty:
        es_data = es_data.dropna()
    if not btc_data.empty:
        btc_data = btc_data.dropna()

    return es_data, btc_data


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


def align_data_on_dates(es_data: pd.DataFrame, btc_data: pd.DataFrame):
    """
    Align EC and BTC data on common dates.
    
    Args:
        es_data: DataFrame containing ES data
        btc_data: DataFrame containing BTC data
        
    Returns:
        tuple: (aligned_es_data, aligned_btc_data) with common date indices
    """
    # Find common dates between the two datasets
    common_dates = es_data.index.intersection(btc_data.index)
    
    # Align both datasets to common dates
    aligned_es_data = es_data.loc[common_dates]
    aligned_btc_data = btc_data.loc[common_dates]
    
    return aligned_es_data, aligned_btc_data
