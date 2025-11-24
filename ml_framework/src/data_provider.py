"""
DataProvider - Handles data loading, validation, and basic preprocessing.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Tuple
from pathlib import Path


class DataProvider:
    """
    Provides data loading and basic preprocessing functionality.
    
    Supports loading from:
    - CSV files
    - Yahoo Finance (via yfinance)
    - Custom data sources
    """
    
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize DataProvider.
        
        Args:
            data_dir: Directory for storing/loading data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data = None
        
    def load_csv(self, filepath: str, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv()
            
        Returns:
            DataFrame with loaded data
        """
        print(f"Loading data from {filepath}...")
        
        # Default parameters
        default_params = {
            'parse_dates': True,
            'index_col': 0
        }
        default_params.update(kwargs)
        
        self.data = pd.read_csv(filepath, **default_params)
        print(f"✓ Loaded {len(self.data)} rows, {len(self.data.columns)} columns")
        
        return self.data
    
    def _get_cache_filename(self, ticker: str, interval: str) -> Path:
        """
        Get cache filename for ticker and interval.
        
        Args:
            ticker: Ticker symbol
            interval: Data interval
            
        Returns:
            Path to cache file
        """
        safe_ticker = ticker.replace('/', '_').replace('-', '_')
        return self.data_dir / f"{safe_ticker}_{interval}.csv"
    
    def _load_cached_data(self, cache_file: Path) -> Optional[pd.DataFrame]:
        """
        Load data from cache file if it exists.
        
        Args:
            cache_file: Path to cache file
            
        Returns:
            DataFrame if cache exists, None otherwise
        """
        if cache_file.exists():
            try:
                df = pd.read_csv(cache_file, parse_dates=True, index_col=0)
                df.columns = [col.lower() for col in df.columns]
                return df
            except Exception as e:
                print(f"⚠ Warning: Could not load cache file: {e}")
                return None
        return None
    
    def _save_to_cache(self, df: pd.DataFrame, cache_file: Path):
        """
        Save data to cache file.
        
        Args:
            df: DataFrame to save
            cache_file: Path to cache file
        """
        try:
            df.to_csv(cache_file)
            print(f"✓ Data cached to {cache_file}")
        except Exception as e:
            print(f"⚠ Warning: Could not save cache: {e}")
    
    def load_yahoo(self, ticker: str, start_date: str, end_date: str, 
                   interval: str = '1d', use_cache: bool = True) -> pd.DataFrame:
        """
        Load data from Yahoo Finance with smart caching.
        
        This method will:
        1. Check if data exists in cache
        2. Load cached data if available
        3. Download only missing data if needed
        4. Merge cached and new data
        5. Save updated data to cache
        
        Args:
            ticker: Stock/crypto ticker symbol (e.g., 'BTC-USD', 'AAPL')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1d, 1h, 5m, etc.)
            use_cache: Whether to use cached data (default: True)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")
        
        from datetime import datetime
        
        # Convert dates to datetime
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Get cache file path
        cache_file = self._get_cache_filename(ticker, interval)
        
        # Try to load cached data
        cached_data = None
        if use_cache:
            cached_data = self._load_cached_data(cache_file)
        
        if cached_data is not None and len(cached_data) > 0:
            print(f"✓ Found cached data: {len(cached_data)} rows")
            print(f"  Cached range: {cached_data.index[0].date()} to {cached_data.index[-1].date()}")
            
            # Check if cached data covers the requested range
            cached_start = cached_data.index[0]
            cached_end = cached_data.index[-1]
            
            # Determine what data needs to be downloaded
            need_download_before = start_dt < cached_start
            need_download_after = end_dt > cached_end
            
            if not need_download_before and not need_download_after:
                # Cache covers the entire range
                print(f"✓ Cache covers requested range, no download needed")
                # Filter to requested range
                self.data = cached_data.loc[start_dt:end_dt]
                print(f"✓ Loaded {len(self.data)} rows from cache")
                return self.data
            
            # Need to download additional data
            new_data_parts = []
            
            if need_download_before:
                print(f"Downloading data before {cached_start.date()}...")
                before_data = yf.download(ticker, start=start_date, 
                                         end=cached_start.strftime('%Y-%m-%d'),
                                         interval=interval, progress=False)
                if len(before_data) > 0:
                    before_data.columns = [col.lower() for col in before_data.columns]
                    new_data_parts.append(before_data)
                    print(f"✓ Downloaded {len(before_data)} rows (before cache)")
            
            if need_download_after:
                print(f"Downloading data after {cached_end.date()}...")
                after_data = yf.download(ticker, 
                                        start=(cached_end + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                                        end=end_date,
                                        interval=interval, progress=False)
                if len(after_data) > 0:
                    after_data.columns = [col.lower() for col in after_data.columns]
                    new_data_parts.append(after_data)
                    print(f"✓ Downloaded {len(after_data)} rows (after cache)")
            
            # Merge all data
            all_data = pd.concat([*new_data_parts, cached_data], axis=0)
            all_data = all_data[~all_data.index.duplicated(keep='last')]
            all_data = all_data.sort_index()
            
            # Save merged data to cache
            self._save_to_cache(all_data, cache_file)
            
            # Filter to requested range
            self.data = all_data.loc[start_dt:end_dt]
            print(f"✓ Total data: {len(self.data)} rows")
            
        else:
            # No cache or cache disabled, download all data
            print(f"Downloading {ticker} data from Yahoo Finance...")
            print(f"Period: {start_date} to {end_date}, Interval: {interval}")
            
            data = yf.download(ticker, start=start_date, end=end_date, 
                              interval=interval, progress=False)
            data.columns = data.columns.get_level_values(0)
            
            if len(data) == 0:
                raise ValueError(f"No data downloaded for {ticker}")
            
            # Standardize column names
            data.columns = [col.lower() for col in data.columns]
            
            # Save to cache
            if use_cache:
                self._save_to_cache(data, cache_file)
            
            self.data = data
            print(f"✓ Downloaded {len(data)} rows")
        
        return self.data
    
    def validate_data(self, df: Optional[pd.DataFrame] = None) -> bool:
        """
        Validate OHLCV data format.
        
        Args:
            df: DataFrame to validate (uses self.data if None)
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        if df is None:
            df = self.data
            
        if df is None:
            raise ValueError("No data to validate")
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for NaN values
        nan_counts = df[required_cols].isna().sum()
        if nan_counts.any():
            print(f"⚠ Warning: Found NaN values:\n{nan_counts[nan_counts > 0]}")
        
        # Check data types
        for col in required_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column '{col}' must be numeric")
        
        print("✓ Data validation passed")
        return True
    
    def clean_data(self, df: Optional[pd.DataFrame] = None, 
                   method: str = 'drop') -> pd.DataFrame:
        """
        Clean data by handling missing values.
        
        Args:
            df: DataFrame to clean (uses self.data if None)
            method: Cleaning method ('drop', 'ffill', 'bfill', 'interpolate')
            
        Returns:
            Cleaned DataFrame
        """
        if df is None:
            df = self.data.copy()
        else:
            df = df.copy()
        
        initial_rows = len(df)
        
        if method == 'drop':
            df = df.dropna()
        elif method == 'ffill':
            df = df.fillna(method='ffill')
        elif method == 'bfill':
            df = df.fillna(method='bfill')
        elif method == 'interpolate':
            df = df.interpolate()
        else:
            raise ValueError(f"Unknown cleaning method: {method}")
        
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            print(f"✓ Cleaned data: removed {removed_rows} rows with NaN values")
        
        self.data = df
        return df
    
    def split_data(self, df: Optional[pd.DataFrame] = None, 
                   train_ratio: float = 0.7, 
                   val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: DataFrame to split (uses self.data if None)
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set (test = 1 - train - val)
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if df is None:
            df = self.data
        
        if df is None:
            raise ValueError("No data to split")
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        print(f"✓ Data split:")
        print(f"  Train: {len(train_df)} rows ({train_ratio*100:.1f}%)")
        print(f"  Val:   {len(val_df)} rows ({val_ratio*100:.1f}%)")
        print(f"  Test:  {len(test_df)} rows ({(1-train_ratio-val_ratio)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def get_data(self) -> pd.DataFrame:
        """Get the current data."""
        if self.data is None:
            raise ValueError("No data loaded. Use load_csv() or load_yahoo() first.")
        return self.data
    
    def save_data(self, filepath: str, df: Optional[pd.DataFrame] = None):
        """
        Save data to CSV file.
        
        Args:
            filepath: Output file path
            df: DataFrame to save (uses self.data if None)
        """
        if df is None:
            df = self.data
        
        if df is None:
            raise ValueError("No data to save")
        
        df.to_csv(filepath)
        print(f"✓ Data saved to {filepath}")
