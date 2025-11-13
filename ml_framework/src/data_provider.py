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
    
    def load_yahoo(self, ticker: str, start_date: str, end_date: str, 
                   interval: str = '1d') -> pd.DataFrame:
        """
        Load data from Yahoo Finance.
        
        Args:
            ticker: Stock/crypto ticker symbol (e.g., 'BTC-USD', 'AAPL')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1d, 1h, 5m, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")
        
        print(f"Downloading {ticker} data from Yahoo Finance...")
        print(f"Period: {start_date} to {end_date}, Interval: {interval}")
        
        # Download data
        data = yf.download(ticker, start=start_date, end=end_date, 
                          interval=interval, progress=False)
        
        # Standardize column names
        data.columns = [col.lower() for col in data.columns]
        
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
