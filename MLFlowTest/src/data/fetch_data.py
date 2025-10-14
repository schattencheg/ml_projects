"""Data fetching utilities for market data."""

import yfinance as yf
import pandas as pd
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time

try:
    from src.utils import config, get_logger, EnvConfig
except ImportError:
    # Handle case when running as script directly
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from src.utils import config, get_logger, EnvConfig

logger = get_logger(__name__)

class DataProvider(ABC):
    """Abstract base class for data providers."""
    
    @abstractmethod
    def fetch_data(self, symbol: str, **kwargs) -> pd.DataFrame:
        """Fetch market data for a given symbol."""
        pass

class YahooFinanceProvider(DataProvider):
    """Yahoo Finance data provider."""
    
    def fetch_data(
        self, 
        symbol: str, 
        period: str = "2y", 
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching data for {symbol} from Yahoo Finance")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Standardize column names
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Add symbol column
            data['Symbol'] = symbol
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise

class AlphaVantageProvider(DataProvider):
    """Alpha Vantage data provider."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with API key.
        
        Args:
            api_key: Alpha Vantage API key. If None, uses environment variable.
        """
        self.api_key = api_key or EnvConfig.ALPHA_VANTAGE_API_KEY
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required")
        
        self.base_url = "https://www.alphavantage.co/query"
    
    def fetch_data(
        self, 
        symbol: str, 
        function: str = "TIME_SERIES_DAILY",
        outputsize: str = "full"
    ) -> pd.DataFrame:
        """Fetch data from Alpha Vantage.
        
        Args:
            symbol: Stock symbol
            function: API function (TIME_SERIES_DAILY, TIME_SERIES_WEEKLY, etc.)
            outputsize: Output size (compact or full)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching data for {symbol} from Alpha Vantage")
            
            params = {
                'function': function,
                'symbol': symbol,
                'outputsize': outputsize,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            
            if 'Note' in data:
                raise ValueError(f"API Limit: {data['Note']}")
            
            # Extract time series data
            time_series_key = None
            for key in data.keys():
                if 'Time Series' in key:
                    time_series_key = key
                    break
            
            if not time_series_key:
                raise ValueError("No time series data found in response")
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Standardize column names
            column_mapping = {
                '1. open': 'Open',
                '2. high': 'High', 
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            }
            df = df.rename(columns=column_mapping)
            df = df.astype(float)
            
            # Add symbol column
            df['Symbol'] = symbol
            
            logger.info(f"Successfully fetched {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise

class DataFetcher:
    """Main data fetcher class that manages multiple providers."""
    
    def __init__(self):
        """Initialize data fetcher with configured providers."""
        self.providers = {}
        
        # Initialize Yahoo Finance provider (always available)
        self.providers['yahoo'] = YahooFinanceProvider()
        
        # Initialize Alpha Vantage provider if API key is available
        if EnvConfig.ALPHA_VANTAGE_API_KEY:
            self.providers['alphavantage'] = AlphaVantageProvider()
        
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
    
    def fetch_symbol_data(
        self, 
        symbol: str, 
        provider: str = "yahoo",
        save_to_file: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """Fetch data for a single symbol.
        
        Args:
            symbol: Stock symbol
            provider: Data provider ('yahoo' or 'alphavantage')
            save_to_file: Whether to save data to file
            **kwargs: Additional arguments for the provider
            
        Returns:
            DataFrame with market data
        """
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not available")
        
        data = self.providers[provider].fetch_data(symbol, **kwargs)
        
        if save_to_file:
            filename = f"{symbol}_{provider}_data.csv"
            filepath = self.data_dir / filename
            data.to_csv(filepath)
            logger.info(f"Data saved to {filepath}")
        
        return data
    
    def fetch_multiple_symbols(
        self, 
        symbols: List[str], 
        provider: str = "yahoo",
        delay: float = 0.1,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            provider: Data provider
            delay: Delay between requests to avoid rate limiting
            **kwargs: Additional arguments for the provider
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        for symbol in symbols:
            try:
                data = self.fetch_symbol_data(symbol, provider, **kwargs)
                results[symbol] = data
                
                if delay > 0:
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {str(e)}")
                continue
        
        return results
    
    def load_saved_data(self, symbol: str, provider: str = "yahoo") -> pd.DataFrame:
        """Load previously saved data.
        
        Args:
            symbol: Stock symbol
            provider: Data provider
            
        Returns:
            DataFrame with market data
        """
        filename = f"{symbol}_{provider}_data.csv"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"No saved data found for {symbol}")
        
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logger.info(f"Loaded data from {filepath}")
        return data

def main():
    """Main function for command-line usage."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Fetch market data")
    parser.add_argument("--symbol", help="Stock symbol (default: AAPL)")
    parser.add_argument("--provider", default="yahoo", choices=["yahoo", "alphavantage"])
    parser.add_argument("--period", default="1mo", help="Data period (default: 1mo)")
    parser.add_argument("--interval", default="1d", help="Data interval (default: 1d)")
    
    # Parse args, but handle case where no symbol is provided
    if len(sys.argv) == 1:
        # No arguments provided, run with defaults
        print("No arguments provided. Running with default symbol: AAPL")
        symbol = "AAPL"
        provider = "yahoo"
        period = "1mo"
        interval = "1d"
    else:
        args = parser.parse_args()
        symbol = args.symbol or "AAPL"
        provider = args.provider
        period = args.period
        interval = args.interval
    
    print(f"Fetching data for {symbol}...")
    print(f"Provider: {provider}, Period: {period}, Interval: {interval}")
    
    try:
        fetcher = DataFetcher()
        data = fetcher.fetch_symbol_data(
            symbol=symbol,
            provider=provider,
            period=period,
            interval=interval
        )
        
        print(f"\n✅ Successfully fetched {len(data)} records for {symbol}")
        print(f"Date range: {data.index.min().date()} to {data.index.max().date()}")
        print(f"Columns: {list(data.columns)}")
        print("\nFirst 5 rows:")
        print(data.head())
        print(f"\nLast 5 rows:")
        print(data.tail())
        
    except Exception as e:
        print(f"❌ Error fetching data: {str(e)}")
        print("\nUsage examples:")
        print("  python src/data/fetch_data.py --symbol AAPL")
        print("  python src/data/fetch_data.py --symbol GOOGL --period 6mo")
        print("  python src/data/fetch_data.py --symbol TSLA --provider yahoo --period 1y")

if __name__ == "__main__":
    main()
