import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import os


class DataLoader:
    """
    Class to handle loading and preprocessing of OHLC financial data
    """
    
    def __init__(self, data_path: str):
        """
        Initialize DataLoader
        
        Args:
            data_path: Path to the raw data file
        """
        self.data_path = data_path
        self.data = None
    
    def load_data(self) -> pd.DataFrame:
        """
        Load OHLC data from CSV file
        
        Expected format:
        - Date column (parsed as datetime)
        - Open, High, Low, Close columns
        - Volume column (optional)
        
        Returns:
            DataFrame with OHLC data
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # Load the data
        df = pd.read_csv(self.data_path, parse_dates=['Date'])
        df = df.sort_values('Date')
        
        # Validate required columns
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        self.data = df
        return df
    
    def split_by_year(self) -> Dict[int, pd.DataFrame]:
        """
        Split the data by year
        
        Returns:
            Dictionary with year as key and DataFrame for that year as value
        """
        if self.data is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")
        
        yearly_data = {}
        for year in sorted(self.data['Date'].dt.year.unique()):
            yearly_data[year] = self.data[self.data['Date'].dt.year == year].copy()
        
        return yearly_data
    
    def create_features(self, df: pd.DataFrame, lookback_window: int = 10) -> pd.DataFrame:
        """
        Create technical indicators and features for model training
        
        Args:
            df: DataFrame with OHLC data
            lookback_window: Number of previous days to use for features
        
        Returns:
            DataFrame with additional features
        """
        df_features = df.copy()
        
        # Basic price features
        df_features['Price_Range'] = df_features['High'] - df_features['Low']
        df_features['Price_Change'] = df_features['Close'] - df_features['Open']
        df_features['High_Low_Ratio'] = df_features['High'] / df_features['Low']
        df_features['Close_to_High'] = (df_features['High'] - df_features['Close']) / (df_features['High'] - df_features['Low'] + 1e-8)
        df_features['Close_to_Low'] = (df_features['Close'] - df_features['Low']) / (df_features['High'] - df_features['Low'] + 1e-8)
        
        # Moving averages
        for window in [5, 10, 20]:
            df_features[f'MA_{window}'] = df_features['Close'].rolling(window=window).mean()
            df_features[f'Volume_MA_{window}'] = df_features.get('Volume', pd.Series([0]*len(df_features))).rolling(window=window).mean()
        
        # Price-based technical indicators
        df_features['RSI'] = self._calculate_rsi(df_features['Close'])
        df_features['MACD'], df_features['MACD_Signal'] = self._calculate_macd(df_features['Close'])
        
        # Lag features (using previous days' values)
        for lag in range(1, min(lookback_window + 1, len(df_features))):
            for col in ['Open', 'High', 'Low', 'Close']:
                df_features[f'{col}_lag_{lag}'] = df_features[col].shift(lag)
        
        # Return features
        df_features['Next_Open'] = df_features['Open'].shift(-1)
        df_features['Next_High'] = df_features['High'].shift(-1)
        df_features['Next_Low'] = df_features['Low'].shift(-1)
        df_features['Next_Close'] = df_features['Close'].shift(-1)
        
        # Drop rows with NaN values (mostly at the beginning due to lookback features)
        df_features = df_features.dropna()
        
        return df_features
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal

    def prepare_data_for_training(self, df: pd.DataFrame, target_cols: List[str] = ['Next_Close']) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and targets for model training
        
        Args:
            df: DataFrame with features (output from create_features)
            target_cols: List of target columns to predict
        
        Returns:
            Tuple of (features, targets)
        """
        # Select feature columns (exclude target columns and Date)
        feature_cols = [col for col in df.columns 
                       if not col.startswith('Next_') and col != 'Date' and col in df.columns]
        
        X = df[feature_cols].values
        y = df[target_cols].values if len(target_cols) > 1 else df[target_cols[0]].values
        
        return X, y