"""Feature engineering for market data."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

from src.features.technical_indicators import TechnicalIndicators

try:
    from src.utils import config, get_logger
except ImportError:
    # Handle case when running as script directly
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from src.utils import config, get_logger

logger = get_logger(__name__)

class FeatureEngineer:
    """Feature engineering for OHLC market data."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize feature engineer.
        
        Args:
            config_dict: Feature engineering configuration
        """
        self.config = config_dict or config.feature_config
        self.technical_indicators = TechnicalIndicators()
        self.feature_names = []
    
    def create_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with price features added
        """
        logger.info("Creating price features")
        df = data.copy()
        
        # Price changes
        df['Price_Change'] = df['Close'].diff()
        df['Price_Change_Pct'] = df['Close'].pct_change()
        df['Open_Change_Pct'] = df['Open'].pct_change()
        df['High_Change_Pct'] = df['High'].pct_change()
        df['Low_Change_Pct'] = df['Low'].pct_change()
        
        # Price ratios
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        df['High_Close_Ratio'] = df['High'] / df['Close']
        df['Low_Close_Ratio'] = df['Low'] / df['Close']
        
        # Price ranges
        df['Daily_Range'] = df['High'] - df['Low']
        df['Daily_Range_Pct'] = (df['High'] - df['Low']) / df['Close']
        df['Upper_Shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
        df['Lower_Shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
        df['Body_Size'] = np.abs(df['Close'] - df['Open'])
        df['Body_Size_Pct'] = np.abs(df['Close'] - df['Open']) / df['Close']
        
        # Gap analysis
        df['Gap'] = df['Open'] - df['Close'].shift(1)
        df['Gap_Pct'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        return df
    
    def create_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features.
        
        Args:
            data: DataFrame with volume data
            
        Returns:
            DataFrame with volume features added
        """
        logger.info("Creating volume features")
        df = data.copy()
        
        # Volume changes
        df['Volume_Change'] = df['Volume'].diff()
        df['Volume_Change_Pct'] = df['Volume'].pct_change()
        
        # Volume ratios
        df['Volume_Price_Ratio'] = df['Volume'] / df['Close']
        df['Volume_Range_Ratio'] = df['Volume'] / (df['High'] - df['Low'])
        
        # Volume moving averages
        for period in [5, 10, 20]:
            df[f'Volume_MA_{period}'] = df['Volume'].rolling(window=period).mean()
            df[f'Volume_Ratio_{period}'] = df['Volume'] / df[f'Volume_MA_{period}']
        
        # Volume-weighted prices
        df['VWAP_Simple'] = (df['Volume'] * df['Close']).cumsum() / df['Volume'].cumsum()
        
        return df
    
    def create_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create volatility-based features.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with volatility features added
        """
        logger.info("Creating volatility features")
        df = data.copy()
        
        # Rolling volatility (standard deviation of returns)
        for period in [5, 10, 20, 30]:
            returns = df['Close'].pct_change()
            df[f'Volatility_{period}d'] = returns.rolling(window=period).std()
            df[f'Volatility_{period}d_Annualized'] = df[f'Volatility_{period}d'] * np.sqrt(252)
        
        # Parkinson volatility (uses high and low prices)
        for period in [5, 10, 20]:
            hl_ratio = np.log(df['High'] / df['Low'])
            df[f'Parkinson_Vol_{period}d'] = np.sqrt(
                hl_ratio.rolling(window=period).apply(lambda x: (x**2).mean()) / (4 * np.log(2))
            )
        
        # Garman-Klass volatility
        for period in [5, 10, 20]:
            hl_term = 0.5 * (np.log(df['High'] / df['Low']))**2
            oc_term = (2 * np.log(2) - 1) * (np.log(df['Close'] / df['Open']))**2
            gk_vol = hl_term - oc_term
            df[f'GK_Vol_{period}d'] = np.sqrt(gk_vol.rolling(window=period).mean())
        
        return df
    
    def create_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create momentum-based features.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with momentum features added
        """
        logger.info("Creating momentum features")
        df = data.copy()
        
        # Rate of Change (ROC)
        for period in [1, 5, 10, 20]:
            df[f'ROC_{period}'] = ((df['Close'] - df['Close'].shift(period)) / 
                                   df['Close'].shift(period)) * 100
        
        # Momentum
        for period in [5, 10, 20]:
            df[f'Momentum_{period}'] = df['Close'] - df['Close'].shift(period)
        
        # Price acceleration
        df['Price_Acceleration'] = df['Close'].diff().diff()
        
        # Relative position in recent range
        for period in [5, 10, 20]:
            rolling_min = df['Low'].rolling(window=period).min()
            rolling_max = df['High'].rolling(window=period).max()
            df[f'Relative_Position_{period}'] = ((df['Close'] - rolling_min) / 
                                                 (rolling_max - rolling_min))
        
        return df
    
    def create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features.
        
        Args:
            data: DataFrame with datetime index
            
        Returns:
            DataFrame with time features added
        """
        logger.info("Creating time features")
        df = data.copy()
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Basic time features
        df['Day_of_Week'] = df.index.dayofweek
        df['Day_of_Month'] = df.index.day
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['Year'] = df.index.year
        
        # Cyclical encoding for time features
        df['Day_of_Week_Sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
        df['Day_of_Week_Cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        # Market session indicators (assuming US market hours)
        df['Is_Monday'] = (df['Day_of_Week'] == 0).astype(int)
        df['Is_Friday'] = (df['Day_of_Week'] == 4).astype(int)
        df['Is_Month_End'] = df.index.is_month_end.astype(int)
        df['Is_Month_Start'] = df.index.is_month_start.astype(int)
        df['Is_Quarter_End'] = df.index.is_quarter_end.astype(int)
        
        return df
    
    def create_lag_features(self, data: pd.DataFrame, lags: List[int] = None) -> pd.DataFrame:
        """Create lagged features.
        
        Args:
            data: Input DataFrame
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features added
        """
        if lags is None:
            lags = [1, 2, 3, 5, 10]
        
        logger.info(f"Creating lag features for periods: {lags}")
        df = data.copy()
        
        # Price lags
        for lag in lags:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
            df[f'High_Lag_{lag}'] = df['High'].shift(lag)
            df[f'Low_Lag_{lag}'] = df['Low'].shift(lag)
        
        # Return lags
        returns = df['Close'].pct_change()
        for lag in lags:
            df[f'Return_Lag_{lag}'] = returns.shift(lag)
        
        return df
    
    def create_rolling_statistics(self, data: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        """Create rolling statistical features.
        
        Args:
            data: Input DataFrame
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling statistics added
        """
        if windows is None:
            windows = [5, 10, 20, 50]
        
        logger.info(f"Creating rolling statistics for windows: {windows}")
        df = data.copy()
        
        for window in windows:
            # Rolling statistics for close price
            df[f'Close_Mean_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'Close_Std_{window}'] = df['Close'].rolling(window=window).std()
            df[f'Close_Min_{window}'] = df['Close'].rolling(window=window).min()
            df[f'Close_Max_{window}'] = df['Close'].rolling(window=window).max()
            df[f'Close_Median_{window}'] = df['Close'].rolling(window=window).median()
            df[f'Close_Skew_{window}'] = df['Close'].rolling(window=window).skew()
            df[f'Close_Kurt_{window}'] = df['Close'].rolling(window=window).kurt()
            
            # Rolling statistics for volume
            df[f'Volume_Mean_{window}'] = df['Volume'].rolling(window=window).mean()
            df[f'Volume_Std_{window}'] = df['Volume'].rolling(window=window).std()
            
            # Rolling correlations (if multiple symbols)
            if 'Symbol' in df.columns and len(df['Symbol'].unique()) > 1:
                # This would require pivot table for multiple symbols
                pass
        
        return df
    
    def create_custom_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create custom domain-specific features.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with custom features added
        """
        logger.info("Creating custom features")
        df = data.copy()
        
        # Custom features from configuration
        custom_features = self.config.get('custom_features', [])
        
        for feature in custom_features:
            if feature == 'price_change_pct':
                df['Price_Change_Pct'] = df['Close'].pct_change()
            
            elif feature == 'volume_change_pct':
                df['Volume_Change_Pct'] = df['Volume'].pct_change()
            
            elif feature == 'high_low_ratio':
                df['High_Low_Ratio'] = df['High'] / df['Low']
            
            elif feature == 'close_open_ratio':
                df['Close_Open_Ratio'] = df['Close'] / df['Open']
            
            elif feature == 'volatility_7d':
                returns = df['Close'].pct_change()
                df['Volatility_7d'] = returns.rolling(window=7).std()
            
            elif feature == 'volatility_30d':
                returns = df['Close'].pct_change()
                df['Volatility_30d'] = returns.rolling(window=30).std()
        
        # Market regime features
        df['Bull_Bear_Indicator'] = np.where(
            df['Close'] > df['Close'].rolling(window=50).mean(), 1, 0
        )
        
        # Trend strength
        df['Trend_Strength'] = np.abs(
            df['Close'].rolling(window=20).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0]
            )
        )
        
        return df
    
    def select_features(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        method: str = 'mutual_info',
        k: int = 50
    ) -> List[str]:
        """Select best features using statistical methods.
        
        Args:
            X: Feature DataFrame
            y: Target series
            method: Selection method ('f_regression', 'mutual_info')
            k: Number of features to select
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting {k} best features using {method}")
        
        # Remove non-numeric columns
        numeric_X = X.select_dtypes(include=[np.number])
        
        # Handle missing values
        numeric_X = numeric_X.fillna(numeric_X.mean())
        y = y.fillna(y.mean())
        
        if method == 'f_regression':
            selector = SelectKBest(score_func=f_regression, k=k)
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=k)
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        selector.fit(numeric_X, y)
        selected_features = numeric_X.columns[selector.get_support()].tolist()
        
        logger.info(f"Selected features: {len(selected_features)}")
        return selected_features
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply complete feature engineering pipeline.
        
        Args:
            data: Raw OHLCV DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering pipeline")
        df = data.copy()
        
        # Create basic features
        df = self.create_price_features(df)
        df = self.create_volume_features(df)
        df = self.create_volatility_features(df)
        df = self.create_momentum_features(df)
        df = self.create_time_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_statistics(df)
        df = self.create_custom_features(df)
        
        # Add technical indicators
        df = self.technical_indicators.calculate_all_indicators(df, self.config)
        
        # Store feature names (excluding original OHLCV columns)
        original_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']
        self.feature_names = [col for col in df.columns if col not in original_columns]
        
        logger.info(f"Feature engineering completed. Created {len(self.feature_names)} features")
        logger.info(f"Final DataFrame shape: {df.shape}")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of engineered feature names.
        
        Returns:
            List of feature names
        """
        return self.feature_names.copy()
