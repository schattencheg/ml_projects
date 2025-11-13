"""
FeaturesGenerator - Generates technical indicators and features from raw OHLC data.
"""

import pandas as pd
import numpy as np
from typing import Optional, List


class FeaturesGenerator:
    """
    Generates technical indicators and features for ML models.
    
    Supports:
    - Moving averages (SMA, EMA)
    - Momentum indicators (RSI, MACD)
    - Volatility indicators (Bollinger Bands, ATR)
    - Volume indicators
    - Price patterns
    """
    
    def __init__(self):
        """Initialize FeaturesGenerator."""
        self.feature_names = []
        
    def generate_features(self, df: pd.DataFrame, 
                         feature_set: str = 'basic') -> pd.DataFrame:
        """
        Generate features from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            feature_set: Feature set to generate ('basic', 'advanced', 'all')
            
        Returns:
            DataFrame with original data + generated features
        """
        print(f"Generating {feature_set} features...")
        
        df_features = df.copy()
        
        if feature_set in ['basic', 'all']:
            df_features = self._add_basic_features(df_features)
        
        if feature_set in ['advanced', 'all']:
            df_features = self._add_advanced_features(df_features)
        
        # Remove NaN rows created by indicators
        initial_rows = len(df_features)
        df_features = df_features.dropna()
        removed_rows = initial_rows - len(df_features)
        
        print(f"✓ Generated {len(self.feature_names)} features")
        if removed_rows > 0:
            print(f"  Removed {removed_rows} rows with NaN values")
        
        return df_features
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators."""
        
        # Moving Averages
        for period in [5, 10, 20, 50]:
            col_name = f'sma_{period}'
            df[col_name] = df['close'].rolling(window=period).mean()
            self.feature_names.append(col_name)
        
        # Exponential Moving Averages
        for period in [12, 26]:
            col_name = f'ema_{period}'
            df[col_name] = df['close'].ewm(span=period, adjust=False).mean()
            self.feature_names.append(col_name)
        
        # Price momentum
        for period in [1, 5, 10]:
            col_name = f'returns_{period}'
            df[col_name] = df['close'].pct_change(period)
            self.feature_names.append(col_name)
        
        # Volatility
        for period in [5, 10, 20]:
            col_name = f'volatility_{period}'
            df[col_name] = df['close'].pct_change().rolling(window=period).std()
            self.feature_names.append(col_name)
        
        return df
    
    def _add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced technical indicators."""
        
        # RSI (Relative Strength Index)
        df['rsi_14'] = self._calculate_rsi(df['close'], period=14)
        self.feature_names.append('rsi_14')
        
        # MACD
        macd, signal, hist = self._calculate_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        self.feature_names.extend(['macd', 'macd_signal', 'macd_hist'])
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['close'])
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        self.feature_names.extend(['bb_upper', 'bb_middle', 'bb_lower', 'bb_width'])
        
        # ATR (Average True Range)
        df['atr_14'] = self._calculate_atr(df, period=14)
        self.feature_names.append('atr_14')
        
        # Volume features (if volume exists)
        if 'volume' in df.columns:
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            self.feature_names.extend(['volume_sma_20', 'volume_ratio'])
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, 
                       fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def _calculate_bollinger_bands(self, prices: pd.Series, 
                                   period: int = 20, std_dev: int = 2):
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def create_target(self, df: pd.DataFrame, 
                     target_type: str = 'classification',
                     future_bars: int = 5,
                     threshold: float = 0.02) -> pd.DataFrame:
        """
        Create target variable for ML models.
        
        Args:
            df: DataFrame with features
            target_type: 'classification' or 'regression'
            future_bars: Number of bars to look ahead
            threshold: Threshold for classification (e.g., 2% price change)
            
        Returns:
            DataFrame with target column added
        """
        print(f"Creating {target_type} target (future_bars={future_bars})...")
        
        df = df.copy()
        
        if target_type == 'regression':
            # Predict future price change
            df['target'] = df['close'].pct_change(future_bars).shift(-future_bars)
            
        elif target_type == 'classification':
            # Predict if price will increase by threshold
            future_return = df['close'].pct_change(future_bars).shift(-future_bars)
            df['target'] = (future_return > threshold).astype(int)
            
            # Count classes
            class_counts = df['target'].value_counts()
            print(f"  Class distribution:")
            print(f"    No Increase (0): {class_counts.get(0, 0)}")
            print(f"    Increase (1):    {class_counts.get(1, 0)}")
        
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
        
        # Remove rows with NaN target
        initial_rows = len(df)
        df = df.dropna(subset=['target'])
        removed_rows = initial_rows - len(df)
        
        print(f"✓ Target created, removed {removed_rows} rows with NaN target")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of generated feature names."""
        return self.feature_names.copy()
    
    def select_features(self, df: pd.DataFrame, 
                       exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Select only feature columns (exclude OHLCV and target).
        
        Args:
            df: DataFrame with all columns
            exclude_cols: Additional columns to exclude
            
        Returns:
            DataFrame with only feature columns
        """
        default_exclude = ['open', 'high', 'low', 'close', 'volume', 'target']
        
        if exclude_cols:
            default_exclude.extend(exclude_cols)
        
        feature_cols = [col for col in df.columns if col not in default_exclude]
        
        return df[feature_cols]
