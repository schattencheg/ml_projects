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
        
        # ==================== PRICE-BASED FEATURES ====================
        
        # Returns and log returns
        for period in [1, 3, 6, 12, 24]:
            col_name = f'return_{period}'
            df[col_name] = df['close'].pct_change(period)
            self.feature_names.append(col_name)
            
            col_name = f'log_return_{period}'
            df[col_name] = np.log(df['close'] / df['close'].shift(period))
            self.feature_names.append(col_name)
        
        # Price momentum
        for period in [3, 6, 12, 24, 48]:
            col_name = f'momentum_{period}'
            df[col_name] = df['close'] - df['close'].shift(period)
            self.feature_names.append(col_name)
            
            col_name = f'momentum_pct_{period}'
            df[col_name] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
            self.feature_names.append(col_name)
        
        # High-Low spread
        df['hl_spread'] = df['high'] - df['low']
        df['hl_spread_pct'] = (df['high'] - df['low']) / df['close']
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        self.feature_names.extend(['hl_spread', 'hl_spread_pct', 'close_position'])
        
        # ==================== MOVING AVERAGES ====================
        
        ma_periods = [5, 10, 20, 50, 100, 200]
        
        for period in ma_periods:
            # Simple Moving Average
            col_name = f'sma_{period}'
            df[col_name] = df['close'].rolling(window=period).mean()
            self.feature_names.append(col_name)
            
            # Exponential Moving Average
            col_name = f'ema_{period}'
            df[col_name] = df['close'].ewm(span=period, adjust=False).mean()
            self.feature_names.append(col_name)
        
        # Price ratios to moving averages
        for period in ma_periods:
            col_name = f'price_to_sma_{period}'
            df[col_name] = df['close'] / df[f'sma_{period}']
            self.feature_names.append(col_name)
            
            col_name = f'price_to_ema_{period}'
            df[col_name] = df['close'] / df[f'ema_{period}']
            self.feature_names.append(col_name)
        
        # Moving average crossovers
        df['sma_cross_5_20'] = df['sma_5'] - df['sma_20']
        df['sma_cross_10_50'] = df['sma_10'] - df['sma_50']
        df['ema_cross_5_20'] = df['ema_5'] - df['ema_20']
        self.feature_names.extend(['sma_cross_5_20', 'sma_cross_10_50', 'ema_cross_5_20'])
        
        # ==================== VOLATILITY FEATURES ====================
        
        # Rolling standard deviation
        for period in [5, 10, 20, 50]:
            col_name = f'volatility_{period}'
            df[col_name] = df['return_1'].rolling(window=period).std()
            self.feature_names.append(col_name)
            
            col_name = f'price_std_{period}'
            df[col_name] = df['close'].rolling(window=period).std()
            self.feature_names.append(col_name)
        
        # Bollinger Bands
        for period in [20, 50]:
            rolling_mean = df['close'].rolling(window=period).mean()
            rolling_std = df['close'].rolling(window=period).std()
            
            df[f'bb_upper_{period}'] = rolling_mean + (rolling_std * 2)
            df[f'bb_lower_{period}'] = rolling_mean - (rolling_std * 2)
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / rolling_mean
            df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-10)
            
            self.feature_names.extend([
                f'bb_upper_{period}', f'bb_lower_{period}', 
                f'bb_width_{period}', f'bb_position_{period}'
            ])
        
        return df
    
    def _add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced technical indicators."""
        
        # ==================== MOMENTUM INDICATORS ====================
        
        # RSI (Relative Strength Index)
        for period in [7, 14, 21]:
            col_name = f'rsi_{period}'
            df[col_name] = self._calculate_rsi(df['close'], period=period)
            self.feature_names.append(col_name)
        
        # MACD
        macd, signal, hist = self._calculate_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        self.feature_names.extend(['macd', 'macd_signal', 'macd_hist'])
        
        # Stochastic Oscillator
        for period in [14, 21]:
            low_min = df['low'].rolling(window=period).min()
            high_max = df['high'].rolling(window=period).max()
            df[f'stoch_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
            df[f'stoch_{period}_sma'] = df[f'stoch_{period}'].rolling(window=3).mean()
            self.feature_names.extend([f'stoch_{period}', f'stoch_{period}_sma'])
        
        # ATR (Average True Range)
        df['atr_14'] = self._calculate_atr(df, period=14)
        self.feature_names.append('atr_14')
        
        # ==================== VOLUME FEATURES ====================
        
        if 'volume' in df.columns:
            # Volume moving averages
            for period in [5, 10, 20]:
                col_name = f'volume_sma_{period}'
                df[col_name] = df['volume'].rolling(window=period).mean()
                self.feature_names.append(col_name)
                
                col_name = f'volume_ratio_{period}'
                df[col_name] = df['volume'] / df[f'volume_sma_{period}']
                self.feature_names.append(col_name)
            
            # Volume-weighted features
            df['vwap_20'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
            self.feature_names.append('vwap_20')
            
            # On-Balance Volume (OBV)
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            df['obv_ema_20'] = df['obv'].ewm(span=20, adjust=False).mean()
            self.feature_names.extend(['obv', 'obv_ema_20'])
            
            # Volume-Price Trend (VPT)
            df['vpt'] = (df['volume'] * ((df['close'] - df['close'].shift(1)) / df['close'].shift(1))).fillna(0).cumsum()
            self.feature_names.append('vpt')
            
            # Money Flow Index (MFI)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * df['volume']
            
            for period in [14, 28]:
                positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
                negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()
                mfi_ratio = positive_flow / (negative_flow + 1e-10)
                df[f'mfi_{period}'] = 100 - (100 / (1 + mfi_ratio))
                self.feature_names.append(f'mfi_{period}')
        
        # ==================== PATTERN FEATURES ====================
        
        # Candlestick patterns
        df['body'] = df['close'] - df['open']
        df['body_pct'] = df['body'] / df['open']
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['shadow_ratio'] = (df['upper_shadow'] + df['lower_shadow']) / (np.abs(df['body']) + 1e-10)
        
        self.feature_names.extend(['body', 'body_pct', 'upper_shadow', 'lower_shadow', 'shadow_ratio'])
        
        # Pattern indicators
        df['is_doji'] = (np.abs(df['body']) / (df['high'] - df['low'] + 1e-10) < 0.1).astype(int)
        df['is_hammer'] = ((df['lower_shadow'] > 2 * np.abs(df['body'])) & 
                          (df['upper_shadow'] < np.abs(df['body']))).astype(int)
        
        self.feature_names.extend(['is_doji', 'is_hammer'])
        
        # ==================== TIME-BASED FEATURES ====================
        
        # Check if we have timestamp or datetime index
        if 'timestamp' in df.columns:
            time_col = pd.to_datetime(df['timestamp'])
        elif isinstance(df.index, pd.DatetimeIndex):
            time_col = df.index
        else:
            time_col = None
        
        if time_col is not None:
            df['hour'] = time_col.hour
            df['day_of_week'] = time_col.dayofweek
            df['day_of_month'] = time_col.day
            df['month'] = time_col.month
            
            # Cyclical encoding for time features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            self.feature_names.extend([
                'hour', 'day_of_week', 'day_of_month', 'month',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
            ])
        
        # ==================== STATISTICAL FEATURES ====================
        
        # Rolling statistics
        for period in [10, 20, 50]:
            col_name = f'price_skew_{period}'
            df[col_name] = df['close'].rolling(window=period).skew()
            self.feature_names.append(col_name)
            
            col_name = f'price_kurt_{period}'
            df[col_name] = df['close'].rolling(window=period).kurt()
            self.feature_names.append(col_name)
            
            if 'volume' in df.columns:
                col_name = f'volume_skew_{period}'
                df[col_name] = df['volume'].rolling(window=period).skew()
                self.feature_names.append(col_name)
        
        # Price percentile in rolling window
        for period in [20, 50, 100]:
            col_name = f'price_percentile_{period}'
            df[col_name] = df['close'].rolling(window=period).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan
            )
            self.feature_names.append(col_name)
        
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
                     future_bars: int = 15,
                     threshold: float = 0.02,
                     num_classes: int = 3) -> pd.DataFrame:
        """
        Create target variable for ML models.
        
        Args:
            df: DataFrame with features
            target_type: 'classification' or 'regression'
            future_bars: Number of bars to look ahead
            threshold: Threshold for classification (e.g., 2% price change)
            num_classes: Number of classes (2 or 3)
                - 2 classes: 0 (no increase), 1 (increase)
                - 3 classes: -1 (decrease), 0 (neutral), +1 (increase)
            
        Returns:
            DataFrame with target column added
        """
        print(f"Creating {target_type} target (future_bars={future_bars}, num_classes={num_classes})...")
        
        df = df.copy()
        
        if target_type == 'regression':
            # Predict future price change
            df['target'] = df['close'].pct_change(future_bars).shift(-future_bars)
            
        elif target_type == 'classification':
            # Calculate future return
            future_return = df['close'].pct_change(future_bars).shift(-future_bars)
            
            if num_classes == 2:
                # Binary classification: 0 (no increase), 1 (increase)
                df['target'] = (future_return > threshold).astype(int)
                
                # Count classes
                class_counts = df['target'].value_counts().sort_index()
                print(f"  Class distribution (2 classes):")
                print(f"    No Increase (0): {class_counts.get(0, 0)}")
                print(f"    Increase (1):    {class_counts.get(1, 0)}")
                
            elif num_classes == 3:
                # Three-class classification: -1 (decrease), 0 (neutral), +1 (increase)
                df['target'] = 0  # Default to neutral
                df.loc[future_return > threshold, 'target'] = 1   # Increase
                df.loc[future_return < -threshold, 'target'] = -1  # Decrease
                
                # Count classes
                class_counts = df['target'].value_counts().sort_index()
                print(f"  Class distribution (3 classes):")
                print(f"    Decrease (-1): {class_counts.get(-1, 0)}")
                print(f"    Neutral (0):   {class_counts.get(0, 0)}")
                print(f"    Increase (+1): {class_counts.get(1, 0)}")
            
            else:
                raise ValueError(f"num_classes must be 2 or 3, got {num_classes}")
        
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
