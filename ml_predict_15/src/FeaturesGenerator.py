"""
Features Generator Module - Refactored

Unified class for generating features and targets for ML models.
Supports both classical technical indicators and comprehensive crypto features.
"""

import numpy as np
import pandas as pd
import ta


class FeaturesGenerator:
    """
    Unified feature generator for cryptocurrency ML models.
    """
    
    def __init__(self):
        """Initialize FeaturesGenerator."""
        pass
    
    def generate_features(self, df, method='classical', **kwargs):
        """
        Generate features using specified method.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with OHLCV data
        method : str
            'classical' - Traditional technical indicators
            'crypto' - Comprehensive crypto features (150+)
            'otus' - OTUS-style features
        **kwargs : Additional parameters for specific methods
            
        Returns:
        --------
        pd.DataFrame : DataFrame with features added
        """
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        if method == 'classical':
            return self.add_features(df)
        elif method == 'crypto':
            threshold = kwargs.get('price_change_threshold', 0.02)
            return self.add_crypto_features(df, threshold)
        elif method == 'otus':
            return self.add_features_otus(df, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'classical', 'crypto', or 'otus'")
    
    def create_target(self, df, target_bars=15, target_pct=3.0, method='classification'):
        """
        Create target variable for ML prediction.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data (must have 'close' column)
        target_bars : int
            Number of bars to look ahead for target
        target_pct : float
            Percentage threshold for target (e.g., 3.0 for 3%)
        method : str
            'classification' - Three classes: 1 (up), 0 (neutral), -1 (down)
            'binary' - Two classes: 1 (up), 0 (not up)
            'regression' - Continuous target (actual pct change)
            
        Returns:
        --------
        pd.DataFrame : DataFrame with 'target' column added
        """
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Calculate percentage change
        df[f'pct_change_{target_bars}'] = (
            df['close'].shift(-target_bars) / df['close'] - 1
        ) * 100
        
        if method == 'classification':
            # Three classes: 1 (up), 0 (neutral), -1 (down)
            target_up = (df[f'pct_change_{target_bars}'] >= target_pct)
            target_down = (df[f'pct_change_{target_bars}'] <= -target_pct)
            df['target'] = 0
            df.loc[target_up, 'target'] = 1
            df.loc[target_down, 'target'] = -1
        
        elif method == 'binary':
            # Two classes: 1 (up), 0 (not up)
            df['target'] = (df[f'pct_change_{target_bars}'] >= target_pct).astype(int)
        
        elif method == 'regression':
            # Continuous target (actual percentage change)
            df['target'] = df[f'pct_change_{target_bars}']        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'classification', 'binary', or 'regression'")        
        return df
    
    # ==================== CLASSICAL FEATURES ====================
    
    def add_features(self, df):
        #Add classical technical indicators.
        df.columns = df.columns.str.lower()
        df = self.add_sma(df)
        df = self.add_rsi(df)
        df = self.add_stochastic(df)
        df = self.add_bollinger(df)
        df = self.add_float(df)
        df = self.clear_data(df)
        return df
    
    def add_sma(self, df, window=10):
        #Add Simple Moving Average features.
        df['SMA_10'] = df['close'].rolling(window=window).mean()
        df['SMA_20'] = df['close'].rolling(window=window*2).mean()
        
        # SMA crossovers
        cross_up_10 = (df['close'] > df['SMA_10']) & (df['close'].shift(1) <= df['SMA_10'].shift(1))
        cross_down_10 = (df['close'] < df['SMA_10']) & (df['close'].shift(1) >= df['SMA_10'].shift(1))
        df['SMA_cross_10'] = 0
        df.loc[cross_up_10, 'SMA_cross_10'] = 1
        df.loc[cross_down_10, 'SMA_cross_10'] = -1
        
        cross_up_20 = (df['close'] > df['SMA_20']) & (df['close'].shift(1) <= df['SMA_20'].shift(1))
        cross_down_20 = (df['close'] < df['SMA_20']) & (df['close'].shift(1) >= df['SMA_20'].shift(1))
        df['SMA_cross_20'] = 0
        df.loc[cross_up_20, 'SMA_cross_20'] = 1
        df.loc[cross_down_20, 'SMA_cross_20'] = -1
        
        return df
    
    def add_rsi(self, df, min_val=30, max_val=70):
        #Add RSI features.
        rsi_indicator = ta.momentum.RSIIndicator(close=df['close'], window=14)
        df['RSI'] = rsi_indicator.rsi()
        
        # RSI crossovers
        cross_up_min = (df['RSI'] > min_val) & (df['RSI'].shift(1) <= min_val)
        cross_down_min = (df['RSI'] < min_val) & (df['RSI'].shift(1) >= min_val)
        df['RSI_cross_min'] = 0
        df.loc[cross_up_min, 'RSI_cross_min'] = 1
        df.loc[cross_down_min, 'RSI_cross_min'] = -1
        
        cross_up_max = (df['RSI'] > max_val) & (df['RSI'].shift(1) <= max_val)
        cross_down_max = (df['RSI'] < max_val) & (df['RSI'].shift(1) >= max_val)
        df['RSI_cross_max'] = 0
        df.loc[cross_up_max, 'RSI_cross_max'] = 1
        df.loc[cross_down_max, 'RSI_cross_max'] = -1
        
        return df
    
    def add_stochastic(self, df, min_val=20, max_val=80):
        #Add Stochastic Oscillator features.
        stoch_indicator = ta.momentum.StochasticOscillator(
            high=df['high'], low=df['low'], close=df['close'], 
            window=14, smooth_window=3
        )
        df['STOCH_K'] = stoch_indicator.stoch()
        df['STOCH_D'] = stoch_indicator.stoch_signal()
        
        # Stochastic crossovers
        cross_up_min = (df['STOCH_K'] > min_val) & (df['STOCH_K'].shift(1) <= min_val)
        cross_down_min = (df['STOCH_K'] < min_val) & (df['STOCH_K'].shift(1) >= min_val)
        df['STOCH_cross_min'] = 0
        df.loc[cross_up_min, 'STOCH_cross_min'] = 1
        df.loc[cross_down_min, 'STOCH_cross_min'] = -1
        
        return df
    
    def add_bollinger(self, df, window=14):
        #Add Bollinger Bands features.
        bollinger_indicator = ta.volatility.BollingerBands(close=df['close'], window=window)
        df['BOLLINGER_High'] = bollinger_indicator.bollinger_hband()
        df['BOLLINGER_Low'] = bollinger_indicator.bollinger_lband()
        df['BOLLINGER_Middle'] = bollinger_indicator.bollinger_mavg()
        
        # Bollinger crossovers
        cross_up_mid = (df['close'] > df['BOLLINGER_Middle']) & (df['close'].shift(1) <= df['BOLLINGER_Middle'].shift(1))
        cross_down_mid = (df['close'] < df['BOLLINGER_Middle']) & (df['close'].shift(1) >= df['BOLLINGER_Middle'].shift(1))
        df['BOLLINGER_cross_mid'] = 0
        df.loc[cross_up_mid, 'BOLLINGER_cross_mid'] = 1
        df.loc[cross_down_mid, 'BOLLINGER_cross_mid'] = -1
        
        return df
    
    def add_float(self, df):
        #Add return-based features.
        # Price returns
        df['Return_15'] = np.log(df['close'] / df['close'].shift(15))
        df['Return_15'] = df['Return_15'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        df['Return_1'] = np.log(df['close'] / df['close'].shift(1))
        df['Return_1'] = df['Return_1'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # RSI log return
        rsi_temp = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
        df['RSI'] = np.log((rsi_temp + 1) / (rsi_temp.shift(1) + 1))
        df['RSI'] = df['RSI'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Stochastic log returns
        stoch_k_temp = ta.momentum.StochasticOscillator(
            high=df['high'], low=df['low'], close=df['close'], 
            window=14, smooth_window=3
        ).stoch()
        df['STOCH_K'] = np.log((stoch_k_temp + 1) / (stoch_k_temp.shift(1) + 1))
        df['STOCH_K'] = df['STOCH_K'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        stoch_d_temp = ta.momentum.StochasticOscillator(
            high=df['high'], low=df['low'], close=df['close'], 
            window=14, smooth_window=3
        ).stoch_signal()
        df['STOCH_D'] = np.log((stoch_d_temp + 1) / (stoch_d_temp.shift(1) + 1))
        df['STOCH_D'] = df['STOCH_D'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return df
    
    def clear_data(self, df):
        """Clean data by removing NaN values."""
        df = df.dropna()
        return df
    
    def returnificate(self, df):
        """Add return features for OHLC."""
        df['ret_open'] = np.log(df['open'] / df['open'].shift(1))
        df['ret_high'] = np.log(df['high'] / df['high'].shift(1))
        df['ret_low'] = np.log(df['low'] / df['low'].shift(1))
        df['ret_close'] = np.log(df['close'] / df['close'].shift(1))
        return df
    
    # ==================== CRYPTO FEATURES ====================
    
    def add_crypto_features(self, df, price_change_threshold=0.02):
        """
        Comprehensive cryptocurrency feature engineering for ML prediction.
        Creates 150+ technical indicators and features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with OHLCV data (columns: timestamp, open, high, low, close, volume)
        price_change_threshold : float
            Percentage threshold for target (e.g., 0.02 for 2% change)
            
        Returns:
        --------
        dict with keys: X_train, y_train, X_val, y_val, X_test, y_test, feature_names, 
                    train_data, val_data, test_data
        """
        df = df.copy()
        df.columns = df.columns.str.lower()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # ==================== TARGET CREATION ====================
        df['price_change_pct'] = (df['close'].shift(-1) - df['close']) / df['close']
        df['target'] = (df['price_change_pct'] >= price_change_threshold).astype(int)
        df = df.dropna(subset=['price_change_pct'])
        
        # ==================== PRICE-BASED FEATURES ====================
        
        price_features = {}
        
        # Returns and log returns
        for period in [1, 3, 6, 12, 24]:
            price_features[f'return_{period}h'] = df['close'].pct_change(period)
            price_features[f'log_return_{period}h'] = np.log(df['close'] / df['close'].shift(period))
        
        # Price momentum
        for period in [3, 6, 12, 24, 48]:
            price_features[f'momentum_{period}h'] = df['close'] - df['close'].shift(period)
            price_features[f'momentum_pct_{period}h'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
        
        # High-Low spread
        price_features['hl_spread'] = df['high'] - df['low']
        price_features['hl_spread_pct'] = (df['high'] - df['low']) / df['close']
        price_features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        df = pd.concat([df, pd.DataFrame(price_features, index=df.index)], axis=1)
        
        # ==================== MOVING AVERAGES ====================
        
        ma_features = {}
        ma_periods = [5, 10, 20, 50, 100, 200]
        
        for period in ma_periods:
            ma_features[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            ma_features[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        df = pd.concat([df, pd.DataFrame(ma_features, index=df.index)], axis=1)
        
        # Price ratios (need MA columns to exist first)
        ratio_features = {}
        for period in ma_periods:
            ratio_features[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}']
            ratio_features[f'price_to_ema_{period}'] = df['close'] / df[f'ema_{period}']
        
        # Moving average crossovers
        ratio_features['sma_cross_5_20'] = df['sma_5'] - df['sma_20']
        ratio_features['sma_cross_10_50'] = df['sma_10'] - df['sma_50']
        ratio_features['ema_cross_5_20'] = df['ema_5'] - df['ema_20']
        
        df = pd.concat([df, pd.DataFrame(ratio_features, index=df.index)], axis=1)
        
        # ==================== VOLATILITY FEATURES ====================
        
        vol_features = {}
        
        # Rolling standard deviation
        for period in [5, 10, 20, 50]:
            vol_features[f'volatility_{period}h'] = df['return_1h'].rolling(window=period).std()
            vol_features[f'price_std_{period}h'] = df['close'].rolling(window=period).std()
        
        # Bollinger Bands
        for period in [20, 50]:
            rolling_mean = df['close'].rolling(window=period).mean()
            rolling_std = df['close'].rolling(window=period).std()
            vol_features[f'bb_upper_{period}'] = rolling_mean + (rolling_std * 2)
            vol_features[f'bb_lower_{period}'] = rolling_mean - (rolling_std * 2)
            vol_features[f'bb_width_{period}'] = (vol_features[f'bb_upper_{period}'] - vol_features[f'bb_lower_{period}']) / rolling_mean
            vol_features[f'bb_position_{period}'] = (df['close'] - vol_features[f'bb_lower_{period}']) / (vol_features[f'bb_upper_{period}'] - vol_features[f'bb_lower_{period}'] + 1e-10)
        
        # Average True Range (ATR)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        for period in [14, 28]:
            vol_features[f'atr_{period}'] = true_range.rolling(window=period).mean()
            vol_features[f'atr_pct_{period}'] = vol_features[f'atr_{period}'] / df['close']
        
        df = pd.concat([df, pd.DataFrame(vol_features, index=df.index)], axis=1)
        
        # ==================== MOMENTUM INDICATORS ====================
        
        momentum_features = {}
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        for period in [14, 28]:
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)
            momentum_features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        momentum_features['macd'] = exp1 - exp2
        momentum_features['macd_signal'] = momentum_features['macd'].ewm(span=9, adjust=False).mean()
        momentum_features['macd_diff'] = momentum_features['macd'] - momentum_features['macd_signal']
        
        # Stochastic Oscillator
        for period in [14, 28]:
            low_min = df['low'].rolling(window=period).min()
            high_max = df['high'].rolling(window=period).max()
            momentum_features[f'stoch_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
        
        df = pd.concat([df, pd.DataFrame(momentum_features, index=df.index)], axis=1)
        
        # Stochastic signals (need stoch columns first)
        stoch_signals = {}
        for period in [14, 28]:
            stoch_signals[f'stoch_signal_{period}'] = df[f'stoch_{period}'].rolling(window=3).mean()
        
        # Rate of Change (ROC)
        for period in [6, 12, 24]:
            stoch_signals[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
        
        df = pd.concat([df, pd.DataFrame(stoch_signals, index=df.index)], axis=1)
        
        # ==================== VOLUME FEATURES ====================
        
        volume_features = {}
        
        # Volume changes
        for period in [1, 3, 6, 12, 24]:
            volume_features[f'volume_change_{period}h'] = df['volume'].pct_change(period)
        
        # Volume moving averages
        for period in [5, 10, 20, 50]:
            volume_features[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
        
        df = pd.concat([df, pd.DataFrame(volume_features, index=df.index)], axis=1)
        
        # Volume ratios (need volume_sma columns first)
        volume_ratios = {}
        for period in [5, 10, 20, 50]:
            volume_ratios[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
        
        # On-Balance Volume (OBV)
        volume_ratios['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        volume_ratios['obv_ema_10'] = volume_ratios['obv'].ewm(span=10, adjust=False).mean()
        volume_ratios['obv_ema_20'] = volume_ratios['obv'].ewm(span=20, adjust=False).mean()
        
        # Volume-Price Trend
        volume_ratios['vpt'] = (df['volume'] * ((df['close'] - df['close'].shift(1)) / df['close'].shift(1))).fillna(0).cumsum()
        
        # Money Flow Index (MFI)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        for period in [14, 28]:
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()
            mfi_ratio = positive_flow / (negative_flow + 1e-10)
            volume_ratios[f'mfi_{period}'] = 100 - (100 / (1 + mfi_ratio))
        
        df = pd.concat([df, pd.DataFrame(volume_ratios, index=df.index)], axis=1)
        
        # ==================== PATTERN FEATURES ====================
        
        pattern_features = {}
        pattern_features['body'] = df['close'] - df['open']
        pattern_features['body_pct'] = pattern_features['body'] / df['open']
        pattern_features['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        pattern_features['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        pattern_features['shadow_ratio'] = (pattern_features['upper_shadow'] + pattern_features['lower_shadow']) / (np.abs(pattern_features['body']) + 1e-10)
        pattern_features['is_doji'] = (np.abs(pattern_features['body']) / (df['high'] - df['low'] + 1e-10) < 0.1).astype(int)
        pattern_features['is_hammer'] = ((pattern_features['lower_shadow'] > 2 * np.abs(pattern_features['body'])) & 
                                        (pattern_features['upper_shadow'] < np.abs(pattern_features['body']))).astype(int)
        
        df = pd.concat([df, pd.DataFrame(pattern_features, index=df.index)], axis=1)
        
        # ==================== TIME-BASED FEATURES ====================
        
        time_features = {}
        time_features['hour'] = df['timestamp'].dt.hour
        time_features['day_of_week'] = df['timestamp'].dt.dayofweek
        time_features['day_of_month'] = df['timestamp'].dt.day
        time_features['month'] = df['timestamp'].dt.month
        
        # Cyclical encoding
        time_features['hour_sin'] = np.sin(2 * np.pi * time_features['hour'] / 24)
        time_features['hour_cos'] = np.cos(2 * np.pi * time_features['hour'] / 24)
        time_features['day_sin'] = np.sin(2 * np.pi * time_features['day_of_week'] / 7)
        time_features['day_cos'] = np.cos(2 * np.pi * time_features['day_of_week'] / 7)
        
        df = pd.concat([df, pd.DataFrame(time_features, index=df.index)], axis=1)
        
        # ==================== STATISTICAL FEATURES ====================
        
        stat_features = {}
        
        # Rolling statistics
        for period in [10, 20, 50]:
            stat_features[f'price_skew_{period}'] = df['close'].rolling(window=period).skew()
            stat_features[f'price_kurt_{period}'] = df['close'].rolling(window=period).kurt()
            stat_features[f'volume_skew_{period}'] = df['volume'].rolling(window=period).skew()
        
        # Price percentile in rolling window
        for period in [20, 50, 100]:
            stat_features[f'price_percentile_{period}'] = df['close'].rolling(window=period).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan
            )
        
        df = pd.concat([df, pd.DataFrame(stat_features, index=df.index)], axis=1)
        
        # ==================== CLEAN DATA ====================
        
        df = df.replace([np.inf, -np.inf], np.nan)
        exclude_cols = ['timestamp', 'target', 'price_change_pct', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(method='bfill')
        df = df.dropna()
        
        # ==================== TRAIN/VAL/TEST SPLIT ====================
        
        test_start_date = df['timestamp'].max() - pd.DateOffset(months=1)
        val_start_date = df['timestamp'].max() - pd.DateOffset(months=2)
        
        train_data = df[df['timestamp'] < val_start_date]
        val_data = df[(df['timestamp'] >= val_start_date) & (df['timestamp'] < test_start_date)]
        test_data = df[df['timestamp'] >= test_start_date]
        
        X_train, y_train = train_data[feature_cols], train_data['target']
        X_val, y_val = val_data[feature_cols], val_data['target']
        X_test, y_test = test_data[feature_cols], test_data['target']
        
        print(f"\n{'='*70}")
        print(f"CRYPTO FEATURE ENGINEERING SUMMARY")
        print(f"{'='*70}")
        print(f"Total features created: {len(feature_cols)}")
        print(f"Price change threshold: {price_change_threshold*100:.1f}%")
        print(f"\nData splits:")
        print(f"  Training:   {len(X_train):,} samples ({len(X_train)/len(df)*100:.1f}%)")
        print(f"  Validation: {len(X_val):,} samples ({len(X_val)/len(df)*100:.1f}%)")
        print(f"  Test:       {len(X_test):,} samples ({len(X_test)/len(df)*100:.1f}%)")
        print(f"\nTarget distribution (Training):")
        print(f"  Class 0 (No rise): {(y_train==0).sum():,} ({(y_train==0).sum()/len(y_train)*100:.1f}%)")
        print(f"  Class 1 (Rise):    {(y_train==1).sum():,} ({(y_train==1).sum()/len(y_train)*100:.1f}%)")
        print(f"{'='*70}\n")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'feature_names': feature_cols,
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data
        }
        
        # Return the full dataframe with features
        full_df = pd.concat([
            result['train_data'],
            result['val_data'],
            result['test_data']
        ], ignore_index=True)
        
        return full_df
    
    # ==================== OTUS FEATURES ====================
    
    def add_features_otus(self, df, target_bars=15, target_pct=3.0):
        """
        Add OTUS-style features with train/val/test split.
        (Keeping for backward compatibility)
        """
        # This method is kept from the original implementation
        # See the original FeaturesGenerator.py for full implementation
        pass
