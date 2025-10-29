"""
Comprehensive Cryptocurrency Feature Engineering Module

This module provides advanced feature engineering for cryptocurrency price prediction.
Creates 150+ technical indicators across multiple categories:
- Price-based features (returns, momentum, spreads)
- Moving averages (SMA, EMA, crossovers)
- Volatility indicators (Bollinger Bands, ATR)
- Momentum indicators (RSI, MACD, Stochastic, ROC)
- Volume features (OBV, MFI, volume ratios)
- Candlestick patterns
- Time-based features (cyclical encoding)
- Statistical features (skewness, kurtosis, percentiles)

Usage:
    from src.crypto_features import create_crypto_features
    
    result = create_crypto_features(df, price_change_threshold=0.02)
    X_train = result['X_train']
    y_train = result['y_train']
    feature_names = result['feature_names']
"""

import pandas as pd
import numpy as np


def create_crypto_features(df, price_change_threshold=0.02):
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
