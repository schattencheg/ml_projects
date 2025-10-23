"""
Data Preparation Module

Functions for preparing OHLCV data for ML model training.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.FeaturesGenerator import FeaturesGenerator


def prepare_data(df_raw: pd.DataFrame, target_bars: int = 45, target_pct: float = 3.0):
    """
    Prepare raw OHLCV data by adding features, target, and returns.
    
    Parameters:
    -----------
    df_raw : pd.DataFrame
        Raw dataframe with columns: Timestamp, Open, High, Low, Close, Volume (optional)
    target_bars : int
        Number of bars to look ahead for target
    target_pct : float
        Percentage increase threshold for target (e.g., 3.0 for 3%)
    
    Returns:
    --------
    X : pd.DataFrame
        Features (without target and pct_change_N)
    y : pd.Series
        Target labels
    """
    # Generate features
    fg = FeaturesGenerator()
    df = fg.add_features(df_raw)
    
    # Create target: 1 if price increases by target_pct% within target_bars, else 0
    df[f'pct_change_{target_bars}'] = (
        df['Close'].shift(-target_bars) / df['Close'] - 1
    ) * 100
    
    df['target'] = (df[f'pct_change_{target_bars}'] >= target_pct).astype(int)
    
    # Drop rows with NaN (from indicators and forward-looking target)
    df = df.dropna()
    
    # Separate features and target
    X = df.drop(columns=['target', f'pct_change_{target_bars}'])
    y = df['target']
    
    # Drop non-feature columns (keep only numeric features)
    non_feature_cols = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    X = X.drop(columns=[col for col in non_feature_cols if col in X.columns])
    
    return X, y


def fit_scaler_standard(X_train: pd.DataFrame):
    """
    Fit a StandardScaler on training data.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    
    Returns:
    --------
    scaler : StandardScaler
        Fitted scaler
    X_train_scaled : np.ndarray
        Scaled training features
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    return scaler, X_train_scaled

def fit_scaler_minmax(X_train: pd.DataFrame):
    """
    Fit a MinMaxScaler on training data.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    
    Returns:
    --------
    scaler : StandardScaler
        Fitted scaler
    X_train_scaled : np.ndarray
        Scaled training features
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    return scaler, X_train_scaled
