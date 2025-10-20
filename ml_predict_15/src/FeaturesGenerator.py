import pandas as pd
import numpy as np

class FeaturesGenerator:
    def __init__(self):
        pass

    def returnificate(self, df):
        df['ret_Open'] = np.log(df['Open'] / df['Open'].shift(1))
        df['ret_High'] = np.log(df['High'] / df['High'].shift(1))
        df['ret_Low'] = np.log(df['Low'] / df['Low'].shift(1))
        df['ret_Close'] = np.log(df['Close'] / df['Close'].shift(1))
        return df

    def add_features(self, df):
        # SMA small moving average
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_15'] = df['Close'].rolling(window=15).mean()
        # SMA cross
        df['SMA_5_10'] = df['SMA_5'] - df['SMA_10']
        df['SMA_10_15'] = df['SMA_10'] - df['SMA_15']
        return df

    def add_target(self, df, N=45, M=3.0):
        """
        Add target column showing if Close is greater than N bars ago by M percent.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLC data
        N : int, default=15
            Number of bars to look back
        M : float, default=1.0
            Percent threshold (e.g., 1.0 means 1% increase)
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with added 'target' column (1 if Close > Close_N_bars_ago * (1 + M/100), else 0)
        """
        # Calculate the Close price N bars ago
        close_n_bars_ago = df['Close'].shift(N)
        
        # Calculate the threshold: Close N bars ago * (1 + M/100)
        threshold = close_n_bars_ago * (1 + M / 100.0)
        
        # Create binary target: 1 if current Close > threshold, else 0
        df['target'] = (df['Close'] > threshold).astype(int)
        
        # Optional: Also add the actual percentage change for analysis
        df['pct_change_N'] = ((df['Close'] - close_n_bars_ago) / close_n_bars_ago * 100.0)
        
        return df
