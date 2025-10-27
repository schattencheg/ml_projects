import pandas as pd
import numpy as np
import ta

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
        self.add_sma(df)
        self.add_rsi(df)
        self.add_stochastic(df)
        self.add_bollinger(df)
        return df

    def add_sma(self, df, window = 10):
        # SMA small moving average
        df['SMA_10'] = df['Close'].rolling(window=window).mean()
        df['SMA_20'] = df['Close'].rolling(window=window*2).mean()
        # SMA cross
        df['SMA_cross_10'] = (df['Close'] <= df['SMA_10'].shift(1) ) & (df['Close'] > df['SMA_10'])
        df['SMA_cross_10'] = df['SMA_cross_10'].astype(int)
        df['SMA_cross'] = (df['SMA_20'].shift(1) < df['SMA_10'].shift(1)) & (df['SMA_20'] > df['SMA_10'])
        df['SMA_cross'] = df['SMA_cross'].astype(int)
        # DROP SMA columns
        df.drop(['SMA_10', 'SMA_20'], axis=1, inplace=True)
        return df

    def add_rsi(self, df, min = 30, max = 70):
        # Calculate RSI using ta library
        rsi_indicator = ta.momentum.RSIIndicator(close=df['Close'], window=14)
        df['RSI'] = rsi_indicator.rsi()
        
        # RSI crossover signals
        df['RSI_cross_min'] = (df['RSI'].shift(1) < min) & (df['RSI'] > min)
        df['RSI_cross_min'] = df['RSI_cross_min'].astype(int)
        #df['RSI_cross_max'] = (df['RSI'].shift(1) > max) & (df['RSI'] < max)
        #df['RSI_cross_max'] = df['RSI_cross_max'].astype(int)
        
        # DROP RSI column
        df.drop(['RSI'], axis=1, inplace=True)
        return df

    def add_stochastic(self, df, min = 20, max = 80):
        # Calculate Stochastic Oscillator using ta library
        stoch_indicator = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
        df['STOCH_K'] = stoch_indicator.stoch()  # %K line
        df['STOCH_D'] = stoch_indicator.stoch_signal()  # %D line (signal)
        
        # STOCH crossover signals
        df['STOCH_cross_min'] = (df['STOCH_K'].shift(1) < min) & (df['STOCH_K'] > min)
        df['STOCH_cross_min'] = df['STOCH_cross_min'].astype(int)
        #df['STOCH_cross_max'] = (df['STOCH_K'].shift(1) > max) & (df['STOCH_K'] < max)
        #df['STOCH_cross_max'] = df['STOCH_cross_max'].astype(int)
        
        # DROP STOCH columns
        df.drop(['STOCH_K', 'STOCH_D'], axis=1, inplace=True)
        return df

    def add_bollinger(self, df, window = 14):
        bollinger_indicator = ta.volatility.BollingerBands(close=df['Close'], window=window)
        df['BOLLINGER_High'] = bollinger_indicator.bollinger_hband()
        df['BOLLINGER_Low'] = bollinger_indicator.bollinger_lband()
        df['BOLLINGER_Middle'] = bollinger_indicator.bollinger_mavg()
        
        # BOLLINGER crossover signals
        #df['BOLLINGER_cross_min'] = (df['BOLLINGER_High'].shift(1) < df['Close']) & (df['BOLLINGER_High'] > df['Close'])
        #df['BOLLINGER_cross_min'] = df['BOLLINGER_cross_min'].astype(int)
        df['BOLLINGER_cross_mid'] = (df['Close'] < df['BOLLINGER_Low'].shift(1)) & (df['Close'] > df['BOLLINGER_Low'])
        df['BOLLINGER_cross_mid'] = df['BOLLINGER_cross_mid'].astype(int)
        #df['BOLLINGER_cross_max'] = (df['Close'] > df['BOLLINGER_High'].shift(1)) & (df['Close'] < df['BOLLINGER_High'])
        #df['BOLLINGER_cross_max'] = df['BOLLINGER_cross_max'].astype(int)

        # DROP BOLLINGER columns
        df.drop(['BOLLINGER_High', 'BOLLINGER_Low', 'BOLLINGER_Middle'], axis=1, inplace=True)
        return df

    def add_target(self, df, N=45, M=3.0):
        """
        Add target column showing if Close is greater than N bars ahead by M percent.
        
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
            DataFrame with added 'target' column (1 if Close > Close_N_bars_ahead * (1 + M/100), else 0)
        """
        # Calculate the Close price N bars ahead
        close_n_bars_ahead = df['Close'].shift(-N)
        
        # Calculate the threshold: Close * (1 + M/100)
        threshold = df['Close'] * (1 + M / 100.0)
        
        # Create binary target: 1 if Close_N_bars_ahead > threshold, else 0
        df['target'] = (close_n_bars_ahead > threshold).astype(int)
        
        # Optional: Also add the actual percentage change for analysis
        df['pct_change_N'] = ((close_n_bars_ahead - df['Close']) / df['Close'] * 100.0)
        
        return df
