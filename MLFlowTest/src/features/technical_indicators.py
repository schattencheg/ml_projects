"""Technical indicators for market data analysis."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import ta

try:
    from src.utils import get_logger
except ImportError:
    # Handle case when running as script directly
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from src.utils import get_logger

logger = get_logger(__name__)

class TechnicalIndicators:
    """Technical indicators calculator for OHLC data."""
    
    def __init__(self):
        """Initialize technical indicators calculator."""
        pass
    
    def sma(self, data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average.
        
        Args:
            data: Price series
            period: Period for moving average
            
        Returns:
            SMA series
        """
        return data.rolling(window=period).mean()
    
    def ema(self, data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average.
        
        Args:
            data: Price series
            period: Period for moving average
            
        Returns:
            EMA series
        """
        return data.ewm(span=period).mean()
    
    def rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index.
        
        Args:
            data: Price series
            period: Period for RSI calculation
            
        Returns:
            RSI series
        """
        return ta.momentum.RSIIndicator(data, window=period).rsi()
    
    def macd(
        self, 
        data: pd.Series, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> Dict[str, pd.Series]:
        """MACD (Moving Average Convergence Divergence).
        
        Args:
            data: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            
        Returns:
            Dictionary with MACD, signal, and histogram
        """
        macd_indicator = ta.trend.MACD(data, window_fast=fast, window_slow=slow, window_sign=signal)
        
        return {
            'MACD': macd_indicator.macd(),
            'MACD_Signal': macd_indicator.macd_signal(),
            'MACD_Histogram': macd_indicator.macd_diff()
        }
    
    def bollinger_bands(
        self, 
        data: pd.Series, 
        period: int = 20, 
        std: float = 2.0
    ) -> Dict[str, pd.Series]:
        """Bollinger Bands.
        
        Args:
            data: Price series
            period: Period for moving average
            std: Standard deviation multiplier
            
        Returns:
            Dictionary with upper, middle, and lower bands
        """
        bb_indicator = ta.volatility.BollingerBands(data, window=period, window_dev=std)
        
        return {
            'BB_Upper': bb_indicator.bollinger_hband(),
            'BB_Middle': bb_indicator.bollinger_mavg(),
            'BB_Lower': bb_indicator.bollinger_lband(),
            'BB_Width': bb_indicator.bollinger_wband(),
            'BB_Percent': bb_indicator.bollinger_pband()
        }
    
    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Period for ATR calculation
            
        Returns:
            ATR series
        """
        return ta.volatility.AverageTrueRange(high, low, close, window=period).average_true_range()
    
    def stochastic(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        k_period: int = 14,
        d_period: int = 3
    ) -> Dict[str, pd.Series]:
        """Stochastic Oscillator.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            k_period: %K period
            d_period: %D period
            
        Returns:
            Dictionary with %K and %D
        """
        stoch_indicator = ta.momentum.StochasticOscillator(
            high, low, close, window=k_period, smooth_window=d_period
        )
        
        return {
            'Stoch_K': stoch_indicator.stoch(),
            'Stoch_D': stoch_indicator.stoch_signal()
        }
    
    def williams_r(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        period: int = 14
    ) -> pd.Series:
        """Williams %R.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Period for calculation
            
        Returns:
            Williams %R series
        """
        return ta.momentum.WilliamsRIndicator(high, low, close, lbp=period).williams_r()
    
    def cci(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        period: int = 20
    ) -> pd.Series:
        """Commodity Channel Index.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Period for calculation
            
        Returns:
            CCI series
        """
        return ta.trend.CCIIndicator(high, low, close, window=period).cci()
    
    def adx(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        period: int = 14
    ) -> Dict[str, pd.Series]:
        """Average Directional Index.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Period for calculation
            
        Returns:
            Dictionary with ADX, +DI, and -DI
        """
        adx_indicator = ta.trend.ADXIndicator(high, low, close, window=period)
        
        return {
            'ADX': adx_indicator.adx(),
            'ADX_Pos': adx_indicator.adx_pos(),
            'ADX_Neg': adx_indicator.adx_neg()
        }
    
    def obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume.
        
        Args:
            close: Close price series
            volume: Volume series
            
        Returns:
            OBV series
        """
        return ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    
    def vwap(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        volume: pd.Series
    ) -> pd.Series:
        """Volume Weighted Average Price.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            volume: Volume series
            
        Returns:
            VWAP series
        """
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
    
    def ichimoku(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series,
        conversion_period: int = 9,
        base_period: int = 26,
        span_b_period: int = 52,
        displacement: int = 26
    ) -> Dict[str, pd.Series]:
        """Ichimoku Cloud.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            conversion_period: Conversion line period
            base_period: Base line period
            span_b_period: Span B period
            displacement: Displacement for spans
            
        Returns:
            Dictionary with Ichimoku components
        """
        ichimoku_indicator = ta.trend.IchimokuIndicator(
            high, low, window1=conversion_period, window2=base_period, window3=span_b_period
        )
        
        return {
            'Ichimoku_Conversion': ichimoku_indicator.ichimoku_conversion_line(),
            'Ichimoku_Base': ichimoku_indicator.ichimoku_base_line(),
            'Ichimoku_A': ichimoku_indicator.ichimoku_a(),
            'Ichimoku_B': ichimoku_indicator.ichimoku_b()
        }
    
    def fibonacci_retracement(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        period: int = 20
    ) -> Dict[str, pd.Series]:
        """Fibonacci Retracement Levels.
        
        Args:
            high: High price series
            low: Low price series
            period: Period for high/low calculation
            
        Returns:
            Dictionary with Fibonacci levels
        """
        rolling_high = high.rolling(window=period).max()
        rolling_low = low.rolling(window=period).min()
        diff = rolling_high - rolling_low
        
        levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        fib_levels = {}
        
        for level in levels:
            fib_levels[f'Fib_{level}'] = rolling_high - (diff * level)
        
        return fib_levels
    
    def pivot_points(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series
    ) -> Dict[str, pd.Series]:
        """Pivot Points.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            
        Returns:
            Dictionary with pivot points and support/resistance levels
        """
        # Use previous day's data for pivot calculation
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)
        
        pivot = (prev_high + prev_low + prev_close) / 3
        
        return {
            'Pivot': pivot,
            'R1': 2 * pivot - prev_low,
            'R2': pivot + (prev_high - prev_low),
            'R3': prev_high + 2 * (pivot - prev_low),
            'S1': 2 * pivot - prev_high,
            'S2': pivot - (prev_high - prev_low),
            'S3': prev_low - 2 * (prev_high - pivot)
        }
    
    def calculate_all_indicators(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Calculate all configured technical indicators.
        
        Args:
            data: OHLCV DataFrame
            config: Configuration dictionary for indicators
            
        Returns:
            DataFrame with all indicators added
        """
        logger.info("Calculating technical indicators")
        df = data.copy()
        
        # Extract OHLCV data
        high = df['High']
        low = df['Low']
        open_price = df['Open']
        close = df['Close']
        volume = df['Volume']
        
        # Calculate indicators based on configuration
        for indicator_config in config.get('technical_indicators', []):
            indicator_name = indicator_config['name']
            
            try:
                if indicator_name == 'SMA':
                    for period in indicator_config['periods']:
                        df[f'SMA_{period}'] = self.sma(close, period)
                
                elif indicator_name == 'EMA':
                    for period in indicator_config['periods']:
                        df[f'EMA_{period}'] = self.ema(close, period)
                
                elif indicator_name == 'RSI':
                    period = indicator_config['period']
                    df['RSI'] = self.rsi(close, period)
                
                elif indicator_name == 'MACD':
                    fast = indicator_config['fast']
                    slow = indicator_config['slow']
                    signal = indicator_config['signal']
                    macd_data = self.macd(close, fast, slow, signal)
                    for key, series in macd_data.items():
                        df[key] = series
                
                elif indicator_name == 'Bollinger_Bands':
                    period = indicator_config['period']
                    std = indicator_config['std']
                    bb_data = self.bollinger_bands(close, period, std)
                    for key, series in bb_data.items():
                        df[key] = series
                
                elif indicator_name == 'ATR':
                    period = indicator_config['period']
                    df['ATR'] = self.atr(high, low, close, period)
                
                elif indicator_name == 'Volume_SMA':
                    for period in indicator_config['periods']:
                        df[f'Volume_SMA_{period}'] = self.sma(volume, period)
                
                # Add more indicators as needed
                
            except Exception as e:
                logger.error(f"Error calculating {indicator_name}: {str(e)}")
                continue
        
        logger.info(f"Technical indicators calculated. New shape: {df.shape}")
        return df
