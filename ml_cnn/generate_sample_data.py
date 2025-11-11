#!/usr/bin/env python3
"""
Generate sample OHLCV data for testing the time series classification notebook.
This creates realistic synthetic financial data with proper OHLC relationships.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_ohlcv_data(n_days=1000, start_price=100, volatility=0.02, start_date='2020-01-01'):
    """
    Generate synthetic OHLCV data with realistic properties.
    
    Args:
        n_days: Number of trading days to generate
        start_price: Starting price
        volatility: Daily volatility (standard deviation of returns)
        start_date: Start date for the time series
    
    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate date range
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    
    # Generate returns with some autocorrelation and trend
    returns = np.random.normal(0.0005, volatility, n_days)  # Small positive drift
    
    # Add autocorrelation to make it more realistic
    for i in range(1, len(returns)):
        returns[i] += 0.1 * returns[i-1]
    
    # Add some regime changes (volatility clustering)
    regime_changes = np.random.choice([0, 1], size=n_days, p=[0.95, 0.05])
    returns[regime_changes == 1] *= 3  # Higher volatility periods
    
    # Generate close prices
    close_prices = [start_price]
    for ret in returns[1:]:
        close_prices.append(close_prices[-1] * (1 + ret))
    
    # Generate OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, close_prices)):
        # Generate realistic OHLC from close price
        daily_range = abs(np.random.normal(0, close * 0.015))  # Daily range
        
        # Generate open price (influenced by previous close)
        if i == 0:
            open_price = close + np.random.normal(0, close * 0.005)
        else:
            gap = np.random.normal(0, close * 0.003)  # Overnight gap
            open_price = close + gap
        
        # Generate high and low
        high = max(open_price, close) + np.random.uniform(0, daily_range)
        low = min(open_price, close) - np.random.uniform(0, daily_range)
        
        # Ensure OHLC relationships are maintained
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Generate volume (correlated with price volatility and range)
        base_volume = 1000000  # Base volume
        volatility_factor = 1 + abs(returns[i]) * 20  # Higher volume on volatile days
        range_factor = 1 + (high - low) / close * 10  # Higher volume on wide range days
        volume = int(np.random.lognormal(np.log(base_volume), 0.5) * volatility_factor * range_factor)
        
        data.append({
            'date': date,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    
    return df

if __name__ == "__main__":
    # Generate sample data
    print("Generating sample OHLCV data...")
    
    # Generate different datasets
    datasets = {
        'sample_data_1000.csv': generate_ohlcv_data(n_days=1000, start_price=100, volatility=0.02),
        'sample_data_2000.csv': generate_ohlcv_data(n_days=2000, start_price=150, volatility=0.025),
        'high_vol_data.csv': generate_ohlcv_data(n_days=1500, start_price=200, volatility=0.04)
    }
    
    # Save datasets
    for filename, df in datasets.items():
        filepath = f"data/{filename}"
        df.to_csv(filepath)
        print(f"Saved {len(df)} rows to {filepath}")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
        print(f"  Average volume: {df['volume'].mean():,.0f}")
        print()
    
    print("✅ Sample data generation completed!")
    print("You can now use these CSV files in the Jupyter notebook.")
