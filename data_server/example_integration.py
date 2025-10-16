"""
Example: Integration with Backtesting and ML Projects

This example shows how to use the Data Provider Server with:
1. Backtesting strategies
2. ML model training
3. Data analysis
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# Base URL for the Data Provider Server
BASE_URL = "http://localhost:5001"


def get_ohlc_data(ticker, resolution='1d', period='max', limit=None):
    """
    Get OHLC data from the server
    
    Args:
        ticker: Ticker symbol (e.g., 'BTC-USD', 'SPY')
        resolution: Data resolution ('1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo')
        period: Data period ('1d', '5d', '1mo', '1y', '5y', 'max')
        limit: Limit number of rows (optional)
    
    Returns:
        pandas DataFrame with OHLC data
    """
    params = {
        'resolution': resolution,
        'period': period
    }
    if limit:
        params['limit'] = limit
    
    try:
        response = requests.get(f"{BASE_URL}/api/data/{ticker}", params=params)
        response.raise_for_status()
        
        data = response.json()
        df = pd.DataFrame(data['data'])
        
        # Convert timestamp to datetime and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Ensure column names are lowercase
        df.columns = [col.lower() for col in df.columns]
        
        print(f"✓ Loaded {len(df)} rows for {ticker} ({resolution}, {period})")
        return df
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to server!")
        print("Make sure the server is running: python server.py")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def example_1_simple_analysis():
    """Example 1: Simple data analysis"""
    print("\n" + "="*60)
    print("Example 1: Simple Data Analysis")
    print("="*60)
    
    # Get Bitcoin daily data
    df = get_ohlc_data('BTC-USD', resolution='1d', period='1y')
    
    if df is not None:
        # Calculate basic statistics
        print(f"\nPrice Statistics:")
        print(f"  Current Price: ${df['close'].iloc[-1]:,.2f}")
        print(f"  52-Week High: ${df['high'].max():,.2f}")
        print(f"  52-Week Low: ${df['low'].min():,.2f}")
        print(f"  Average Volume: {df['volume'].mean():,.0f}")
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        print(f"\nReturns:")
        print(f"  Daily Avg Return: {df['returns'].mean():.2%}")
        print(f"  Daily Volatility: {df['returns'].std():.2%}")
        print(f"  Total Return: {(df['close'].iloc[-1] / df['close'].iloc[0] - 1):.2%}")


def example_2_moving_average_strategy():
    """Example 2: Simple Moving Average Crossover Strategy"""
    print("\n" + "="*60)
    print("Example 2: Moving Average Crossover Strategy")
    print("="*60)
    
    # Get data
    df = get_ohlc_data('BTC-USD', resolution='1d', period='1y')
    
    if df is not None:
        # Calculate moving averages
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['SMA_20'] > df['SMA_50'], 'signal'] = 1  # Buy signal
        df.loc[df['SMA_20'] < df['SMA_50'], 'signal'] = -1  # Sell signal
        
        # Calculate strategy returns
        df['returns'] = df['close'].pct_change()
        df['strategy_returns'] = df['signal'].shift(1) * df['returns']
        
        # Calculate cumulative returns
        df['cumulative_returns'] = (1 + df['returns']).cumprod()
        df['cumulative_strategy_returns'] = (1 + df['strategy_returns']).cumprod()
        
        # Performance metrics
        buy_hold_return = df['cumulative_returns'].iloc[-1] - 1
        strategy_return = df['cumulative_strategy_returns'].iloc[-1] - 1
        
        print(f"\nStrategy Performance:")
        print(f"  Buy & Hold Return: {buy_hold_return:.2%}")
        print(f"  Strategy Return: {strategy_return:.2%}")
        print(f"  Outperformance: {(strategy_return - buy_hold_return):.2%}")
        
        # Count trades
        df['position_change'] = df['signal'].diff()
        num_trades = (df['position_change'] != 0).sum()
        print(f"  Number of Trades: {num_trades}")


def example_3_multi_asset_comparison():
    """Example 3: Compare multiple assets"""
    print("\n" + "="*60)
    print("Example 3: Multi-Asset Comparison")
    print("="*60)
    
    # Get data for multiple assets using batch request
    response = requests.post(f"{BASE_URL}/api/batch", json={
        'tickers': ['BTC-USD', 'ETH-USD', 'SPY', 'AAPL'],
        'resolution': '1d',
        'period': '1y',
        'limit': 365
    })
    
    if response.status_code == 200:
        batch_data = response.json()
        
        print(f"\n✓ Loaded {batch_data['success_count']} assets")
        
        # Process each asset
        returns = {}
        for ticker, info in batch_data['results'].items():
            df = pd.DataFrame(info['data'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Calculate total return
            total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
            returns[ticker] = total_return
        
        # Display comparison
        print(f"\n1-Year Returns:")
        for ticker, ret in sorted(returns.items(), key=lambda x: x[1], reverse=True):
            print(f"  {ticker:12s}: {ret:>7.2f}%")


def example_4_ml_features():
    """Example 4: Generate ML features for model training"""
    print("\n" + "="*60)
    print("Example 4: ML Feature Generation")
    print("="*60)
    
    # Get data
    df = get_ohlc_data('BTC-USD', resolution='1d', period='2y')
    
    if df is not None:
        # Generate technical indicators as features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'SMA_{window}'] = df['close'].rolling(window=window).mean()
            df[f'price_to_SMA_{window}'] = df['close'] / df[f'SMA_{window}']
        
        # Volatility
        df['volatility_20'] = df['returns'].rolling(window=20).std()
        
        # Volume features
        df['volume_SMA_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_SMA_20']
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Target: Next day return
        df['target'] = df['returns'].shift(-1)
        
        # Remove NaN values
        df_clean = df.dropna()
        
        print(f"\nGenerated Features:")
        print(f"  Total rows: {len(df_clean)}")
        print(f"  Features: {len(df_clean.columns)}")
        print(f"\nFeature columns:")
        for col in df_clean.columns:
            print(f"  - {col}")
        
        print(f"\nSample data:")
        print(df_clean[['close', 'returns', 'SMA_20', 'RSI', 'target']].tail())


def example_5_download_for_backtesting():
    """Example 5: Download data for backtesting framework"""
    print("\n" + "="*60)
    print("Example 5: Download Data for Backtesting")
    print("="*60)
    
    # Download data for backtesting
    tickers = ['BTC-USD', 'ETH-USD', 'SPY']
    
    for ticker in tickers:
        # Download CSV
        response = requests.get(f"{BASE_URL}/api/data/{ticker}/csv", params={
            'resolution': '1d',
            'period': 'max'
        })
        
        if response.status_code == 200:
            filename = f"backtest_data_{ticker.replace('-', '_')}.csv"
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"✓ Downloaded: {filename}")
            
            # Load and verify
            df = pd.read_csv(filename)
            print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")


def main():
    """Run all examples"""
    print("="*60)
    print("Data Provider Server - Integration Examples")
    print("="*60)
    print(f"\nServer URL: {BASE_URL}")
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        if response.status_code == 200:
            print("✓ Server is running")
        else:
            print("❌ Server returned error")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running!")
        print("\nPlease start the server first:")
        print("  python server.py")
        return
    
    # Run examples
    try:
        example_1_simple_analysis()
        example_2_moving_average_strategy()
        example_3_multi_asset_comparison()
        example_4_ml_features()
        example_5_download_for_backtesting()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")


if __name__ == '__main__':
    main()
