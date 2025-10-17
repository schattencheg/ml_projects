"""
WebSocket Example Usage for DataProviderLocal

This file demonstrates how to use the WebSocket functionality
to retrieve OHLC data in real-time from the data server.

Make sure the server is running before executing this script:
    python server.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.Data.DataProviderLocal import DataProviderLocal
from src.Background.enums import DataResolution, DataPeriod


def example_1_simple_socket_call():
    """Example 1: Simple WebSocket call to get data"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Simple WebSocket Call")
    print("="*60)
    
    provider = DataProviderLocal()
    
    # Get BTC-USD daily data via WebSocket
    df = provider.get_data_socket(
        ticker='BTC-USD',
        resolution=DataResolution.DAY_01,
        period=DataPeriod.YEAR_MAX,
        local_only=True,
        limit=10
    )
    
    if df is not None:
        print(f"\n✓ Received {len(df)} rows for BTC-USD")
        print("\nLast 5 rows:")
        print(df.tail())
    else:
        print("\n✗ Failed to retrieve data")


def example_2_with_callbacks():
    """Example 2: WebSocket with callbacks for real-time updates"""
    print("\n" + "="*60)
    print("EXAMPLE 2: WebSocket with Callbacks")
    print("="*60)
    
    # Define callback functions
    def on_data_received(data):
        print(f"\n✓ Callback: Data received for {data['ticker']}")
        print(f"  - Rows: {data['rows']}")
        print(f"  - Date range: {data['start_date']} to {data['end_date']}")
        print(f"  - Resolution: {data['resolution']}")
        print(f"  - Period: {data['period']}")
    
    def on_error(error):
        print(f"\n✗ Callback: Error occurred - {error}")
    
    provider = DataProviderLocal()
    
    # Get ETH-USD hourly data with callbacks
    df = provider.get_data_socket(
        ticker='ETH-USD',
        resolution=DataResolution.HOUR_01,
        period=DataPeriod.MONTH_01,
        local_only=True,
        callback=on_data_received,
        error_callback=on_error,
        limit=20
    )
    
    if df is not None:
        print(f"\n✓ DataFrame shape: {df.shape}")
        print("\nFirst 3 rows:")
        print(df.head(3))


def example_3_batch_request():
    """Example 3: Batch request for multiple tickers via WebSocket"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Batch WebSocket Request")
    print("="*60)
    
    provider = DataProviderLocal()
    
    tickers = ['BTC-USD', 'ETH-USD', 'SPY', 'AAPL']
    
    print(f"\nRequesting data for: {', '.join(tickers)}")
    
    results = provider.get_batch_data_socket(
        tickers=tickers,
        resolution=DataResolution.DAY_01,
        period=DataPeriod.YEAR_01,
        local_only=True,
        limit=5
    )
    
    print(f"\n✓ Successfully retrieved data for {len(results)} tickers\n")
    
    for ticker, df in results.items():
        print(f"{ticker}:")
        print(f"  - Rows: {len(df)}")
        print(f"  - Date range: {df.index[0]} to {df.index[-1]}")
        print(f"  - Last Close: ${df['Close'].iloc[-1]:.2f}")
        print()


def example_4_batch_with_progress():
    """Example 4: Batch request with progress callback"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Batch Request with Progress Tracking")
    print("="*60)
    
    # Define progress callback
    def on_progress(progress_info):
        ticker = progress_info.get('ticker', 'Unknown')
        progress = progress_info.get('progress', '')
        print(f"  → Processing {ticker} ({progress})")
    
    def on_error(error):
        print(f"\n✗ Error: {error}")
    
    provider = DataProviderLocal()
    
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD']
    
    print(f"\nRequesting data for {len(tickers)} tickers with progress tracking:\n")
    
    results = provider.get_batch_data_socket(
        tickers=tickers,
        resolution=DataResolution.DAY_01,
        period=DataPeriod.MONTH_06,
        local_only=True,
        progress_callback=on_progress,
        error_callback=on_error,
        limit=10
    )
    
    print(f"\n✓ Batch request completed!")
    print(f"  - Success: {len(results)} tickers")
    print(f"  - Failed: {len(tickers) - len(results)} tickers\n")
    
    for ticker, df in results.items():
        print(f"{ticker}: {len(df)} rows, Latest: ${df['Close'].iloc[-1]:.2f}")


def example_5_comparison_rest_vs_socket():
    """Example 5: Compare REST API vs WebSocket performance"""
    print("\n" + "="*60)
    print("EXAMPLE 5: REST API vs WebSocket Comparison")
    print("="*60)
    
    import time
    
    provider = DataProviderLocal()
    ticker = 'BTC-USD'
    
    # Test REST API
    print("\nTesting REST API...")
    start_time = time.time()
    df_rest = provider.get_data(
        ticker=ticker,
        resolution=DataResolution.DAY_01,
        period=DataPeriod.YEAR_01
    )
    rest_time = time.time() - start_time
    
    # Test WebSocket
    print("Testing WebSocket...")
    start_time = time.time()
    df_socket = provider.get_data_socket(
        ticker=ticker,
        resolution=DataResolution.DAY_01,
        period=DataPeriod.YEAR_01,
        local_only=True
    )
    socket_time = time.time() - start_time
    
    print(f"\nResults for {ticker}:")
    print(f"  REST API:")
    print(f"    - Time: {rest_time:.3f} seconds")
    print(f"    - Rows: {len(df_rest) if df_rest is not None else 0}")
    print(f"  WebSocket:")
    print(f"    - Time: {socket_time:.3f} seconds")
    print(f"    - Rows: {len(df_socket) if df_socket is not None else 0}")
    
    if rest_time > 0 and socket_time > 0:
        speedup = rest_time / socket_time
        print(f"\n  WebSocket is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than REST API")


def example_6_persistent_connection():
    """Example 6: Reuse WebSocket connection for multiple requests"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Persistent WebSocket Connection")
    print("="*60)
    
    provider = DataProviderLocal()
    
    # Connect once
    print("\nConnecting to WebSocket server...")
    if not provider._connect_socket():
        print("✗ Failed to connect")
        return
    
    print("✓ Connected successfully\n")
    
    # Make multiple requests without disconnecting
    tickers = ['BTC-USD', 'ETH-USD', 'SPY']
    
    for ticker in tickers:
        print(f"Fetching {ticker}...")
        df = provider.get_data_socket(
            ticker=ticker,
            resolution=DataResolution.DAY_01,
            period=DataPeriod.MONTH_01,
            local_only=True,
            auto_disconnect=False,  # Keep connection alive
            limit=5
        )
        
        if df is not None:
            print(f"  ✓ Received {len(df)} rows, Last Close: ${df['Close'].iloc[-1]:.2f}\n")
        else:
            print(f"  ✗ Failed to retrieve data\n")
    
    # Disconnect manually
    print("Disconnecting...")
    provider._disconnect_socket()
    print("✓ Disconnected")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("WebSocket Data Provider Examples")
    print("="*60)
    print("\nMake sure the server is running: python server.py")
    print("Press Ctrl+C to stop at any time\n")
    
    try:
        # Run examples
        example_1_simple_socket_call()
        
        input("\nPress Enter to continue to Example 2...")
        example_2_with_callbacks()
        
        input("\nPress Enter to continue to Example 3...")
        example_3_batch_request()
        
        input("\nPress Enter to continue to Example 4...")
        example_4_batch_with_progress()
        
        input("\nPress Enter to continue to Example 5...")
        example_5_comparison_rest_vs_socket()
        
        input("\nPress Enter to continue to Example 6...")
        example_6_persistent_connection()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n\n✗ Error running examples: {str(e)}")


if __name__ == '__main__':
    main()
