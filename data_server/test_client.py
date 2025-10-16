"""
Test client for Data Provider Server
Demonstrates how to use the API endpoints
"""
import requests
import pandas as pd
from typing import Dict, List

# Base URL for the server
BASE_URL = "http://localhost:5001"


def test_health():
    """Test health check endpoint"""
    print("\n=== Testing Health Check ===")
    response = requests.get(f"{BASE_URL}/api/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")


def test_list_instruments():
    """Test list instruments endpoint"""
    print("\n=== Testing List Instruments ===")
    response = requests.get(f"{BASE_URL}/api/instruments")
    data = response.json()
    print(f"Crypto instruments: {len(data['crypto'])}")
    print(f"Regular market instruments: {len(data['regular_market'])}")
    print(f"Total: {data['total']}")


def test_get_data(ticker: str, resolution: str = '1d', period: str = 'max', limit: int = 10):
    """Test get data endpoint"""
    print(f"\n=== Testing Get Data: {ticker} ===")
    response = requests.get(f"{BASE_URL}/api/data/{ticker}", params={
        'resolution': resolution,
        'period': period,
        'limit': limit
    })
    
    if response.status_code == 200:
        data = response.json()
        print(f"Ticker: {data['ticker']}")
        print(f"Resolution: {data['resolution']}")
        print(f"Period: {data['period']}")
        print(f"Rows: {data['rows']}")
        print(f"Date range: {data['start_date']} to {data['end_date']}")
        
        # Convert to DataFrame for display
        df = pd.DataFrame(data['data'])
        print(f"\nFirst few rows:")
        print(df.head())
        print(f"\nLast few rows:")
        print(df.tail())
        
        return df
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.json()}")
        return None


def test_batch_request(tickers: List[str], resolution: str = '1d', period: str = '1y', limit: int = 5):
    """Test batch request endpoint"""
    print(f"\n=== Testing Batch Request ===")
    print(f"Tickers: {tickers}")
    
    response = requests.post(f"{BASE_URL}/api/batch", json={
        'tickers': tickers,
        'resolution': resolution,
        'period': period,
        'limit': limit
    })
    
    if response.status_code == 200:
        data = response.json()
        print(f"Success count: {data['success_count']}")
        print(f"Error count: {data['error_count']}")
        
        for ticker, info in data['results'].items():
            print(f"\n{ticker}:")
            print(f"  Rows: {info['rows']}")
            print(f"  Date range: {info['start_date']} to {info['end_date']}")
            
            # Show first row
            if info['data']:
                print(f"  First row: {info['data'][0]}")
        
        if data['errors']:
            print(f"\nErrors: {data['errors']}")
        
        return data
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.json()}")
        return None


def test_refresh_data(ticker: str, resolution: str = '1d', period: str = 'max'):
    """Test refresh data endpoint"""
    print(f"\n=== Testing Refresh Data: {ticker} ===")
    response = requests.post(f"{BASE_URL}/api/refresh/{ticker}", params={
        'resolution': resolution,
        'period': period
    })
    
    if response.status_code == 200:
        data = response.json()
        print(f"Status: {data['status']}")
        print(f"Message: {data['message']}")
        print(f"Rows: {data['rows']}")
        print(f"Date range: {data['start_date']} to {data['end_date']}")
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.json()}")


def test_download_csv(ticker: str, resolution: str = '1d', period: str = 'max'):
    """Test CSV download endpoint"""
    print(f"\n=== Testing CSV Download: {ticker} ===")
    response = requests.get(f"{BASE_URL}/api/data/{ticker}/csv", params={
        'resolution': resolution,
        'period': period
    })
    
    if response.status_code == 200:
        filename = f"{ticker}_{resolution}_{period}.csv"
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {filename}")
        
        # Read and display
        df = pd.read_csv(filename)
        print(f"Rows: {len(df)}")
        print(f"\nFirst few rows:")
        print(df.head())
    else:
        print(f"Error: {response.status_code}")


def main():
    """Run all tests"""
    print("\n")
    print("=" * 60)
    print("Data Provider Server - Test Client")
    print("=" * 60)
    
    try:
        # Test 1: Health check
        test_health()
        
        # Test 2: List instruments
        test_list_instruments()
        
        # Test 3: Get Bitcoin data
        test_get_data('BTC-USD', resolution='1d', period='max', limit=10)
        
        # Test 4: Get Ethereum hourly data
        test_get_data('ETH-USD', resolution='1h', period='1mo', limit=10)
        
        # Test 5: Get SPY data
        test_get_data('SPY', resolution='1d', period='1y', limit=10)
        
        # Test 6: Batch request
        test_batch_request(['BTC-USD', 'ETH-USD', 'SPY'], resolution='1d', period='1mo', limit=5)
        
        # Test 7: Refresh data (commented out to avoid unnecessary downloads)
        # test_refresh_data('BTC-USD', resolution='1d', period='max')
        
        # Test 8: Download CSV (commented out to avoid file creation)
        # test_download_csv('BTC-USD', resolution='1d', period='max')
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to the server!")
        print("Make sure the server is running: python server.py")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")


if __name__ == '__main__':
    main()
