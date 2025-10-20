"""
Test MLflow Server Connection

This script tests the connectivity to the MLflow server
and provides diagnostic information.
"""

import requests
import sys

BASE_URL = "http://localhost:5001"
MLFLOW_URL = "http://localhost:5000"


def test_data_server():
    """Test if data server is running"""
    print("\n" + "="*60)
    print("Testing Data Provider Server")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if response.status_code == 200:
            print(f"✓ Data Provider Server is running at {BASE_URL}")
            return True
        else:
            print(f"✗ Data Provider Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to Data Provider Server at {BASE_URL}")
        print("  Make sure the server is running: python server.py")
        return False
    except Exception as e:
        print(f"✗ Error connecting to Data Provider Server: {str(e)}")
        return False


def test_mlflow_status_endpoint():
    """Test MLflow status via data server endpoint"""
    print("\n" + "="*60)
    print("Testing MLflow Status Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/api/mlflow/status", timeout=5)
        data = response.json()
        
        print(f"\nMLflow Status:")
        print(f"  Enabled: {data.get('enabled', 'Unknown')}")
        print(f"  Status: {data.get('status', 'Unknown')}")
        print(f"  URL: {data.get('url', 'Unknown')}")
        print(f"  Message: {data.get('message', 'No message')}")
        
        if response.status_code == 200 and data.get('status') == 'running':
            print(f"\n✓ MLflow server is accessible via status endpoint")
            return True
        else:
            print(f"\n⚠ MLflow server status: {data.get('status', 'unknown')}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to Data Provider Server at {BASE_URL}")
        return False
    except Exception as e:
        print(f"✗ Error checking MLflow status: {str(e)}")
        return False


def test_mlflow_direct():
    """Test direct connection to MLflow server"""
    print("\n" + "="*60)
    print("Testing Direct MLflow Connection")
    print("="*60)
    
    # Test root endpoint
    try:
        print(f"\nTesting {MLFLOW_URL}/ ...")
        response = requests.get(f"{MLFLOW_URL}/", timeout=5)
        if response.status_code == 200:
            print(f"✓ MLflow UI is accessible at {MLFLOW_URL}")
            ui_accessible = True
        else:
            print(f"⚠ MLflow UI returned status code: {response.status_code}")
            ui_accessible = False
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to MLflow server at {MLFLOW_URL}")
        ui_accessible = False
    except Exception as e:
        print(f"✗ Error connecting to MLflow UI: {str(e)}")
        ui_accessible = False
    
    # Test health endpoint
    try:
        print(f"\nTesting {MLFLOW_URL}/health ...")
        response = requests.get(f"{MLFLOW_URL}/health", timeout=5)
        if response.status_code == 200:
            print(f"✓ MLflow health endpoint is responding")
            health_ok = True
        else:
            print(f"⚠ MLflow health endpoint returned status code: {response.status_code}")
            health_ok = False
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to MLflow health endpoint")
        health_ok = False
    except Exception as e:
        print(f"⚠ MLflow health endpoint error: {str(e)}")
        health_ok = False
    
    # Test API endpoint
    try:
        print(f"\nTesting {MLFLOW_URL}/api/2.0/mlflow/experiments/list ...")
        response = requests.get(f"{MLFLOW_URL}/api/2.0/mlflow/experiments/list", timeout=5)
        if response.status_code == 200:
            data = response.json()
            experiments = data.get('experiments', [])
            print(f"✓ MLflow API is responding")
            print(f"  Found {len(experiments)} experiment(s)")
            api_ok = True
        else:
            print(f"⚠ MLflow API returned status code: {response.status_code}")
            api_ok = False
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to MLflow API")
        api_ok = False
    except Exception as e:
        print(f"⚠ MLflow API error: {str(e)}")
        api_ok = False
    
    return ui_accessible or health_ok or api_ok


def test_mlflow_with_python_client():
    """Test MLflow using Python client"""
    print("\n" + "="*60)
    print("Testing MLflow Python Client")
    print("="*60)
    
    try:
        import mlflow
        
        # Set tracking URI
        mlflow.set_tracking_uri(MLFLOW_URL)
        print(f"\nTracking URI set to: {MLFLOW_URL}")
        
        # Try to list experiments
        try:
            experiments = mlflow.search_experiments()
            print(f"✓ MLflow Python client can connect")
            print(f"  Found {len(experiments)} experiment(s):")
            for exp in experiments[:5]:  # Show first 5
                print(f"    - {exp.name} (ID: {exp.experiment_id})")
            return True
        except Exception as e:
            print(f"✗ MLflow Python client error: {str(e)}")
            return False
            
    except ImportError:
        print("⚠ MLflow Python package is not installed")
        print("  Install it with: pip install mlflow")
        return False
    except Exception as e:
        print(f"✗ Error using MLflow Python client: {str(e)}")
        return False


def provide_recommendations(results):
    """Provide recommendations based on test results"""
    print("\n" + "="*60)
    print("Recommendations")
    print("="*60 + "\n")
    
    if not results['data_server']:
        print("❌ Data Provider Server is not running")
        print("   → Start the server: python server.py")
        return
    
    if results['mlflow_status'] and results['mlflow_direct']:
        print("✅ All systems operational!")
        print(f"   → Data Server: {BASE_URL}")
        print(f"   → MLflow UI: {MLFLOW_URL}")
        print("\n   You can now use both servers for your ML workflows.")
        return
    
    if not results['mlflow_status'] and not results['mlflow_direct']:
        print("❌ MLflow server is not accessible")
        print("\n   Possible solutions:")
        print("   1. Wait a few more seconds - MLflow may still be starting")
        print("   2. Check if MLflow is enabled in config.py (MLFLOW_ENABLED = True)")
        print("   3. Verify MLflow is installed: pip install mlflow")
        print("   4. Check if port 5000 is available")
        print("   5. Try starting MLflow manually:")
        print("      mlflow server --host 127.0.0.1 --port 5000")
        print(f"\n   6. Check server logs in data_server.log for errors")
        return
    
    if results['mlflow_status'] and not results['mlflow_direct']:
        print("⚠ MLflow status endpoint works but direct connection fails")
        print("\n   This is unusual. Try:")
        print("   1. Wait a few seconds and test again")
        print("   2. Check firewall settings")
        print(f"   3. Try accessing {MLFLOW_URL} in your browser")
        return
    
    if not results['mlflow_status'] and results['mlflow_direct']:
        print("⚠ MLflow is accessible directly but status endpoint reports issues")
        print("\n   MLflow is working! The status endpoint may need a moment to update.")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("MLflow Server Connection Test")
    print("="*60)
    print("\nThis script will test connectivity to:")
    print(f"  - Data Provider Server: {BASE_URL}")
    print(f"  - MLflow Server: {MLFLOW_URL}")
    
    results = {
        'data_server': False,
        'mlflow_status': False,
        'mlflow_direct': False,
        'mlflow_client': False
    }
    
    # Run tests
    results['data_server'] = test_data_server()
    
    if results['data_server']:
        results['mlflow_status'] = test_mlflow_status_endpoint()
        results['mlflow_direct'] = test_mlflow_direct()
        results['mlflow_client'] = test_mlflow_with_python_client()
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60 + "\n")
    
    print(f"Data Provider Server:  {'✓ PASS' if results['data_server'] else '✗ FAIL'}")
    print(f"MLflow Status API:     {'✓ PASS' if results['mlflow_status'] else '✗ FAIL'}")
    print(f"MLflow Direct Access:  {'✓ PASS' if results['mlflow_direct'] else '✗ FAIL'}")
    print(f"MLflow Python Client:  {'✓ PASS' if results['mlflow_client'] else '✗ FAIL'}")
    
    # Provide recommendations
    provide_recommendations(results)
    
    print("\n" + "="*60 + "\n")
    
    # Exit code
    if results['data_server'] and (results['mlflow_status'] or results['mlflow_direct']):
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
