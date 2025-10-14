"""Simple test script to verify the setup works."""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from utils import get_logger, config, EnvConfig
        print("✅ Utils import successful")
        
        from data import DataFetcher
        print("✅ Data fetcher import successful")
        
        from features import FeatureEngineer
        print("✅ Feature engineer import successful")
        
        from models import RandomForestModel
        print("✅ Models import successful")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {str(e)}")
        return False

def test_data_fetch():
    """Test basic data fetching."""
    print("\nTesting data fetching...")
    
    try:
        from data import DataFetcher
        
        fetcher = DataFetcher()
        print("✅ DataFetcher created successfully")
        
        # Test with a small amount of data
        data = fetcher.fetch_symbol_data("AAPL", period="5d", save_to_file=False)
        print(f"✅ Data fetched successfully: {data.shape}")
        print(f"   Columns: {list(data.columns)}")
        print(f"   Date range: {data.index.min()} to {data.index.max()}")
        
        return True
    except Exception as e:
        print(f"❌ Data fetch failed: {str(e)}")
        return False

def test_basic_model():
    """Test basic model creation and training."""
    print("\nTesting basic model...")
    
    try:
        import numpy as np
        from models import RandomForestModel
        
        # Create sample data
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randn(100, 1)
        
        # Create and train model
        config = {'n_estimators': 5, 'random_state': 42}
        model = RandomForestModel(config)
        
        results = model.train(X, y)
        print("✅ Model training successful")
        
        # Test prediction
        predictions = model.predict(X[:5])
        print(f"✅ Model prediction successful: {predictions.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model test failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("🚀 OHLC Project Setup Test")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Fetch Test", test_data_fetch),
        ("Model Test", test_basic_model)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 20)
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 TEST SUMMARY")
    print("=" * 40)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Your setup is working correctly.")
        print("\nNext steps:")
        print("1. Run: python scripts/run_example.py")
        print("2. Or train a model: python src/train.py --symbol AAPL --model rf")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check your internet connection for data fetching")
        print("3. Ensure you're in the correct directory")

if __name__ == "__main__":
    main()
