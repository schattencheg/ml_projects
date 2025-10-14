"""Simple test script to verify basic functionality."""

import sys
import os
from pathlib import Path

# Set up paths
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def test_basic_imports():
    """Test basic imports step by step."""
    print("🔍 Testing basic imports...")
    
    try:
        print("  - Testing utils...")
        from utils import get_logger
        logger = get_logger(__name__)
        print("  ✅ Utils imported successfully")
        
        print("  - Testing data fetcher...")
        from data.fetch_data import DataFetcher
        print("  ✅ DataFetcher imported successfully")
        
        print("  - Testing preprocessor...")
        from data.preprocessor import DataPreprocessor
        print("  ✅ DataPreprocessor imported successfully")
        
        print("  - Testing base model...")
        from models.base_model import BaseModel
        print("  ✅ BaseModel imported successfully")
        
        print("  - Testing random forest model...")
        from models.random_forest_model import RandomForestModel
        print("  ✅ RandomForestModel imported successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_data_fetching():
    """Test data fetching functionality."""
    print("\n📊 Testing data fetching...")
    
    try:
        from data.fetch_data import DataFetcher
        
        fetcher = DataFetcher()
        print("  ✅ DataFetcher created")
        
        # Fetch small amount of data
        print("  - Fetching AAPL data (5 days)...")
        data = fetcher.fetch_symbol_data("AAPL", period="5d", save_to_file=False)
        print(f"  ✅ Data fetched: {data.shape}")
        print(f"     Columns: {list(data.columns)}")
        print(f"     Date range: {data.index.min().date()} to {data.index.max().date()}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data fetching failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model creation."""
    print("\n🤖 Testing model creation...")
    
    try:
        from models.random_forest_model import RandomForestModel
        import numpy as np
        
        # Create simple model
        config = {
            'n_estimators': 5,
            'max_depth': 3,
            'random_state': 42
        }
        
        model = RandomForestModel(config)
        print("  ✅ RandomForestModel created")
        
        # Create dummy data
        X = np.random.randn(50, 10)
        y = np.random.randn(50, 1)
        
        # Train model
        results = model.train(X, y)
        print("  ✅ Model trained successfully")
        print(f"     Training score: {results.get('train_score', 'N/A')}")
        
        # Test prediction
        predictions = model.predict(X[:5])
        print(f"  ✅ Predictions made: {predictions.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Simple MLFlow Test - Step by Step")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Data Fetching", test_data_fetching),
        ("Model Creation", test_model_creation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        success = test_func()
        results.append((test_name, success))
        
        if not success:
            print(f"\n⚠️  {test_name} failed. Stopping here.")
            break
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Basic setup is working.")
        print("\nNext steps:")
        print("1. Try: python run_simple_example.py")
        print("2. Or: python src/data/fetch_data.py")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
