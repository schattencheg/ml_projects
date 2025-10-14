"""Simple example script that demonstrates the OHLC prediction pipeline."""

import sys
import os
from pathlib import Path

# Ensure we're in the right directory and set up paths
project_root = Path(__file__).parent
os.chdir(project_root)
sys.path.insert(0, str(project_root / "src"))

def main():
    """Run a simple example."""
    print("🚀 OHLC Prediction - Simple Example")
    print("=" * 50)
    
    try:
        # Test imports first
        print("📦 Testing imports...")
        
        # Import step by step to identify issues
        print("  - Importing utils...")
        from utils import get_logger
        logger = get_logger(__name__)
        
        print("  - Importing data fetcher...")
        from data.fetch_data import DataFetcher
        
        print("  - Importing preprocessor...")
        from data.preprocessor import DataPreprocessor
        
        print("  - Importing feature engineer...")
        from features.feature_engineer import FeatureEngineer
        
        print("  - Importing random forest model...")
        from models.random_forest_model import RandomForestModel
        
        print("✅ All imports successful!")
        
        # Initialize components
        print("\n📊 Initializing components...")
        logger = get_logger(__name__)
        data_fetcher = DataFetcher()
        feature_engineer = FeatureEngineer()
        
        # Fetch sample data
        print("\n📈 Fetching sample data for AAPL...")
        symbol = "AAPL"
        raw_data = data_fetcher.fetch_symbol_data(
            symbol=symbol, 
            period="3y",  # 6 months of data
            save_to_file=True
        )
        print(f"✅ Data fetched: {raw_data.shape}")
        print(f"   Date range: {raw_data.index.min().date()} to {raw_data.index.max().date()}")
        
        # Basic data preprocessing
        print("\n🔧 Processing features...")
        from data import DataPreprocessor
        preprocessor = DataPreprocessor()
        
        # Clean data
        cleaned_data = preprocessor.clean_data(raw_data)
        print(f"✅ Data cleaned: {cleaned_data.shape}")
        
        # Engineer features (simplified)
        processed_data = feature_engineer.engineer_features(cleaned_data)
        processed_data = processed_data.dropna()
        print(f"✅ Features engineered: {processed_data.shape}")
        
        # Prepare model data
        print("\n🤖 Preparing model data...")
        target_columns = ['Close']
        feature_columns = [col for col in processed_data.columns 
                          if col not in ['Symbol'] + target_columns]
        
        X = processed_data[feature_columns].values
        y = processed_data[target_columns].values
        
        # Simple train/test split (last 20% for testing)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"✅ Data split - Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Train a simple Random Forest model
        print("\n🌳 Training Random Forest model...")
        model_config = {
            'n_estimators': 50,
            'max_depth': 10,
            'random_state': 42
        }
        
        model = RandomForestModel(model_config)
        training_results = model.train(X_train, y_train)
        print(f"✅ Model trained! Training R² score: {training_results['train_score']:.4f}")
        
        # Evaluate model
        print("\n📊 Evaluating model...")
        metrics = model.evaluate(X_test, y_test)
        
        print("📈 Model Performance:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        # Make sample predictions
        print("\n🔮 Making sample predictions...")
        predictions = model.predict(X_test[:5])
        actual = y_test[:5]
        
        print("Sample Predictions vs Actual:")
        for i in range(5):
            pred_val = predictions[i] if predictions.ndim == 1 else predictions[i, 0]
            actual_val = actual[i] if actual.ndim == 1 else actual[i, 0]
            print(f"   Day {i+1}: Predicted=${pred_val:.2f}, Actual=${actual_val:.2f}")
        
        # Save model
        print("\n💾 Saving model...")
        model_path = f"models/{symbol}_rf_simple_model.pkl"
        Path("models").mkdir(exist_ok=True)
        model.save_model(model_path)
        print(f"✅ Model saved to: {model_path}")
        
        print("\n🎉 Simple example completed successfully!")
        print("=" * 50)
        print("📁 Generated files:")
        print(f"   - {model_path}")
        print(f"   - data/{symbol}_yahoo_data.csv")
        print("\n🚀 Next steps:")
        print("   1. Run the full example: python scripts/run_example.py")
        print("   2. Start MLflow: python scripts/setup_mlflow.py")
        print("   3. Train more models: python src/train.py --symbol AAPL --model rf")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure you installed requirements: pip install -r requirements.txt")
        print("   2. Check internet connection for data fetching")
        print("   3. Run the setup test: python test_setup.py")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
