"""
MLflow Neural Networks Demo

Demonstrates MLflow tracking with neural networks:
- Automatic experiment tracking
- Neural network specific logging
- Model comparison and versioning
- Loading models from MLflow registry
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ModelsManager import ModelsManager
from FeaturesGenerator import FeaturesGenerator
from Trainer import Trainer
from MLflowTracker import MLflowTracker, create_tracker

def demo_mlflow_with_neural_networks():
    """Demonstrate MLflow tracking with neural networks."""
    print("="*80)
    print("MLFLOW NEURAL NETWORKS TRACKING DEMO")
    print("="*80)
    
    # Check MLflow server availability
    tracker = create_tracker()
    if not tracker.is_available():
        print("\n⚠ MLflow server not available!")
        print("To start MLflow server:")
        print("1. Run: start_mlflow.bat")
        print("2. Or: mlflow server --host 127.0.0.1 --port 5000")
        print("3. Keep the server running and try again")
        return
    
    print("✓ MLflow server available")
    
    # Load sample data
    print("\n1. Loading sample data...")
    try:
        df = pd.read_csv("../data/hour/btc.csv")
        print(f"✓ Data loaded: {len(df):,} rows")
    except FileNotFoundError:
        print("✗ Sample data not found. Creating synthetic data...")
        # Create synthetic data for demo
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
        np.random.seed(42)
        prices = 30000 + np.cumsum(np.random.randn(len(dates)) * 100)
        volumes = np.random.randint(100, 10000, len(dates))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * (1 + np.random.uniform(0, 0.02, len(dates))),
            'low': prices * (1 - np.random.uniform(0, 0.02, len(dates))),
            'close': prices,
            'volume': volumes
        })
        print(f"✓ Synthetic data created: {len(df):,} rows")
    
    # Generate features
    print("\n2. Generating features...")
    fg = FeaturesGenerator()
    features_response = fg.generate_features(df.head(2000), method='crypto')  # Use subset for demo
    
    # Extract features and target
    top_features = ['atr_pct_28', 'atr_pct_14', 'volatility_20h', 'hl_spread_pct', 'volatility_50h']
    X = features_response['X_train'][top_features].dropna()
    y = features_response['y_train'].loc[X.index]
    
    print(f"✓ Features prepared: {X.shape}")
    print(f"  Target distribution: {y.value_counts().to_dict()}")
    
    # Create models manager with neural networks
    print("\n3. Creating models with neural networks...")
    models_manager = ModelsManager(
        models_dir='../models',
        include_neural_networks=True,
        sequence_length=15,  # Shorter for demo
        epochs=3,           # Very few epochs for demo
        batch_size=16       # Small batch for demo
    )
    
    # Enable only a few models for demo
    models_manager.enable_model('logistic_regression', True)
    models_manager.enable_model('xgboost', True)
    models_manager.enable_neural_network('cnn_simple', True)
    models_manager.enable_neural_network('lstm_simple', True)
    
    # Disable other neural networks for faster demo
    for model_name in ['cnn_deep', 'cnn_residual', 'lstm_stacked', 'lstm_attention']:
        models_manager.enable_neural_network(model_name, False)
    
    models = models_manager.create_models(enabled_only=True)
    print(f"✓ Created {len(models)} models for demo")
    
    # Train with MLflow tracking
    print("\n4. Training models with MLflow tracking...")
    trainer = Trainer(
        use_smote=True,
        optimize_threshold=True,
        use_scaler=True,
        use_mlflow=True,
        mlflow_experiment="ml_predict_15/neural_networks_demo",
        mlflow_tracking_uri="http://localhost:5000"
    )
    
    try:
        trained_models, scaler, results, best_model_name = trainer.train(
            models=models,
            X_train=X.values,
            y_train=y.values
        )
        
        print(f"\n✓ Training completed with MLflow tracking!")
        print(f"  Best model: {best_model_name}")
        print(f"  View results at: http://localhost:5000")
        
        # Print training results
        trainer.print_results()
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return
    
    # Demonstrate loading models from MLflow
    print("\n5. Loading models from MLflow...")
    try:
        from MLflowTracker import MLflowTracker
        
        # Load the best model from MLflow registry
        loaded_model = MLflowTracker.load_model_from_mlflow(
            model_name=f"ml_predict_15_{best_model_name}",
            version="latest",
            tracking_uri="http://localhost:5000"
        )
        
        if loaded_model:
            print(f"✓ Loaded model from MLflow: {best_model_name}")
            
            # Test prediction
            if hasattr(loaded_model, 'predict'):
                test_predictions = loaded_model.predict(X.values[-10:])
                print(f"✓ Test predictions: {test_predictions}")
        else:
            print("⚠ Could not load model from MLflow (may need time to register)")
            
    except Exception as e:
        print(f"⚠ Model loading failed: {e}")
    
    print(f"\n{'='*80}")
    print(f"MLFLOW DEMO COMPLETED!")
    print(f"{'='*80}")
    print(f"\nWhat was tracked:")
    print(f"✓ Training parameters (SMOTE, scaling, model configs)")
    print(f"✓ Model metrics (accuracy, F1, precision, recall)")
    print(f"✓ Neural network architectures and parameters")
    print(f"✓ Best model registered in MLflow")
    print(f"✓ Training artifacts (results, configs, summaries)")
    
    print(f"\nNext steps:")
    print(f"1. Open MLflow UI: http://localhost:5000")
    print(f"2. Navigate to experiment: ml_predict_15/neural_networks_demo")
    print(f"3. Compare different runs and models")
    print(f"4. Download models and artifacts")
    print(f"5. Use models for predictions")

def demo_mlflow_comparison():
    """Demonstrate comparing multiple MLflow runs."""
    print("\n" + "="*80)
    print("MLFLOW COMPARISON DEMO")
    print("="*80)
    
    print("\nThis demo shows how to compare multiple training runs:")
    print("1. Run this demo multiple times with different configurations")
    print("2. Open MLflow UI: http://localhost:5000")
    print("3. Select multiple runs in the experiment")
    print("4. Click 'Compare' to see side-by-side comparison")
    print("5. Analyze which configurations work best")
    
    print("\nMLflow UI Features:")
    print("• Parameters comparison (hyperparameters, data size, etc.)")
    print("• Metrics comparison (accuracy, F1, training time)")
    print("• Model artifacts (download trained models)")
    print("• Charts and visualizations")
    print("• Model versioning and registry")

def main():
    """Main demo function."""
    print("MLFLOW NEURAL NETWORKS TRACKING DEMONSTRATION")
    print("This script demonstrates comprehensive MLflow tracking")
    print("for both traditional ML models and neural networks.\n")
    
    try:
        # Demo MLflow with neural networks
        demo_mlflow_with_neural_networks()
        
        # Demo comparison features
        demo_mlflow_comparison()
        
        print("\n" + "="*80)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nMLflow Integration Summary:")
        print("✓ Automatic experiment tracking for all models")
        print("✓ Neural network specific parameter logging")
        print("✓ Model versioning and registry")
        print("✓ Comprehensive metrics and artifacts")
        print("✓ Easy model loading and deployment")
        print("✓ Web UI for experiment comparison")
        
        print("\nTo use MLflow in your workflow:")
        print("1. Start MLflow server: start_mlflow.bat")
        print("2. Train models with use_mlflow=True (default)")
        print("3. View results in web UI: http://localhost:5000")
        print("4. Compare experiments and select best models")
        print("5. Load models for production use")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        print("Please ensure:")
        print("1. MLflow server is running: start_mlflow.bat")
        print("2. All dependencies are installed: pip install -r requirements.txt")
        print("3. TensorFlow is available: pip install tensorflow keras")

if __name__ == "__main__":
    main()
