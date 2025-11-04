"""
Neural Networks Demo Script

Demonstrates how to use the new neural network models:
- Multiple 1D-CNN variants
- Multiple LSTM variants
- GAN implementation
- Integration with existing ML pipeline
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ModelsManager import ModelsManager
from NeuralNetworksManager import NeuralNetworksManager, CryptocurrencyGAN
from FeaturesGenerator import FeaturesGenerator

def demo_neural_networks():
    """Demonstrate neural network models."""
    print("="*80)
    print("NEURAL NETWORKS DEMO")
    print("="*80)
    
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
    features_response = fg.generate_features(df.head(1000), method='crypto')  # Use subset for demo
    
    # Extract features and target
    top_features = ['atr_pct_28', 'atr_pct_14', 'volatility_20h', 'hl_spread_pct', 'volatility_50h']
    X = features_response['X_train'][top_features].dropna()
    y = features_response['y_train'].loc[X.index]
    
    print(f"✓ Features prepared: {X.shape}")
    print(f"  Feature columns: {list(X.columns)}")
    print(f"  Target distribution: {y.value_counts().to_dict()}")
    
    # Create neural networks manager
    print("\n3. Creating Neural Networks Manager...")
    nn_manager = NeuralNetworksManager(
        sequence_length=20,  # Shorter for demo
        epochs=5,           # Fewer epochs for demo
        batch_size=16       # Smaller batch for demo
    )
    
    # Print available models
    nn_manager.print_config()
    
    # Disable some models for faster demo
    models_to_disable = ['cnn_deep', 'cnn_residual', 'lstm_stacked', 'lstm_attention']
    for model_name in models_to_disable:
        nn_manager.enable_model(model_name, False)
    
    print(f"\n4. Creating neural network models (demo subset)...")
    neural_models = nn_manager.create_models(enabled_only=True)
    
    # Train one model as example
    if neural_models:
        print(f"\n5. Training example model...")
        model_name = list(neural_models.keys())[0]
        model = neural_models[model_name]
        
        print(f"Training {model_name}...")
        try:
            # Fit the model
            model.fit(X.values, y.values)
            print(f"✓ {model_name} trained successfully")
            
            # Make predictions
            predictions = model.predict(X.values[-50:])  # Predict on last 50 samples
            probabilities = model.predict_proba(X.values[-50:])
            
            print(f"✓ Predictions made: {len(predictions)} samples")
            print(f"  Prediction distribution: {np.bincount(predictions)}")
            print(f"  Average confidence: {np.max(probabilities, axis=1).mean():.3f}")
            
        except Exception as e:
            print(f"✗ Training failed: {e}")
    
    print(f"\n6. Integration with ModelsManager...")
    # Create models manager with neural networks
    models_manager = ModelsManager(
        models_dir='../models',
        include_neural_networks=True,
        sequence_length=20,
        epochs=5,
        batch_size=16
    )
    
    # Configure neural networks
    models_manager.configure_neural_networks(epochs=3)  # Even fewer for demo
    
    # Disable most neural networks for demo
    for model_name in models_to_disable:
        models_manager.enable_neural_network(model_name, False)
    
    # Print full configuration
    models_manager.print_config()
    
    print(f"\n✓ Neural networks successfully integrated with ModelsManager!")

def demo_gan():
    """Demonstrate GAN implementation."""
    print("\n" + "="*80)
    print("GAN DEMO")
    print("="*80)
    
    print("\n1. Creating sample price data...")
    # Create sample price data
    np.random.seed(42)
    n_samples = 500
    prices = 30000 + np.cumsum(np.random.randn(n_samples) * 100)
    volumes = np.random.randint(100, 10000, n_samples)
    
    # Create features matrix
    X = np.column_stack([
        prices,
        volumes,
        np.random.randn(n_samples),  # Additional feature
        np.random.randn(n_samples),  # Additional feature
        np.random.randn(n_samples)   # Additional feature
    ])
    
    print(f"✓ Sample data created: {X.shape}")
    
    print("\n2. Creating and training GAN...")
    gan = CryptocurrencyGAN(sequence_length=20, latent_dim=50)
    
    try:
        # Train GAN (very few epochs for demo)
        print("Training GAN (this may take a moment)...")
        gan.train(X, epochs=10, batch_size=16, sample_interval=5)
        
        print("\n3. Generating synthetic sequences...")
        synthetic_sequences = gan.generate_sequences(n_samples=5)
        print(f"✓ Generated {len(synthetic_sequences)} synthetic sequences")
        print(f"  Shape: {synthetic_sequences.shape}")
        
        print("\n4. Extracting discriminator features...")
        features = gan.get_discriminator_features(X)
        print(f"✓ Extracted features: {features.shape}")
        
        print(f"\n✓ GAN demo completed successfully!")
        
    except Exception as e:
        print(f"✗ GAN demo failed: {e}")
        print("Note: GAN training requires more data and computational resources")

def main():
    """Main demo function."""
    print("NEURAL NETWORKS AND GAN DEMONSTRATION")
    print("This script demonstrates the new neural network capabilities")
    print("including multiple CNN and LSTM variants, plus GAN implementation.\n")
    
    try:
        # Demo neural networks
        demo_neural_networks()
        
        # Demo GAN
        demo_gan()
        
        print("\n" + "="*80)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nSummary of implemented features:")
        print("✓ Multiple 1D-CNN variants (simple, deep, residual, attention, dilated)")
        print("✓ Multiple LSTM variants (simple, bidirectional, stacked, attention)")
        print("✓ Hybrid CNN-LSTM models")
        print("✓ GRU variants (simple, bidirectional)")
        print("✓ GAN for cryptocurrency price prediction")
        print("✓ KerasClassifierWrapper for sklearn compatibility")
        print("✓ Automatic sequence creation for time series data")
        print("✓ Integration with existing ModelsManager")
        
        print("\nNext steps:")
        print("1. Use these models in your main training pipeline")
        print("2. Adjust hyperparameters (sequence_length, epochs, batch_size)")
        print("3. Enable/disable specific models based on your needs")
        print("4. Experiment with the GAN for data augmentation")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        print("Please check that all dependencies are installed:")
        print("  pip install tensorflow keras")

if __name__ == "__main__":
    main()
