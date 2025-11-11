"""
Example script demonstrating the enhanced CNN optimizer with:
- Custom model building using optimized neuron counts
- Optimization results visualization
- Model comparison and evaluation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.enhanced_cnn_optimizer import EnhancedCNNOptimizer
from src.data_loader import load_data
from src.features_generator import FeaturesGenerator

def main():
    print("\n" + "="*80)
    print("ENHANCED CNN OPTIMIZATION EXAMPLE")
    print("="*80 + "\n")
    
    # 1. Load and prepare data
    print("Step 1: Loading data...")
    ticker = 'BTC-USD'
    df = load_data(ticker, start_date='2014-01-01', end_date='2025-11-10')
    print(f"✓ Loaded {len(df)} samples for {ticker}")
    
    # 2. Generate features
    print("\nStep 2: Generating features...")
    fg = FeaturesGenerator()
    df_features, features_names = fg.generate_features(df)
    print(f"✓ Generated {len(df_features.columns)} features")
    
    # 3. Create target labels (binary classification)
    print("\nStep 3: Creating target labels...")
    future_period = 15
    threshold = 0.01
    n_trials = 30  # Increase for better results (e.g., 50-100)

    future_prices = df_features['close'].shift(-future_period)
    price_changes = (future_prices - df_features['close']) / df_features['close']
    target = (price_changes > threshold).astype(int)
    target = target[:-future_period]  # Remove last NaN values
    df_features = df_features.iloc[:-future_period]
    print(f"✓ Created binary targets (positive: {target.sum()}, negative: {len(target) - target.sum()})")
    
    # 4. Create sequences for CNN
    print("\nStep 4: Creating sequences...")
    sequence_length = future_period
    
    # Select only numeric columns
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    df_numeric = df_features[numeric_cols]
    
    X, y = [], []
    for i in range(len(df_numeric) - sequence_length):
        X.append(df_numeric.iloc[i:i + sequence_length].values)
        y.append(target.iloc[i + sequence_length])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"✓ Created {len(X)} sequences of shape {X.shape[1:]}")
    
    # 5. Split data
    print("\nStep 5: Splitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"✓ Train: {len(X_train)} samples")
    print(f"✓ Validation: {len(X_val)} samples")
    print(f"✓ Test: {len(X_test)} samples")
    
    # 6. Initialize optimizer
    print("\nStep 6: Initializing optimizer...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    optimizer = EnhancedCNNOptimizer(X_train, y_train, X_val, y_val, input_shape)
    print(f"✓ Optimizer initialized with input shape: {input_shape}")
    
    # 7. Run optimization
    print("\nStep 7: Running hyperparameter optimization...")
    print("This will test different architectures and neuron counts...")
    study = optimizer.optimize(n_trials=n_trials, show_progress=True)
    
    # 8. Visualize optimization results
    print("\nStep 8: Visualizing optimization results...")
    optimizer.plot_optimization_results(study, save_path='optimization_results.png')
    
    # 9. Compare top models on test data
    print("\nStep 9: Comparing top models on test data...")
    comparison_df = optimizer.compare_optimized_models(study, X_test, y_test, top_n=5)
    
    # 10. Save comparison results
    print("\nStep 10: Saving comparison results...")
    comparison_df.to_csv('model_comparison_results.csv', index=False)
    print("✓ Comparison results saved to: model_comparison_results.csv")
    
    # 11. Save best model with all artifacts
    print("\nStep 11: Saving best model...")
    save_dir = optimizer.save_best_model(study, model_name='btc_trend_cnn', base_dir='models')
    print(f"✓ Best model saved to: {save_dir}")
    
    # 12. Print summary
    print("\n" + "="*80)
    print("OPTIMIZATION SUMMARY (Maximizing Recall)")
    print("="*80)
    print(f"Total trials: {n_trials}")
    print(f"Best validation Recall: {study.best_value:.4f}")
    print(f"Best architecture: {study.best_params['architecture']}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        if key != 'architecture':
            print(f"  {key}: {value}")
    
    print(f"\nTop model test performance:")
    best_model = comparison_df.iloc[0]
    print(f"  Test Accuracy: {best_model['Test_Accuracy']:.4f}")
    print(f"  Test Precision: {best_model['Test_Precision']:.4f}")
    print(f"  Test Recall: {best_model['Test_Recall']:.4f} ← OPTIMIZED METRIC")
    print(f"  Test F1: {best_model['Test_F1']:.4f}")
    print("="*80 + "\n")
    
    print("✓ Optimization complete!")
    print("✓ Check 'optimization_results.png' for visualization")
    print("✓ Check 'model_comparison_results.csv' for detailed results")
    print(f"✓ Check '{save_dir}' for saved model and all artifacts")

if __name__ == '__main__':
    main()
