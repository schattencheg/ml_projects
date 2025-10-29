"""
Train and Save All Models

This script trains all ML models and saves them to the models folder
for later use in backtesting.

Run this script first before running backtest examples.
"""

import pandas as pd
from src.model_training import train


def main():
    """Train and save all models."""
    
    print("="*80)
    print("TRAIN AND SAVE ALL MODELS")
    print("="*80)
    
    # Load training data
    print("\n[1/2] Loading training data...")
    path_train = "data/btc_2022.csv"
    df_train = pd.read_csv(path_train)
    print(f"  Training data: {df_train.shape[0]} rows")
    
    # Train models
    print("\n[2/2] Training models...")
    print("  This may take a few minutes...\n")
    
    # Prepare data
    X, y = prepare_data(df_train, target_bars, target_pct)
    models, scaler, train_results, best_model_name = train(X, y)
    
    # Summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    
    print(f"\n✓ Trained {len(models)} models")
    print(f"✓ Best model: {best_model_name}")
    print(f"✓ All models and scaler saved to 'models/' directory")
    
    print("\nModel Performance Summary:")
    print("-"*80)
    for model_name, results in train_results.items():
        print(f"  {model_name:25s} - Accuracy: {results['accuracy']:.4f}, F1: {results['f1']:.4f}")
    
    print("\n" + "="*80)
    print("Next Steps:")
    print("  1. Run 'python backtest_quick_start.py' for quick backtesting")
    print("  2. Run 'python backtest_example.py' for advanced examples")
    print("="*80)


if __name__ == "__main__":
    main()
