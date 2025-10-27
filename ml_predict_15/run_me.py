"""
Main Training and Testing Script

This script trains ML models on historical data and tests them on future data.
All helper functions have been moved to modular files in the src/ directory.
"""

import pandas as pd
from src.model_training import train, test
from src.visualization import print_model_summary


# Data paths
PATH_TRAIN = "data/btc_2022.csv"
PATH_TEST = "data/btc_2023.csv"


def main():
    """Main execution function."""
    
    # Load data
    print("Loading training data...")
    df_train = pd.read_csv(PATH_TRAIN)
    print("Loading test data...")
    df_test = pd.read_csv(PATH_TEST)
    
    # Train models
    print("\n" + "="*80)
    print("TRAINING MODELS")
    print("="*80)
    # Parameters:
    # - target_bars=45: Look ahead 45 bars (e.g., 45 hours for hourly data)
    # - target_pct=3.0: Predict if price increases by 3% or more
    # Note: Larger target_pct (e.g., 3-5%) creates harder but more meaningful predictions
    #       Smaller target_pct (e.g., 0.5-1%) may result in similar model performance
    models, scaler, train_results, best_model_name = train(
        df_train, 
        target_bars=45, 
        target_pct=3.0,  # Changed from 1.0 to 3.0 for more meaningful predictions
        use_smote=True,
        use_gpu=False,
        n_jobs=-1
    )
    
    # Print training summary
    print_model_summary(train_results)
    
    # Test on next year data
    print("\n" + "="*80)
    print("TESTING MODELS")
    print("="*80)
    test_metrics = test(models, scaler, df_test)
    
    # Print test summary
    print("\n" + "="*80)
    print("SUMMARY: Test Metrics (Next Year Data)")
    print("="*80)
    for model_name, metrics in test_metrics.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    
    print("\n" + "="*80)
    print("TRAINING AND TESTING COMPLETE!")
    print("="*80)
    print(f"\nBest Model: {best_model_name.upper()}")
    print(f"All models saved to: models/")
    print(f"\nNext steps:")
    print(f"  - Run 'python backtest_quick_start.py' for backtesting")
    print(f"  - Run 'python backtest_example.py' for advanced examples")


if __name__ == "__main__":
    main()
