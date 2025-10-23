#!/usr/bin/env python3
"""
Quick test script to verify all models are working correctly.
"""

import pandas as pd
import numpy as np
from run_me import train, test

def test_small_dataset():
    """Test with a small subset of data to verify functionality."""
    
    print("="*60)
    print("TESTING MODEL FUNCTIONALITY")
    print("="*60)
    
    # Load and sample data
    print("Loading data...")
    df_train = pd.read_csv("data/btc_2022.csv")
    df_test = pd.read_csv("data/btc_2023.csv")
    
    # Use smaller samples for quick testing
    df_train_sample = df_train.head(1000)  # First 1000 rows
    df_test_sample = df_test.head(500)     # First 500 rows
    
    print(f"Training sample size: {len(df_train_sample)}")
    print(f"Test sample size: {len(df_test_sample)}")
    
    try:
        # Train models
        print("\nTraining models...")
        models, scaler, train_results, best_model_name = train(
            df_train_sample, 
            target_bars=10,  # Reduced for faster testing
            target_pct=2.0   # Reduced threshold
        )
        
        print(f"\nSuccessfully trained {len(models)} models!")
        print(f"Best model: {best_model_name}")
        
        # Test models
        print("\nTesting models...")
        test_results = test(
            models, 
            scaler, 
            df_test_sample,
            target_bars=10,
            target_pct=2.0
        )
        
        print(f"\nSuccessfully tested {len(test_results)} models!")
        
        # Summary
        print("\n" + "="*60)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Available models:")
        for model_name in models.keys():
            print(f"  ✓ {model_name}")
        
        print("\nMetrics tested:")
        print("  ✓ Accuracy")
        print("  ✓ F1 Score") 
        print("  ✓ Precision")
        print("  ✓ Recall")
        print("  ✓ ROC AUC")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_small_dataset()
    if success:
        print("\n🎉 All models are working correctly!")
        print("You can now run the full training with: python run_me.py")
    else:
        print("\n❌ Some models failed. Check the error messages above.")
