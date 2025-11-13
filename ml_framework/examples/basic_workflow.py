"""
Basic Workflow Example - Demonstrates the complete ML framework workflow.

This example shows:
1. Loading data
2. Generating features
3. Creating target variable
4. Training models
5. Testing models
6. Backtesting strategy
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_provider import DataProvider
from src.features_generator import FeaturesGenerator
from src.model_manager import ModelManager
from src.ml_trainer import ML_Trainer
from src.ml_tester import ML_Tester
from src.backtester import Backtester


def main():
    """Run complete ML workflow."""
    
    print("\n" + "="*70)
    print("ML FRAMEWORK - BASIC WORKFLOW EXAMPLE")
    print("="*70)
    
    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    print("\n[STEP 1] Loading data...")
    
    data_provider = DataProvider()
    
    # Load from Yahoo Finance (you can also use load_csv)
    df = data_provider.load_yahoo(
        ticker='BTC-USD',
        start_date='2020-01-01',
        end_date='2023-12-31',
        interval='1d'
    )
    
    # Validate and clean data
    data_provider.validate_data(df)
    df = data_provider.clean_data(df, method='drop')
    
    # ========================================================================
    # STEP 2: Generate Features
    # ========================================================================
    print("\n[STEP 2] Generating features...")
    
    features_gen = FeaturesGenerator()
    
    # Generate technical indicators
    df_features = features_gen.generate_features(df, feature_set='basic')
    
    # Create target variable (predict if price will increase by 2% in next 5 days)
    df_features = features_gen.create_target(
        df_features,
        target_type='classification',
        future_bars=5,
        threshold=0.02
    )
    
    print(f"✓ Dataset ready: {len(df_features)} rows, {len(df_features.columns)} columns")
    
    # ========================================================================
    # STEP 3: Split Data
    # ========================================================================
    print("\n[STEP 3] Splitting data...")
    
    train_df, val_df, test_df = data_provider.split_data(
        df_features,
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    # ========================================================================
    # STEP 4: Setup Models
    # ========================================================================
    print("\n[STEP 4] Setting up models...")
    
    model_manager = ModelManager()
    model_configs = model_manager.get_models()
    
    # ========================================================================
    # STEP 5: Train Models
    # ========================================================================
    print("\n[STEP 5] Training models...")
    
    trainer = ML_Trainer()
    
    # Get feature columns (exclude OHLCV and target)
    feature_cols = features_gen.get_feature_names()
    
    training_results = trainer.train(
        df=train_df,
        target_col='target',
        feature_cols=feature_cols,
        model_configs=model_configs,
        test_size=0.0,  # We already split the data
        scale_features=True
    )
    
    # ========================================================================
    # STEP 6: Test Models
    # ========================================================================
    print("\n[STEP 6] Testing models...")
    
    tester = ML_Tester()
    
    test_results = tester.evaluate(
        df=test_df,
        models=training_results['models'],
        scaler=training_results['scaler'],
        target_col='target',
        feature_cols=feature_cols
    )
    
    # ========================================================================
    # STEP 7: Save Models
    # ========================================================================
    print("\n[STEP 7] Saving models...")
    
    save_dir = model_manager.save_models(
        models=training_results['models'],
        scaler=training_results['scaler'],
        metadata={
            'best_model': training_results['best_model'],
            'feature_cols': feature_cols,
            'ticker': 'BTC-USD'
        }
    )
    
    # ========================================================================
    # STEP 8: Backtest Best Model
    # ========================================================================
    print("\n[STEP 8] Backtesting best model...")
    
    best_model_name = training_results['best_model']
    best_model = training_results['models'][best_model_name]
    
    backtester = Backtester(
        initial_capital=10000.0,
        position_size=1.0,
        commission=0.001
    )
    
    backtest_results = backtester.run(
        df=test_df,
        model=best_model,
        scaler=training_results['scaler'],
        feature_cols=feature_cols,
        price_col='close'
    )
    
    # Plot results
    backtester.plot_results()
    
    # ========================================================================
    # STEP 9: Summary
    # ========================================================================
    print("\n" + "="*70)
    print("WORKFLOW COMPLETE")
    print("="*70)
    print(f"\n✓ Data loaded: {len(df)} rows")
    print(f"✓ Features generated: {len(feature_cols)}")
    print(f"✓ Models trained: {len(training_results['models'])}")
    print(f"✓ Best model: {best_model_name}")
    print(f"✓ Models saved to: {save_dir}")
    print(f"✓ Backtest return: {backtest_results['metrics']['total_return']*100:.2f}%")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
