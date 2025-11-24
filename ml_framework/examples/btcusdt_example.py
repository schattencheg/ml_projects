"""
BTCUSDT Complete ML Workflow Example

This example demonstrates the complete ML framework workflow for BTCUSDT:
1. Download BTCUSDT data from Yahoo Finance
2. Generate technical indicators as features
3. Create target variable for price prediction
4. Train multiple ML models
5. Test and evaluate models
6. Save trained models
7. Backtest the best model

Author: ML Framework
Date: 2024-11-24
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_provider import DataProvider
from src.features_generator import FeaturesGenerator
from src.managers.model_manager import ModelManager
from src.ml_trainer import ML_Trainer
from src.ml_tester import ML_Tester
from src.backtester import Backtester


def main():
    """Run complete ML workflow for BTCUSDT."""
    
    print("\n" + "="*80)
    print("BTCUSDT ML FRAMEWORK - COMPLETE WORKFLOW")
    print("="*80)
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    TICKER = 'BTC-USD'  # Yahoo Finance uses BTC-USD for BTCUSDT
    START_DATE = '2020-01-01'
    END_DATE = '2024-11-24'
    INTERVAL = '1d'  # Daily data
    
    # Target configuration
    FUTURE_BARS = 5  # Predict 5 days ahead
    THRESHOLD = 0.02  # 2% price change threshold
    
    # Model configuration
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Backtesting configuration
    INITIAL_CAPITAL = 10000.0
    POSITION_SIZE = 1.0  # 100% of capital
    COMMISSION = 0.001  # 0.1% commission
    
    print(f"\nConfiguration:")
    print(f"  Ticker: {TICKER}")
    print(f"  Period: {START_DATE} to {END_DATE}")
    print(f"  Interval: {INTERVAL}")
    print(f"  Target: Predict {THRESHOLD*100}% change in {FUTURE_BARS} bars")
    print(f"  Data split: Train={TRAIN_RATIO*100}%, Val={VAL_RATIO*100}%, Test={TEST_RATIO*100}%")
    
    # ========================================================================
    # STEP 1: Download Data
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 1] DOWNLOADING BTCUSDT DATA")
    print("="*80)
    
    data_provider = DataProvider(data_dir='data')
    
    # Download from Yahoo Finance
    df = data_provider.load_yahoo(
        ticker=TICKER,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL
    )
    
    print(f"\n✓ Downloaded {len(df)} rows of data")
    print(f"✓ Date range: {df.index[0]} to {df.index[-1]}")
    print(f"✓ Columns: {list(df.columns)}")
    
    # Validate data
    data_provider.validate_data(df)
    
    # Clean data (remove NaN values)
    df = data_provider.clean_data(df, method='drop')
    
    # Save raw data
    raw_data_path = 'data/btcusdt_raw.csv'
    data_provider.save_data(raw_data_path, df)
    
    # Display basic statistics
    print(f"\nBTCUSDT Price Statistics:")
    print(f"  Open:  ${df['open'].iloc[-1]:,.2f}")
    print(f"  High:  ${df['high'].max():,.2f}")
    print(f"  Low:   ${df['low'].min():,.2f}")
    print(f"  Close: ${df['close'].iloc[-1]:,.2f}")
    print(f"  Volume: {df['volume'].mean():,.0f} (avg)")
    
    # ========================================================================
    # STEP 2: Generate Features
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 2] GENERATING TECHNICAL FEATURES")
    print("="*80)
    
    features_gen = FeaturesGenerator()
    
    # Generate technical indicators
    # Options: 'basic', 'advanced', 'all'
    df_features = features_gen.generate_features(df, feature_set='advanced')
    
    print(f"\n✓ Generated features")
    print(f"✓ Total columns: {len(df_features.columns)}")
    
    # ========================================================================
    # STEP 3: Create Target Variable
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 3] CREATING TARGET VARIABLE")
    print("="*80)
    
    # Create classification target: will price increase by THRESHOLD% in FUTURE_BARS?
    df_features = features_gen.create_target(
        df_features,
        target_type='classification',
        future_bars=FUTURE_BARS,
        threshold=THRESHOLD
    )
    
    # Remove rows with NaN (from feature generation and target creation)
    df_features = df_features.dropna()
    
    print(f"\n✓ Target variable created: 'target'")
    print(f"✓ Dataset ready: {len(df_features)} rows")
    
    # Check class distribution
    target_counts = df_features['target'].value_counts()
    print(f"\nClass Distribution:")
    for label, count in target_counts.items():
        percentage = (count / len(df_features)) * 100
        print(f"  Class {label}: {count} samples ({percentage:.1f}%)")
    
    # ========================================================================
    # STEP 4: Split Data
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 4] SPLITTING DATA")
    print("="*80)
    
    train_df, val_df, test_df = data_provider.split_data(
        df_features,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO
    )
    
    # ========================================================================
    # STEP 5: Setup Models
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 5] SETTING UP ML MODELS")
    print("="*80)
    
    model_manager = ModelManager()
    model_configs = model_manager.get_models()
    
    print(f"\n✓ Available models: {len(model_configs)}")
    for model_name in model_configs.keys():
        print(f"  - {model_name}")
    
    # ========================================================================
    # STEP 6: Train Models
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 6] TRAINING MODELS")
    print("="*80)
    
    trainer = ML_Trainer()
    
    # Get feature columns (exclude OHLCV and target)
    feature_cols = features_gen.get_feature_names()
    
    print(f"\nTraining with {len(feature_cols)} features...")
    print(f"Training samples: {len(train_df)}")
    
    training_results = trainer.train(
        df=train_df,
        target_col='target',
        feature_cols=feature_cols,
        model_configs=model_configs,
        test_size=0.0,  # We already split the data
        scale_features=True,
        use_smote=True  # Handle class imbalance
    )
    
    print(f"\n✓ Training complete!")
    print(f"✓ Best model: {training_results['best_model']}")
    print(f"✓ Best accuracy: {training_results['best_accuracy']:.4f}")
    
    # Display training results
    print("\nTraining Results Summary:")
    print("-" * 60)
    for model_name, metrics in training_results['results'].items():
        print(f"{model_name:25s} | Accuracy: {metrics['accuracy']:.4f} | "
              f"F1: {metrics['f1_score']:.4f}")
    print("-" * 60)
    
    # ========================================================================
    # STEP 7: Test Models on Validation Set
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 7] TESTING MODELS ON VALIDATION SET")
    print("="*80)
    
    tester = ML_Tester()
    
    print(f"\nValidation samples: {len(val_df)}")
    
    val_results = tester.evaluate(
        df=val_df,
        models=training_results['models'],
        scaler=training_results['scaler'],
        target_col='target',
        feature_cols=feature_cols
    )
    
    print("\nValidation Results Summary:")
    print("-" * 60)
    for model_name, metrics in val_results['results'].items():
        print(f"{model_name:25s} | Accuracy: {metrics['accuracy']:.4f} | "
              f"F1: {metrics['f1_score']:.4f}")
    print("-" * 60)
    
    # ========================================================================
    # STEP 8: Test Models on Test Set
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 8] TESTING MODELS ON TEST SET")
    print("="*80)
    
    print(f"\nTest samples: {len(test_df)}")
    
    test_results = tester.evaluate(
        df=test_df,
        models=training_results['models'],
        scaler=training_results['scaler'],
        target_col='target',
        feature_cols=feature_cols
    )
    
    print("\nTest Results Summary:")
    print("-" * 60)
    for model_name, metrics in test_results['results'].items():
        print(f"{model_name:25s} | Accuracy: {metrics['accuracy']:.4f} | "
              f"F1: {metrics['f1_score']:.4f} | Precision: {metrics['precision']:.4f} | "
              f"Recall: {metrics['recall']:.4f}")
    print("-" * 60)
    
    # ========================================================================
    # STEP 9: Save Models
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 9] SAVING MODELS")
    print("="*80)
    
    save_dir = model_manager.save_models(
        models=training_results['models'],
        scaler=training_results['scaler'],
        metadata={
            'ticker': TICKER,
            'start_date': START_DATE,
            'end_date': END_DATE,
            'interval': INTERVAL,
            'best_model': training_results['best_model'],
            'best_accuracy': training_results['best_accuracy'],
            'feature_cols': feature_cols,
            'future_bars': FUTURE_BARS,
            'threshold': THRESHOLD,
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df)
        }
    )
    
    print(f"\n✓ Models saved to: {save_dir}")
    
    # ========================================================================
    # STEP 10: Backtest Best Model
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 10] BACKTESTING BEST MODEL")
    print("="*80)
    
    best_model_name = training_results['best_model']
    best_model = training_results['models'][best_model_name]
    
    print(f"\nBacktesting: {best_model_name}")
    print(f"Initial capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Position size: {POSITION_SIZE*100}%")
    print(f"Commission: {COMMISSION*100}%")
    
    backtester = Backtester(
        initial_capital=INITIAL_CAPITAL,
        position_size=POSITION_SIZE,
        commission=COMMISSION
    )
    
    backtest_results = backtester.run(
        df=test_df,
        model=best_model,
        scaler=training_results['scaler'],
        feature_cols=feature_cols,
        price_col='close'
    )
    
    # Display backtest metrics
    metrics = backtest_results['metrics']
    print("\nBacktest Results:")
    print("-" * 60)
    print(f"Total Return:     {metrics['total_return']*100:>10.2f}%")
    print(f"Annual Return:    {metrics['annual_return']*100:>10.2f}%")
    print(f"Sharpe Ratio:     {metrics['sharpe_ratio']:>10.2f}")
    print(f"Max Drawdown:     {metrics['max_drawdown']*100:>10.2f}%")
    print(f"Win Rate:         {metrics['win_rate']*100:>10.2f}%")
    print(f"Total Trades:     {metrics['total_trades']:>10d}")
    print(f"Final Capital:    ${metrics['final_capital']:>10,.2f}")
    print("-" * 60)
    
    # Plot backtest results
    try:
        backtester.plot_results()
        print("\n✓ Backtest plot saved")
    except Exception as e:
        print(f"\n⚠ Could not create plot: {e}")
    
    # ========================================================================
    # STEP 11: Summary
    # ========================================================================
    print("\n" + "="*80)
    print("WORKFLOW COMPLETE - SUMMARY")
    print("="*80)
    
    print(f"\n📊 Data:")
    print(f"  ✓ Ticker: {TICKER}")
    print(f"  ✓ Total rows: {len(df)}")
    print(f"  ✓ Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  ✓ Features generated: {len(feature_cols)}")
    
    print(f"\n🤖 Models:")
    print(f"  ✓ Models trained: {len(training_results['models'])}")
    print(f"  ✓ Best model: {best_model_name}")
    print(f"  ✓ Training accuracy: {training_results['best_accuracy']:.4f}")
    print(f"  ✓ Test accuracy: {test_results['results'][best_model_name]['accuracy']:.4f}")
    
    print(f"\n💰 Backtest:")
    print(f"  ✓ Total return: {metrics['total_return']*100:.2f}%")
    print(f"  ✓ Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  ✓ Win rate: {metrics['win_rate']*100:.2f}%")
    print(f"  ✓ Total trades: {metrics['total_trades']}")
    
    print(f"\n💾 Saved:")
    print(f"  ✓ Models: {save_dir}")
    print(f"  ✓ Raw data: {raw_data_path}")
    
    print("\n" + "="*80)
    print("🎉 BTCUSDT ML workflow completed successfully!")
    print("="*80 + "\n")
    
    # Return results for further analysis
    return {
        'data': df_features,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'models': training_results['models'],
        'scaler': training_results['scaler'],
        'best_model': best_model_name,
        'training_results': training_results,
        'test_results': test_results,
        'backtest_results': backtest_results,
        'save_dir': save_dir
    }


if __name__ == '__main__':
    # Run the complete workflow
    results = main()
    
    # Optional: Access results for further analysis
    # print("\nYou can now access the results dictionary for further analysis:")
    # print("  - results['data']: Full dataset with features")
    # print("  - results['models']: Trained models")
    # print("  - results['best_model']: Name of best performing model")
    # print("  - results['backtest_results']: Backtesting metrics")
