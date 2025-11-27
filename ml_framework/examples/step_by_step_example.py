"""
Example: Step-by-step usage of individual managers.

This example shows how to use each manager individually for more control.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src import (
    DataProvider,
    FeaturesGenerator,
    ModelManager,
    TrainManager,
    ScalerManager,
    BacktestManager,
    ResultManager,
    VisualizationManager
)

def main():
    """Step-by-step pipeline execution."""
    
    # Create timestamped run directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = Path('results') / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("STEP-BY-STEP PIPELINE EXAMPLE")
    print("="*70)
    
    # Step 1: Load and prepare data
    print("\n[1/8] Loading data...")
    data_provider = DataProvider()
    df = data_provider.load_yahoo('BTC-USD', '2020-01-01', '2023-12-31')
    df = data_provider.clean_data(df)
    
    # Step 2: Generate features
    print("\n[2/8] Generating features...")
    features_gen = FeaturesGenerator()
    df = features_gen.generate_features(df, feature_set='basic')
    df = features_gen.create_target(df, future_bars=5, threshold=0.02)
    
    # Step 3: Split data
    print("\n[3/8] Splitting data...")
    train_df, val_df, test_df = data_provider.split_data(df, train_size=0.7, val_size=0.15)
    feature_cols = features_gen.get_feature_names()
    
    # Step 4: Setup models
    print("\n[4/8] Setting up models...")
    model_manager = ModelManager()
    model_manager.enable_model('logistic_regression', True)
    model_manager.enable_model('random_forest', True)
    model_manager.enable_model('xgboost', True)
    models = model_manager.create_models()
    
    # Step 5: Train models
    print("\n[5/8] Training models...")
    train_manager = TrainManager(use_scaler=True, scaler_type='standard')
    train_output = train_manager.train(
        models=models,
        train_data=train_df,
        target_col='target',
        feature_cols=feature_cols,
        val_data=val_df
    )
    
    # Step 6: Test models
    print("\n[6/8] Testing models...")
    test_results = train_manager.test(
        test_data=test_df,
        target_col='target',
        feature_cols=feature_cols
    )
    
    # Step 7: Backtest models
    print("\n[7/8] Backtesting models...")
    backtest_manager = BacktestManager(backend='nolib', initial_capital=10000)
    
    result_manager = ResultManager()
    result_manager.add_train_results(train_output['results'])
    result_manager.add_test_results(test_results)
    
    for model_name, model in train_output['models'].items():
        backtest_results = backtest_manager.run(
            data=test_df,
            model=model,
            scaler_manager=train_output['scaler_manager'],
            feature_cols=feature_cols
        )
        result_manager.add_backtest_results(model_name, backtest_results)
    
    # Step 8: Generate reports and save
    print("\n[8/8] Generating reports and saving...")
    
    # Save models
    model_manager.save_models(
        models=train_output['models'],
        save_dir=run_dir,
        metadata={'feature_cols': feature_cols, 'target_col': 'target'}
    )
    
    # Save scaler
    if train_output['scaler_manager']:
        train_output['scaler_manager'].save(run_dir)
    
    # Save results
    result_manager.save_results(run_dir)
    
    # Generate visualizations
    viz_manager = VisualizationManager()
    viz_manager.create_train_report(result_manager.train_results, run_dir)
    viz_manager.create_test_report(result_manager.test_results, run_dir)
    viz_manager.create_backtest_report(result_manager.backtest_results, run_dir, test_results=result_manager.test_results)
    
    # Print summary
    result_manager.print_summary()
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {run_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
