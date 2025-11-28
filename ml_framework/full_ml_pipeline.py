"""
Full ML Pipeline - Complete Workflow

This script demonstrates the complete ML workflow:
1. Load and prepare data
2. Generate features
3. Feature importance analysis
4. Train ALL available models
5. Evaluate models on test set
6. Save all artifacts to timestamped folder
7. Load saved models and run backtests
8. Generate comprehensive visualizations

Author: ML Framework
Date: 2025-11-26
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_provider import DataProvider
from src.features_generator import FeaturesGenerator
from src.models_lib import (
    XGBoostModel, CatBoostModel,
    LogisticRegressionModel, RandomForestModel
)
from src.backtesting import (
    BacktestNoLib, 
    BacktestBacktrader, 
    BacktestBacktestingPy
)
from src.managers import (
    ModelManager,
    TrainManager,
    TestManager,
    ResultManager,
    VisualizationManager,
    FeatureSelector,
    RunManager,
    ScalerManager
)


def main(verbose: bool = True):
    """
    Run complete ML pipeline with all models.
    
    Args:
        verbose: If True, print detailed progress (default: True)
    """
    
    if verbose:
        print("\n" + "="*80)
        print("FULL ML PIPELINE - ALL MODELS")
        print("="*80)
    
    #region Configuration
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    # Data configuration
    TICKER = 'BTC-USD'
    START_DATE = '2020-01-01'
    END_DATE = '2024-11-24'
    INTERVAL = '1d'
    
    # Target configuration
    FUTURE_BARS = 15
    THRESHOLD = 0.1
    NUM_CLASSES = 3  # -1 (down), 0 (neutral), +1 (up)
    
    # Backtest configuration
    INITIAL_CAPITAL = 10000.0
    COMMISSION = 0.001
    POSITION_SIZE = 0.02
    BARS_TO_HOLD = FUTURE_BARS
    
    # Feature selection
    FEATURE_SELECTION_METHOD = 'tree'
    MIN_FEATURES = 5
    
    if verbose:
        print(f"\nConfiguration:")
        print(f"  Ticker: {TICKER}")
        print(f"  Period: {START_DATE} to {END_DATE}")
        print(f"  Interval: {INTERVAL}")
        print(f"  Target: {NUM_CLASSES}-class classification")
        print(f"  Future bars: {FUTURE_BARS}, Threshold: {THRESHOLD*100}%")
        print(f"  Initial Capital: ${INITIAL_CAPITAL:,.2f}")
        print(f"  Commission: {COMMISSION*100}%")
        print(f"  Position Size: {POSITION_SIZE*100}%")
    #endregion
    
    #region Step 1: Load Data
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("[STEP 1] LOADING DATA")
        print("="*80)
    
    data_provider = DataProvider(data_dir='data')
    df = data_provider.load_yahoo(
        ticker=TICKER,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL,
        use_cache=True
    )
    
    if verbose:
        print(f"\n✓ Data loaded: {len(df)} rows")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        print(f"  Columns: {list(df.columns)}")
    #endregion
    
    #region Step 2: Generate Features
    # ========================================================================
    # STEP 2: GENERATE FEATURES
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("[STEP 2] GENERATING FEATURES")
        print("="*80)
    
    features_gen = FeaturesGenerator()
    #df_features = features_gen.generate_features(df, feature_set='advanced')
    df_features = features_gen.generate_features(df, feature_set='all')
    df_features = features_gen.create_target(
        df_features,
        target_type='classification',
        future_bars=FUTURE_BARS,
        threshold=THRESHOLD,
        num_classes=NUM_CLASSES
    )
    df_features = df_features.dropna()
    
    feature_cols = features_gen.get_feature_names()
    
    if verbose:
        print(f"\n✓ Features generated: {len(feature_cols)} features")
        print(f"✓ Dataset ready: {len(df_features)} rows")
        print(f"\nFeature list:")
        for i, col in enumerate(feature_cols, 1):
            print(f"  {i:2d}. {col}")
        
        # Target distribution
        print(f"\nTarget distribution:")
        target_counts = df_features['target'].value_counts().sort_index()
        for label, count in target_counts.items():
            pct = count / len(df_features) * 100
            label_name = {-1: 'Down', 0: 'Neutral', 1: 'Up'}.get(label, str(label))
            print(f"  {label_name:8s} ({label:+d}): {count:5d} ({pct:5.1f}%)")
    #endregion
    
    #region Step 3: Split Data
    # ========================================================================
    # STEP 3: SPLIT DATA (Temporal)
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("[STEP 3] SPLITTING DATA (TEMPORAL)")
        print("="*80)
    
    train_df, val_df, test_df = data_provider.split_data(
        df_features,
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['target'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values
    
    if verbose:
        print(f"\n✓ Data split (temporal):")
        print(f"  Train: {len(train_df):5d} samples ({train_df.index[0]} to {train_df.index[-1]})")
        print(f"  Val:   {len(val_df):5d} samples ({val_df.index[0]} to {val_df.index[-1]})")
        print(f"  Test:  {len(test_df):5d} samples ({test_df.index[0]} to {test_df.index[-1]})")
    #endregion
    
    #region Step 4: Initialize Run Manager
    # ========================================================================
    # STEP 4: INITIALIZE RUN MANAGER
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("[STEP 4] INITIALIZING RUN MANAGER")
        print("="*80)
    
    run_manager = RunManager(base_dir='results', verbose=verbose)
    run_manager.initialize()
    run_dir = run_manager.get_run_dir()
    
    if verbose:
        print(f"\n✓ Run directory created: {run_dir}")
    #endregion
    
    #region Step 5: Feature Importance Analysis
    # ========================================================================
    # STEP 5: FEATURE IMPORTANCE ANALYSIS
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("[STEP 5] FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
    
    feature_selector = FeatureSelector(method=FEATURE_SELECTION_METHOD, verbose=verbose)
    X_train_df = train_df[feature_cols]
    y_train_series = train_df['target']
    
    feature_selector.fit(X_train_df, y_train_series)
    feature_importance = feature_selector.get_feature_importance()
    selected_features = feature_selector.get_selected_features()
    dropped_features = feature_selector.get_dropped_features()
    
    # Generate feature importance report
    viz_manager = VisualizationManager(verbose=verbose)
    correlation_matrix = X_train_df.corr()
    
    viz_manager.create_feature_importance_report(
        feature_importance=feature_importance,
        save_dir=run_dir,
        selected_features=selected_features,
        dropped_features=dropped_features,
        method=FEATURE_SELECTION_METHOD,
        correlation_matrix=correlation_matrix,
        show=False
    )
    
    # Save feature importance
    run_manager.save_feature_importance(
        feature_importance=feature_importance,
        selected_features=selected_features,
        dropped_features=dropped_features,
        method=FEATURE_SELECTION_METHOD
    )
    
    if verbose:
        feature_selector.print_summary()
    #endregion
    
    #region Step 6: Scale Features
    # ========================================================================
    # STEP 6: SCALE FEATURES
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("[STEP 6] SCALING FEATURES")
        print("="*80)
    
    scaler_manager = ScalerManager(scaler_type='standard')
    X_train_scaled = scaler_manager.fit_transform(X_train)
    X_val_scaled = scaler_manager.transform(X_val)
    X_test_scaled = scaler_manager.transform(X_test)
    
    # Save scaler
    run_manager.save_scaler(scaler_manager)
    
    if verbose:
        print(f"\n✓ Features scaled using StandardScaler")
        print(f"  Train mean: {X_train_scaled.mean():.6f}")
        print(f"  Train std:  {X_train_scaled.std():.6f}")
    #endregion
    
    #region Step 7: Create All Models Using ModelManager
    # ========================================================================
    # STEP 7: CREATE ALL MODELS USING MODELMANAGER
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("[STEP 7] CREATING ALL MODELS USING MODELMANAGER")
        print("="*80)
    
    # Initialize ModelManager
    model_manager = ModelManager(results_dir='results', use_gpu=False, verbose=verbose)
    
    # Add traditional ML models
    if verbose:
        print("\nAdding traditional ML models...")
    
    model_manager.add_model('logistic_regression', custom_name='LogisticRegression')
    model_manager.add_model('linear_regression', custom_name='LinearRegression')
    model_manager.add_model('random_forest', custom_name='RandomForest', n_jobs=-1)
    model_manager.add_model('xgboost', custom_name='XGBoost')
    
    # CatBoost (may fail if not installed)
    try:
        model_manager.add_model('catboost', custom_name='CatBoost')
    except Exception as e:
        if verbose:
            print(f"  ✗ CatBoost not available: {e}")
    
    # Add CNN models (predefined architectures)
    if verbose:
        print("\nAdding CNN models...")
    
    input_shape = (len(feature_cols), 1)
    
    try:
        model_manager.add_predefined_model('simple_cnn_small', 'CNN_Small', input_shape, NUM_CLASSES, 0.001)
    except Exception as e:
        if verbose:
            print(f"  ✗ CNN_Small failed: {e}")
    
    try:
        model_manager.add_predefined_model('simple_cnn_medium', 'CNN_Medium', input_shape, NUM_CLASSES, 0.001)
    except Exception as e:
        if verbose:
            print(f"  ✗ CNN_Medium failed: {e}")
    
    try:
        model_manager.add_predefined_model('simple_cnn_large', 'CNN_Large', input_shape, NUM_CLASSES, 0.0005)
    except Exception as e:
        if verbose:
            print(f"  ✗ CNN_Large failed: {e}")
    
    # Add LSTM models (predefined architectures)
    if verbose:
        print("\nAdding LSTM models...")
    
    try:
        model_manager.add_predefined_model('lstm_small', 'LSTM_Small', input_shape, NUM_CLASSES, 0.001)
    except Exception as e:
        if verbose:
            print(f"  ✗ LSTM_Small failed: {e}")
    
    try:
        model_manager.add_predefined_model('lstm_medium', 'LSTM_Medium', input_shape, NUM_CLASSES, 0.001)
    except Exception as e:
        if verbose:
            print(f"  ✗ LSTM_Medium failed: {e}")
    
    try:
        model_manager.add_predefined_model('lstm_large', 'LSTM_Large', input_shape, NUM_CLASSES, 0.0005)
    except Exception as e:
        if verbose:
            print(f"  ✗ LSTM_Large failed: {e}")
    
    # Add Hybrid CNN-LSTM model
    if verbose:
        print("\nAdding Hybrid model...")
    
    try:
        model_manager.add_predefined_model('hybrid_cnn_lstm', 'Hybrid_CNN_LSTM', input_shape, NUM_CLASSES, 0.001)
    except Exception as e:
        if verbose:
            print(f"  ✗ Hybrid_CNN_LSTM failed: {e}")
    
    # Get all models
    models = model_manager.get_models()
    
    if verbose:
        print("\n" + "="*70)
        print(f"SUMMARY: Created {len(models)} models successfully")
        print("="*70)
        for i, name in enumerate(models.keys(), 1):
            print(f"  {i:2d}. {name}")
    #endregion
    
    #region Step 8: Train All Models
    # ========================================================================
    # STEP 8: TRAIN ALL MODELS USING TRAINMANAGER
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("[STEP 8] TRAINING ALL MODELS USING TRAINMANAGER")
        print("="*80)
    
    # Initialize TrainManager (scaler already handled in Step 6)
    train_manager = TrainManager(use_scaler=False, verbose=verbose)
    
    # Prepare training data with scaled features
    train_df_scaled = train_df.copy()
    train_df_scaled[feature_cols] = X_train_scaled
    
    val_df_scaled = val_df.copy()
    val_df_scaled[feature_cols] = X_val_scaled
    
    # Train all models using TrainManager
    train_output = train_manager.train(
        models=models,
        train_data=train_df_scaled,
        target_col='target',
        feature_cols=feature_cols,
        val_data=val_df_scaled,
        scale_features=False  # Already scaled
    )
    
    # Get results
    training_results = train_manager.get_train_results()
    trained_models = train_manager.get_trained_models()
    
    # Summary
    successful_models = {k: v for k, v in training_results.items() if v.get('status') == 'success'}
    if verbose:
        print(f"\n✓ Successfully trained {len(successful_models)}/{len(models)} models")
    #endregion
    
    #region Step 9: Evaluate All Models on Test Set
    # ========================================================================
    # STEP 9: EVALUATE ALL MODELS ON TEST SET USING TESTMANAGER
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("[STEP 9] EVALUATING ALL MODELS ON TEST SET USING TESTMANAGER")
        print("="*80)
    
    # Initialize TestManager
    test_manager = TestManager(verbose=verbose)
    
    # Filter to only successfully trained models
    models_to_test = {k: v for k, v in trained_models.items() 
                      if training_results.get(k, {}).get('status') == 'success'}
    
    # Prepare test data with scaled features
    test_df_scaled = test_df.copy()
    test_df_scaled[feature_cols] = X_test_scaled
    
    # Test all models using TestManager
    test_results = test_manager.test(
        models=models_to_test,
        test_data=test_df_scaled,
        target_col='target',
        feature_cols=feature_cols,
        X_test_scaled=X_test_scaled,
        y_test=y_test
    )
    
    # Get comparison DataFrame and best model
    comparison_df = test_manager.get_comparison_dataframe()
    best_model_name = test_manager.get_best_model(metric='accuracy')
    
    if verbose:
        print(f"\n{'='*80}")
        print("MODEL COMPARISON - TEST SET")
        print(f"{'='*80}")
        print(comparison_df.to_string())
        print(f"\n🏆 Best model: {best_model_name} (Accuracy: {comparison_df.loc[best_model_name, 'accuracy']:.4f})")
    #endregion
    
    #region Step 10: Save All Models and Artifacts
    # ========================================================================
    # STEP 10: SAVE ALL MODELS AND ARTIFACTS
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("[STEP 10] SAVING ALL MODELS AND ARTIFACTS")
        print("="*80)
    
    # Prepare metrics for saving
    model_metrics = {}
    for name in trained_models.keys():
        if name in test_results and test_results[name].get('status') == 'success':
            model_metrics[name] = {
                'accuracy': test_results[name]['accuracy'],
                'f1_score': test_results[name]['f1_score'],
                'precision': test_results[name]['precision'],
                'recall': test_results[name]['recall'],
                'training_time': training_results[name].get('training_time', 0),
                'val_accuracy': training_results[name].get('val_accuracy', 0)
            }
    
    # Save models (trained_models already from train_manager)
    run_manager.save_models(models=trained_models, metrics=model_metrics)
    
    # Save datasets
    run_manager.save_datasets(
        X_train=train_df[feature_cols], y_train=train_df['target'],
        X_val=val_df[feature_cols], y_val=val_df['target'],
        X_test=test_df[feature_cols], y_test=test_df['target'],
        df_full=df_features
    )
    
    # Save configuration
    run_manager.save_config({
        'ticker': TICKER,
        'start_date': START_DATE,
        'end_date': END_DATE,
        'interval': INTERVAL,
        'future_bars': FUTURE_BARS,
        'threshold': THRESHOLD,
        'num_classes': NUM_CLASSES,
        'initial_capital': INITIAL_CAPITAL,
        'commission': COMMISSION,
        'position_size': POSITION_SIZE,
        'bars_to_hold': BARS_TO_HOLD,
        'feature_selection_method': FEATURE_SELECTION_METHOD,
        'feature_count': len(feature_cols),
        'models_trained': list(trained_models.keys()),
        'best_model': best_model_name
    })
    
    # Save test metrics
    run_manager.save_metrics(test_results)
    
    if verbose:
        print(f"\n✓ All artifacts saved to: {run_dir}")
        print(f"  - Models: {len(trained_models)}")
        print(f"  - Scaler: standard")
        print(f"  - Datasets: train, val, test, full")
        print(f"  - Feature importance report")
        print(f"  - Configuration and metrics")
    #endregion
    
    #region Step 11: Load Models and Run Backtests
    # ========================================================================
    # STEP 11: LOAD MODELS FROM FOLDER AND RUN BACKTESTS
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("[STEP 11] LOADING MODELS AND RUNNING BACKTESTS")
        print("="*80)
    
    # Load models from saved folder
    if verbose:
        print(f"\nLoading models from: {run_dir}")
    
    loaded_models = run_manager.load_models()
    loaded_scaler = run_manager.load_scaler()  # Returns sklearn scaler object
    
    if verbose:
        print(f"✓ Loaded {len(loaded_models)} models: {list(loaded_models.keys())}")
        print(f"✓ Loaded scaler: {type(loaded_scaler).__name__}")
    
    # Prepare backtest data
    backtest_df = df_features.copy()
    
    # Ensure OHLC columns are present
    ohlc_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in ohlc_cols:
        if col not in backtest_df.columns and col in df.columns:
            backtest_df[col] = df[col]
    
    # Run backtests for each model using BacktestNoLib (provides trades for OHLC visualization)
    backtest_results = {}

    for model_name, model in loaded_models.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Backtesting {model_name}...")
            print(f"{'='*60}")
        
        try:
            # Use BacktestNoLib directly to get trades for OHLC visualization
            backtest = BacktestNoLib(
                initial_capital=INITIAL_CAPITAL,
                commission=COMMISSION,
                position_size=POSITION_SIZE,
                bars_to_hold=BARS_TO_HOLD
            )
            
            start_time = time.time()
            results = backtest.run(
                df=backtest_df,
                model=model,
                scaler=loaded_scaler,
                feature_cols=feature_cols,
                price_col='close'
            )
            elapsed_time = time.time() - start_time
            
            if verbose:
                backtest.print_results()
            
            backtest_results[model_name] = {
                'backtest': backtest,
                'results': results,
                'metrics': backtest.get_metrics(),
                'trades': backtest.get_trades(),  # Individual trades for OHLC visualization
                'execution_time': elapsed_time,
                'success': True
            }
            
            if verbose:
                print(f"Execution time: {elapsed_time:.2f}s")
            
        except Exception as e:
            if verbose:
                print(f"  ✗ Backtest failed: {e}")
            backtest_results[model_name] = {
                'success': False,
                'error': str(e)
            }
    
    # Backtest comparison
    if verbose:
        print(f"\n{'='*80}")
        print("BACKTEST COMPARISON - ALL MODELS")
        print(f"{'='*80}")
    
    comparison_data = {}
    for name, result in backtest_results.items():
        if result['success']:
            metrics = result['metrics']
            comparison_data[name] = {
                'Final Capital ($)': metrics.get('final_capital', 0),
                'Total Return (%)': metrics.get('total_return', 0) * 100,
                'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                'Max Drawdown (%)': metrics.get('max_drawdown', 0) * 100,
                'Total Trades': metrics.get('total_trades', 0),
                'Win Rate (%)': metrics.get('win_rate', 0) * 100
            }
    
    if comparison_data:
        bt_comparison_df = pd.DataFrame(comparison_data).T
        bt_comparison_df = bt_comparison_df.sort_values('Total Return (%)', ascending=False)
        if verbose:
            print(bt_comparison_df.to_string())
        
        best_backtest = bt_comparison_df.index[0]
        if verbose:
            print(f"\n🏆 Best backtest: {best_backtest} (Return: {bt_comparison_df.loc[best_backtest, 'Total Return (%)']:.2f}%)")
    #endregion
    
    #region Step 12: Generate Comprehensive Visualizations
    # ========================================================================
    # STEP 12: GENERATE COMPREHENSIVE VISUALIZATIONS
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("[STEP 12] GENERATING COMPREHENSIVE VISUALIZATIONS")
        print("="*80)
    
    results_manager = ResultManager(verbose=verbose)
    
    # Add training results
    for name, result in training_results.items():
        if result['status'] == 'success':
            results_manager.train_results[name] = {
                'status': 'success',
                'train_accuracy': result.get('train_accuracy', 0),  # Will add this
                'val_accuracy': result.get('val_accuracy', 0),
                'training_time': result.get('training_time', 0)
            }
    
    # Add test results
    for name, result in test_results.items():
        results_manager.test_results[name] = {
            'status': 'success',
            'accuracy': result.get('accuracy', 0),
            'precision': result.get('precision', 0),
            'recall': result.get('recall', 0),
            'f1_score': result.get('f1_score', 0)
        }
    
    # Add all backtest results
    for name, result in backtest_results.items():
        if result['success']:
            equity_curve = result['results'].get('equity_curve', [])
            if isinstance(equity_curve, pd.Series):
                equity_curve = equity_curve.tolist()
            elif not isinstance(equity_curve, list):
                equity_curve = list(equity_curve) if hasattr(equity_curve, '__iter__') else []
            
            results_manager.add_backtest_results(
                model_name=name,
                results={
                    'status': 'success',
                    'equity_curve': equity_curve,
                    'trades': result['trades'],
                    'metrics': result['metrics'],
                    'initial_capital': INITIAL_CAPITAL
                }
            )
    
    # Prepare visualization data
    viz_data = results_manager.prepare_backtest_visualization_data(backtest_df)
    
    # Generate backtest report
    report_path = viz_manager.create_backtest_report(
        backtest_results=viz_data,
        save_dir=run_dir,
        df=backtest_df,
        test_results=results_manager.test_results,
        train_results=results_manager.train_results,
        show=True
    )
    
    if verbose:
        print(f"\n✓ Backtest report generated: {report_path}")
    #endregion
    
    #region Step 13: Final Summary
    # ========================================================================
    # STEP 13: FINAL SUMMARY
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("PIPELINE COMPLETE - FINAL SUMMARY")
        print("="*80)
        
        run_manager.print_summary()
        
        print(f"\n{'='*80}")
        print("RESULTS OVERVIEW")
        print(f"{'='*80}")
        
        print(f"\n📊 Data:")
        print(f"  - Total samples: {len(df_features)}")
        print(f"  - Features: {len(feature_cols)}")
        print(f"  - Selected features: {len(selected_features)}")
        
        print(f"\n🤖 Models:")
        print(f"  - Trained: {len(trained_models)}")
        print(f"  - Best (Test): {best_model_name}")
        if comparison_data:
            print(f"  - Best (Backtest): {best_backtest}")
        
        print(f"\n📁 Artifacts saved to: {run_dir}")
        print(f"  - models/")
        print(f"  - scalers/")
        print(f"  - datasets/")
        print(f"  - features/")
        print(f"  - reports/")
        
        print("\n" + "="*80)
        print("🎉 Full ML Pipeline completed successfully!")
        print("="*80 + "\n")
    #endregion
    
    return {
        'run_dir': run_dir,
        'models': trained_models,
        'test_results': test_results,
        'backtest_results': backtest_results,
        'best_model': best_model_name,
        'best_backtest': best_backtest if comparison_data else None
    }


if __name__ == '__main__':
    results = main()
