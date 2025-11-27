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
Date: 2024-11-26
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
from src.backtesting import BacktestNoLib, BacktestBacktrader, BacktestBacktestingPy
from src.managers.model_manager import ModelManager
from src.managers.train_manager import TrainManager
from src.managers.result_manager import ResultManager
from src.managers.visualization_manager import VisualizationManager
from src.managers.feature_selector import FeatureSelector
from src.managers.run_manager import RunManager
from src.managers.scaler_manager import ScalerManager
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report


def main():
    """Run complete ML pipeline with all models."""
    
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
    
    print(f"\n✓ Data loaded: {len(df)} rows")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  Columns: {list(df.columns)}")
    #endregion
    
    #region Step 2: Generate Features
    # ========================================================================
    # STEP 2: GENERATE FEATURES
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 2] GENERATING FEATURES")
    print("="*80)
    
    features_gen = FeaturesGenerator()
    df_features = features_gen.generate_features(df, feature_set='advanced')
    df_features = features_gen.create_target(
        df_features,
        target_type='classification',
        future_bars=FUTURE_BARS,
        threshold=THRESHOLD,
        num_classes=NUM_CLASSES
    )
    df_features = df_features.dropna()
    
    feature_cols = features_gen.get_feature_names()
    
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
    
    print(f"\n✓ Data split (temporal):")
    print(f"  Train: {len(train_df):5d} samples ({train_df.index[0]} to {train_df.index[-1]})")
    print(f"  Val:   {len(val_df):5d} samples ({val_df.index[0]} to {val_df.index[-1]})")
    print(f"  Test:  {len(test_df):5d} samples ({test_df.index[0]} to {test_df.index[-1]})")
    #endregion
    
    #region Step 4: Initialize Run Manager
    # ========================================================================
    # STEP 4: INITIALIZE RUN MANAGER
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 4] INITIALIZING RUN MANAGER")
    print("="*80)
    
    run_manager = RunManager(base_dir='results')
    run_manager.initialize()
    run_dir = run_manager.get_run_dir()
    
    print(f"\n✓ Run directory created: {run_dir}")
    #endregion
    
    #region Step 5: Feature Importance Analysis
    # ========================================================================
    # STEP 5: FEATURE IMPORTANCE ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 5] FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    feature_selector = FeatureSelector(method=FEATURE_SELECTION_METHOD)
    X_train_df = train_df[feature_cols]
    y_train_series = train_df['target']
    
    feature_selector.fit(X_train_df, y_train_series)
    feature_importance = feature_selector.get_feature_importance()
    selected_features = feature_selector.get_selected_features()
    dropped_features = feature_selector.get_dropped_features()
    
    # Generate feature importance report
    viz_manager = VisualizationManager()
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
    
    feature_selector.print_summary()
    #endregion
    
    #region Step 6: Scale Features
    # ========================================================================
    # STEP 6: SCALE FEATURES
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 6] SCALING FEATURES")
    print("="*80)
    
    scaler_manager = ScalerManager(scaler_type='standard')
    X_train_scaled = scaler_manager.fit_transform(X_train)
    X_val_scaled = scaler_manager.transform(X_val)
    X_test_scaled = scaler_manager.transform(X_test)
    
    # Save scaler
    run_manager.save_scaler(scaler_manager)
    
    print(f"\n✓ Features scaled using StandardScaler")
    print(f"  Train mean: {X_train_scaled.mean():.6f}")
    print(f"  Train std:  {X_train_scaled.std():.6f}")
    #endregion
    
    #region Step 7: Create All Models Using ModelManager
    # ========================================================================
    # STEP 7: CREATE ALL MODELS USING MODELMANAGER
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 7] CREATING ALL MODELS USING MODELMANAGER")
    print("="*80)
    
    # Initialize ModelManager
    model_manager = ModelManager(results_dir='results', use_gpu=False)
    
    # Enable ALL available models
    model_manager.enable_model('logistic_regression', True)
    model_manager.enable_model('random_forest', True)
    model_manager.enable_model('xgboost', True)
    model_manager.enable_model('catboost', True)
    model_manager.enable_model('linear_regression', True)
    
    # Print configuration
    model_manager.print_config()
    
    # Create all enabled models
    models = {}
    
    print("\n" + "="*70)
    print("CREATING TRADITIONAL ML MODELS")
    print("="*70)
    
    # 1. Logistic Regression
    print("\n  1. Creating Logistic Regression...")
    try:
        models['LogisticRegression'] = model_manager.create_model('logistic_regression')
        print("     ✓ Success")
    except Exception as e:
        print(f"     ✗ Failed: {e}")
    
    # 2. Linear Regression
    print("  2. Creating Linear Regression...")
    try:
        models['LinearRegression'] = model_manager.create_model('linear_regression')
        print("     ✓ Success")
    except Exception as e:
        print(f"     ✗ Failed: {e}")
    
    # 3. Random Forest
    print("  3. Creating Random Forest...")
    try:
        models['RandomForest'] = model_manager.create_model('random_forest', n_jobs=-1)
        print("     ✓ Success")
    except Exception as e:
        print(f"     ✗ Failed: {e}")
    
    # 4. XGBoost
    print("  4. Creating XGBoost...")
    try:
        models['XGBoost'] = model_manager.create_model('xgboost')
        print("     ✓ Success")
    except Exception as e:
        print(f"     ✗ Failed: {e}")
    
    # 5. CatBoost
    print("  5. Creating CatBoost...")
    try:
        models['CatBoost'] = model_manager.create_model('catboost')
        print("     ✓ Success")
    except Exception as e:
        print(f"     ✗ Failed: {e}")
    
    print("\n" + "="*70)
    print("CREATING CNN MODELS (3 ARCHITECTURES)")
    print("="*70)
    
    # 6. CNN Small
    print("\n  6. Creating CNN_Small (predefined: simple_cnn_small)...")
    try:
        models['CNN_Small'] = model_manager.create_from_predefined(
            architecture_name='simple_cnn_small',
            name='CNN_Small',
            input_shape=(len(feature_cols), 1),
            num_classes=NUM_CLASSES,
            learning_rate=0.001
        )
        print("     ✓ Success - 2 conv layers [32, 64], 1 dense [64]")
    except Exception as e:
        print(f"     ✗ Failed: {e}")
    
    # 7. CNN Medium
    print("  7. Creating CNN_Medium (predefined: simple_cnn_medium)...")
    try:
        models['CNN_Medium'] = model_manager.create_from_predefined(
            architecture_name='simple_cnn_medium',
            name='CNN_Medium',
            input_shape=(len(feature_cols), 1),
            num_classes=NUM_CLASSES,
            learning_rate=0.001
        )
        print("     ✓ Success - 3 conv layers [64, 128, 256], 2 dense [128, 64]")
    except Exception as e:
        print(f"     ✗ Failed: {e}")
    
    # 8. CNN Large
    print("  8. Creating CNN_Large (predefined: simple_cnn_large)...")
    try:
        models['CNN_Large'] = model_manager.create_from_predefined(
            architecture_name='simple_cnn_large',
            name='CNN_Large',
            input_shape=(len(feature_cols), 1),
            num_classes=NUM_CLASSES,
            learning_rate=0.0005  # Lower LR for larger model
        )
        print("     ✓ Success - 4 conv layers [128, 256, 512, 512], 2 dense [256, 128]")
    except Exception as e:
        print(f"     ✗ Failed: {e}")
    
    print("\n" + "="*70)
    print("CREATING LSTM MODELS (3 ARCHITECTURES)")
    print("="*70)
    
    # 9. LSTM Small
    print("\n  9. Creating LSTM_Small (predefined: lstm_small)...")
    try:
        models['LSTM_Small'] = model_manager.create_from_predefined(
            architecture_name='lstm_small',
            name='LSTM_Small',
            input_shape=(len(feature_cols), 1),
            num_classes=NUM_CLASSES,
            learning_rate=0.001
        )
        print("     ✓ Success - 1 LSTM layer [64], 1 dense [32]")
    except Exception as e:
        print(f"     ✗ Failed: {e}")
    
    # 10. LSTM Medium
    print("  10. Creating LSTM_Medium (predefined: lstm_medium)...")
    try:
        models['LSTM_Medium'] = model_manager.create_from_predefined(
            architecture_name='lstm_medium',
            name='LSTM_Medium',
            input_shape=(len(feature_cols), 1),
            num_classes=NUM_CLASSES,
            learning_rate=0.001
        )
        print("     ✓ Success - 2 LSTM layers [128, 64], 2 dense [64, 32]")
    except Exception as e:
        print(f"     ✗ Failed: {e}")
    
    # 11. LSTM Large
    print("  11. Creating LSTM_Large (predefined: lstm_large)...")
    try:
        models['LSTM_Large'] = model_manager.create_from_predefined(
            architecture_name='lstm_large',
            name='LSTM_Large',
            input_shape=(len(feature_cols), 1),
            num_classes=NUM_CLASSES,
            learning_rate=0.0005  # Lower LR for larger model
        )
        print("     ✓ Success - 3 LSTM layers [256, 128, 64], 2 dense [128, 64]")
    except Exception as e:
        print(f"     ✗ Failed: {e}")
    
    print("\n" + "="*70)
    print("CREATING HYBRID MODEL")
    print("="*70)
    
    # 12. Hybrid CNN-LSTM
    print("\n  12. Creating Hybrid_CNN_LSTM (predefined: hybrid_cnn_lstm)...")
    try:
        models['Hybrid_CNN_LSTM'] = model_manager.create_from_predefined(
            architecture_name='hybrid_cnn_lstm',
            name='Hybrid_CNN_LSTM',
            input_shape=(len(feature_cols), 1),
            num_classes=NUM_CLASSES,
            learning_rate=0.001
        )
        print("     ✓ Success - 2 conv [64, 128] + 1 LSTM [64], 2 dense [64, 32]")
    except Exception as e:
        print(f"     ✗ Failed: {e}")
    
    print("\n" + "="*70)
    print(f"SUMMARY: Created {len(models)} models successfully")
    print("="*70)
    print(f"\nModels created:")
    for i, name in enumerate(models.keys(), 1):
        print(f"  {i:2d}. {name}")
    
    print(f"\n\nAll available predefined architectures:")
    for arch_name in model_manager.get_predefined_architectures():
        print(f"  - {arch_name}")
    #endregion
    
    #region Step 8: Train All Models
    # ========================================================================
    # STEP 8: TRAIN ALL MODELS
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 8] TRAINING ALL MODELS")
    print("="*80)
    
    training_results = {}
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            model.fit(X_train_scaled, y_train)
            training_time = time.time() - start_time
            
            # Training accuracy
            y_train_pred = model.predict(X_train_scaled)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            
            # Validation accuracy
            y_val_pred = model.predict(X_val_scaled)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred, average='weighted')
            val_precision = precision_score(y_val, y_val_pred, average='weighted')
            val_recall = recall_score(y_val, y_val_pred, average='weighted')
            
            training_results[name] = {
                'status': 'success',
                'training_time': training_time,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'val_f1': val_f1,
                'val_precision': val_precision,
                'val_recall': val_recall
            }
            
            print(f"  ✓ Training complete in {training_time:.2f}s")
            print(f"  Training Accuracy:   {train_accuracy:.4f}")
            print(f"  Validation Accuracy: {val_accuracy:.4f}")
            print(f"  Validation F1:       {val_f1:.4f}")
            print(f"  Validation Precision: {val_precision:.4f}")
            print(f"  Validation Recall:    {val_recall:.4f}")
            
        except Exception as e:
            training_results[name] = {
                'status': 'failed',
                'error': str(e)
            }
            print(f"  ✗ Training failed: {e}")
    
    # Summary
    successful_models = {k: v for k, v in training_results.items() if v['status'] == 'success'}
    print(f"\n✓ Successfully trained {len(successful_models)}/{len(models)} models")
    #endregion
    
    #region Step 9: Evaluate All Models on Test Set
    # ========================================================================
    # STEP 9: EVALUATE ALL MODELS ON TEST SET
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 9] EVALUATING ALL MODELS ON TEST SET")
    print("="*80)
    
    test_results = {}
    
    for name, model in models.items():
        if training_results[name]['status'] != 'success':
            continue
            
        print(f"\n{'='*60}")
        print(f"Evaluating {name}...")
        print(f"{'='*60}")
        
        try:
            y_pred = model.predict(X_test_scaled)
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            
            test_results[name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'predictions': y_pred
            }
            
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  F1 Score:  {f1:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            
            # Classification report
            print(f"\n  Classification Report:")
            report = classification_report(y_test, y_pred, target_names=['Down', 'Neutral', 'Up'])
            for line in report.split('\n'):
                print(f"    {line}")
                
        except Exception as e:
            print(f"  ✗ Evaluation failed: {e}")
    
    # Comparison table
    print(f"\n{'='*80}")
    print("MODEL COMPARISON - TEST SET")
    print(f"{'='*80}")
    
    comparison_df = pd.DataFrame(test_results).T
    comparison_df = comparison_df[['accuracy', 'f1_score', 'precision', 'recall']]
    comparison_df = comparison_df.sort_values('accuracy', ascending=False)
    print(comparison_df.to_string())
    
    best_model_name = comparison_df.index[0]
    print(f"\n🏆 Best model: {best_model_name} (Accuracy: {comparison_df.loc[best_model_name, 'accuracy']:.4f})")
    #endregion
    
    #region Step 10: Save All Models and Artifacts
    # ========================================================================
    # STEP 10: SAVE ALL MODELS AND ARTIFACTS
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 10] SAVING ALL MODELS AND ARTIFACTS")
    print("="*80)
    
    # Prepare metrics for saving
    model_metrics = {}
    for name in models.keys():
        if name in test_results:
            model_metrics[name] = {
                'accuracy': test_results[name]['accuracy'],
                'f1_score': test_results[name]['f1_score'],
                'precision': test_results[name]['precision'],
                'recall': test_results[name]['recall'],
                'training_time': training_results[name].get('training_time', 0),
                'val_accuracy': training_results[name].get('val_accuracy', 0),
                'val_f1': training_results[name].get('val_f1', 0)
            }
    
    # Save models
    trained_models = {k: v for k, v in models.items() if training_results[k]['status'] == 'success'}
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
    print("\n" + "="*80)
    print("[STEP 11] LOADING MODELS AND RUNNING BACKTESTS")
    print("="*80)
    
    # Load models from saved folder
    print(f"\nLoading models from: {run_dir}")
    
    loaded_models = run_manager.load_models()
    loaded_scaler = run_manager.load_scaler()  # Returns sklearn scaler object
    
    print(f"✓ Loaded {len(loaded_models)} models: {list(loaded_models.keys())}")
    print(f"✓ Loaded scaler: {type(loaded_scaler).__name__}")
    
    # Prepare backtest data
    backtest_df = df_features.copy()
    
    # Ensure OHLC columns are present
    ohlc_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in ohlc_cols:
        if col not in backtest_df.columns and col in df.columns:
            backtest_df[col] = df[col]
    
    # Run backtests for each model
    backtest_results = {}
    backtester = BacktestNoLib

    for model_name, model in loaded_models.items():
        print(f"\n{'='*60}")
        print(f"Backtesting {model_name}...")
        print(f"{'='*60}")
        
        try:
            backtest = backtester(
                initial_capital=INITIAL_CAPITAL,
                commission=COMMISSION,
                position_size=POSITION_SIZE,
                bars_to_hold=BARS_TO_HOLD
            )
            
            start_time = time.time()
            results = backtest.run(
                df=backtest_df,
                model=model,
                scaler=loaded_scaler,  # Already the sklearn scaler object
                feature_cols=feature_cols,
                price_col='close'
            )
            elapsed_time = time.time() - start_time
            
            backtest.print_results()
            
            backtest_results[model_name] = {
                'backtest': backtest,
                'results': results,
                'metrics': backtest.get_metrics(),
                'trades': backtest.get_trades(),
                'execution_time': elapsed_time,
                'success': True
            }
            
            print(f"Execution time: {elapsed_time:.2f}s")
            
        except Exception as e:
            print(f"  ✗ Backtest failed: {e}")
            backtest_results[model_name] = {
                'success': False,
                'error': str(e)
            }
    
    # Backtest comparison
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
        print(bt_comparison_df.to_string())
        
        best_backtest = bt_comparison_df.index[0]
        print(f"\n🏆 Best backtest: {best_backtest} (Return: {bt_comparison_df.loc[best_backtest, 'Total Return (%)']:.2f}%)")
    #endregion
    
    #region Step 12: Generate Comprehensive Visualizations
    # ========================================================================
    # STEP 12: GENERATE COMPREHENSIVE VISUALIZATIONS
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 12] GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("="*80)
    
    results_manager = ResultManager()
    
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
    
    print(f"\n✓ Backtest report generated: {report_path}")
    #endregion
    
    #region Step 13: Final Summary
    # ========================================================================
    # STEP 13: FINAL SUMMARY
    # ========================================================================
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
