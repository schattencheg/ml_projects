"""
BTCUSDT Backtest Comparison Example

This example demonstrates:
1. Download BTCUSDT data with smart caching
2. Generate features
3. Train ML models
4. Compare all three backtesting backends:
   - BacktestNoLib (custom implementation)
   - BacktestBacktrader (Backtrader library)
   - BacktestBacktestingPy (backtesting.py library)

Author: ML Framework
Date: 2024-11-24
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_provider import DataProvider
from src.features_generator import FeaturesGenerator
from src.models_lib import LogisticRegressionModel, RandomForestModel
from src.backtesting import BacktestNoLib, BacktestBacktrader, BacktestBacktestingPy
from src.managers.result_manager import ResultManager
from src.managers.visualization_manager import VisualizationManager
from src.managers.feature_selector import FeatureSelector
from src.managers.run_manager import RunManager
from src.managers.scaler_manager import ScalerManager


def main():
    """Run complete ML workflow with backtest comparison."""
    
    print("\n" + "="*80)
    print("BTCUSDT ML FRAMEWORK - BACKTEST COMPARISON")
    print("="*80)
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    TICKER = 'BTC-USD'
    START_DATE = '2020-01-01'
    END_DATE = '2024-11-24'
    INTERVAL = '1d'
    
    FUTURE_BARS = 15
    THRESHOLD = 0.05
    
    # Backtest configuration
    INITIAL_CAPITAL = 10000.0
    COMMISSION = 0.001
    POSITION_SIZE = 0.02  # 2% of capital per trade
    BARS_TO_HOLD = FUTURE_BARS  # Exit after N bars (same as prediction horizon)
    
    print(f"\nConfiguration:")
    print(f"  Ticker: {TICKER}")
    print(f"  Period: {START_DATE} to {END_DATE}")
    print(f"  Target: Predict {THRESHOLD*100}% change in {FUTURE_BARS} bars")
    print(f"  Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"  Commission: {COMMISSION*100}%")
    print(f"  Position Size: {POSITION_SIZE*100}% of capital")
    print(f"  Bars to Hold: {BARS_TO_HOLD}")
    
    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    #region 'Load Data'
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
    #endregion

    # ========================================================================
    # STEP 2: Generate Features
    # ========================================================================
    #region 'Generate Features'

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
        num_classes=3  # Three-class: -1 (decrease), 0 (neutral), +1 (increase)
    )
    df_features = df_features.dropna()
    
    print(f"\n✓ Features generated: {len(df_features.columns)} columns")
    print(f"✓ Dataset ready: {len(df_features)} rows")
    #endregion
    
    # ========================================================================
    # STEP 3: Split Data
    # ========================================================================
    #region 'Split Data'
    print("\n" + "="*80)
    print("[STEP 3] SPLITTING DATA")
    print("="*80)
    
    train_df, val_df, test_df = data_provider.split_data(
        df_features,
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    feature_cols = features_gen.get_feature_names()
    
    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values
    
    print(f"\n✓ Train: {len(train_df)} samples")
    print(f"✓ Val: {len(val_df)} samples")
    print(f"✓ Test: {len(test_df)} samples")
    #endregion
    
    # ========================================================================
    # STEP 4: Feature Importance Analysis (BEFORE training)
    # ========================================================================
    #region 'Feature Importance'
    print("\n" + "="*80)
    print("[STEP 4] FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    # Initialize RunManager for artifact storage
    run_manager = RunManager(base_dir='models')
    run_manager.initialize()
    
    # Initialize VisualizationManager
    viz_manager = VisualizationManager()
    
    # Analyze feature importance using tree-based method
    feature_selector = FeatureSelector(method='tree')
    X_train_df = train_df[feature_cols]
    y_train_series = train_df['target']
    
    feature_selector.fit(X_train_df, y_train_series)
    feature_importance = feature_selector.get_feature_importance()
    selected_features = feature_selector.get_selected_features()
    dropped_features = feature_selector.get_dropped_features()
    
    # Generate feature importance report
    correlation_matrix = X_train_df.corr()
    viz_manager.create_feature_importance_report(
        feature_importance=feature_importance,
        save_dir=run_manager.get_run_dir(),
        selected_features=selected_features,
        dropped_features=dropped_features,
        method='tree',
        correlation_matrix=correlation_matrix,
        show=False  # Don't open browser yet
    )
    
    # Save feature importance to run
    run_manager.save_feature_importance(
        feature_importance=feature_importance,
        selected_features=selected_features,
        dropped_features=dropped_features,
        method='tree'
    )
    
    feature_selector.print_summary()
    #endregion
    
    # ========================================================================
    # STEP 5: Train Model
    # ========================================================================
    #region 'Train Model'
    print("\n" + "="*80)
    print("[STEP 5] TRAINING MODEL")
    print("="*80)
    
    # Scale features using ScalerManager
    scaler_manager = ScalerManager(scaler_type='standard')
    X_train_scaled = scaler_manager.fit_transform(X_train)
    X_val_scaled = scaler_manager.transform(val_df[feature_cols].values)
    X_test_scaled = scaler_manager.transform(X_test)
    
    # Use Random Forest for better performance
    model = RandomForestModel(
        name='RandomForest',
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    print("\nTraining Random Forest...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    from sklearn.metrics import accuracy_score, f1_score
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n✓ Model trained")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # Save model, scaler, and datasets
    run_manager.save_models(
        models={'RandomForest': model},
        metrics={'RandomForest': {'accuracy': accuracy, 'f1_score': f1}}
    )
    run_manager.save_scaler(scaler_manager)
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
        'initial_capital': INITIAL_CAPITAL,
        'commission': COMMISSION,
        'position_size': POSITION_SIZE,
        'bars_to_hold': BARS_TO_HOLD,
        'model': 'RandomForest',
        'feature_count': len(feature_cols)
    })
    #endregion
    
    # ========================================================================
    # STEP 6: Run Backtests
    # ========================================================================
    #region 'Run Backtests'
    print("\n" + "="*80)
    print("[STEP 6] RUNNING BACKTESTS - COMPARING ALL BACKENDS")
    print("="*80)
    
    # Prepare test data with all required columns
    test_df_bt = test_df.copy()
    
    # Initialize backtesting engines
    # NoLib uses RiskManager for:
    # - Position sizing (2% of current capital)
    # - Exit after exactly N bars
    # - No TP/SL
    # - Cooldown: new positions only on bar after previous closes
    backtests = {
        'NoLib': BacktestNoLib(
            initial_capital=INITIAL_CAPITAL,
            commission=COMMISSION,
            position_size=POSITION_SIZE,  # 2% of capital
            bars_to_hold=BARS_TO_HOLD     # Exit after N bars
        ),
        'Backtrader': BacktestBacktrader(
            initial_capital=INITIAL_CAPITAL,
            commission=COMMISSION,
            position_size=POSITION_SIZE
        ),
        'BacktestingPy': BacktestBacktestingPy(
            initial_capital=INITIAL_CAPITAL,
            commission=COMMISSION,
            position_size=POSITION_SIZE
        )
    }
    
    # Run backtests
    results_manager = ResultManager()
    results_comparison = {}

    for name, backtest in backtests.items():
        print(f"\n{'='*80}")
        print(f"Running {name} Backtest")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            test_df_bt = df_features.copy() # Evaluating backtest on whole period (Train+Test+Val)
            
            # Preserve original OHLC columns for visualization
            # Merge original OHLC data back if not present
            ohlc_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in ohlc_cols:
                if col not in test_df_bt.columns and col in df.columns:
                    test_df_bt[col] = df[col]
            
            results = backtest.run(
                df=test_df_bt,
                model=model,
                scaler=scaler_manager.scaler,
                feature_cols=feature_cols,
                price_col='close'
            )
            
            elapsed_time = time.time() - start_time
            
            # Print results
            backtest.print_results()
            print(f"Execution time: {elapsed_time:.2f} seconds")
            
            # Store for comparison
            results_comparison[name] = {
                'metrics': backtest.get_metrics(),
                'execution_time': elapsed_time,
                'success': True,
                'backtest': backtest  # Store backtest object for visualization
            }

            metrics = backtest.calculate_metrics()
            
            # Get equity curve and convert to list if it's a Series
            equity_curve = results.get('equity_curve', [])
            if isinstance(equity_curve, pd.Series):
                equity_curve = equity_curve.tolist()
            elif not isinstance(equity_curve, list):
                equity_curve = list(equity_curve) if hasattr(equity_curve, '__iter__') else []
            
            # Add to results manager
            results_manager.add_backtest_results(
                model_name=name,
                results={
                    'status': 'success',
                    'equity_curve': equity_curve,
                    'trades': backtest.get_trades(),
                    'metrics': metrics,
                    'initial_capital': INITIAL_CAPITAL
                }
            )
            
            print(f"\n✓ {name} backtest complete")
            print(f"  Total Return: {metrics.get('total_return', 0)*100:.2f}%")
            print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"  Total Trades: {metrics.get('total_trades', 0)}")

            
        except Exception as e:
            print(f"❌ Error running {name}: {e}")
            print(f"   This backend may require additional dependencies")
            results_comparison[name] = {
                'metrics': {},
                'execution_time': 0,
                'success': False,
                'error': str(e)
            }
    #endregion
    
    # ========================================================================
    # STEP 7: Compare Results
    # ========================================================================
    #region 'Compare Results'
    print("\n" + "="*80)
    print("[STEP 7] BACKTEST COMPARISON SUMMARY")
    print("="*80)
    
    # Create comparison table
    comparison_df = pd.DataFrame()
    
    for name, result in results_comparison.items():
        if result['success']:
            metrics = result['metrics']
            comparison_df[name] = pd.Series({
                'Final Capital ($)': metrics.get('final_capital', 0),
                'Total Return (%)': metrics.get('total_return', 0) * 100,
                'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                'Max Drawdown (%)': metrics.get('max_drawdown', 0) * 100,
                'Total Trades': metrics.get('total_trades', 0),
                'Win Rate (%)': metrics.get('win_rate', 0) * 100,
                'Execution Time (s)': result['execution_time']
            })
        else:
            comparison_df[name] = pd.Series({
                'Final Capital ($)': 'N/A',
                'Total Return (%)': 'N/A',
                'Sharpe Ratio': 'N/A',
                'Max Drawdown (%)': 'N/A',
                'Total Trades': 'N/A',
                'Win Rate (%)': 'N/A',
                'Execution Time (s)': 'N/A'
            })
    
    print("\n" + comparison_df.to_string())
    #endregion
    
    # ========================================================================
    # STEP 8: Generate Comprehensive Visualizations
    # ========================================================================
    #region 'Generate Comprehensive Visualizations'
    print("\n" + "="*80)
    print("[STEP 8] GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("="*80)
    
    # Initialize managers
    results_manager = ResultManager()
    viz_manager = VisualizationManager()
    
    # Add backtest results to ResultManager
    for name, result_data in results_comparison.items():
        if result_data['success']:
            print(f"\nAdding {name} results to visualization manager...")
            
            # Get the backtest object from stored results
            backtest = result_data['backtest']
            
            # Get equity curve and convert to list if it's a Series
            equity_curve = backtest.get_results().get('equity_curve', [])
            if isinstance(equity_curve, pd.Series):
                equity_curve = equity_curve.tolist()
            elif not isinstance(equity_curve, list):
                equity_curve = list(equity_curve) if hasattr(equity_curve, '__iter__') else []
            
            results_manager.add_backtest_results(
                model_name=name,
                results={
                    'status': 'success',
                    'equity_curve': equity_curve,
                    'trades': backtest.get_trades(),
                    'metrics': backtest.get_metrics(),
                    'initial_capital': INITIAL_CAPITAL
                }
            )
    
    # Prepare visualization data
    print("\nPreparing visualization data...")
    viz_data = results_manager.prepare_backtest_visualization_data(test_df_bt)
    
    # Generate comprehensive backtest report with OHLC and trades
    print("\nGenerating comprehensive HTML report...")
    report_path = viz_manager.create_backtest_report(
        backtest_results=viz_data,
        save_dir=Path('results'),
        df=test_df_bt,  # Pass OHLC data for trade visualization
        test_results=results_manager.test_results,  # Include test metrics in report
        train_results=None  # No train results in this example
    )
    
    if report_path:
        print(f"\n✓ Comprehensive visualization report generated!")
        print(f"  Report includes:")
        print(f"    1. Equity curves comparison (all backtests on one chart)")
        print(f"    2. OHLC charts with trade markers for each backtest")
        print(f"    3. Performance metrics comparison")
        print(f"    4. Returns distributions")
        print(f"\n📊 Open the report to view: {report_path}")
    #endregion
    
    # ========================================================================
    # STEP 9: Summary
    # ========================================================================
    #region 'Summary'
    
    print("\n" + "="*80)
    print("WORKFLOW COMPLETE - SUMMARY")
    print("="*80)
    
    successful_backtests = sum(1 for r in results_comparison.values() if r['success'])
    
    print(f"\n✓ Data: {len(df_features)} samples")
    print(f"✓ Features: {len(feature_cols)}")
    print(f"✓ Model: Random Forest (Accuracy: {accuracy:.4f}, F1: {f1:.4f})")
    print(f"✓ Backtests run: {successful_backtests}/{len(backtests)}")
    
    if successful_backtests > 0:
        # Find best performing backtest
        best_backend = None
        best_return = -float('inf')
        
        for name, result in results_comparison.items():
            if result['success']:
                ret = result['metrics'].get('total_return', -float('inf'))
                if ret > best_return:
                    best_return = ret
                    best_backend = name
        
        if best_backend:
            print(f"\n🏆 Best performing backend: {best_backend}")
            print(f"   Total Return: {best_return*100:.2f}%")
    
    # Print run summary
    run_manager.print_summary()
    
    print("\n" + "="*80)
    print("🎉 Backtest comparison completed successfully!")
    print(f"📁 All artifacts saved to: {run_manager.get_run_dir()}")
    print("="*80 + "\n")
    #endregion
    return results_comparison


if __name__ == '__main__':
    results = main()
