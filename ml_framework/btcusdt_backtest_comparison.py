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
    THRESHOLD = 0.02
    
    # Backtest configuration
    INITIAL_CAPITAL = 10000.0
    COMMISSION = 0.001
    POSITION_SIZE = 1.0
    
    print(f"\nConfiguration:")
    print(f"  Ticker: {TICKER}")
    print(f"  Period: {START_DATE} to {END_DATE}")
    print(f"  Target: Predict {THRESHOLD*100}% change in {FUTURE_BARS} bars")
    print(f"  Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"  Commission: {COMMISSION*100}%")
    
    # ========================================================================
    # STEP 1: Load Data
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
    
    # ========================================================================
    # STEP 2: Generate Features
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
        num_classes=3  # Three-class: -1 (decrease), 0 (neutral), +1 (increase)
    )
    df_features = df_features.dropna()
    
    print(f"\n✓ Features generated: {len(df_features.columns)} columns")
    print(f"✓ Dataset ready: {len(df_features)} rows")
    
    # ========================================================================
    # STEP 3: Split Data
    # ========================================================================
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
    
    # ========================================================================
    # STEP 4: Train Model
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 4] TRAINING MODEL")
    print("="*80)
    
    # Use Random Forest for better performance
    model = RandomForestModel(
        name='RandomForest',
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)
    
    # Evaluate
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Re-train on scaled data
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n✓ Model trained")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # ========================================================================
    # STEP 5: Run Backtests
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 5] RUNNING BACKTESTS - COMPARING ALL BACKENDS")
    print("="*80)
    
    # Prepare test data with all required columns
    test_df_bt = test_df.copy()
    
    # Initialize backtesting engines
    backtests = {
        'NoLib': BacktestNoLib(
            initial_capital=INITIAL_CAPITAL,
            commission=COMMISSION,
            position_size=POSITION_SIZE,
            stop_loss=0.05,  # 5% stop loss
            take_profit=0.10  # 10% take profit
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
    
    results_comparison = {}
    
    for name, backtest in backtests.items():
        print(f"\n{'='*80}")
        print(f"Running {name} Backtest")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            test_df_bt = df_features.copy() # Evaluating backtest on whole period (Train+Test+Val)
            results = backtest.run(
                df=test_df_bt,
                model=model,
                scaler=scaler,
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
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Error running {name}: {e}")
            print(f"   This backend may require additional dependencies")
            results_comparison[name] = {
                'metrics': {},
                'execution_time': 0,
                'success': False,
                'error': str(e)
            }
    
    # ========================================================================
    # STEP 6: Compare Results
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 6] BACKTEST COMPARISON SUMMARY")
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
    
    # ========================================================================
    # STEP 7: Recommendations
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 7] RECOMMENDATIONS")
    print("="*80)
    
    print("\n📊 Backend Comparison:")
    print("\n1. BacktestNoLib (Custom):")
    print("   ✓ Pros: Simple, transparent, easy to customize")
    print("   ✓ Pros: No external dependencies")
    print("   ✓ Pros: Includes stop loss and take profit")
    print("   ✗ Cons: Manual implementation, slower")
    
    print("\n2. Backtrader:")
    print("   ✓ Pros: Event-driven, realistic simulation")
    print("   ✓ Pros: Live trading ready")
    print("   ✓ Pros: Extensive features")
    print("   ✗ Cons: Requires backtrader library")
    print("   ✗ Cons: Steeper learning curve")
    
    print("\n3. Backtesting.py:")
    print("   ✓ Pros: Fast vectorized backtesting")
    print("   ✓ Pros: Built-in optimization")
    print("   ✓ Pros: Interactive visualizations")
    print("   ✗ Cons: Requires backtesting library")
    print("   ✗ Cons: Less realistic than event-driven")
    
    print("\n💡 Use Case Recommendations:")
    print("  • Quick prototyping: BacktestNoLib or BacktestingPy")
    print("  • Parameter optimization: BacktestingPy")
    print("  • Realistic simulation: Backtrader")
    print("  • Production/Live trading: Backtrader")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
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
    
    print("\n" + "="*80)
    print("🎉 Backtest comparison completed successfully!")
    print("="*80 + "\n")
    
    return results_comparison


if __name__ == '__main__':
    results = main()
