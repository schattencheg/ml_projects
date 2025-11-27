"""
Example: Backtest Visualization with Equity Curves and Trade Markers

This example demonstrates:
1. Running backtests with multiple backends
2. Generating comprehensive visualization reports
3. Equity curves comparison across all backtests
4. OHLC charts with trade entry/exit markers
5. Using ResultManager and VisualizationManager

Author: ML Framework
Date: 2024-11-25
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_provider import DataProvider
from src.features_generator import FeaturesGenerator
from src.models_lib import RandomForestModel
from src.backtesting import BacktestNoLib, BacktestBacktrader, BacktestBacktestingPy
from src.managers.result_manager import ResultManager
from src.managers.visualization_manager import VisualizationManager
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score


def main():
    """Run backtest with comprehensive visualization."""
    
    print("\n" + "="*80)
    print("BACKTEST VISUALIZATION EXAMPLE")
    print("="*80)
    
    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 1] LOADING DATA")
    print("="*80)
    
    TICKER = 'BTC-USD'
    START_DATE = '2023-01-01'
    END_DATE = '2024-11-25'
    INTERVAL = '1d'
    
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
        future_bars=15,
        threshold=0.02,
        num_classes=3  # Three-class: -1, 0, +1
    )
    df_features = df_features.dropna()
    
    print(f"\n✓ Features generated: {len(df_features.columns)} columns")
    print(f"✓ Dataset ready: {len(df_features)} rows")
    
    # ========================================================================
    # STEP 3: Split Data and Train Model
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 3] TRAINING MODEL")
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
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    model = RandomForestModel(
        name='RandomForest',
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    print("\nTraining Random Forest...")
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n✓ Model trained")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # Store test results for visualization
    from sklearn.metrics import precision_score, recall_score
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results_manager.add_test_results({
        'RandomForest': {
            'status': 'success',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    })
    
    # ========================================================================
    # STEP 4: Run Backtests
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 4] RUNNING BACKTESTS")
    print("="*80)
    
    INITIAL_CAPITAL = 10000.0
    COMMISSION = 0.001
    POSITION_SIZE = 0.02  # 2% of capital per trade
    BARS_TO_HOLD = 15     # Exit after N bars
    
    # Prepare test data
    test_df_bt = test_df.copy()
    
    # Initialize backtesting engines
    # NoLib uses RiskManager for position sizing and exit timing
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
    
    for name, backtest in backtests.items():
        print(f"\n{'='*70}")
        print(f"Running {name} backtest...")
        print(f"{'='*70}")
        
        try:
            results = backtest.run(
                df=test_df_bt,
                model=model,
                scaler=scaler,
                feature_cols=feature_cols,
                price_col='close'
            )
            
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
            print(f"\n✗ {name} backtest failed: {str(e)}")
            results_manager.add_backtest_results(
                model_name=name,
                results={'status': 'failed', 'error': str(e)}
            )
    
    # ========================================================================
    # STEP 5: Generate Visualizations
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 5] GENERATING VISUALIZATIONS")
    print("="*80)
    
    viz_manager = VisualizationManager()
    
    # Prepare visualization data
    viz_data = results_manager.prepare_backtest_visualization_data(test_df_bt)
    
    # Create comprehensive backtest report with OHLC and trades
    report_path = viz_manager.create_backtest_report(
        backtest_results=viz_data,
        save_dir=Path('results'),
        df=test_df_bt,  # Pass OHLC data for trade visualization
        test_results=results_manager.test_results,  # Include test metrics in report
        train_results=None  # No train results in this example
    )
    
    print(f"\n✓ Visualization report generated")
    print(f"  Report path: {report_path}")
    
    # ========================================================================
    # STEP 6: Summary
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 6] SUMMARY")
    print("="*80)
    
    results_manager.print_summary()
    
    # Print comparison
    comparison_df = results_manager.compare_backtests()
    if not comparison_df.empty:
        print("\nBacktest Comparison:")
        print(comparison_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("🎉 Backtest visualization example completed!")
    print("="*80)
    print(f"\n📊 Open the HTML report to view:")
    print(f"   1. Equity curves comparison (all backtests on one chart)")
    print(f"   2. OHLC charts with trade markers for each backtest")
    print(f"   3. Performance metrics comparison")
    print(f"   4. Returns distributions")
    print(f"\n📁 Report location: {report_path}")
    print("="*80 + "\n")
    
    return results_manager


if __name__ == '__main__':
    results = main()
