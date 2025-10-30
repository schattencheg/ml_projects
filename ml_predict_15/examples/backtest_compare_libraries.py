"""
Compare backtesting.py vs Backtrader

This script runs the same strategy using both libraries and compares results.
This helps you understand the differences and choose the right library for your needs.
"""

import pandas as pd
import numpy as np
from src.BacktestingPyStrategy import MLBacktesterPy
from src.BacktraderStrategy import MLBacktesterBT
from src.data_preparation import prepare_data
from src.model_loader import load_all_models, load_scaler
from src.FeaturesGenerator import FeaturesGenerator
import matplotlib.pyplot as plt
import time


def main():
    """Main execution function."""
    
    print("="*80)
    print("COMPARING BACKTESTING.PY VS BACKTRADER")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    df_train, df_test = prepare_data('data/6e_2007_2019.csv')
    print(f"Test data shape: {df_test.shape}")
    
    # Load pre-trained models
    print("\nLoading models...")
    models = load_all_models()
    scaler = load_scaler()
    
    if not models:
        print("\nNo models found! Please run train_and_save_models.py first.")
        return
    
    # Select model
    model_name = list(models.keys())[0]
    model = models[model_name]
    print(f"Using model: {model_name}")
    
    # Prepare test data with features
    fg = FeaturesGenerator()
    df_test_with_features = fg.generate(df_test.copy())
    
    # Get feature columns
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'Timestamp']
    X_columns = [col for col in df_test_with_features.columns if col not in exclude_cols]
    
    print(f"Number of features: {len(X_columns)}")
    
    # Common parameters
    params = {
        'probability_threshold': 0.6,
        'trailing_stop_pct': 2.0,
        'take_profit_pct': 5.0,
        'position_size_pct': 1.0
    }
    
    print("\n" + "="*80)
    print("STRATEGY PARAMETERS")
    print("="*80)
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # =========================================================================
    # Test 1: backtesting.py
    # =========================================================================
    print("\n" + "="*80)
    print("RUNNING BACKTEST WITH BACKTESTING.PY")
    print("="*80)
    
    backtester_py = MLBacktesterPy(
        initial_cash=10000.0,
        commission=0.001,
        margin=1.0,
        trade_on_close=False
    )
    
    start_time = time.time()
    
    stats_py, trades_py = backtester_py.run_backtest(
        df=df_test_with_features,
        model=model,
        scaler=scaler,
        X_columns=X_columns,
        plot=False,
        **params
    )
    
    time_py = time.time() - start_time
    
    print(f"\nExecution time: {time_py:.2f} seconds")
    
    # =========================================================================
    # Test 2: Backtrader
    # =========================================================================
    print("\n" + "="*80)
    print("RUNNING BACKTEST WITH BACKTRADER")
    print("="*80)
    
    backtester_bt = MLBacktesterBT(
        initial_cash=10000.0,
        commission=0.001,
        slippage_perc=0.0,
        slippage_fixed=0.0
    )
    
    start_time = time.time()
    
    results_bt, trades_bt = backtester_bt.run_backtest(
        df=df_test_with_features,
        model=model,
        scaler=scaler,
        X_columns=X_columns,
        plot=False,
        printlog=False,
        **params
    )
    
    time_bt = time.time() - start_time
    
    print(f"\nExecution time: {time_bt:.2f} seconds")
    
    # =========================================================================
    # Compare Results
    # =========================================================================
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    comparison = pd.DataFrame([
        {
            'Library': 'backtesting.py',
            'Execution Time (s)': time_py,
            'Final Value ($)': stats_py['Equity Final [$]'],
            'Return (%)': stats_py['Return [%]'],
            'Total Trades': stats_py['# Trades'],
            'Win Rate (%)': stats_py['Win Rate [%]'],
            'Sharpe Ratio': stats_py['Sharpe Ratio'],
            'Max Drawdown (%)': stats_py['Max. Drawdown [%]']
        },
        {
            'Library': 'Backtrader',
            'Execution Time (s)': time_bt,
            'Final Value ($)': results_bt['final_value'],
            'Return (%)': results_bt['total_return_pct'],
            'Total Trades': results_bt['total_trades'],
            'Win Rate (%)': results_bt['win_rate'],
            'Sharpe Ratio': results_bt['sharpe_ratio'],
            'Max Drawdown (%)': results_bt['max_drawdown']
        }
    ])
    
    print("\n" + comparison.to_string(index=False))
    
    # Calculate differences
    print("\n" + "="*80)
    print("DIFFERENCES")
    print("="*80)
    
    return_diff = abs(stats_py['Return [%]'] - results_bt['total_return_pct'])
    trades_diff = abs(stats_py['# Trades'] - results_bt['total_trades'])
    speed_ratio = time_bt / time_py if time_py > 0 else 0
    
    print(f"\nReturn Difference:       {return_diff:.2f}%")
    print(f"Trades Difference:       {trades_diff}")
    print(f"Speed Ratio (BT/BP):     {speed_ratio:.2f}x")
    print(f"backtesting.py is:       {speed_ratio:.1f}x faster")
    
    # =========================================================================
    # Visualize Comparison
    # =========================================================================
    print("\n" + "="*80)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('backtesting.py vs Backtrader Comparison', fontsize=16, fontweight='bold')
    
    libraries = ['backtesting.py', 'Backtrader']
    
    # Execution Time
    axes[0, 0].bar(libraries, [time_py, time_bt], color=['#2ecc71', '#3498db'])
    axes[0, 0].set_title('Execution Time', fontweight='bold')
    axes[0, 0].set_ylabel('Seconds')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Return %
    axes[0, 1].bar(libraries, [stats_py['Return [%]'], results_bt['total_return_pct']], 
                   color=['#2ecc71', '#3498db'])
    axes[0, 1].set_title('Total Return %', fontweight='bold')
    axes[0, 1].set_ylabel('Return %')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Total Trades
    axes[0, 2].bar(libraries, [stats_py['# Trades'], results_bt['total_trades']], 
                   color=['#2ecc71', '#3498db'])
    axes[0, 2].set_title('Total Trades', fontweight='bold')
    axes[0, 2].set_ylabel('Number of Trades')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Win Rate
    axes[1, 0].bar(libraries, [stats_py['Win Rate [%]'], results_bt['win_rate']], 
                   color=['#2ecc71', '#3498db'])
    axes[1, 0].set_title('Win Rate %', fontweight='bold')
    axes[1, 0].set_ylabel('Win Rate %')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Sharpe Ratio
    axes[1, 1].bar(libraries, [stats_py['Sharpe Ratio'], results_bt['sharpe_ratio']], 
                   color=['#2ecc71', '#3498db'])
    axes[1, 1].set_title('Sharpe Ratio', fontweight='bold')
    axes[1, 1].set_ylabel('Sharpe Ratio')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Max Drawdown
    axes[1, 2].bar(libraries, [stats_py['Max. Drawdown [%]'], results_bt['max_drawdown']], 
                   color=['#2ecc71', '#3498db'])
    axes[1, 2].set_title('Max Drawdown %', fontweight='bold')
    axes[1, 2].set_ylabel('Drawdown %')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/library_comparison.png', dpi=300, bbox_inches='tight')
    print("\nComparison plot saved to: plots/library_comparison.png")
    plt.show()
    
    # =========================================================================
    # Summary and Recommendations
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*80)
    
    print("\n📊 Performance Comparison:")
    print(f"  • backtesting.py executed in {time_py:.2f}s")
    print(f"  • Backtrader executed in {time_bt:.2f}s")
    print(f"  • Speed difference: {speed_ratio:.1f}x")
    
    print("\n💰 Return Comparison:")
    if abs(return_diff) < 1.0:
        print(f"  • Results are very similar (diff: {return_diff:.2f}%)")
        print(f"  • Both libraries produce consistent results ✓")
    else:
        print(f"  • Results differ by {return_diff:.2f}%")
        print(f"  • This may be due to execution model differences")
    
    print("\n🎯 When to Use Each Library:")
    print("\n  Use backtesting.py when:")
    print("    ✓ You need fast iterations")
    print("    ✓ You want to optimize parameters")
    print("    ✓ You prefer interactive visualizations")
    print("    ✓ You're doing statistical analysis")
    
    print("\n  Use Backtrader when:")
    print("    ✓ You need realistic order execution")
    print("    ✓ You plan to move to live trading")
    print("    ✓ You need complex strategies")
    print("    ✓ You want detailed control")
    
    print("\n💡 Best Practice:")
    print("  • Use backtesting.py for rapid development and optimization")
    print("  • Validate final strategy with Backtrader for realism")
    print("  • Compare results between both for confidence")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    import os
    os.makedirs('plots', exist_ok=True)
    
    main()
