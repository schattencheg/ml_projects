"""
Example: Backtesting with backtesting.py Library

This example demonstrates how to use the backtesting.py library with ML models:
1. Load trained ML models
2. Run backtests using backtesting.py
3. Optimize strategy parameters
4. Compare different models
"""

import pandas as pd
import numpy as np
from src.BacktestingPyStrategy import MLBacktesterPy
from src.data_preparation import prepare_data
from src.model_loader import load_all_models, load_scaler, list_available_models
import matplotlib.pyplot as plt


def main():
    """Main execution function."""
    
    print("="*80)
    print("ML BACKTESTING WITH BACKTESTING.PY LIBRARY")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    df_train, df_test = prepare_data('data/6e_2007_2019.csv')
    print(f"Training data shape: {df_train.shape}")
    print(f"Test data shape: {df_test.shape}")
    
    # Load pre-trained models
    print("\n" + "="*80)
    print("LOADING PRE-TRAINED MODELS")
    print("="*80)
    
    models = load_all_models()
    scaler = load_scaler()
    
    if not models:
        print("\nNo models found! Please run train_and_save_models.py first.")
        return
    
    print("\nAvailable models:")
    for model_name in models.keys():
        print(f"  ✓ Loaded: {model_name}")
    
    print(f"Loaded scaler: scaler")
    
    # Select best model (or use first available)
    best_model_name = list(models.keys())[0]
    best_model = models[best_model_name]
    
    print(f"\n  Default model for examples: {best_model_name}")
    print(f"  Total models loaded: {len(models)}")
    
    # Prepare test data with features
    from src.FeaturesGenerator import FeaturesGenerator
    
    fg = FeaturesGenerator()
    df_test_with_features = fg.generate(df_test.copy())
    
    # Get feature columns (exclude OHLCV and target)
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'Timestamp']
    X_columns = [col for col in df_test_with_features.columns if col not in exclude_cols]
    
    print(f"\nTest data with features shape: {df_test_with_features.shape}")
    print(f"Number of features: {len(X_columns)}")
    
    # Initialize backtester
    backtester = MLBacktesterPy(
        initial_cash=10000.0,
        commission=0.001,  # 0.1%
        margin=1.0,
        trade_on_close=False
    )
    
    # =========================================================================
    # EXAMPLE 1: Basic Backtest
    # =========================================================================
    print("\n" + "="*80)
    print("EXAMPLE 1: BASIC BACKTEST WITH BACKTESTING.PY")
    print("="*80)
    
    stats, trades = backtester.run_backtest(
        df=df_test_with_features,
        model=best_model,
        scaler=scaler,
        X_columns=X_columns,
        probability_threshold=0.6,
        trailing_stop_pct=2.0,
        take_profit_pct=5.0,
        position_size_pct=1.0,
        plot=True
    )
    
    # Print results
    backtester.print_results(stats)
    
    # =========================================================================
    # EXAMPLE 2: Parameter Optimization
    # =========================================================================
    print("\n" + "="*80)
    print("EXAMPLE 2: PARAMETER OPTIMIZATION")
    print("="*80)
    
    print("\nOptimizing probability threshold and trailing stop...")
    
    best_stats, heatmap = backtester.optimize(
        df=df_test_with_features,
        model=best_model,
        scaler=scaler,
        X_columns=X_columns,
        probability_threshold_range=(0.5, 0.75, 0.05),
        trailing_stop_range=(1.0, 4.0, 0.5),
        maximize='Return [%]',
        return_heatmap=True
    )
    
    print("\nOptimization Results:")
    print(f"  Best Probability Threshold: {best_stats._strategy.probability_threshold:.2f}")
    print(f"  Best Trailing Stop:         {best_stats._strategy.trailing_stop_pct:.2f}%")
    print(f"  Best Return:                {best_stats['Return [%]']:.2f}%")
    print(f"  Sharpe Ratio:               {best_stats['Sharpe Ratio']:.2f}")
    print(f"  Max Drawdown:               {best_stats['Max. Drawdown [%]']:.2f}%")
    
    # Plot heatmap
    if heatmap is not None:
        plt.figure(figsize=(12, 8))
        plt.imshow(heatmap, cmap='RdYlGn', aspect='auto')
        plt.colorbar(label='Return [%]')
        plt.title('Optimization Heatmap: Return [%]', fontsize=14, fontweight='bold')
        plt.xlabel('Trailing Stop %')
        plt.ylabel('Probability Threshold')
        plt.tight_layout()
        plt.savefig('plots/optimization_heatmap_backtestingpy.png', dpi=300)
        print("\nHeatmap saved to: plots/optimization_heatmap_backtestingpy.png")
        plt.show()
    
    # =========================================================================
    # EXAMPLE 3: Compare Multiple Models
    # =========================================================================
    print("\n" + "="*80)
    print("EXAMPLE 3: COMPARE MULTIPLE MODELS")
    print("="*80)
    
    if len(models) > 1:
        comparison_results = []
        
        for model_name, model in models.items():
            print(f"\nBacktesting {model_name}...")
            
            # Run backtest without plotting
            stats, trades = backtester.run_backtest(
                df=df_test_with_features,
                model=model,
                scaler=scaler,
                X_columns=X_columns,
                probability_threshold=0.6,
                trailing_stop_pct=2.0,
                take_profit_pct=5.0,
                position_size_pct=1.0,
                plot=False
            )
            
            comparison_results.append({
                'model': model_name,
                'return_pct': stats['Return [%]'],
                'sharpe_ratio': stats['Sharpe Ratio'],
                'max_drawdown': stats['Max. Drawdown [%]'],
                'total_trades': stats['# Trades'],
                'win_rate': stats['Win Rate [%]']
            })
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.sort_values('return_pct', ascending=False)
        
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        print(comparison_df.to_string(index=False))
        
        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Return %
        axes[0, 0].bar(comparison_df['model'], comparison_df['return_pct'])
        axes[0, 0].set_title('Total Return %', fontweight='bold')
        axes[0, 0].set_ylabel('Return %')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Sharpe Ratio
        axes[0, 1].bar(comparison_df['model'], comparison_df['sharpe_ratio'])
        axes[0, 1].set_title('Sharpe Ratio', fontweight='bold')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Max Drawdown
        axes[1, 0].bar(comparison_df['model'], comparison_df['max_drawdown'])
        axes[1, 0].set_title('Max Drawdown %', fontweight='bold')
        axes[1, 0].set_ylabel('Drawdown %')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Win Rate
        axes[1, 1].bar(comparison_df['model'], comparison_df['win_rate'])
        axes[1, 1].set_title('Win Rate %', fontweight='bold')
        axes[1, 1].set_ylabel('Win Rate %')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/model_comparison_backtestingpy.png', dpi=300)
        print("\nComparison plot saved to: plots/model_comparison_backtestingpy.png")
        plt.show()
    else:
        print("\nOnly one model available. Train more models to compare.")
    
    # =========================================================================
    # EXAMPLE 4: Conservative vs Aggressive Strategy
    # =========================================================================
    print("\n" + "="*80)
    print("EXAMPLE 4: CONSERVATIVE VS AGGRESSIVE STRATEGY")
    print("="*80)
    
    print("\nConservative Strategy:")
    print("  - High probability threshold (0.7)")
    print("  - Tight trailing stop (1.5%)")
    print("  - Small position size (50%)")
    
    conservative_stats, _ = backtester.run_backtest(
        df=df_test_with_features,
        model=best_model,
        scaler=scaler,
        X_columns=X_columns,
        probability_threshold=0.7,
        trailing_stop_pct=1.5,
        take_profit_pct=4.0,
        position_size_pct=0.5,
        plot=False
    )
    
    print("\nAggressive Strategy:")
    print("  - Low probability threshold (0.55)")
    print("  - Wide trailing stop (3.5%)")
    print("  - Large position size (100%)")
    
    aggressive_stats, _ = backtester.run_backtest(
        df=df_test_with_features,
        model=best_model,
        scaler=scaler,
        X_columns=X_columns,
        probability_threshold=0.55,
        trailing_stop_pct=3.5,
        take_profit_pct=8.0,
        position_size_pct=1.0,
        plot=False
    )
    
    # Compare strategies
    print("\n" + "="*80)
    print("STRATEGY COMPARISON")
    print("="*80)
    
    strategy_comparison = pd.DataFrame([
        {
            'strategy': 'Conservative',
            'return_pct': conservative_stats['Return [%]'],
            'sharpe_ratio': conservative_stats['Sharpe Ratio'],
            'max_drawdown': conservative_stats['Max. Drawdown [%]'],
            'total_trades': conservative_stats['# Trades'],
            'win_rate': conservative_stats['Win Rate [%]']
        },
        {
            'strategy': 'Aggressive',
            'return_pct': aggressive_stats['Return [%]'],
            'sharpe_ratio': aggressive_stats['Sharpe Ratio'],
            'max_drawdown': aggressive_stats['Max. Drawdown [%]'],
            'total_trades': aggressive_stats['# Trades'],
            'win_rate': aggressive_stats['Win Rate [%]']
        }
    ])
    
    print(strategy_comparison.to_string(index=False))
    
    print("\n" + "="*80)
    print("BACKTESTING COMPLETE!")
    print("="*80)
    print("\nKey Features of backtesting.py:")
    print("  ✓ Fast vectorized backtesting")
    print("  ✓ Built-in parameter optimization")
    print("  ✓ Interactive plots with Bokeh")
    print("  ✓ Comprehensive statistics")
    print("  ✓ Easy-to-use API")
    print("\nAll plots have been saved to the 'plots/' directory.")


if __name__ == "__main__":
    # Create plots directory if it doesn't exist
    import os
    os.makedirs('plots', exist_ok=True)
    
    main()
