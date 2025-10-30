"""
Example: Backtesting with Backtrader Library

This example demonstrates how to use the Backtrader library with ML models:
1. Load trained ML models
2. Run backtests using Backtrader
3. Optimize strategy parameters
4. Compare different models
"""

import pandas as pd
import numpy as np
from src.BacktraderStrategy import MLBacktesterBT
from src.data_preparation import prepare_data
from src.model_loader import load_all_models, load_scaler, list_available_models
import matplotlib.pyplot as plt


def main():
    """Main execution function."""
    
    print("="*80)
    print("ML BACKTESTING WITH BACKTRADER LIBRARY")
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
    backtester = MLBacktesterBT(
        initial_cash=10000.0,
        commission=0.001,  # 0.1%
        slippage_perc=0.0,
        slippage_fixed=0.0
    )
    
    # =========================================================================
    # EXAMPLE 1: Basic Backtest
    # =========================================================================
    print("\n" + "="*80)
    print("EXAMPLE 1: BASIC BACKTEST WITH BACKTRADER")
    print("="*80)
    
    results, trades = backtester.run_backtest(
        df=df_test_with_features,
        model=best_model,
        scaler=scaler,
        X_columns=X_columns,
        probability_threshold=0.6,
        trailing_stop_pct=2.0,
        take_profit_pct=5.0,
        position_size_pct=1.0,
        plot=True,
        printlog=False
    )
    
    # Print results
    backtester.print_results(results)
    
    # Print trade details
    if not trades.empty:
        print("\n" + "="*80)
        print("TRADE DETAILS (Last 10 trades)")
        print("="*80)
        print(trades.tail(10).to_string(index=False))
    
    # =========================================================================
    # EXAMPLE 2: Parameter Optimization
    # =========================================================================
    print("\n" + "="*80)
    print("EXAMPLE 2: PARAMETER OPTIMIZATION")
    print("="*80)
    
    print("\nOptimizing probability threshold and trailing stop...")
    print("This may take a few minutes...")
    
    optimization_results = backtester.optimize(
        df=df_test_with_features,
        model=best_model,
        scaler=scaler,
        X_columns=X_columns,
        probability_threshold_range=(0.5, 0.75, 0.1),
        trailing_stop_range=(1.0, 4.0, 1.0)
    )
    
    # Show top 10 results
    print("\n" + "="*80)
    print("TOP 10 OPTIMIZATION RESULTS")
    print("="*80)
    
    opt_df = pd.DataFrame(optimization_results[:10])
    print(opt_df.to_string(index=False))
    
    # Get best parameters
    best_params = optimization_results[0]
    print("\n" + "="*80)
    print("BEST PARAMETERS")
    print("="*80)
    print(f"  Probability Threshold: {best_params['probability_threshold']:.2f}")
    print(f"  Trailing Stop:         {best_params['trailing_stop_pct']:.2f}%")
    print(f"  Final Value:           ${best_params['final_value']:,.2f}")
    print(f"  Sharpe Ratio:          {best_params['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:          {best_params['max_drawdown']:.2f}%")
    
    # Plot optimization results
    opt_full_df = pd.DataFrame(optimization_results)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Scatter plot: Sharpe vs Drawdown
    scatter = axes[0].scatter(
        opt_full_df['max_drawdown'],
        opt_full_df['sharpe_ratio'],
        c=opt_full_df['final_value'],
        cmap='RdYlGn',
        s=100,
        alpha=0.6
    )
    axes[0].set_xlabel('Max Drawdown %')
    axes[0].set_ylabel('Sharpe Ratio')
    axes[0].set_title('Optimization Results: Risk vs Reward', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0], label='Final Value ($)')
    
    # Bar plot: Top 10 by Final Value
    top_10 = opt_full_df.nlargest(10, 'final_value')
    top_10['label'] = top_10.apply(
        lambda x: f"P:{x['probability_threshold']:.1f}\nS:{x['trailing_stop_pct']:.1f}",
        axis=1
    )
    axes[1].bar(range(len(top_10)), top_10['final_value'])
    axes[1].set_xticks(range(len(top_10)))
    axes[1].set_xticklabels(top_10['label'], rotation=45, ha='right')
    axes[1].set_ylabel('Final Value ($)')
    axes[1].set_title('Top 10 Parameter Combinations', fontweight='bold')
    axes[1].axhline(y=backtester.initial_cash, color='red', linestyle='--', alpha=0.5, label='Initial Cash')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/optimization_backtrader.png', dpi=300)
    print("\nOptimization plot saved to: plots/optimization_backtrader.png")
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
            results, trades = backtester.run_backtest(
                df=df_test_with_features,
                model=model,
                scaler=scaler,
                X_columns=X_columns,
                probability_threshold=0.6,
                trailing_stop_pct=2.0,
                take_profit_pct=5.0,
                position_size_pct=1.0,
                plot=False,
                printlog=False
            )
            
            comparison_results.append({
                'model': model_name,
                'return_pct': results['total_return_pct'],
                'sharpe_ratio': results['sharpe_ratio'],
                'max_drawdown': results['max_drawdown'],
                'total_trades': results['total_trades'],
                'win_rate': results['win_rate']
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
        plt.savefig('plots/model_comparison_backtrader.png', dpi=300)
        print("\nComparison plot saved to: plots/model_comparison_backtrader.png")
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
    
    conservative_results, _ = backtester.run_backtest(
        df=df_test_with_features,
        model=best_model,
        scaler=scaler,
        X_columns=X_columns,
        probability_threshold=0.7,
        trailing_stop_pct=1.5,
        take_profit_pct=4.0,
        position_size_pct=0.5,
        plot=False,
        printlog=False
    )
    
    print("\nAggressive Strategy:")
    print("  - Low probability threshold (0.55)")
    print("  - Wide trailing stop (3.5%)")
    print("  - Large position size (100%)")
    
    aggressive_results, _ = backtester.run_backtest(
        df=df_test_with_features,
        model=best_model,
        scaler=scaler,
        X_columns=X_columns,
        probability_threshold=0.55,
        trailing_stop_pct=3.5,
        take_profit_pct=8.0,
        position_size_pct=1.0,
        plot=False,
        printlog=False
    )
    
    # Compare strategies
    print("\n" + "="*80)
    print("STRATEGY COMPARISON")
    print("="*80)
    
    strategy_comparison = pd.DataFrame([
        {
            'strategy': 'Conservative',
            'return_pct': conservative_results['total_return_pct'],
            'sharpe_ratio': conservative_results['sharpe_ratio'],
            'max_drawdown': conservative_results['max_drawdown'],
            'total_trades': conservative_results['total_trades'],
            'win_rate': conservative_results['win_rate']
        },
        {
            'strategy': 'Aggressive',
            'return_pct': aggressive_results['total_return_pct'],
            'sharpe_ratio': aggressive_results['sharpe_ratio'],
            'max_drawdown': aggressive_results['max_drawdown'],
            'total_trades': aggressive_results['total_trades'],
            'win_rate': aggressive_results['win_rate']
        }
    ])
    
    print(strategy_comparison.to_string(index=False))
    
    # Plot strategy comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    strategies = strategy_comparison['strategy']
    
    # Return comparison
    axes[0].bar(strategies, strategy_comparison['return_pct'])
    axes[0].set_title('Total Return %', fontweight='bold')
    axes[0].set_ylabel('Return %')
    axes[0].grid(True, alpha=0.3)
    
    # Risk comparison
    axes[1].bar(strategies, strategy_comparison['max_drawdown'])
    axes[1].set_title('Max Drawdown %', fontweight='bold')
    axes[1].set_ylabel('Drawdown %')
    axes[1].grid(True, alpha=0.3)
    
    # Sharpe comparison
    axes[2].bar(strategies, strategy_comparison['sharpe_ratio'])
    axes[2].set_title('Sharpe Ratio', fontweight='bold')
    axes[2].set_ylabel('Sharpe Ratio')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/strategy_comparison_backtrader.png', dpi=300)
    print("\nStrategy comparison plot saved to: plots/strategy_comparison_backtrader.png")
    plt.show()
    
    print("\n" + "="*80)
    print("BACKTESTING COMPLETE!")
    print("="*80)
    print("\nKey Features of Backtrader:")
    print("  ✓ Event-driven backtesting (realistic)")
    print("  ✓ Flexible strategy development")
    print("  ✓ Built-in analyzers and observers")
    print("  ✓ Support for multiple data feeds")
    print("  ✓ Live trading integration")
    print("\nAll plots have been saved to the 'plots/' directory.")


if __name__ == "__main__":
    # Create plots directory if it doesn't exist
    import os
    os.makedirs('plots', exist_ok=True)
    
    main()
