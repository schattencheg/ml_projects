"""
Example: Backtesting with ML Model Predictions and Trailing Stop Loss

This example demonstrates how to:
1. Load trained ML models
2. Run backtests using model predictions as signals
3. Use trailing stop loss for risk management
4. Compare different models and parameters
"""

import pandas as pd
import numpy as np
from src.MLBacktester import MLBacktester
from src.data_preparation import prepare_data
from src.model_loader import load_all_models, load_scaler, list_available_models
import matplotlib.pyplot as plt


def backtest_single_model(
    model_name: str,
    model,
    scaler,
    df_test: pd.DataFrame,
    X_columns: list,
    initial_capital: float = 10000.0,
    trailing_stop_pct: float = 2.0,
    take_profit_pct: float = None,
    probability_threshold: float = 0.6,
    position_size: float = 1.0
):
    """
    Run backtest for a single model.
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    model : sklearn model
        Trained model
    scaler : sklearn scaler
        Fitted scaler
    df_test : pd.DataFrame
        Test dataframe with OHLCV data and features
    X_columns : list
        List of feature column names
    initial_capital : float
        Starting capital
    trailing_stop_pct : float
        Trailing stop loss percentage
    take_profit_pct : float, optional
        Take profit percentage
    probability_threshold : float
        Minimum probability to enter trade
    position_size : float
        Fraction of capital to use per trade
        
    Returns:
    --------
    backtester : MLBacktester
        Backtester instance with results
    results : dict
        Backtest results
    """
    # Initialize backtester
    backtester = MLBacktester(
        initial_capital=initial_capital,
        position_size=position_size,
        trailing_stop_pct=trailing_stop_pct,
        take_profit_pct=take_profit_pct,
        commission=0.001,  # 0.1% commission
        slippage=0.0005,   # 0.05% slippage
        use_probability_threshold=True,
        probability_threshold=probability_threshold,
        max_holding_bars=None  # No max holding period
    )
    
    print(f"\n{'='*80}")
    print(f"BACKTESTING MODEL: {model_name.upper()}")
    print(f"{'='*80}")
    
    # Run backtest
    results = backtester.run_backtest(
        df=df_test,
        model=model,
        scaler=scaler,
        X_columns=X_columns,
        close_column='Close',
        timestamp_column='Timestamp'
    )
    
    # Print results
    backtester.print_results(results)
    
    return backtester, results


def compare_trailing_stops(
    model_name: str,
    model,
    scaler,
    df_test: pd.DataFrame,
    X_columns: list,
    trailing_stops: list = [1.0, 2.0, 3.0, 5.0]
):
    """
    Compare different trailing stop percentages.
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    model : sklearn model
        Trained model
    scaler : sklearn scaler
        Fitted scaler
    df_test : pd.DataFrame
        Test dataframe
    X_columns : list
        Feature column names
    trailing_stops : list
        List of trailing stop percentages to test
        
    Returns:
    --------
    comparison_df : pd.DataFrame
        Comparison results
    """
    print(f"\n{'='*80}")
    print(f"COMPARING TRAILING STOP PERCENTAGES FOR {model_name.upper()}")
    print(f"{'='*80}")
    
    results_list = []
    
    for trailing_stop in trailing_stops:
        backtester = MLBacktester(
            initial_capital=10000.0,
            position_size=1.0,
            trailing_stop_pct=trailing_stop,
            take_profit_pct=None,
            commission=0.001,
            slippage=0.0005,
            use_probability_threshold=True,
            probability_threshold=0.6
        )
        
        results = backtester.run_backtest(
            df=df_test,
            model=model,
            scaler=scaler,
            X_columns=X_columns
        )
        
        results_list.append({
            'trailing_stop_pct': trailing_stop,
            'total_return_pct': results['total_return_pct'],
            'total_trades': results['total_trades'],
            'win_rate': results['win_rate'],
            'profit_factor': results['profit_factor'],
            'max_drawdown': results['max_drawdown'],
            'sharpe_ratio': results['sharpe_ratio']
        })
    
    comparison_df = pd.DataFrame(results_list)
    
    print("\nTrailing Stop Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(comparison_df['trailing_stop_pct'], comparison_df['total_return_pct'], 
                    marker='o', linewidth=2)
    axes[0, 0].set_title('Total Return % vs Trailing Stop %')
    axes[0, 0].set_xlabel('Trailing Stop %')
    axes[0, 0].set_ylabel('Total Return %')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(comparison_df['trailing_stop_pct'], comparison_df['win_rate'], 
                    marker='o', linewidth=2, color='green')
    axes[0, 1].set_title('Win Rate vs Trailing Stop %')
    axes[0, 1].set_xlabel('Trailing Stop %')
    axes[0, 1].set_ylabel('Win Rate %')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(comparison_df['trailing_stop_pct'], comparison_df['max_drawdown'], 
                    marker='o', linewidth=2, color='red')
    axes[1, 0].set_title('Max Drawdown vs Trailing Stop %')
    axes[1, 0].set_xlabel('Trailing Stop %')
    axes[1, 0].set_ylabel('Max Drawdown %')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(comparison_df['trailing_stop_pct'], comparison_df['sharpe_ratio'], 
                    marker='o', linewidth=2, color='purple')
    axes[1, 1].set_title('Sharpe Ratio vs Trailing Stop %')
    axes[1, 1].set_xlabel('Trailing Stop %')
    axes[1, 1].set_ylabel('Sharpe Ratio')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/trailing_stop_comparison.png', dpi=300, bbox_inches='tight')
    print("\nComparison plot saved to: plots/trailing_stop_comparison.png")
    plt.show()
    
    return comparison_df


def compare_models(
    models: dict,
    scaler,
    df_test: pd.DataFrame,
    X_columns: list,
    trailing_stop_pct: float = 2.0
):
    """
    Compare backtest results across different models.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    scaler : sklearn scaler
        Fitted scaler
    df_test : pd.DataFrame
        Test dataframe
    X_columns : list
        Feature column names
    trailing_stop_pct : float
        Trailing stop percentage to use
        
    Returns:
    --------
    comparison_df : pd.DataFrame
        Comparison results
    """
    print(f"\n{'='*80}")
    print(f"COMPARING MODELS WITH {trailing_stop_pct}% TRAILING STOP")
    print(f"{'='*80}")
    
    results_list = []
    
    for model_name, model in models.items():
        
        backtester = MLBacktester(
            initial_capital=10000.0,
            position_size=1.0,
            trailing_stop_pct=trailing_stop_pct,
            take_profit_pct=None,
            commission=0.001,
            slippage=0.0005,
            use_probability_threshold=True,
            probability_threshold=0.6
        )
        
        results = backtester.run_backtest(
            df=df_test,
            model=model,
            scaler=scaler,
            X_columns=X_columns
        )
        
        results_list.append({
            'model': model_name,
            'total_return_pct': results['total_return_pct'],
            'total_trades': results['total_trades'],
            'win_rate': results['win_rate'],
            'profit_factor': results['profit_factor'],
            'max_drawdown': results['max_drawdown'],
            'sharpe_ratio': results['sharpe_ratio'],
            #'avg_bars_held': results['avg_bars_held']
        })
    
    comparison_df = pd.DataFrame(results_list)
    comparison_df = comparison_df.sort_values('total_return_pct', ascending=False)
    
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].barh(comparison_df['model'], comparison_df['total_return_pct'])
    axes[0, 0].set_title('Total Return % by Model')
    axes[0, 0].set_xlabel('Total Return %')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    axes[0, 1].barh(comparison_df['model'], comparison_df['win_rate'], color='green')
    axes[0, 1].set_title('Win Rate by Model')
    axes[0, 1].set_xlabel('Win Rate %')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    axes[1, 0].barh(comparison_df['model'], comparison_df['max_drawdown'], color='red')
    axes[1, 0].set_title('Max Drawdown by Model')
    axes[1, 0].set_xlabel('Max Drawdown %')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    axes[1, 1].barh(comparison_df['model'], comparison_df['sharpe_ratio'], color='purple')
    axes[1, 1].set_title('Sharpe Ratio by Model')
    axes[1, 1].set_xlabel('Sharpe Ratio')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
    print("\nComparison plot saved to: plots/model_comparison.png")
    plt.show()
    
    return comparison_df


def main():
    """Main execution function."""
    
    # Load data
    print("Loading data...")
    path_train = "data/btc_2022.csv"
    path_test = "data/btc_2023.csv"
    
    df_train = pd.read_csv(path_train)
    df_test = pd.read_csv(path_test)
    
    print(f"Training data shape: {df_train.shape}")
    print(f"Test data shape: {df_test.shape}")
    
    # Load pre-trained models
    print("\n" + "="*80)
    print("LOADING PRE-TRAINED MODELS")
    print("="*80)
    
    print("\nAvailable models:")
    available_models = list_available_models()
    
    if len(available_models) == 0:
        print("\n❌ No trained models found!")
        print("Please run 'python run_me.py' first to train models.")
        return
    
    models = load_all_models()
    scaler = load_scaler()
    
    # Use the first model as default
    best_model_name = available_models[0]
    print(f"\n  Default model for examples: {best_model_name}")
    print(f"  Total models loaded: {len(models)}")
    
    # Prepare test data to get feature columns
    X_test, y_test = prepare_data(df_test, target_bars=45, target_pct=3.0)
    X_columns = X_test.columns.tolist()
    
    # Add features to test dataframe for backtesting
    from src.FeaturesGenerator import FeaturesGenerator
    fg = FeaturesGenerator()
    df_test_with_features = fg.add_features(df_test)
    
    # Ensure we have the required columns
    df_test_with_features = df_test_with_features[['Timestamp', 'Open', 'High', 'Low', 'Close'] + X_columns].dropna()
    
    print(f"\nTest data with features shape: {df_test_with_features.shape}")
    
    # Example 1: Backtest the best model
    print("\n" + "="*80)
    print("EXAMPLE 1: BACKTEST BEST MODEL")
    print("="*80)
    
    best_model = models[best_model_name]
    
    backtester, results = backtest_single_model(
        model_name=best_model_name,
        model=best_model,
        scaler=scaler,
        df_test=df_test_with_features,
        X_columns=X_columns,
        initial_capital=10000.0,
        trailing_stop_pct=2.0,
        take_profit_pct=5.0,  # 5% take profit
        probability_threshold=0.6,
        position_size=1.0  # Use 100% of capital
    )
    
    # Plot results
    backtester.plot_results(
        results=results,
        df=df_test_with_features,
        save_path=f'plots/backtest_{best_model_name}.png'
    )
    
    if False:
        # Example 2: Compare different trailing stop percentages
        print("\n" + "="*80)
        print("EXAMPLE 2: COMPARE TRAILING STOP PERCENTAGES")
        print("="*80)
        
        trailing_stop_comparison = compare_trailing_stops(
            model_name=best_model_name,
            model=best_model,
            scaler=scaler,
            df_test=df_test_with_features,
            X_columns=X_columns,
            trailing_stops=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
        )
    
    if False:
        # Example 3: Compare all models
        print("\n" + "="*80)
        print("EXAMPLE 3: COMPARE ALL MODELS")
        print("="*80)
    
        model_comparison = compare_models(
            models=models,
            scaler=scaler,
            df_test=df_test_with_features,
            X_columns=X_columns,
            trailing_stop_pct=2.0
        )
    
    if False:
        # Example 4: Conservative vs Aggressive strategy
        print("\n" + "="*80)
        print("EXAMPLE 4: CONSERVATIVE VS AGGRESSIVE STRATEGY")
        print("="*80)
    
        print("\nConservative Strategy (50% position size, 1.5% trailing stop, 0.7 prob threshold):")
        conservative_backtester, conservative_results = backtest_single_model(
            model_name=f"{best_model_name}_conservative",
            model=best_model,
            scaler=scaler,
            df_test=df_test_with_features,
            X_columns=X_columns,
            initial_capital=10000.0,
            trailing_stop_pct=1.5,
            take_profit_pct=4.0,
            probability_threshold=0.7,  # Higher threshold
            position_size=0.5  # Use only 50% of capital
        )
    
        print("\nAggressive Strategy (100% position size, 3% trailing stop, 0.55 prob threshold):")
        aggressive_backtester, aggressive_results = backtest_single_model(
            model_name=f"{best_model_name}_aggressive",
            model=best_model,
            scaler=scaler,
            df_test=df_test_with_features,
            X_columns=X_columns,
            initial_capital=10000.0,
            trailing_stop_pct=3.0,
            take_profit_pct=8.0,
            probability_threshold=0.55,  # Lower threshold
            position_size=1.0  # Use 100% of capital
        )
    
        # Compare strategies
        print("\n" + "="*80)
        print("STRATEGY COMPARISON")
        print("="*80)
        
        strategy_comparison = pd.DataFrame([
            {
                'strategy': 'Conservative',
                'total_return_pct': conservative_results['total_return_pct'],
                'total_trades': conservative_results['total_trades'],
                'win_rate': conservative_results['win_rate'],
                'max_drawdown': conservative_results['max_drawdown'],
                'sharpe_ratio': conservative_results['sharpe_ratio']
            },
            {
                'strategy': 'Aggressive',
                'total_return_pct': aggressive_results['total_return_pct'],
                'total_trades': aggressive_results['total_trades'],
                'win_rate': aggressive_results['win_rate'],
                'max_drawdown': aggressive_results['max_drawdown'],
                'sharpe_ratio': aggressive_results['sharpe_ratio']
            }
        ])
        
        print(strategy_comparison.to_string(index=False))
    
    print("\n" + "="*80)
    print("BACKTESTING COMPLETE!")
    print("="*80)
    print("\nAll plots have been saved to the 'plots/' directory.")
    print("\nKey Takeaways:")
    print("1. Trailing stop loss helps protect profits and limit losses")
    print("2. Different models may perform differently in backtesting")
    print("3. Position sizing and probability thresholds affect risk/reward")
    print("4. Conservative strategies typically have lower returns but also lower drawdowns")


if __name__ == "__main__":
    main()
