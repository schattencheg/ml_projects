"""
Quick Start: ML Backtesting with Trailing Stop Loss

This is a simple example to get you started quickly with ML backtesting.
Run this file to see a complete backtest in action.
"""

import pandas as pd
from src.MLBacktester import BacktestNoLib
from src.data_preparation import prepare_data
from src.FeaturesGenerator import FeaturesGenerator
from src.model_loader import load_all_models, load_scaler, list_available_models


def main():
    """Quick start example for ML backtesting."""
    
    print("="*80)
    print("ML BACKTESTING - QUICK START")
    print("="*80)
    
    # Step 1: Load data
    print("\n[1/5] Loading data...")
    df_train = pd.read_csv("data/btc_2022.csv")
    df_test = pd.read_csv("data/btc_2023.csv")
    print(f"  Training data: {df_train.shape[0]} rows")
    print(f"  Test data: {df_test.shape[0]} rows")
    
    # Step 2: Load pre-trained models
    print("\n[2/5] Loading pre-trained models...")
    print("\nAvailable models:")
    available_models = list_available_models()
    
    if len(available_models) == 0:
        print("\n❌ No trained models found!")
        print("Please run 'python run_me.py' first to train models.")
        return
    
    models = load_all_models()
    scaler = load_scaler()
    
    # Use the first model as the best (or you can specify which one)
    best_model_name = available_models[0]
    print(f"\n  Using model: {best_model_name}")
    print(f"  Total models loaded: {len(models)}")
    
    # Step 3: Prepare test data with features
    print("\n[3/5] Preparing test data with features...")
    fg = FeaturesGenerator(df_test)
    df_test_with_features = fg.generate_all_features()
    
    # Get feature columns
    X_test, y_test = prepare_data(df_test, target_bars=45, target_pct=3.0)
    X_columns = X_test.columns.tolist()
    
    # Ensure we have all required columns
    required_cols = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'] + X_columns
    df_test_with_features = df_test_with_features[required_cols].dropna()
    print(f"  Test data with features: {df_test_with_features.shape[0]} rows")
    
    # Step 4: Initialize backtester
    print("\n[4/5] Initializing backtester...")
    backtester = BacktestNoLib(
        initial_capital=10000.0,        # Start with $10,000
        position_size=1.0,              # Use 100% of capital per trade
        trailing_stop_pct=2.0,          # 2% trailing stop loss
        take_profit_pct=5.0,            # 5% take profit target
        commission=0.001,               # 0.1% commission
        slippage=0.0005,                # 0.05% slippage
        use_probability_threshold=True,
        probability_threshold=0.6,      # Only enter when model is 60%+ confident
        max_holding_bars=None           # No max holding period
    )
    
    print("  Backtester settings:")
    print(f"    - Initial capital: ${backtester.initial_capital:,.2f}")
    print(f"    - Position size: {backtester.position_size * 100:.0f}%")
    print(f"    - Trailing stop: {backtester.trailing_stop_pct}%")
    print(f"    - Take profit: {backtester.take_profit_pct}%")
    print(f"    - Probability threshold: {backtester.probability_threshold}")
    
    # Step 5: Run backtest
    print("\n[5/5] Running backtest...")
    best_model = models[best_model_name]
    
    results = backtester.run_backtest(
        df=df_test_with_features,
        model=best_model,
        scaler=scaler,
        X_columns=X_columns,
        close_column='Close',
        timestamp_column='Timestamp'
    )
    
    # Display results
    backtester.print_results(results)
    
    # Create visualization
    print("\n[6/6] Creating visualization...")
    backtester.plot_results(
        results=results,
        df=df_test_with_features,
        close_column='Close',
        timestamp_column='Timestamp',
        save_path='plots/backtest_quick_start.png'
    )
    
    # Summary
    print("\n" + "="*80)
    print("QUICK START COMPLETE!")
    print("="*80)
    
    print("\nKey Results:")
    print(f"  ✓ Total Return: ${results['total_return']:,.2f} ({results['total_return_pct']:.2f}%)")
    print(f"  ✓ Buy & Hold Return: {results['buy_and_hold_return_pct']:.2f}%")
    print(f"  ✓ Total Trades: {results['total_trades']}")
    print(f"  ✓ Win Rate: {results['win_rate']:.2f}%")
    print(f"  ✓ Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"  ✓ Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    
    print("\nNext Steps:")
    print("  1. Check the plot saved to: plots/backtest_quick_start.png")
    print("  2. Try different trailing stop percentages (1%, 2%, 3%, etc.)")
    print("  3. Adjust probability threshold (0.5, 0.6, 0.7, etc.)")
    print("  4. Test different models from the trained models dictionary")
    print("  5. Run 'python backtest_example.py' for more advanced examples")
    print("  6. Read BACKTEST_GUIDE.md for detailed documentation")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
