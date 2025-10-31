"""
Main Training and Testing Script - Refactored

Uses the new class-based architecture:
- ModelsManager: Model lifecycle
- FeaturesGenerator: Feature generation
- Trainer: Training logic
- Tester: Testing logic
- ReportManager: Report generation
- HealthManager: Health monitoring
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mplfinance as mpf
from datetime import datetime

# Import new classes
from src.ModelsManager import ModelsManager
from src.FeaturesGenerator import FeaturesGenerator
from src.Trainer import Trainer
from src.Tester import Tester
from src.ReportManager import ReportManager
from src.HealthManager import HealthManager
# Backtesting
from src.BacktestNoLib import BacktestNoLib
from src.BacktestBacktrader import BacktestBacktraderML
from src.BacktestBacktesting import BacktestBacktestingML


# Data paths
PATH_TRAIN = "data/hour/btc.csv"
PATH_TEST = "data/hour/btc_2025.csv"
PATH_MODELS = "models"
os.environ["PATH_TRAIN"] = PATH_TRAIN
os.environ["PATH_TEST"] = PATH_TEST
os.environ["PATH_MODELS"] = PATH_MODELS


def _create_models_comparison_plot(all_results, all_equity_curves):
    """
    Create a comprehensive comparison plot for all tested models.
    
    Parameters:
    -----------
    all_results : dict
        Dictionary of model_name -> results dict
    all_equity_curves : dict
        Dictionary of model_name -> equity curve DataFrame
    """
    if not all_results:
        print("⚠ No results to plot - all_results is empty")
        return
    
    if not all_equity_curves:
        print("⚠ No equity curves to plot - all_equity_curves is empty")
        print(f"  Available results keys: {list(all_results.keys())}")
        return
    
    # Filter valid equity curves
    valid_equity_curves = {}
    for model_name, equity_df in all_equity_curves.items():
        if equity_df is not None and len(equity_df) > 0:
            if 'equity' in equity_df.columns and 'timestamp' in equity_df.columns:
                valid_equity_curves[model_name] = equity_df
            else:
                print(f"⚠ Skipping {model_name}: missing required columns (has: {list(equity_df.columns)})")
        else:
            print(f"⚠ Skipping {model_name}: equity curve is empty or None")
    
    if not valid_equity_curves:
        print("⚠ No valid equity curves to plot")
        print(f"  Total equity curves: {len(all_equity_curves)}")
        return
    
    print(f"\nPlotting {len(valid_equity_curves)} model(s)...")
    
    # Create figure with 2 subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
    
    # Subplot 1: Equity curves
    ax1 = fig.add_subplot(gs[0])
    
    # Define colors for models
    colors = plt.cm.tab10(np.linspace(0, 1, len(valid_equity_curves)))
    
    # Plot each model's equity curve
    for i, (model_name, equity_df) in enumerate(valid_equity_curves.items()):
        # Normalize to percentage return
        initial_capital = all_results[model_name].get('initial_capital', 10000)
        equity_pct = (equity_df['equity'] / initial_capital - 1) * 100
        
        ax1.plot(equity_df['timestamp'], equity_pct, 
                label=model_name.replace('_', ' ').title(),
                linewidth=2, color=colors[i], alpha=0.8)
    
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Return (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Comparison - Equity Curves', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis
    ax1.tick_params(axis='x', rotation=45)
    
    # Subplot 2: Summary table
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    
    # Create summary data
    summary_data = []
    for model_name, results in all_results.items():
        summary_data.append([
            model_name.replace('_', ' ').title(),
            f"${results.get('final_value', 0):,.0f}",
            f"{results.get('total_return_pct', 0):.2f}%",
            f"{results.get('total_trades', 0)}",
            f"{results.get('win_rate', 0):.1f}%",
            f"{results.get('sharpe_ratio', 0):.2f}",
            f"{results.get('max_drawdown', 0):.1f}%"
        ])
    
    # Sort by total return
    summary_data.sort(key=lambda x: float(x[2].rstrip('%')), reverse=True)
    
    # Create table
    columns = ['Model', 'Final Value', 'Return', 'Trades', 'Win Rate', 'Sharpe', 'Max DD']
    
    table = ax2.table(cellText=summary_data, colLabels=columns,
                     cellLoc='center', loc='center',
                     colWidths=[0.20, 0.15, 0.12, 0.10, 0.12, 0.10, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')
    
    # Style rows - alternate colors
    for i in range(1, len(summary_data) + 1):
        for j in range(len(columns)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')
            else:
                cell.set_facecolor('white')
    
    # Highlight best model (first row after sorting)
    for j in range(len(columns)):
        cell = table[(1, j)]
        cell.set_facecolor('#90EE90')
        cell.set_text_props(weight='bold')
    
    ax2.set_title('Performance Summary (Sorted by Return)', 
                 fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = 'backtest_results'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(output_dir, f'models_comparison_{timestamp}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to: {filepath}")
    
    plt.show()
    plt.close()


def _create_ohlc_with_trades_plot(df_ohlc, trades_df, model_name):
    """
    Create OHLC candlestick chart with trade entry and exit markers.
    
    Parameters:
    -----------
    df_ohlc : pd.DataFrame
        DataFrame with OHLC data (must have: timestamp, open, high, low, close, volume)
    trades_df : pd.DataFrame
        DataFrame with trades (must have: entry_time, exit_time, entry_price, exit_price, net_pnl)
    model_name : str
        Name of the model for the title
    """
    if df_ohlc is None or len(df_ohlc) == 0:
        print("⚠ No OHLC data to plot")
        return
    
    if trades_df is None or len(trades_df) == 0:
        print("⚠ No trades to plot")
        return
    
    # Prepare OHLC data for mplfinance
    df_plot = df_ohlc.copy()
    
    # Ensure timestamp is datetime and set as index
    if 'timestamp' in df_plot.columns:
        df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'])
        df_plot.set_index('timestamp', inplace=True)
    elif not isinstance(df_plot.index, pd.DatetimeIndex):
        df_plot.index = pd.to_datetime(df_plot.index)
    
    # Ensure column names are capitalized for mplfinance
    column_mapping = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df_plot.columns:
            df_plot.rename(columns={old_col: new_col}, inplace=True)
    
    # Prepare trade markers
    entry_markers = []
    exit_markers = []
    
    for _, trade in trades_df.iterrows():
        entry_time = pd.to_datetime(trade['entry_time'])
        exit_time = pd.to_datetime(trade['exit_time'])
        
        # Find closest timestamps in OHLC data
        if entry_time in df_plot.index:
            entry_price = trade['entry_price']
            entry_markers.append((entry_time, entry_price))
        
        if exit_time in df_plot.index:
            exit_price = trade['exit_price']
            is_win = trade.get('net_pnl', 0) > 0
            exit_markers.append((exit_time, exit_price, is_win))
    
    # Create additional plots for markers
    apds = []
    
    # Entry markers (green triangles pointing up)
    if entry_markers:
        entry_times = [m[0] for m in entry_markers]
        entry_prices = [m[1] for m in entry_markers]
        entry_scatter = mpf.make_addplot(
            pd.Series([np.nan] * len(df_plot), index=df_plot.index),
            type='scatter',
            markersize=100,
            marker='^',
            color='green',
            secondary_y=False
        )
        # Add entry points manually
        for time, price in entry_markers:
            if time in df_plot.index:
                idx = df_plot.index.get_loc(time)
                entry_data = pd.Series([np.nan] * len(df_plot), index=df_plot.index)
                entry_data.iloc[idx] = price
                apds.append(mpf.make_addplot(entry_data, type='scatter', markersize=100, 
                                            marker='^', color='lime', secondary_y=False))
    
    # Exit markers (red for losses, blue for wins)
    for time, price, is_win in exit_markers:
        if time in df_plot.index:
            idx = df_plot.index.get_loc(time)
            exit_data = pd.Series([np.nan] * len(df_plot), index=df_plot.index)
            exit_data.iloc[idx] = price
            color = 'dodgerblue' if is_win else 'red'
            apds.append(mpf.make_addplot(exit_data, type='scatter', markersize=100,
                                        marker='v', color=color, secondary_y=False))
    
    # Create the plot
    fig, axes = mpf.plot(
        df_plot,
        type='candle',
        style='charles',
        title=f'OHLC Chart with Trades - {model_name.replace("_", " ").title()}',
        ylabel='Price',
        volume=True,
        addplot=apds if apds else None,
        figsize=(16, 10),
        returnfig=True,
        warn_too_much_data=len(df_plot) + 1  # Suppress warning
    )
    
    # Add legend
    ax = axes[0]
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='lime', 
               markersize=10, label='Entry'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='dodgerblue',
               markersize=10, label='Exit (Win)'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='red',
               markersize=10, label='Exit (Loss)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # Add trade statistics text
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['net_pnl'] > 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    stats_text = f'Total Trades: {total_trades}  |  Wins: {winning_trades}  |  Win Rate: {win_rate:.1f}%'
    ax.text(0.5, 0.98, stats_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', horizontalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save plot
    output_dir = 'backtest_results'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(output_dir, f'ohlc_trades_{model_name}_{timestamp}.png')
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ OHLC chart saved to: {filepath}")
    
    plt.close(fig)


def main_train(features_method='crypto'):
    """Main execution function."""
    
    print("="*80)
    print("ML PREDICTION SYSTEM - TRAINING & TESTING")
    print("="*80)
    
    # ==================== STEP 1: LOAD DATA ====================
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    
    print(f"Loading training data from: {PATH_TRAIN}")
    df_train = pd.read_csv(PATH_TRAIN)
    print(f"✓ Training data loaded: {len(df_train):,} rows")
    
    print(f"Loading test data from: {PATH_TEST}")
    df_test = pd.read_csv(PATH_TEST)
    print(f"✓ Test data loaded: {len(df_test):,} rows")
    
    # ==================== STEP 2: GENERATE FEATURES ====================
    print("\n" + "="*80)
    print("STEP 2: GENERATING FEATURES")
    print("="*80)
    
    fg = FeaturesGenerator()
    top_features = ['atr_pct_28', 'atr_pct_14', 'volatility_20h', 'hl_spread_pct', 'volatility_50h', 'shadow_ratio', 'volatility_5h', 'volatility_10h', 'volume_skew_50', 'vpt', 'volume_skew_10', 'volume_skew_20',
        'price_to_ema_200', 'momentum_pct_48h', 'price_to_ema_100', 'momentum_pct_3h', 'price_to_sma_100', 'bb_width_20', 'bb_width_50', 'price_to_ema_20', 'volume_sma_5', 'stoch_14', 'price_to_sma_200',
        'price_to_ema_50', 'volume_sma_50', 'stoch_signal_28', 'mfi_14', 'price_to_sma_10', 'momentum_pct_6h','price_to_ema_10']
    
    print("Generating features...")
    features_generated = fg.generate_features(df_train, method=features_method)
    df_train_features = features_generated['X_train'][top_features]
    df_train_target = features_generated['y_train']
    df_test_features = features_generated['X_test'][top_features]
    df_test_target = features_generated['y_test']
    
    print(f"✓ Features generated: {len(df_train_features.columns)} columns")
    
    df_train_with_target = pd.concat([df_train_features, df_train_target], axis=1)
    df_test_with_target = pd.concat([df_test_features, df_test_target], axis=1)
    
    # Drop NaN rows
    df_train_with_target = df_train_with_target.dropna()
    df_test_with_target = df_test_with_target.dropna()
    
    print(f"✓ Features prepared")
    print(f"  Training samples: {len(df_train_with_target):,}")
    print(f"  Test samples: {len(df_test_with_target):,}")
    
    # Separate features and target
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target', 'pct_change_15']
    feature_cols = [col for col in df_train_with_target.columns if col not in exclude_cols]
    
    X_train = df_train_with_target[feature_cols]
    y_train = df_train_with_target['target']
    X_test = df_test_with_target[feature_cols]
    y_test = df_test_with_target['target']
    
    print(f"✓ Features prepared: {len(feature_cols)} features")
    
    # ==================== STEP 4: CREATE MODELS ====================
    print("\n" + "="*80)
    print("STEP 4: CREATING MODELS")
    print("="*80)
    
    models_manager = ModelsManager(models_dir='models')
    
    # Show configuration
    print("\nEnabled models:")
    enabled = models_manager.get_enabled_models()
    for model_name in enabled:
        print(f"  ✓ {model_name}")
    
    # Create models
    models = models_manager.create_models(enabled_only=True)
    
    # ==================== STEP 5: TRAIN MODELS ====================
    print("\n" + "="*80)
    print("STEP 5: TRAINING MODELS")
    print("="*80)
    
    trainer = Trainer(
        use_smote=True,           # Apply SMOTE for imbalanced data
        optimize_threshold=True,  # Optimize probability threshold
        use_scaler=True          # Scale features
    )
    
    trained_models, scaler, train_results, best_model_name = trainer.train(
        models=models,
        X_train=X_train,
        y_train=y_train
    )
    
    # Print training results
    trainer.print_results()
    
    # ==================== STEP 6: SAVE MODELS ====================
    print("\n" + "="*80)
    print("STEP 6: SAVING MODELS")
    print("="*80)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    saved_paths = models_manager.save_models(
        models=trained_models,
        scaler=scaler,
        suffix=timestamp
    )
    
    print(f"\n✓ Models saved with timestamp: {timestamp}")
    
    # ==================== STEP 7: TEST MODELS ====================
    print("\n" + "="*80)
    print("STEP 7: TESTING MODELS")
    print("="*80)
    
    tester = Tester(scaler=scaler)
    
    # Extract optimal thresholds from training
    optimal_thresholds = {
        name: results['optimal_threshold'] 
        for name, results in train_results.items()
    }
    
    test_results = tester.test(
        models=trained_models,
        X_test=X_test,
        y_test=y_test,
        optimal_thresholds=optimal_thresholds
    )
    
    # Print test results
    tester.print_results()
    
    # Print detailed report for best model
    best_test_model = tester.get_best_model_name()
    print(f"\nDetailed report for best model: {best_test_model}")
    tester.print_detailed_report(best_test_model, y_test, target_names=['No Rise', 'Rise'])
    
    # ==================== STEP 8: GENERATE REPORTS ====================
    print("\n" + "="*80)
    print("STEP 8: GENERATING REPORTS")
    print("="*80)
    
    report_manager = ReportManager(output_dir='reports')
    
    # Create full report with all visualizations
    full_report = report_manager.export_full_report(
        train_results=train_results,
        test_results=test_results,
        y_test=y_test,
        filename=f"ml_report_{timestamp}",
        target_names=['No Rise', 'Rise']
    )
    
    # ==================== STEP 9: MONITOR HEALTH ====================
    print("\n" + "="*80)
    print("STEP 9: SETTING UP HEALTH MONITORING")
    print("="*80)
    
    health_manager = HealthManager(
        performance_threshold=0.05,  # 5% degradation threshold
        time_threshold_days=30       # Retrain after 30 days
    )
    
    # Set baseline for best model
    best_model_metrics = test_results[best_test_model]['metrics']
    
    health_manager.set_baseline(
        model_name=best_test_model,
        metrics=best_model_metrics,
        timestamp=datetime.now()
    )
    
    print(f"\n✓ Health baseline set for {best_test_model}")
    print(f"  Performance threshold: 5%")
    print(f"  Time threshold: 30 days")
    
    # ==================== SUMMARY ====================
    print("\n" + "="*80)
    print("TRAINING AND TESTING COMPLETE!")
    print("="*80)
    
    print(f"\n📊 Training Summary:")
    print(f"  • Best training model: {best_model_name}")
    print(f"  • Models trained: {len(trained_models)}")
    print(f"  • Total training time: {trainer.training_time:.2f}s ({trainer.training_time/60:.2f} min)")
    print(f"  • Average time per model: {trainer.training_time/len(trained_models):.2f}s")
    
    print(f"\n📈 Testing Summary:")
    print(f"  • Best test model: {best_test_model}")
    print(f"  • Test Accuracy: {best_model_metrics['accuracy']:.4f}")
    print(f"  • Test F1 Score: {best_model_metrics['f1']:.4f}")
    print(f"  • Test Precision: {best_model_metrics['precision']:.4f}")
    print(f"  • Test Recall: {best_model_metrics['recall']:.4f}")
    
    print(f"\n💾 Saved Files:")
    print(f"  • Models: models/ (timestamp: {timestamp})")
    print(f"  • Reports: reports/")
    print(f"    - Training report (CSV + PNG)")
    print(f"    - Test report (CSV + PNG)")
    print(f"    - Comparison report (CSV + PNG)")
    
    print(f"\n🏥 Health Monitoring:")
    print(f"  • Baseline set for {best_test_model}")
    print(f"  • Monitor regularly for performance degradation")
    print(f"  • Retrain when health check recommends")
    
    print(f"\n🚀 Next Steps:")
    print(f"  1. Review reports in 'reports/' folder")
    print(f"  2. Check visualizations (PNG files)")
    print(f"  3. Run backtesting:")
    print(f"     python examples/backtest_quick_start.py")
    print(f"  4. Monitor model health regularly")
    print(f"  5. Use best model for predictions")
    
    print("\n" + "="*80 + "\n")

def main_backtest(features_method='crypto'):
    """Main execution function for backtesting."""
    # Load models
    models_manager = ModelsManager(PATH_MODELS)
    models, scaler, metadata = models_manager.load_models('latest')

    # Backtest
    df_test = pd.read_csv(PATH_TEST)

    # Generate features
    print("Generating features...")
    fg = FeaturesGenerator()
    top_features = ['atr_pct_28', 'atr_pct_14', 'volatility_20h', 'hl_spread_pct', 'volatility_50h', 'shadow_ratio', 'volatility_5h', 'volatility_10h', 'volume_skew_50', 'vpt', 'volume_skew_10', 'volume_skew_20',
        'price_to_ema_200', 'momentum_pct_48h', 'price_to_ema_100', 'momentum_pct_3h', 'price_to_sma_100', 'bb_width_20', 'bb_width_50', 'price_to_ema_20', 'volume_sma_5', 'stoch_14', 'price_to_sma_200',
        'price_to_ema_50', 'volume_sma_50', 'stoch_signal_28', 'mfi_14', 'price_to_sma_10', 'momentum_pct_6h','price_to_ema_10']
    features_response = fg.generate_features(df_test, method=features_method)
    df_test_features = features_response['df'][['timestamp', 'open', 'high', 'low', 'close', 'volume'] + top_features]
    
    # Validate data
    print(f"\nData shape: {df_test_features.shape}")
    print(f"Date range: {df_test_features['timestamp'].min()} to {df_test_features['timestamp'].max()}")
    
    # Check for NaN/inf values
    nan_counts = df_test_features.isna().sum()
    if nan_counts.any():
        print(f"\n⚠ Warning: Found NaN values:")
        print(nan_counts[nan_counts > 0])
    
    # Drop rows with NaN in OHLCV columns
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    df_test_features = df_test_features.dropna(subset=ohlcv_cols)
    print(f"\nData after cleaning: {df_test_features.shape}")
    
    # Run backtests
    print(f"\n{'='*80}")
    print(f"RUNNING BACKTESTS FOR {len(models)} MODELS")
    print(f"{'='*80}\n")
    
    #backtester = BacktestBacktraderML()
    backtester = BacktestNoLib()
    #backtester = BacktestBacktestingML()
    
    # Store results for all models
    all_results = {}
    all_equity_curves = {}
    
    for i, (model_name, model) in enumerate(models.items(), 1):
        print(f"\n{'='*80}")
        print(f"BACKTEST {i}/{len(models)}: {model_name.upper()}")
        print(f"{'='*80}")
        
        results, trades = backtester.run_backtest(
            df = df_test_features,
            model = model,
            scaler = scaler,
            X_columns = top_features,
            probability_threshold = 0.6,
            trailing_stop_pct = 2.0,
            take_profit_pct = None,
            position_size_pct = 1.0,
            plot = False,  # Disable plotting to avoid memory issues
            printlog = False
        )
        
        # Store results
        all_results[model_name] = results
        
        # Debug: Print all result keys
        print(f"\n  Debug - Result keys for {model_name}: {list(results.keys())}")
        
        # Store equity curve with validation
        if 'equity_curve' in results:
            equity_curve = results['equity_curve']
            print(f"  Debug - Equity curve type: {type(equity_curve)}")
            if equity_curve is not None:
                if hasattr(equity_curve, '__len__'):
                    print(f"  Debug - Equity curve length: {len(equity_curve)}")
                    if len(equity_curve) > 0:
                        all_equity_curves[model_name] = equity_curve
                        print(f"  ✓ Stored equity curve: {len(equity_curve)} points")
                    else:
                        print(f"  ⚠ Equity curve is empty for {model_name}")
                else:
                    print(f"  ⚠ Equity curve has no length attribute")
            else:
                print(f"  ⚠ Equity curve is None for {model_name}")
        else:
            print(f"  ⚠ No equity_curve in results for {model_name}")
        
        # Print results
        print(f"\nResults for {model_name}:")
        print(f"  Final Value: ${results['final_value']:,.2f}")
        print(f"  Total Return: {results['total_return_pct']:.2f}%")
        print(f"  Total Trades: {results['total_trades']}")
        print(f"  Win Rate: {results['win_rate']:.2f}%")
        print(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {results.get('max_drawdown', 0):.2f}%")
        
        # Create visualizations
        print(f"\nCreating visualizations for {model_name}...")
        backtester.create_comprehensive_visualizations(trades)
        
        # Create OHLC chart with trades
        _create_ohlc_with_trades_plot(df_test_features, trades, model_name)
    
    # Create final comparison plot
    print(f"\n{'='*80}")
    print(f"CREATING FINAL COMPARISON PLOT")
    print(f"{'='*80}")
    _create_models_comparison_plot(all_results, all_equity_curves)


if __name__ == "__main__":
    #main_train()
    main_backtest()
