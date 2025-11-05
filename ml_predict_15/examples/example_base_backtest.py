"""
Example usage of the BaseBacktest class with enhanced reporting and visualization.

This script demonstrates how to use the new base class functionality
for detailed trade analysis and comprehensive visualizations.
"""

import pandas as pd
import numpy as np
from src.BacktestNoLib import BacktestNoLib
from src.BaseBacktest import BaseBacktest
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Mock data generator for demonstration
def generate_mock_data(n_samples=1000):
    """Generate mock OHLCV data with features for testing."""
    np.random.seed(42)
    
    # Generate dates
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_samples)]
    
    # Generate price data with trend and noise
    trend = np.linspace(100, 150, n_samples)
    noise = np.random.normal(0, 2, n_samples)
    close_prices = trend + noise + np.random.normal(0, 1, n_samples)
    
    # Generate OHLC from close
    open_prices = close_prices + np.random.normal(0, 0.5, n_samples)
    high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.normal(0, 1, n_samples))
    low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.normal(0, 1, n_samples))
    volumes = np.random.randint(1000, 10000, n_samples)
    
    # Generate some technical indicators as features
    sma_5 = pd.Series(close_prices).rolling(5).mean().fillna(method='bfill')
    sma_20 = pd.Series(close_prices).rolling(20).mean().fillna(method='bfill')
    rsi = np.random.uniform(20, 80, n_samples)  # Mock RSI
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes,
        'sma_5': sma_5,
        'sma_20': sma_20,
        'rsi': rsi,
        'price_change': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])  # Target
    })
    
    return df

# Mock model class for demonstration
class MockModel:
    """Mock ML model for demonstration."""
    
    def predict(self, X):
        """Generate mock predictions."""
        # Simple strategy: buy when RSI < 30, sell when RSI > 70
        predictions = []
        for i in range(len(X)):
            rsi = X[i, 2]  # Assuming RSI is the 3rd feature
            if rsi < 30:
                predictions.append(1)  # Buy signal
            else:
                predictions.append(0)  # No signal
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Generate mock probabilities."""
        predictions = self.predict(X)
        probabilities = []
        for pred in predictions:
            if pred == 1:
                prob = np.random.uniform(0.6, 0.9)  # High confidence for buy
                probabilities.append([1-prob, prob])
            else:
                prob = np.random.uniform(0.1, 0.4)  # Low confidence for no action
                probabilities.append([1-prob, prob])
        return np.array(probabilities)

# Mock scaler class
class MockScaler:
    """Mock scaler for demonstration."""
    
    def transform(self, X):
        """Mock scaling - just return normalized data."""
        return (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

def demonstrate_base_backtest():
    """Demonstrate the enhanced backtest functionality."""
    
    print("="*80)
    print("ENHANCED BACKTEST DEMONSTRATION")
    print("="*80)
    
    # Generate mock data
    print("\n1. Generating mock data...")
    df = generate_mock_data(500)
    print(f"   Generated {len(df)} data points")
    
    # Create mock model and scaler
    model = MockModel()
    scaler = MockScaler()
    feature_columns = ['sma_5', 'sma_20', 'rsi']
    
    # Initialize backtest
    print("\n2. Initializing backtest...")
    backtest = BacktestNoLib(
        initial_capital=10000,
        position_size=0.95,
        trailing_stop_pct=3.0,
        commission=0.001,
        probability_threshold=0.65
    )
    
    # Run backtest
    print("\n3. Running backtest...")
    results, trades_df = backtest.run_backtest(
        df=df,
        model=model,
        scaler=scaler,
        X_columns=feature_columns,
        plot=False,
        printlog=False
    )
    
    # Print summary using base class method
    print("\n4. Results Summary:")
    backtest.print_summary(results)
    
    # Generate detailed report
    print("\n5. Generating detailed report...")
    report = backtest.generate_detailed_report(results)
    print("\nDetailed Report Generated!")
    
    # Create comprehensive visualizations
    print("\n6. Creating visualizations...")
    try:
        viz_files = backtest.create_comprehensive_visualizations(
            results=results,
            df=df,
            save_dir="backtest_results",
            show_plots=False  # Set to True to display plots
        )
        
        print("   Visualizations created:")
        for viz_name, file_path in viz_files.items():
            if file_path:
                print(f"   - {viz_name}: {file_path}")
    except Exception as e:
        print(f"   Warning: Could not create all visualizations: {e}")
    
    # Export results
    print("\n7. Exporting results...")
    try:
        exported_files = backtest.export_results(
            results=results,
            export_dir="backtest_results"
        )
        
        print("   Files exported:")
        for file_type, file_path in exported_files.items():
            print(f"   - {file_type}: {file_path}")
    except Exception as e:
        print(f"   Warning: Could not export all files: {e}")
    
    # Display key metrics
    print("\n8. Key Performance Metrics:")
    print(f"   - Total Trades: {results.get('total_trades', 0)}")
    print(f"   - Win Rate: {results.get('win_rate', 0):.1f}%")
    print(f"   - Total Return: {results.get('total_return_pct', 0):.2f}%")
    print(f"   - Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
    print(f"   - Max Drawdown: {results.get('max_drawdown', 0):.2f}%")
    
    if results.get('total_trades', 0) > 0:
        print(f"   - Average Win: ${results.get('avg_win', 0):.2f}")
        print(f"   - Average Loss: ${results.get('avg_loss', 0):.2f}")
        print(f"   - Profit Factor: {results.get('profit_factor', 0):.2f}")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nThe BaseBacktest class provides:")
    print("✓ Comprehensive performance metrics calculation")
    print("✓ Detailed trade analysis and reporting")
    print("✓ Advanced visualization suite")
    print("✓ Export functionality for results and charts")
    print("✓ Consistent interface across all backtest implementations")
    
    return results, trades_df

if __name__ == "__main__":
    # Run the demonstration
    results, trades = demonstrate_base_backtest()
    
    print(f"\nDemo completed! Check the 'backtest_results' folder for exported files.")
