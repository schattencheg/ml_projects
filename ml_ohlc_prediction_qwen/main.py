#!/usr/bin/env python3
"""
Main execution script for ML OHLC Price Prediction project.

This script:
1. Loads OHLC data
2. Trains multiple models with yearly retraining
3. Evaluates model performance
4. Generates visualizations and reports
"""
import sys
import os
from typing import List

# Add src to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import DataLoader
from src.models.lstm_model import LSTMModel
from src.models.random_forest_model import RandomForestModel
from src.models.linear_model import LinearModel
from src.training import YearlyModelTrainer
from src.visualization import (visualize_yearly_performance, 
                               compare_model_performance, 
                               plot_yearly_trends, 
                               generate_performance_report)
from src.utils import format_results_for_display


def main():
    """
    Main execution function
    """
    print("ML OHLC Price Prediction Project")
    print("=" * 40)
    
    # Define the path to the data file
    # Note: You'll need to update this path to point to your actual data file
    data_path = "data/raw/ohlc_data.csv"
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please place your OHLC data in the specified location.")
        print("Expected format: CSV with columns [Date, Open, High, Low, Close, Volume]")
        return
    
    # Initialize models to compare
    models = [
        LSTMModel(name="LSTM", n_units=50, n_layers=2, sequence_length=10),
        RandomForestModel(name="RandomForest", n_estimators=100, max_depth=10),
        LinearModel(name="LinearRegression")
    ]
    
    print("Initialized models:")
    for model in models:
        print(f"  - {model.name}")
    
    # Create the yearly model trainer
    trainer = YearlyModelTrainer(data_path, models)
    
    print(f"\nStarting yearly training and evaluation...")
    
    # Train and evaluate models yearly
    try:
        yearly_results = trainer.train_and_evaluate_yearly(lookback_window=10)
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return
    
    print(f"\nTraining completed!")
    
    # Get results as DataFrame
    results_df = trainer.get_yearly_summary()
    
    if not results_df.empty:
        print("\nYearly Results Summary:")
        print(format_results_for_display(results_df))
        
        # Save results
        trainer.save_results()
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        
        # 1. Yearly performance heatmap
        visualize_yearly_performance(results_df)
        
        # 2. Model comparison
        compare_model_performance(results_df)
        
        # 3. Yearly trends
        plot_yearly_trends(results_df)
        
        # 4. Generate performance report
        generate_performance_report(results_df)
        
        print("\nAll visualizations and reports have been generated!")
        print("Check the 'results' directory for outputs.")
    else:
        print("No results to display - training may have failed or had insufficient data")


def create_sample_data():
    """
    Create sample OHLC data for demonstration purposes
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    print("Creating sample OHLC data...")
    
    # Create dates for 5 years of daily data
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate synthetic OHLC data
    n = len(dates)
    np.random.seed(42)  # For reproducibility
    
    # Start with an initial price
    prices = [100.0]  # Starting price
    
    for i in range(1, n):
        # Random walk with some volatility
        change_percent = np.random.normal(0.0005, 0.02)  # 0.05% average daily return, 2% std dev
        new_price = prices[-1] * (1 + change_percent)
        prices.append(new_price)
    
    # Add some volatility for high/low
    opens = []
    highs = []
    lows = []
    closes = []
    
    for i in range(n):
        if i == 0:
            open_price = prices[i]
        else:
            # Open price is yesterday's close
            open_price = closes[-1] if closes else prices[i]
        
        close_price = prices[i]
        
        # Determine high and low based on open/close and some random variation
        price_range = abs(close_price - open_price) * np.random.uniform(0.5, 2.0)
        high_price = max(open_price, close_price) + np.random.uniform(0, price_range)
        low_price = min(open_price, close_price) - np.random.uniform(0, price_range)
        
        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        closes.append(close_price)
    
    # Create the DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': np.random.uniform(1000000, 5000000, n).astype(int)  # Random volumes
    })
    
    # Save to CSV
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/ohlc_data.csv', index=False)
    
    print(f"Sample data created with {n} rows and saved to data/raw/ohlc_data.csv")
    print("First 5 rows:")
    print(df.head())


if __name__ == "__main__":
    # Check if sample data exists, create if not
    if not os.path.exists("data/raw/ohlc_data.csv"):
        print("Sample data not found. Creating sample OHLC data...")
        create_sample_data()
    
    main()