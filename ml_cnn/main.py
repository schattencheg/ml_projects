import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
from src.data_loader import load_data, align_data_on_dates
from src.features_generator import FeaturesGenerator
from src.trend_predictor import TrendPredictor


tickers = ['BTC-USD', 'ES']
tickers = ['ES']
models = []

def main():
    """
    Main function to run the complete pipeline:
    1. Load EC and BTC data from Yahoo Finance
    2. Generate features from OHLC data
    3. Train trend prediction model with profit threshold
    4. Make predictions
    """
    print("Starting trend prediction pipeline...")
    
    # Initialize features generator
    features_gen = FeaturesGenerator()

    print("Loading data from Yahoo Finance...")
    for ticker in tickers:
        data = load_data(ticker, start_date="2000-01-01", end_date="2025-11-10")
    
        # Align data on common dates
        print(f"{ticker} data shape: {data.shape}")
        data = align_data_on_dates(data)
        print(f"After alignment - {ticker} data shape: {data.shape}")
    
    
        # For this example, let's use BTC data for prediction
        # You can modify this to combine EC and BTC data as needed
        #ohlc_data = data[['timestamp', 'open', 'high', 'low', 'close']].copy()
        price_series = data['close']
        
        # Generate features
        print("Generating features from OHLC data...")
        features_df, feature_cols = features_gen.generate_features(data)
    
        # Initialize trend predictor with profit threshold
        profit_threshold = 0.02  # 2% profit threshold
        future_period = 15
        trend_predictor = TrendPredictor(profit_threshold=profit_threshold, sequence_length=10)
    
        # Train the model
        print("Training trend prediction model...")
        trend_predictor.train(
            features_df=features_df,
            price_series=price_series,
            optimize_hyperparams=True,  # Set to True to use Optuna optimization
            n_trials=10,  # Number of optimization trials
            future_period=future_period,
            threshold=profit_threshold
        )
    
        # Make predictions
        print("Making predictions...")
        predictions, probabilities = trend_predictor.predict_with_probability(features_df)
        
        # Print results summary
        print(f"Number of predictions: {len(predictions)}")
        print(f"Up trend predictions: {np.sum(predictions)} ({np.mean(predictions)*100:.2f}%)")
        print(f"Average prediction probability: {np.mean(probabilities):.3f}")
        
        # Create a results DataFrame
        results_df = pd.DataFrame({
            'Date': price_series.index[-len(predictions):],
            'Prediction': predictions,
            'Probability': probabilities,
            'Actual_Price': price_series.values[-len(predictions):]
        })
    
        # Display first few predictions
        print("\nFirst 10 predictions:")
        print(results_df.head(10))
        
        print("\nTrend prediction pipeline completed successfully!")


if __name__ == "__main__":
    main()
