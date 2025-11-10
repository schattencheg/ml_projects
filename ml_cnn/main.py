import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
from src.data_loader import load_es_and_btc_data, align_data_on_dates
from src.features_generator import FeaturesGenerator
from src.trend_predictor import TrendPredictor

def main():
    """
    Main function to run the complete pipeline:
    1. Load EC and BTC data from Yahoo Finance
    2. Generate features from OHLC data
    3. Train trend prediction model with profit threshold
    4. Make predictions
    """
    print("Starting trend prediction pipeline...")
    
    # Load EC and BTC data
    print("Loading EC and BTC data from Yahoo Finance...")
    es_data, btc_data = load_es_and_btc_data(start_date="2000-01-01", end_date="2025-11-10")
    
    # Align data on common dates
    print(f"EC data shape: {es_data.shape}")
    print(f"BTC data shape: {btc_data.shape}")
    es_data, btc_data = align_data_on_dates(es_data, btc_data)
    print(f"After alignment - EC data shape: {es_data.shape}, BTC data shape: {btc_data.shape}")
    
    # Initialize features generator
    features_gen = FeaturesGenerator()
    
    # For this example, let's use BTC data for prediction
    # You can modify this to combine EC and BTC data as needed
    ohlc_data = btc_data[['Open', 'High', 'Low', 'Close']].copy()
    price_series = btc_data['Close']
    
    # Generate features
    print("Generating features from OHLC data...")
    features_df = features_gen.generate_features(ohlc_data)
    
    # Initialize trend predictor with profit threshold
    profit_threshold = 0.02  # 2% profit threshold
    trend_predictor = TrendPredictor(profit_threshold=profit_threshold, sequence_length=10)
    
    # Train the model
    print("Training trend prediction model...")
    trend_predictor.train(
        features_df=features_df,
        price_series=price_series,
        optimize_hyperparams=True,  # Set to True to use Optuna optimization
        n_trials=10  # Number of optimization trials
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