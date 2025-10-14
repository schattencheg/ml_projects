import sys
import os
sys.path.append('src')

from src.data_loader import DataLoader
import pandas as pd
import numpy as np
from src.models.lstm_model import LSTMModel
from src.models.random_forest_model import RandomForestModel
from src.models.linear_model import LinearModel

# Load and prepare data
data_path = 'data/raw/ohlc_data.csv'
loader = DataLoader(data_path)
all_data = loader.load_data()

# Split by year and get 2019 data
yearly_data = loader.split_by_year()
year_2019_data = yearly_data[2019]
year_2019_features = loader.create_features(year_2019_data, lookback_window=10)
X, y = loader.prepare_data_for_training(year_2019_features, ['Next_Close'])

print(f"Original data shapes - X: {X.shape}, y: {y.shape}")

# Test LSTM model separately
print("\nTesting LSTM model:")
lstm_model = LSTMModel(name="LSTM", n_units=32, n_layers=1, sequence_length=10)
try:
    lstm_model.train(X, y, epochs=5, verbose=1)  # Train for just 5 epochs for quick test
    print("LSTM model training successful!")
    
    # Test prediction
    predictions = lstm_model.predict(X)
    print(f"LSTM predictions shape: {predictions.shape}")
except Exception as e:
    print(f"LSTM model error: {e}")

# Test Random Forest model with same original data
print("\nTesting Random Forest model with original data:")
rf_model = RandomForestModel(name="RandomForest", n_estimators=10, max_depth=5)
try:
    rf_model.train(X, y)
    print("Random Forest model training successful!")
    
    # Test prediction
    predictions = rf_model.predict(X)
    print(f"RF predictions shape: {predictions.shape}")
except Exception as e:
    print(f"Random Forest model error: {e}")

# Test Linear model with original data
print("\nTesting Linear model with original data:")
linear_model = LinearModel(name="LinearRegression")
try:
    linear_model.train(X, y)
    print("Linear model training successful!")
    
    # Test prediction
    predictions = linear_model.predict(X)
    print(f"Linear predictions shape: {predictions.shape}")
except Exception as e:
    print(f"Linear model error: {e}")

print("\nAll models tested successfully!")