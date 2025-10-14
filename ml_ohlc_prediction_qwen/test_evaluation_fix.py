import sys
import os
sys.path.append('src')

from src.data_loader import DataLoader
from src.models.lstm_model import LSTMModel
from src.models.random_forest_model import RandomForestModel
from src.models.linear_model import LinearModel
from src.evaluation import calculate_metrics
import numpy as np
from datetime import datetime

# Load and prepare data (mimicking YearlyModelTrainer)
data_path = 'data/raw/ohlc_data.csv'
loader = DataLoader(data_path)
all_data = loader.load_data()
yearly_data = loader.split_by_year()
year_2019_data = yearly_data[2019]
year_2019_features = loader.create_features(year_2019_data, lookback_window=10)
X, y = loader.prepare_data_for_training(year_2019_features, ['Next_Close'])

print(f"Original data shapes - X: {X.shape}, y: {y.shape}")

# Initialize models
models = [
    LSTMModel(name="LSTM", n_units=32, n_layers=1, sequence_length=10),
    RandomForestModel(name="RandomForest", n_estimators=10, max_depth=5),
    LinearModel(name="LinearRegression")
]

# Train each model on the same data (mimicking the loop in YearlyModelTrainer)
for model in models:
    print(f"\nTraining {model.name}...")
    
    try:
        if model.name == "LSTM":
            # Train the LSTM model with epochs parameter
            trained_model = model.train(X, y, epochs=3)
            
            # For LSTM, we need to make predictions on appropriately shaped data
            lstm_predictions = trained_model.predict(X)
            print(f"  LSTM predictions shape: {lstm_predictions.shape}")
            
            # LSTM predictions will have different size due to sequence reshaping
            # So we take the corresponding y values for evaluation
            y_eval = y[trained_model.sequence_length-1:]  # Adjust for sequence length
            print(f"  Adjusted y shape for evaluation: {y_eval.shape}")
            
            # Calculate metrics with adjusted arrays
            metrics = calculate_metrics(y_eval, lstm_predictions)
            print(f"  {model.name} - RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}")
            
        else:
            # Train other models without epochs parameter
            trained_model = model.train(X, y)
            
            # For other models, predict and evaluate with original data
            predictions = trained_model.predict(X)
            metrics = calculate_metrics(y, predictions)
            print(f"  {model.name} - RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}")
        
        print(f"  Training and evaluation completed successfully!")
        
    except Exception as e:
        print(f"  Error with {model.name}: {e}")
        import traceback
        traceback.print_exc()

print("\nTest completed!")