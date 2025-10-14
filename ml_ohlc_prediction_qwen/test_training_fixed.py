import sys
import os
sys.path.append('src')

from src.data_loader import DataLoader
from src.models.lstm_model import LSTMModel
from src.models.random_forest_model import RandomForestModel
from src.models.linear_model import LinearModel
from src.evaluation import evaluate_model
import pandas as pd
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
    
    # Check shapes before training
    print(f"  Before training - X: {X.shape}, y: {y.shape}")
    
    try:
        if model.name == "LSTM":
            # Train the LSTM model with epochs parameter
            trained_model = model.train(X, y, epochs=3)
        else:
            # Train other models without epochs parameter
            trained_model = model.train(X, y)
        
        # Evaluate the model
        metrics = evaluate_model(trained_model, X, y, model.name)
        print(f"  {model.name} - RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}")
        print(f"  Training completed successfully!")
        
    except Exception as e:
        print(f"  Error training {model.name}: {e}")
        import traceback
        traceback.print_exc()

print("\nTest completed!")