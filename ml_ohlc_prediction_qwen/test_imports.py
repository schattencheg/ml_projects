import sys
import os
sys.path.append('src')

from src.data_loader import DataLoader
from src.models.lstm_model import LSTMModel
from src.models.random_forest_model import RandomForestModel
from src.models.linear_model import LinearModel
from src.training import YearlyModelTrainer
from src.evaluation import evaluate_model
from src.utils import format_results_for_display
import pandas as pd
import numpy as np

print('All modules imported successfully')

# Check if data exists
data_path = 'data/raw/ohlc_data.csv'
if os.path.exists(data_path):
    print(f'Data file exists at {data_path}')
    loader = DataLoader(data_path)
    data = loader.load_data()
    print(f'Loaded {len(data)} rows of data')
    print('First few rows:')
    print(data.head())
else:
    print('Data file not found at expected location')