import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import os
import json
import pickle
from datetime import datetime


def ensure_directory_exists(path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary
    
    Args:
        path: Path to the directory
    """
    os.makedirs(path, exist_ok=True)


def save_dict_to_json(data: Dict, filepath: str) -> None:
    """
    Save a dictionary to a JSON file
    
    Args:
        data: Dictionary to save
        filepath: Path to save the file
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_dict_from_json(filepath: str) -> Dict:
    """
    Load a dictionary from a JSON file
    
    Args:
        filepath: Path to the file to load
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_model_results(results: Dict, model_name: str, results_dir: str = "results/models") -> None:
    """
    Save model results to a file
    
    Args:
        results: Results dictionary to save
        model_name: Name of the model
        results_dir: Directory to save results
    """
    ensure_directory_exists(results_dir)
    filepath = os.path.join(results_dir, f"{model_name}_results.json")
    save_dict_to_json(results, filepath)


def calculate_direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the accuracy of direction prediction (up/down)
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Direction prediction accuracy
    """
    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    
    if len(true_direction) == 0 or len(pred_direction) == 0:
        return 0.0
    
    # Take the minimum length to compare
    min_len = min(len(true_direction), len(pred_direction))
    accuracy = np.mean(true_direction[:min_len] == pred_direction[:min_len])
    
    return accuracy


def create_rolling_window_data(X: np.ndarray, y: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create rolling window data for time series prediction
    
    Args:
        X: Feature matrix
        y: Target values
        window_size: Size of the rolling window
        
    Returns:
        Tuple of (X_rolled, y_rolled) with rolling window data
    """
    if len(X) < window_size:
        return X, y
    
    X_rolled = []
    y_rolled = []
    
    for i in range(len(X) - window_size + 1):
        X_rolled.append(X[i:i+window_size])
        y_rolled.append(y[i+window_size-1])  # Predict the last value in the window
    
    return np.array(X_rolled), np.array(y_rolled)


def format_results_for_display(results: List[Dict]) -> pd.DataFrame:
    """
    Format results for display in a readable DataFrame
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Formatted DataFrame
    """
    if results is None or (isinstance(results, pd.DataFrame) and results.empty):
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Round numeric columns for readability
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].round(6)
    
    return df