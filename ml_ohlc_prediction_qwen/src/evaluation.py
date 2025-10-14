import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from .models.base_model import BaseModel


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics for predictions
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }


def evaluate_model(model: BaseModel, X_test: np.ndarray, y_test: np.ndarray, 
                   model_name: str = None) -> Dict[str, float]:
    """
    Evaluate a trained model on test data
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        model_name: Name of the model for reporting
        
    Returns:
        Dictionary of evaluation metrics
    """
    if model_name is None:
        model_name = model.name
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    # Add model name to metrics
    metrics['Model'] = model_name
    
    return metrics


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                     title: str = "Actual vs Predicted Values", 
                     figsize: Tuple[int, int] = (12, 6)):
    """
    Plot actual vs predicted values
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Actual vs Predicted scatter
    axes[0].scatter(y_true, y_pred, alpha=0.5)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual')
    axes[0].set_ylabel('Predicted')
    axes[0].set_title('Actual vs Predicted')
    
    # Plot 2: Time series comparison
    axes[1].plot(y_true, label='Actual', alpha=0.7)
    axes[1].plot(y_pred, label='Predicted', alpha=0.7)
    axes[1].set_xlabel('Time Index')
    axes[1].set_ylabel('Value')
    axes[1].set_title('Time Series Comparison')
    axes[1].legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_model_comparison(results: List[Dict[str, float]], 
                         title: str = "Model Performance Comparison",
                         figsize: Tuple[int, int] = (12, 8)):
    """
    Plot comparison of different models' performance
    
    Args:
        results: List of dictionaries with model metrics
        title: Plot title
        figsize: Figure size
    """
    df_results = pd.DataFrame(results)
    
    # Create a figure with subplots for different metrics
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    metrics_to_plot = ['RMSE', 'MAE', 'R2', 'MAPE']
    
    for i, metric in enumerate(metrics_to_plot):
        row = i // 2
        col = i % 2
        
        ax = axes[row, col]
        
        # Create horizontal bar plot
        models = df_results['Model']
        values = df_results[metric]
        
        ax.barh(models, values)
        ax.set_xlabel(metric)
        ax.set_title(f'{metric} Comparison')
        
        # Add value labels on bars
        for j, v in enumerate(values):
            ax.text(v, j, f'{v:.4f}', va='center', fontsize=9)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()