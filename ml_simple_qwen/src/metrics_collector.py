"""
Metrics Collector Module

This module collects and stores performance metrics for different ML models
during the continuous training and evaluation process. It maintains a dataframe
with metrics that can be analyzed after the process completes.
"""

import numpy as np
import pandas as pd
from datetime import datetime


class MetricsCollector:
    """
    Collects and manages performance metrics for ML models during training.
    
    This class stores various metrics like Mean Squared Error (MSE), 
    Mean Absolute Error (MAE), R-squared, and others for each model 
    at different time points during the continuous learning process.
    
    Educational Notes:
    - Metrics tracking is essential for monitoring model performance over time
    - Different metrics reveal different aspects of model performance
    - MSE penalizes large errors more heavily than MAE
    - R-squared indicates how much variance in the target is explained by the model
    """
    
    def __init__(self):
        """
        Initialize the metrics collector with an empty results dataframe.
        """
        self.results = []
        self.metrics_history = pd.DataFrame()
    
    def calculate_mse(self, y_true, y_pred):
        """
        Calculate Mean Squared Error between true and predicted values.
        
        Args:
            y_true (np.ndarray): True target values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            float: Mean Squared Error
        """
        return np.mean((y_true - y_pred) ** 2)
    
    def calculate_mae(self, y_true, y_pred):
        """
        Calculate Mean Absolute Error between true and predicted values.
        
        Args:
            y_true (np.ndarray): True target values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            float: Mean Absolute Error
        """
        return np.mean(np.abs(y_true - y_pred))
    
    def calculate_r2(self, y_true, y_pred):
        """
        Calculate R-squared (coefficient of determination).
        
        Args:
            y_true (np.ndarray): True target values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            float: R-squared value
        """
        ss_res = np.sum((y_true - y_pred) ** 2)  # Sum of squares of residuals
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # Total sum of squares
        r2 = 1 - (ss_res / ss_tot)
        return r2
    
    def calculate_rmse(self, y_true, y_pred):
        """
        Calculate Root Mean Squared Error.
        
        Args:
            y_true (np.ndarray): True target values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            float: Root Mean Squared Error
        """
        mse = self.calculate_mse(y_true, y_pred)
        return np.sqrt(mse)
    
    def record_metrics(self, model_name, y_true, y_pred, step, additional_params=None):
        """
        Record performance metrics for a model at a specific step.
        
        Args:
            model_name (str): Name of the model
            y_true (np.ndarray): True target values
            y_pred (np.ndarray): Predicted values
            step (int): Current step in the training process
            additional_params (dict): Additional parameters to record (e.g., model params)
        """
        if additional_params is None:
            additional_params = {}
        
        # Calculate metrics
        mse = self.calculate_mse(y_true, y_pred)
        mae = self.calculate_mae(y_true, y_pred)
        r2 = self.calculate_r2(y_true, y_pred)
        rmse = self.calculate_rmse(y_true, y_pred)
        
        # Create a record with all the metrics
        record = {
            'timestamp': datetime.now(),
            'step': step,
            'model_name': model_name,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': rmse,
            'n_samples': len(y_true)
        }
        
        # Add any additional parameters
        for key, value in additional_params.items():
            record[key] = value
        
        # Append to results list
        self.results.append(record)
        
        # Update the metrics history dataframe
        self.metrics_history = pd.DataFrame(self.results)
    
    def get_metrics_summary(self):
        """
        Get a summary of collected metrics by model.
        
        Returns:
            pd.DataFrame: Summary statistics by model
        """
        if self.metrics_history.empty:
            return pd.DataFrame()
        
        # Group by model name and calculate summary statistics
        summary = self.metrics_history.groupby('model_name').agg({
            'mse': ['mean', 'std', 'min', 'max'],
            'mae': ['mean', 'std', 'min', 'max'],
            'r2': ['mean', 'std', 'min', 'max'],
            'rmse': ['mean', 'std', 'min', 'max'],
            'step': ['count']
        }).round(4)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns]
        summary = summary.reset_index()
        
        return summary
    
    def get_all_metrics(self):
        """
        Get the complete dataframe of all collected metrics.
        
        Returns:
            pd.DataFrame: All collected metrics
        """
        return self.metrics_history.copy()
    
    def reset(self):
        """
        Reset the metrics collector to start fresh.
        """
        self.results = []
        self.metrics_history = pd.DataFrame()
    
    def save_metrics(self, filename):
        """
        Save the collected metrics to a CSV file.
        
        Args:
            filename (str): Path to save the metrics CSV file
        """
        self.metrics_history.to_csv(filename, index=False)
        print(f"Metrics saved to {filename}")
    
    def print_metrics_summary(self):
        """
        Print a formatted summary of the collected metrics.
        """
        if self.metrics_history.empty:
            print("No metrics recorded yet.")
            return
        
        print("\n" + "="*60)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*60)
        
        summary = self.get_metrics_summary()
        
        for _, row in summary.iterrows():
            model_name = row['model_name']
            print(f"\nModel: {model_name}")
            print(f"  Samples processed: {row['step_count']}")
            print(f"  MSE - Mean: {row['mse_mean']:.4f}, Std: {row['mse_std']:.4f}")
            print(f"  MAE - Mean: {row['mae_mean']:.4f}, Std: {row['mae_std']:.4f}")
            print(f"  R2  - Mean: {row['r2_mean']:.4f}, Std: {row['r2_std']:.4f}")
            print(f"  RMSE - Mean: {row['rmse_mean']:.4f}, Std: {row['rmse_std']:.4f}")
        
        print("\n" + "="*60)


# Example usage
if __name__ == "__main__":
    # Create a metrics collector
    collector = MetricsCollector()
    
    # Simulate some predictions and true values
    np.random.seed(42)
    y_true = np.random.randn(20)
    y_pred_model1 = y_true + 0.1*np.random.randn(20)  # Model 1: fairly accurate
    y_pred_model2 = y_true + 0.5*np.random.randn(20)  # Model 2: less accurate
    
    # Record metrics for model 1
    for step in range(5):
        start_idx = step * 4
        end_idx = (step + 1) * 4
        collector.record_metrics(
            model_name='model_1',
            y_true=y_true[start_idx:end_idx],
            y_pred=y_pred_model1[start_idx:end_idx],
            step=step
        )
    
    # Record metrics for model 2
    for step in range(5):
        start_idx = step * 4
        end_idx = (step + 1) * 4
        collector.record_metrics(
            model_name='model_2',
            y_true=y_true[start_idx:end_idx],
            y_pred=y_pred_model2[start_idx:end_idx],
            step=step
        )
    
    # Print summary
    collector.print_metrics_summary()
    
    # Show all metrics
    print("\nAll collected metrics:")
    print(collector.get_all_metrics())
    
    # Show summary
    print("\nSummary by model:")
    print(collector.get_metrics_summary())