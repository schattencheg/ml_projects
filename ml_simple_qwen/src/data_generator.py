"""
Data Generator Module

This module creates synthetic time series data for training and testing models.
It simulates real-world scenarios where data comes in sequentially.
"""

import numpy as np
import pandas as pd


class DataGenerator:
    """
    Generates synthetic time series data for ML model training and evaluation.
    
    The data simulates a scenario where we have features that influence a target value,
    with some added noise to make it realistic. This serves as a substitute for 
    real-world streaming data.
    """
    
    def __init__(self, seed=42):
        """
        Initialize the data generator with a random seed for reproducibility.
        
        Args:
            seed (int): Random seed for reproducible results
        """
        np.random.seed(seed)
        self.seed = seed
        
    def generate_linear_data(self, n_samples=1000, n_features=3, noise_level=0.1):
        """
        Generate linear relationship data with some noise.
        
        Args:
            n_samples (int): Number of data points to generate
            n_features (int): Number of feature columns
            noise_level (float): Level of noise to add (0.0 = no noise)
            
        Returns:
            pd.DataFrame: Generated data with features and target
        """
        # Generate random features
        X = np.random.randn(n_samples, n_features)
        
        # Create a linear relationship: y = w0*x0 + w1*x1 + w2*x2 + ... + noise
        weights = np.random.uniform(-2, 2, n_features)  # Random weights
        y = X @ weights + np.random.normal(0, noise_level, n_samples)
        
        # Combine into a DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        data = pd.DataFrame(X, columns=feature_names)
        data['target'] = y
        
        return data
    
    def generate_nonlinear_data(self, n_samples=1000, n_features=3, noise_level=0.1):
        """
        Generate non-linear relationship data (e.g., polynomial or sinusoidal).
        
        Args:
            n_samples (int): Number of data points to generate
            n_features (int): Number of feature columns
            noise_level (float): Level of noise to add (0.0 = no noise)
            
        Returns:
            pd.DataFrame: Generated data with features and target
        """
        # Generate random features
        X = np.random.randn(n_samples, n_features)
        
        # Create a non-linear relationship
        y = (X[:, 0] * X[:, 1] + 
             np.sin(X[:, 2]) * 0.5 + 
             X[:, 0]**2 * 0.3 + 
             np.random.normal(0, noise_level, n_samples))
        
        # Combine into a DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        data = pd.DataFrame(X, columns=feature_names)
        data['target'] = y
        
        return data
    
    def stream_data(self, data, batch_size=10):
        """
        Generator that yields data in batches to simulate streaming.
        
        Args:
            data (pd.DataFrame): The dataset to stream
            batch_size (int): Number of samples per batch
            
        Yields:
            pd.DataFrame: Batch of data
        """
        n_samples = len(data)
        for i in range(0, n_samples, batch_size):
            yield data.iloc[i:i+batch_size].copy()


# Example usage
if __name__ == "__main__":
    # Create a data generator
    generator = DataGenerator()
    
    # Generate some linear data
    print("Generating linear data...")
    linear_data = generator.generate_linear_data(n_samples=500, n_features=4)
    print(f"Linear data shape: {linear_data.shape}")
    print(f"First 5 rows:\n{linear_data.head()}")
    
    # Generate some non-linear data
    print("\nGenerating non-linear data...")
    nonlinear_data = generator.generate_nonlinear_data(n_samples=300, n_features=3)
    print(f"Non-linear data shape: {nonlinear_data.shape}")
    print(f"First 5 rows:\n{nonlinear_data.head()}")
    
    # Demonstrate streaming
    print("\nStreaming data in batches of 50...")
    for i, batch in enumerate(generator.stream_data(linear_data, batch_size=50)):
        print(f"Batch {i+1} shape: {batch.shape}")
        if i >= 2:  # Just show first 3 batches
            break