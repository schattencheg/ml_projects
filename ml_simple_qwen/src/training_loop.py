"""
Training Loop Module

This module implements the main continuous training loop that:
1. Trains models on historical data
2. Evaluates models on new data
3. Retrains models every N values
4. Collects metrics throughout the process
"""

import numpy as np
import pandas as pd
from .model_manager import ModelManager
from .data_generator import DataGenerator
from .metrics_collector import MetricsCollector


class ContinuousTrainingLoop:
    """
    Implements the main continuous training loop for ML models.
    
    This class orchestrates the process of:
    - Using historical data to train models
    - Evaluating models on new incoming data
    - Retraining models periodically (every N values)
    - Collecting metrics throughout the process
    
    Educational Notes:
    - Continuous learning is crucial when data distributions change over time (concept drift)
    - Regular retraining helps models adapt to new patterns in the data
    - Performance metrics help track model degradation and the effectiveness of retraining
    - Batch processing is efficient for handling streaming data
    """
    
    def __init__(self, model_manager, metrics_collector):
        """
        Initialize the continuous training loop.
        
        Args:
            model_manager (ModelManager): Manager for ML models
            metrics_collector (MetricsCollector): Collector for performance metrics
        """
        self.model_manager = model_manager
        self.metrics_collector = metrics_collector
        self.step_count = 0
    
    def run_continuous_training(self, data, initial_train_size, retrain_interval, 
                               batch_size=10, feature_cols=None, target_col='target'):
        """
        Run the continuous training process.
        
        In a real-world scenario, this would work with streaming data that arrives continuously.
        Here we simulate this by splitting the dataset into an initial training set and 
        subsequent streaming data.
        
        Args:
            data (pd.DataFrame): Complete dataset with features and target
            initial_train_size (int): Number of samples to use for initial training
            retrain_interval (int): How often to retrain models (every N samples)
            batch_size (int): Size of data batches for evaluation
            feature_cols (list): List of feature column names (if None, use all except target)
            target_col (str): Name of the target column
        """
        if feature_cols is None:
            feature_cols = [col for col in data.columns if col != target_col]
        
        print(f"Starting continuous training...")
        print(f"- Initial training size: {initial_train_size}")
        print(f"- Retrain interval: {retrain_interval}")
        print(f"- Batch size: {batch_size}")
        print(f"- Features: {feature_cols}")
        print(f"- Target: {target_col}")
        
        # Split data into initial training set and streaming data
        initial_train_data = data.iloc[:initial_train_size].copy()
        streaming_data = data.iloc[initial_train_size:].copy().reset_index(drop=True)
        
        print(f"- Initial training samples: {len(initial_train_data)}")
        print(f"- Streaming samples: {len(streaming_data)}")
        
        # Train all models on initial data
        self._train_all_models(initial_train_data, feature_cols, target_col)
        
        # Process streaming data in batches
        for batch_start in range(0, len(streaming_data), batch_size):
            batch_end = min(batch_start + batch_size, len(streaming_data))
            batch = streaming_data.iloc[batch_start:batch_end].copy()
            
            # Evaluate models on the batch
            self._evaluate_models(batch, feature_cols, target_col)
            
            # Update step count
            self.step_count += 1
            
            # Retrain models if interval reached
            if self.step_count % (retrain_interval // batch_size) == 0:
                print(f"\n[STEP {self.step_count}] Retraining models...")
                
                # Prepare retraining data (recent samples)
                retrain_start = max(0, batch_end - retrain_interval)
                retrain_data = data.iloc[retrain_start:batch_end].copy()
                
                self._train_all_models(retrain_data, feature_cols, target_col)
        
        print(f"\nContinuous training completed after {self.step_count} steps.")
    
    def _train_all_models(self, train_data, feature_cols, target_col):
        """
        Train all available models on the given training data.
        
        Args:
            train_data (pd.DataFrame): Training data
            feature_cols (list): List of feature column names
            target_col (str): Name of the target column
        """
        X = train_data[feature_cols].values
        y = train_data[target_col].values
        
        print(f"Training {len(self.model_manager.get_trained_models())} models on {len(train_data)} samples...")
        
        for model_name in self.model_manager.get_trained_models():
            print(f"  Training {model_name}...")
            self.model_manager.train_model(model_name, X, y)
    
    def _evaluate_models(self, batch, feature_cols, target_col):
        """
        Evaluate all trained models on the given batch and record metrics.
        
        Args:
            batch (pd.DataFrame): Batch of data to evaluate on
            feature_cols (list): List of feature column names
            target_col (str): Name of the target column
        """
        X = batch[feature_cols].values
        y_true = batch[target_col].values
        
        print(f"Evaluating models on batch of {len(batch)} samples (Step {self.step_count+1})...")
        
        for model_name in self.model_manager.get_trained_models():
            # Get model predictions
            y_pred = self.model_manager.predict_model(model_name, X)
            
            # Record metrics
            model_params = self.model_manager.get_model(model_name).get_params()
            self.metrics_collector.record_metrics(
                model_name=model_name,
                y_true=y_true,
                y_pred=y_pred,
                step=self.step_count,
                additional_params=model_params
            )
            
            # Calculate and print current batch metrics
            mse = self.metrics_collector.calculate_mse(y_true, y_pred)
            mae = self.metrics_collector.calculate_mae(y_true, y_pred)
            r2 = self.metrics_collector.calculate_r2(y_true, y_pred)
            
            print(f"  {model_name}: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")


# Example usage and testing
if __name__ == "__main__":
    # Create required components
    model_manager = ModelManager()
    metrics_collector = MetricsCollector()
    data_generator = DataGenerator()
    
    # Register models
    from model_manager import LinearModel, DummyModel
    model_manager.register_model('linear', LinearModel)
    model_manager.register_model('dummy', DummyModel)
    
    # Create model instances
    model_manager.create_model('linear')
    model_manager.create_model('dummy')
    
    # Generate some data for testing
    print("Generating synthetic data...")
    data = data_generator.generate_linear_data(n_samples=200, n_features=3, noise_level=0.1)
    
    # Create the continuous training loop
    training_loop = ContinuousTrainingLoop(model_manager, metrics_collector)
    
    # Run the continuous training process
    training_loop.run_continuous_training(
        data=data,
        initial_train_size=50,      # Use first 50 samples for initial training
        retrain_interval=30,        # Retrain every 30 new samples
        batch_size=10,              # Process 10 samples at a time
        feature_cols=['feature_0', 'feature_1', 'feature_2'],
        target_col='target'
    )
    
    # Print final metrics summary
    metrics_collector.print_metrics_summary()
    
    # Show all collected metrics
    all_metrics = metrics_collector.get_all_metrics()
    print(f"\nTotal metrics records collected: {len(all_metrics)}")
    print("\nFirst few records:")
    print(all_metrics.head())
    
    print(f"\nLast few records:")
    print(all_metrics.tail())