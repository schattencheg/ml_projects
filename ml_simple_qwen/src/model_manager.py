"""
Model Manager Module

This module handles the registration, initialization, and management of different ML models.
It provides a clean interface for training, predicting, and managing multiple models.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    Abstract base class for all ML models in the system.
    Ensures all models have a consistent interface.
    
    Educational Notes:
    - Using an abstract base class enforces a consistent interface across all models
    - This makes it easy to swap different algorithms without changing the training loop
    - The ABC pattern ensures all models implement required methods
    """
    
    @abstractmethod
    def train(self, X, y):
        """
        Train the model on the given data.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target values
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Make predictions on the given data.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Predicted values
        """
        pass
    
    @abstractmethod
    def get_params(self):
        """
        Return model-specific parameters for logging purposes.
        
        Returns:
            dict: Model parameters
        """
        pass


class ModelManager:
    """
    Manages multiple ML models, handles registration, training, and prediction.
    
    This class provides a unified interface to work with different models,
    making it easy to compare their performance and manage them in the 
    continuous learning pipeline.
    """
    
    def __init__(self):
        """
        Initialize the model manager with an empty registry.
        """
        self.models = {}  # Dictionary to hold model instances
        self.model_classes = {}  # Dictionary to hold model classes
    
    def register_model(self, name, model_class):
        """
        Register a new model class with the manager.
        
        Args:
            name (str): Name to identify the model
            model_class (BaseModel): Class of the model to register
        """
        if not issubclass(model_class, BaseModel):
            raise ValueError(f"Model class must inherit from BaseModel")
        
        self.model_classes[name] = model_class
        print(f"Registered model: {name}")
    
    def create_model(self, name, **kwargs):
        """
        Create an instance of a registered model.
        
        Args:
            name (str): Name of the registered model
            **kwargs: Arguments to pass to the model constructor
            
        Returns:
            BaseModel: Initialized model instance
        """
        if name not in self.model_classes:
            raise ValueError(f"Model '{name}' not registered")
        
        # Create an instance of the model
        model_instance = self.model_classes[name](**kwargs)
        self.models[name] = model_instance
        print(f"Created model instance: {name}")
        return model_instance
    
    def get_model(self, name):
        """
        Get a model instance by name.
        
        Args:
            name (str): Name of the model
            
        Returns:
            BaseModel: Model instance
        """
        if name not in self.models:
            raise ValueError(f"Model '{name}' not created yet")
        
        return self.models[name]
    
    def train_model(self, name, X, y):
        """
        Train a specific model on the provided data.
        
        Args:
            name (str): Name of the model to train
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target values
        """
        model = self.get_model(name)
        model.train(X, y)
        print(f"Trained model: {name}")
    
    def predict_model(self, name, X):
        """
        Make predictions using a specific model.
        
        Args:
            name (str): Name of the model to use
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Predicted values
        """
        model = self.get_model(name)
        predictions = model.predict(X)
        return predictions
    
    def get_available_models(self):
        """
        Get a list of available registered models.
        
        Returns:
            list: List of model names
        """
        return list(self.model_classes.keys())
    
    def get_trained_models(self):
        """
        Get a list of models that have been created/initialized.
        
        Returns:
            list: List of model names that have instances
        """
        return list(self.models.keys())


# Example implementation of a simple linear model
class LinearModel(BaseModel):
    """
    Simple linear regression model using normal equation.
    This serves as an example implementation of the BaseModel interface.
    """
    
    def __init__(self):
        """
        Initialize the linear model with no coefficients yet.
        """
        self.coefficients = None
        self.intercept = None
    
    def train(self, X, y):
        """
        Train the linear model using the normal equation: w = (X^T * X)^(-1) * X^T * y
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target values
        """
        # Add bias column (intercept) to features
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Calculate coefficients using normal equation
        try:
            XtX_inv = np.linalg.inv(X_with_bias.T @ X_with_bias)
            coefficients = XtX_inv @ X_with_bias.T @ y
            
            # Separate intercept and feature coefficients
            self.intercept = coefficients[0]
            self.coefficients = coefficients[1:]
        except np.linalg.LinAlgError:
            # Handle singular matrix case (regularized solution)
            reg_strength = 1e-6
            XtX_reg = X_with_bias.T @ X_with_bias + reg_strength * np.eye(X_with_bias.shape[1])
            XtX_inv = np.linalg.inv(XtX_reg)
            coefficients = XtX_inv @ X_with_bias.T @ y
            
            self.intercept = coefficients[0]
            self.coefficients = coefficients[1:]
    
    def predict(self, X):
        """
        Make predictions using the trained coefficients.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Predicted values
        """
        if self.coefficients is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Calculate predictions: y = X * coefficients + intercept
        predictions = X @ self.coefficients + self.intercept
        return predictions
    
    def get_params(self):
        """
        Return model parameters.
        
        Returns:
            dict: Model coefficients and intercept
        """
        return {
            'coefficients': self.coefficients.tolist() if self.coefficients is not None else None,
            'intercept': self.intercept
        }


# Example implementation of a dummy model for testing
class DummyModel(BaseModel):
    """
    A simple dummy model that predicts the mean of training targets.
    Useful for testing and as a baseline model.
    """
    
    def __init__(self):
        """
        Initialize the dummy model with no stored mean.
        """
        self.mean_value = None
    
    def train(self, X, y):
        """
        Train the dummy model by storing the mean of the target values.
        
        Args:
            X (np.ndarray): Feature matrix (ignored for dummy model)
            y (np.ndarray): Target values
        """
        self.mean_value = np.mean(y)
    
    def predict(self, X):
        """
        Make predictions by returning the stored mean value.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Predicted values (all the same mean value)
        """
        if self.mean_value is None:
            raise ValueError("Model must be trained before making predictions")
        
        return np.full(X.shape[0], self.mean_value)
    
    def get_params(self):
        """
        Return model parameters.
        
        Returns:
            dict: Mean value used for predictions
        """
        return {'mean_value': self.mean_value}


# Example usage and testing
if __name__ == "__main__":
    # Create a model manager
    manager = ModelManager()
    
    # Register some models
    manager.register_model('linear', LinearModel)
    manager.register_model('dummy', DummyModel)
    
    # Show available models
    print("Available models:", manager.get_available_models())
    
    # Create model instances
    linear_model = manager.create_model('linear')
    dummy_model = manager.create_model('dummy')
    
    # Show created models
    print("Created models:", manager.get_trained_models())
    
    # Generate some test data
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + 0.5*np.random.randn(100)  # Linear relationship with noise
    
    # Train the models
    manager.train_model('linear', X, y)
    manager.train_model('dummy', X, y)
    
    # Make predictions
    test_X = X[:5]  # First 5 samples as test
    linear_pred = manager.predict_model('linear', test_X)
    dummy_pred = manager.predict_model('dummy', test_X)
    
    print(f"Linear model predictions: {linear_pred}")
    print(f"Dummy model predictions: {dummy_pred}")
    print(f"Actual values: {y[:5]}")