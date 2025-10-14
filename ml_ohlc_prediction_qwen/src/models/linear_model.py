import numpy as np
import joblib
from typing import Any, Dict
from sklearn.linear_model import LinearRegression
from .base_model import BaseModel


class LinearModel(BaseModel):
    """
    Linear Regression for OHLC prediction
    """
    
    def __init__(self, name: str = "LinearRegression", 
                 fit_intercept: bool = True,
                 normalize: bool = False):
        super().__init__(name)
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.model = LinearRegression(fit_intercept=fit_intercept)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> 'LinearModel':
        """
        Train the Linear Regression model
        
        Args:
            X_train: Training features
            y_train: Training targets
            **kwargs: Additional arguments
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the Linear Regression model
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk
        
        Args:
            path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        joblib.dump(self.model, path)
    
    def load_model(self, path: str) -> 'LinearModel':
        """
        Load a trained model from disk
        
        Args:
            path: Path to load the model from
            
        Returns:
            The loaded model instance
        """
        self.model = joblib.load(path)
        self.is_trained = True
        return self