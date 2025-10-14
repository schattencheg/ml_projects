"""
Sklearn Linear Model

This module implements a linear regression model using sklearn as an example 
of integrating external ML libraries into our system.
"""

import numpy as np
from ..model_manager import BaseModel

# Try to import sklearn, with a fallback if not available
try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. SklearnLinearModel will not work.")


class SklearnLinearModel(BaseModel):
    """
    Linear regression model using sklearn implementation.
    
    This serves as an example of how to integrate external ML libraries
    into our model management system while maintaining the BaseModel interface.
    """
    
    def __init__(self):
        """
        Initialize the sklearn linear model.
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for SklearnLinearModel but is not installed")
        
        self.model = LinearRegression()
        self.is_trained = False
    
    def train(self, X, y):
        """
        Train the sklearn linear model on the given data.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target values
        """
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X):
        """
        Make predictions using the trained sklearn model.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Predicted values
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def get_params(self):
        """
        Return sklearn model parameters for logging.
        
        Returns:
            dict: Model coefficients and intercept from sklearn
        """
        if not self.is_trained:
            return {'coefficients': None, 'intercept': None}
        
        return {
            'coefficients': self.model.coef_.tolist(),
            'intercept': self.model.intercept_
        }