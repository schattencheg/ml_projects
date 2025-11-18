"""
LinearRegressionModel - Linear regression and logistic regression wrappers.
"""

import numpy as np
from typing import Optional
from .base_model import BaseModel
from sklearn.linear_model import LogisticRegression, LinearRegression


class LinearRegressionModel(BaseModel):
    """
    Linear regression model with automatic target conversion.
    Note: For classification, use LogisticRegressionModel instead.
    """
    
    def __init__(self, name: str = "LinearRegression", **params):
        """
        Initialize Linear Regression model.
        
        Args:
            name: Model name
            **params: LinearRegression parameters
        """
        super().__init__(name)
        
        default_params = {
            'n_jobs': -1
        }
        default_params.update(params)
        
        self.params = default_params
        self.model = None
        
    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit Linear Regression model."""
        self.model = LinearRegression(**self.params)
        self.model.fit(X, y, **kwargs)
        
    def _predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Make predictions and round to nearest integer."""
        predictions = self.model.predict(X, **kwargs)
        return np.round(predictions).astype(int)
    
    def get_coefficients(self) -> Optional[np.ndarray]:
        """
        Get model coefficients.
        
        Returns:
            Coefficient array
        """
        if self.model is None:
            return None
        return self.model.coef_


class LogisticRegressionModel(BaseModel):
    """
    Logistic regression model with automatic target conversion.
    """
    
    def __init__(self, name: str = "LogisticRegression", **params):
        """
        Initialize Logistic Regression model.
        
        Args:
            name: Model name
            **params: LogisticRegression parameters
        """
        super().__init__(name)
        
        default_params = {
            'max_iter': 1000,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(params)
        
        self.params = default_params
        self.model = None
        
    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit Logistic Regression model."""
        self.model = LogisticRegression(**self.params)
        self.model.fit(X, y, **kwargs)
        
    def _predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X, **kwargs)
    
    def _predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X, **kwargs)
    
    def get_coefficients(self) -> Optional[np.ndarray]:
        """
        Get model coefficients.
        
        Returns:
            Coefficient array
        """
        if self.model is None:
            return None
        return self.model.coef_


class RandomForestModel(BaseModel):
    """
    Random Forest classifier with automatic target conversion.
    """
    
    def __init__(self, name: str = "RandomForest", **params):
        """
        Initialize Random Forest model.
        
        Args:
            name: Model name
            **params: RandomForestClassifier parameters
        """
        super().__init__(name)
        
        from sklearn.ensemble import RandomForestClassifier
        
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(params)
        
        self.params = default_params
        self.model = None
        
    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit Random Forest model."""
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X, y, **kwargs)
        
    def _predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X, **kwargs)
    
    def _predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X, **kwargs)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores.
        
        Returns:
            Feature importance array
        """
        if self.model is None:
            return None
        return self.model.feature_importances_
