"""
BaseModel - Base class for all ML models with automatic target conversion.
"""

import numpy as np
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Base class for all ML models.
    
    Features:
    - Automatic target conversion from -1/0/1 to 0/1/2 for compatibility
    - Unified interface for all models (sklearn, XGBoost, CatBoost, DL)
    - Automatic reverse conversion of predictions
    """
    
    def __init__(self, name: str = "BaseModel"):
        """
        Initialize BaseModel.
        
        Args:
            name: Name of the model
        """
        self.name = name
        self.model = None
        self.target_mapping = None  # Maps original targets to converted targets
        self.reverse_mapping = None  # Maps converted targets back to original
        self.is_fitted = False
        
    def _convert_targets(self, y: np.ndarray) -> np.ndarray:
        """
        Convert targets to sequential integers starting from 0.
        Handles -1/0/1 -> 0/1/2 conversion and other mappings.
        
        Args:
            y: Original target array
            
        Returns:
            Converted target array
        """
        unique_values = np.unique(y)
        
        # Create mapping from original to sequential integers
        self.target_mapping = {val: idx for idx, val in enumerate(sorted(unique_values))}
        self.reverse_mapping = {idx: val for val, idx in self.target_mapping.items()}
        
        # Convert targets
        y_converted = np.array([self.target_mapping[val] for val in y])
        
        return y_converted
    
    def _reverse_convert_predictions(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Convert predictions back to original target values.
        
        Args:
            y_pred: Predicted values (as sequential integers)
            
        Returns:
            Predictions in original target space
        """
        if self.reverse_mapping is None:
            return y_pred
        
        # Convert predictions back
        y_original = np.array([self.reverse_mapping[val] for val in y_pred])
        
        return y_original
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseModel':
        """
        Fit the model with automatic target conversion.
        
        Args:
            X: Feature matrix
            y: Target array
            **kwargs: Additional arguments for the specific model
            
        Returns:
            Self
        """
        # Convert targets
        y_converted = self._convert_targets(y)
        
        # Fit the underlying model
        self._fit(X, y_converted, **kwargs)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make predictions with automatic reverse conversion.
        
        Args:
            X: Feature matrix
            **kwargs: Additional arguments for the specific model
            
        Returns:
            Predictions in original target space
        """
        if not self.is_fitted:
            raise ValueError(f"Model {self.name} is not fitted yet")
        
        # Get predictions from underlying model
        y_pred = self._predict(X, **kwargs)
        
        # Convert back to original target space
        y_original = self._reverse_convert_predictions(y_pred)
        
        return y_original
    
    def predict_proba(self, X: np.ndarray, **kwargs) -> Optional[np.ndarray]:
        """
        Predict class probabilities if supported.
        
        Args:
            X: Feature matrix
            **kwargs: Additional arguments for the specific model
            
        Returns:
            Class probabilities or None if not supported
        """
        if not self.is_fitted:
            raise ValueError(f"Model {self.name} is not fitted yet")
        
        return self._predict_proba(X, **kwargs)
    
    def score(self, X: np.ndarray, y: np.ndarray, **kwargs) -> float:
        """
        Calculate accuracy score.
        
        Args:
            X: Feature matrix
            y: True target values
            **kwargs: Additional arguments
            
        Returns:
            Accuracy score
        """
        y_pred = self.predict(X, **kwargs)
        return np.mean(y_pred == y)
    
    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        Internal fit method to be implemented by subclasses.
        
        Args:
            X: Feature matrix
            y: Converted target array (0, 1, 2, ...)
            **kwargs: Additional arguments
        """
        pass
    
    @abstractmethod
    def _predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Internal predict method to be implemented by subclasses.
        
        Args:
            X: Feature matrix
            **kwargs: Additional arguments
            
        Returns:
            Predictions as sequential integers (0, 1, 2, ...)
        """
        pass
    
    def _predict_proba(self, X: np.ndarray, **kwargs) -> Optional[np.ndarray]:
        """
        Internal predict_proba method to be implemented by subclasses.
        Default implementation returns None.
        
        Args:
            X: Feature matrix
            **kwargs: Additional arguments
            
        Returns:
            Class probabilities or None
        """
        return None
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dictionary of parameters
        """
        if hasattr(self.model, 'get_params'):
            return self.model.get_params()
        return {}
    
    def set_params(self, **params):
        """
        Set model parameters.
        
        Args:
            **params: Parameters to set
        """
        if hasattr(self.model, 'set_params'):
            self.model.set_params(**params)
        return self
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
