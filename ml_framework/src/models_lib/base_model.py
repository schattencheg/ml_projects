"""
BaseModel - Abstract base class for all ML models with automatic target conversion.
"""

import numpy as np
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Base class for ML models with automatic target conversion (-1/0/1 ↔ 0/1/2).
    
    Subclasses must implement: _fit(), _predict()
    Optional override: _predict_proba()
    """
    
    def __init__(self, name: str = "BaseModel"):
        self.name = name
        self.model = None
        self.target_mapping = None
        self.reverse_mapping = None
        self.is_fitted = False
    
    def _convert_targets(self, y: np.ndarray) -> np.ndarray:
        """Convert targets to sequential integers (0, 1, 2, ...)."""
        unique = sorted(np.unique(y))
        self.target_mapping = {v: i for i, v in enumerate(unique)}
        self.reverse_mapping = {i: v for v, i in self.target_mapping.items()}
        return np.array([self.target_mapping[v] for v in y])
    
    def _reverse_predictions(self, y_pred: np.ndarray) -> np.ndarray:
        """Convert predictions back to original target values."""
        if self.reverse_mapping is None:
            return y_pred
        return np.array([self.reverse_mapping[v] for v in y_pred])
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseModel':
        """Fit model with automatic target conversion."""
        self._fit(X, self._convert_targets(y), **kwargs)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict with automatic reverse conversion to original target space."""
        if not self.is_fitted:
            raise ValueError(f"Model {self.name} is not fitted")
        return self._reverse_predictions(self._predict(X, **kwargs))
    
    def predict_proba(self, X: np.ndarray, **kwargs) -> Optional[np.ndarray]:
        """Predict class probabilities (if supported)."""
        if not self.is_fitted:
            raise ValueError(f"Model {self.name} is not fitted")
        return self._predict_proba(X, **kwargs)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score."""
        return np.mean(self.predict(X) == y)
    
    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Internal fit - implement in subclass."""
        pass
    
    @abstractmethod
    def _predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Internal predict - implement in subclass."""
        pass
    
    def _predict_proba(self, X: np.ndarray, **kwargs) -> Optional[np.ndarray]:
        """Internal predict_proba - override in subclass if supported."""
        return None
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.model.get_params() if hasattr(self.model, 'get_params') else {}
    
    def set_params(self, **params) -> 'BaseModel':
        """Set model parameters."""
        if hasattr(self.model, 'set_params'):
            self.model.set_params(**params)
        return self
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
