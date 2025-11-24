"""
CatBoostModel - CatBoost classifier wrapper.
"""

import numpy as np
from typing import Optional
from src.models_lib.base_model import BaseModel

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class CatBoostModel(BaseModel):
    """
    CatBoost classifier with automatic target conversion.
    """
    
    def __init__(self, name: str = "CatBoost", use_gpu: bool = False, **params):
        """
        Initialize CatBoost model.
        
        Args:
            name: Model name
            use_gpu: Whether to use GPU acceleration
            **params: CatBoost parameters
        """
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not installed. Install with: pip install catboost")
        
        super().__init__(name)
        
        # Default parameters
        default_params = {
            'iterations': 100,
            'depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbose': False,
            'thread_count': -1
        }
        
        # Add GPU parameters if requested
        if use_gpu:
            default_params.update({
                'task_type': 'GPU',
                'devices': '0'
            })
            print(f"✓ {name}: GPU acceleration enabled")
        
        default_params.update(params)
        
        self.params = default_params
        self.model = None
        self.use_gpu = use_gpu
        
    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit CatBoost model."""
        self.model = CatBoostClassifier(**self.params)
        self.model.fit(X, y, **kwargs)
        
    def _predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X, **kwargs).flatten().astype(int)
    
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
        return self.model.get_feature_importance()
