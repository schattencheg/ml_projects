"""
CatBoostModel - CatBoost classifier wrapper.
"""

import numpy as np
from typing import Optional
from src.models_lib.base_model import BaseModel

try:
    from catboost import CatBoostClassifier
    CB_AVAILABLE = True
except ImportError:
    CB_AVAILABLE = False


class CatBoostModel(BaseModel):
    """CatBoost classifier with automatic target conversion."""
    
    DEFAULT_PARAMS = {
        'iterations': 100, 'depth': 6, 'learning_rate': 0.1,
        'random_state': 42, 'verbose': False, 'thread_count': -1
    }
    GPU_PARAMS = {'task_type': 'GPU', 'devices': '0'}
    
    def __init__(self, name: str = "CatBoost", use_gpu: bool = False, **params):
        if not CB_AVAILABLE:
            raise ImportError("CatBoost required. Install: pip install catboost")
        super().__init__(name)
        
        self.params = {**self.DEFAULT_PARAMS, **(self.GPU_PARAMS if use_gpu else {}), **params}
        self.use_gpu = use_gpu
    
    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        self.model = CatBoostClassifier(**self.params)
        self.model.fit(X, y, **kwargs)
    
    def _predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        return self.model.predict(X, **kwargs).flatten().astype(int)
    
    def _predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        return self.model.predict_proba(X, **kwargs)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores."""
        return self.model.get_feature_importance() if self.model else None
