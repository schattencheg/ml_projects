"""
XGBoostModel - XGBoost classifier wrapper.
"""

import numpy as np
from typing import Optional
from src.models_lib.base_model import BaseModel

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


class XGBoostModel(BaseModel):
    """XGBoost classifier with automatic target conversion."""
    
    DEFAULT_PARAMS = {
        'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1,
        'random_state': 42, 'n_jobs': -1
    }
    GPU_PARAMS = {'tree_method': 'gpu_hist', 'gpu_id': 0, 'predictor': 'gpu_predictor'}
    
    def __init__(self, name: str = "XGBoost", use_gpu: bool = False, **params):
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost required. Install: pip install xgboost")
        super().__init__(name)
        
        self.params = {**self.DEFAULT_PARAMS, **(self.GPU_PARAMS if use_gpu else {}), **params}
        self.use_gpu = use_gpu
    
    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y, **kwargs)
    
    def _predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        return self.model.predict(X, **kwargs)
    
    def _predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        return self.model.predict_proba(X, **kwargs)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores."""
        return self.model.feature_importances_ if self.model else None
