"""
XGBoostModel - XGBoost classifier wrapper.
"""

import numpy as np
from typing import Optional
from .base_model import BaseModel

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class XGBoostModel(BaseModel):
    """
    XGBoost classifier with automatic target conversion.
    """
    
    def __init__(self, name: str = "XGBoost", use_gpu: bool = False, **params):
        """
        Initialize XGBoost model.
        
        Args:
            name: Model name
            use_gpu: Whether to use GPU acceleration
            **params: XGBoost parameters
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
        
        super().__init__(name)
        
        # Default parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Add GPU parameters if requested
        if use_gpu:
            default_params.update({
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'predictor': 'gpu_predictor'
            })
            print(f"✓ {name}: GPU acceleration enabled")
        
        default_params.update(params)
        
        self.params = default_params
        self.model = None
        self.use_gpu = use_gpu
        
    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit XGBoost model."""
        self.model = xgb.XGBClassifier(**self.params)
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
