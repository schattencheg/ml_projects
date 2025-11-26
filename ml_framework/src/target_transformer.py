"""
Target Transformer - Converts targets between three-class (-1,0,+1) and ML format (0,1,2).
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Literal

# Deep learning model identifiers
_DL_MODULES = ('keras', 'tensorflow', 'torch', 'pytorch')
_DL_CLASSES = ('SimpleCNN', 'DeepCNN', 'ResidualCNN')


class TargetTransformer:
    """Transforms targets between three-class (-1,0,+1) and classic ML (0,1,2) formats."""
    
    @staticmethod
    def to_ml_format(y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Convert -1,0,+1 to 0,1,2 format."""
        return np.array(y, dtype=int) + 1
    
    @staticmethod
    def to_three_class(y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Convert 0,1,2 to -1,0,+1 format."""
        return np.array(y, dtype=int) - 1
    
    @staticmethod
    def is_deep_learning(model) -> bool:
        """Check if model is a deep learning model."""
        module = model.__class__.__module__.lower()
        name = model.__class__.__name__
        return any(m in module for m in _DL_MODULES) or name in _DL_CLASSES
    
    def get_model_type(self, model) -> Literal['classic_ml', 'deep_learning']:
        """Detect model type."""
        return 'deep_learning' if self.is_deep_learning(model) else 'classic_ml'
    
    def transform(self, y: Union[np.ndarray, pd.Series], model) -> Tuple[np.ndarray, str]:
        """Transform targets for model. Returns (transformed_y, model_type)."""
        model_type = self.get_model_type(model)
        if model_type == 'classic_ml':
            return self.to_ml_format(y), model_type
        return np.array(y, dtype=int), model_type
    
    def inverse_transform(self, y_pred: Union[np.ndarray, pd.Series], model_type: str) -> np.ndarray:
        """Transform predictions back to three-class format."""
        if model_type == 'classic_ml':
            return self.to_three_class(y_pred)
        return np.array(y_pred, dtype=int)


# Singleton instance
_transformer = TargetTransformer()

def get_target_transformer() -> TargetTransformer:
    """Get global TargetTransformer instance."""
    return _transformer

def transform_for_model(y: Union[np.ndarray, pd.Series], model, verbose: bool = False) -> Tuple[np.ndarray, str]:
    """Transform targets for the given model. Returns (transformed_y, model_type)."""
    return _transformer.transform(y, model)

def inverse_transform_predictions(y_pred: Union[np.ndarray, pd.Series], model_type: str) -> np.ndarray:
    """Transform predictions back to three-class format (-1, 0, +1)."""
    return _transformer.inverse_transform(y_pred, model_type)
