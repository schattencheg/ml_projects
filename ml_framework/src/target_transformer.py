"""
Target Transformer - Automatic target transformation for different model types.

Handles conversion between:
- Three-class format: -1 (decrease), 0 (neutral), +1 (increase)
- Classic ML format: 0, 1, 2
- Deep learning format: -1, 0, +1
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Literal


class TargetTransformer:
    """
    Transforms targets between different formats for ML and DL models.
    
    Formats:
    - 'three_class': -1, 0, +1 (original format)
    - 'classic_ml': 0, 1, 2 (for sklearn models)
    - 'deep_learning': -1, 0, +1 (for neural networks)
    """
    
    def __init__(self):
        """Initialize target transformer."""
        self.original_format = 'three_class'
        
    def transform_for_classic_ml(self, y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Transform targets from three-class (-1, 0, +1) to classic ML format (0, 1, 2).
        
        Mapping:
        -1 (decrease) → 0
         0 (neutral)  → 1
        +1 (increase) → 2
        
        Args:
            y: Target array in three-class format (-1, 0, +1)
            
        Returns:
            Target array in classic ML format (0, 1, 2)
        """
        y_array = np.array(y)
        y_transformed = np.zeros_like(y_array, dtype=int)
        
        y_transformed[y_array == -1] = 0  # Decrease
        y_transformed[y_array == 0] = 1   # Neutral
        y_transformed[y_array == 1] = 2   # Increase
        
        return y_transformed
    
    def transform_for_deep_learning(self, y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Transform targets for deep learning (keeps three-class format).
        
        Format: -1, 0, +1
        
        Args:
            y: Target array in three-class format
            
        Returns:
            Target array in deep learning format (-1, 0, +1)
        """
        # Deep learning uses the same format as three-class
        return np.array(y, dtype=int)
    
    def inverse_transform_classic_ml(self, y_pred: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Transform predictions from classic ML format (0, 1, 2) back to three-class (-1, 0, +1).
        
        Mapping:
        0 → -1 (decrease)
        1 →  0 (neutral)
        2 → +1 (increase)
        
        Args:
            y_pred: Predictions in classic ML format (0, 1, 2)
            
        Returns:
            Predictions in three-class format (-1, 0, +1)
        """
        y_array = np.array(y_pred)
        y_transformed = np.zeros_like(y_array, dtype=int)
        
        y_transformed[y_array == 0] = -1  # Decrease
        y_transformed[y_array == 1] = 0   # Neutral
        y_transformed[y_array == 2] = 1   # Increase
        
        return y_transformed
    
    def inverse_transform_deep_learning(self, y_pred: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Transform predictions from deep learning format back to three-class.
        
        Args:
            y_pred: Predictions in deep learning format (-1, 0, +1)
            
        Returns:
            Predictions in three-class format (-1, 0, +1)
        """
        # Deep learning uses the same format as three-class
        return np.array(y_pred, dtype=int)
    
    def get_model_type(self, model: any) -> Literal['classic_ml', 'deep_learning']:
        """
        Detect model type automatically.
        
        Args:
            model: ML model instance
            
        Returns:
            'classic_ml' or 'deep_learning'
        """
        model_class_name = model.__class__.__name__
        model_module = model.__class__.__module__
        
        # Check if it's a deep learning model
        if 'keras' in model_module.lower() or 'tensorflow' in model_module.lower():
            return 'deep_learning'
        elif 'torch' in model_module.lower() or 'pytorch' in model_module.lower():
            return 'deep_learning'
        elif model_class_name in ['SimpleCNN', 'DeepCNN', 'ResidualCNN']:
            return 'deep_learning'
        else:
            return 'classic_ml'
    
    def prepare_targets(self, y: Union[np.ndarray, pd.Series], 
                       model: any) -> Tuple[np.ndarray, str]:
        """
        Automatically prepare targets for the given model type.
        
        Args:
            y: Target array in three-class format (-1, 0, +1)
            model: ML model instance
            
        Returns:
            Tuple of (transformed_targets, model_type)
        """
        model_type = self.get_model_type(model)
        
        if model_type == 'classic_ml':
            y_transformed = self.transform_for_classic_ml(y)
        else:  # deep_learning
            y_transformed = self.transform_for_deep_learning(y)
        
        return y_transformed, model_type
    
    def inverse_transform(self, y_pred: Union[np.ndarray, pd.Series],
                         model_type: str) -> np.ndarray:
        """
        Transform predictions back to three-class format.
        
        Args:
            y_pred: Model predictions
            model_type: 'classic_ml' or 'deep_learning'
            
        Returns:
            Predictions in three-class format (-1, 0, +1)
        """
        if model_type == 'classic_ml':
            return self.inverse_transform_classic_ml(y_pred)
        else:  # deep_learning
            return self.inverse_transform_deep_learning(y_pred)
    
    def print_transformation_info(self, y_original: np.ndarray, 
                                 y_transformed: np.ndarray,
                                 model_type: str):
        """
        Print information about the transformation.
        
        Args:
            y_original: Original targets
            y_transformed: Transformed targets
            model_type: Model type
        """
        print(f"\n{'='*60}")
        print(f"TARGET TRANSFORMATION - {model_type.upper()}")
        print(f"{'='*60}")
        
        if model_type == 'classic_ml':
            print("Mapping: -1→0, 0→1, +1→2")
        else:
            print("Mapping: -1→-1, 0→0, +1→+1 (no change)")
        
        print(f"\nOriginal distribution:")
        unique, counts = np.unique(y_original, return_counts=True)
        for val, count in zip(unique, counts):
            print(f"  {val:+2d}: {count:5d} samples")
        
        print(f"\nTransformed distribution:")
        unique, counts = np.unique(y_transformed, return_counts=True)
        for val, count in zip(unique, counts):
            print(f"  {val:+2d}: {count:5d} samples")
        
        print(f"{'='*60}\n")


# Global instance
_target_transformer = None


def get_target_transformer() -> TargetTransformer:
    """
    Get global TargetTransformer instance (singleton).
    
    Returns:
        TargetTransformer instance
    """
    global _target_transformer
    if _target_transformer is None:
        _target_transformer = TargetTransformer()
    return _target_transformer


# Convenience functions
def transform_for_model(y: Union[np.ndarray, pd.Series], 
                       model: any,
                       verbose: bool = False) -> Tuple[np.ndarray, str]:
    """
    Transform targets for the given model.
    
    Args:
        y: Target array in three-class format (-1, 0, +1)
        model: ML model instance
        verbose: Print transformation info
        
    Returns:
        Tuple of (transformed_targets, model_type)
    """
    transformer = get_target_transformer()
    y_transformed, model_type = transformer.prepare_targets(y, model)
    
    if verbose:
        transformer.print_transformation_info(np.array(y), y_transformed, model_type)
    
    return y_transformed, model_type


def inverse_transform_predictions(y_pred: Union[np.ndarray, pd.Series],
                                  model_type: str) -> np.ndarray:
    """
    Transform predictions back to three-class format.
    
    Args:
        y_pred: Model predictions
        model_type: 'classic_ml' or 'deep_learning'
        
    Returns:
        Predictions in three-class format (-1, 0, +1)
    """
    transformer = get_target_transformer()
    return transformer.inverse_transform(y_pred, model_type)
