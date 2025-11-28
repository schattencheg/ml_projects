"""
Keras Model Wrappers - Wrapper classes for Keras models to integrate with BaseModel.

These wrapper classes must be at module level (not inside functions) to be picklable.
"""

import numpy as np
from src.models_lib.base_model import BaseModel

try:
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


def _ensure_3d(X):
    """
    Ensure input is 3D for Keras sequential models (LSTM, CNN, Hybrid).
    
    Keras expects shape (samples, timesteps, features).
    If input is 2D (samples, features), reshape to (samples, features, 1).
    """
    if isinstance(X, np.ndarray) and X.ndim == 2:
        return X.reshape(X.shape[0], X.shape[1], 1)
    return X


class LSTMModelWrapper(BaseModel):
    """
    Wrapper for LSTM Keras models to integrate with BaseModel.
    
    This class wraps a Keras Sequential LSTM model and provides the BaseModel
    interface with automatic target conversion.
    
    Automatically reshapes 2D input (samples, features) to 3D (samples, features, 1)
    for compatibility with LSTM layers.
    
    Args:
        keras_model: Compiled Keras model
        model_name: Name for the model
    """
    
    def __init__(self, keras_model, model_name):
        super().__init__(name=model_name)
        self.model = keras_model
    
    def _fit(self, X, y, **kwargs):
        """Internal fit method - trains the Keras model with automatic class weighting."""
        X = _ensure_3d(X)
        
        # Compute class weights if not provided to handle class imbalance
        if 'class_weight' not in kwargs:
            unique, counts = np.unique(y, return_counts=True)
            total = len(y)
            # Inverse frequency weighting
            class_weight = {cls: total / (len(unique) * count) for cls, count in zip(unique, counts)}
            kwargs['class_weight'] = class_weight
        
        return self.model.fit(X, y, **kwargs)
    
    def _predict(self, X, **kwargs):
        """Internal predict method - returns class predictions."""
        X = _ensure_3d(X)
        return self.model.predict(X, verbose=0).argmax(axis=1)
    
    def _predict_proba(self, X, **kwargs):
        """Internal predict_proba method - returns class probabilities."""
        X = _ensure_3d(X)
        return self.model.predict(X, verbose=0)


class HybridModelWrapper(BaseModel):
    """
    Wrapper for Hybrid CNN-LSTM Keras models to integrate with BaseModel.
    
    This class wraps a Keras Sequential model that combines CNN and LSTM layers
    and provides the BaseModel interface with automatic target conversion.
    
    Automatically reshapes 2D input (samples, features) to 3D (samples, features, 1)
    for compatibility with Conv1D and LSTM layers.
    
    Args:
        keras_model: Compiled Keras model
        model_name: Name for the model
    """
    
    def __init__(self, keras_model, model_name):
        super().__init__(name=model_name)
        self.model = keras_model
    
    def _fit(self, X, y, **kwargs):
        """Internal fit method - trains the Keras model with automatic class weighting."""
        X = _ensure_3d(X)
        
        # Compute class weights if not provided to handle class imbalance
        if 'class_weight' not in kwargs:
            unique, counts = np.unique(y, return_counts=True)
            total = len(y)
            # Inverse frequency weighting
            class_weight = {cls: total / (len(unique) * count) for cls, count in zip(unique, counts)}
            kwargs['class_weight'] = class_weight
        
        return self.model.fit(X, y, **kwargs)
    
    def _predict(self, X, **kwargs):
        """Internal predict method - returns class predictions."""
        X = _ensure_3d(X)
        return self.model.predict(X, verbose=0).argmax(axis=1)
    
    def _predict_proba(self, X, **kwargs):
        """Internal predict_proba method - returns class probabilities."""
        X = _ensure_3d(X)
        return self.model.predict(X, verbose=0)
