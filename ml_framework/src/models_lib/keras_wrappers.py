"""
Keras Model Wrappers - Wrapper classes for Keras models to integrate with BaseModel.

These wrapper classes must be at module level (not inside functions) to be picklable.
"""

from src.models_lib.base_model import BaseModel

try:
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class LSTMModelWrapper(BaseModel):
    """
    Wrapper for LSTM Keras models to integrate with BaseModel.
    
    This class wraps a Keras Sequential LSTM model and provides the BaseModel
    interface with automatic target conversion.
    
    Args:
        keras_model: Compiled Keras model
        model_name: Name for the model
    """
    
    def __init__(self, keras_model, model_name):
        super().__init__(name=model_name)
        self.model = keras_model
    
    def _fit(self, X, y, **kwargs):
        """Internal fit method - trains the Keras model."""
        return self.model.fit(X, y, **kwargs)
    
    def _predict(self, X, **kwargs):
        """Internal predict method - returns class predictions."""
        return self.model.predict(X, verbose=0).argmax(axis=1)
    
    def _predict_proba(self, X, **kwargs):
        """Internal predict_proba method - returns class probabilities."""
        return self.model.predict(X, verbose=0)


class HybridModelWrapper(BaseModel):
    """
    Wrapper for Hybrid CNN-LSTM Keras models to integrate with BaseModel.
    
    This class wraps a Keras Sequential model that combines CNN and LSTM layers
    and provides the BaseModel interface with automatic target conversion.
    
    Args:
        keras_model: Compiled Keras model
        model_name: Name for the model
    """
    
    def __init__(self, keras_model, model_name):
        super().__init__(name=model_name)
        self.model = keras_model
    
    def _fit(self, X, y, **kwargs):
        """Internal fit method - trains the Keras model."""
        return self.model.fit(X, y, **kwargs)
    
    def _predict(self, X, **kwargs):
        """Internal predict method - returns class predictions."""
        return self.model.predict(X, verbose=0).argmax(axis=1)
    
    def _predict_proba(self, X, **kwargs):
        """Internal predict_proba method - returns class probabilities."""
        return self.model.predict(X, verbose=0)
