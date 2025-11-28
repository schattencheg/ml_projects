"""
CNN Models - 1D Convolutional Neural Networks for time series classification.
"""

import numpy as np
from typing import Optional, Tuple
from abc import abstractmethod
from src.models_lib.base_model import BaseModel

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def _setup_gpu():
    """Configure GPU memory growth for TensorFlow."""
    if not TF_AVAILABLE:
        return
    for gpu in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


class BaseCNN(BaseModel):
    """Base class for CNN models with shared functionality."""
    
    def __init__(self, name: str, num_classes: int = 3, epochs: int = 50, 
                 batch_size: int = 32, use_gpu: bool = True, **params):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required. Install: pip install tensorflow")
        super().__init__(name)
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.params = params
        self.model = None
        if use_gpu:
            _setup_gpu()
    
    @abstractmethod
    def _build_model(self, input_shape: Tuple, num_classes: int):
        """Build the model architecture. Must be implemented by subclasses."""
        pass
    
    def _ensure_3d(self, X: np.ndarray) -> np.ndarray:
        """Ensure input is 3D (samples, timesteps, features)."""
        return X.reshape(X.shape[0], X.shape[1], 1) if len(X.shape) == 2 else X
    
    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        X = self._ensure_3d(X)
        if self.model is None:
            self.model = self._build_model(X.shape[1:], len(np.unique(y)))
        
        fit_params = {'epochs': self.epochs, 'batch_size': self.batch_size,
                      'validation_split': 0.2, 'verbose': 0}
        fit_params.update(kwargs)
        
        # Compute class weights if not provided to handle class imbalance
        if 'class_weight' not in fit_params:
            unique, counts = np.unique(y, return_counts=True)
            total = len(y)
            # Inverse frequency weighting
            fit_params['class_weight'] = {cls: total / (len(unique) * count) for cls, count in zip(unique, counts)}
        
        self.model.fit(X, y, **fit_params)
    
    def _predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        X = self._ensure_3d(X)
        return np.argmax(self.model.predict(X, verbose=0), axis=1)
    
    def _predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        return self.model.predict(self._ensure_3d(X), verbose=0)


def _conv_block(filters: int, double: bool = False):
    """Create a conv block: Conv1D -> BN -> (optional second Conv1D -> BN) -> Pool -> Dropout."""
    block = [
        layers.Conv1D(filters, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
    ]
    if double:
        block += [
            layers.Conv1D(filters, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
        ]
    block += [layers.MaxPooling1D(2), layers.Dropout(0.3)]
    return block


class SimpleCNN(BaseCNN):
    """Simple 2-block CNN for time series classification."""
    
    def __init__(self, name: str = "SimpleCNN", num_classes: int = 3, **params):
        super().__init__(name, num_classes, epochs=50, **params)
    
    def _build_model(self, input_shape: Tuple, num_classes: int):
        model = models.Sequential([
            layers.Input(shape=input_shape),
            *_conv_block(64),
            *_conv_block(128),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model


class DeepCNN(BaseCNN):
    """Deeper 3-block CNN with double convolutions per block."""
    
    def __init__(self, name: str = "DeepCNN", num_classes: int = 3, **params):
        super().__init__(name, num_classes, epochs=100, **params)
    
    def _build_model(self, input_shape: Tuple, num_classes: int):
        model = models.Sequential([
            layers.Input(shape=input_shape),
            *_conv_block(64, double=True),
            *_conv_block(128, double=True),
            *_conv_block(256, double=True),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model


class ResidualCNN(BaseCNN):
    """CNN with residual connections for better gradient flow."""
    
    def __init__(self, name: str = "ResidualCNN", num_classes: int = 3, **params):
        super().__init__(name, num_classes, epochs=100, **params)
    
    def _residual_block(self, x, filters: int):
        """Create a residual block with skip connection."""
        fx = layers.Conv1D(filters, 3, padding='same')(x)
        fx = layers.BatchNormalization()(fx)
        fx = layers.Activation('relu')(fx)
        fx = layers.Conv1D(filters, 3, padding='same')(fx)
        fx = layers.BatchNormalization()(fx)
        
        # Match dimensions if needed
        if x.shape[-1] != filters:
            x = layers.Conv1D(filters, 1, padding='same')(x)
        
        return layers.Activation('relu')(layers.Add()([x, fx]))
    
    def _build_model(self, input_shape: Tuple, num_classes: int):
        inputs = layers.Input(shape=input_shape)
        
        x = layers.Conv1D(64, 7, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        for filters, dropout in [(64, 0.3), (128, 0.3), (256, 0.4)]:
            x = self._residual_block(x, filters)
            x = layers.MaxPooling1D(2)(x)
            x = layers.Dropout(dropout)(x)
        
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
