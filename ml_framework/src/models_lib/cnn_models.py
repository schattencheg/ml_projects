"""
CNN Models - Convolutional Neural Network implementations.
"""

import numpy as np
from typing import Optional, Tuple
from .base_model import BaseModel

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class SimpleCNN(BaseModel):
    """
    Simple CNN architecture for time series classification.
    """
    
    def __init__(self, name: str = "SimpleCNN", 
                 input_shape: Optional[Tuple] = None,
                 num_classes: int = 3,
                 **params):
        """
        Initialize Simple CNN model.
        
        Args:
            name: Model name
            input_shape: Input shape (timesteps, features) or None to infer
            num_classes: Number of output classes
            **params: Additional parameters
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed. Install with: pip install tensorflow")
        
        super().__init__(name)
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.params = params
        self.model = None
        
    def _build_model(self, input_shape: Tuple, num_classes: int):
        """Build Simple CNN architecture."""
        model = models.Sequential([
            layers.Input(shape=input_shape),
            
            # Conv Block 1
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            # Conv Block 2
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit Simple CNN model."""
        # Reshape if needed (add channel dimension if not present)
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Build model if not already built
        if self.model is None:
            input_shape = X.shape[1:]
            num_classes = len(np.unique(y))
            self.model = self._build_model(input_shape, num_classes)
        
        # Default training parameters
        fit_params = {
            'epochs': 50,
            'batch_size': 32,
            'validation_split': 0.2,
            'verbose': 0
        }
        fit_params.update(kwargs)
        
        self.model.fit(X, y, **fit_params)
        
    def _predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Make predictions."""
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        predictions = self.model.predict(X, verbose=0, **kwargs)
        return np.argmax(predictions, axis=1)
    
    def _predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict class probabilities."""
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        return self.model.predict(X, verbose=0, **kwargs)


class DeepCNN(BaseModel):
    """
    Deeper CNN architecture with more layers.
    """
    
    def __init__(self, name: str = "DeepCNN", 
                 input_shape: Optional[Tuple] = None,
                 num_classes: int = 3,
                 **params):
        """
        Initialize Deep CNN model.
        
        Args:
            name: Model name
            input_shape: Input shape (timesteps, features) or None to infer
            num_classes: Number of output classes
            **params: Additional parameters
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed. Install with: pip install tensorflow")
        
        super().__init__(name)
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.params = params
        self.model = None
        
    def _build_model(self, input_shape: Tuple, num_classes: int):
        """Build Deep CNN architecture."""
        model = models.Sequential([
            layers.Input(shape=input_shape),
            
            # Conv Block 1
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            # Conv Block 2
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            # Conv Block 3
            layers.Conv1D(256, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(256, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.4),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit Deep CNN model."""
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        if self.model is None:
            input_shape = X.shape[1:]
            num_classes = len(np.unique(y))
            self.model = self._build_model(input_shape, num_classes)
        
        fit_params = {
            'epochs': 100,
            'batch_size': 32,
            'validation_split': 0.2,
            'verbose': 0
        }
        fit_params.update(kwargs)
        
        self.model.fit(X, y, **fit_params)
        
    def _predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Make predictions."""
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        predictions = self.model.predict(X, verbose=0, **kwargs)
        return np.argmax(predictions, axis=1)
    
    def _predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict class probabilities."""
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        return self.model.predict(X, verbose=0, **kwargs)


class ResidualCNN(BaseModel):
    """
    CNN with residual connections for better gradient flow.
    """
    
    def __init__(self, name: str = "ResidualCNN", 
                 input_shape: Optional[Tuple] = None,
                 num_classes: int = 3,
                 **params):
        """
        Initialize Residual CNN model.
        
        Args:
            name: Model name
            input_shape: Input shape (timesteps, features) or None to infer
            num_classes: Number of output classes
            **params: Additional parameters
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed. Install with: pip install tensorflow")
        
        super().__init__(name)
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.params = params
        self.model = None
        
    def _residual_block(self, x, filters: int, kernel_size: int = 3):
        """Create a residual block."""
        # Main path
        fx = layers.Conv1D(filters, kernel_size, padding='same')(x)
        fx = layers.BatchNormalization()(fx)
        fx = layers.Activation('relu')(fx)
        fx = layers.Conv1D(filters, kernel_size, padding='same')(fx)
        fx = layers.BatchNormalization()(fx)
        
        # Shortcut path
        if x.shape[-1] != filters:
            x = layers.Conv1D(filters, 1, padding='same')(x)
        
        # Add and activate
        out = layers.Add()([x, fx])
        out = layers.Activation('relu')(out)
        
        return out
        
    def _build_model(self, input_shape: Tuple, num_classes: int):
        """Build Residual CNN architecture."""
        inputs = layers.Input(shape=input_shape)
        
        # Initial conv
        x = layers.Conv1D(64, 7, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Residual blocks
        x = self._residual_block(x, 64)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = self._residual_block(x, 128)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = self._residual_block(x, 256)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.4)(x)
        
        # Dense layers
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit Residual CNN model."""
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        if self.model is None:
            input_shape = X.shape[1:]
            num_classes = len(np.unique(y))
            self.model = self._build_model(input_shape, num_classes)
        
        fit_params = {
            'epochs': 100,
            'batch_size': 32,
            'validation_split': 0.2,
            'verbose': 0
        }
        fit_params.update(kwargs)
        
        self.model.fit(X, y, **fit_params)
        
    def _predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Make predictions."""
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        predictions = self.model.predict(X, verbose=0, **kwargs)
        return np.argmax(predictions, axis=1)
    
    def _predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict class probabilities."""
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        return self.model.predict(X, verbose=0, **kwargs)
