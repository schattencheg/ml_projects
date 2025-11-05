"""
Neural Networks Manager Module

Handles creation and management of various neural network architectures:
- Multiple 1D-CNN variants
- Multiple LSTM variants  
- GAN implementation
- Sequence creation utilities
"""

import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_X_y, check_array
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Import centralized model configuration
try:
    from src.ModelConfig import get_model_config
    MODEL_CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ModelConfig not available: {e}")
    MODEL_CONFIG_AVAILABLE = False

# Set TensorFlow to use CPU only to avoid GPU memory issues during development
tf.config.set_visible_devices([], 'GPU')

class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper to make Keras models compatible with sklearn.
    """
    
    def __init__(self, build_fn, sequence_length=30, epochs=50, batch_size=32, 
                 validation_split=0.2, verbose=0, **kwargs):
        self.build_fn = build_fn
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.verbose = verbose
        self.kwargs = kwargs
        self.model = None
        self.scaler = MinMaxScaler()
        self.classes_ = None
        self.label_map_ = None  # Store label mapping for inverse transform
        self.n_samples_dropped_ = 0  # Number of samples dropped during sequence creation
        
    def _create_sequences(self, X, y=None):
        """Create sequences for time series data."""
        if len(X) < self.sequence_length:
            raise ValueError(f"Not enough data points. Need at least {self.sequence_length}, got {len(X)}")
        
        X_seq = []
        y_seq = [] if y is not None else None
        
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
            if y is not None:
                y_seq.append(y[i])
        
        X_seq = np.array(X_seq)
        if y is not None:
            y_seq = np.array(y_seq)
            return X_seq, y_seq
        return X_seq
    
    def fit(self, X, y):
        """Fit the model."""
        X, y = check_X_y(X, y)
        
        # Store classes
        self.classes_ = np.unique(y)
        
        # Ensure labels are integers (required for sparse_categorical_crossentropy)
        # Map labels to 0, 1, 2, ... if they aren't already
        if not np.issubdtype(y.dtype, np.integer):
            y = y.astype(np.int32)
        
        # If labels are not starting from 0, remap them
        unique_labels = np.unique(y)
        if not np.array_equal(unique_labels, np.arange(len(unique_labels))):
            self.label_map_ = {idx: label for idx, label in enumerate(unique_labels)}
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
            y = np.array([label_map[label] for label in y])
        else:
            self.label_map_ = None
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y)
        
        # Store how many samples were dropped
        self.n_samples_dropped_ = len(y) - len(y_seq)
        
        # Ensure y_seq is int32 and validate labels
        y_seq = y_seq.astype(np.int32)
        unique_y_seq = np.unique(y_seq)
        num_classes = len(unique_y_seq)
        
        # Validate that labels are 0, 1, 2, ... n-1
        if not np.array_equal(unique_y_seq, np.arange(num_classes)):
            raise ValueError(
                f"Labels must be consecutive integers starting from 0. "
                f"Got: {unique_y_seq}, expected: {np.arange(num_classes)}"
            )
        
        # Build model
        input_shape = (self.sequence_length, X.shape[1])
        self.model = self.build_fn(input_shape=input_shape, **self.kwargs)
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Add callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        # Train model
        self.model.fit(
            X_seq, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=self.verbose
        )
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        X = check_array(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        X_seq = self._create_sequences(X_scaled)
        
        # Predict
        predictions = self.model.predict(X_seq, verbose=0)
        predicted_indices = np.argmax(predictions, axis=1)
        
        # Map back to original labels if needed
        if self.label_map_ is not None:
            predicted_indices = np.array([self.label_map_[idx] for idx in predicted_indices])
        
        # Pad predictions to match original input length
        # Use the most common class for the dropped samples
        if self.n_samples_dropped_ > 0:
            # Use the mode of predictions for padding
            pad_value = np.bincount(predicted_indices).argmax()
            padded_predictions = np.full(len(X), pad_value, dtype=predicted_indices.dtype)
            padded_predictions[self.n_samples_dropped_:] = predicted_indices
            return padded_predictions
        
        return predicted_indices
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        X = check_array(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        X_seq = self._create_sequences(X_scaled)
        
        # Predict probabilities
        probabilities = self.model.predict(X_seq, verbose=0)
        
        # Pad probabilities to match original input length
        if self.n_samples_dropped_ > 0:
            # Create uniform probabilities for dropped samples
            n_classes = probabilities.shape[1]
            uniform_proba = np.full((len(X), n_classes), 1.0 / n_classes)
            uniform_proba[self.n_samples_dropped_:] = probabilities
            return uniform_proba
        
        return probabilities


class NeuralNetworksManager:
    """
    Manages various neural network architectures for cryptocurrency prediction.
    """
    
    def __init__(self, sequence_length=30, epochs=50, batch_size=32):
        """
        Initialize NeuralNetworksManager.
        
        Parameters:
        -----------
        sequence_length : int
            Length of input sequences
        epochs : int
            Number of training epochs
        batch_size : int
            Training batch size
        """
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Get centralized model configuration
        if MODEL_CONFIG_AVAILABLE:
            self.model_config_manager = get_model_config()
            # Set build_fn references in centralized config
            self._setup_build_functions()
        else:
            raise ImportError("ModelConfig is required but not available")
        
        # Legacy: Keep reference for backward compatibility (deprecated)
        self.model_configs = {
            # 1D-CNN Variants
            'cnn_simple': {
                'enabled': True,
                'build_fn': self._build_cnn_simple,
                'description': 'Simple 1D-CNN with 2 conv layers'
            },
            'cnn_deep': {
                'enabled': True,
                'build_fn': self._build_cnn_deep,
                'description': 'Deep 1D-CNN with 4 conv layers'
            },
            'cnn_residual': {
                'enabled': True,
                'build_fn': self._build_cnn_residual,
                'description': '1D-CNN with residual connections'
            },
            'cnn_attention': {
                'enabled': True,
                'build_fn': self._build_cnn_attention,
                'description': '1D-CNN with attention mechanism'
            },
            'cnn_dilated': {
                'enabled': True,
                'build_fn': self._build_cnn_dilated,
                'description': '1D-CNN with dilated convolutions'
            },
            
            # LSTM Variants
            'lstm_simple': {
                'enabled': True,
                'build_fn': self._build_lstm_simple,
                'description': 'Simple LSTM with 2 layers'
            },
            'lstm_bidirectional': {
                'enabled': True,
                'build_fn': self._build_lstm_bidirectional,
                'description': 'Bidirectional LSTM'
            },
            'lstm_stacked': {
                'enabled': True,
                'build_fn': self._build_lstm_stacked,
                'description': 'Stacked LSTM with 3 layers'
            },
            'lstm_attention': {
                'enabled': True,
                'build_fn': self._build_lstm_attention,
                'description': 'LSTM with attention mechanism'
            },
            'lstm_cnn_hybrid': {
                'enabled': True,
                'build_fn': self._build_lstm_cnn_hybrid,
                'description': 'Hybrid CNN-LSTM model'
            },
            
            # GRU Variants
            'gru_simple': {
                'enabled': True,
                'build_fn': self._build_gru_simple,
                'description': 'Simple GRU model'
            },
            'gru_bidirectional': {
                'enabled': True,
                'build_fn': self._build_gru_bidirectional,
                'description': 'Bidirectional GRU'
            }
        }
    
    def _setup_build_functions(self):
        """Set build_fn references in centralized config for neural network models."""
        # Map build functions to centralized config
        build_fn_mapping = {
            'cnn_simple': self._build_cnn_simple,
            'cnn_deep': self._build_cnn_deep,
            'cnn_residual': self._build_cnn_residual,
            'cnn_attention': self._build_cnn_attention,
            'cnn_dilated': self._build_cnn_dilated,
            'lstm_simple': self._build_lstm_simple,
            'lstm_bidirectional': self._build_lstm_bidirectional,
            'lstm_stacked': self._build_lstm_stacked,
            'lstm_attention': self._build_lstm_attention,
            'lstm_cnn_hybrid': self._build_lstm_cnn_hybrid,
            'gru_simple': self._build_gru_simple,
            'gru_bidirectional': self._build_gru_bidirectional
        }
        
        # Set build_fn in centralized config
        for model_name, build_fn in build_fn_mapping.items():
            if model_name in self.model_config_manager.neural_network_models:
                self.model_config_manager.neural_network_models[model_name]['build_fn'] = build_fn
    
    # ==================== 1D-CNN VARIANTS ====================
    
    def _build_cnn_simple(self, input_shape, **kwargs):
        """Build simple 1D-CNN model."""
        model = models.Sequential([
            layers.Conv1D(32, 3, activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            layers.Conv1D(64, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            layers.GlobalMaxPooling1D(),
            layers.Dense(50, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(3, activation='softmax')  # 3 classes
        ])
        return model
    
    def _build_cnn_deep(self, input_shape, **kwargs):
        """Build deep 1D-CNN model."""
        model = models.Sequential([
            layers.Conv1D(32, 3, activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv1D(32, 3, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Dropout(0.25),
            
            layers.Conv1D(64, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.Conv1D(64, 3, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Dropout(0.25),
            
            layers.Conv1D(128, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.Conv1D(128, 3, activation='relu'),
            layers.GlobalMaxPooling1D(),
            layers.Dropout(0.5),
            
            layers.Dense(100, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(50, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(3, activation='softmax')  # 3 classes
        ])
        return model
    
    def _build_cnn_residual(self, input_shape, **kwargs):
        """Build 1D-CNN with residual connections."""
        inputs = layers.Input(shape=input_shape)
        
        # First block
        x = layers.Conv1D(32, 3, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        
        # Residual block 1
        residual = x
        x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(32, 3, padding='same')(x)
        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.3)(x)
        
        # Residual block 2
        residual = layers.Conv1D(64, 1, padding='same')(x)  # Match dimensions
        x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(64, 3, padding='same')(x)
        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layers
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dense(50, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(3, activation='softmax')  # 3 classes(x)
        
        model = models.Model(inputs, outputs)
        return model
    
    def _build_cnn_attention(self, input_shape, **kwargs):
        """Build 1D-CNN with attention mechanism."""
        inputs = layers.Input(shape=input_shape)
        
        # CNN layers
        x = layers.Conv1D(32, 3, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(64, 3, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(128, 3, activation='relu')(x)
        
        # Attention mechanism
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(128)(attention)
        attention = layers.Permute([2, 1])(attention)
        
        # Apply attention
        x = layers.Multiply()([x, attention])
        x = layers.GlobalMaxPooling1D()(x)
        
        # Output layers
        x = layers.Dense(50, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(3, activation='softmax')  # 3 classes(x)
        
        model = models.Model(inputs, outputs)
        return model
    
    def _build_cnn_dilated(self, input_shape, **kwargs):
        """Build 1D-CNN with dilated convolutions."""
        model = models.Sequential([
            layers.Conv1D(32, 3, dilation_rate=1, activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv1D(32, 3, dilation_rate=2, activation='relu'),
            layers.BatchNormalization(),
            layers.Conv1D(64, 3, dilation_rate=4, activation='relu'),
            layers.BatchNormalization(),
            layers.Conv1D(64, 3, dilation_rate=8, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.GlobalMaxPooling1D(),
            layers.Dense(50, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(3, activation='softmax')  # 3 classes
        ])
        return model
    
    # ==================== LSTM VARIANTS ====================
    
    def _build_lstm_simple(self, input_shape, **kwargs):
        """Build simple LSTM model."""
        model = models.Sequential([
            layers.LSTM(50, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.3),
            layers.LSTM(50),
            layers.Dropout(0.3),
            layers.Dense(25, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(3, activation='softmax')  # 3 classes
        ])
        return model
    
    def _build_lstm_bidirectional(self, input_shape, **kwargs):
        """Build bidirectional LSTM model."""
        model = models.Sequential([
            layers.Bidirectional(layers.LSTM(50, return_sequences=True), input_shape=input_shape),
            layers.Dropout(0.3),
            layers.Bidirectional(layers.LSTM(50)),
            layers.Dropout(0.3),
            layers.Dense(25, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(3, activation='softmax')  # 3 classes
        ])
        return model
    
    def _build_lstm_stacked(self, input_shape, **kwargs):
        """Build stacked LSTM model."""
        model = models.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(32),
            layers.Dropout(0.3),
            layers.Dense(25, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(3, activation='softmax')  # 3 classes
        ])
        return model
    
    def _build_lstm_attention(self, input_shape, **kwargs):
        """Build LSTM with attention mechanism."""
        inputs = layers.Input(shape=input_shape)
        
        # LSTM layer
        lstm_out = layers.LSTM(50, return_sequences=True)(inputs)
        lstm_out = layers.Dropout(0.3)(lstm_out)
        
        # Attention mechanism
        attention = layers.Dense(1, activation='tanh')(lstm_out)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(50)(attention)
        attention = layers.Permute([2, 1])(attention)
        
        # Apply attention
        attended = layers.Multiply()([lstm_out, attention])
        attended = layers.GlobalMaxPooling1D()(attended)
        
        # Output layers
        x = layers.Dense(25, activation='relu')(attended)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(3, activation='softmax')  # 3 classes(x)
        
        model = models.Model(inputs, outputs)
        return model
    
    def _build_lstm_cnn_hybrid(self, input_shape, **kwargs):
        """Build hybrid CNN-LSTM model."""
        model = models.Sequential([
            layers.Conv1D(32, 3, activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv1D(64, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            layers.LSTM(50, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(50),
            layers.Dropout(0.3),
            
            layers.Dense(25, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(3, activation='softmax')  # 3 classes
        ])
        return model
    
    # ==================== GRU VARIANTS ====================
    
    def _build_gru_simple(self, input_shape, **kwargs):
        """Build simple GRU model."""
        model = models.Sequential([
            layers.GRU(50, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.3),
            layers.GRU(50),
            layers.Dropout(0.3),
            layers.Dense(25, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(3, activation='softmax')  # 3 classes
        ])
        return model
    
    def _build_gru_bidirectional(self, input_shape, **kwargs):
        """Build bidirectional GRU model."""
        model = models.Sequential([
            layers.Bidirectional(layers.GRU(50, return_sequences=True), input_shape=input_shape),
            layers.Dropout(0.3),
            layers.Bidirectional(layers.GRU(50)),
            layers.Dropout(0.3),
            layers.Dense(25, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(3, activation='softmax')  # 3 classes
        ])
        return model
    
    # ==================== MODEL MANAGEMENT ====================
    
    def create_models(self, enabled_only=True):
        """
        Create neural network model instances.
        
        Parameters:
        -----------
        enabled_only : bool
            If True, only create enabled models
            
        Returns:
        --------
        dict : Dictionary of model_name -> KerasClassifierWrapper instance
        """
        models = {}
        
        # Create from centralized config
        for name, config in self.model_config_manager.neural_network_models.items():
            if enabled_only and not config['enabled']:
                continue
            
            try:
                wrapper = KerasClassifierWrapper(
                    build_fn=config['build_fn'],
                    sequence_length=self.sequence_length,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_split=0.2,
                    verbose=0
                )
                models[name] = wrapper
                print(f"✓ Created neural network: {name} - {config['description']}")
            except Exception as e:
                print(f"✗ Failed to create neural network {name}: {e}")
        
        print(f"\nTotal neural networks created: {len(models)}")
        return models
    
    def enable_model(self, model_name, enabled=True):
        """Enable or disable a neural network model."""
        # Delegate to centralized config
        self.model_config_manager.enable_model(model_name, enabled)
    
    def get_enabled_models(self):
        """Get list of enabled neural network model names."""
        # Get from centralized config
        return self.model_config_manager.get_enabled_neural_network_models()
    
    def print_config(self):
        """Print current neural network configuration."""
        # Print neural network specific settings
        print(f"\n{'='*70}")
        print(f"NEURAL NETWORKS TRAINING CONFIGURATION")
        print(f"{'='*70}")
        print(f"  • Sequence length: {self.sequence_length}")
        print(f"  • Epochs: {self.epochs}")
        print(f"  • Batch size: {self.batch_size}")
        print(f"{'='*70}\n")
        
        # Note: Model enable/disable status is managed by centralized ModelConfig
        # Use model_config_manager.print_config() to see full configuration


class CryptocurrencyGAN:
    """
    Generative Adversarial Network for cryptocurrency price prediction.
    
    This GAN learns to generate realistic price sequences and can be used
    for data augmentation or as a feature extractor for prediction.
    """
    
    def __init__(self, sequence_length=30, latent_dim=100):
        """
        Initialize CryptocurrencyGAN.
        
        Parameters:
        -----------
        sequence_length : int
            Length of price sequences
        latent_dim : int
            Dimension of latent space
        """
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.generator = None
        self.discriminator = None
        self.gan = None
        self.scaler = MinMaxScaler()
        
    def _build_generator(self, input_dim):
        """Build generator network."""
        model = models.Sequential([
            layers.Dense(128, input_dim=self.latent_dim),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(),
            
            layers.Dense(256),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(),
            
            layers.Dense(512),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(),
            
            layers.Dense(self.sequence_length * input_dim, activation='tanh'),
            layers.Reshape((self.sequence_length, input_dim))
        ])
        return model
    
    def _build_discriminator(self, input_shape):
        """Build discriminator network."""
        model = models.Sequential([
            layers.Conv1D(32, 3, strides=2, padding='same', input_shape=input_shape),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.25),
            
            layers.Conv1D(64, 3, strides=2, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.25),
            
            layers.Conv1D(128, 3, strides=2, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def _create_sequences(self, X):
        """Create sequences for GAN training."""
        if len(X) < self.sequence_length:
            raise ValueError(f"Not enough data points. Need at least {self.sequence_length}, got {len(X)}")
        
        sequences = []
        for i in range(self.sequence_length, len(X)):
            sequences.append(X[i-self.sequence_length:i])
        
        return np.array(sequences)
    
    def build_models(self, input_dim):
        """Build and compile GAN models."""
        # Build generator
        self.generator = self._build_generator(input_dim)
        
        # Build discriminator
        input_shape = (self.sequence_length, input_dim)
        self.discriminator = self._build_discriminator(input_shape)
        self.discriminator.compile(
            optimizer=optimizers.Adam(0.0002, 0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Build GAN
        self.discriminator.trainable = False
        gan_input = layers.Input(shape=(self.latent_dim,))
        generated = self.generator(gan_input)
        validity = self.discriminator(generated)
        self.gan = models.Model(gan_input, validity)
        self.gan.compile(
            optimizer=optimizers.Adam(0.0002, 0.5),
            loss='binary_crossentropy'
        )
    
    def train(self, X, epochs=1000, batch_size=32, sample_interval=100):
        """
        Train the GAN.
        
        Parameters:
        -----------
        X : array-like
            Training data
        epochs : int
            Number of training epochs
        batch_size : int
            Training batch size
        sample_interval : int
            Interval for sampling generated data
        """
        # Scale and create sequences
        X_scaled = self.scaler.fit_transform(X)
        X_sequences = self._create_sequences(X_scaled)
        
        # Build models
        self.build_models(X.shape[1])
        
        # Training loop
        for epoch in range(epochs):
            # Train discriminator
            idx = np.random.randint(0, X_sequences.shape[0], batch_size)
            real_sequences = X_sequences[idx]
            
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            generated_sequences = self.generator.predict(noise, verbose=0)
            
            d_loss_real = self.discriminator.train_on_batch(real_sequences, np.ones((batch_size, 1)))
            d_loss_fake = self.discriminator.train_on_batch(generated_sequences, np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch(noise, np.ones((batch_size, 1)))
            
            # Print progress
            if epoch % sample_interval == 0:
                print(f"Epoch {epoch}/{epochs} - D loss: {d_loss[0]:.4f}, acc: {d_loss[1]:.4f} - G loss: {g_loss:.4f}")
    
    def generate_sequences(self, n_samples=100):
        """Generate synthetic price sequences."""
        if self.generator is None:
            raise ValueError("Generator not trained. Call train() first.")
        
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        generated = self.generator.predict(noise, verbose=0)
        
        # Inverse transform to original scale
        generated_reshaped = generated.reshape(-1, generated.shape[-1])
        generated_scaled = self.scaler.inverse_transform(generated_reshaped)
        generated_sequences = generated_scaled.reshape(generated.shape)
        
        return generated_sequences
    
    def get_discriminator_features(self, X):
        """Use discriminator as feature extractor."""
        if self.discriminator is None:
            raise ValueError("Discriminator not trained. Call train() first.")
        
        # Scale and create sequences
        X_scaled = self.scaler.transform(X)
        X_sequences = self._create_sequences(X_scaled)
        
        # Extract features from discriminator's intermediate layers
        feature_extractor = models.Model(
            inputs=self.discriminator.input,
            outputs=self.discriminator.layers[-2].output  # Before final dense layer
        )
        
        features = feature_extractor.predict(X_sequences, verbose=0)
        return features


# Example usage and testing
if __name__ == "__main__":
    print("Testing Neural Networks Manager...")
    
    # Create manager
    nn_manager = NeuralNetworksManager(sequence_length=30, epochs=10, batch_size=16)
    
    # Print configuration
    nn_manager.print_config()
    
    # Create models (disabled for testing to avoid long training times)
    print("\nCreating neural network models...")
    # models = nn_manager.create_models(enabled_only=True)
    
    print("\n✓ Neural Networks Manager created successfully!")
    print("  - Multiple 1D-CNN variants implemented")
    print("  - Multiple LSTM variants implemented") 
    print("  - GRU variants implemented")
    print("  - GAN implementation included")
    print("  - KerasClassifierWrapper for sklearn compatibility")
