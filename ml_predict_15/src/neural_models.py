"""
Neural Network Model Builders

This module contains functions to create LSTM, CNN, and hybrid neural network models
for time series classification.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Neural Network imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


def create_sequences(X, y, sequence_length=60):
    """
    Create sequences for LSTM/CNN models from feature data.
    
    Parameters:
    -----------
    X : np.ndarray or pd.DataFrame
        Feature data
    y : np.ndarray or pd.Series
        Target data
    sequence_length : int
        Length of each sequence
    
    Returns:
    --------
    X_seq : np.ndarray
        Sequences of shape (samples, sequence_length, features)
    y_seq : np.ndarray
        Corresponding targets
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    X_seq, y_seq = [], []
    
    for i in range(sequence_length, len(X)):
        X_seq.append(X[i-sequence_length:i])
        y_seq.append(y[i])
    
    return np.array(X_seq), np.array(y_seq)


def create_lstm_model(input_shape, dropout_rate=0.2):
    """
    Create a simple LSTM model for binary classification.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (sequence_length, features)
    dropout_rate : float
        Dropout rate for regularization
    
    Returns:
    --------
    model : Sequential
        Compiled LSTM model
    """
    if not TENSORFLOW_AVAILABLE:
        return None
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(50, return_sequences=False),
        Dropout(dropout_rate),
        Dense(25, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_cnn_model(input_shape, dropout_rate=0.2):
    """
    Create a 1D CNN model for pattern recognition in sequential data.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (sequence_length, features)
    dropout_rate : float
        Dropout rate for regularization
    
    Returns:
    --------
    model : Sequential
        Compiled CNN model
    """
    if not TENSORFLOW_AVAILABLE:
        return None
    
    model = Sequential([
        # First Conv block
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rate),
        
        # Second Conv block
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rate),
        
        # Third Conv block
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rate),
        
        # Dense layers
        Flatten(),
        Dense(50, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_hybrid_lstm_cnn_model(input_shape, dropout_rate=0.2):
    """
    Create a hybrid LSTM-CNN model combining both approaches.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (sequence_length, features)
    dropout_rate : float
        Dropout rate for regularization
    
    Returns:
    --------
    model : Sequential
        Compiled hybrid model
    """
    if not TENSORFLOW_AVAILABLE:
        return None
    
    model = Sequential([
        # CNN layers for feature extraction
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rate),
        
        # LSTM layers for sequence modeling
        LSTM(50, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(50, return_sequences=False),
        Dropout(dropout_rate),
        
        # Dense layers
        Dense(25, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


class KerasClassifierWrapper:
    """
    Wrapper to make Keras models compatible with sklearn-style interface.
    """
    
    def __init__(self, model_builder, input_shape, sequence_length=60, **kwargs):
        self.model_builder = model_builder
        self.input_shape = input_shape
        self.sequence_length = sequence_length
        self.model = None
        self.kwargs = kwargs
        self.scaler = MinMaxScaler()
    
    def fit(self, X, y):
        """
        Fit the Keras model.
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = create_sequences(X_scaled, y, self.sequence_length)
        
        # Build model
        self.model = self.model_builder(self.input_shape, **self.kwargs)
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=0
        )
        
        # Train
        self.model.fit(
            X_seq, y_seq,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        return self
    
    def predict(self, X):
        """
        Predict class labels.
        """
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        X_seq, _ = create_sequences(X_scaled, np.zeros(len(X)), self.sequence_length)
        
        # Predict
        y_pred_proba = self.model.predict(X_seq, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Pad predictions to match original length
        padding = np.zeros(self.sequence_length)
        y_pred_full = np.concatenate([padding, y_pred])
        
        return y_pred_full
    
    def predict_proba(self, X):
        """
        Predict class probabilities for ROC AUC calculation.
        """
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        X_seq, _ = create_sequences(X_scaled, np.zeros(len(X)), self.sequence_length)
        
        # Predict probabilities
        y_pred_proba = self.model.predict(X_seq, verbose=0).flatten()
        
        # Pad predictions to match original length
        padding = np.zeros(self.sequence_length)
        y_pred_proba_full = np.concatenate([padding, y_pred_proba])
        
        # Return in sklearn format (n_samples, n_classes)
        return np.column_stack([1 - y_pred_proba_full, y_pred_proba_full])
