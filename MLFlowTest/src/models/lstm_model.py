"""LSTM model for OHLC price prediction."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.models.base_model import BaseModel
from src.utils import get_logger

logger = get_logger(__name__)

class LSTMModel(BaseModel):
    """LSTM model for time series prediction."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LSTM model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.sequence_length = config.get('sequence_length', 60)
        self.hidden_units = config.get('hidden_units', [128, 64, 32])
        self.dropout = config.get('dropout', 0.2)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.epochs = config.get('epochs', 100)
        self.batch_size = config.get('batch_size', 32)
        
    def build_model(self, input_shape: Tuple[int, int], output_shape: int) -> None:
        """Build LSTM model architecture.
        
        Args:
            input_shape: Shape of input sequences (sequence_length, n_features)
            output_shape: Number of output targets
        """
        logger.info(f"Building LSTM model with input shape {input_shape} and output shape {output_shape}")
        
        self.model = Sequential()
        
        # First LSTM layer
        self.model.add(LSTM(
            units=self.hidden_units[0],
            return_sequences=len(self.hidden_units) > 1,
            input_shape=input_shape
        ))
        self.model.add(Dropout(self.dropout))
        self.model.add(BatchNormalization())
        
        # Additional LSTM layers
        for i, units in enumerate(self.hidden_units[1:], 1):
            return_sequences = i < len(self.hidden_units) - 1
            self.model.add(LSTM(units=units, return_sequences=return_sequences))
            self.model.add(Dropout(self.dropout))
            self.model.add(BatchNormalization())
        
        # Output layer
        self.model.add(Dense(output_shape))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"LSTM model built successfully")
        logger.info(f"Model summary:\n{self.model.summary()}")
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Train the LSTM model.
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            
        Returns:
            Training history
        """
        logger.info("Starting LSTM model training")
        
        if self.model is None:
            # Determine output shape
            output_shape = y_train.shape[1] if len(y_train.shape) > 1 else 1
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.build_model(input_shape, output_shape)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=20,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7
            )
        ]
        
        # Validation data
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        logger.info("LSTM model training completed")
        
        return {
            'history': history.history,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history.get('val_loss', [None])[-1]
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with LSTM model.
        
        Args:
            X: Input sequences
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X)
        return predictions
    
    def predict_sequence(
        self, 
        initial_sequence: np.ndarray, 
        n_steps: int,
        feature_columns: Optional[list] = None
    ) -> np.ndarray:
        """Predict multiple steps into the future.
        
        Args:
            initial_sequence: Initial sequence to start prediction from
            n_steps: Number of steps to predict
            feature_columns: Indices of features that should be updated with predictions
            
        Returns:
            Multi-step predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if len(initial_sequence.shape) == 2:
            initial_sequence = initial_sequence.reshape(1, *initial_sequence.shape)
        
        predictions = []
        current_sequence = initial_sequence.copy()
        
        for _ in range(n_steps):
            # Predict next step
            next_pred = self.model.predict(current_sequence, verbose=0)
            predictions.append(next_pred[0])
            
            # Update sequence for next prediction
            if feature_columns is not None:
                # Create new step with updated features
                new_step = current_sequence[0, -1, :].copy()
                for i, col_idx in enumerate(feature_columns):
                    if i < len(next_pred[0]):
                        new_step[col_idx] = next_pred[0][i]
                
                # Shift sequence and add new step
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, :] = new_step
            else:
                # Simple case: just use predictions as new features
                new_step = np.zeros_like(current_sequence[0, -1, :])
                new_step[:len(next_pred[0])] = next_pred[0]
                
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, :] = new_step
        
        return np.array(predictions)
    
    def get_attention_weights(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Get attention weights if model supports it.
        
        Args:
            X: Input sequences
            
        Returns:
            Attention weights or None
        """
        # This would require modifying the model architecture to include attention
        # For now, return None
        return None
    
    def save_model(self, filepath: str) -> None:
        """Save LSTM model.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        # Save Keras model
        model_path = filepath.replace('.pkl', '.h5')
        self.model.save(model_path)
        
        # Save metadata
        super().save_model(filepath)
        
        logger.info(f"LSTM model saved to {filepath} and {model_path}")
    
    def load_model(self, filepath: str) -> None:
        """Load LSTM model.
        
        Args:
            filepath: Path to load model from
        """
        # Load Keras model
        model_path = filepath.replace('.pkl', '.h5')
        self.model = tf.keras.models.load_model(model_path)
        
        # Load metadata
        super().load_model(filepath)
        
        logger.info(f"LSTM model loaded from {filepath} and {model_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get LSTM model information.
        
        Returns:
            Dictionary with model information
        """
        info = super().get_model_info()
        info.update({
            'sequence_length': self.sequence_length,
            'hidden_units': self.hidden_units,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'total_params': self.model.count_params() if self.model else 0
        })
        return info
