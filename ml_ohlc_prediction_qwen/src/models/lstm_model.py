import numpy as np
import joblib
from typing import Any, Dict
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from .base_model import BaseModel


class LSTMModel(BaseModel):
    """
    LSTM Neural Network for OHLC prediction
    """
    
    def __init__(self, name: str = "LSTM", 
                 n_units: int = 50, 
                 n_layers: int = 2, 
                 dropout: float = 0.2,
                 sequence_length: int = 10):
        super().__init__(name)
        self.n_units = n_units
        self.n_layers = n_layers
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.model = None
    
    def _reshape_data_for_lstm(self, X: np.ndarray) -> np.ndarray:
        """
        Reshape data for LSTM input (samples, timesteps, features)
        """
        if len(X.shape) == 2:
            # If we have a 2D array, we need to reshape it for LSTM
            # This assumes we have flattened sequential data
            samples = X.shape[0] - self.sequence_length + 1
            features = X.shape[1]
            X_lstm = np.zeros((samples, self.sequence_length, features))
            
            for i in range(samples):
                X_lstm[i] = X[i:i + self.sequence_length]
                
            return X_lstm
        return X
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> 'LSTMModel':
        """
        Train the LSTM model
        
        Args:
            X_train: Training features
            y_train: Training targets
            **kwargs: Additional arguments
        """
        # Prepare data for LSTM
        X_lstm = self._reshape_data_for_lstm(X_train)
        
        if len(y_train.shape) == 1:
            y_lstm = y_train[self.sequence_length-1:]
        else:
            y_lstm = y_train[self.sequence_length-1:]
        
        # Build model if not already built
        if self.model is None:
            self._build_model(X_lstm.shape[1:])
        
        # Train the model
        batch_size = kwargs.get('batch_size', 32)
        epochs = kwargs.get('epochs', 50)
        validation_split = kwargs.get('validation_split', 0.1)
        
        history = self.model.fit(
            X_lstm, y_lstm,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            verbose=kwargs.get('verbose', 0)
        )
        
        self.is_trained = True
        return self
    
    def _build_model(self, input_shape):
        """
        Build the LSTM model architecture
        """
        self.model = Sequential()
        
        # Add first LSTM layer
        self.model.add(LSTM(
            units=self.n_units,
            return_sequences=True if self.n_layers > 1 else False,
            input_shape=input_shape
        ))
        self.model.add(Dropout(self.dropout))
        
        # Add additional LSTM layers if needed
        for i in range(self.n_layers - 1):
            return_seq = i < self.n_layers - 2  # Return sequences for all but the last layer
            self.model.add(LSTM(units=self.n_units, return_sequences=return_seq))
            self.model.add(Dropout(self.dropout))
        
        # Output layer
        self.model.add(Dense(units=1))
        
        # Compile model
        learning_rate = 0.001
        if 'learning_rate' in locals():
            learning_rate = locals()['learning_rate']
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mean_squared_error',
            metrics=['mae']
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the LSTM model
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_lstm = self._reshape_data_for_lstm(X)
        predictions = self.model.predict(X_lstm, verbose=0)
        return predictions.flatten()
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk
        
        Args:
            path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Save the Keras model
        self.model.save(path)
    
    def load_model(self, path: str) -> 'LSTMModel':
        """
        Load a trained model from disk
        
        Args:
            path: Path to load the model from
            
        Returns:
            The loaded model instance
        """
        from tensorflow.keras.models import load_model
        self.model = load_model(path)
        self.is_trained = True
        return self