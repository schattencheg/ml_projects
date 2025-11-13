import optuna
import tensorflow as tf
from tensorflow.keras import optimizers, layers, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .cnn_architectures import CNNArchitectures


class HyperparameterOptimizer:
    """
    Class to optimize hyperparameters using Optuna for CNN models.
    """
    
    def __init__(self, X_train, y_train, X_val, y_val, input_shape):
        """
        Initialize optimizer with training and validation data.
        
        Args:
            X_train: Training features (will be converted to float32 numpy array)
            y_train: Training labels (will be converted to int32 numpy array)
            X_val: Validation features (will be converted to float32 numpy array)
            y_val: Validation labels (will be converted to int32 numpy array)
            input_shape: Shape of input data
        """
        # Convert to numpy arrays with proper dtypes to avoid "Invalid dtype: object" error
        # Handle DataFrames with Timestamp or other non-numeric columns
        X_train = self._prepare_features(X_train)
        X_val = self._prepare_features(X_val)
            
        self.X_train = np.array(X_train, dtype=np.float32)
        self.y_train = np.array(y_train, dtype=np.int32).flatten()
        self.X_val = np.array(X_val, dtype=np.float32)
        self.y_val = np.array(y_val, dtype=np.int32).flatten()
        self.input_shape = input_shape
    
    def _prepare_features(self, X):
        """
        Prepare features by handling non-numeric data (like Timestamps).
        Works with 2D and 3D arrays.
        
        Args:
            X: Input features (DataFrame, 2D array, or 3D array)
            
        Returns:
            Cleaned numpy array with only numeric values
        """
        if isinstance(X, pd.DataFrame):
            # Select only numeric columns
            return X.select_dtypes(include=[np.number]).values
        
        elif isinstance(X, np.ndarray):
            if X.dtype == object:
                # Handle 3D arrays (sequences) - shape: (samples, sequence_length, features)
                if X.ndim == 3:
                    cleaned_sequences = []
                    for sequence in X:
                        # Convert each sequence to DataFrame
                        df = pd.DataFrame(sequence)
                        
                        # Try to convert all columns to numeric, coercing errors to NaN
                        for col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        # Now select numeric columns (should be all columns that converted successfully)
                        numeric_df = df.select_dtypes(include=[np.number])
                        
                        # Drop columns that are all NaN (couldn't be converted)
                        numeric_df = numeric_df.dropna(axis=1, how='all')
                        
                        cleaned_sequences.append(numeric_df.values)
                    return np.array(cleaned_sequences, dtype=np.float32)
                
                # Handle 2D arrays
                elif X.ndim == 2:
                    try:
                        return X.astype(np.float32)
                    except (ValueError, TypeError):
                        df = pd.DataFrame(X)
                        # Try to convert to numeric
                        for col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        numeric_df = df.select_dtypes(include=[np.number])
                        numeric_df = numeric_df.dropna(axis=1, how='all')
                        return numeric_df.values
                
                # Handle 1D arrays
                else:
                    try:
                        return X.astype(np.float32)
                    except (ValueError, TypeError):
                        # Filter out non-numeric values
                        return np.array([x for x in X if isinstance(x, (int, float, np.number))], dtype=np.float32)
            else:
                # Already numeric dtype
                return X
        
        # For other types, try direct conversion
        return np.array(X, dtype=np.float32)
        
    def objective(self, trial):
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation accuracy to maximize
        """
        # Suggest architecture type
        architecture_type = trial.suggest_categorical('architecture', ['simple', 'deeper', 'cnn_lstm'])
        
        # Suggest optimizer and learning rate
        optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd'])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        
        # Suggest batch size
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        
        # Suggest epochs (with a reasonable range)
        epochs = trial.suggest_int('epochs', 10, 50, step=10)
        
        # Build model using CNNArchitectures
        # Note: We optimize architecture selection (simple/deeper/cnn_lstm) and training
        # hyperparameters (optimizer, learning_rate, batch_size, epochs).
        # The CNN architectures themselves use predefined layer configurations.
        if architecture_type == 'simple':
            model = CNNArchitectures.simple_cnn(self.input_shape)
        elif architecture_type == 'deeper':
            model = CNNArchitectures.deeper_cnn(self.input_shape)
        elif architecture_type == 'cnn_lstm':
            model = CNNArchitectures.cnn_with_lstm(self.input_shape)
        
        # Compile model with suggested hyperparameters
        if optimizer_name == 'adam':
            optimizer = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = optimizers.RMSprop(learning_rate=learning_rate)
        else:  # sgd
            optimizer = optimizers.SGD(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model with early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        model.fit(
            self.X_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.X_val, self.y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate model on validation set
        val_predictions = model.predict(self.X_val)
        val_predictions_binary = (val_predictions > 0.5).astype(int)
        val_accuracy = accuracy_score(self.y_val, val_predictions_binary)
        
        return val_accuracy
    
    def optimize(self, n_trials: int = 50):
        """
        Run Optuna optimization.
        
        Args:
            n_trials: Number of optimization trials to run
            
        Returns:
            Optuna study with best parameters
        """
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        
        return study
