import optuna
import tensorflow as tf
from tensorflow.keras import optimizers
from sklearn.metrics import accuracy_score
import numpy as np
from .cnn_architectures import CNNArchitectures


class HyperparameterOptimizer:
    """
    Class to optimize hyperparameters using Optuna for CNN models.
    """
    
    def __init__(self, X_train, y_train, X_val, y_val, input_shape):
        """
        Initialize optimizer with training and validation data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            input_shape: Shape of input data
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.input_shape = input_shape
        
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
        
        # Suggest hyperparameters based on architecture
        if architecture_type == 'simple':
            filters_1 = trial.suggest_int('filters_1', 16, 64, step=16)
            filters_2 = trial.suggest_int('filters_2', 32, 128, step=16)
            filters_3 = trial.suggest_int('filters_3', 64, 256, step=32)
            dense_units = trial.suggest_int('dense_units', 20, 100, step=10)
        elif architecture_type == 'deeper':
            filters_1 = trial.suggest_int('filters_1', 32, 128, step=16)
            filters_2 = trial.suggest_int('filters_2', 32, 128, step=16)
            filters_3 = trial.suggest_int('filters_3', 64, 256, step=32)
            dense_units_1 = trial.suggest_int('dense_units_1', 256, 512, step=64)
            dense_units_2 = trial.suggest_int('dense_units_2', 128, 256, step=32)
            dropout_1 = trial.suggest_float('dropout_1', 0.1, 0.5)
            dropout_2 = trial.suggest_float('dropout_2', 0.2, 0.7)
        elif architecture_type == 'cnn_lstm':
            filters_1 = trial.suggest_int('filters_1', 32, 128, step=16)
            filters_2 = trial.suggest_int('filters_2', 32, 128, step=16)
            lstm_units_1 = trial.suggest_int('lstm_units_1', 50, 200, step=25)
            lstm_units_2 = trial.suggest_int('lstm_units_2', 50, 200, step=25)
            dense_units_1 = trial.suggest_int('dense_units_1', 128, 512, step=64)
            dense_units_2 = trial.suggest_int('dense_units_2', 64, 256, step=32)
            dropout_1 = trial.suggest_float('dropout_1', 0.2, 0.7)
            dropout_2 = trial.suggest_float('dropout_2', 0.2, 0.7)
            dropout_3 = trial.suggest_float('dropout_3', 0.2, 0.7)
        
        # Suggest optimizer and learning rate
        optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd'])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        
        # Suggest batch size
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        
        # Suggest epochs (with a reasonable range)
        epochs = trial.suggest_int('epochs', 10, 50, step=10)
        
        # Build model based on architecture type
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