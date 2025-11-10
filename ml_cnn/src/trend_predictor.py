import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from .hyperparameter_optimizer import HyperparameterOptimizer
from .cnn_architectures import CNNArchitectures


class TrendPredictor:
    """
    Trend prediction model with profit threshold.
    """
    
    def __init__(self, profit_threshold: float = 0.02, sequence_length: int = 10):
        """
        Initialize the TrendPredictor.
        
        Args:
            profit_threshold: Minimum profit percentage to consider a trend profitable
            sequence_length: Length of sequences for CNN input
        """
        self.profit_threshold = profit_threshold
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False
        
    def prepare_data(self, features_df: pd.DataFrame, target_series: pd.Series):
        """
        Prepare data for training by creating sequences and scaling features.
        
        Args:
            features_df: DataFrame with features
            target_series: Series with target values (price changes)
            
        Returns:
            tuple: (X, y) prepared for model training
        """
        # Scale the features
        scaled_features = self.scaler.fit_transform(features_df.values)
        
        # Create sequences for CNN
        X, y = [], []
        for i in range(len(scaled_features) - self.sequence_length):
            X.append(scaled_features[i:i + self.sequence_length])
            y.append(target_series.iloc[i + self.sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Convert target to binary based on profit threshold
        # 1 if profit > threshold, 0 otherwise
        y_binary = (y > self.profit_threshold).astype(int)
        
        return X, y_binary
    
    def create_target_labels(self, price_series: pd.Series):
        """
        Create target labels based on future price changes.
        
        Args:
            price_series: Series with closing prices
            
        Returns:
            Series with target labels (percentage change)
        """
        # Calculate future price change (next period)
        future_prices = price_series.shift(-1)
        price_changes = (future_prices - price_series) / price_series
        
        return price_changes[:-1]  # Remove the last NaN value
    
    def train(self, features_df: pd.DataFrame, price_series: pd.Series, optimize_hyperparams: bool = True, n_trials: int = 20):
        """
        Train the trend prediction model.
        
        Args:
            features_df: DataFrame with features
            price_series: Series with target prices
            optimize_hyperparams: Whether to optimize hyperparameters using Optuna
            n_trials: Number of optimization trials if optimizing
        """
        # Create target labels based on future price changes
        target_labels = self.create_target_labels(price_series)
        
        # Prepare data for training
        X, y = self.prepare_data(features_df, target_labels)
        
        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        if optimize_hyperparams:
            # Optimize hyperparameters using Optuna
            input_shape = (X_train.shape[1], X_train.shape[2])
            optimizer = HyperparameterOptimizer(X_train, y_train, X_val, y_val, input_shape)
            study = optimizer.optimize(n_trials=n_trials)
            
            # Get best parameters
            best_params = study.best_params
            
            # Build model with best parameters
            if best_params['architecture'] == 'simple':
                self.model = CNNArchitectures.simple_cnn(input_shape)
            elif best_params['architecture'] == 'deeper':
                self.model = CNNArchitectures.deeper_cnn(input_shape)
            elif best_params['architecture'] == 'cnn_lstm':
                self.model = CNNArchitectures.cnn_with_lstm(input_shape)
                
            # Compile model with best hyperparameters
            if best_params['optimizer'] == 'adam':
                optimizer = tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
            elif best_params['optimizer'] == 'rmsprop':
                optimizer = tf.keras.optimizers.RMSprop(learning_rate=best_params['learning_rate'])
            else:  # sgd
                optimizer = tf.keras.optimizers.SGD(learning_rate=best_params['learning_rate'])
            
            self.model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Train model
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            self.model.fit(
                X_train, y_train,
                batch_size=best_params['batch_size'],
                epochs=best_params['epochs'],
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=1
            )
            
        else:
            # Use default simple CNN model
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.model = CNNArchitectures.simple_cnn(input_shape)
            
            self.model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            self.model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=50,
                validation_data=(X_val, y_val),
                verbose=1
            )
        
        self.is_trained = True
    
    def predict(self, features_df: pd.DataFrame):
        """
        Predict trend direction based on features.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Array with predictions (0 for down trend, 1 for up trend)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale features
        scaled_features = self.scaler.transform(features_df.values)
        
        # Create sequences for prediction
        X_pred = []
        for i in range(len(scaled_features) - self.sequence_length + 1):
            X_pred.append(scaled_features[i:i + self.sequence_length])
        
        X_pred = np.array(X_pred)
        
        # Make predictions
        predictions = self.model.predict(X_pred)
        predictions_binary = (predictions > 0.5).astype(int)
        
        return predictions_binary.flatten()
    
    def predict_with_probability(self, features_df: pd.DataFrame):
        """
        Predict trend direction with probability scores.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            tuple: (predictions, probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale features
        scaled_features = self.scaler.transform(features_df.values)
        
        # Create sequences for prediction
        X_pred = []
        for i in range(len(scaled_features) - self.sequence_length + 1):
            X_pred.append(scaled_features[i:i + self.sequence_length])
        
        X_pred = np.array(X_pred)
        
        # Make predictions
        probabilities = self.model.predict(X_pred).flatten()
        predictions = (probabilities > 0.5).astype(int)
        
        return predictions, probabilities