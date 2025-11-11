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
        
    def prepare_data(self, features_df: pd.DataFrame, target_series):
        """
        Prepare data for training by creating sequences and scaling features.
        
        Args:
            features_df: DataFrame with features
            target_series: Array or Series with target values (price changes or binary labels)
            
        Returns:
            tuple: (X, y) prepared for model training
        """
        # Scale the features
        float_cols = features_df.select_dtypes(include=['float64']).columns
        scaled_features = features_df.copy()
        scaled_features[float_cols] = self.scaler.fit_transform(features_df[float_cols].values)
        
        # Convert target_series to numpy array if it's a pandas Series
        if isinstance(target_series, pd.Series):
            target_array = target_series.values
        else:
            target_array = target_series
        
        # Create sequences for CNN
        # Note: target_array may be shorter than features_df due to future_period removal
        # We need to ensure we don't go out of bounds
        max_index = min(len(scaled_features), len(target_array))
        
        X, y = [], []
        for i in range(max_index - self.sequence_length):
            X.append(scaled_features.iloc[i:i + self.sequence_length].values)
            y.append(target_array[i + self.sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        return X, y
    
    def _prepare_features_for_training(self, X):
        """
        Prepare features for training by ensuring proper numeric dtypes.
        Handles arrays with object dtype that may contain Timestamps.
        
        Args:
            X: Input features (numpy array)
            
        Returns:
            Cleaned numpy array with float32 dtype
        """
        if isinstance(X, np.ndarray) and X.dtype == object:
            # Handle 3D arrays (sequences)
            if X.ndim == 3:
                cleaned_sequences = []
                for sequence in X:
                    df = pd.DataFrame(sequence)
                    # Convert to numeric, coercing errors to NaN
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    numeric_df = df.select_dtypes(include=[np.number])
                    numeric_df = numeric_df.dropna(axis=1, how='all')
                    cleaned_sequences.append(numeric_df.values)
                return np.array(cleaned_sequences, dtype=np.float32)
        
        # Already numeric or other type
        return np.array(X, dtype=np.float32)
    
    def create_target_labels(self, price_series: pd.Series, future_period: int = 15, threshold: float = 0.02):
        """
        Create target labels based on future price changes.
        
        Args:
            price_series: Series with closing prices
            
        Returns:
            Series with target labels (percentage change)
        """
        # Calculate future price change (next period)
        future_prices = price_series.shift(-future_period)
        price_changes = (future_prices - price_series) / price_series

        # Convert target to binary based on profit threshold
        # 1 if profit > threshold, -1 if profit < -threshold, 0 otherwise
        indices_pos = (price_changes > threshold).astype(int)
        indices_neg = (price_changes < -threshold).astype(int)
        y_binary = np.zeros_like(price_changes)
        y_binary[indices_pos] = 1
        y_binary[indices_neg] = -1        
        return y_binary[:-future_period]  # Remove the last NaN value
    
    def train(self, features_df: pd.DataFrame, 
                    price_series: pd.Series, 
                    optimize_hyperparams: bool = True, 
                    n_trials: int = 20,
                    future_period: int = 15,
                    threshold: float = 0.02):
        """
        Train the trend prediction model.
        
        Args:
            features_df: DataFrame with features
            price_series: Series with target prices
            optimize_hyperparams: Whether to optimize hyperparameters using Optuna
            n_trials: Number of optimization trials if optimizing
            future_period: Number of periods to look ahead for price change
            threshold: Profit threshold for trend prediction
        """
        # Create target labels based on future price changes
        target_labels = self.create_target_labels(price_series, future_period, threshold)
        
        # Prepare data for training
        X, y = self.prepare_data(features_df, target_labels)
        
        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Ensure data has proper numeric dtypes (handle object dtypes with Timestamps)
        X_train = self._prepare_features_for_training(X_train)
        X_val = self._prepare_features_for_training(X_val)
        y_train = np.array(y_train, dtype=np.int32)
        y_val = np.array(y_val, dtype=np.int32)
        
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
        
        # Scale features - select only numeric columns
        float_cols = features_df.select_dtypes(include=[np.number]).columns
        scaled_features = features_df.copy()
        
        # Only transform the numeric columns that were used during training
        # The scaler expects the same number of features it was trained on
        if len(float_cols) > 0:
            scaled_features[float_cols] = self.scaler.transform(features_df[float_cols].values)
        
        # Create sequences for prediction
        X_pred = []
        for i in range(len(scaled_features) - self.sequence_length + 1):
            X_pred.append(scaled_features.iloc[i:i + self.sequence_length].values)
        
        X_pred = np.array(X_pred)
        
        # Clean data to ensure proper numeric dtypes (remove any Timestamps)
        X_pred = self._prepare_features_for_training(X_pred)
        
        # Make predictions
        probabilities = self.model.predict(X_pred).flatten()
        predictions = (probabilities > 0.5).astype(int)
        
        return predictions, probabilities
