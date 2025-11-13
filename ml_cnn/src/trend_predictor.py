import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from datetime import datetime
import os
from .hyperparameter_optimizer import HyperparameterOptimizer
from .cnn_architectures import CNNArchitectures
from .mlflow_tracker import MLflowTracker


class TrendPredictor:
    """
    Trend prediction model with profit threshold.
    """
    
    def __init__(self, profit_threshold: float = 0.02, sequence_length: int = 10, 
                 use_mlflow: bool = True, mlflow_uri: str = 'http://localhost:5000'):
        """
        Initialize the TrendPredictor.
        
        Args:
            profit_threshold: Minimum profit percentage to consider a trend profitable
            sequence_length: Length of sequences for CNN input
            use_mlflow: Whether to use MLflow tracking (default: True)
            mlflow_uri: MLflow tracking URI (default: http://localhost:5000)
        """
        self.profit_threshold = profit_threshold
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False
        self.feature_columns = None  # Store feature columns used during training
        self.input_shape = None  # Store input shape used during training
        
        # Initialize MLflow tracker
        self.mlflow_tracker = MLflowTracker(
            tracking_uri=mlflow_uri,
            experiment_name='trend_predictor',
            enabled=use_mlflow
        )
        
    def prepare_data(self, features_df: pd.DataFrame, target_series):
        """
        Prepare data for training by creating sequences and scaling features.
        
        Args:
            features_df: DataFrame with features
            target_series: Array or Series with target values (price changes or binary labels)
            
        Returns:
            tuple: (X, y) prepared for model training
        """
        # Scale the features - select only float64 columns
        float_cols = features_df.select_dtypes(include=['float64']).columns
        # Store the feature columns for later use in prediction
        self.feature_columns = float_cols.tolist()
        
        # Use only the float columns for training
        features_to_use = features_df[float_cols].copy()
        scaled_features = self.scaler.fit_transform(features_to_use.values)
        scaled_features_df = pd.DataFrame(scaled_features, columns=float_cols, index=features_df.index)
        
        # Convert target_series to numpy array if it's a pandas Series
        if isinstance(target_series, pd.Series):
            target_array = target_series.values
        else:
            target_array = target_series
        
        # Create sequences for CNN
        # Note: target_array may be shorter than features_df due to future_period removal
        # We need to ensure we don't go out of bounds
        max_index = min(len(scaled_features_df), len(target_array))
        
        X, y = [], []
        for i in range(max_index - self.sequence_length):
            X.append(scaled_features_df.iloc[i:i + self.sequence_length].values)
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
                    threshold: float = 0.02,
                    model_name: str = None,
                    ticker: str = None):
        """
        Train the trend prediction model.
        
        Args:
            features_df: DataFrame with features
            price_series: Series with target prices
            optimize_hyperparams: Whether to optimize hyperparameters using Optuna
            n_trials: Number of optimization trials if optimizing
            future_period: Number of periods to look ahead for price change
            threshold: Profit threshold for trend prediction
            model_name: Name for the model (for MLflow tracking)
            ticker: Ticker symbol (for MLflow tracking)
        """
        # Generate model name if not provided
        if model_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"trend_predictor_{timestamp}"
        
        # Start MLflow training run
        run_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if ticker:
            run_name = f"{ticker}_{run_name}"
        
        # Create target labels based on future price changes
        target_labels = self.create_target_labels(price_series, future_period, threshold)
        
        # Prepare data for training
        X, y = self.prepare_data(features_df, target_labels)
        
        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        
        # Ensure data has proper numeric dtypes (handle object dtypes with Timestamps)
        X_train = self._prepare_features_for_training(X_train)
        X_val = self._prepare_features_for_training(X_val)
        y_train = np.array(y_train, dtype=np.int32)
        y_val = np.array(y_val, dtype=np.int32)
        
        # Prepare parameters for MLflow logging
        training_params = {
            'profit_threshold': self.profit_threshold,
            'sequence_length': self.sequence_length,
            'optimize_hyperparams': optimize_hyperparams,
            'n_trials': n_trials if optimize_hyperparams else 0,
            'future_period': future_period,
            'threshold': threshold,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'num_features': X_train.shape[2],
            'model_name': model_name
        }
        if ticker:
            training_params['ticker'] = ticker
        
        # Store input shape for later validation
        self.input_shape = (X_train.shape[1], X_train.shape[2])
        
        if optimize_hyperparams:
            # Optimize hyperparameters using Optuna
            optimizer = HyperparameterOptimizer(X_train, y_train, X_val, y_val, self.input_shape)
            study = optimizer.optimize(n_trials=n_trials)
            
            # Get best parameters
            best_params = study.best_params
            
            # Build model with best parameters
            if best_params['architecture'] == 'simple':
                self.model = CNNArchitectures.simple_cnn(self.input_shape)
            elif best_params['architecture'] == 'deeper':
                self.model = CNNArchitectures.deeper_cnn(self.input_shape)
            elif best_params['architecture'] == 'cnn_lstm':
                self.model = CNNArchitectures.cnn_with_lstm(self.input_shape)
                
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
            
            history = self.model.fit(
                X_train, y_train,
                batch_size=best_params['batch_size'],
                epochs=best_params['epochs'],
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=1
            )
            
            # Add best params to training_params
            training_params.update(best_params)
            training_params['architecture'] = best_params['architecture']
            
        else:
            # Use default simple CNN model
            self.model = CNNArchitectures.simple_cnn(self.input_shape)
            
            self.model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            history = self.model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=50,
                validation_data=(X_val, y_val),
                verbose=1
            )
            
            # Add default params
            training_params['architecture'] = 'simple'
            training_params['optimizer'] = 'adam'
            training_params['batch_size'] = 32
            training_params['epochs'] = 50
        
        self.is_trained = True
        
        # Calculate final validation metrics
        y_val_pred = self.model.predict(X_val, verbose=0)
        y_val_pred_binary = (y_val_pred > 0.5).astype(int).flatten()
        
        val_accuracy = accuracy_score(y_val, y_val_pred_binary)
        val_precision = precision_score(y_val, y_val_pred_binary, zero_division=0)
        val_recall = recall_score(y_val, y_val_pred_binary, zero_division=0)
        val_f1 = f1_score(y_val, y_val_pred_binary, zero_division=0)
        
        metrics = {
            'val_accuracy': float(val_accuracy),
            'val_precision': float(val_precision),
            'val_recall': float(val_recall),
            'val_f1_score': float(val_f1),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1])
        }
        
        # Log to MLflow
        print("\n" + "="*80)
        print("LOGGING TO MLFLOW")
        print("="*80)
        
        # Create a simple MLflow run for this training session
        import mlflow
        import mlflow.keras
        
        if self.mlflow_tracker.is_enabled():
            try:
                with mlflow.start_run(run_name=run_name):
                    # Log all parameters
                    for key, value in training_params.items():
                        mlflow.log_param(key, value)
                    
                    # Log all metrics
                    for key, value in metrics.items():
                        mlflow.log_metric(key, value)
                    
                    # Log model
                    mlflow.keras.log_model(
                        self.model, 
                        "model",
                        registered_model_name=f"trend_predictor_{model_name}"
                    )
                    
                    # Set tags
                    mlflow.set_tag('model_type', 'cnn_trend_predictor')
                    mlflow.set_tag('framework', 'tensorflow')
                    if ticker:
                        mlflow.set_tag('ticker', ticker)
                    
                    run_id = mlflow.active_run().info.run_id
                    print(f"✓ Model logged to MLflow registry: trend_predictor_{model_name}")
                    print(f"✓ MLflow Run ID: {run_id}")
                    print(f"  View at: {self.mlflow_tracker.get_tracking_uri()}")
                    
            except Exception as e:
                print(f"✗ MLflow logging failed: {str(e)}")
                print("  Model training completed but not logged to MLflow")
        else:
            print("MLflow tracking is disabled")
        
        print("="*80 + "\n")
        
        # Print training summary
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Model: {model_name}")
        if ticker:
            print(f"Ticker: {ticker}")
        print(f"\nValidation Metrics:")
        print(f"  Accuracy:  {val_accuracy:.4f}")
        print(f"  Precision: {val_precision:.4f}")
        print(f"  Recall:    {val_recall:.4f}")
        print(f"  F1 Score:  {val_f1:.4f}")
        print("="*80 + "\n")
    
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
        
        if self.feature_columns is None:
            raise ValueError("Model was not properly trained - feature columns not stored")
        
        # Use only the feature columns that were used during training
        scaled_features = self.scaler.transform(features_df[self.feature_columns].values)
        
        # Create sequences for prediction
        X_pred = []
        for i in range(len(scaled_features) - self.sequence_length + 1):
            X_pred.append(scaled_features[i:i + self.sequence_length])
        
        X_pred = np.array(X_pred)
        
        # Make predictions
        predictions = self.model.predict(X_pred)
        predictions_binary = (predictions > 0.5).astype(int)
        
        return predictions_binary.flatten()
    
    def predict_with_probability(self, features_df: pd.DataFrame, feature_cols: list = None):
        """
        Predict trend direction with probability scores.
        
        Args:
            features_df: DataFrame with features
            feature_cols: Optional list of feature columns to use (deprecated, kept for compatibility)
            
        Returns:
            tuple: (predictions, probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.feature_columns is None:
            raise ValueError("Model was not properly trained - feature columns not stored")
        
        if self.input_shape is None:
            raise ValueError("Model was not properly trained - input shape not stored")
        
        # Use only the feature columns that were used during training
        # This ensures the scaler gets the same features it was trained on
        scaled_features = features_df[self.feature_columns].copy()
        
        # Transform using the fitted scaler
        scaled_features[self.feature_columns] = self.scaler.transform(features_df[self.feature_columns].values)
        
        # Create sequences for prediction
        X_pred = []
        for i in range(len(scaled_features) - self.sequence_length + 1):
            X_pred.append(scaled_features.iloc[i:i + self.sequence_length].values)
        
        if len(X_pred) == 0:
            raise ValueError(f"Not enough data to create sequences. Need at least {self.sequence_length} samples.")
        
        X_pred = np.array(X_pred, dtype=np.float32)
        
        # Clean data to ensure proper numeric dtypes (remove any Timestamps)
        X_pred = self._prepare_features_for_training(X_pred)
        
        # Validate shape matches expected input
        expected_shape = (X_pred.shape[0], self.input_shape[0], self.input_shape[1])
        if X_pred.shape != expected_shape:
            print(f"Warning: Input shape {X_pred.shape} doesn't match expected {expected_shape}")
            print(f"Attempting to reshape...")
            # Try to fix the shape if possible
            if X_pred.shape[1:] != self.input_shape:
                raise ValueError(
                    f"Feature shape mismatch. Expected {self.input_shape}, got {X_pred.shape[1:]}. "
                    f"Model was trained with {self.input_shape[1]} features."
                )
        
        # Ensure data is float32 and contiguous
        X_pred = np.ascontiguousarray(X_pred, dtype=np.float32)
        
        # Make predictions with verbose=0 to suppress output
        try:
            probabilities = self.model.predict(X_pred, verbose=0).flatten()
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            print(f"Input shape: {X_pred.shape}, dtype: {X_pred.dtype}")
            print(f"Expected input shape: {self.input_shape}")
            raise
        
        predictions = (probabilities > 0.5).astype(int)
        
        return predictions, probabilities
