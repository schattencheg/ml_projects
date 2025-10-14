"""Data preprocessing utilities."""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pathlib import Path

try:
    from src.utils import config, get_logger
except ImportError:
    # Handle case when running as script directly
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from src.utils import config, get_logger

logger = get_logger(__name__)

class DataPreprocessor:
    """Data preprocessing for OHLC market data."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize preprocessor with configuration.
        
        Args:
            config_dict: Preprocessing configuration dictionary
        """
        self.config = config_dict or config.data_config.get('preprocessing', {})
        self.scalers = {}
        
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean raw market data.
        
        Args:
            data: Raw market data DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning")
        df = data.copy()
        
        # Remove duplicates
        initial_len = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_len:
            logger.info(f"Removed {initial_len - len(df)} duplicate rows")
        
        # Handle missing values
        fill_method = self.config.get('fill_method', 'forward')
        if fill_method == 'forward':
            df = df.fillna(method='ffill')
        elif fill_method == 'backward':
            df = df.fillna(method='bfill')
        elif fill_method == 'interpolate':
            df = df.interpolate()
        
        # Remove remaining NaN values
        df = df.dropna()
        
        # Remove outliers if configured
        if self.config.get('remove_outliers', False):
            df = self._remove_outliers(df)
        
        # Ensure minimum periods
        min_periods = self.config.get('min_periods', 30)
        if len(df) < min_periods:
            raise ValueError(f"Insufficient data: {len(df)} < {min_periods} required periods")
        
        logger.info(f"Data cleaning completed. Final shape: {df.shape}")
        return df
    
    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using z-score method.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with outliers removed
        """
        threshold = self.config.get('outlier_threshold', 3.0)
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        for col in numeric_columns:
            if col in data.columns:
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outliers = z_scores > threshold
                data = data[~outliers]
        
        logger.info(f"Outlier removal completed with threshold {threshold}")
        return data
    
    def create_sequences(
        self, 
        data: pd.DataFrame, 
        sequence_length: int = 60,
        target_columns: List[str] = None,
        prediction_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series modeling.
        
        Args:
            data: Input DataFrame
            sequence_length: Length of input sequences
            target_columns: Columns to predict
            prediction_horizon: Number of steps to predict ahead
            
        Returns:
            Tuple of (X, y) arrays
        """
        if target_columns is None:
            target_columns = ['Close']
        
        logger.info(f"Creating sequences with length {sequence_length}")
        
        # Select features and targets
        feature_columns = [col for col in data.columns if col not in ['Symbol']]
        X_data = data[feature_columns].values
        y_data = data[target_columns].values
        
        X, y = [], []
        
        for i in range(sequence_length, len(X_data) - prediction_horizon + 1):
            # Input sequence
            X.append(X_data[i-sequence_length:i])
            
            # Target (future values)
            if prediction_horizon == 1:
                y.append(y_data[i])
            else:
                y.append(y_data[i:i+prediction_horizon])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created {len(X)} sequences with shape X: {X.shape}, y: {y.shape}")
        return X, y
    
    def scale_data(
        self, 
        data: pd.DataFrame, 
        scaler_type: str = 'standard',
        fit_scaler: bool = True
    ) -> pd.DataFrame:
        """Scale numerical data.
        
        Args:
            data: Input DataFrame
            scaler_type: Type of scaler ('standard', 'minmax')
            fit_scaler: Whether to fit the scaler
            
        Returns:
            Scaled DataFrame
        """
        logger.info(f"Scaling data with {scaler_type} scaler")
        
        df = data.copy()
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if scaler_type not in self.scalers or fit_scaler:
            if scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaler type: {scaler_type}")
            
            if fit_scaler:
                scaler.fit(df[numeric_columns])
                self.scalers[scaler_type] = scaler
        else:
            scaler = self.scalers[scaler_type]
        
        df[numeric_columns] = scaler.transform(df[numeric_columns])
        return df
    
    def inverse_scale_data(
        self, 
        data: np.ndarray, 
        scaler_type: str = 'standard',
        columns: List[str] = None
    ) -> np.ndarray:
        """Inverse transform scaled data.
        
        Args:
            data: Scaled data array
            scaler_type: Type of scaler used
            columns: Column names (for partial inverse transform)
            
        Returns:
            Original scale data
        """
        if scaler_type not in self.scalers:
            raise ValueError(f"Scaler {scaler_type} not fitted")
        
        scaler = self.scalers[scaler_type]
        
        if columns is not None:
            # Create dummy array with all features
            n_features = len(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else scaler.n_features_in_
            dummy_array = np.zeros((data.shape[0], n_features))
            
            # Fill in the columns we want to inverse transform
            for i, col in enumerate(columns):
                if hasattr(scaler, 'feature_names_in_'):
                    col_idx = list(scaler.feature_names_in_).index(col)
                else:
                    col_idx = i
                dummy_array[:, col_idx] = data[:, i]
            
            # Inverse transform and extract relevant columns
            inverse_transformed = scaler.inverse_transform(dummy_array)
            result = np.zeros_like(data)
            for i, col in enumerate(columns):
                if hasattr(scaler, 'feature_names_in_'):
                    col_idx = list(scaler.feature_names_in_).index(col)
                else:
                    col_idx = i
                result[:, i] = inverse_transformed[:, col_idx]
            
            return result
        else:
            return scaler.inverse_transform(data)
    
    def split_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        test_size: float = 0.2,
        validation_size: float = 0.1,
        shuffle: bool = False
    ) -> Tuple[np.ndarray, ...]:
        """Split data into train, validation, and test sets.
        
        Args:
            X: Feature array
            y: Target array
            test_size: Proportion of test data
            validation_size: Proportion of validation data
            shuffle: Whether to shuffle data before splitting
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info(f"Splitting data: test_size={test_size}, validation_size={validation_size}")
        
        n_samples = len(X)
        
        if shuffle:
            indices = np.random.permutation(n_samples)
            X = X[indices]
            y = y[indices]
        
        # Calculate split indices
        test_idx = int(n_samples * (1 - test_size))
        val_idx = int(test_idx * (1 - validation_size))
        
        # Split data
        X_train = X[:val_idx]
        X_val = X[val_idx:test_idx]
        X_test = X[test_idx:]
        
        y_train = y[:val_idx]
        y_val = y[val_idx:test_idx]
        y_test = y[test_idx:]
        
        logger.info(f"Data split completed:")
        logger.info(f"  Train: {len(X_train)} samples")
        logger.info(f"  Validation: {len(X_val)} samples")
        logger.info(f"  Test: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_scalers(self, filepath: str):
        """Save fitted scalers to file.
        
        Args:
            filepath: Path to save scalers
        """
        import joblib
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scalers, filepath)
        logger.info(f"Scalers saved to {filepath}")
    
    def load_scalers(self, filepath: str):
        """Load scalers from file.
        
        Args:
            filepath: Path to load scalers from
        """
        import joblib
        
        self.scalers = joblib.load(filepath)
        logger.info(f"Scalers loaded from {filepath}")
    
    def get_preprocessing_pipeline(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply complete preprocessing pipeline.
        
        Args:
            data: Raw market data
            
        Returns:
            Preprocessed data
        """
        logger.info("Starting preprocessing pipeline")
        
        # Clean data
        data = self.clean_data(data)
        
        # Scale data
        data = self.scale_data(data, scaler_type='standard')
        
        logger.info("Preprocessing pipeline completed")
        return data
