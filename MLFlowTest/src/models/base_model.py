"""Base model class for OHLC prediction models."""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import joblib
from pathlib import Path

try:
    from src.utils import get_logger
except ImportError:
    # Handle case when running as script directly
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils import get_logger

logger = get_logger(__name__)

class BaseModel(ABC):
    """Abstract base class for all prediction models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize base model.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model = None
        self.is_trained = False
        self.feature_names = []
        self.target_columns = []
        
    @abstractmethod
    def build_model(self) -> None:
        """Build the model architecture."""
        pass
    
    @abstractmethod
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Training history/metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        pass
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        predictions = self.predict(X_test)
        
        # Handle multi-output case
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            metrics = {}
            for i, target in enumerate(self.target_columns):
                y_true = y_test[:, i] if len(y_test.shape) > 1 else y_test
                y_pred = predictions[:, i] if len(predictions.shape) > 1 else predictions
                
                metrics[f'{target}_MAE'] = mean_absolute_error(y_true, y_pred)
                metrics[f'{target}_MSE'] = mean_squared_error(y_true, y_pred)
                metrics[f'{target}_RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
                metrics[f'{target}_R2'] = r2_score(y_true, y_pred)
                
                # MAPE (Mean Absolute Percentage Error)
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                metrics[f'{target}_MAPE'] = mape
        else:
            metrics = {
                'MAE': mean_absolute_error(y_test, predictions),
                'MSE': mean_squared_error(y_test, predictions),
                'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
                'R2': r2_score(y_test, predictions),
                'MAPE': np.mean(np.abs((y_test - predictions) / y_test)) * 100
            }
        
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """Save model to file.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'config': self.config,
            'feature_names': self.feature_names,
            'target_columns': self.target_columns,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load model from file.
        
        Args:
            filepath: Path to load model from
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.config = model_data['config']
        self.feature_names = model_data['feature_names']
        self.target_columns = model_data['target_columns']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if supported by model.
        
        Returns:
            Dictionary of feature importances or None
        """
        return None
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Get prediction probabilities if supported.
        
        Args:
            X: Input features
            
        Returns:
            Prediction probabilities or None
        """
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': self.__class__.__name__,
            'config': self.config,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names),
            'target_columns': self.target_columns
        }
