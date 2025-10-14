"""Random Forest model for OHLC price prediction."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from src.models.base_model import BaseModel

try:
    from src.utils import get_logger
except ImportError:
    # Handle case when running as script directly
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils import get_logger

logger = get_logger(__name__)

class RandomForestModel(BaseModel):
    """Random Forest model for OHLC prediction."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Random Forest model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.n_estimators = config.get('n_estimators', 100)
        self.max_depth = config.get('max_depth', 10)
        self.min_samples_split = config.get('min_samples_split', 5)
        self.min_samples_leaf = config.get('min_samples_leaf', 2)
        self.random_state = config.get('random_state', 42)
        self.n_jobs = config.get('n_jobs', -1)
        
    def build_model(self, n_outputs: int = 1) -> None:
        """Build Random Forest model.
        
        Args:
            n_outputs: Number of output targets
        """
        logger.info(f"Building Random Forest model with {n_outputs} outputs")
        
        base_model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        if n_outputs > 1:
            self.model = MultiOutputRegressor(base_model, n_jobs=self.n_jobs)
        else:
            self.model = base_model
        
        logger.info("Random Forest model built successfully")
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Train Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (not used for Random Forest)
            y_val: Validation targets (not used for Random Forest)
            
        Returns:
            Training information
        """
        logger.info("Starting Random Forest model training")
        
        # Handle sequence data by flattening if necessary
        if len(X_train.shape) == 3:
            # Reshape from (samples, sequence_length, features) to (samples, sequence_length * features)
            X_train = X_train.reshape(X_train.shape[0], -1)
            logger.info(f"Reshaped training data from 3D to 2D: {X_train.shape}")
        
        if self.model is None:
            n_outputs = y_train.shape[1] if len(y_train.shape) > 1 else 1
            self.build_model(n_outputs)
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate training score
        train_score = self.model.score(X_train, y_train)
        
        logger.info(f"Random Forest model training completed. Training R² score: {train_score:.4f}")
        
        return {
            'train_score': train_score,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with Random Forest model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Handle sequence data by flattening if necessary
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        
        predictions = self.model.predict(X)
        return predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from Random Forest.
        
        Returns:
            Dictionary of feature importances
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            # Single output model
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'estimators_'):
            # Multi-output model
            importances = np.mean([
                estimator.feature_importances_ 
                for estimator in self.model.estimators_
            ], axis=0)
        else:
            return {}
        
        # Create feature names if not available
        if not self.feature_names:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        else:
            feature_names = self.feature_names
        
        # Handle case where we flattened sequence data
        if len(feature_names) != len(importances):
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        importance_dict = dict(zip(feature_names, importances))
        
        # Sort by importance
        importance_dict = dict(sorted(
            importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        return importance_dict
    
    def get_top_features(self, n_features: int = 10) -> Dict[str, float]:
        """Get top N most important features.
        
        Args:
            n_features: Number of top features to return
            
        Returns:
            Dictionary of top features and their importance
        """
        all_importance = self.get_feature_importance()
        return dict(list(all_importance.items())[:n_features])
    
    def predict_with_uncertainty(self, X: np.ndarray) -> tuple:
        """Get predictions with uncertainty estimates.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, std_dev)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Handle sequence data
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        
        # Get predictions from all trees
        if hasattr(self.model, 'estimators_'):
            # Single output model
            tree_predictions = np.array([
                tree.predict(X) for tree in self.model.estimators_
            ])
            predictions = np.mean(tree_predictions, axis=0)
            std_dev = np.std(tree_predictions, axis=0)
        elif hasattr(self.model, 'estimators_') and hasattr(self.model.estimators_[0], 'estimators_'):
            # Multi-output model
            predictions = self.model.predict(X)
            # For multi-output, calculate std for each output separately
            std_devs = []
            for i, estimator in enumerate(self.model.estimators_):
                tree_preds = np.array([tree.predict(X) for tree in estimator.estimators_])
                std_devs.append(np.std(tree_preds, axis=0))
            std_dev = np.column_stack(std_devs)
        else:
            # Fallback
            predictions = self.model.predict(X)
            std_dev = np.zeros_like(predictions)
        
        return predictions, std_dev
    
    def partial_dependence(self, X: np.ndarray, feature_idx: int, n_points: int = 100) -> tuple:
        """Calculate partial dependence for a feature.
        
        Args:
            X: Input features
            feature_idx: Index of feature to analyze
            n_points: Number of points for partial dependence
            
        Returns:
            Tuple of (feature_values, partial_dependence)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained for partial dependence")
        
        # Handle sequence data
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        
        # Get feature range
        feature_min = X[:, feature_idx].min()
        feature_max = X[:, feature_idx].max()
        feature_values = np.linspace(feature_min, feature_max, n_points)
        
        # Calculate partial dependence
        partial_deps = []
        for value in feature_values:
            X_modified = X.copy()
            X_modified[:, feature_idx] = value
            pred = self.model.predict(X_modified).mean()
            partial_deps.append(pred)
        
        return feature_values, np.array(partial_deps)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Random Forest model information.
        
        Returns:
            Dictionary with model information
        """
        info = super().get_model_info()
        info.update({
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state
        })
        
        if self.is_trained:
            info['oob_score'] = getattr(self.model, 'oob_score_', None)
        
        return info
