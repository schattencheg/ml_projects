"""XGBoost model for OHLC price prediction."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

from src.models.base_model import BaseModel
from src.utils import get_logger

logger = get_logger(__name__)

class XGBoostModel(BaseModel):
    """XGBoost model for OHLC prediction."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize XGBoost model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.n_estimators = config.get('n_estimators', 100)
        self.max_depth = config.get('max_depth', 6)
        self.learning_rate = config.get('learning_rate', 0.1)
        self.subsample = config.get('subsample', 0.8)
        self.colsample_bytree = config.get('colsample_bytree', 0.8)
        self.random_state = config.get('random_state', 42)
        self.n_jobs = config.get('n_jobs', -1)
        self.early_stopping_rounds = config.get('early_stopping_rounds', 20)
        
    def build_model(self, n_outputs: int = 1) -> None:
        """Build XGBoost model.
        
        Args:
            n_outputs: Number of output targets
        """
        logger.info(f"Building XGBoost model with {n_outputs} outputs")
        
        base_model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=1
        )
        
        if n_outputs > 1:
            self.model = MultiOutputRegressor(base_model, n_jobs=self.n_jobs)
        else:
            self.model = base_model
        
        logger.info("XGBoost model built successfully")
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Training information
        """
        logger.info("Starting XGBoost model training")
        
        # Handle sequence data by flattening if necessary
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
            logger.info(f"Reshaped training data from 3D to 2D: {X_train.shape}")
        
        if X_val is not None and len(X_val.shape) == 3:
            X_val = X_val.reshape(X_val.shape[0], -1)
        
        if self.model is None:
            n_outputs = y_train.shape[1] if len(y_train.shape) > 1 else 1
            self.build_model(n_outputs)
        
        # Prepare training arguments
        fit_params = {}
        
        # Add early stopping if validation data is provided and model supports it
        if X_val is not None and y_val is not None:
            if hasattr(self.model, 'fit') and not isinstance(self.model, MultiOutputRegressor):
                # Single output XGBoost with early stopping
                fit_params['eval_set'] = [(X_val, y_val)]
                fit_params['early_stopping_rounds'] = self.early_stopping_rounds
                fit_params['verbose'] = False
        
        # Train model
        self.model.fit(X_train, y_train, **fit_params)
        self.is_trained = True
        
        # Calculate training score
        train_score = self.model.score(X_train, y_train)
        val_score = None
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
        
        logger.info(f"XGBoost model training completed. Training R² score: {train_score:.4f}")
        if val_score is not None:
            logger.info(f"Validation R² score: {val_score:.4f}")
        
        training_info = {
            'train_score': train_score,
            'val_score': val_score,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate
        }
        
        # Add best iteration if available
        if hasattr(self.model, 'best_iteration'):
            training_info['best_iteration'] = self.model.best_iteration
        
        return training_info
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with XGBoost model.
        
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
        """Get feature importance from XGBoost.
        
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
    
    def get_booster_importance(self, importance_type: str = 'gain') -> Dict[str, float]:
        """Get feature importance from XGBoost booster.
        
        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover')
            
        Returns:
            Dictionary of feature importances
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        if isinstance(self.model, MultiOutputRegressor):
            # For multi-output, get importance from first estimator
            booster = self.model.estimators_[0].get_booster()
        else:
            booster = self.model.get_booster()
        
        importance_dict = booster.get_score(importance_type=importance_type)
        
        # Sort by importance
        importance_dict = dict(sorted(
            importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        return importance_dict
    
    def plot_importance(self, max_num_features: int = 20, importance_type: str = 'gain'):
        """Plot feature importance.
        
        Args:
            max_num_features: Maximum number of features to plot
            importance_type: Type of importance to plot
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to plot importance")
        
        try:
            if isinstance(self.model, MultiOutputRegressor):
                # Plot for first estimator
                xgb.plot_importance(
                    self.model.estimators_[0], 
                    max_num_features=max_num_features,
                    importance_type=importance_type
                )
            else:
                xgb.plot_importance(
                    self.model, 
                    max_num_features=max_num_features,
                    importance_type=importance_type
                )
        except Exception as e:
            logger.warning(f"Could not plot importance: {str(e)}")
    
    def get_shap_values(self, X: np.ndarray, max_samples: int = 1000):
        """Get SHAP values for interpretability.
        
        Args:
            X: Input features
            max_samples: Maximum number of samples to compute SHAP for
            
        Returns:
            SHAP values
        """
        try:
            import shap
            
            if not self.is_trained:
                raise ValueError("Model must be trained to get SHAP values")
            
            # Handle sequence data
            if len(X.shape) == 3:
                X = X.reshape(X.shape[0], -1)
            
            # Limit samples for performance
            if len(X) > max_samples:
                X = X[:max_samples]
            
            if isinstance(self.model, MultiOutputRegressor):
                # For multi-output, use first estimator
                explainer = shap.TreeExplainer(self.model.estimators_[0])
            else:
                explainer = shap.TreeExplainer(self.model)
            
            shap_values = explainer.shap_values(X)
            return shap_values
            
        except ImportError:
            logger.warning("SHAP not installed. Install with: pip install shap")
            return None
        except Exception as e:
            logger.error(f"Error computing SHAP values: {str(e)}")
            return None
    
    def predict_with_leaf_index(self, X: np.ndarray) -> tuple:
        """Get predictions along with leaf indices.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, leaf_indices)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Handle sequence data
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        
        predictions = self.model.predict(X)
        
        try:
            if isinstance(self.model, MultiOutputRegressor):
                # Get leaf indices from first estimator
                leaf_indices = self.model.estimators_[0].apply(X)
            else:
                leaf_indices = self.model.apply(X)
        except:
            leaf_indices = None
        
        return predictions, leaf_indices
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get XGBoost model information.
        
        Returns:
            Dictionary with model information
        """
        info = super().get_model_info()
        info.update({
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'random_state': self.random_state
        })
        
        if self.is_trained:
            try:
                if isinstance(self.model, MultiOutputRegressor):
                    info['best_iteration'] = getattr(self.model.estimators_[0], 'best_iteration', None)
                else:
                    info['best_iteration'] = getattr(self.model, 'best_iteration', None)
            except:
                pass
        
        return info
