"""Ensemble model combining multiple prediction models."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from src.models.base_model import BaseModel
from src.models.lstm_model import LSTMModel
from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.utils import get_logger

logger = get_logger(__name__)

class EnsembleModel(BaseModel):
    """Ensemble model combining multiple base models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ensemble model.
        
        Args:
            config: Ensemble configuration
        """
        super().__init__(config)
        self.base_models = {}
        self.weights = None
        self.meta_model = None
        self.ensemble_method = config.get('method', 'weighted_average')  # 'weighted_average', 'stacking', 'voting'
        self.use_meta_model = config.get('use_meta_model', False)
        
    def add_model(self, name: str, model: BaseModel) -> None:
        """Add a base model to the ensemble.
        
        Args:
            name: Model name
            model: Model instance
        """
        self.base_models[name] = model
        logger.info(f"Added {name} model to ensemble")
    
    def build_model(self) -> None:
        """Build ensemble model."""
        logger.info(f"Building ensemble with {len(self.base_models)} base models")
        
        if len(self.base_models) == 0:
            raise ValueError("No base models added to ensemble")
        
        # Initialize weights equally
        n_models = len(self.base_models)
        self.weights = np.ones(n_models) / n_models
        
        # Initialize meta model if using stacking
        if self.ensemble_method == 'stacking' or self.use_meta_model:
            self.meta_model = LinearRegression()
        
        logger.info("Ensemble model built successfully")
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Train ensemble model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Training information
        """
        logger.info("Starting ensemble model training")
        
        if len(self.base_models) == 0:
            raise ValueError("No base models to train")
        
        training_results = {}
        base_predictions = []
        
        # Train each base model
        for name, model in self.base_models.items():
            logger.info(f"Training {name} model")
            try:
                result = model.train(X_train, y_train, X_val, y_val)
                training_results[name] = result
                
                # Get predictions for meta-model training
                if self.ensemble_method == 'stacking' or self.use_meta_model:
                    pred = model.predict(X_train)
                    base_predictions.append(pred)
                
            except Exception as e:
                logger.error(f"Error training {name} model: {str(e)}")
                continue
        
        # Optimize ensemble weights or train meta-model
        if self.ensemble_method == 'weighted_average':
            self._optimize_weights(X_val, y_val)
        elif self.ensemble_method == 'stacking' and len(base_predictions) > 0:
            self._train_meta_model(base_predictions, y_train)
        
        self.is_trained = True
        logger.info("Ensemble model training completed")
        
        return {
            'base_models': training_results,
            'ensemble_method': self.ensemble_method,
            'weights': self.weights.tolist() if self.weights is not None else None
        }
    
    def _optimize_weights(self, X_val: Optional[np.ndarray], y_val: Optional[np.ndarray]) -> None:
        """Optimize ensemble weights using validation data.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
        """
        if X_val is None or y_val is None:
            logger.info("No validation data provided, using equal weights")
            return
        
        logger.info("Optimizing ensemble weights")
        
        # Get predictions from all models
        predictions = []
        for name, model in self.base_models.items():
            if model.is_trained:
                pred = model.predict(X_val)
                predictions.append(pred)
        
        if len(predictions) == 0:
            logger.warning("No trained models for weight optimization")
            return
        
        predictions = np.array(predictions)
        
        # Simple grid search for optimal weights
        best_weights = None
        best_score = float('inf')
        
        # Try different weight combinations
        from itertools import product
        
        weight_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        n_models = len(predictions)
        
        if n_models == 2:
            for w1 in weight_options:
                w2 = 1.0 - w1
                weights = np.array([w1, w2])
                ensemble_pred = np.average(predictions, axis=0, weights=weights)
                score = mean_squared_error(y_val, ensemble_pred)
                
                if score < best_score:
                    best_score = score
                    best_weights = weights
        else:
            # For more models, use random search
            np.random.seed(42)
            for _ in range(100):
                weights = np.random.random(n_models)
                weights = weights / weights.sum()
                
                ensemble_pred = np.average(predictions, axis=0, weights=weights)
                score = mean_squared_error(y_val, ensemble_pred)
                
                if score < best_score:
                    best_score = score
                    best_weights = weights
        
        if best_weights is not None:
            self.weights = best_weights
            logger.info(f"Optimized weights: {dict(zip(self.base_models.keys(), self.weights))}")
        else:
            logger.warning("Weight optimization failed, using equal weights")
    
    def _train_meta_model(self, base_predictions: List[np.ndarray], y_train: np.ndarray) -> None:
        """Train meta-model for stacking.
        
        Args:
            base_predictions: Predictions from base models
            y_train: Training targets
        """
        logger.info("Training meta-model for stacking")
        
        # Stack predictions as features
        X_meta = np.column_stack(base_predictions)
        
        # Train meta-model
        self.meta_model.fit(X_meta, y_train)
        logger.info("Meta-model training completed")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions.
        
        Args:
            X: Input features
            
        Returns:
            Ensemble predictions
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        # Get predictions from all base models
        predictions = []
        model_names = []
        
        for name, model in self.base_models.items():
            if model.is_trained:
                pred = model.predict(X)
                predictions.append(pred)
                model_names.append(name)
        
        if len(predictions) == 0:
            raise ValueError("No trained base models available for prediction")
        
        predictions = np.array(predictions)
        
        # Combine predictions based on ensemble method
        if self.ensemble_method == 'weighted_average':
            # Use optimized weights
            weights = self.weights[:len(predictions)]
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
            
        elif self.ensemble_method == 'voting':
            # Simple average
            ensemble_pred = np.mean(predictions, axis=0)
            
        elif self.ensemble_method == 'stacking' and self.meta_model is not None:
            # Use meta-model
            X_meta = predictions.T  # Transpose to get (samples, models)
            ensemble_pred = self.meta_model.predict(X_meta)
            
        else:
            # Fallback to simple average
            ensemble_pred = np.mean(predictions, axis=0)
        
        return ensemble_pred
    
    def predict_with_individual_models(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from individual models and ensemble.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary with predictions from each model and ensemble
        """
        results = {}
        
        # Get individual predictions
        for name, model in self.base_models.items():
            if model.is_trained:
                results[name] = model.predict(X)
        
        # Get ensemble prediction
        results['ensemble'] = self.predict(X)
        
        return results
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get ensemble model weights.
        
        Returns:
            Dictionary of model weights
        """
        if self.weights is None:
            return {}
        
        model_names = list(self.base_models.keys())
        return dict(zip(model_names, self.weights))
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get aggregated feature importance from base models.
        
        Returns:
            Dictionary of aggregated feature importances
        """
        all_importances = {}
        
        for name, model in self.base_models.items():
            if model.is_trained and hasattr(model, 'get_feature_importance'):
                try:
                    importance = model.get_feature_importance()
                    for feature, value in importance.items():
                        if feature not in all_importances:
                            all_importances[feature] = []
                        all_importances[feature].append(value)
                except:
                    continue
        
        # Average importances across models
        avg_importances = {}
        for feature, values in all_importances.items():
            avg_importances[feature] = np.mean(values)
        
        # Sort by importance
        avg_importances = dict(sorted(
            avg_importances.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        return avg_importances
    
    def get_prediction_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get prediction uncertainty based on model disagreement.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (mean_predictions, std_predictions)
        """
        predictions = []
        
        for name, model in self.base_models.items():
            if model.is_trained:
                pred = model.predict(X)
                predictions.append(pred)
        
        if len(predictions) == 0:
            raise ValueError("No trained models available")
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred
    
    def save_model(self, filepath: str) -> None:
        """Save ensemble model.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained ensemble")
        
        # Save base models individually
        base_model_paths = {}
        for name, model in self.base_models.items():
            if model.is_trained:
                model_path = filepath.replace('.pkl', f'_{name}.pkl')
                model.save_model(model_path)
                base_model_paths[name] = model_path
        
        # Save ensemble metadata
        ensemble_data = {
            'config': self.config,
            'weights': self.weights,
            'meta_model': self.meta_model,
            'ensemble_method': self.ensemble_method,
            'base_model_paths': base_model_paths,
            'is_trained': self.is_trained
        }
        
        import joblib
        joblib.dump(ensemble_data, filepath)
        logger.info(f"Ensemble model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load ensemble model.
        
        Args:
            filepath: Path to load model from
        """
        import joblib
        
        # Load ensemble metadata
        ensemble_data = joblib.load(filepath)
        
        self.config = ensemble_data['config']
        self.weights = ensemble_data['weights']
        self.meta_model = ensemble_data['meta_model']
        self.ensemble_method = ensemble_data['ensemble_method']
        self.is_trained = ensemble_data['is_trained']
        
        # Load base models
        base_model_paths = ensemble_data['base_model_paths']
        for name, model_path in base_model_paths.items():
            # Determine model type and create instance
            if 'lstm' in name.lower():
                model = LSTMModel({})
            elif 'random_forest' in name.lower() or 'rf' in name.lower():
                model = RandomForestModel({})
            elif 'xgboost' in name.lower() or 'xgb' in name.lower():
                model = XGBoostModel({})
            else:
                continue
            
            model.load_model(model_path)
            self.base_models[name] = model
        
        logger.info(f"Ensemble model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get ensemble model information.
        
        Returns:
            Dictionary with ensemble information
        """
        info = super().get_model_info()
        info.update({
            'ensemble_method': self.ensemble_method,
            'n_base_models': len(self.base_models),
            'base_models': list(self.base_models.keys()),
            'weights': self.get_model_weights(),
            'use_meta_model': self.meta_model is not None
        })
        
        return info
