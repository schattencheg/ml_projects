"""
ML_Trainer - Trains machine learning models on prepared data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import time


class ML_Trainer:
    """
    Trains machine learning models on prepared data.
    
    Features:
    - Multiple model training
    - Progress tracking
    - Performance metrics
    - Model comparison
    """
    
    def __init__(self):
        """Initialize ML_Trainer."""
        self.trained_models = {}
        self.scaler = None
        self.results = {}
        
    def train(self, df: pd.DataFrame,
             target_col: str = 'target',
             feature_cols: Optional[list] = None,
             model_configs: Optional[Dict] = None,
             test_size: float = 0.2,
             scale_features: bool = True) -> Dict[str, Any]:
        """
        Train ML models on data.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            feature_cols: List of feature columns (uses all except target if None)
            model_configs: Dictionary of model configurations
            test_size: Proportion of data for testing
            scale_features: Whether to scale features
            
        Returns:
            Dictionary with trained models, scaler, and results
        """
        print("\n" + "="*70)
        print("TRAINING ML MODELS")
        print("="*70)
        
        # Prepare data
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_col]
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        print(f"\nDataset split:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Test:  {len(X_test)} samples")
        print(f"  Features: {len(feature_cols)}")
        
        # Scale features
        if scale_features:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            print(f"  ✓ Features scaled")
        
        # Get model configurations
        if model_configs is None:
            model_configs = self._get_default_models()
        
        print(f"\n{'='*70}")
        print(f"TRAINING {len(model_configs)} MODELS")
        print(f"{'='*70}\n")
        
        # Train each model
        total_time = 0
        for model_name, config in model_configs.items():
            if not config.get('enabled', True):
                continue
            
            print(f"Training {model_name}...", end=' ')
            start_time = time.time()
            
            # Create and train model
            model = self._create_model(model_name, config.get('params', {}))
            model.fit(X_train, y_train)
            
            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            training_time = time.time() - start_time
            
            # Store results
            self.trained_models[model_name] = model
            self.results[model_name] = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'training_time': training_time
            }
            
            total_time += training_time
            print(f"✓ Completed in {training_time:.2f}s")
            print(f"  Train accuracy: {train_score:.4f}")
            print(f"  Test accuracy:  {test_score:.4f}\n")
        
        # Summary
        print(f"{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Total training time: {total_time:.2f} seconds")
        print(f"Average time per model: {total_time/len(self.trained_models):.2f} seconds")
        
        # Find best model
        best_model_name = max(self.results.items(), 
                             key=lambda x: x[1]['test_accuracy'])[0]
        best_accuracy = self.results[best_model_name]['test_accuracy']
        
        print(f"\nBest model: {best_model_name}")
        print(f"Test accuracy: {best_accuracy:.4f}")
        print(f"{'='*70}\n")
        
        return {
            'models': self.trained_models,
            'scaler': self.scaler,
            'results': self.results,
            'best_model': best_model_name,
            'feature_cols': feature_cols
        }
    
    def _get_default_models(self) -> Dict[str, Dict]:
        """Get default model configurations."""
        return {
            'logistic_regression': {
                'enabled': True,
                'params': {'max_iter': 1000, 'random_state': 42}
            },
            'random_forest': {
                'enabled': True,
                'params': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
            }
        }
    
    def _create_model(self, model_name: str, params: Dict):
        """Create model instance based on name and parameters."""
        if model_name == 'logistic_regression':
            return LogisticRegression(**params)
        elif model_name == 'random_forest':
            return RandomForestClassifier(**params)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def get_trained_models(self) -> Dict[str, Any]:
        """Get dictionary of trained models."""
        return self.trained_models
    
    def get_results(self) -> Dict[str, Dict]:
        """Get training results."""
        return self.results
    
    def print_results(self):
        """Print training results summary."""
        if not self.results:
            print("No results available. Train models first.")
            return
        
        print("\n" + "="*70)
        print("TRAINING RESULTS SUMMARY")
        print("="*70)
        
        # Create DataFrame for better formatting
        results_data = []
        for model_name, metrics in self.results.items():
            results_data.append({
                'Model': model_name,
                'Train Acc': f"{metrics['train_accuracy']:.4f}",
                'Test Acc': f"{metrics['test_accuracy']:.4f}",
                'Time (s)': f"{metrics['training_time']:.2f}"
            })
        
        df_results = pd.DataFrame(results_data)
        print(df_results.to_string(index=False))
        print("="*70 + "\n")
