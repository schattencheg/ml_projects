"""
TrainManager - Manages model training and testing.
Unified manager for both training and testing operations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)

from ..models_lib import BaseModel
from .scaler_manager import ScalerManager


class TrainManager:
    """
    Manages model training and testing operations.
    
    Features:
    - Train multiple models
    - Test/evaluate models
    - Progress tracking
    - Performance metrics
    - Comprehensive results
    """
    
    def __init__(self, use_scaler: bool = True, scaler_type: str = 'standard'):
        """
        Initialize TrainManager.
        
        Args:
            use_scaler: Whether to use feature scaling
            scaler_type: Type of scaler to use
        """
        self.use_scaler = use_scaler
        self.scaler_type = scaler_type
        self.scaler_manager = None
        
        self.trained_models = {}
        self.train_results = {}
        self.test_results = {}
        
    def train(self,
             models: Dict[str, BaseModel],
             train_data: pd.DataFrame,
             target_col: str = 'target',
             feature_cols: Optional[List[str]] = None,
             val_data: Optional[pd.DataFrame] = None,
             scale_features: bool = True,
             **fit_kwargs) -> Dict[str, Any]:
        """
        Train multiple models.
        
        Args:
            models: Dictionary of model instances to train
            train_data: Training DataFrame
            target_col: Name of target column
            feature_cols: List of feature columns (None = all except target)
            val_data: Validation DataFrame (optional)
            scale_features: Whether to scale features
            **fit_kwargs: Additional arguments for model.fit()
            
        Returns:
            Dictionary with training results
        """
        print("\n" + "="*70)
        print("TRAINING MODELS")
        print("="*70)
        
        # Prepare data
        if feature_cols is None:
            feature_cols = [col for col in train_data.columns if col != target_col]
        
        X_train = train_data[feature_cols].values
        y_train = train_data[target_col].values
        
        print(f"\nTraining dataset:")
        print(f"  Samples: {len(X_train)}")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Target classes: {np.unique(y_train)}")
        
        # Scale features
        if scale_features and self.use_scaler:
            self.scaler_manager = ScalerManager(scaler_type=self.scaler_type)
            X_train = self.scaler_manager.fit_transform(train_data[feature_cols]).values
            print(f"  ✓ Features scaled using {self.scaler_type} scaler")
        
        # Prepare validation data if provided
        X_val, y_val = None, None
        if val_data is not None:
            X_val = val_data[feature_cols].values
            y_val = val_data[target_col].values
            if self.scaler_manager is not None:
                X_val = self.scaler_manager.transform(val_data[feature_cols]).values
            print(f"  Validation samples: {len(X_val)}")
        
        print(f"\n{'='*70}")
        print(f"TRAINING {len(models)} MODELS")
        print(f"{'='*70}\n")
        
        # Train each model
        total_time = 0
        for model_name, model in models.items():
            print(f"Training {model_name}...", end=' ')
            start_time = time.time()
            
            try:
                # Train model
                model.fit(X_train, y_train, **fit_kwargs)
                
                # Evaluate on training data
                y_train_pred = model.predict(X_train)
                train_accuracy = accuracy_score(y_train, y_train_pred)
                
                # Evaluate on validation data if available
                val_accuracy = None
                if X_val is not None:
                    y_val_pred = model.predict(X_val)
                    val_accuracy = accuracy_score(y_val, y_val_pred)
                
                training_time = time.time() - start_time
                
                # Store results
                self.trained_models[model_name] = model
                self.train_results[model_name] = {
                    'train_accuracy': train_accuracy,
                    'val_accuracy': val_accuracy,
                    'training_time': training_time,
                    'status': 'success'
                }
                
                total_time += training_time
                print(f"✓ Completed in {training_time:.2f}s")
                print(f"  Train accuracy: {train_accuracy:.4f}")
                if val_accuracy is not None:
                    print(f"  Val accuracy:   {val_accuracy:.4f}")
                print()
                
            except Exception as e:
                print(f"✗ Failed: {str(e)}\n")
                self.train_results[model_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Summary
        successful_models = [name for name, res in self.train_results.items() 
                           if res.get('status') == 'success']
        
        print(f"{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Successful models: {len(successful_models)}/{len(models)}")
        print(f"Total training time: {total_time:.2f} seconds")
        
        if successful_models:
            avg_time = total_time / len(successful_models)
            print(f"Average time per model: {avg_time:.2f} seconds")
            
            # Find best model
            best_metric = 'val_accuracy' if val_data is not None else 'train_accuracy'
            valid_results = {name: res for name, res in self.train_results.items() 
                           if res.get('status') == 'success' and res.get(best_metric) is not None}
            
            if valid_results:
                best_model_name = max(valid_results.items(), 
                                    key=lambda x: x[1][best_metric])[0]
                best_accuracy = valid_results[best_model_name][best_metric]
                
                print(f"\nBest model: {best_model_name}")
                print(f"{best_metric.replace('_', ' ').title()}: {best_accuracy:.4f}")
        
        print(f"{'='*70}\n")
        
        return {
            'models': self.trained_models,
            'scaler_manager': self.scaler_manager,
            'results': self.train_results,
            'feature_cols': feature_cols
        }
    
    def test(self,
            test_data: pd.DataFrame,
            models: Optional[Dict[str, BaseModel]] = None,
            target_col: str = 'target',
            feature_cols: Optional[List[str]] = None,
            scaler_manager: Optional[ScalerManager] = None) -> Dict[str, Dict]:
        """
        Test/evaluate models on test data.
        
        Args:
            test_data: Test DataFrame
            models: Dictionary of models to test (None = use trained models)
            target_col: Name of target column
            feature_cols: List of feature columns
            scaler_manager: ScalerManager instance (None = use internal)
            
        Returns:
            Dictionary with test results for each model
        """
        print("\n" + "="*70)
        print("TESTING MODELS")
        print("="*70)
        
        # Use provided models or trained models
        if models is None:
            models = self.trained_models
        
        if not models:
            raise ValueError("No models to test")
        
        # Use provided scaler or internal scaler
        if scaler_manager is None:
            scaler_manager = self.scaler_manager
        
        # Prepare data
        if feature_cols is None:
            feature_cols = [col for col in test_data.columns if col != target_col]
        
        X_test = test_data[feature_cols].values
        y_test = test_data[target_col].values
        
        # Scale features if scaler available
        if scaler_manager is not None:
            X_test = scaler_manager.transform(test_data[feature_cols]).values
        
        print(f"\nTest dataset:")
        print(f"  Samples: {len(X_test)}")
        print(f"  Features: {len(feature_cols)}\n")
        
        # Evaluate each model
        for model_name, model in models.items():
            print(f"{'='*70}")
            print(f"Model: {model_name.upper()}")
            print(f"{'='*70}")
            
            try:
                # Predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                cm = confusion_matrix(y_test, y_pred)
                
                # Store results
                self.test_results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'confusion_matrix': cm,
                    'predictions': y_pred,
                    'status': 'success'
                }
                
                # Print metrics
                print(f"\nPerformance Metrics:")
                print(f"  Accuracy:  {accuracy:.4f}")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall:    {recall:.4f}")
                print(f"  F1 Score:  {f1:.4f}")
                
                print(f"\nConfusion Matrix:")
                print(cm)
                print()
                
            except Exception as e:
                print(f"✗ Testing failed: {str(e)}\n")
                self.test_results[model_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Summary
        self._print_test_summary()
        
        return self.test_results
    
    def _print_test_summary(self):
        """Print test results summary."""
        if not self.test_results:
            return
        
        successful_results = {name: res for name, res in self.test_results.items() 
                            if res.get('status') == 'success'}
        
        if not successful_results:
            print("No successful test results to display")
            return
        
        print("="*70)
        print("TEST RESULTS SUMMARY")
        print("="*70)
        
        # Create summary DataFrame
        summary_data = []
        for model_name, metrics in successful_results.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1 Score': f"{metrics['f1_score']:.4f}"
            })
        
        df_summary = pd.DataFrame(summary_data)
        
        # Sort by F1 score
        df_summary['F1_numeric'] = df_summary['F1 Score'].astype(float)
        df_summary = df_summary.sort_values('F1_numeric', ascending=False)
        df_summary = df_summary.drop('F1_numeric', axis=1)
        
        print("\n" + df_summary.to_string(index=False))
        
        # Best model
        best_model = max(successful_results.items(), 
                        key=lambda x: x[1]['f1_score'])[0]
        best_f1 = successful_results[best_model]['f1_score']
        
        print(f"\nBest model: {best_model}")
        print(f"F1 Score: {best_f1:.4f}")
        print("="*70 + "\n")
    
    def get_trained_models(self) -> Dict[str, BaseModel]:
        """Get dictionary of trained models."""
        return self.trained_models
    
    def get_train_results(self) -> Dict[str, Dict]:
        """Get training results."""
        return self.train_results
    
    def get_test_results(self) -> Dict[str, Dict]:
        """Get test results."""
        return self.test_results
    
    def get_scaler_manager(self) -> Optional[ScalerManager]:
        """Get scaler manager instance."""
        return self.scaler_manager
