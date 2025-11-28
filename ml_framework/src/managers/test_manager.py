"""
TestManager - Manages model testing and evaluation.
Dedicated manager for testing/evaluation operations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)

from ..models_lib import BaseModel
from .scaler_manager import ScalerManager


class TestManager:
    """
    Manages model testing and evaluation operations.
    
    Features:
    - Test multiple models on test data
    - Comprehensive performance metrics
    - Confusion matrix analysis
    - Classification reports
    - Model comparison and ranking
    
    Example:
        >>> test_mgr = TestManager(verbose=False)
        >>> results = test_mgr.test(models, test_data)
        >>> best = test_mgr.get_best_model(metric='accuracy')
        >>> df = test_mgr.get_comparison_dataframe()
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize TestManager.
        
        Args:
            verbose: If True, print status messages (default: True)
        """
        self.test_results = {}
        self.predictions = {}
        self.verbose = verbose
        
    def test(self,
            models: Dict[str, BaseModel],
            test_data: pd.DataFrame,
            target_col: str = 'target',
            feature_cols: Optional[List[str]] = None,
            scaler_manager: Optional[ScalerManager] = None,
            X_test_scaled: Optional[np.ndarray] = None,
            y_test: Optional[np.ndarray] = None) -> Dict[str, Dict]:
        """
        Test/evaluate models on test data.
        
        Args:
            models: Dictionary of models to test
            test_data: Test DataFrame (used if X_test_scaled not provided)
            target_col: Name of target column
            feature_cols: List of feature columns
            scaler_manager: ScalerManager instance for scaling
            X_test_scaled: Pre-scaled test features (optional)
            y_test: Test labels (optional, extracted from test_data if not provided)
            
        Returns:
            Dictionary with test results for each model
        """
        if self.verbose:
            print("\n" + "="*70)
            print("TESTING MODELS")
            print("="*70)
        
        if not models:
            raise ValueError("No models to test")
        
        # Prepare data
        if X_test_scaled is None:
            if feature_cols is None:
                feature_cols = [col for col in test_data.columns if col != target_col]
            
            X_test = test_data[feature_cols].values
            
            # Scale features if scaler available
            if scaler_manager is not None:
                X_test_scaled = scaler_manager.transform(test_data[feature_cols])
                if isinstance(X_test_scaled, pd.DataFrame):
                    X_test_scaled = X_test_scaled.values
            else:
                X_test_scaled = X_test
        
        if y_test is None:
            y_test = test_data[target_col].values
        
        n_features = X_test_scaled.shape[1] if len(X_test_scaled.shape) > 1 else 1
        
        if self.verbose:
            print(f"\nTest dataset:")
            print(f"  Samples: {len(X_test_scaled)}")
            print(f"  Features: {n_features}\n")
        
        # Evaluate each model
        for model_name, model in models.items():
            if self.verbose:
                print(f"{'='*70}")
                print(f"Model: {model_name.upper()}")
                print(f"{'='*70}")
            
            try:
                # Predictions
                y_pred = model.predict(X_test_scaled)
                
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
                
                self.predictions[model_name] = y_pred
                
                # Print metrics
                if self.verbose:
                    print(f"\nPerformance Metrics:")
                    print(f"  Accuracy:  {accuracy:.4f}")
                    print(f"  Precision: {precision:.4f}")
                    print(f"  Recall:    {recall:.4f}")
                    print(f"  F1 Score:  {f1:.4f}")
                    
                    print(f"\nConfusion Matrix:")
                    print(cm)
                    
                    # Classification report
                    print(f"\nClassification Report:")
                    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
                    target_names = [self._get_label_name(label) for label in sorted(unique_labels)]
                    report = classification_report(y_test, y_pred, target_names=target_names)
                    for line in report.split('\n'):
                        print(f"  {line}")
                
            except Exception as e:
                if self.verbose:
                    print(f"✗ Testing failed: {str(e)}\n")
                self.test_results[model_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Summary
        if self.verbose:
            self._print_test_summary()
        
        return self.test_results
    
    def _get_label_name(self, label: int) -> str:
        """Get human-readable label name."""
        label_names = {-1: 'Down', 0: 'Neutral', 1: 'Up'}
        return label_names.get(label, str(label))
    
    def _print_test_summary(self):
        """Print test results summary."""
        if not self.test_results:
            return
        
        successful_results = {name: res for name, res in self.test_results.items() 
                            if res.get('status') == 'success'}
        
        if not successful_results:
            print("\nNo successful test results to display")
            return
        
        print("\n" + "="*70)
        print("TEST RESULTS SUMMARY")
        print("="*70)
        
        # Create summary DataFrame
        summary_data = []
        for model_name, metrics in successful_results.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1_score']
            })
        
        df_summary = pd.DataFrame(summary_data)
        
        # Sort by accuracy
        df_summary = df_summary.sort_values('Accuracy', ascending=False)
        
        # Format for display
        display_df = df_summary.copy()
        for col in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
        
        print("\n" + display_df.to_string(index=False))
        
        # Best model
        best_model = df_summary.iloc[0]['Model']
        best_accuracy = df_summary.iloc[0]['Accuracy']
        
        print(f"\n🏆 Best model: {best_model}")
        print(f"   Accuracy: {best_accuracy:.4f}")
        print("="*70 + "\n")
    
    def get_test_results(self) -> Dict[str, Dict]:
        """Get test results."""
        return self.test_results
    
    def get_predictions(self) -> Dict[str, np.ndarray]:
        """Get predictions for all models."""
        return self.predictions
    
    def get_best_model(self, metric: str = 'accuracy') -> str:
        """
        Get the name of the best performing model.
        
        Args:
            metric: Metric to use for comparison ('accuracy', 'f1_score', etc.)
            
        Returns:
            Name of the best model
        """
        successful_results = {name: res for name, res in self.test_results.items() 
                            if res.get('status') == 'success'}
        
        if not successful_results:
            raise ValueError("No successful test results available")
        
        best_model = max(successful_results.items(), 
                        key=lambda x: x[1].get(metric, 0))[0]
        return best_model
    
    def get_comparison_dataframe(self) -> pd.DataFrame:
        """
        Get a DataFrame comparing all model results.
        
        Returns:
            DataFrame with model comparison
        """
        successful_results = {name: res for name, res in self.test_results.items() 
                            if res.get('status') == 'success'}
        
        if not successful_results:
            return pd.DataFrame()
        
        comparison_data = {}
        for name, res in successful_results.items():
            comparison_data[name] = {
                'accuracy': res['accuracy'],
                'f1_score': res['f1_score'],
                'precision': res['precision'],
                'recall': res['recall']
            }
        
        df = pd.DataFrame(comparison_data).T
        df = df.sort_values('accuracy', ascending=False)
        return df
