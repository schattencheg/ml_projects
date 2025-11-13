"""
ML_Tester - Evaluates trained models and generates performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)


class ML_Tester:
    """
    Evaluates trained ML models and generates performance metrics.
    
    Features:
    - Model evaluation on test data
    - Multiple performance metrics
    - Confusion matrix
    - Classification reports
    - Model comparison
    """
    
    def __init__(self):
        """Initialize ML_Tester."""
        self.test_results = {}
        
    def evaluate(self, df: pd.DataFrame,
                models: Dict[str, Any],
                scaler: Optional[Any] = None,
                target_col: str = 'target',
                feature_cols: Optional[list] = None) -> Dict[str, Dict]:
        """
        Evaluate models on test data.
        
        Args:
            df: Test DataFrame with features and target
            models: Dictionary of trained models
            scaler: Fitted scaler (optional)
            target_col: Name of target column
            feature_cols: List of feature columns
            
        Returns:
            Dictionary with test results for each model
        """
        print("\n" + "="*70)
        print("EVALUATING MODELS ON TEST DATA")
        print("="*70)
        
        # Prepare data
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_col]
        
        X_test = df[feature_cols].values
        y_test = df[target_col].values
        
        # Scale features if scaler provided
        if scaler is not None:
            X_test = scaler.transform(X_test)
        
        print(f"\nTest dataset:")
        print(f"  Samples: {len(X_test)}")
        print(f"  Features: {len(feature_cols)}\n")
        
        # Evaluate each model
        for model_name, model in models.items():
            print(f"{'='*70}")
            print(f"Model: {model_name.upper()}")
            print(f"{'='*70}")
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Store results
            self.test_results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'predictions': y_pred
            }
            
            # Print metrics
            print(f"\nPerformance Metrics:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1 Score:  {f1:.4f}")
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            print(f"\nConfusion Matrix:")
            print(cm)
            print()
        
        # Summary
        self._print_summary()
        
        return self.test_results
    
    def _print_summary(self):
        """Print evaluation summary."""
        if not self.test_results:
            return
        
        print("="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        
        # Create summary DataFrame
        summary_data = []
        for model_name, metrics in self.test_results.items():
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
        best_model = max(self.test_results.items(), 
                        key=lambda x: x[1]['f1_score'])[0]
        best_f1 = self.test_results[best_model]['f1_score']
        
        print(f"\nBest model: {best_model}")
        print(f"F1 Score: {best_f1:.4f}")
        print("="*70 + "\n")
    
    def get_results(self) -> Dict[str, Dict]:
        """Get test results."""
        return self.test_results
    
    def get_predictions(self, model_name: str) -> np.ndarray:
        """
        Get predictions for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Array of predictions
        """
        if model_name not in self.test_results:
            raise ValueError(f"No results for model: {model_name}")
        
        return self.test_results[model_name]['predictions']
    
    def compare_models(self, metric: str = 'f1_score') -> pd.DataFrame:
        """
        Compare models by a specific metric.
        
        Args:
            metric: Metric to compare ('accuracy', 'precision', 'recall', 'f1_score')
            
        Returns:
            DataFrame with model comparison
        """
        if not self.test_results:
            raise ValueError("No test results available")
        
        comparison_data = []
        for model_name, metrics in self.test_results.items():
            comparison_data.append({
                'Model': model_name,
                'Score': metrics[metric]
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('Score', ascending=False)
        
        return df_comparison
    
    def print_classification_report(self, model_name: str, 
                                   y_true: np.ndarray):
        """
        Print detailed classification report for a model.
        
        Args:
            model_name: Name of the model
            y_true: True labels
        """
        if model_name not in self.test_results:
            raise ValueError(f"No results for model: {model_name}")
        
        y_pred = self.test_results[model_name]['predictions']
        
        print(f"\n{'='*70}")
        print(f"CLASSIFICATION REPORT: {model_name.upper()}")
        print(f"{'='*70}\n")
        
        print(classification_report(y_true, y_pred))
        print("="*70 + "\n")
