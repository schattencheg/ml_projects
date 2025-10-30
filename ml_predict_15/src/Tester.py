"""
Tester Module

Handles testing of trained ML models on test data.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


class Tester:
    """
    Tests trained ML models on test data.
    """
    
    def __init__(self, scaler=None):
        """
        Initialize Tester.
        
        Parameters:
        -----------
        scaler : sklearn scaler
            Fitted scaler from training
        """
        self.scaler = scaler
        self.results = {}
        self.best_model_name = None
    
    def test(self, models, X_test, y_test, optimal_thresholds=None):
        """
        Test multiple models on test data.
        
        Parameters:
        -----------
        models : dict
            Dictionary of model_name -> trained_model
        X_test : pd.DataFrame or np.ndarray
            Test features
        y_test : pd.Series or np.ndarray
            Test labels
        optimal_thresholds : dict (optional)
            Dictionary of model_name -> optimal_threshold
            
        Returns:
        --------
        dict : Test results for each model
        """
        print(f"\n{'='*70}")
        print(f"TESTING MODELS")
        print(f"{'='*70}\n")
        
        # Scale features if scaler is available
        if self.scaler is not None:
            X_test_scaled = self.scaler.transform(X_test)
            print(f"✓ Features scaled using provided scaler")
        else:
            X_test_scaled = X_test
        
        # Test each model
        for model_name, model in models.items():
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Apply optimal threshold if available
            if optimal_thresholds and model_name in optimal_thresholds:
                threshold = optimal_thresholds[model_name]
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test_scaled)[:, 1]
                    y_pred = (y_proba >= threshold).astype(int)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred)
            
            # Store results
            self.results[model_name] = {
                'predictions': y_pred,
                'metrics': metrics
            }
            
            print(f"✓ {model_name}: Acc={metrics['accuracy']:.4f}, "
                  f"F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, "
                  f"Recall={metrics['recall']:.4f}")
        
        # Find best model
        self.best_model_name = max(
            self.results.keys(),
            key=lambda k: self.results[k]['metrics']['f1']
        )
        
        print(f"\n{'='*70}")
        print(f"TESTING COMPLETE")
        print(f"{'='*70}")
        print(f"Best model on test set: {self.best_model_name}")
        print(f"{'='*70}\n")
        
        return self.results
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def print_results(self):
        """Print test results summary."""
        if not self.results:
            print("No results to display. Test models first.")
            return
        
        print(f"\n{'='*70}")
        print(f"TEST RESULTS SUMMARY")
        print(f"{'='*70}\n")
        
        results_data = []
        for model_name, data in self.results.items():
            metrics = data['metrics']
            
            row = {
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1']
            }
            
            results_data.append(row)
        
        df_results = pd.DataFrame(results_data)
        df_results = df_results.sort_values('F1 Score', ascending=False)
        
        print(df_results.to_string(index=False))
        print(f"\n{'='*70}\n")
    
    def print_detailed_report(self, model_name, y_test, target_names=None):
        """
        Print detailed classification report for a specific model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        y_test : pd.Series or np.ndarray
            True labels
        target_names : list (optional)
            Names of target classes
        """
        if model_name not in self.results:
            print(f"Model {model_name} not found in results.")
            return
        
        y_pred = self.results[model_name]['predictions']
        
        print(f"\n{'='*70}")
        print(f"DETAILED REPORT: {model_name}")
        print(f"{'='*70}\n")
        
        # Classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        print(f"\n{'='*70}\n")
    
    def get_best_model_name(self):
        """
        Get the name of the best performing model.
        
        Returns:
        --------
        str : Name of best model
        """
        return self.best_model_name
    
    def get_predictions(self, model_name):
        """
        Get predictions for a specific model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
            
        Returns:
        --------
        np.ndarray : Predictions
        """
        if model_name not in self.results:
            return None
        
        return self.results[model_name]['predictions']
    
    def compare_models(self, metric='f1'):
        """
        Compare models by a specific metric.
        
        Parameters:
        -----------
        metric : str
            Metric to compare ('accuracy', 'precision', 'recall', 'f1')
            
        Returns:
        --------
        pd.DataFrame : Sorted comparison of models
        """
        if not self.results:
            print("No results to compare. Test models first.")
            return None
        
        comparison_data = []
        for model_name, data in self.results.items():
            metrics = data['metrics']
            comparison_data.append({
                'Model': model_name,
                'Metric Value': metrics[metric]
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('Metric Value', ascending=False)
        
        return df_comparison
