"""
Trainer Module

Handles training of multiple ML models with progress tracking, SMOTE, and threshold optimization.
"""

import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Warning: imbalanced-learn not installed. SMOTE will not be available.")


class Trainer:
    """
    Trains multiple ML models with automatic handling of imbalanced data.
    """
    
    def __init__(self, use_smote=True, optimize_threshold=True, use_scaler=True):
        """
        Initialize Trainer.
        
        Parameters:
        -----------
        use_smote : bool
            Whether to apply SMOTE for imbalanced data
        optimize_threshold : bool
            Whether to optimize probability threshold
        use_scaler : bool
            Whether to scale features
        """
        self.use_smote = use_smote and SMOTE_AVAILABLE
        self.optimize_threshold = optimize_threshold
        self.use_scaler = use_scaler
        self.scaler = None
        self.results = {}
        self.best_model_name = None
        self.training_time = 0
    
    def train(self, models, X_train, y_train, X_val=None, y_val=None):
        """
        Train multiple models.
        
        Parameters:
        -----------
        models : dict
            Dictionary of model_name -> model_instance
        X_train : pd.DataFrame or np.ndarray
            Training features
        y_train : pd.Series or np.ndarray
            Training labels
        X_val : pd.DataFrame or np.ndarray (optional)
            Validation features
        y_val : pd.Series or np.ndarray (optional)
            Validation labels
            
        Returns:
        --------
        tuple : (trained_models, scaler, results, best_model_name)
        """
        print(f"\n{'='*70}")
        print(f"TRAINING MODELS")
        print(f"{'='*70}\n")
        
        # Scale features
        if self.use_scaler:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            if X_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
            print(f"✓ Features scaled using StandardScaler")
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val if X_val is not None else None
        
        # Check class imbalance
        unique, counts = np.unique(y_train, return_counts=True)
        class_dist = dict(zip(unique, counts))
        print(f"\nClass distribution (training):")
        for cls, count in class_dist.items():
            print(f"  Class {cls}: {count:,} ({count/len(y_train)*100:.1f}%)")
        
        # Apply SMOTE if needed
        if self.use_smote and len(unique) > 1:
            imbalance_ratio = max(counts) / min(counts)
            if imbalance_ratio > 1.5:
                print(f"\n✓ Applying SMOTE (imbalance ratio: {imbalance_ratio:.2f})")
                smote = SMOTE(random_state=42)
                X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
                print(f"  Resampled size: {len(X_train_scaled):,}")
        
        # Train each model
        trained_models = {}
        start_time = time.time()
        
        for model_name in tqdm(models.keys(), desc="Training models"):
            model = models[model_name]
            
            model_start = time.time()
            model.fit(X_train_scaled, y_train)
            model_time = time.time() - model_start
            
            # Evaluate on training set
            y_train_pred = model.predict(X_train_scaled)
            train_metrics = self._calculate_metrics(y_train, y_train_pred, model_time)
            
            # Evaluate on validation set if provided
            val_metrics = {}
            if X_val is not None and y_val is not None:
                y_val_pred = model.predict(X_val_scaled)
                val_metrics = self._calculate_metrics(y_val, y_val_pred)
            
            # Optimize threshold if requested
            optimal_threshold = 0.5
            if self.optimize_threshold and hasattr(model, 'predict_proba'):
                if X_val is not None and y_val is not None:
                    optimal_threshold = self._find_optimal_threshold(
                        model, X_val_scaled, y_val
                    )
            
            # Store results
            self.results[model_name] = {
                'model': model,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'optimal_threshold': optimal_threshold,
                'training_time': model_time
            }
            
            trained_models[model_name] = model
            
            print(f"✓ {model_name}: Train Acc={train_metrics['accuracy']:.4f}, "
                  f"F1={train_metrics['f1']:.4f}, Time={model_time:.2f}s")
        
        self.training_time = time.time() - start_time
        
        # Find best model
        self.best_model_name = max(
            self.results.keys(),
            key=lambda k: self.results[k]['train_metrics']['f1']
        )
        
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Total training time: {self.training_time:.2f} seconds ({self.training_time/60:.2f} minutes)")
        print(f"Average time per model: {self.training_time/len(models):.2f} seconds")
        print(f"Best model: {self.best_model_name}")
        print(f"{'='*70}\n")
        
        return trained_models, self.scaler, self.results, self.best_model_name
    
    def _calculate_metrics(self, y_true, y_pred, training_time=None):
        """Calculate classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        if training_time is not None:
            metrics['training_time'] = training_time
        
        return metrics
    
    def _find_optimal_threshold(self, model, X_val, y_val):
        """Find optimal probability threshold for binary classification."""
        if not hasattr(model, 'predict_proba'):
            return 0.5
        
        y_proba = model.predict_proba(X_val)[:, 1]
        
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in np.arange(0.1, 0.9, 0.05):
            y_pred = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold
    
    def print_results(self):
        """Print training results summary."""
        if not self.results:
            print("No results to display. Train models first.")
            return
        
        print(f"\n{'='*70}")
        print(f"TRAINING RESULTS SUMMARY")
        print(f"{'='*70}\n")
        
        results_data = []
        for model_name, data in self.results.items():
            train_metrics = data['train_metrics']
            val_metrics = data['val_metrics']
            
            row = {
                'Model': model_name,
                'Train Acc': train_metrics['accuracy'],
                'Train F1': train_metrics['f1'],
                'Train Time (s)': train_metrics.get('training_time', 0)
            }
            
            if val_metrics:
                row['Val Acc'] = val_metrics['accuracy']
                row['Val F1'] = val_metrics['f1']
            
            results_data.append(row)
        
        df_results = pd.DataFrame(results_data)
        df_results = df_results.sort_values('Train F1', ascending=False)
        
        print(df_results.to_string(index=False))
        print(f"\n{'='*70}\n")
    
    def get_best_model(self):
        """
        Get the best performing model.
        
        Returns:
        --------
        tuple : (model_name, model_instance)
        """
        if self.best_model_name is None:
            return None, None
        
        return self.best_model_name, self.results[self.best_model_name]['model']
