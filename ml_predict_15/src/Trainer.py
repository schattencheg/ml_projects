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

# MLflow integration
try:
    from .MLflowTracker import MLflowTracker
    MLFLOW_TRACKER_AVAILABLE = True
except ImportError:
    MLFLOW_TRACKER_AVAILABLE = False


class Trainer:
    """
    Trains multiple ML models with automatic handling of imbalanced data.
    """
    
    def __init__(self, use_smote=True, optimize_threshold=True, use_scaler=True, 
                 use_mlflow=True, mlflow_experiment="ml_predict_15/crypto_prediction",
                 mlflow_tracking_uri="http://localhost:5000"):
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
        use_mlflow : bool
            Whether to use MLflow tracking
        mlflow_experiment : str
            MLflow experiment name
        mlflow_tracking_uri : str
            MLflow tracking server URI
        """
        self.use_smote = use_smote and SMOTE_AVAILABLE
        self.optimize_threshold = optimize_threshold
        self.use_scaler = use_scaler
        self.scaler = None
        self.results = {}
        self.best_model_name = None
        self.training_time = 0
        
        # MLflow setup
        self.use_mlflow = use_mlflow and MLFLOW_TRACKER_AVAILABLE
        self.mlflow_tracker = None
        if self.use_mlflow:
            self.mlflow_tracker = MLflowTracker(
                experiment_name=mlflow_experiment,
                tracking_uri=mlflow_tracking_uri,
                enable_tracking=True
            )
    
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
        
        # Start MLflow run
        mlflow_run_id = None
        if self.use_mlflow and self.mlflow_tracker and self.mlflow_tracker.is_available():
            mlflow_run_id = self.mlflow_tracker.start_run()
            
            # Log training configuration
            training_config = {
                'use_smote': self.use_smote,
                'optimize_threshold': self.optimize_threshold,
                'use_scaler': self.use_scaler,
                'num_models': len(models),
                'model_names': list(models.keys()),
                'training_samples': len(X_train),
                'validation_samples': len(X_val) if X_val is not None else 0,
                'feature_count': X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0])
            }
            self.mlflow_tracker.log_params(training_config)
        
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
        
        # Log results to MLflow
        if mlflow_run_id and self.mlflow_tracker:
            self._log_results_to_mlflow(trained_models, class_dist)
        
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
    
    def _log_results_to_mlflow(self, trained_models, class_dist):
        """Log training results to MLflow."""
        try:
            # Prepare metrics for logging
            all_metrics = {}
            
            # Log class distribution
            for cls, count in class_dist.items():
                all_metrics[f"class_{cls}_count"] = count
                all_metrics[f"class_{cls}_percentage"] = count / sum(class_dist.values()) * 100
            
            # Log model-specific metrics
            for model_name, results in self.results.items():
                train_metrics = results['train_metrics']
                val_metrics = results.get('val_metrics', {})
                
                # Training metrics
                for metric_name, value in train_metrics.items():
                    all_metrics[f"{model_name}_train_{metric_name}"] = value
                
                # Validation metrics
                for metric_name, value in val_metrics.items():
                    all_metrics[f"{model_name}_val_{metric_name}"] = value
                
                # Other metrics
                all_metrics[f"{model_name}_optimal_threshold"] = results['optimal_threshold']
            
            # Best model metrics
            if self.best_model_name in self.results:
                best_results = self.results[self.best_model_name]
                for metric_name, value in best_results['train_metrics'].items():
                    all_metrics[f"best_{metric_name}"] = value
            
            # Summary metrics
            all_metrics["total_training_time"] = self.training_time
            all_metrics["avg_training_time"] = self.training_time / len(trained_models)
            all_metrics["num_models_trained"] = len(trained_models)
            
            # Log all metrics
            self.mlflow_tracker.log_metrics(all_metrics)
            
            # Log best model
            if self.best_model_name in trained_models:
                best_model = trained_models[self.best_model_name]
                self.mlflow_tracker.log_model(best_model, self.best_model_name)
                
                # Log neural network specific info if applicable
                if hasattr(best_model, 'model') or 'neural' in self.best_model_name.lower():
                    self.mlflow_tracker.log_neural_network_info(best_model, self.best_model_name)
            
            # Log all neural network models
            for model_name, model in trained_models.items():
                if hasattr(model, 'model') or any(nn_type in model_name.lower() 
                                                 for nn_type in ['cnn', 'lstm', 'gru', 'neural']):
                    self.mlflow_tracker.log_neural_network_info(model, model_name)
            
            # Prepare artifacts
            results_df = pd.DataFrame({
                name: {
                    **results['train_metrics'],
                    **{f"val_{k}": v for k, v in results.get('val_metrics', {}).items()},
                    'optimal_threshold': results['optimal_threshold']
                }
                for name, results in self.results.items()
            }).T
            
            artifacts = {
                "training_results": results_df,
                "class_distribution": pd.DataFrame(list(class_dist.items()), 
                                                 columns=['class', 'count']),
                "training_summary": {
                    "best_model": self.best_model_name,
                    "total_time": self.training_time,
                    "num_models": len(trained_models),
                    "model_names": list(trained_models.keys())
                }
            }
            
            self.mlflow_tracker.log_artifacts(artifacts)
            
            # End MLflow run
            self.mlflow_tracker.end_run()
            
        except Exception as e:
            print(f"⚠ Failed to log results to MLflow: {e}")
            if self.mlflow_tracker:
                try:
                    self.mlflow_tracker.end_run()
                except:
                    pass
    
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
