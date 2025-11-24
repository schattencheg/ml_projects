"""
HyperparameterOptimizer - Find optimal parameters for ML models.
Supports grid search, random search, and Bayesian optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Literal, Callable
from pathlib import Path
import time
import json
from sklearn.model_selection import cross_val_score
import warnings

try:
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False


class HyperparameterOptimizer:
    """
    Optimizes hyperparameters for ML models.
    
    Features:
    - Multiple optimization methods (grid, random, bayesian)
    - Cross-validation support
    - Parallel processing
    - Results tracking and visualization
    - Save/load optimization results
    """
    
    def __init__(self, 
                 method: Literal['grid', 'random', 'bayesian'] = 'random',
                 cv: int = 5,
                 n_jobs: int = -1,
                 verbose: int = 1):
        """
        Initialize HyperparameterOptimizer.
        
        Args:
            method: Optimization method
                - 'grid': Grid search (exhaustive)
                - 'random': Random search (faster)
                - 'bayesian': Bayesian optimization (smart)
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs (-1 = all cores)
            verbose: Verbosity level
        """
        self.method = method
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.best_params = {}
        self.best_scores = {}
        self.optimization_results = {}
        
    def optimize(self,
                model_name: str,
                model_class: Any,
                param_space: Dict[str, Any],
                X_train: np.ndarray,
                y_train: np.ndarray,
                scoring: str = 'accuracy',
                n_iter: int = 50,
                timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a model.
        
        Args:
            model_name: Name of the model
            model_class: Model class or instance
            param_space: Parameter search space
            X_train: Training features
            y_train: Training targets
            scoring: Scoring metric
            n_iter: Number of iterations (for random/bayesian)
            timeout: Timeout in seconds (None = no timeout)
            
        Returns:
            Dictionary with optimization results
        """
        print("\n" + "="*70)
        print(f"OPTIMIZING HYPERPARAMETERS - {model_name.upper()}")
        print("="*70)
        print(f"Method: {self.method}")
        print(f"CV folds: {self.cv}")
        print(f"Scoring: {scoring}")
        
        start_time = time.time()
        
        try:
            if self.method == 'grid':
                results = self._optimize_grid(model_class, param_space, X_train, y_train, scoring)
            elif self.method == 'random':
                results = self._optimize_random(model_class, param_space, X_train, y_train, scoring, n_iter)
            elif self.method == 'bayesian':
                results = self._optimize_bayesian(model_class, param_space, X_train, y_train, scoring, n_iter)
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            optimization_time = time.time() - start_time
            
            # Store results
            self.best_params[model_name] = results['best_params']
            self.best_scores[model_name] = results['best_score']
            self.optimization_results[model_name] = {
                **results,
                'optimization_time': optimization_time,
                'method': self.method
            }
            
            print(f"\n✓ Optimization completed in {optimization_time:.2f}s")
            print(f"Best score: {results['best_score']:.4f}")
            print(f"Best parameters:")
            for param, value in results['best_params'].items():
                print(f"  {param}: {value}")
            print("="*70 + "\n")
            
            return results
            
        except Exception as e:
            print(f"\n✗ Optimization failed: {e}")
            print("="*70 + "\n")
            raise
    
    def _optimize_grid(self,
                      model_class: Any,
                      param_space: Dict[str, Any],
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      scoring: str) -> Dict[str, Any]:
        """Perform grid search optimization."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for grid search")
        
        print(f"\nPerforming grid search...")
        print(f"Parameter space size: {np.prod([len(v) for v in param_space.values()])}")
        
        # Create base model
        base_model = model_class() if callable(model_class) else model_class
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_space,
            cv=self.cv,
            scoring=scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            return_train_score=True
        )
        
        grid_search.fit(X_train, y_train)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': pd.DataFrame(grid_search.cv_results_),
            'best_estimator': grid_search.best_estimator_
        }
    
    def _optimize_random(self,
                        model_class: Any,
                        param_space: Dict[str, Any],
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        scoring: str,
                        n_iter: int) -> Dict[str, Any]:
        """Perform random search optimization."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for random search")
        
        print(f"\nPerforming random search...")
        print(f"Number of iterations: {n_iter}")
        
        # Create base model
        base_model = model_class() if callable(model_class) else model_class
        
        # Perform random search
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_space,
            n_iter=n_iter,
            cv=self.cv,
            scoring=scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=42,
            return_train_score=True
        )
        
        random_search.fit(X_train, y_train)
        
        return {
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'cv_results': pd.DataFrame(random_search.cv_results_),
            'best_estimator': random_search.best_estimator_
        }
    
    def _optimize_bayesian(self,
                          model_class: Any,
                          param_space: Dict[str, Any],
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          scoring: str,
                          n_iter: int) -> Dict[str, Any]:
        """Perform Bayesian optimization."""
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize is required for Bayesian optimization. "
                            "Install with: pip install scikit-optimize")
        
        print(f"\nPerforming Bayesian optimization...")
        print(f"Number of iterations: {n_iter}")
        
        # Create base model
        base_model = model_class() if callable(model_class) else model_class
        
        # Perform Bayesian search
        bayes_search = BayesSearchCV(
            estimator=base_model,
            search_spaces=param_space,
            n_iter=n_iter,
            cv=self.cv,
            scoring=scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=42,
            return_train_score=True
        )
        
        bayes_search.fit(X_train, y_train)
        
        return {
            'best_params': bayes_search.best_params_,
            'best_score': bayes_search.best_score_,
            'cv_results': pd.DataFrame(bayes_search.cv_results_),
            'best_estimator': bayes_search.best_estimator_
        }
    
    def optimize_multiple(self,
                         models_config: Dict[str, Dict[str, Any]],
                         X_train: np.ndarray,
                         y_train: np.ndarray,
                         scoring: str = 'accuracy',
                         n_iter: int = 50) -> Dict[str, Dict[str, Any]]:
        """
        Optimize hyperparameters for multiple models.
        
        Args:
            models_config: Dictionary with model configs
                {
                    'model_name': {
                        'model_class': ModelClass,
                        'param_space': {...}
                    }
                }
            X_train: Training features
            y_train: Training targets
            scoring: Scoring metric
            n_iter: Number of iterations per model
            
        Returns:
            Dictionary with results for each model
        """
        print("\n" + "="*70)
        print(f"OPTIMIZING {len(models_config)} MODELS")
        print("="*70)
        
        all_results = {}
        
        for model_name, config in models_config.items():
            try:
                results = self.optimize(
                    model_name=model_name,
                    model_class=config['model_class'],
                    param_space=config['param_space'],
                    X_train=X_train,
                    y_train=y_train,
                    scoring=scoring,
                    n_iter=n_iter
                )
                all_results[model_name] = results
            except Exception as e:
                print(f"✗ Failed to optimize {model_name}: {e}")
                all_results[model_name] = {'error': str(e)}
        
        self._print_comparison()
        
        return all_results
    
    def get_best_params(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get best parameters for a model.
        
        Args:
            model_name: Model name (None = all models)
            
        Returns:
            Best parameters dictionary
        """
        if model_name is None:
            return self.best_params
        
        if model_name not in self.best_params:
            raise ValueError(f"No optimization results for model: {model_name}")
        
        return self.best_params[model_name]
    
    def get_best_score(self, model_name: str) -> float:
        """Get best score for a model."""
        if model_name not in self.best_scores:
            raise ValueError(f"No optimization results for model: {model_name}")
        
        return self.best_scores[model_name]
    
    def _print_comparison(self):
        """Print comparison of optimized models."""
        if not self.best_scores:
            return
        
        print("\n" + "="*70)
        print("OPTIMIZATION COMPARISON")
        print("="*70)
        
        # Sort by score
        sorted_models = sorted(self.best_scores.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n{'Model':<30s} {'Best Score':<15s} {'Time (s)':<10s}")
        print("-" * 70)
        
        for model_name, score in sorted_models:
            opt_time = self.optimization_results[model_name].get('optimization_time', 0)
            print(f"{model_name:<30s} {score:<15.4f} {opt_time:<10.2f}")
        
        print("="*70 + "\n")
    
    def save_results(self, save_dir: Path):
        """
        Save optimization results.
        
        Args:
            save_dir: Directory to save results
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save best parameters
        params_path = save_dir / "best_params.json"
        with open(params_path, 'w') as f:
            json.dump(self.best_params, f, indent=2, default=str)
        print(f"✓ Saved best parameters to {params_path}")
        
        # Save best scores
        scores_path = save_dir / "best_scores.json"
        with open(scores_path, 'w') as f:
            json.dump(self.best_scores, f, indent=2)
        print(f"✓ Saved best scores to {scores_path}")
        
        # Save detailed results (without estimators)
        for model_name, results in self.optimization_results.items():
            model_dir = save_dir / model_name
            model_dir.mkdir(exist_ok=True)
            
            # Save CV results if available
            if 'cv_results' in results:
                cv_path = model_dir / "cv_results.csv"
                results['cv_results'].to_csv(cv_path, index=False)
            
            # Save summary
            summary = {
                'best_params': results['best_params'],
                'best_score': results['best_score'],
                'optimization_time': results.get('optimization_time'),
                'method': results.get('method')
            }
            summary_path = model_dir / "summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
        
        print(f"✓ Saved detailed results to {save_dir}")
    
    def plot_optimization_history(self, 
                                  model_name: str, 
                                  save_path: Optional[Path] = None):
        """
        Plot optimization history.
        
        Args:
            model_name: Model name
            save_path: Path to save plot (None = display)
        """
        if model_name not in self.optimization_results:
            raise ValueError(f"No results for model: {model_name}")
        
        results = self.optimization_results[model_name]
        
        if 'cv_results' not in results:
            print("No CV results available for plotting")
            return
        
        import matplotlib.pyplot as plt
        
        cv_results = results['cv_results']
        
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Score progression
        plt.subplot(1, 2, 1)
        plt.plot(cv_results['mean_test_score'], marker='o', label='Test Score')
        plt.plot(cv_results['mean_train_score'], marker='s', alpha=0.5, label='Train Score')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.title(f'{model_name} - Optimization Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Best score so far
        plt.subplot(1, 2, 2)
        best_so_far = cv_results['mean_test_score'].cummax()
        plt.plot(best_so_far, marker='o', color='green')
        plt.xlabel('Iteration')
        plt.ylabel('Best Score So Far')
        plt.title(f'{model_name} - Best Score Progress')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"✓ Saved optimization plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def get_default_param_spaces() -> Dict[str, Dict[str, Any]]:
        """
        Get default parameter spaces for common models.
        
        Returns:
            Dictionary with parameter spaces
        """
        param_spaces = {
            'xgboost': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'min_child_weight': [1, 3, 5]
            },
            'catboost': {
                'iterations': [50, 100, 200, 300],
                'depth': [4, 6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'l2_leaf_reg': [1, 3, 5, 7]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'logistic_regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        return param_spaces
    
    def __repr__(self) -> str:
        n_optimized = len(self.best_params)
        return f"HyperparameterOptimizer(method='{self.method}', optimized={n_optimized})"
