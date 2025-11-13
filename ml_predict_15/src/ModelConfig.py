"""
Model Configuration Module

Centralized configuration for all ML models (traditional and neural networks).
Provides a single source of truth for model enable/disable settings and parameters.
"""

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb


class ModelConfig:
    """
    Centralized configuration for all ML models.
    
    This class provides a unified interface for managing both traditional ML models
    and neural network models, including their enable/disable states and parameters.
    """
    
    def __init__(self):
        """Initialize model configurations."""
        
        self.model_names = {
            'logistic_regression': True,
            'ridge_classifier': True,
            'naive_bayes': True,
            'decision_tree': True,
            'random_forest': False,
            'gradient_boosting': False,
            'knn': False,
            'svm': False,
            'xgboost': True,
            'lightgbm': False,
            'cnn_simple': True,
            'cnn_deep': False,
            'cnn_residual': False,
            'cnn_attention': False,
            'cnn_dilated': False,
            'lstm_simple': False,
            'lstm_bidirectional': False,
            'lstm_stacked': False,
            'lstm_attention': False,
            'lstm_cnn_hybrid': False,
            'gru_simple': False,
            'gru_bidirectional': False
        }

        # Traditional ML Models Configuration
        self.traditional_models = {
            'logistic_regression': {
                'enabled': self.model_names['logistic_regression'],
                'class': LogisticRegression,
                'params': {'max_iter': 1000, 'random_state': 42, 'class_weight': 'balanced'},
                'description': 'Logistic Regression with balanced class weights',
                'training_time': '~2-5 sec'
            },
            'ridge_classifier': {
                'enabled': self.model_names['ridge_classifier'],
                'class': RidgeClassifier,
                'params': {'random_state': 42, 'class_weight': 'balanced'},
                'description': 'Ridge Classifier with L2 regularization',
                'training_time': '~2-5 sec'
            },
            'naive_bayes': {
                'enabled': self.model_names['naive_bayes'],
                'class': GaussianNB,
                'params': {},
                'description': 'Gaussian Naive Bayes',
                'training_time': '~3-5 sec'
            },
            'decision_tree': {
                'enabled': self.model_names['decision_tree'],
                'class': DecisionTreeClassifier,
                'params': {'max_depth': 10, 'random_state': 42, 'class_weight': 'balanced'},
                'description': 'Decision Tree with max depth 10',
                'training_time': '~5-10 sec'
            },
            'random_forest': {
                'enabled': self.model_names['random_forest'],
                'class': RandomForestClassifier,
                'params': {
                    'n_estimators': 100, 
                    'max_depth': 10, 
                    'random_state': 42,
                    'n_jobs': -1, 
                    'class_weight': 'balanced'
                },
                'description': 'Random Forest with 100 trees',
                'training_time': '~15-30 sec'
            },
            'gradient_boosting': {
                'enabled': self.model_names['gradient_boosting'],
                'class': GradientBoostingClassifier,
                'params': {'n_estimators': 100, 'max_depth': 5, 'random_state': 42},
                'description': 'Gradient Boosting (slower than XGBoost)',
                'training_time': '~60+ sec'
            },
            'knn': {
                'enabled': self.model_names['knn'],
                'class': KNeighborsClassifier,
                'params': {'n_neighbors': 5, 'n_jobs': -1},
                'description': 'K-Nearest Neighbors (slow on large datasets)',
                'training_time': '~50-120 sec'
            },
            'svm': {
                'enabled': self.model_names['svm'],
                'class': SVC,
                'params': {
                    'kernel': 'rbf', 
                    'random_state': 42, 
                    'probability': True,
                    'class_weight': 'balanced'
                },
                'description': 'Support Vector Machine with RBF kernel',
                'training_time': '~100+ sec'
            },
            'xgboost': {
                'enabled': self.model_names['xgboost'],
                'class': xgb.XGBClassifier,
                'params': {
                    'n_estimators': 100, 
                    'max_depth': 5, 
                    'random_state': 42,
                    'tree_method': 'hist', 
                    'n_jobs': -1
                },
                'description': 'XGBoost with histogram-based tree method',
                'training_time': '~3-5 sec'
            },
            'lightgbm': {
                'enabled': self.model_names['lightgbm'],
                'class': lgb.LGBMClassifier,
                'params': {
                    'n_estimators': 100, 
                    'max_depth': 5, 
                    'random_state': 42,
                    'n_jobs': -1, 
                    'verbose': -1
                },
                'description': 'LightGBM with fast training',
                'training_time': '~3-5 sec'
            }
        }
        
        # Neural Network Models Configuration
        # Note: build_fn will be set by NeuralNetworksManager
        self.neural_network_models = {
            # 1D-CNN Variants
            'cnn_simple': {
                'enabled': self.model_names['cnn_simple'],
                'build_fn': None,  # Set by NeuralNetworksManager
                'description': 'Simple 1D-CNN with 2 conv layers',
                'training_time': 'Variable (depends on epochs)',
                'category': 'CNN'
            },
            'cnn_deep': {
                'enabled': self.model_names['cnn_deep'],
                'build_fn': None,
                'description': 'Deep 1D-CNN with 4 conv layers',
                'training_time': 'Variable (depends on epochs)',
                'category': 'CNN'
            },
            'cnn_residual': {
                'enabled': self.model_names['cnn_residual'],
                'build_fn': None,
                'description': '1D-CNN with residual connections',
                'training_time': 'Variable (depends on epochs)',
                'category': 'CNN'
            },
            'cnn_attention': {
                'enabled': self.model_names['cnn_attention'],
                'build_fn': None,
                'description': '1D-CNN with attention mechanism',
                'training_time': 'Variable (depends on epochs)',
                'category': 'CNN'
            },
            'cnn_dilated': {
                'enabled': self.model_names['cnn_dilated'],
                'build_fn': None,
                'description': '1D-CNN with dilated convolutions',
                'training_time': 'Variable (depends on epochs)',
                'category': 'CNN'
            },
            
            # LSTM Variants
            'lstm_simple': {
                'enabled': self.model_names['lstm_simple'],
                'build_fn': None,
                'description': 'Simple LSTM with 2 layers',
                'training_time': 'Variable (depends on epochs)',
                'category': 'LSTM'
            },
            'lstm_bidirectional': {
                'enabled': self.model_names['lstm_bidirectional'],
                'build_fn': None,
                'description': 'Bidirectional LSTM',
                'training_time': 'Variable (depends on epochs)',
                'category': 'LSTM'
            },
            'lstm_stacked': {
                'enabled': self.model_names['lstm_stacked'],
                'build_fn': None,
                'description': 'Stacked LSTM with 3 layers',
                'training_time': 'Variable (depends on epochs)',
                'category': 'LSTM'
            },
            'lstm_attention': {
                'enabled': self.model_names['lstm_attention'],
                'build_fn': None,
                'description': 'LSTM with attention mechanism',
                'training_time': 'Variable (depends on epochs)',
                'category': 'LSTM'
            },
            'lstm_cnn_hybrid': {
                'enabled': self.model_names['lstm_cnn_hybrid'],
                'build_fn': None,
                'description': 'Hybrid CNN-LSTM model',
                'training_time': 'Variable (depends on epochs)',
                'category': 'Hybrid'
            },
            
            # GRU Variants
            'gru_simple': {
                'enabled': self.model_names['gru_simple'],
                'build_fn': None,
                'description': 'Simple GRU model',
                'training_time': 'Variable (depends on epochs)',
                'category': 'GRU'
            },
            'gru_bidirectional': {
                'enabled': self.model_names['gru_bidirectional'],
                'build_fn': None,
                'description': 'Bidirectional GRU',
                'training_time': 'Variable (depends on epochs)',
                'category': 'GRU'
            }
        }
    
    def enable_model(self, model_name, enabled=True):
        """
        Enable or disable a model (traditional or neural network).
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        enabled : bool
            Whether to enable or disable
        
        Returns:
        --------
        bool : True if successful, False if model not found
        """
        if model_name in self.traditional_models:
            self.traditional_models[model_name]['enabled'] = enabled
            status = "enabled" if enabled else "disabled"
            print(f"✓ Traditional model '{model_name}' {status}")
            return True
        elif model_name in self.neural_network_models:
            self.neural_network_models[model_name]['enabled'] = enabled
            status = "enabled" if enabled else "disabled"
            print(f"✓ Neural network model '{model_name}' {status}")
            return True
        else:
            print(f"✗ Model '{model_name}' not found")
            return False
    
    def enable_all_traditional(self, enabled=True):
        """Enable or disable all traditional ML models."""
        for model_name in self.traditional_models:
            self.traditional_models[model_name]['enabled'] = enabled
        status = "enabled" if enabled else "disabled"
        print(f"✓ All traditional models {status}")
    
    def enable_all_neural_networks(self, enabled=True):
        """Enable or disable all neural network models."""
        for model_name in self.neural_network_models:
            self.neural_network_models[model_name]['enabled'] = enabled
        status = "enabled" if enabled else "disabled"
        print(f"✓ All neural network models {status}")
    
    def enable_by_category(self, category, enabled=True):
        """
        Enable or disable neural network models by category.
        
        Parameters:
        -----------
        category : str
            Category name ('CNN', 'LSTM', 'GRU', 'Hybrid')
        enabled : bool
            Whether to enable or disable
        """
        count = 0
        for model_name, config in self.neural_network_models.items():
            if config.get('category') == category:
                config['enabled'] = enabled
                count += 1
        
        status = "enabled" if enabled else "disabled"
        print(f"✓ {count} {category} models {status}")
    
    def get_enabled_traditional_models(self):
        """Get list of enabled traditional model names."""
        return [name for name, config in self.traditional_models.items() if config['enabled']]
    
    def get_enabled_neural_network_models(self):
        """Get list of enabled neural network model names."""
        return [name for name, config in self.neural_network_models.items() if config['enabled']]
    
    def get_all_enabled_models(self):
        """Get list of all enabled model names (traditional + neural networks)."""
        return self.get_enabled_traditional_models() + self.get_enabled_neural_network_models()
    
    def get_disabled_traditional_models(self):
        """Get list of disabled traditional model names."""
        return [name for name, config in self.traditional_models.items() if not config['enabled']]
    
    def get_disabled_neural_network_models(self):
        """Get list of disabled neural network model names."""
        return [name for name, config in self.neural_network_models.items() if not config['enabled']]
    
    def print_config(self, show_disabled=True):
        """
        Print current model configuration.
        
        Parameters:
        -----------
        show_disabled : bool
            Whether to show disabled models
        """
        print(f"\n{'='*80}")
        print(f"CENTRALIZED MODEL CONFIGURATION")
        print(f"{'='*80}")
        
        # Traditional ML Models
        print(f"\nTRADITIONAL ML MODELS:")
        enabled_trad = self.get_enabled_traditional_models()
        disabled_trad = self.get_disabled_traditional_models()
        
        print(f"\nEnabled models ({len(enabled_trad)}):")
        for name in enabled_trad:
            config = self.traditional_models[name]
            print(f"  ✓ {name:25s} - {config['description']}")
            print(f"    Training time: {config['training_time']}")
        
        if show_disabled and disabled_trad:
            print(f"\nDisabled models ({len(disabled_trad)}):")
            for name in disabled_trad:
                config = self.traditional_models[name]
                print(f"  ✗ {name:25s} - {config['description']}")
        
        # Neural Network Models
        print(f"\n{'-'*80}")
        print(f"NEURAL NETWORK MODELS:")
        enabled_nn = self.get_enabled_neural_network_models()
        disabled_nn = self.get_disabled_neural_network_models()
        
        # Group by category
        categories = {}
        for name in enabled_nn:
            category = self.neural_network_models[name].get('category', 'Other')
            if category not in categories:
                categories[category] = []
            categories[category].append(name)
        
        print(f"\nEnabled models ({len(enabled_nn)}):")
        for category, models in sorted(categories.items()):
            print(f"\n  {category} Models:")
            for name in models:
                config = self.neural_network_models[name]
                print(f"    ✓ {name:25s} - {config['description']}")
        
        if show_disabled and disabled_nn:
            print(f"\nDisabled models ({len(disabled_nn)}):")
            for name in disabled_nn:
                config = self.neural_network_models[name]
                print(f"  ✗ {name:25s} - {config['description']}")
        
        # Summary
        print(f"\n{'-'*80}")
        print(f"SUMMARY:")
        print(f"  Total models: {len(self.traditional_models) + len(self.neural_network_models)}")
        print(f"  Traditional ML: {len(enabled_trad)} enabled, {len(disabled_trad)} disabled")
        print(f"  Neural Networks: {len(enabled_nn)} enabled, {len(disabled_nn)} disabled")
        print(f"  Total enabled: {len(enabled_trad) + len(enabled_nn)}")
        print(f"{'='*80}\n")
    
    def get_model_info(self, model_name):
        """
        Get detailed information about a specific model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        
        Returns:
        --------
        dict : Model configuration or None if not found
        """
        if model_name in self.traditional_models:
            return {
                'type': 'traditional',
                'config': self.traditional_models[model_name]
            }
        elif model_name in self.neural_network_models:
            return {
                'type': 'neural_network',
                'config': self.neural_network_models[model_name]
            }
        else:
            return None
    
    def export_config(self):
        """
        Export current configuration as a dictionary.
        
        Returns:
        --------
        dict : Configuration dictionary
        """
        return {
            'traditional_models': {
                name: {
                    'enabled': config['enabled'],
                    'description': config['description'],
                    'training_time': config['training_time']
                }
                for name, config in self.traditional_models.items()
            },
            'neural_network_models': {
                name: {
                    'enabled': config['enabled'],
                    'description': config['description'],
                    'category': config.get('category', 'Other')
                }
                for name, config in self.neural_network_models.items()
            }
        }


# Singleton instance for global access
_model_config_instance = None

def get_model_config():
    """
    Get the singleton ModelConfig instance.
    
    Returns:
    --------
    ModelConfig : Singleton instance
    """
    global _model_config_instance
    if _model_config_instance is None:
        _model_config_instance = ModelConfig()
    return _model_config_instance


# Example usage
if __name__ == "__main__":
    print("Testing Model Configuration...")
    
    # Create config
    config = ModelConfig()
    
    # Print configuration
    config.print_config()
    
    # Test enable/disable
    print("\nTesting enable/disable functionality:")
    config.enable_model('random_forest', True)
    config.enable_model('cnn_simple', False)
    
    # Test category enable/disable
    print("\nDisabling all CNN models:")
    config.enable_by_category('CNN', False)
    
    # Print updated config
    config.print_config()
    
    print("\n✓ Model Configuration tested successfully!")
