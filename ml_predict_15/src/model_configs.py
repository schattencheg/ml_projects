"""
Model Configurations Module

Functions for creating and configuring ML models.
"""

import multiprocessing
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Check for additional ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from src.neural_models import (
    create_lstm_model, create_cnn_model, create_hybrid_lstm_cnn_model,
    KerasClassifierWrapper, TENSORFLOW_AVAILABLE
)


def detect_hardware():
    """
    Detect available hardware for acceleration.
    
    Returns:
    --------
    hardware_info : dict
        Dictionary with hardware capabilities
    """
    hardware_info = {
        'cpu_cores': multiprocessing.cpu_count(),
        'gpu_available': False,
        'gpu_device': None,
        'cuda_available': False
    }
    
    # Check for CUDA/GPU support
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            hardware_info['gpu_available'] = True
            hardware_info['cuda_available'] = True
            hardware_info['gpu_device'] = gpus[0].name
            print(f"✓ GPU detected: {gpus[0].name}")
    except:
        pass
    
    # Check XGBoost GPU support
    if XGBOOST_AVAILABLE:
        try:
            import xgboost as xgb
            # Try to create a GPU-enabled booster
            test_params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
            hardware_info['xgboost_gpu'] = True
        except:
            hardware_info['xgboost_gpu'] = False
    
    # Check LightGBM GPU support
    if LIGHTGBM_AVAILABLE:
        try:
            import lightgbm as lgb
            # LightGBM GPU support check
            hardware_info['lightgbm_gpu'] = False  # Requires special compilation
        except:
            hardware_info['lightgbm_gpu'] = False
    
    print(f"✓ CPU cores available: {hardware_info['cpu_cores']}")
    
    return hardware_info


# Detect hardware at module load
HARDWARE_INFO = detect_hardware()

# Model enabled/disabled configuration
# Set to False to disable a model from training
# Current config: Only fast models (<10 min training time)
MODEL_ENABLED_CONFIG = {
    'logistic_regression': True,      # Fast: ~2-5 seconds
    'ridge_classifier': True,         # Fast: ~2-5 seconds
    'naive_bayes': True,              # Fast: ~3-5 seconds
    'knn_k_neighbours': False,        # SLOW: 50-120 seconds (disabled)
    'decision_tree': True,            # Fast: ~5-10 seconds
    'random_forest': True,            # Medium: ~15-30 seconds (acceptable)
    'gradient_boosting': False,       # SLOW: 60+ seconds (disabled, use XGBoost)
    'svm_support_vector_classification': False,  # SLOW: 100+ seconds (disabled)
    'xgboost': True,                  # Fast: ~3-5 seconds
    'lightgbm': True,                 # Fast: ~3-5 seconds
    'lstm': False,                    # SLOW: Neural network (disabled)
    'cnn': False,                     # SLOW: Neural network (disabled)
    'hybrid_lstm_cnn': False,         # SLOW: Neural network (disabled)
}


def get_model_configs(use_gpu=False, n_jobs=-1):
    """
    Get configurations for all available models with hardware acceleration.
    
    Parameters:
    -----------
    use_gpu : bool
        Whether to use GPU acceleration (if available)
    n_jobs : int
        Number of CPU cores to use (-1 = all cores, 1 = single core)
    
    Returns:
    --------
    models : dict
        Dictionary of model_name -> (model_instance, params_dict, enabled)
    """
    # Adjust n_jobs if needed
    if n_jobs == -1:
        # Use all cores - 1 to keep system responsive
        n_jobs = max(1, HARDWARE_INFO['cpu_cores'] - 1)
    
    models = {}
    
    # 1. Logistic Regression (supports multi-class with 'ovr' or 'multinomial')
    models['logistic_regression'] = (
        LogisticRegression(
            max_iter=1000, 
            class_weight='balanced',
            n_jobs=n_jobs,
            random_state=42,
            multi_class='ovr',  # One-vs-Rest for multi-class
            solver='lbfgs'  # Supports multi-class
        ),
        {'C': [0.001, 0.01, 0.1, 1, 10]},
        MODEL_ENABLED_CONFIG.get('logistic_regression', True)
    )
    
    # 2. Ridge Classifier
    models['ridge_classifier'] = (
        RidgeClassifier(
            class_weight='balanced',
            random_state=42
        ),
        {'alpha': [0.1, 1.0, 10.0]},
        MODEL_ENABLED_CONFIG.get('ridge_classifier', True)
    )
    
    # 3. Naive Bayes
    models['naive_bayes'] = (
        GaussianNB(),
        {'var_smoothing': [1e-9, 1e-8, 1e-7]},
        MODEL_ENABLED_CONFIG.get('naive_bayes', True)
    )
    
    # 4. K-Nearest Neighbors
    models['knn_k_neighbours'] = (
        KNeighborsClassifier(
            n_jobs=n_jobs
        ),
        {'n_neighbors': [3, 5, 7, 9]},
        MODEL_ENABLED_CONFIG.get('knn_k_neighbours', True)
    )
    
    # 5. Decision Tree
    models['decision_tree'] = (
        DecisionTreeClassifier(
            class_weight='balanced',
            random_state=42
        ),
        {'max_depth': [5, 10, 15, 20]},
        MODEL_ENABLED_CONFIG.get('decision_tree', True)
    )
    
    # 6. Random Forest
    models['random_forest'] = (
        RandomForestClassifier(
            class_weight='balanced',
            n_jobs=n_jobs,
            random_state=42
        ),
        {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]},
        MODEL_ENABLED_CONFIG.get('random_forest', True)
    )
    
    # 7. Gradient Boosting
    models['gradient_boosting'] = (
        GradientBoostingClassifier(
            random_state=42
        ),
        {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
        MODEL_ENABLED_CONFIG.get('gradient_boosting', True)
    )
    
    # 8. Support Vector Machine
    models['svm_support_vector_classification'] = (
        SVC(
            probability=True,
            class_weight='balanced',
            random_state=42
        ),
        {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']},
        MODEL_ENABLED_CONFIG.get('svm_support_vector_classification', True)
    )
    
    # 9. XGBoost (if available)
    if XGBOOST_AVAILABLE:
        xgb_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'random_state': 42,
            'n_jobs': n_jobs,
            'eval_metric': 'logloss'
        }
        
        # Add GPU support if requested and available
        if use_gpu and HARDWARE_INFO.get('xgboost_gpu', False):
            xgb_params['tree_method'] = 'gpu_hist'
            xgb_params['gpu_id'] = 0
        
        models['xgboost'] = (
            xgb.XGBClassifier(**xgb_params),
            {'n_estimators': [50, 100, 200], 'max_depth': [3, 6, 9]},
            MODEL_ENABLED_CONFIG.get('xgboost', True)
        )
    
    # 10. LightGBM (if available)
    if LIGHTGBM_AVAILABLE:
        lgb_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'random_state': 42,
            'n_jobs': n_jobs,
            'verbose': -1
        }
        
        # Add GPU support if requested and available
        if use_gpu and HARDWARE_INFO.get('lightgbm_gpu', False):
            lgb_params['device'] = 'gpu'
        
        models['lightgbm'] = (
            lgb.LGBMClassifier(**lgb_params),
            {'n_estimators': [50, 100, 200], 'max_depth': [3, 6, 9]},
            MODEL_ENABLED_CONFIG.get('lightgbm', True)
        )
    
    return models


def add_neural_network_models(models, input_shape, sequence_length=60):
    """
    Add neural network models to the models dictionary.
    
    Parameters:
    -----------
    models : dict
        Existing models dictionary
    input_shape : tuple
        Input shape for neural networks (sequence_length, features)
    sequence_length : int
        Sequence length for time series models
    
    Returns:
    --------
    models : dict
        Updated models dictionary with neural networks
    """
    if not TENSORFLOW_AVAILABLE:
        print("Warning: TensorFlow not available. Skipping neural network models.")
        return models
    
    # 11. LSTM
    models['lstm'] = (
        KerasClassifierWrapper(
            model_builder=create_lstm_model,
            input_shape=input_shape,
            sequence_length=sequence_length,
            epochs=50,
            batch_size=32,
            verbose=0
        ),
        {'sequence_length': sequence_length},
        MODEL_ENABLED_CONFIG.get('lstm', True)
    )
    
    # 12. CNN
    models['cnn'] = (
        KerasClassifierWrapper(
            model_builder=create_cnn_model,
            input_shape=input_shape,
            sequence_length=sequence_length,
            epochs=50,
            batch_size=32,
            verbose=0
        ),
        {'sequence_length': sequence_length},
        MODEL_ENABLED_CONFIG.get('cnn', True)
    )
    
    # 13. Hybrid LSTM-CNN
    models['hybrid_lstm_cnn'] = (
        KerasClassifierWrapper(
            model_builder=create_hybrid_lstm_cnn_model,
            input_shape=input_shape,
            sequence_length=sequence_length,
            epochs=50,
            batch_size=32,
            verbose=0
        ),
        {'sequence_length': sequence_length},
        MODEL_ENABLED_CONFIG.get('hybrid_lstm_cnn', True)
    )
    
    return models
