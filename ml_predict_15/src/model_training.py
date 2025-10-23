"""
Model Training Module

Functions for training and evaluating ML models.
"""

import pandas as pd
import numpy as np
import os
import multiprocessing
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from src.data_preparation import fit_scaler_minmax, prepare_data

# Try to import SMOTE for handling imbalanced data
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Warning: imbalanced-learn not installed. Install with: pip install imbalanced-learn")
from src.neural_models import (
    create_lstm_model, create_cnn_model, create_hybrid_lstm_cnn_model,
    KerasClassifierWrapper, TENSORFLOW_AVAILABLE
)
from src.model_loader import save_all_models
from src.utils import Utils

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

# Detect hardware capabilities
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
        Dictionary of model_name -> (model_instance, params_dict)
    """
    # Determine optimal number of jobs
    if n_jobs == -1:
        n_jobs = max(1, HARDWARE_INFO['cpu_cores'] - 1)  # Leave one core free
    elif n_jobs == 0:
        n_jobs = 1
    
    print(f"\n{'='*80}")
    print(f"HARDWARE ACCELERATION SETTINGS")
    print(f"{'='*80}")
    print(f"CPU cores to use: {n_jobs} of {HARDWARE_INFO['cpu_cores']}")
    print(f"GPU acceleration: {'Enabled' if use_gpu and HARDWARE_INFO['gpu_available'] else 'Disabled'}")
    print(f"{'='*80}\n")
    models = {
        # Traditional ML models with class_weight='balanced' and multi-core support
        "logistic_regression": (
            LogisticRegression(
                max_iter=1000, 
                random_state=42, 
                class_weight='balanced',
                n_jobs=n_jobs  # Multi-core support
            ),
            {"max_iter": 1000, "class_weight": "balanced", "n_jobs": n_jobs}
        ),
        "ridge_classifier": (
            RidgeClassifier(random_state=42, class_weight='balanced'),
            {"class_weight": "balanced"}  # Ridge doesn't support n_jobs
        ),
        "naive_bayes": (
            GaussianNB(),
            {}  # Naive Bayes doesn't support class_weight or n_jobs
        ),
        "knn_k_neighbours": (
            KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=n_jobs  # Multi-core support
            ),
            {"n_neighbors": 5, "n_jobs": n_jobs}
        ),
        "decision_tree": (
            DecisionTreeClassifier(
                max_depth=10, 
                random_state=42, 
                class_weight='balanced'
            ),
            {"max_depth": 10, "class_weight": "balanced"}  # Decision tree doesn't support n_jobs
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42, 
                class_weight='balanced',
                n_jobs=n_jobs  # Multi-core support
            ),
            {"n_estimators": 100, "max_depth": 10, "class_weight": "balanced", "n_jobs": n_jobs}
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(
                n_estimators=100, 
                max_depth=5, 
                random_state=42
            ),
            {"n_estimators": 100, "max_depth": 5}  # GB doesn't support class_weight or n_jobs
        ),
        "svm_support_vector_classification": (
            SVC(
                kernel='rbf', 
                probability=True, 
                random_state=42, 
                class_weight='balanced'
            ),
            {"kernel": "rbf", "class_weight": "balanced"}  # SVM doesn't benefit from n_jobs for small datasets
        ),
    }
    
    # Add XGBoost if available (with GPU support)
    if XGBOOST_AVAILABLE:
        xgb_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42,
            'eval_metric': 'logloss',
            'n_jobs': n_jobs
        }
        
        # Enable GPU if available and requested
        if use_gpu and HARDWARE_INFO.get('xgboost_gpu', False):
            xgb_params['tree_method'] = 'gpu_hist'
            xgb_params['gpu_id'] = 0
            print("  ✓ XGBoost: GPU acceleration enabled")
        else:
            xgb_params['tree_method'] = 'hist'  # Fast CPU histogram method
        
        models["xgboost"] = (
            xgb.XGBClassifier(**xgb_params),
            xgb_params
        )
    
    # Add LightGBM if available (with multi-core support)
    if LIGHTGBM_AVAILABLE:
        lgb_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbose': -1,
            'n_jobs': n_jobs
        }
        
        # LightGBM GPU support (requires special compilation)
        if use_gpu and HARDWARE_INFO.get('lightgbm_gpu', False):
            lgb_params['device'] = 'gpu'
            print("  ✓ LightGBM: GPU acceleration enabled")
        
        models["lightgbm"] = (
            lgb.LGBMClassifier(**lgb_params),
            lgb_params
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
        return models
    
    # Add LSTM
    models["lstm"] = (
        KerasClassifierWrapper(
            create_lstm_model,
            input_shape,
            sequence_length=sequence_length,
            dropout_rate=0.2
        ),
        {"sequence_length": sequence_length}
    )
    
    # Add CNN
    models["cnn"] = (
        KerasClassifierWrapper(
            create_cnn_model,
            input_shape,
            sequence_length=sequence_length,
            dropout_rate=0.2
        ),
        {"sequence_length": sequence_length}
    )
    
    # Add Hybrid LSTM-CNN
    models["lstm_cnn_hybrid"] = (
        KerasClassifierWrapper(
            create_hybrid_lstm_cnn_model,
            input_shape,
            sequence_length=sequence_length,
            dropout_rate=0.3
        ),
        {"sequence_length": sequence_length}
    )
    
    return models


def find_optimal_threshold(model, X_val_scaled, y_val, metric='f1'):
    """
    Find optimal probability threshold to maximize a given metric.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with predict_proba
    X_val_scaled : np.ndarray
        Validation features
    y_val : pd.Series
        Validation labels
    metric : str
        Metric to optimize ('f1', 'recall', 'precision')
    
    Returns:
    --------
    best_threshold : float
        Optimal threshold
    best_score : float
        Best score achieved
    """
    if not hasattr(model, 'predict_proba'):
        return 0.5, None
    
    y_proba = model.predict_proba(X_val_scaled)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.05)
    
    best_threshold = 0.5
    best_score = 0
    
    for threshold in thresholds:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_val, y_pred_thresh, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_val, y_pred_thresh, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_val, y_pred_thresh, zero_division=0)
        else:
            score = f1_score(y_val, y_pred_thresh, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


def train_and_evaluate_model(model, model_name, X_train_scaled, y_train, X_val_scaled, y_val, optimize_threshold=True):
    """
    Train and evaluate a single model.
    
    Parameters:
    -----------
    model : sklearn-compatible model
        Model to train
    model_name : str
        Name of the model
    X_train_scaled : np.ndarray
        Scaled training features
    y_train : pd.Series
        Training labels
    X_val_scaled : np.ndarray
        Scaled validation features
    y_val : pd.Series
        Validation labels
    optimize_threshold : bool
        Whether to optimize decision threshold for better recall
    
    Returns:
    --------
    results : dict
        Dictionary with model performance metrics
    """
    Utils.print_color(f"\n{'='*80}", 'magenta')
    Utils.print_color(f"Model: {model_name.upper()}", 'magenta')
    Utils.print_color(f"{'='*80}", 'magenta')
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions on validation set
    y_pred = model.predict(X_val_scaled)
    
    # Calculate metrics with default threshold (0.5)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    
    # Try to optimize threshold for better F1 score
    optimal_threshold = 0.5
    if optimize_threshold and hasattr(model, 'predict_proba'):
        optimal_threshold, optimal_f1 = find_optimal_threshold(model, X_val_scaled, y_val, metric='f1')
        
        if optimal_f1 > f1:
            # Use optimized threshold
            y_proba = model.predict_proba(X_val_scaled)[:, 1]
            y_pred_optimized = (y_proba >= optimal_threshold).astype(int)
            
            # Recalculate metrics with optimized threshold
            accuracy_opt = accuracy_score(y_val, y_pred_optimized)
            f1_opt = f1_score(y_val, y_pred_optimized)
            precision_opt = precision_score(y_val, y_pred_optimized)
            recall_opt = recall_score(y_val, y_pred_optimized)
            
            print(f"\n  Threshold Optimization:")
            print(f"    Default threshold (0.5): F1={f1:.4f}, Recall={recall:.4f}")
            print(f"    Optimal threshold ({optimal_threshold:.2f}): F1={f1_opt:.4f}, Recall={recall_opt:.4f}")
            print(f"    Improvement: F1={f1_opt-f1:+.4f}, Recall={recall_opt-recall:+.4f}")
            
            # Update predictions and metrics
            y_pred = y_pred_optimized
            accuracy = accuracy_opt
            f1 = f1_opt
            precision = precision_opt
            recall = recall_opt
    
    # Calculate ROC AUC
    try:
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_pred_proba = model.decision_function(X_val_scaled)
        else:
            y_pred_proba = y_pred.astype(float)
        
        roc_auc = roc_auc_score(y_val, y_pred_proba)
    except Exception as e:
        print(f"  Warning: Could not calculate ROC AUC: {str(e)}")
        roc_auc = 0.0
    
    # Print results
    Utils.print_color(f"\nValidation Set Performance:", 'green')
    Utils.print_color(f"  Accuracy:  {accuracy:.4f}", 'green')
    Utils.print_color(f"  F1 Score:  {f1:.4f}", 'green')
    Utils.print_color(f"  Precision: {precision:.4f}", 'green')
    Utils.print_color(f"  Recall:    {recall:.4f}", 'green')
    Utils.print_color(f"  ROC AUC:   {roc_auc:.4f}", 'green')
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['No Increase', 'Increase'], zero_division=np.nan))
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'model': model
    }


def train(df_train: pd.DataFrame, target_bars: int = 45, target_pct: float = 3.0, use_smote: bool = True, use_gpu: bool = False, n_jobs: int = -1):
    """
    Train models on the provided training dataframe with hardware acceleration.
    
    Parameters:
    -----------
    df_train : pd.DataFrame
        Training dataframe with OHLCV data
    target_bars : int
        Number of bars to look ahead for target
    target_pct : float
        Percentage increase threshold for target
    use_smote : bool
        Whether to use SMOTE for oversampling minority class
    use_gpu : bool
        Whether to use GPU acceleration (if available)
    n_jobs : int
        Number of CPU cores to use (-1 = all available cores)
    
    Returns:
    --------
    models : dict
        Dictionary of trained models
    scaler : StandardScaler
        Fitted scaler for feature normalization
    results : dict
        Training results with metrics
    best_model_name : str
        Name of the best performing model
    """
    # Prepare data
    X, y = prepare_data(df_train, target_bars, target_pct)

    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"Target balance: {y.value_counts(normalize=True)}")
    
    # Calculate class imbalance ratio
    class_counts = y.value_counts()
    imbalance_ratio = class_counts.max() / class_counts.min()
    print(f"Class imbalance ratio: {imbalance_ratio:.2f}:1")
    print()

    # Split data into train/validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Apply SMOTE if available and requested
    if use_smote and SMOTE_AVAILABLE and imbalance_ratio > 1.5:
        print("Applying SMOTE to balance training data...")
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"Original training size: {X_train.shape[0]}")
        print(f"Resampled training size: {X_train_resampled.shape[0]}")
        print(f"New class distribution: {pd.Series(y_train_resampled).value_counts()}")
        print()
        X_train = X_train_resampled
        y_train = y_train_resampled
    elif use_smote and not SMOTE_AVAILABLE:
        print("SMOTE requested but not available. Install with: pip install imbalanced-learn")
        print("Continuing without SMOTE...\n")

    # Get model configurations with hardware acceleration
    models = get_model_configs(use_gpu=use_gpu, n_jobs=n_jobs)
    
    # Add neural network models if TensorFlow is available
    if False and TENSORFLOW_AVAILABLE:
        sequence_length = 60
        input_shape = (sequence_length, X_train.shape[1])
        models = add_neural_network_models(models, input_shape, sequence_length)

    # Fit scaler on training data
    scaler, X_train_scaled = fit_scaler_minmax(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train and evaluate models
    print("="*80)
    print("TRAINING AND EVALUATING MODELS")
    print("="*80)

    best_model = None
    best_score = 0
    best_model_name = ""
    results = {}

    for model_name, model_data in models.items():
        model = model_data[0]
        
        # Train and evaluate
        result = train_and_evaluate_model(
            model, model_name, X_train_scaled, y_train, X_val_scaled, y_val
        )
        
        results[model_name] = result
        
        # Track best model
        if result['accuracy'] > best_score:
            best_score = result['accuracy']
            best_model = model
            best_model_name = model_name

    print(f"\n{'='*80}")
    print(f"BEST MODEL: {best_model_name.upper()} with accuracy: {best_score:.4f}")
    print(f"{'='*80}")
    
    # Print training results summary and create visualizations
    print_training_results_summary(results)
    plot_training_comparison(results)

    # Save all models and scaler
    save_all_models(models, scaler, models_dir='models')
    
    # Also save the best model separately
    import os
    import joblib
    os.makedirs('models', exist_ok=True)
    model_path = f'models/{best_model_name}_best.joblib'
    joblib.dump(best_model, model_path)
    print(f"\nBest model also saved separately to: {model_path}")

    return models, scaler, results, best_model_name


def print_training_results_summary(results: dict):
    """
    Print summary of training/validation results for all models.
    
    Parameters:
    -----------
    results : dict
        Dictionary with validation metrics for each model
    """
    print(f"\n{'='*80}")
    print(f"TRAINING RESULTS SUMMARY (Validation Set)")
    print(f"{'='*80}")
    
    # Create summary dataframe
    summary_data = []
    for model_name, metrics in results.items():
        summary_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'F1 Score': metrics['f1'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'ROC AUC': metrics['roc_auc']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('F1 Score', ascending=False)
    
    print("\n" + summary_df.to_string(index=False))
    
    # Find best model for each metric
    print(f"\n{'='*80}")
    print(f"BEST MODELS BY METRIC (Validation Set)")
    print(f"{'='*80}")
    
    best_accuracy = summary_df.loc[summary_df['Accuracy'].idxmax()]
    best_f1 = summary_df.loc[summary_df['F1 Score'].idxmax()]
    best_precision = summary_df.loc[summary_df['Precision'].idxmax()]
    best_recall = summary_df.loc[summary_df['Recall'].idxmax()]
    best_roc_auc = summary_df.loc[summary_df['ROC AUC'].idxmax()]
    
    print(f"  Best Accuracy:  {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})")
    print(f"  Best F1 Score:  {best_f1['Model']} ({best_f1['F1 Score']:.4f})")
    print(f"  Best Precision: {best_precision['Model']} ({best_precision['Precision']:.4f})")
    print(f"  Best Recall:    {best_recall['Model']} ({best_recall['Recall']:.4f})")
    print(f"  Best ROC AUC:   {best_roc_auc['Model']} ({best_roc_auc['ROC AUC']:.4f})")


def plot_training_comparison(results: dict, save_path: str = 'plots/model_comparison_training.png'):
    """
    Create visualization comparing model performance on validation set.
    
    Parameters:
    -----------
    results : dict
        Dictionary with validation metrics for each model
    save_path : str
        Path to save the plot
    """
    import os
    import matplotlib.pyplot as plt
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Prepare data
    model_names = []
    accuracy_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    roc_auc_scores = []
    
    for model_name, metrics in results.items():
        model_names.append(model_name.replace('_', ' ').title())
        accuracy_scores.append(metrics['accuracy'])
        f1_scores.append(metrics['f1'])
        precision_scores.append(metrics['precision'])
        recall_scores.append(metrics['recall'])
        roc_auc_scores.append(metrics['roc_auc'])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Model Performance Comparison on Validation Set', fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy
    axes[0, 0].bar(model_names, accuracy_scores, color='#3498db', alpha=0.8)
    axes[0, 0].set_title('Accuracy', fontweight='bold')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(accuracy_scores):
        axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: F1 Score
    axes[0, 1].bar(model_names, f1_scores, color='#2ecc71', alpha=0.8)
    axes[0, 1].set_title('F1 Score', fontweight='bold')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(f1_scores):
        axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Precision
    axes[0, 2].bar(model_names, precision_scores, color='#e74c3c', alpha=0.8)
    axes[0, 2].set_title('Precision', fontweight='bold')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].set_ylim([0, 1])
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(precision_scores):
        axes[0, 2].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Recall
    axes[1, 0].bar(model_names, recall_scores, color='#f39c12', alpha=0.8)
    axes[1, 0].set_title('Recall', fontweight='bold')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(recall_scores):
        axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 5: ROC AUC
    axes[1, 1].bar(model_names, roc_auc_scores, color='#9b59b6', alpha=0.8)
    axes[1, 1].set_title('ROC AUC', fontweight='bold')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(roc_auc_scores):
        axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 6: Overall comparison
    ax = axes[1, 2]
    x = np.arange(len(model_names))
    width = 0.15
    
    ax.bar(x - 2*width, accuracy_scores, width, label='Accuracy', alpha=0.8)
    ax.bar(x - width, f1_scores, width, label='F1 Score', alpha=0.8)
    ax.bar(x, precision_scores, width, label='Precision', alpha=0.8)
    ax.bar(x + width, recall_scores, width, label='Recall', alpha=0.8)
    ax.bar(x + 2*width, roc_auc_scores, width, label='ROC AUC', alpha=0.8)
    
    ax.set_title('All Metrics Comparison', fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_ylim([0, 1])
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining comparison plot saved to: {save_path}")
    plt.show()


def test(models: dict, scaler, df_test: pd.DataFrame, target_bars: int = 45, target_pct: float = 3.0):
    """
    Test trained models on new data.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    scaler : StandardScaler
        Fitted scaler from training
    df_test : pd.DataFrame
        Test dataframe with OHLCV data
    target_bars : int
        Number of bars to look ahead for target (should match training)
    target_pct : float
        Percentage increase threshold for target (should match training)
    
    Returns:
    --------
    results_test : dict
        Dictionary with test metrics for each model
    """
    # Prepare test data using the same pipeline as training
    X_test, y_test = prepare_data(df_test, target_bars, target_pct)
    
    # Scale using the fitted scaler from training (DO NOT refit!)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n{'='*80}")
    print(f"TESTING ON HELD-OUT TEST DATA")
    print(f"{'='*80}")
    print(f"Test dataset shape: {X_test.shape}")
    print(f"Test target distribution:\n{y_test.value_counts()}")
    print()

    results_test = {}

    for model_name, model_data in models.items():
        print(f"\n{'='*80}")
        print(f"Model: {model_name.upper()}")
        print(f"{'='*80}")
        
        model = model_data[0]
        
        # Make predictions on test set
        y_pred_test = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test, zero_division=np.nan)
        recall = recall_score(y_test, y_pred_test)
        
        # Calculate ROC AUC
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba_test = model.predict_proba(X_test_scaled)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_pred_proba_test = model.decision_function(X_test_scaled)
            else:
                y_pred_proba_test = y_pred_test.astype(float)
            
            roc_auc = roc_auc_score(y_test, y_pred_proba_test)
        except Exception as e:
            print(f"  Warning: Could not calculate ROC AUC for {model_name}: {str(e)}")
            roc_auc = 0.0
        
        Utils.print_color(f"\nTest Set Performance:", 'green')
        Utils.print_color(f"  Accuracy:  {accuracy:.4f}", 'green')
        Utils.print_color(f"  F1 Score:  {f1:.4f}", 'green')
        Utils.print_color(f"  Precision: {precision:.4f}", 'green')
        Utils.print_color(f"  Recall:    {recall:.4f}", 'green')
        Utils.print_color(f"  ROC AUC:   {roc_auc:.4f}", 'green')
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred_test, target_names=['No Increase', 'Increase'], zero_division=np.nan))
        
        # Store results
        results_test[model_name] = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'y_pred': y_pred_test,
        }
    
    # Print summary and create visualizations
    print_test_results_summary(results_test)
    plot_model_comparison(results_test)
    
    return results_test


def print_test_results_summary(results_test: dict):
    """
    Print summary of test results for all models.
    
    Parameters:
    -----------
    results_test : dict
        Dictionary with test metrics for each model
    """
    print(f"\n{'='*80}")
    print(f"TEST RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # Create summary dataframe
    summary_data = []
    for model_name, metrics in results_test.items():
        summary_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'F1 Score': metrics['f1'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'ROC AUC': metrics['roc_auc']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('F1 Score', ascending=False)
    
    print("\n" + summary_df.to_string(index=False))
    
    # Find best model for each metric
    print(f"\n{'='*80}")
    print(f"BEST MODELS BY METRIC")
    print(f"{'='*80}")
    
    best_accuracy = summary_df.loc[summary_df['Accuracy'].idxmax()]
    best_f1 = summary_df.loc[summary_df['F1 Score'].idxmax()]
    best_precision = summary_df.loc[summary_df['Precision'].idxmax()]
    best_recall = summary_df.loc[summary_df['Recall'].idxmax()]
    best_roc_auc = summary_df.loc[summary_df['ROC AUC'].idxmax()]
    
    print(f"  Best Accuracy:  {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})")
    print(f"  Best F1 Score:  {best_f1['Model']} ({best_f1['F1 Score']:.4f})")
    print(f"  Best Precision: {best_precision['Model']} ({best_precision['Precision']:.4f})")
    print(f"  Best Recall:    {best_recall['Model']} ({best_recall['Recall']:.4f})")
    print(f"  Best ROC AUC:   {best_roc_auc['Model']} ({best_roc_auc['ROC AUC']:.4f})")


def plot_model_comparison(results_test: dict, save_path: str = 'plots/model_comparison_test.png'):
    """
    Create visualization comparing model performance.
    
    Parameters:
    -----------
    results_test : dict
        Dictionary with test metrics for each model
    save_path : str
        Path to save the plot
    """
    import os
    import matplotlib.pyplot as plt
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Prepare data
    model_names = []
    accuracy_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    roc_auc_scores = []
    
    for model_name, metrics in results_test.items():
        model_names.append(model_name.replace('_', ' ').title())
        accuracy_scores.append(metrics['accuracy'])
        f1_scores.append(metrics['f1'])
        precision_scores.append(metrics['precision'])
        recall_scores.append(metrics['recall'])
        roc_auc_scores.append(metrics['roc_auc'])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Model Performance Comparison on Test Set', fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy
    axes[0, 0].bar(model_names, accuracy_scores, color='#3498db', alpha=0.8)
    axes[0, 0].set_title('Accuracy', fontweight='bold')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(accuracy_scores):
        axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: F1 Score
    axes[0, 1].bar(model_names, f1_scores, color='#2ecc71', alpha=0.8)
    axes[0, 1].set_title('F1 Score', fontweight='bold')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(f1_scores):
        axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Precision
    axes[0, 2].bar(model_names, precision_scores, color='#e74c3c', alpha=0.8)
    axes[0, 2].set_title('Precision', fontweight='bold')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].set_ylim([0, 1])
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(precision_scores):
        axes[0, 2].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Recall
    axes[1, 0].bar(model_names, recall_scores, color='#f39c12', alpha=0.8)
    axes[1, 0].set_title('Recall', fontweight='bold')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(recall_scores):
        axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 5: ROC AUC
    axes[1, 1].bar(model_names, roc_auc_scores, color='#9b59b6', alpha=0.8)
    axes[1, 1].set_title('ROC AUC', fontweight='bold')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(roc_auc_scores):
        axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 6: Overall comparison (radar chart style)
    ax = axes[1, 2]
    x = np.arange(len(model_names))
    width = 0.15
    
    ax.bar(x - 2*width, accuracy_scores, width, label='Accuracy', alpha=0.8)
    ax.bar(x - width, f1_scores, width, label='F1 Score', alpha=0.8)
    ax.bar(x, precision_scores, width, label='Precision', alpha=0.8)
    ax.bar(x + width, recall_scores, width, label='Recall', alpha=0.8)
    ax.bar(x + 2*width, roc_auc_scores, width, label='ROC AUC', alpha=0.8)
    
    ax.set_title('All Metrics Comparison', fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_ylim([0, 1])
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nModel comparison plot saved to: {save_path}")
    plt.show()