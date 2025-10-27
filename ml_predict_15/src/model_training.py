"""
Model Training Module

Functions for training and evaluating ML models.
"""

import pandas as pd
import numpy as np
import os
import time
import shutil
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Import from new modular structure
from src.data_preparation import fit_scaler_minmax, prepare_data
from src.model_configs import (
    get_model_configs, 
    add_neural_network_models,
    HARDWARE_INFO,
    MODEL_ENABLED_CONFIG
)
from src.model_evaluation import (
    find_optimal_threshold,
    print_training_results_summary,
    print_test_results_summary,
    plot_training_comparison,
    plot_model_comparison
)
from src.model_loader import save_all_models
from src.utils import Utils

# Try to import SMOTE for handling imbalanced data
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Warning: imbalanced-learn not installed. Install with: pip install imbalanced-learn")

# Try to import MLflow for experiment tracking
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow not installed. Install with: pip install mlflow")

# Note: Hardware detection, model configs, and evaluation functions
# have been moved to separate modules:
# - src/model_configs.py: get_model_configs(), add_neural_network_models(), detect_hardware()
# - src/model_evaluation.py: find_optimal_threshold(), plotting and printing functions


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
    
    # Train the model with time tracking
    print(f"Training {model_name}...", end=' ', flush=True)
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    print(f"✓ Completed in {training_time:.2f} seconds")
    
    # Make predictions on validation set
    start_time = time.time()
    y_pred = model.predict(X_val_scaled)
    prediction_time = time.time() - start_time
    print(f"✓ Prediction completed in {prediction_time:.2f} seconds")
    
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
        'model': model,
        'training_time': training_time
    }


def train(df_train: pd.DataFrame, target_bars: int = 45, target_pct: float = 3.0, use_smote: bool = True, use_gpu: bool = False, n_jobs: int = -1, use_mlflow: bool = True, mlflow_tracking_uri: str = "http://localhost:5000"):
    """
    Train models on the provided training dataframe with hardware acceleration and MLflow tracking.
    
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
    use_mlflow : bool
        Whether to use MLflow for experiment tracking
    mlflow_tracking_uri : str
        MLflow tracking server URI (default: http://localhost:5000)
    
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

    # Initialize MLflow tracking if enabled
    mlflow_run = None
    if use_mlflow and MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            
            # Set experiment name
            experiment_name = "ml_predict_15/classification/crypto_price_prediction"
            mlflow.set_experiment(experiment_name)
            
            # Start MLflow run
            run_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            mlflow_run = mlflow.start_run(run_name=run_name)
            
            # Log parameters
            mlflow.log_param("target_bars", target_bars)
            mlflow.log_param("target_pct", target_pct)
            mlflow.log_param("use_smote", use_smote)
            mlflow.log_param("use_gpu", use_gpu)
            mlflow.log_param("n_jobs", n_jobs)
            mlflow.log_param("dataset_shape", f"{X.shape[0]}x{X.shape[1]}")
            mlflow.log_param("train_size", X_train.shape[0])
            mlflow.log_param("val_size", X_val.shape[0])
            mlflow.log_param("class_imbalance_ratio", f"{imbalance_ratio:.2f}")
            mlflow.log_param("smote_applied", use_smote and SMOTE_AVAILABLE and imbalance_ratio > 1.5)
            
            print(f"\n{'='*80}")
            print(f"MLFLOW TRACKING ENABLED")
            print(f"{'='*80}")
            print(f"Tracking URI: {mlflow_tracking_uri}")
            print(f"Experiment: {experiment_name}")
            print(f"Run: {run_name}")
            print(f"Run ID: {mlflow_run.info.run_id}")
            print(f"{'='*80}\n")
        except Exception as e:
            print(f"Warning: MLflow tracking failed to initialize: {e}")
            print("Continuing without MLflow tracking...\n")
            use_mlflow = False
    elif use_mlflow and not MLFLOW_AVAILABLE:
        print("MLflow requested but not available. Install with: pip install mlflow")
        print("Continuing without MLflow tracking...\n")
        use_mlflow = False
    
    # Get model configurations with hardware acceleration
    models = get_model_configs(use_gpu=use_gpu, n_jobs=n_jobs)
    
    # Add neural network models if TensorFlow is available
    if TENSORFLOW_AVAILABLE:
        sequence_length = 60
        input_shape = (sequence_length, X_train.shape[1])
        models = add_neural_network_models(models, input_shape, sequence_length)

    # Fit scaler on training data
    scaler, X_train_scaled = fit_scaler_minmax(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Filter enabled models only
    enabled_models = {name: data for name, data in models.items() if len(data) >= 3 and data[2]}
    disabled_models = {name: data for name, data in models.items() if len(data) < 3 or not data[2]}
    
    # Train and evaluate models
    print("="*80)
    print("TRAINING AND EVALUATING MODELS")
    print("="*80)
    print(f"Total models available: {len(models)}")
    print(f"Enabled models: {len(enabled_models)}")
    if disabled_models:
        print(f"Disabled models: {', '.join(disabled_models.keys())}")
    print()

    best_model = None
    best_score = 0
    best_model_name = ""
    results = {}
    total_training_time = 0

    # Training loop with progress bar (only enabled models)
    with tqdm(total=len(enabled_models), desc="Training Progress", unit="model", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        
        for model_name, model_data in enabled_models.items():
            model = model_data[0]
            pbar.set_description(f"Training {model_name}")
            
            # Train and evaluate
            result = train_and_evaluate_model(
                model, model_name, X_train_scaled, y_train, X_val_scaled, y_val
            )
            
            results[model_name] = result
            total_training_time += result['training_time']
            
            # Log to MLflow if enabled
            if use_mlflow and MLFLOW_AVAILABLE:
                try:
                    # Check if MLflow run is active
                    active_run = mlflow.active_run()
                    if active_run is None:
                        print(f"  ✗ MLflow: No active run found for {model_name}")
                    else:
                        # Log model-specific metrics
                        mlflow.log_metric(f"{model_name}_accuracy", float(result['accuracy']))
                        mlflow.log_metric(f"{model_name}_f1_score", float(result['f1_score']))
                        mlflow.log_metric(f"{model_name}_precision", float(result['precision']))
                        mlflow.log_metric(f"{model_name}_recall", float(result['recall']))
                        mlflow.log_metric(f"{model_name}_roc_auc", float(result['roc_auc']))
                        mlflow.log_metric(f"{model_name}_training_time", float(result['training_time']))
                        print(f"  ✓ MLflow: Logged metrics for {model_name}")
                except Exception as e:
                    print(f"  ✗ MLflow: Failed to log metrics for {model_name}")
                    print(f"     Error: {type(e).__name__}: {str(e)}")
            
            # Track best model
            if result['accuracy'] > best_score:
                best_score = result['accuracy']
                best_model = model
                best_model_name = model_name
            
            pbar.update(1)

    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Total training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
    print(f"Average time per model: {total_training_time/len(models):.2f} seconds")
    print(f"\nBEST MODEL: {best_model_name.upper()} with accuracy: {best_score:.4f}")
    print(f"{'='*80}")
    
    # Print training results summary and create visualizations
    print_training_results_summary(results)
    plot_training_comparison(results)

    # Create timestamped directory for this training session
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    models_dir = f'models/{timestamp}'
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"SAVING MODELS AND RESULTS")
    print(f"{'='*80}")
    print(f"Save directory: {models_dir}")
    
    # Save all enabled models and scaler to timestamped directory
    save_all_models(enabled_models, scaler, models_dir=models_dir)
    
    # Save the best model separately
    import joblib
    best_model_path = os.path.join(models_dir, f'{best_model_name}_best.joblib')
    joblib.dump(best_model, best_model_path)
    print(f"Best model saved to: {best_model_path}")
    
    # Save training results summary to CSV
    summary_data = []
    for model_name, metrics in results.items():
        summary_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'F1_Score': metrics['f1'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'ROC_AUC': metrics['roc_auc'],
            'Training_Time_Seconds': metrics.get('training_time', 0.0)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('F1_Score', ascending=False)
    
    # Add summary statistics
    summary_stats = pd.DataFrame([{
        'Model': 'SUMMARY',
        'Accuracy': '',
        'F1_Score': '',
        'Precision': '',
        'Recall': '',
        'ROC_AUC': '',
        'Training_Time_Seconds': ''
    }, {
        'Model': f'Best Model: {best_model_name}',
        'Accuracy': best_score,
        'F1_Score': results[best_model_name]['f1'],
        'Precision': results[best_model_name]['precision'],
        'Recall': results[best_model_name]['recall'],
        'ROC_AUC': results[best_model_name]['roc_auc'],
        'Training_Time_Seconds': results[best_model_name].get('training_time', 0.0)
    }, {
        'Model': 'Total Training Time',
        'Accuracy': '',
        'F1_Score': '',
        'Precision': '',
        'Recall': '',
        'ROC_AUC': '',
        'Training_Time_Seconds': f'{total_training_time:.2f}s ({total_training_time/60:.2f}min)'
    }, {
        'Model': 'Average Time per Model',
        'Accuracy': '',
        'F1_Score': '',
        'Precision': '',
        'Recall': '',
        'ROC_AUC': '',
        'Training_Time_Seconds': f'{total_training_time/len(models):.2f}s'
    }])
    
    summary_with_stats = pd.concat([summary_df, summary_stats], ignore_index=True)
    
    # Save to CSV
    csv_path = os.path.join(models_dir, 'training_results_summary.csv')
    summary_with_stats.to_csv(csv_path, index=False)
    print(f"Training results saved to: {csv_path}")
    
    # Save training configuration
    config_path = os.path.join(models_dir, 'training_config.txt')
    with open(config_path, 'w') as f:
        f.write(f"Training Configuration\n")
        f.write(f"{'='*80}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Target Bars: {target_bars}\n")
        f.write(f"Target Percentage: {target_pct}%\n")
        f.write(f"SMOTE Enabled: {use_smote}\n")
        f.write(f"GPU Enabled: {use_gpu}\n")
        f.write(f"CPU Cores: {n_jobs}\n")
        f.write(f"Dataset Shape: {X.shape}\n")
        f.write(f"Class Imbalance Ratio: {imbalance_ratio:.2f}:1\n")
        f.write(f"\nTraining Summary\n")
        f.write(f"{'='*80}\n")
        f.write(f"Total Models Trained: {len(models)}\n")
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"Best Accuracy: {best_score:.4f}\n")
        f.write(f"Total Training Time: {total_training_time:.2f}s ({total_training_time/60:.2f}min)\n")
        f.write(f"Average Time per Model: {total_training_time/len(models):.2f}s\n")
    print(f"Training config saved to: {config_path}")
    
    # Copy visualization to timestamped directory
    import shutil
    src_plot = 'plots/model_comparison_training.png'
    if os.path.exists(src_plot):
        dst_plot = os.path.join(models_dir, 'model_comparison_training.png')
        shutil.copy2(src_plot, dst_plot)
        print(f"Training plot copied to: {dst_plot}")
    
    print(f"{'='*80}\n")
    
    # Log to MLflow - best model and artifacts
    if use_mlflow and MLFLOW_AVAILABLE:
        try:
            # Log best model metrics
            mlflow.log_metric("best_accuracy", best_score)
            mlflow.log_metric("best_f1_score", results[best_model_name]['f1'])
            mlflow.log_metric("best_precision", results[best_model_name]['precision'])
            mlflow.log_metric("best_recall", results[best_model_name]['recall'])
            mlflow.log_metric("best_roc_auc", results[best_model_name]['roc_auc'])
            mlflow.log_metric("total_training_time", total_training_time)
            mlflow.log_metric("avg_training_time", total_training_time/len(enabled_models))
            
            # Log best model name as parameter
            mlflow.log_param("best_model_name", best_model_name)
            mlflow.log_param("num_models_trained", len(enabled_models))
            
            # Log the best model to MLflow
            try:
                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    artifact_path="best_model",
                    registered_model_name=f"ml_predict_15_{best_model_name}"
                )
                print(f"✓ Best model logged to MLflow: {best_model_name}")
            except Exception as e:
                print(f"Warning: Failed to log best model to MLflow: {e}")
            
            # Log artifacts (CSV, config, plot)
            try:
                mlflow.log_artifact(csv_path, artifact_path="results")
                mlflow.log_artifact(config_path, artifact_path="config")
                if os.path.exists(dst_plot):
                    mlflow.log_artifact(dst_plot, artifact_path="plots")
                print(f"✓ Artifacts logged to MLflow")
            except Exception as e:
                print(f"Warning: Failed to log artifacts to MLflow: {e}")
            
            # End MLflow run
            mlflow.end_run()
            
            print(f"\n{'='*80}")
            print(f"MLFLOW TRACKING COMPLETE")
            print(f"{'='*80}")
            print(f"View results at: {mlflow_tracking_uri}")
            print(f"Run ID: {mlflow_run.info.run_id}")
            print(f"{'='*80}\n")
        except Exception as e:
            print(f"Warning: MLflow logging failed: {e}")
            try:
                mlflow.end_run()
            except:
                pass

    return enabled_models, scaler, results, best_model_name


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
            'ROC AUC': metrics['roc_auc'],
            'Train Time (s)': metrics.get('training_time', 0.0)
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
