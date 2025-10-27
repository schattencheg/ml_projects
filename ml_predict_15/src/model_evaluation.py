"""
Model Evaluation Module

Functions for evaluating ML models and generating metrics/visualizations.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


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
    y_proba = model.predict_proba(X_val_scaled)[:, 1]
    
    best_threshold = 0.5
    best_score = 0
    
    # Test thresholds from 0.1 to 0.9
    for threshold in np.arange(0.1, 1.0, 0.05):
        y_pred = (y_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_val, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_val, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_val, y_pred, zero_division=0)
        else:
            score = f1_score(y_val, y_pred, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


def print_training_results_summary(results: dict):
    """
    Print summary of training/validation results for all models.
    
    Parameters:
    -----------
    results : dict
        Dictionary with validation metrics for each model
    """
    print("\n" + "="*80)
    print("TRAINING RESULTS SUMMARY (Validation Set)")
    print("="*80)
    
    # Create DataFrame for better formatting
    summary_data = []
    for model_name, metrics in results.items():
        summary_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'F1 Score': f"{metrics['f1']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'ROC AUC': f"{metrics['roc_auc']:.4f}",
            'Train Time (s)': f"{metrics.get('training_time', 0.0):.2f}"
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Sort by F1 Score
    df_summary = df_summary.sort_values('F1 Score', ascending=False)
    
    print(df_summary.to_string(index=False))
    print("="*80 + "\n")


def print_test_results_summary(results_test: dict):
    """
    Print summary of test results for all models.
    
    Parameters:
    -----------
    results_test : dict
        Dictionary with test metrics for each model
    """
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    
    # Create DataFrame for better formatting
    summary_data = []
    for model_name, metrics in results_test.items():
        summary_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'F1 Score': f"{metrics['f1']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'ROC AUC': f"{metrics['roc_auc']:.4f}"
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Sort by F1 Score
    df_summary = df_summary.sort_values('F1 Score', ascending=False)
    
    print(df_summary.to_string(index=False))
    print("="*80 + "\n")


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
    training_times = []
    
    for model_name, metrics in results.items():
        model_names.append(model_name.replace('_', ' ').title())
        accuracy_scores.append(metrics['accuracy'])
        f1_scores.append(metrics['f1'])
        precision_scores.append(metrics['precision'])
        recall_scores.append(metrics['recall'])
        roc_auc_scores.append(metrics['roc_auc'])
        training_times.append(metrics.get('training_time', 0.0))
    
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
    
    # Plot 6: Training Time
    axes[1, 2].bar(model_names, training_times, color='#1abc9c', alpha=0.8)
    axes[1, 2].set_title('Training Time', fontweight='bold')
    axes[1, 2].set_ylabel('Seconds')
    axes[1, 2].tick_params(axis='x', rotation=45)
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(training_times):
        axes[1, 2].text(i, v + 0.5, f'{v:.1f}s', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nModel comparison plot saved to: {save_path}")
    plt.show()


def plot_model_comparison(results_test: dict, save_path: str = 'plots/model_comparison_test.png'):
    """
    Create visualization comparing model performance on test set.
    
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
