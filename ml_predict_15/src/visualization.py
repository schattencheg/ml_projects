"""
Visualization Module

Functions for creating visualizations of model performance.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def print_model_summary(results: dict):
    """
    Print a comprehensive summary of all model performances.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model results
    """
    print(f"\n{'='*80}")
    print("MODEL PERFORMANCE SUMMARY")
    print(f"{'='*80}\n")
    
    # Create summary dataframe
    summary_data = []
    for model_name, metrics in results.items():
        summary_data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'F1 Score': f"{metrics['f1']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'ROC AUC': f"{metrics['roc_auc']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by accuracy
    summary_df['Accuracy_num'] = summary_df['Accuracy'].astype(float)
    summary_df = summary_df.sort_values('Accuracy_num', ascending=False)
    summary_df = summary_df.drop('Accuracy_num', axis=1)
    
    print(summary_df.to_string(index=False))
    print(f"\n{'='*80}\n")


def create_visualizations(results: dict, close_test: pd.Series, y_test: pd.Series, idx_test: pd.Index):
    """
    Create and save visualizations for model performance.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model results with keys: accuracy, f1, precision, recall, y_pred, model
    close_test : pd.Series
        Test set Close prices (not scaled)
    y_test : pd.Series
        Test set actual target labels
    idx_test : pd.Index
        Test set indices
    """
    import os
    os.makedirs('plots', exist_ok=True)
    
    # 1. Model Comparison Bar Chart
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    f1_scores = [results[m]['f1'] for m in models]
    precisions = [results[m]['precision'] for m in models]
    recalls = [results[m]['recall'] for m in models]
    
    axes[0, 0].barh(models, accuracies, color='steelblue')
    axes[0, 0].set_xlabel('Accuracy')
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    axes[0, 1].barh(models, f1_scores, color='green')
    axes[0, 1].set_xlabel('F1 Score')
    axes[0, 1].set_title('Model F1 Score Comparison')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    axes[1, 0].barh(models, precisions, color='orange')
    axes[1, 0].set_xlabel('Precision')
    axes[1, 0].set_title('Model Precision Comparison')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    axes[1, 1].barh(models, recalls, color='red')
    axes[1, 1].set_xlabel('Recall')
    axes[1, 1].set_title('Model Recall Comparison')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: plots/model_comparison.png")
    plt.close()
    
    # 2. Prediction vs Actual for Best Model
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    y_pred_best = results[best_model_name]['y_pred']
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Plot close prices
    ax.plot(idx_test, close_test, label='Close Price', color='black', alpha=0.7)
    
    # Mark actual increases
    actual_increases = idx_test[y_test == 1]
    ax.scatter(actual_increases, close_test[y_test == 1], 
              color='green', marker='^', s=50, label='Actual Increase', alpha=0.6)
    
    # Mark predicted increases
    pred_increases = idx_test[y_pred_best == 1]
    ax.scatter(pred_increases, close_test[y_pred_best == 1], 
              color='blue', marker='v', s=50, label='Predicted Increase', alpha=0.6)
    
    ax.set_title(f'Predictions vs Actual - {best_model_name.upper()}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'plots/predictions_{best_model_name}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: plots/predictions_{best_model_name}.png")
    plt.close()
    
    print("\nVisualization complete!")
