import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os
from datetime import datetime
from .evaluation import plot_model_comparison, plot_predictions


def visualize_yearly_performance(results_df: pd.DataFrame, 
                                save_path: str = "results/visualizations/yearly_performance.png"):
    """
    Visualize model performance by year
    
    Args:
        results_df: DataFrame with results by year and model
        save_path: Path to save the visualization
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if results_df.empty:
        print("No results to visualize")
        return
    
    # Create a figure with subplots for different metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = ['RMSE', 'MAE', 'R2', 'MAPE']
    
    for i, metric in enumerate(metrics):
        row = i // 2
        col = i % 2
        
        ax = axes[row, col]
        
        # Pivot the data for the current metric
        pivot_df = results_df.pivot(index='Year', columns='Model', values=metric)
        
        # Create heatmap
        sns.heatmap(pivot_df, annot=True, fmt='.4f', ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title(f'{metric} by Year and Model')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Yearly performance visualization saved to {save_path}")


def compare_model_performance(results_df: pd.DataFrame, 
                             save_path: str = "results/visualizations/model_comparison.png"):
    """
    Compare overall model performance
    
    Args:
        results_df: DataFrame with results by year and model
        save_path: Path to save the visualization
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if results_df.empty:
        print("No results to visualize")
        return
    
    # Group by model and calculate average metrics
    model_avg = results_df.groupby('Model').agg({
        'RMSE': 'mean',
        'MAE': 'mean', 
        'R2': 'mean',
        'MAPE': 'mean'
    }).reset_index()
    
    # Create a bar plot for each metric
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['RMSE', 'MAE', 'R2', 'MAPE']
    
    for i, metric in enumerate(metrics):
        row = i // 2
        col = i % 2
        
        ax = axes[row, col]
        
        # Sort models by the metric value (descending for R2, ascending for others)
        if metric == 'R2':
            sorted_df = model_avg.sort_values(by=metric, ascending=False)
        else:
            sorted_df = model_avg.sort_values(by=metric, ascending=True)
        
        ax.bar(sorted_df['Model'], sorted_df[metric])
        ax.set_title(f'Average {metric} by Model')
        ax.set_ylabel(metric)
        
        # Rotate x-axis labels if they're long
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Model comparison visualization saved to {save_path}")


def plot_yearly_trends(results_df: pd.DataFrame, 
                       save_path: str = "results/visualizations/yearly_trends.png"):
    """
    Plot yearly trends for each model
    
    Args:
        results_df: DataFrame with results by year and model
        save_path: Path to save the visualization
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if results_df.empty:
        print("No results to visualize")
        return
    
    # Get unique models
    models = results_df['Model'].unique()
    
    # Create subplots for each metric
    metrics = ['RMSE', 'MAE', 'R2', 'MAPE']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Yearly Trends by Model', fontsize=16)
    
    for i, metric in enumerate(metrics):
        row = i // 2
        col = i % 2
        
        ax = axes[row, col]
        
        for model in models:
            model_data = results_df[results_df['Model'] == model]
            if not model_data.empty:
                ax.plot(model_data['Year'], model_data[metric], 
                       marker='o', label=model, linewidth=2, markersize=6)
        
        ax.set_title(f'{metric} Over Time')
        ax.set_xlabel('Year')
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Yearly trends visualization saved to {save_path}")


def generate_performance_report(results_df: pd.DataFrame, 
                               save_path: str = "results/performance_report.txt"):
    """
    Generate a text report summarizing model performance
    
    Args:
        results_df: DataFrame with results by year and model
        save_path: Path to save the report
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if results_df.empty:
        print("No results to report")
        return
    
    with open(save_path, 'w') as f:
        f.write("OHLC Price Prediction Model Performance Report\n")
        f.write("="*50 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall performance by model
        f.write("1. Overall Performance by Model:\n")
        f.write("-" * 30 + "\n")
        
        model_avg = results_df.groupby('Model').agg({
            'RMSE': 'mean',
            'MAE': 'mean', 
            'R2': 'mean',
            'MAPE': 'mean'
        }).round(6)
        
        for model in model_avg.index:
            f.write(f"\n{model}:\n")
            f.write(f"  RMSE: {model_avg.loc[model, 'RMSE']:.6f}\n")
            f.write(f"  MAE: {model_avg.loc[model, 'MAE']:.6f}\n")
            f.write(f"  R2: {model_avg.loc[model, 'R2']:.6f}\n")
            f.write(f"  MAPE: {model_avg.loc[model, 'MAPE']:.6f}%\n")
        
        # Best model by metric
        f.write("\n\n2. Best Performing Models by Metric:\n")
        f.write("-" * 35 + "\n")
        
        metrics = ['RMSE', 'MAE', 'R2', 'MAPE']
        for metric in metrics:
            if metric == 'R2':
                # For R2, higher is better
                best_model = model_avg[metric].idxmax()
                best_value = model_avg[metric].max()
            else:
                # For other metrics, lower is better
                best_model = model_avg[metric].idxmin()
                best_value = model_avg[metric].min()
                
            f.write(f"  Best {metric}: {best_model} ({best_value:.6f})\n")
        
        # Yearly performance
        f.write("\n\n3. Yearly Performance:\n")
        f.write("-" * 20 + "\n")
        
        yearly_avg = results_df.groupby('Year').agg({
            'RMSE': 'mean',
            'MAE': 'mean', 
            'R2': 'mean',
            'MAPE': 'mean'
        }).round(6)
        
        for year in sorted(yearly_avg.index):
            f.write(f"\n{year}:\n")
            f.write(f"  Average RMSE: {yearly_avg.loc[year, 'RMSE']:.6f}\n")
            f.write(f"  Average MAE: {yearly_avg.loc[year, 'MAE']:.6f}\n")
            f.write(f"  Average R2: {yearly_avg.loc[year, 'R2']:.6f}\n")
            f.write(f"  Average MAPE: {yearly_avg.loc[year, 'MAPE']:.6f}%\n")
    
    print(f"Performance report saved to {save_path}")


def visualize_predictions_over_time(y_true_list: List[np.ndarray], 
                                   y_pred_list: List[np.ndarray],
                                   years: List[int],
                                   model_name: str,
                                   save_path: str = "results/visualizations/predictions_over_time.png"):
    """
    Visualize predictions over time for a specific model across different years
    
    Args:
        y_true_list: List of true values for each year
        y_pred_list: List of predicted values for each year
        years: List of years
        model_name: Name of the model
        save_path: Path to save the visualization
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    n_years = len(years)
    if n_years == 0:
        return
    
    # Calculate the number of rows and columns for subplots
    cols = min(3, n_years)
    rows = (n_years + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, year in enumerate(years):
        ax = axes[i]
        y_true = y_true_list[i]
        y_pred = y_pred_list[i]
        
        ax.plot(y_true, label='Actual', alpha=0.7)
        ax.plot(y_pred, label='Predicted', alpha=0.7)
        ax.set_title(f'{model_name} - {year}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
    
    # Hide any unused subplots
    for i in range(len(years), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Predictions over time visualization saved to {save_path}")