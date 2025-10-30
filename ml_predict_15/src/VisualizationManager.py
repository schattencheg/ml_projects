"""
Visualization Manager Module

Handles all visualization creation for ML model training and testing results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class VisualizationManager:
    """
    Creates visualizations for ML model training and testing results.
    """
    
    def __init__(self, output_dir='visualizations', style='seaborn-v0_8-darkgrid'):
        """
        Initialize VisualizationManager.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save visualizations
        style : str
            Matplotlib style to use
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        try:
            plt.style.use(style)
        except:
            # Fallback to default if style not available
            pass
        
        # Set default figure parameters
        plt.rcParams['figure.figsize'] = (15, 10)
        plt.rcParams['font.size'] = 10
    
    def create_training_visualizations(self, df_summary, filename, output_dir=None):
        """
        Create training visualizations.
        
        Parameters:
        -----------
        df_summary : pd.DataFrame
            Summary DataFrame with training metrics
        filename : str
            Filename to save (without extension)
        output_dir : str (optional)
            Override default output directory
            
        Returns:
        --------
        str : Path to saved figure
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Results', fontsize=16, fontweight='bold')
        
        # 1. Accuracy comparison
        ax1 = axes[0, 0]
        df_plot = df_summary.sort_values('Train Accuracy', ascending=True)
        ax1.barh(df_plot['Model'], df_plot['Train Accuracy'], color='skyblue')
        ax1.set_xlabel('Accuracy')
        ax1.set_title('Training Accuracy by Model')
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. F1 Score comparison
        ax2 = axes[0, 1]
        df_plot = df_summary.sort_values('Train F1', ascending=True)
        ax2.barh(df_plot['Model'], df_plot['Train F1'], color='lightcoral')
        ax2.set_xlabel('F1 Score')
        ax2.set_title('Training F1 Score by Model')
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Training time
        ax3 = axes[1, 0]
        df_plot = df_summary.sort_values('Train Time (s)', ascending=True)
        ax3.barh(df_plot['Model'], df_plot['Train Time (s)'], color='lightgreen')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_title('Training Time by Model')
        ax3.grid(axis='x', alpha=0.3)
        
        # 4. Metrics heatmap
        ax4 = axes[1, 1]
        metrics_cols = ['Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1']
        heatmap_data = df_summary[['Model'] + metrics_cols].set_index('Model')
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax4)
        ax4.set_title('Training Metrics Heatmap')
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return fig_path
    
    def create_test_visualizations(self, df_summary, test_results, y_test, 
                                   filename, target_names=None, output_dir=None):
        """
        Create test visualizations.
        
        Parameters:
        -----------
        df_summary : pd.DataFrame
            Summary DataFrame with test metrics
        test_results : dict
            Test results dictionary
        y_test : array-like
            True test labels
        filename : str
            Filename to save (without extension)
        target_names : list (optional)
            Names of target classes
        output_dir : str (optional)
            Override default output directory
            
        Returns:
        --------
        str : Path to saved figure
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Test Results', fontsize=16, fontweight='bold')
        
        # 1. Accuracy comparison
        ax1 = axes[0, 0]
        df_plot = df_summary.sort_values('Accuracy', ascending=True)
        ax1.barh(df_plot['Model'], df_plot['Accuracy'], color='skyblue')
        ax1.set_xlabel('Accuracy')
        ax1.set_title('Test Accuracy by Model')
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. F1 Score comparison
        ax2 = axes[0, 1]
        df_plot = df_summary.sort_values('F1 Score', ascending=True)
        ax2.barh(df_plot['Model'], df_plot['F1 Score'], color='lightcoral')
        ax2.set_xlabel('F1 Score')
        ax2.set_title('Test F1 Score by Model')
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Metrics comparison
        ax3 = axes[1, 0]
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        x = np.arange(len(df_summary))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            ax3.bar(x + i*width, df_summary[metric], width, label=metric)
        
        ax3.set_xlabel('Model')
        ax3.set_ylabel('Score')
        ax3.set_title('Test Metrics Comparison')
        ax3.set_xticks(x + width * 1.5)
        ax3.set_xticklabels(df_summary['Model'], rotation=45, ha='right')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Confusion matrix for best model
        ax4 = axes[1, 1]
        best_model = df_summary.iloc[0]['Model']
        y_pred = test_results[best_model]['predictions']
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
        ax4.set_title(f'Confusion Matrix - {best_model}')
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Actual')
        
        if target_names:
            ax4.set_xticklabels(target_names)
            ax4.set_yticklabels(target_names)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return fig_path
    
    def create_comparison_visualizations(self, df_comparison, filename, output_dir=None):
        """
        Create train vs test comparison visualizations.
        
        Parameters:
        -----------
        df_comparison : pd.DataFrame
            Comparison DataFrame
        filename : str
            Filename to save (without extension)
        output_dir : str (optional)
            Override default output directory
            
        Returns:
        --------
        str : Path to saved figure
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Train vs Test Comparison', fontsize=16, fontweight='bold')
        
        # 1. Accuracy comparison
        ax1 = axes[0, 0]
        x = np.arange(len(df_comparison))
        width = 0.35
        ax1.bar(x - width/2, df_comparison['Train Acc'], width, label='Train', color='skyblue')
        ax1.bar(x + width/2, df_comparison['Test Acc'], width, label='Test', color='lightcoral')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Train vs Test Accuracy')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df_comparison['Model'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. F1 Score comparison
        ax2 = axes[0, 1]
        ax2.bar(x - width/2, df_comparison['Train F1'], width, label='Train', color='lightgreen')
        ax2.bar(x + width/2, df_comparison['Test F1'], width, label='Test', color='orange')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Train vs Test F1 Score')
        ax2.set_xticks(x)
        ax2.set_xticklabels(df_comparison['Model'], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Overfitting score
        ax3 = axes[1, 0]
        colors = ['red' if x > 5 else 'green' for x in df_comparison['Overfit Score']]
        ax3.barh(df_comparison['Model'], df_comparison['Overfit Score'], color=colors)
        ax3.axvline(x=5, color='red', linestyle='--', label='Overfit threshold (5%)')
        ax3.set_xlabel('Overfit Score (%)')
        ax3.set_title('Overfitting Analysis')
        ax3.legend()
        ax3.grid(axis='x', alpha=0.3)
        
        # 4. Scatter plot
        ax4 = axes[1, 1]
        ax4.scatter(df_comparison['Train F1'], df_comparison['Test F1'], s=100, alpha=0.6)
        
        # Add diagonal line (perfect generalization)
        min_val = min(df_comparison['Train F1'].min(), df_comparison['Test F1'].min())
        max_val = max(df_comparison['Train F1'].max(), df_comparison['Test F1'].max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect generalization')
        
        # Add labels
        for idx, row in df_comparison.iterrows():
            ax4.annotate(row['Model'], (row['Train F1'], row['Test F1']), 
                        fontsize=8, alpha=0.7)
        
        ax4.set_xlabel('Train F1 Score')
        ax4.set_ylabel('Test F1 Score')
        ax4.set_title('Generalization Analysis')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return fig_path
    
    def create_feature_importance_plot(self, feature_importance, top_n=20, 
                                      filename='feature_importance', output_dir=None):
        """
        Create feature importance visualization.
        
        Parameters:
        -----------
        feature_importance : dict or pd.DataFrame
            Feature importance data (feature_name -> importance)
        top_n : int
            Number of top features to show
        filename : str
            Filename to save (without extension)
        output_dir : str (optional)
            Override default output directory
            
        Returns:
        --------
        str : Path to saved figure
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        # Convert to DataFrame if dict
        if isinstance(feature_importance, dict):
            df = pd.DataFrame(list(feature_importance.items()), 
                            columns=['Feature', 'Importance'])
        else:
            df = feature_importance.copy()
        
        # Sort and get top N
        df = df.sort_values('Importance', ascending=False).head(top_n)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(df['Feature'], df['Importance'], color='steelblue')
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return fig_path
    
    def create_correlation_heatmap(self, df, features=None, top_n=20,
                                  filename='correlation_heatmap', output_dir=None):
        """
        Create correlation heatmap for features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features
        features : list (optional)
            Specific features to include
        top_n : int
            Number of features to show if features not specified
        filename : str
            Filename to save (without extension)
        output_dir : str (optional)
            Override default output directory
            
        Returns:
        --------
        str : Path to saved figure
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        # Select features
        if features is None:
            # Use all numeric columns, limited to top_n
            features = df.select_dtypes(include=[np.number]).columns[:top_n]
        
        # Calculate correlation
        corr = df[features].corr()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5, ax=ax)
        ax.set_title('Feature Correlation Heatmap')
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return fig_path
    
    def create_learning_curve(self, train_scores, val_scores, 
                             filename='learning_curve', output_dir=None):
        """
        Create learning curve visualization.
        
        Parameters:
        -----------
        train_scores : list or array
            Training scores over epochs/iterations
        val_scores : list or array
            Validation scores over epochs/iterations
        filename : str
            Filename to save (without extension)
        output_dir : str (optional)
            Override default output directory
            
        Returns:
        --------
        str : Path to saved figure
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(train_scores) + 1)
        ax.plot(epochs, train_scores, 'b-', label='Training Score', linewidth=2)
        ax.plot(epochs, val_scores, 'r-', label='Validation Score', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Learning Curve')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return fig_path
    
    def create_roc_curve(self, fpr, tpr, auc_score, model_name='Model',
                        filename='roc_curve', output_dir=None):
        """
        Create ROC curve visualization.
        
        Parameters:
        -----------
        fpr : array
            False positive rates
        tpr : array
            True positive rates
        auc_score : float
            AUC score
        model_name : str
            Name of the model
        filename : str
            Filename to save (without extension)
        output_dir : str (optional)
            Override default output directory
            
        Returns:
        --------
        str : Path to saved figure
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.plot(fpr, tpr, 'b-', linewidth=2, 
               label=f'{model_name} (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return fig_path
