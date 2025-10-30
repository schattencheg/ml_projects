"""
Report Manager Module

Creates comprehensive reports for training and testing results with visualizations.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
from src.VisualizationManager import VisualizationManager


class ReportManager:
    """
    Creates comprehensive reports for ML model training and testing.
    """
    
    def __init__(self, output_dir='reports'):
        """
        Initialize ReportManager.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create VisualizationManager instance
        self.viz_manager = VisualizationManager(output_dir=output_dir)
    
    def create_training_report(self, train_results, save=True, filename=None):
        """
        Create comprehensive training report.
        
        Parameters:
        -----------
        train_results : dict
            Training results from Trainer
        save : bool
            Whether to save the report
        filename : str (optional)
            Custom filename for report
            
        Returns:
        --------
        dict : Report data
        """
        if filename is None:
            filename = f"training_report_{self.timestamp}"
        
        print(f"\n{'='*70}")
        print(f"TRAINING REPORT")
        print(f"{'='*70}\n")
        
        # Create summary DataFrame
        summary_data = []
        for model_name, data in train_results.items():
            train_metrics = data['train_metrics']
            val_metrics = data.get('val_metrics', {})
            
            row = {
                'Model': model_name,
                'Train Accuracy': train_metrics['accuracy'],
                'Train Precision': train_metrics['precision'],
                'Train Recall': train_metrics['recall'],
                'Train F1': train_metrics['f1'],
                'Train Time (s)': train_metrics.get('training_time', 0)
            }
            
            if val_metrics:
                row['Val Accuracy'] = val_metrics['accuracy']
                row['Val Precision'] = val_metrics['precision']
                row['Val Recall'] = val_metrics['recall']
                row['Val F1'] = val_metrics['f1']
            
            summary_data.append(row)
        
        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.sort_values('Train F1', ascending=False)
        
        # Print summary
        print(df_summary.to_string(index=False))
        print(f"\n{'='*70}\n")
        
        # Create visualizations
        if save:
            fig_path = self.viz_manager.create_training_visualizations(
                df_summary, filename, output_dir=self.output_dir
            )
            print(f"✓ Training visualizations saved to {fig_path}")
            
            # Save summary to CSV
            csv_path = os.path.join(self.output_dir, f"{filename}.csv")
            df_summary.to_csv(csv_path, index=False)
            print(f"✓ Training report saved to {csv_path}")
        
        return {
            'summary': df_summary,
            'timestamp': self.timestamp,
            'num_models': len(train_results)
        }
    
    def create_test_report(self, test_results, y_test, save=True, filename=None, 
                          target_names=None):
        """
        Create comprehensive test report.
        
        Parameters:
        -----------
        test_results : dict
            Test results from Tester
        y_test : array-like
            True labels
        save : bool
            Whether to save the report
        filename : str (optional)
            Custom filename for report
        target_names : list (optional)
            Names of target classes
            
        Returns:
        --------
        dict : Report data
        """
        if filename is None:
            filename = f"test_report_{self.timestamp}"
        
        print(f"\n{'='*70}")
        print(f"TEST REPORT")
        print(f"{'='*70}\n")
        
        # Create summary DataFrame
        summary_data = []
        for model_name, data in test_results.items():
            metrics = data['metrics']
            
            row = {
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1']
            }
            
            summary_data.append(row)
        
        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.sort_values('F1 Score', ascending=False)
        
        # Print summary
        print(df_summary.to_string(index=False))
        print(f"\n{'='*70}\n")
        
        # Find best model
        best_model = df_summary.iloc[0]['Model']
        print(f"Best Model: {best_model}")
        print(f"Best F1 Score: {df_summary.iloc[0]['F1 Score']:.4f}\n")
        
        # Create visualizations
        if save:
            fig_path = self.viz_manager.create_test_visualizations(
                df_summary, test_results, y_test, filename, 
                target_names=target_names, output_dir=self.output_dir
            )
            print(f"✓ Test visualizations saved to {fig_path}")
            
            # Save summary to CSV
            csv_path = os.path.join(self.output_dir, f"{filename}.csv")
            df_summary.to_csv(csv_path, index=False)
            print(f"✓ Test report saved to {csv_path}")
        
        return {
            'summary': df_summary,
            'best_model': best_model,
            'timestamp': self.timestamp,
            'num_models': len(test_results)
        }
    
    def create_comparison_report(self, train_results, test_results, save=True, 
                                 filename=None):
        """
        Create comparison report between training and test results.
        
        Parameters:
        -----------
        train_results : dict
            Training results
        test_results : dict
            Test results
        save : bool
            Whether to save the report
        filename : str (optional)
            Custom filename
            
        Returns:
        --------
        dict : Comparison data
        """
        if filename is None:
            filename = f"comparison_report_{self.timestamp}"
        
        print(f"\n{'='*70}")
        print(f"TRAIN vs TEST COMPARISON REPORT")
        print(f"{'='*70}\n")
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name in train_results.keys():
            if model_name not in test_results:
                continue
            
            train_metrics = train_results[model_name]['train_metrics']
            test_metrics = test_results[model_name]['metrics']
            
            row = {
                'Model': model_name,
                'Train Acc': train_metrics['accuracy'],
                'Test Acc': test_metrics['accuracy'],
                'Acc Diff': train_metrics['accuracy'] - test_metrics['accuracy'],
                'Train F1': train_metrics['f1'],
                'Test F1': test_metrics['f1'],
                'F1 Diff': train_metrics['f1'] - test_metrics['f1'],
                'Overfit Score': (train_metrics['accuracy'] - test_metrics['accuracy']) * 100
            }
            
            comparison_data.append(row)
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('Test F1', ascending=False)
        
        # Print comparison
        print(df_comparison.to_string(index=False))
        print(f"\n{'='*70}\n")
        
        # Identify overfitting
        overfitting_models = df_comparison[df_comparison['Overfit Score'] > 5]
        if len(overfitting_models) > 0:
            print("⚠️  Models with potential overfitting (>5% accuracy drop):")
            for _, row in overfitting_models.iterrows():
                print(f"  • {row['Model']}: {row['Overfit Score']:.1f}% drop")
            print()
        
        # Create visualizations
        if save:
            fig_path = self.viz_manager.create_comparison_visualizations(
                df_comparison, filename, output_dir=self.output_dir
            )
            print(f"✓ Comparison visualizations saved to {fig_path}")
            
            # Save to CSV
            csv_path = os.path.join(self.output_dir, f"{filename}.csv")
            df_comparison.to_csv(csv_path, index=False)
            print(f"✓ Comparison report saved to {csv_path}")
        
        return {
            'comparison': df_comparison,
            'overfitting_models': overfitting_models,
            'timestamp': self.timestamp
        }
    
    def export_full_report(self, train_results, test_results, y_test, 
                          filename=None, target_names=None):
        """
        Export complete report with all sections.
        
        Parameters:
        -----------
        train_results : dict
            Training results
        test_results : dict
            Test results
        y_test : array-like
            True test labels
        filename : str (optional)
            Base filename for reports
        target_names : list (optional)
            Names of target classes
            
        Returns:
        --------
        dict : All report data
        """
        if filename is None:
            filename = f"full_report_{self.timestamp}"
        
        print(f"\n{'='*70}")
        print(f"EXPORTING FULL REPORT")
        print(f"{'='*70}\n")
        
        # Create all reports
        train_report = self.create_training_report(
            train_results, save=True, filename=f"{filename}_training"
        )
        
        test_report = self.create_test_report(
            test_results, y_test, save=True, filename=f"{filename}_test",
            target_names=target_names
        )
        
        comparison_report = self.create_comparison_report(
            train_results, test_results, save=True, filename=f"{filename}_comparison"
        )
        
        print(f"\n{'='*70}")
        print(f"FULL REPORT EXPORTED")
        print(f"{'='*70}")
        print(f"Location: {self.output_dir}/")
        print(f"Files:")
        print(f"  • {filename}_training.csv")
        print(f"  • {filename}_training.png")
        print(f"  • {filename}_test.csv")
        print(f"  • {filename}_test.png")
        print(f"  • {filename}_comparison.csv")
        print(f"  • {filename}_comparison.png")
        print(f"{'='*70}\n")
        
        return {
            'train_report': train_report,
            'test_report': test_report,
            'comparison_report': comparison_report,
            'timestamp': self.timestamp
        }
