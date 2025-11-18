"""
VisualizationManager - Visualize results with HTML reports.
Creates comprehensive HTML reports with all charts stacked vertically.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class VisualizationManager:
    """
    Manages visualization and HTML report generation.
    
    Features:
    - Generate HTML reports with multiple charts
    - Charts stacked vertically (not subplots)
    - Separate reports for train/test/backtest
    - Interactive plots using Plotly
    """
    
    def __init__(self):
        """Initialize VisualizationManager."""
        self.figures = []
        
    def create_train_report(self,
                           train_results: Dict[str, Any],
                           save_dir: Path) -> str:
        """
        Create HTML report for training results.
        
        Args:
            train_results: Training results dictionary
            save_dir: Directory to save report
            
        Returns:
            Path to saved HTML file
        """
        print("\n" + "="*70)
        print("GENERATING TRAINING VISUALIZATION REPORT")
        print("="*70)
        
        save_dir = Path(save_dir) / 'reports' / 'train'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        figures = []
        
        # Filter successful results
        successful_results = {name: res for name, res in train_results.items() 
                            if res.get('status') == 'success'}
        
        if not successful_results:
            print("No successful training results to visualize")
            return ""
        
        # 1. Training Accuracy Comparison
        fig = self._create_accuracy_comparison(successful_results, 'train')
        figures.append(fig)
        
        # 2. Training Time Comparison
        fig = self._create_training_time_chart(successful_results)
        figures.append(fig)
        
        # 3. Validation Accuracy (if available)
        if any(res.get('val_accuracy') is not None for res in successful_results.values()):
            fig = self._create_accuracy_comparison(successful_results, 'val')
            figures.append(fig)
        
        # Save HTML report
        html_path = save_dir / 'train_report.html'
        self._save_html_report(figures, html_path, "Training Results Report")
        
        print(f"✓ Saved training report: {html_path}")
        print("="*70 + "\n")
        
        return str(html_path)
    
    def create_test_report(self,
                          test_results: Dict[str, Any],
                          save_dir: Path) -> str:
        """
        Create HTML report for test results.
        
        Args:
            test_results: Test results dictionary
            save_dir: Directory to save report
            
        Returns:
            Path to saved HTML file
        """
        print("\n" + "="*70)
        print("GENERATING TEST VISUALIZATION REPORT")
        print("="*70)
        
        save_dir = Path(save_dir) / 'reports' / 'test'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        figures = []
        
        # Filter successful results
        successful_results = {name: res for name, res in test_results.items() 
                            if res.get('status') == 'success'}
        
        if not successful_results:
            print("No successful test results to visualize")
            return ""
        
        # 1. Metrics Comparison (Accuracy, Precision, Recall, F1)
        fig = self._create_metrics_comparison(successful_results)
        figures.append(fig)
        
        # 2. Individual Model Performance
        for model_name, results in successful_results.items():
            fig = self._create_model_performance_chart(model_name, results)
            figures.append(fig)
        
        # 3. Confusion Matrices
        for model_name, results in successful_results.items():
            if 'confusion_matrix' in results:
                fig = self._create_confusion_matrix(model_name, results['confusion_matrix'])
                figures.append(fig)
        
        # Save HTML report
        html_path = save_dir / 'test_report.html'
        self._save_html_report(figures, html_path, "Test Results Report")
        
        print(f"✓ Saved test report: {html_path}")
        print("="*70 + "\n")
        
        return str(html_path)
    
    def create_backtest_report(self,
                              backtest_results: Dict[str, Any],
                              save_dir: Path) -> str:
        """
        Create HTML report for backtest results.
        
        Args:
            backtest_results: Backtest results dictionary
            save_dir: Directory to save report
            
        Returns:
            Path to saved HTML file
        """
        print("\n" + "="*70)
        print("GENERATING BACKTEST VISUALIZATION REPORT")
        print("="*70)
        
        save_dir = Path(save_dir) / 'reports' / 'backtest'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        figures = []
        
        if not backtest_results:
            print("No backtest results to visualize")
            return ""
        
        # 1. Performance Metrics Comparison
        fig = self._create_backtest_metrics_comparison(backtest_results)
        figures.append(fig)
        
        # 2. Individual Equity Curves
        for model_name, results in backtest_results.items():
            if 'equity_curve' in results:
                fig = self._create_equity_curve(model_name, results)
                figures.append(fig)
        
        # 3. Comprehensive Equity Curves (all models together)
        fig = self._create_comprehensive_equity_curves(backtest_results)
        figures.append(fig)
        
        # 4. Returns Distribution
        for model_name, results in backtest_results.items():
            if 'strategy_returns' in results:
                fig = self._create_returns_distribution(model_name, results)
                figures.append(fig)
        
        # Save HTML report
        html_path = save_dir / 'backtest_report.html'
        self._save_html_report(figures, html_path, "Backtest Results Report")
        
        print(f"✓ Saved backtest report: {html_path}")
        print("="*70 + "\n")
        
        return str(html_path)
    
    def _create_accuracy_comparison(self, results: Dict[str, Any], phase: str) -> go.Figure:
        """Create accuracy comparison bar chart."""
        models = list(results.keys())
        metric_key = f'{phase}_accuracy'
        accuracies = [results[m].get(metric_key, 0) for m in models]
        
        fig = go.Figure(data=[
            go.Bar(x=models, y=accuracies, text=[f'{a:.4f}' for a in accuracies],
                  textposition='auto')
        ])
        
        fig.update_layout(
            title=f'{phase.capitalize()} Accuracy Comparison',
            xaxis_title='Model',
            yaxis_title='Accuracy',
            height=400
        )
        
        return fig
    
    def _create_training_time_chart(self, results: Dict[str, Any]) -> go.Figure:
        """Create training time comparison chart."""
        models = list(results.keys())
        times = [results[m].get('training_time', 0) for m in models]
        
        fig = go.Figure(data=[
            go.Bar(x=models, y=times, text=[f'{t:.2f}s' for t in times],
                  textposition='auto')
        ])
        
        fig.update_layout(
            title='Training Time Comparison',
            xaxis_title='Model',
            yaxis_title='Time (seconds)',
            height=400
        )
        
        return fig
    
    def _create_metrics_comparison(self, results: Dict[str, Any]) -> go.Figure:
        """Create comprehensive metrics comparison."""
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig = go.Figure()
        
        for metric in metrics:
            values = [results[m].get(metric, 0) for m in models]
            fig.add_trace(go.Bar(name=metric.replace('_', ' ').title(),
                                x=models, y=values))
        
        fig.update_layout(
            title='Test Metrics Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            height=500
        )
        
        return fig
    
    def _create_model_performance_chart(self, model_name: str, results: Dict[str, Any]) -> go.Figure:
        """Create individual model performance chart."""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        values = [results.get(m, 0) for m in metrics]
        labels = [m.replace('_', ' ').title() for m in metrics]
        
        fig = go.Figure(data=[
            go.Bar(x=labels, y=values, text=[f'{v:.4f}' for v in values],
                  textposition='auto')
        ])
        
        fig.update_layout(
            title=f'{model_name} - Performance Metrics',
            xaxis_title='Metric',
            yaxis_title='Score',
            height=400
        )
        
        return fig
    
    def _create_confusion_matrix(self, model_name: str, cm: np.ndarray) -> go.Figure:
        """Create confusion matrix heatmap."""
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            text=cm,
            texttemplate='%{text}',
            colorscale='Blues'
        ))
        
        fig.update_layout(
            title=f'{model_name} - Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            height=400
        )
        
        return fig
    
    def _create_backtest_metrics_comparison(self, results: Dict[str, Any]) -> go.Figure:
        """Create backtest metrics comparison."""
        models = list(results.keys())
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        
        fig = go.Figure()
        
        for metric in metrics:
            values = [results[m].get(metric, 0) for m in models]
            # Convert to percentage for some metrics
            if metric in ['total_return', 'max_drawdown', 'win_rate']:
                values = [v * 100 for v in values]
            
            fig.add_trace(go.Bar(name=metric.replace('_', ' ').title(),
                                x=models, y=values))
        
        fig.update_layout(
            title='Backtest Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Value',
            barmode='group',
            height=500
        )
        
        return fig
    
    def _create_equity_curve(self, model_name: str, results: Dict[str, Any]) -> go.Figure:
        """Create equity curve for a single model."""
        equity = results.get('equity_curve', np.array([]))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=equity, mode='lines', name='Strategy'))
        
        # Add buy & hold if available
        if 'buy_hold_equity' in results:
            fig.add_trace(go.Scatter(y=results['buy_hold_equity'], 
                                    mode='lines', name='Buy & Hold'))
        
        fig.update_layout(
            title=f'{model_name} - Equity Curve',
            xaxis_title='Time',
            yaxis_title='Equity ($)',
            height=400
        )
        
        return fig
    
    def _create_comprehensive_equity_curves(self, results: Dict[str, Any]) -> go.Figure:
        """Create comprehensive equity curves for all models."""
        fig = go.Figure()
        
        for model_name, res in results.items():
            equity = res.get('equity_curve', np.array([]))
            if len(equity) > 0:
                fig.add_trace(go.Scatter(y=equity, mode='lines', name=model_name))
        
        fig.update_layout(
            title='Comprehensive Equity Curves - All Models',
            xaxis_title='Time',
            yaxis_title='Equity ($)',
            height=500
        )
        
        return fig
    
    def _create_returns_distribution(self, model_name: str, results: Dict[str, Any]) -> go.Figure:
        """Create returns distribution histogram."""
        returns = results.get('strategy_returns', np.array([]))
        
        fig = go.Figure(data=[go.Histogram(x=returns, nbinsx=50)])
        
        fig.update_layout(
            title=f'{model_name} - Returns Distribution',
            xaxis_title='Returns',
            yaxis_title='Frequency',
            height=400
        )
        
        return fig
    
    def _save_html_report(self, figures: List[go.Figure], filepath: Path, title: str):
        """
        Save multiple figures to a single HTML file with charts stacked vertically.
        
        Args:
            figures: List of Plotly figures
            filepath: Path to save HTML file
            title: Report title
        """
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            text-align: center;
            color: #333;
        }}
        .chart-container {{
            background-color: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
"""
        
        # Add each figure as a separate div
        for i, fig in enumerate(figures):
            html_content += f'    <div class="chart-container" id="chart{i}"></div>\n'
        
        # Add JavaScript to render figures
        html_content += "    <script>\n"
        for i, fig in enumerate(figures):
            fig_json = fig.to_json()
            html_content += f"        Plotly.newPlot('chart{i}', {fig_json});\n"
        html_content += "    </script>\n"
        
        html_content += """
</body>
</html>
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
