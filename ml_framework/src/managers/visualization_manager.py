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
    - Jupyter notebook support with inline display
    
    Quick Start:
        >>> viz = VisualizationManager(verbose=False)
        >>> 
        >>> # Create and save reports
        >>> viz.create_backtest_report(backtest_results, save_dir='results')
        >>> viz.create_feature_importance_report(feature_importance, save_dir='results')
        >>> 
        >>> # Display inline (Jupyter)
        >>> viz.show_backtest_results(backtest_results, df=ohlc_data)
        >>> viz.show_feature_importance(feature_importance)
        >>> 
        >>> # Get figures for custom handling
        >>> figs = viz.get_backtest_figures(backtest_results)
    """
    
    def __init__(self, jupyter_mode: Optional[bool] = None, verbose: bool = True):
        """
        Initialize VisualizationManager.
        
        Args:
            jupyter_mode: If True, optimize for Jupyter display. 
                         If None, auto-detect environment.
            verbose: If True, print status messages (default: True)
        """
        self.figures = []
        self._jupyter_mode = jupyter_mode if jupyter_mode is not None else self._is_jupyter()
        self.verbose = verbose
    
    @staticmethod
    def _is_jupyter() -> bool:
        """Detect if running in Jupyter notebook/lab environment."""
        try:
            from IPython import get_ipython
            shell = get_ipython()
            if shell is None:
                return False
            shell_name = shell.__class__.__name__
            if 'ZMQInteractiveShell' in shell_name:  # Jupyter notebook/lab
                return True
            if 'TerminalInteractiveShell' in shell_name:  # IPython terminal
                return False
            return False
        except (ImportError, NameError):
            return False
    
    @property
    def jupyter_mode(self) -> bool:
        """Check if running in Jupyter mode."""
        return self._jupyter_mode
    
    def display_figure(self, fig: go.Figure) -> None:
        """
        Display a figure inline (Jupyter) or show in browser.
        
        Args:
            fig: Plotly figure to display
        """
        if self._jupyter_mode:
            fig.show()
        else:
            fig.show()
    
    def display_figures(self, figures: List[go.Figure]) -> None:
        """
        Display multiple figures inline (Jupyter) or show in browser.
        
        Args:
            figures: List of Plotly figures to display
        """
        for fig in figures:
            self.display_figure(fig)
    
    # =========================================================================
    # JUPYTER-FRIENDLY METHODS (return figures for inline display)
    # =========================================================================
    
    def get_feature_importance_figures(self,
                                        feature_importance: pd.Series,
                                        selected_features: Optional[List[str]] = None,
                                        dropped_features: Optional[List[str]] = None,
                                        method: str = 'unknown',
                                        correlation_matrix: Optional[pd.DataFrame] = None) -> List[go.Figure]:
        """
        Get feature importance figures for Jupyter display.
        
        Args:
            feature_importance: Series with feature names as index and importance as values
            selected_features: List of selected feature names
            dropped_features: List of dropped feature names
            method: Feature selection method used
            correlation_matrix: Optional correlation matrix for heatmap
            
        Returns:
            List of Plotly figures for inline display
        """
        figures = []
        
        # 1. Feature Importance Bar Chart (Top 30)
        figures.append(self._create_feature_importance_bar(feature_importance, top_n=30))
        
        # 2. Feature Importance Horizontal Bar (All features)
        figures.append(self._create_feature_importance_horizontal(feature_importance))
        
        # 3. Cumulative Importance Chart
        figures.append(self._create_cumulative_importance(feature_importance))
        
        # 4. Selected vs Dropped Features Summary
        if selected_features is not None and dropped_features is not None:
            figures.append(self._create_feature_selection_summary(
                feature_importance, selected_features, dropped_features, method
            ))
        
        # 5. Correlation Heatmap (if provided)
        if correlation_matrix is not None:
            figures.append(self._create_correlation_heatmap(correlation_matrix))
        
        # 6. Feature Importance Distribution
        figures.append(self._create_importance_distribution(feature_importance))
        
        return figures
    
    def get_backtest_figures(self,
                             backtest_results: Dict[str, Any],
                             df: Optional[pd.DataFrame] = None) -> List[go.Figure]:
        """
        Get backtest figures for Jupyter display.
        
        Args:
            backtest_results: Backtest results dictionary
            df: Optional DataFrame with OHLC data for trade visualization
            
        Returns:
            List of Plotly figures for inline display
        """
        figures = []
        
        if not backtest_results:
            return figures
        
        # 1. Backtest Metrics Comparison
        figures.append(self._create_backtest_metrics_comparison(backtest_results))
        
        # 2. Equity Curves Comparison
        figures.append(self.create_equity_curves_comparison(backtest_results))
        
        # 3. Individual Model Results
        for model_name, results in backtest_results.items():
            # Individual equity curve
            if 'equity_curve' in results:
                figures.append(self._create_equity_curve(model_name, results))
            
            # OHLC chart with trades
            if df is not None and 'trades' in results and len(results['trades']) > 0:
                figures.append(self.create_ohlc_with_trades(
                    df=df,
                    trades=results['trades'],
                    model_name=model_name
                ))
            
            # Returns distribution
            if 'strategy_returns' in results:
                figures.append(self._create_returns_distribution(model_name, results))
        
        return figures
    
    def get_model_comparison_figures(self, test_results: Dict[str, Any]) -> List[go.Figure]:
        """
        Get model comparison figures for Jupyter display.
        
        Args:
            test_results: Test results dictionary with metrics per model
            
        Returns:
            List of Plotly figures for inline display
        """
        figures = []
        
        if not test_results:
            return figures
        
        # 1. Accuracy Comparison
        figures.append(self._create_accuracy_comparison(test_results, 'test'))
        
        # 2. Metrics Comparison
        figures.append(self._create_metrics_comparison(test_results))
        
        # 3. Individual confusion matrices
        for model_name, results in test_results.items():
            if 'confusion_matrix' in results:
                figures.append(self._create_confusion_matrix(model_name, results['confusion_matrix']))
        
        return figures
    
    def show_feature_importance(self,
                                 feature_importance: pd.Series,
                                 selected_features: Optional[List[str]] = None,
                                 dropped_features: Optional[List[str]] = None,
                                 method: str = 'unknown',
                                 correlation_matrix: Optional[pd.DataFrame] = None) -> None:
        """
        Display feature importance figures inline (Jupyter-friendly).
        
        Args:
            feature_importance: Series with feature names as index and importance as values
            selected_features: List of selected feature names
            dropped_features: List of dropped feature names
            method: Feature selection method used
            correlation_matrix: Optional correlation matrix for heatmap
        """
        figures = self.get_feature_importance_figures(
            feature_importance, selected_features, dropped_features, method, correlation_matrix
        )
        self.display_figures(figures)
    
    def show_backtest_results(self,
                               backtest_results: Dict[str, Any],
                               df: Optional[pd.DataFrame] = None) -> None:
        """
        Display backtest results inline (Jupyter-friendly).
        
        Args:
            backtest_results: Backtest results dictionary
            df: Optional DataFrame with OHLC data for trade visualization
        """
        figures = self.get_backtest_figures(backtest_results, df)
        self.display_figures(figures)
    
    def show_model_comparison(self, test_results: Dict[str, Any]) -> None:
        """
        Display model comparison inline (Jupyter-friendly).
        
        Args:
            test_results: Test results dictionary with metrics per model
        """
        figures = self.get_model_comparison_figures(test_results)
        self.display_figures(figures)
        
    def create_train_report(self,
                           train_results: Dict[str, Any],
                           save_dir: Path,
                           show: bool = True) -> str:
        """
        Create HTML report for training results.
        
        Args:
            train_results: Training results dictionary
            save_dir: Directory to save report
            show: If True, automatically open the report in browser (default: True)
            
        Returns:
            Path to saved HTML file
        """
        if self.verbose:
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
            if self.verbose:
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
        self._save_html_report(figures, html_path, "Training Results Report", show=show)
        
        if self.verbose:
            print(f"✓ Saved training report: {html_path}")
            if show:
                print(f"🎉 Opening report in browser...")
            print("="*70 + "\n")
        
        return str(html_path)
    
    def create_test_report(self,
                          test_results: Dict[str, Any],
                          save_dir: Path,
                          show: bool = True) -> str:
        """
        Create HTML report for test results.
        
        Args:
            test_results: Test results dictionary
            save_dir: Directory to save report
            show: If True, automatically open the report in browser (default: True)
            
        Returns:
            Path to saved HTML file
        """
        if self.verbose:
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
            if self.verbose:
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
        self._save_html_report(figures, html_path, "Test Results Report", show=show)
        
        if self.verbose:
            print(f"✓ Saved test report: {html_path}")
            if show:
                print(f"🎉 Opening report in browser...")
            print("="*70 + "\n")
        
        return str(html_path)
    
    def create_feature_importance_report(self,
                                          feature_importance: pd.Series,
                                          save_dir: Path,
                                          selected_features: Optional[List[str]] = None,
                                          dropped_features: Optional[List[str]] = None,
                                          method: str = 'unknown',
                                          correlation_matrix: Optional[pd.DataFrame] = None,
                                          show: bool = True) -> str:
        """
        Create HTML report for feature importance analysis.
        
        Args:
            feature_importance: Series with feature names as index and importance as values
            save_dir: Directory to save report
            selected_features: List of selected feature names
            dropped_features: List of dropped feature names
            method: Feature selection method used
            correlation_matrix: Optional correlation matrix for heatmap
            show: If True, automatically open the report in browser
            
        Returns:
            Path to saved HTML file
        """
        if self.verbose:
            print("\n" + "="*70)
            print("GENERATING FEATURE IMPORTANCE REPORT")
            print("="*70)
        
        save_dir = Path(save_dir) / 'reports' / 'features'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        figures = []
        
        # 1. Feature Importance Bar Chart (Top 30)
        fig = self._create_feature_importance_bar(feature_importance, top_n=30)
        figures.append(fig)
        
        # 2. Feature Importance Horizontal Bar (All features)
        fig = self._create_feature_importance_horizontal(feature_importance)
        figures.append(fig)
        
        # 3. Cumulative Importance Chart
        fig = self._create_cumulative_importance(feature_importance)
        figures.append(fig)
        
        # 4. Selected vs Dropped Features Summary
        if selected_features is not None and dropped_features is not None:
            fig = self._create_feature_selection_summary(
                feature_importance, selected_features, dropped_features, method
            )
            figures.append(fig)
        
        # 5. Correlation Heatmap (if provided)
        if correlation_matrix is not None:
            fig = self._create_correlation_heatmap(correlation_matrix)
            figures.append(fig)
        
        # 6. Feature Importance Distribution
        fig = self._create_importance_distribution(feature_importance)
        figures.append(fig)
        
        # Save HTML report
        html_path = save_dir / 'feature_importance_report.html'
        self._save_html_report(figures, html_path, "Feature Importance Analysis Report", show=show)
        
        if self.verbose:
            print(f"✓ Saved feature importance report: {html_path}")
            if show:
                print(f"🎉 Opening report in browser...")
            print("="*70 + "\n")
        
        return str(html_path)
    
    def _create_feature_importance_bar(self, importance: pd.Series, top_n: int = 30) -> go.Figure:
        """Create vertical bar chart of top N feature importances."""
        top_features = importance.head(top_n)
        
        fig = go.Figure(data=[
            go.Bar(
                x=top_features.index.tolist(),
                y=top_features.values.tolist(),
                text=[f'{v:.4f}' for v in top_features.values],
                textposition='auto',
                marker_color='steelblue'
            )
        ])
        
        fig.update_layout(
            title=f'<b>Top {top_n} Feature Importance</b>',
            xaxis_title='Feature',
            yaxis_title='Importance',
            height=500,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def _create_feature_importance_horizontal(self, importance: pd.Series) -> go.Figure:
        """Create horizontal bar chart of all feature importances."""
        # Sort ascending for horizontal bar (top features at top)
        sorted_importance = importance.sort_values(ascending=True)
        
        # Limit to reasonable number for display
        if len(sorted_importance) > 50:
            sorted_importance = sorted_importance.tail(50)
        
        fig = go.Figure(data=[
            go.Bar(
                x=sorted_importance.values.tolist(),
                y=sorted_importance.index.tolist(),
                orientation='h',
                text=[f'{v:.4f}' for v in sorted_importance.values],
                textposition='auto',
                marker_color='steelblue'
            )
        ])
        
        fig.update_layout(
            title='<b>Feature Importance (All Features)</b>',
            xaxis_title='Importance',
            yaxis_title='Feature',
            height=max(400, len(sorted_importance) * 20)
        )
        
        return fig
    
    def _create_cumulative_importance(self, importance: pd.Series) -> go.Figure:
        """Create cumulative importance chart."""
        sorted_importance = importance.sort_values(ascending=False)
        cumulative = sorted_importance.cumsum() / sorted_importance.sum() * 100
        
        fig = go.Figure()
        
        # Cumulative line
        fig.add_trace(go.Scatter(
            x=list(range(1, len(cumulative) + 1)),
            y=cumulative.values.tolist(),
            mode='lines+markers',
            name='Cumulative Importance',
            line=dict(color='steelblue', width=2),
            marker=dict(size=4)
        ))
        
        # 80% threshold line
        fig.add_hline(y=80, line_dash="dash", line_color="red",
                      annotation_text="80% threshold")
        
        # 95% threshold line
        fig.add_hline(y=95, line_dash="dash", line_color="orange",
                      annotation_text="95% threshold")
        
        # Find number of features for 80% and 95%
        n_80 = (cumulative <= 80).sum() + 1
        n_95 = (cumulative <= 95).sum() + 1
        
        fig.update_layout(
            title=f'<b>Cumulative Feature Importance</b><br>'
                  f'<sub>{n_80} features for 80%, {n_95} features for 95% of importance</sub>',
            xaxis_title='Number of Features',
            yaxis_title='Cumulative Importance (%)',
            height=450
        )
        
        return fig
    
    def _create_feature_selection_summary(self, 
                                           importance: pd.Series,
                                           selected: List[str],
                                           dropped: List[str],
                                           method: str) -> go.Figure:
        """Create summary chart of selected vs dropped features."""
        # Create DataFrame for visualization
        all_features = importance.index.tolist()
        colors = ['green' if f in selected else 'red' for f in all_features]
        
        sorted_importance = importance.sort_values(ascending=False)
        sorted_colors = ['green' if f in selected else 'red' for f in sorted_importance.index]
        
        fig = go.Figure(data=[
            go.Bar(
                x=sorted_importance.index.tolist()[:30],
                y=sorted_importance.values.tolist()[:30],
                marker_color=sorted_colors[:30],
                text=['Selected' if c == 'green' else 'Dropped' for c in sorted_colors[:30]],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=f'<b>Feature Selection Summary ({method.upper()})</b><br>'
                  f'<sub>Selected: {len(selected)} | Dropped: {len(dropped)}</sub>',
            xaxis_title='Feature',
            yaxis_title='Importance',
            height=500,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def _create_correlation_heatmap(self, corr_matrix: pd.DataFrame) -> go.Figure:
        """Create correlation matrix heatmap."""
        # Limit size for readability
        if len(corr_matrix) > 30:
            # Take top 30 features by variance
            variances = corr_matrix.var()
            top_features = variances.nlargest(30).index.tolist()
            corr_matrix = corr_matrix.loc[top_features, top_features]
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values.tolist(),
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale='RdBu_r',
            zmid=0,
            text=[[f'{v:.2f}' for v in row] for row in corr_matrix.values],
            texttemplate='%{text}',
            textfont={"size": 8}
        ))
        
        fig.update_layout(
            title='<b>Feature Correlation Matrix</b>',
            height=700,
            width=900,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def _create_importance_distribution(self, importance: pd.Series) -> go.Figure:
        """Create distribution histogram of feature importances."""
        fig = go.Figure(data=[
            go.Histogram(
                x=importance.values.tolist(),
                nbinsx=30,
                marker_color='steelblue'
            )
        ])
        
        # Add mean and median lines
        mean_val = importance.mean()
        median_val = importance.median()
        
        fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {mean_val:.4f}")
        fig.add_vline(x=median_val, line_dash="dash", line_color="green",
                      annotation_text=f"Median: {median_val:.4f}")
        
        fig.update_layout(
            title='<b>Feature Importance Distribution</b>',
            xaxis_title='Importance',
            yaxis_title='Count',
            height=400
        )
        
        return fig
    
    def create_backtest_report(self,
                              backtest_results: Dict[str, Any],
                              save_dir: Path,
                              df: Optional[pd.DataFrame] = None,
                              test_results: Optional[Dict[str, Any]] = None,
                              train_results: Optional[Dict[str, Any]] = None,
                              feature_info: Optional[Dict[str, Any]] = None,
                              show: bool = True) -> str:
        """
        Create HTML report for backtest results.
        
        Args:
            backtest_results: Backtest results dictionary
            save_dir: Directory to save report
            df: Optional DataFrame with OHLC data for trade visualization
            test_results: Optional test results dictionary with model metrics
            train_results: Optional train results dictionary with model metrics
            feature_info: Optional feature engineering information
            show: If True, automatically open the report in browser (default: True)
            
        Returns:
            Path to saved HTML file
        """
        if self.verbose:
            print("\n" + "="*70)
            print("GENERATING BACKTEST VISUALIZATION REPORT")
            print("="*70)
        
        save_dir = Path(save_dir) / 'reports' / 'backtest'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        figures = []
        
        if not backtest_results:
            if self.verbose:
                print("No backtest results to visualize")
            return ""
        
        # 1. Train/Test/Val Results Table (if available)
        if train_results or test_results:
            fig = self._create_train_test_val_table(train_results, test_results)
            figures.append(fig)
        
        # 2. Feature Engineering Results Table (if available)
        if feature_info:
            fig = self._create_feature_engineering_table(feature_info)
            figures.append(fig)
        
        # 3. Backtest Performance Summary Table
        fig = self._create_backtest_summary_table(backtest_results, test_results)
        figures.append(fig)
        
        # 4. Equity Curves Comparison (All curves on one chart)
        fig = self.create_equity_curves_comparison(backtest_results)
        figures.append(fig)
        
        # 5. Individual Model Results
        for model_name, results in backtest_results.items():
            # OHLC chart with trades
            if df is not None and 'trades' in results and len(results['trades']) > 0:
                fig = self.create_ohlc_with_trades(
                    df=df,
                    trades=results['trades'],
                    model_name=model_name
                )
                figures.append(fig)
            
            # Returns distribution
            if 'strategy_returns' in results:
                fig = self._create_returns_distribution(model_name, results)
                figures.append(fig)
        
        # Save HTML report with custom summary
        html_path = save_dir / 'backtest_report.html'
        summary_html = self._generate_summary_html(backtest_results, test_results, train_results)
        self._save_html_report_with_summary(figures, html_path, "Backtest Results Report", summary_html, show=show)
        
        if self.verbose:
            print(f"✓ Saved backtest report: {html_path}")
            if show:
                print(f"🎉 Opening report in browser...")
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
    
    def _create_backtest_summary_table(self, 
                                       backtest_results: Dict[str, Any],
                                       test_results: Optional[Dict[str, Any]] = None) -> go.Figure:
        """
        Create comprehensive summary table with test metrics and backtest metrics.
        
        Args:
            backtest_results: Backtest results dictionary
            test_results: Optional test results dictionary with model metrics
            
        Returns:
            Plotly figure with table
        """
        models = list(backtest_results.keys())
        
        # Prepare table data
        headers = ['Model']
        
        # Add test metrics columns if available
        if test_results:
            headers.extend(['Accuracy', 'Precision', 'Recall', 'F1 Score'])
        
        # Add backtest metrics columns
        headers.extend(['Trades', 'Final PnL ($)', 'Total Return (%)', 
                       'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)'])
        
        # Build table rows
        table_data = []
        
        for model_name in models:
            row = [model_name]
            
            # Add test metrics if available
            if test_results and model_name in test_results:
                test_res = test_results[model_name]
                if test_res.get('status') == 'success':
                    row.append(f"{test_res.get('accuracy', 0):.4f}")
                    row.append(f"{test_res.get('precision', 0):.4f}")
                    row.append(f"{test_res.get('recall', 0):.4f}")
                    row.append(f"{test_res.get('f1_score', 0):.4f}")
                else:
                    row.extend(['N/A', 'N/A', 'N/A', 'N/A'])
            elif test_results:
                row.extend(['N/A', 'N/A', 'N/A', 'N/A'])
            
            # Add backtest metrics
            bt_res = backtest_results[model_name]
            
            # Get metrics (could be in 'metrics' dict or at top level)
            metrics = bt_res.get('metrics', bt_res)
            
            # Trades count
            trades_count = len(bt_res.get('trades', []))
            row.append(str(trades_count))
            
            # Final PnL
            initial_capital = bt_res.get('initial_capital', 10000)
            final_capital = metrics.get('final_capital', initial_capital)
            final_pnl = final_capital - initial_capital
            row.append(f"{final_pnl:,.2f}")
            
            # Total Return (%)
            total_return = metrics.get('total_return', 0) * 100
            row.append(f"{total_return:.2f}")
            
            # Sharpe Ratio
            sharpe = metrics.get('sharpe_ratio', 0)
            row.append(f"{sharpe:.3f}")
            
            # Max Drawdown (%)
            max_dd = metrics.get('max_drawdown', 0) * 100
            row.append(f"{max_dd:.2f}")
            
            # Win Rate (%)
            win_rate = metrics.get('win_rate', 0) * 100
            row.append(f"{win_rate:.2f}")
            
            table_data.append(row)
        
        # Transpose data for Plotly table (columns as lists)
        table_values = [[row[i] for row in table_data] for i in range(len(headers))]
        
        # Create color scheme for cells
        # Header colors
        header_colors = ['#2c3e50']  # Model column
        if test_results:
            header_colors.extend(['#3498db'] * 4)  # Test metrics in blue
        header_colors.extend(['#27ae60'] * 6)  # Backtest metrics in green
        
        # Cell colors (alternating rows)
        cell_colors = []
        for col_idx in range(len(headers)):
            if col_idx == 0:  # Model names
                colors = ['#ecf0f1' if i % 2 == 0 else '#ffffff' for i in range(len(models))]
            elif test_results and 1 <= col_idx <= 4:  # Test metrics
                colors = ['#ebf5fb' if i % 2 == 0 else '#ffffff' for i in range(len(models))]
            else:  # Backtest metrics
                colors = ['#eafaf1' if i % 2 == 0 else '#ffffff' for i in range(len(models))]
            cell_colors.append(colors)
        
        # Create table figure
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=[f'<b>{h}</b>' for h in headers],
                fill_color=header_colors,
                align='center',
                font=dict(color='white', size=12, family='Arial'),
                height=35
            ),
            cells=dict(
                values=table_values,
                fill_color=cell_colors,
                align=['left'] + ['center'] * (len(headers) - 1),
                font=dict(color='#2c3e50', size=11, family='Arial'),
                height=30
            )
        )])
        
        fig.update_layout(
            title='<b>Models Performance Summary</b>',
            height=max(200, 100 + len(models) * 35),
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        return fig
    
    def _create_train_test_val_table(self,
                                     train_results: Optional[Dict[str, Any]] = None,
                                     test_results: Optional[Dict[str, Any]] = None) -> go.Figure:
        """
        Create Train/Test/Val results table showing metrics across all datasets.
        
        Args:
            train_results: Training results dictionary
            test_results: Test results dictionary
            
        Returns:
            Plotly figure with table
        """
        # Get all model names
        models = set()
        if train_results:
            models.update(train_results.keys())
        if test_results:
            models.update(test_results.keys())
        models = sorted(list(models))
        
        # Prepare table data
        headers = ['Model', 'Train Accuracy', 'Val Accuracy', 'Test Accuracy', 
                   'Test Precision', 'Test Recall', 'Test F1 Score', 'Training Time (s)']
        
        # Build table rows
        table_data = []
        
        for model_name in models:
            row = [model_name]
            
            # Train results
            if train_results and model_name in train_results:
                train_res = train_results[model_name]
                if train_res.get('status') == 'success':
                    row.append(f"{train_res.get('train_accuracy', 0):.4f}")
                    val_acc = train_res.get('val_accuracy')
                    row.append(f"{val_acc:.4f}" if val_acc is not None else 'N/A')
                else:
                    row.extend(['Failed', 'Failed'])
            else:
                row.extend(['N/A', 'N/A'])
            
            # Test results
            if test_results and model_name in test_results:
                test_res = test_results[model_name]
                if test_res.get('status') == 'success':
                    row.append(f"{test_res.get('accuracy', 0):.4f}")
                    row.append(f"{test_res.get('precision', 0):.4f}")
                    row.append(f"{test_res.get('recall', 0):.4f}")
                    row.append(f"{test_res.get('f1_score', 0):.4f}")
                else:
                    row.extend(['Failed', 'Failed', 'Failed', 'Failed'])
            else:
                row.extend(['N/A', 'N/A', 'N/A', 'N/A'])
            
            # Training time
            if train_results and model_name in train_results:
                train_res = train_results[model_name]
                train_time = train_res.get('training_time', 0)
                row.append(f"{train_time:.2f}")
            else:
                row.append('N/A')
            
            table_data.append(row)
        
        # Transpose data for Plotly table
        table_values = [[row[i] for row in table_data] for i in range(len(headers))]
        
        # Create color scheme
        header_colors = ['#2c3e50', '#e67e22', '#e67e22', '#3498db', 
                        '#3498db', '#3498db', '#3498db', '#95a5a6']
        
        # Cell colors (alternating rows)
        cell_colors = []
        for col_idx in range(len(headers)):
            if col_idx == 0:  # Model names
                colors = ['#ecf0f1' if i % 2 == 0 else '#ffffff' for i in range(len(models))]
            elif col_idx in [1, 2]:  # Train/Val
                colors = ['#fef5e7' if i % 2 == 0 else '#ffffff' for i in range(len(models))]
            elif col_idx in [3, 4, 5, 6]:  # Test metrics
                colors = ['#ebf5fb' if i % 2 == 0 else '#ffffff' for i in range(len(models))]
            else:  # Training time
                colors = ['#f4f6f7' if i % 2 == 0 else '#ffffff' for i in range(len(models))]
            cell_colors.append(colors)
        
        # Create table figure
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=[f'<b>{h}</b>' for h in headers],
                fill_color=header_colors,
                align='center',
                font=dict(color='white', size=12, family='Arial'),
                height=35
            ),
            cells=dict(
                values=table_values,
                fill_color=cell_colors,
                align=['left'] + ['center'] * (len(headers) - 1),
                font=dict(color='#2c3e50', size=11, family='Arial'),
                height=30
            )
        )])
        
        fig.update_layout(
            title='<b>Train/Test/Validation Results</b>',
            height=max(200, 100 + len(models) * 35),
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        return fig
    
    def _create_feature_engineering_table(self, feature_info: Dict[str, Any]) -> go.Figure:
        """
        Create feature engineering results table.
        
        Args:
            feature_info: Dictionary with feature engineering information
            
        Returns:
            Plotly figure with table
        """
        # Extract information
        total_features = feature_info.get('total_features', 0)
        selected_features = feature_info.get('selected_features', 0)
        dropped_features = feature_info.get('dropped_features', 0)
        selection_method = feature_info.get('selection_method', 'Unknown')
        feature_names = feature_info.get('feature_names', [])
        
        # Prepare summary data
        headers = ['Metric', 'Value']
        
        rows = [
            ['Total Features Generated', str(total_features)],
            ['Features Selected', str(selected_features)],
            ['Features Dropped', str(dropped_features)],
            ['Selection Method', selection_method],
            ['Reduction Rate', f"{(dropped_features/total_features*100):.1f}%" if total_features > 0 else 'N/A']
        ]
        
        # Add top features if available
        if 'top_features' in feature_info and feature_info['top_features']:
            top_features = feature_info['top_features'][:5]  # Top 5
            rows.append(['Top 5 Features', ', '.join(top_features)])
        
        # Transpose for Plotly
        table_values = [[row[i] for row in rows] for i in range(2)]
        
        # Create table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=[f'<b>{h}</b>' for h in headers],
                fill_color=['#2c3e50', '#2c3e50'],
                align='left',
                font=dict(color='white', size=12, family='Arial'),
                height=35
            ),
            cells=dict(
                values=table_values,
                fill_color=[['#ecf0f1' if i % 2 == 0 else '#ffffff' for i in range(len(rows))],
                           ['#e8f8f5' if i % 2 == 0 else '#ffffff' for i in range(len(rows))]],
                align=['left', 'left'],
                font=dict(color='#2c3e50', size=11, family='Arial'),
                height=30
            )
        )])
        
        fig.update_layout(
            title='<b>Feature Engineering Summary</b>',
            height=max(250, 100 + len(rows) * 35),
            margin=dict(l=20, r=20, t=60, b=20)
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
    
    def create_equity_curves_comparison(self, backtest_results: Dict[str, Any]) -> go.Figure:
        """
        Create comparison plot of equity curves for all backtests.
        
        Args:
            backtest_results: Dictionary with backtest results for each model/backend
            
        Returns:
            Plotly figure with all equity curves
        """
        fig = go.Figure()
        
        # Add equity curve for each model/backend
        for name, results in backtest_results.items():
            if 'equity_curve' in results and len(results['equity_curve']) > 0:
                equity = results['equity_curve']
                
                # Create hover text with metrics
                metrics = results.get('metrics', {})
                hover_text = (
                    f"<b>{name}</b><br>"
                    f"Final Capital: ${metrics.get('final_capital', 0):,.2f}<br>"
                    f"Total Return: {metrics.get('total_return', 0)*100:.2f}%<br>"
                    f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}<br>"
                    f"Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%"
                )
                
                fig.add_trace(go.Scatter(
                    y=equity,
                    mode='lines',
                    name=name,
                    hovertemplate=hover_text + '<extra></extra>',
                    line=dict(width=2)
                ))
        
        # Add initial capital reference line
        if backtest_results:
            first_result = next(iter(backtest_results.values()))
            initial_capital = first_result.get('initial_capital', 10000)
            max_len = max(len(r.get('equity_curve', [])) for r in backtest_results.values())
            
            fig.add_trace(go.Scatter(
                y=[initial_capital] * max_len,
                mode='lines',
                name='Initial Capital',
                line=dict(color='gray', dash='dash', width=1),
                hovertemplate=f'Initial Capital: ${initial_capital:,.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title='<b>Equity Curves Comparison - All Backtests</b>',
            xaxis_title='Time (bars)',
            yaxis_title='Equity ($)',
            height=600,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)"
            ),
            template='plotly_white'
        )
        
        return fig
    
    def create_ohlc_with_trades(self, 
                                 df: pd.DataFrame,
                                 trades: List[Dict[str, Any]],
                                 model_name: str,
                                 price_col: str = 'close') -> go.Figure:
        """
        Create OHLC candlestick chart with trade entry/exit markers.
        
        Args:
            df: DataFrame with OHLC data (must have 'open', 'high', 'low', 'close' columns)
            trades: List of trade dictionaries with 'entry_idx', 'exit_idx', 'entry_price', 'exit_price', 'pnl'
            model_name: Name of the model/backend
            price_col: Name of the price column (default: 'close')
            
        Returns:
            Plotly figure with OHLC chart and trade markers
        """
        fig = go.Figure()
        
        # Check if OHLC data is available (case-insensitive)
        df_cols_lower = [col.lower() for col in df.columns]
        has_ohlc = all(col in df_cols_lower for col in ['open', 'high', 'low', 'close'])
        
        # Create column mapping (handle both lowercase and uppercase)
        col_map = {}
        for col in df.columns:
            col_map[col.lower()] = col
        
        if has_ohlc:
            # Convert to Python lists to avoid Plotly binary encoding issues
            # which cause candlesticks to not render in saved HTML
            x_values = [str(x) for x in df.index]
            open_values = df[col_map['open']].tolist()
            high_values = df[col_map['high']].tolist()
            low_values = df[col_map['low']].tolist()
            close_values = df[col_map['close']].tolist()
            
            # Add candlestick chart (use mapped column names)
            fig.add_trace(go.Candlestick(
                x=x_values,
                open=open_values,
                high=high_values,
                low=low_values,
                close=close_values,
                name='OHLC',
                increasing_line_color='green',
                decreasing_line_color='red',
                showlegend=True
            ))
            
            # Also add close price line for clarity
            fig.add_trace(go.Scatter(
                x=x_values,
                y=close_values,
                mode='lines',
                name='Close',
                line=dict(color='blue', width=1),
                opacity=0.7,
                showlegend=True
            ))
        else:
            # Fallback to line chart if OHLC not available
            # Try to find close column with case-insensitive search
            close_col = None
            for col in df.columns:
                if col.lower() == 'close':
                    close_col = col
                    break
            
            if close_col is None:
                close_col = price_col
            
            # Convert to Python lists to avoid binary encoding
            x_values = [str(x) for x in df.index]
            y_values = df[close_col].tolist()
            
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            ))
        
        # Add trade markers
        # Normalize trades to list format
        if trades is not None:
            # Convert DataFrame to list of dicts if needed
            if isinstance(trades, pd.DataFrame):
                trades_list = []
                for idx, row in trades.iterrows():
                    trades_list.append({
                        'entry_idx': int(row.get('EntryBar', idx)),
                        'exit_idx': int(row.get('ExitBar', idx)),
                        'entry_price': float(row.get('EntryPrice', 0)),
                        'exit_price': float(row.get('ExitPrice', 0)),
                        'shares': float(row.get('Size', 0)),
                        'pnl': float(row.get('PnL', 0)),
                        'exit_reason': str(row.get('ExitReason', 'signal'))
                    })
                trades = trades_list
            elif not isinstance(trades, list):
                trades = []
        
        if trades:
            entry_indices = []
            entry_prices = []
            entry_hover = []
            
            exit_indices = []
            exit_prices = []
            exit_hover = []
            
            # Separate long and short entries
            long_entry_indices = []
            long_entry_prices = []
            long_entry_hover = []
            short_entry_indices = []
            short_entry_prices = []
            short_entry_hover = []
            
            for i, trade in enumerate(trades):
                entry_idx = trade.get('entry_idx')
                exit_idx = trade.get('exit_idx')
                entry_price = trade.get('entry_price', 0)
                exit_price = trade.get('exit_price', 0)
                pnl = trade.get('pnl', 0)
                exit_reason = trade.get('exit_reason', 'signal')
                shares = trade.get('shares', 0)
                position_type = trade.get('position_type', 'long')
                
                # Determine if LONG or SHORT - check position_type first, fallback to shares sign
                is_long = position_type == 'long' if position_type else shares > 0
                
                # Entry markers - use DataFrame close price for Y to align with candles
                if entry_idx is not None and 0 <= entry_idx < len(df):
                    # Use close price from DataFrame for marker position (aligns with candles)
                    close_col = col_map.get('close', 'close')
                    chart_entry_price = float(df[close_col].iloc[entry_idx])
                    
                    hover_text = (
                        f"<b>Trade #{i+1} - {'LONG' if is_long else 'SHORT'} ENTRY</b><br>"
                        f"Executed: ${entry_price:.2f}<br>"
                        f"Close: ${chart_entry_price:.2f}<br>"
                        f"Shares: {abs(shares):.4f}"
                    )
                    
                    if is_long:
                        # LONG entry: RED ARROW UP
                        long_entry_indices.append(str(df.index[entry_idx]))
                        long_entry_prices.append(chart_entry_price)
                        long_entry_hover.append(hover_text)
                    else:
                        # SHORT entry: RED ARROW DOWN
                        short_entry_indices.append(str(df.index[entry_idx]))
                        short_entry_prices.append(chart_entry_price)
                        short_entry_hover.append(hover_text)
                
                # Exit markers - use DataFrame close price for Y to align with candles
                if exit_idx is not None and 0 <= exit_idx < len(df):
                    close_col = col_map.get('close', 'close')
                    chart_exit_price = float(df[close_col].iloc[exit_idx])
                    
                    exit_indices.append(str(df.index[exit_idx]))
                    exit_prices.append(chart_exit_price)
                    
                    pnl_color = 'green' if pnl > 0 else 'red'
                    exit_hover.append(
                        f"<b>Trade #{i+1} - EXIT</b><br>"
                        f"Price: ${exit_price:.2f}<br>"
                        f"PnL: ${pnl:.2f}<br>"
                        f"Return: {(exit_price/entry_price - 1)*100:.2f}%<br>"
                        f"Reason: {exit_reason}"
                    )
            
            # Add LONG entry markers (RED arrows UP)
            if long_entry_indices:
                fig.add_trace(go.Scatter(
                    x=long_entry_indices,
                    y=long_entry_prices,
                    mode='markers',
                    name='Long Entry',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='red',
                        line=dict(color='darkred', width=1)
                    ),
                    hovertext=long_entry_hover,
                    hovertemplate='%{hovertext}<extra></extra>'
                ))
            
            # Add SHORT entry markers (RED arrows DOWN)
            if short_entry_indices:
                fig.add_trace(go.Scatter(
                    x=short_entry_indices,
                    y=short_entry_prices,
                    mode='markers',
                    name='Short Entry',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='red',
                        line=dict(color='darkred', width=1)
                    ),
                    hovertext=short_entry_hover,
                    hovertemplate='%{hovertext}<extra></extra>'
                ))
            
            # Add exit markers (colored CROSS)
            if exit_indices:
                # Determine colors based on PnL
                exit_colors = ['green' if trade.get('pnl', 0) > 0 else 'red' 
                              for trade in trades]
                
                fig.add_trace(go.Scatter(
                    x=exit_indices,
                    y=exit_prices,
                    mode='markers',
                    name='Exit',
                    marker=dict(
                        symbol='x',
                        size=12,
                        color=exit_colors,
                        line=dict(width=2)
                    ),
                    hovertext=exit_hover,
                    hovertemplate='%{hovertext}<extra></extra>'
                ))
        
        fig.update_layout(
            title=f'<b>{model_name} - Price Chart with Trades</b>',
            xaxis_title='Date/Time',
            yaxis_title='Price ($)',
            height=600,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255, 255, 255, 0.8)"
            ),
            template='plotly_white'
        )
        
        return fig
    
    def _save_html_report(self, figures: List[go.Figure], filepath: Path, title: str, show: bool = True):
        """
        Save multiple figures to a single HTML file with charts stacked vertically.
        
        Args:
            figures: List of Plotly figures
            filepath: Path to save HTML file
            title: Report title
            show: If True, automatically open the report in the default browser (default: True)
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
        
        # Add each figure using Plotly's native to_html() which handles
        # binary data encoding and all trace types correctly (including candlestick/OHLC)
        for i, fig in enumerate(figures):
            html_content += '    <div class="chart-container">\n'
            # Use to_html with full_html=False to get just the div+script for this figure
            fig_html = fig.to_html(full_html=False, include_plotlyjs=False)
            html_content += f'        {fig_html}\n'
            html_content += '    </div>\n'
        
        html_content += """
</body>
</html>
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Automatically open in browser if requested
        if show:
            import webbrowser
            import os
            # Convert to absolute path and open in browser
            abs_path = os.path.abspath(filepath)
            webbrowser.open('file://' + abs_path)
    
    def _generate_all_models_rows(self,
                                   backtest_results: Dict[str, Any],
                                   test_results: Optional[Dict[str, Any]] = None,
                                   train_results: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate table rows for all models performance.
        
        Args:
            backtest_results: Backtest results dictionary
            test_results: Optional test results dictionary
            train_results: Optional train results dictionary
            
        Returns:
            HTML string with table rows
        """
        # Get all model names from all sources
        models = set()
        if backtest_results:
            models.update(backtest_results.keys())
        if test_results:
            models.update(test_results.keys())
        if train_results:
            models.update(train_results.keys())
        
        # If no models found, return empty
        if not models:
            return "<tr><td colspan='8' style='padding: 10px; text-align: center;'>No model data available</td></tr>"
        
        models = sorted(list(models))
        
        rows_html = ""
        for i, model_name in enumerate(models):
            # Alternating row colors
            bg_color = "rgba(255,255,255,0.05)" if i % 2 == 0 else "rgba(255,255,255,0.1)"
            
            # Get train metrics - try different possible keys
            train_acc = "N/A"
            val_acc = "N/A"
            train_time = "N/A"
            if train_results and model_name in train_results:
                train_res = train_results[model_name]
                # Check if it's a success status or just has data
                if isinstance(train_res, dict):
                    if train_res.get('status') == 'success' or 'train_accuracy' in train_res:
                        train_acc_val = train_res.get('train_accuracy') or train_res.get('accuracy', 0)
                        if train_acc_val and train_acc_val > 0:
                            train_acc = f"{train_acc_val:.4f}"
                        
                        val = train_res.get('val_accuracy') or train_res.get('validation_accuracy')
                        if val is not None and val > 0:
                            val_acc = f"{val:.4f}"
                        
                        time_val = train_res.get('training_time') or train_res.get('time', 0)
                        if time_val and time_val > 0:
                            train_time = f"{time_val:.2f}s"
            
            # Get test metrics - try different possible keys
            test_acc = "N/A"
            test_precision = "N/A"
            test_recall = "N/A"
            test_f1 = "N/A"
            if test_results and model_name in test_results:
                test_res = test_results[model_name]
                # Check if it's a success status or just has data
                if isinstance(test_res, dict):
                    if test_res.get('status') == 'success' or 'accuracy' in test_res:
                        acc_val = test_res.get('accuracy', 0)
                        if acc_val and acc_val > 0:
                            test_acc = f"{acc_val:.4f}"
                        
                        prec_val = test_res.get('precision', 0)
                        if prec_val and prec_val > 0:
                            test_precision = f"{prec_val:.4f}"
                        
                        rec_val = test_res.get('recall', 0)
                        if rec_val and rec_val > 0:
                            test_recall = f"{rec_val:.4f}"
                        
                        f1_val = test_res.get('f1_score') or test_res.get('f1', 0)
                        if f1_val and f1_val > 0:
                            test_f1 = f"{f1_val:.4f}"
            
            rows_html += f"""
                            <tr style="background: {bg_color};">
                                <td style="padding: 8px; border-bottom: 1px solid rgba(255,255,255,0.1);">{model_name}</td>
                                <td style="padding: 8px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.1);">{train_acc}</td>
                                <td style="padding: 8px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.1);">{val_acc}</td>
                                <td style="padding: 8px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.1);">{test_acc}</td>
                                <td style="padding: 8px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.1);">{test_precision}</td>
                                <td style="padding: 8px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.1);">{test_recall}</td>
                                <td style="padding: 8px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.1);">{test_f1}</td>
                                <td style="padding: 8px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.1);">{train_time}</td>
                            </tr>
"""
        
        return rows_html
    
    def _generate_summary_html(self,
                               backtest_results: Dict[str, Any],
                               test_results: Optional[Dict[str, Any]] = None,
                               train_results: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate HTML summary section.
        
        Args:
            backtest_results: Backtest results dictionary
            test_results: Optional test results dictionary
            train_results: Optional train results dictionary
            
        Returns:
            HTML string for summary section
        """
        # Find best model
        best_model = None
        best_return = -float('inf')
        best_sharpe = -float('inf')
        
        for model_name, results in backtest_results.items():
            metrics = results.get('metrics', results)
            total_return = metrics.get('total_return', 0)
            sharpe = metrics.get('sharpe_ratio', 0)
            
            if total_return > best_return:
                best_return = total_return
                best_model = model_name
            if sharpe > best_sharpe:
                best_sharpe = sharpe
        
        # Get best model details
        best_backtest = backtest_results.get(best_model, {})
        best_metrics = best_backtest.get('metrics', best_backtest)
        
        total_trades = len(best_backtest.get('trades', []))
        initial_capital = best_backtest.get('initial_capital', 10000)
        final_capital = best_metrics.get('final_capital', initial_capital)
        final_pnl = final_capital - initial_capital
        win_rate = best_metrics.get('win_rate', 0) * 100
        max_dd = best_metrics.get('max_drawdown', 0) * 100
        
        # Get test metrics
        test_accuracy = 'N/A'
        test_f1 = 'N/A'
        if test_results and best_model in test_results:
            test_res = test_results[best_model]
            if test_res.get('status') == 'success':
                test_accuracy = f"{test_res.get('accuracy', 0):.4f}"
                test_f1 = f"{test_res.get('f1_score', 0):.4f}"
        
        # Get training time
        train_time = 'N/A'
        if train_results and best_model in train_results:
            train_res = train_results[best_model]
            if train_res.get('status') == 'success':
                train_time = f"{train_res.get('training_time', 0):.2f}s"
        
        pnl_color = '#4ade80' if final_pnl >= 0 else '#f87171'
        return_color = '#4ade80' if best_return >= 0 else '#f87171'
        
        summary_html = f"""
        <div style="font-family: Arial, sans-serif; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin: 20px 0;">
            <h1 style="text-align: center; margin-bottom: 30px; font-size: 32px;">📊 Results Summary</h1>
            
            <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                <h2 style="font-size: 24px; margin-bottom: 15px;">🏆 Best Performing Model</h2>
                <p style="font-size: 28px; font-weight: bold; margin: 10px 0;">{best_model}</p>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-bottom: 20px;">
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
                    <h3 style="font-size: 16px; margin-bottom: 8px; opacity: 0.9;">💰 Final PnL</h3>
                    <p style="font-size: 24px; font-weight: bold; margin: 0; color: {pnl_color};">${final_pnl:,.2f}</p>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
                    <h3 style="font-size: 16px; margin-bottom: 8px; opacity: 0.9;">📈 Total Return</h3>
                    <p style="font-size: 24px; font-weight: bold; margin: 0; color: {return_color};">{best_return*100:.2f}%</p>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
                    <h3 style="font-size: 16px; margin-bottom: 8px; opacity: 0.9;">📊 Sharpe Ratio</h3>
                    <p style="font-size: 24px; font-weight: bold; margin: 0;">{best_sharpe:.3f}</p>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
                    <h3 style="font-size: 16px; margin-bottom: 8px; opacity: 0.9;">🎯 Win Rate</h3>
                    <p style="font-size: 24px; font-weight: bold; margin: 0;">{win_rate:.1f}%</p>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
                    <h3 style="font-size: 16px; margin-bottom: 8px; opacity: 0.9;">🔻 Max Drawdown</h3>
                    <p style="font-size: 24px; font-weight: bold; margin: 0; color: #f87171;">{max_dd:.2f}%</p>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
                    <h3 style="font-size: 16px; margin-bottom: 8px; opacity: 0.9;">🔄 Total Trades</h3>
                    <p style="font-size: 24px; font-weight: bold; margin: 0;">{total_trades}</p>
                </div>
            </div>
            
            <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                <h2 style="font-size: 20px; margin-bottom: 15px;">📋 Model Performance Metrics</h2>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
                    <div>
                        <p style="margin: 5px 0; opacity: 0.9;">Test Accuracy:</p>
                        <p style="margin: 5px 0; font-weight: bold; font-size: 18px;">{test_accuracy}</p>
                    </div>
                    <div>
                        <p style="margin: 5px 0; opacity: 0.9;">Test F1 Score:</p>
                        <p style="margin: 5px 0; font-weight: bold; font-size: 18px;">{test_f1}</p>
                    </div>
                    <div>
                        <p style="margin: 5px 0; opacity: 0.9;">Training Time:</p>
                        <p style="margin: 5px 0; font-weight: bold; font-size: 18px;">{train_time}</p>
                    </div>
                </div>
            </div>
            
            <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 8px; margin-top: 20px;">
                <h2 style="font-size: 20px; margin-bottom: 15px;">📊 All Models Performance</h2>
                <div style="overflow-x: auto;">
                    <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                        <thead>
                            <tr style="background: rgba(255,255,255,0.2);">
                                <th style="padding: 8px; text-align: left; border-bottom: 2px solid rgba(255,255,255,0.3);">Model</th>
                                <th style="padding: 8px; text-align: center; border-bottom: 2px solid rgba(255,255,255,0.3); font-size: 12px;">Train Acc</th>
                                <th style="padding: 8px; text-align: center; border-bottom: 2px solid rgba(255,255,255,0.3); font-size: 12px;">Val Acc</th>
                                <th style="padding: 8px; text-align: center; border-bottom: 2px solid rgba(255,255,255,0.3); font-size: 12px;">Test Acc</th>
                                <th style="padding: 8px; text-align: center; border-bottom: 2px solid rgba(255,255,255,0.3); font-size: 12px;">Precision</th>
                                <th style="padding: 8px; text-align: center; border-bottom: 2px solid rgba(255,255,255,0.3); font-size: 12px;">Recall</th>
                                <th style="padding: 8px; text-align: center; border-bottom: 2px solid rgba(255,255,255,0.3); font-size: 12px;">F1 Score</th>
                                <th style="padding: 8px; text-align: center; border-bottom: 2px solid rgba(255,255,255,0.3); font-size: 12px;">Train Time</th>
                            </tr>
                        </thead>
                        <tbody>
{self._generate_all_models_rows(backtest_results, test_results, train_results)}
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.3);">
                <p style="font-size: 14px; opacity: 0.8; margin: 0;">Generated by ML Framework - Backtest Analysis System</p>
            </div>
        </div>
        """
        
        return summary_html
    
    def _save_html_report_with_summary(self, 
                                       figures: List[go.Figure], 
                                       filepath: Path, 
                                       title: str,
                                       summary_html: str,
                                       show: bool = True):
        """
        Save multiple figures to a single HTML file with charts stacked vertically and a summary section.
        
        Args:
            figures: List of Plotly figures
            filepath: Path to save HTML file
            title: Report title
            summary_html: HTML content for summary section
            show: If True, automatically open the report in the default browser (default: True)
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
        
        # Add each figure
        for i, fig in enumerate(figures):
            html_content += '    <div class="chart-container">\n'
            fig_html = fig.to_html(full_html=False, include_plotlyjs=False)
            html_content += f'        {fig_html}\n'
            html_content += '    </div>\n'
        
        # Add summary section at the end
        html_content += f"""
    <div class="chart-container">
        {summary_html}
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Automatically open in browser if requested
        if show:
            import webbrowser
            import os
            abs_path = os.path.abspath(filepath)
            webbrowser.open('file://' + abs_path)
