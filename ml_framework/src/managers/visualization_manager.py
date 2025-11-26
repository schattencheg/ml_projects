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
        self._save_html_report(figures, html_path, "Training Results Report", show=show)
        
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
        self._save_html_report(figures, html_path, "Test Results Report", show=show)
        
        print(f"✓ Saved test report: {html_path}")
        if show:
            print(f"🎉 Opening report in browser...")
        print("="*70 + "\n")
        
        return str(html_path)
    
    def create_backtest_report(self,
                              backtest_results: Dict[str, Any],
                              save_dir: Path,
                              df: Optional[pd.DataFrame] = None,
                              show: bool = True) -> str:
        """
        Create HTML report for backtest results.
        
        Args:
            backtest_results: Backtest results dictionary
            save_dir: Directory to save report
            df: Optional DataFrame with OHLC data for trade visualization
            show: If True, automatically open the report in browser (default: True)
            
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
        
        # 2. Equity Curves Comparison (NEW - All curves on one chart)
        fig = self.create_equity_curves_comparison(backtest_results)
        figures.append(fig)
        
        # 3. Individual Model Results
        for model_name, results in backtest_results.items():
            # Individual equity curve
            if 'equity_curve' in results:
                fig = self._create_equity_curve(model_name, results)
                figures.append(fig)
            
            # OHLC chart with trades (NEW)
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
        
        # Save HTML report
        html_path = save_dir / 'backtest_report.html'
        self._save_html_report(figures, html_path, "Backtest Results Report", show=show)
        
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
                
                # Determine if LONG or SHORT based on shares sign
                is_long = shares > 0
                
                # Entry markers
                if entry_idx is not None and entry_idx < len(df):
                    hover_text = (
                        f"<b>Trade #{i+1} - {'LONG' if is_long else 'SHORT'} ENTRY</b><br>"
                        f"Price: ${entry_price:.2f}<br>"
                        f"Shares: {abs(shares):.2f}"
                    )
                    
                    if is_long:
                        # LONG entry: RED ARROW UP
                        long_entry_indices.append(str(df.index[entry_idx]))
                        long_entry_prices.append(float(entry_price))
                        long_entry_hover.append(hover_text)
                    else:
                        # SHORT entry: RED ARROW DOWN
                        short_entry_indices.append(str(df.index[entry_idx]))
                        short_entry_prices.append(float(entry_price))
                        short_entry_hover.append(hover_text)
                
                # Exit markers (same color CROSS)
                if exit_idx is not None and exit_idx < len(df):
                    exit_indices.append(str(df.index[exit_idx]))
                    exit_prices.append(float(exit_price))
                    
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
