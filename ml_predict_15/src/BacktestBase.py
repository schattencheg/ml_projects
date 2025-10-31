"""
Base Backtest Class for ML Trading Strategies

This module provides a comprehensive base class for all backtesting implementations
with detailed reporting, visualization, and trade analysis capabilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import warnings
from pathlib import Path
import json

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class BacktestBase(ABC):
    """
    Abstract base class for all backtesting implementations.
    
    Provides common functionality for:
    - Trade tracking and analysis
    - Performance metrics calculation
    - Detailed reporting
    - Advanced visualization
    - Export capabilities
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        **kwargs
    ):
        """
        Initialize base backtest class.
        
        Parameters:
        -----------
        initial_capital : float
            Starting capital
        commission : float
            Commission per trade as fraction
        slippage : float
            Slippage per trade as fraction
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # Results storage
        self.trades = []
        self.equity_curve = []
        self.signals = []
        self.daily_returns = []
        
        # Metrics cache
        self._metrics_cache = {}
        self._last_calculation = None
        
    @abstractmethod
    def run_backtest(self, *args, **kwargs) -> Tuple[Dict, pd.DataFrame]:
        """
        Run the backtest. Must be implemented by subclasses.
        
        Returns:
        --------
        results : Dict
            Backtest results and metrics
        trades : pd.DataFrame
            Trade history
        """
        pass
    
    def calculate_comprehensive_metrics(
        self, 
        df: pd.DataFrame = None,
        benchmark_returns: pd.Series = None
    ) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Parameters:
        -----------
        df : pd.DataFrame, optional
            Original price data for benchmark calculation
        benchmark_returns : pd.Series, optional
            Benchmark returns for comparison
            
        Returns:
        --------
        metrics : Dict
            Comprehensive performance metrics
        """
        if not self.trades:
            return self._empty_metrics()
        
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Basic metrics
        metrics = self._calculate_basic_metrics(trades_df, equity_df)
        
        # Risk metrics
        metrics.update(self._calculate_risk_metrics(equity_df))
        
        # Trade analysis
        metrics.update(self._calculate_trade_analysis(trades_df))
        
        # Time-based analysis
        metrics.update(self._calculate_time_analysis(trades_df))
        
        # Benchmark comparison
        if df is not None:
            metrics.update(self._calculate_benchmark_metrics(df, equity_df))
        
        # Advanced ratios
        metrics.update(self._calculate_advanced_ratios(equity_df))
        
        return metrics
    
    def _calculate_basic_metrics(self, trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> Dict:
        """Calculate basic performance metrics."""
        final_value = equity_df['equity'].iloc[-1] if len(equity_df) > 0 else self.initial_capital
        total_return = final_value - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'total_trades': len(trades_df),
        }
    
    def _calculate_risk_metrics(self, equity_df: pd.DataFrame) -> Dict:
        """Calculate risk-related metrics."""
        if len(equity_df) < 2:
            return {'sharpe_ratio': 0, 'max_drawdown': 0, 'volatility': 0}
        
        # Calculate returns
        equity_df = equity_df.copy()
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        # Sharpe ratio (annualized)
        mean_return = equity_df['returns'].mean()
        std_return = equity_df['returns'].std()
        sharpe_ratio = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0
        
        # Maximum drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].min() * 100
        
        # Volatility (annualized)
        volatility = std_return * np.sqrt(252) * 100
        
        # Sortino ratio
        downside_returns = equity_df['returns'][equity_df['returns'] < 0]
        downside_std = downside_returns.std()
        sortino_ratio = (mean_return / downside_std * np.sqrt(252)) if downside_std > 0 else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
        }
    
    def _calculate_trade_analysis(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate trade-specific metrics."""
        if len(trades_df) == 0:
            return {}
        
        # Win/Loss analysis
        winning_trades = trades_df[trades_df['net_pnl'] > 0] if 'net_pnl' in trades_df.columns else trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['net_pnl'] <= 0] if 'net_pnl' in trades_df.columns else trades_df[trades_df['pnl'] <= 0]
        
        pnl_col = 'net_pnl' if 'net_pnl' in trades_df.columns else 'pnl'
        
        win_rate = len(winning_trades) / len(trades_df) * 100
        avg_win = winning_trades[pnl_col].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades[pnl_col].mean() if len(losing_trades) > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades[pnl_col].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades[pnl_col].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Consecutive wins/losses
        trades_df['win'] = trades_df[pnl_col] > 0
        trades_df['streak'] = (trades_df['win'] != trades_df['win'].shift()).cumsum()
        streak_analysis = trades_df.groupby(['streak', 'win']).size()
        
        max_consecutive_wins = streak_analysis[streak_analysis.index.get_level_values(1) == True].max() if len(streak_analysis) > 0 else 0
        max_consecutive_losses = streak_analysis[streak_analysis.index.get_level_values(1) == False].max() if len(streak_analysis) > 0 else 0
        
        return {
            'won_trades': len(winning_trades),
            'lost_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'best_trade': winning_trades[pnl_col].max() if len(winning_trades) > 0 else 0,
            'worst_trade': losing_trades[pnl_col].min() if len(losing_trades) > 0 else 0,
            'profit_factor': profit_factor,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
        }
    
    def _calculate_time_analysis(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate time-based metrics."""
        if len(trades_df) == 0 or 'entry_time' not in trades_df.columns:
            return {}
        
        # Holding period analysis
        if 'bars_held' in trades_df.columns:
            avg_holding_period = trades_df['bars_held'].mean()
            max_holding_period = trades_df['bars_held'].max()
            min_holding_period = trades_df['bars_held'].min()
        else:
            avg_holding_period = max_holding_period = min_holding_period = 0
        
        # Monthly/yearly analysis
        trades_df['entry_month'] = pd.to_datetime(trades_df['entry_time']).dt.month
        trades_df['entry_year'] = pd.to_datetime(trades_df['entry_time']).dt.year
        
        pnl_col = 'net_pnl' if 'net_pnl' in trades_df.columns else 'pnl'
        monthly_performance = trades_df.groupby('entry_month')[pnl_col].sum()
        best_month = monthly_performance.idxmax() if len(monthly_performance) > 0 else None
        worst_month = monthly_performance.idxmin() if len(monthly_performance) > 0 else None
        
        return {
            'avg_holding_period': avg_holding_period,
            'max_holding_period': max_holding_period,
            'min_holding_period': min_holding_period,
            'best_month': best_month,
            'worst_month': worst_month,
        }
    
    def _calculate_benchmark_metrics(self, df: pd.DataFrame, equity_df: pd.DataFrame) -> Dict:
        """Calculate benchmark comparison metrics."""
        if 'close' not in df.columns and 'Close' not in df.columns:
            return {}
        
        close_col = 'close' if 'close' in df.columns else 'Close'
        buy_hold_return = (df[close_col].iloc[-1] / df[close_col].iloc[0] - 1) * 100
        
        # Calculate correlation with benchmark
        if len(equity_df) > 1 and len(df) > 1:
            # Align data
            min_len = min(len(equity_df), len(df))
            portfolio_returns = equity_df['equity'].iloc[:min_len].pct_change().dropna()
            benchmark_returns = df[close_col].iloc[:min_len].pct_change().dropna()
            
            if len(portfolio_returns) > 1 and len(benchmark_returns) > 1:
                correlation = portfolio_returns.corr(benchmark_returns)
                beta = portfolio_returns.cov(benchmark_returns) / benchmark_returns.var()
            else:
                correlation = beta = 0
        else:
            correlation = beta = 0
        
        return {
            'buy_and_hold_return_pct': buy_hold_return,
            'correlation_with_benchmark': correlation,
            'beta': beta,
        }
    
    def _calculate_advanced_ratios(self, equity_df: pd.DataFrame) -> Dict:
        """Calculate advanced performance ratios."""
        if len(equity_df) < 2:
            return {}
        
        equity_df = equity_df.copy()
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        # Calmar ratio (annual return / max drawdown)
        annual_return = (equity_df['equity'].iloc[-1] / equity_df['equity'].iloc[0]) ** (252 / len(equity_df)) - 1
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        max_dd = abs(equity_df['drawdown'].min())
        calmar_ratio = annual_return / max_dd if max_dd > 0 else 0
        
        # Information ratio (assuming benchmark return is 0)
        mean_excess_return = equity_df['returns'].mean()
        tracking_error = equity_df['returns'].std()
        information_ratio = mean_excess_return / tracking_error if tracking_error > 0 else 0
        
        return {
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            'annual_return': annual_return * 100,
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics when no trades exist."""
        return {
            'initial_capital': self.initial_capital,
            'final_value': self.initial_capital,
            'total_return': 0,
            'total_return_pct': 0,
            'total_trades': 0,
            'won_trades': 0,
            'lost_trades': 0,
            'win_rate': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
        }
    
    def generate_detailed_report(self, results: Dict, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive text report.
        
        Parameters:
        -----------
        results : Dict
            Backtest results
        save_path : str, optional
            Path to save the report
            
        Returns:
        --------
        report : str
            Formatted report text
        """
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("COMPREHENSIVE BACKTEST REPORT")
        report_lines.append("=" * 100)
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Capital Summary
        report_lines.append("CAPITAL SUMMARY")
        report_lines.append("-" * 50)
        report_lines.append(f"Initial Capital:           ${results.get('initial_capital', 0):,.2f}")
        report_lines.append(f"Final Value:               ${results.get('final_value', 0):,.2f}")
        report_lines.append(f"Total Return:              ${results.get('total_return', 0):,.2f}")
        report_lines.append(f"Total Return %:            {results.get('total_return_pct', 0):.2f}%")
        if 'buy_and_hold_return_pct' in results:
            report_lines.append(f"Buy & Hold Return %:       {results['buy_and_hold_return_pct']:.2f}%")
        if 'annual_return' in results:
            report_lines.append(f"Annualized Return %:       {results['annual_return']:.2f}%")
        report_lines.append("")
        
        # Trade Statistics
        report_lines.append("TRADE STATISTICS")
        report_lines.append("-" * 50)
        report_lines.append(f"Total Trades:              {results.get('total_trades', 0)}")
        report_lines.append(f"Winning Trades:            {results.get('won_trades', 0)}")
        report_lines.append(f"Losing Trades:             {results.get('lost_trades', 0)}")
        report_lines.append(f"Win Rate:                  {results.get('win_rate', 0):.2f}%")
        report_lines.append(f"Average Win:               ${results.get('avg_win', 0):.2f}")
        report_lines.append(f"Average Loss:              ${results.get('avg_loss', 0):.2f}")
        report_lines.append(f"Best Trade:                ${results.get('best_trade', 0):.2f}")
        report_lines.append(f"Worst Trade:               ${results.get('worst_trade', 0):.2f}")
        report_lines.append(f"Profit Factor:             {results.get('profit_factor', 0):.2f}")
        if 'max_consecutive_wins' in results:
            report_lines.append(f"Max Consecutive Wins:      {results['max_consecutive_wins']}")
        if 'max_consecutive_losses' in results:
            report_lines.append(f"Max Consecutive Losses:    {results['max_consecutive_losses']}")
        report_lines.append("")
        
        # Risk Metrics
        report_lines.append("RISK METRICS")
        report_lines.append("-" * 50)
        report_lines.append(f"Maximum Drawdown:          {results.get('max_drawdown', 0):.2f}%")
        report_lines.append(f"Volatility (Annual):       {results.get('volatility', 0):.2f}%")
        report_lines.append(f"Sharpe Ratio:              {results.get('sharpe_ratio', 0):.2f}")
        if 'sortino_ratio' in results:
            report_lines.append(f"Sortino Ratio:             {results['sortino_ratio']:.2f}")
        if 'calmar_ratio' in results:
            report_lines.append(f"Calmar Ratio:              {results['calmar_ratio']:.2f}")
        if 'information_ratio' in results:
            report_lines.append(f"Information Ratio:         {results['information_ratio']:.2f}")
        report_lines.append("")
        
        # Time Analysis
        if 'avg_holding_period' in results:
            report_lines.append("TIME ANALYSIS")
            report_lines.append("-" * 50)
            report_lines.append(f"Avg Holding Period:        {results['avg_holding_period']:.1f} bars")
            report_lines.append(f"Max Holding Period:        {results.get('max_holding_period', 0):.0f} bars")
            report_lines.append(f"Min Holding Period:        {results.get('min_holding_period', 0):.0f} bars")
            if results.get('best_month'):
                report_lines.append(f"Best Month:                {results['best_month']}")
            if results.get('worst_month'):
                report_lines.append(f"Worst Month:               {results['worst_month']}")
            report_lines.append("")
        
        # Benchmark Comparison
        if 'correlation_with_benchmark' in results:
            report_lines.append("BENCHMARK COMPARISON")
            report_lines.append("-" * 50)
            report_lines.append(f"Correlation:               {results['correlation_with_benchmark']:.2f}")
            report_lines.append(f"Beta:                      {results.get('beta', 0):.2f}")
            report_lines.append("")
        
        report_lines.append("=" * 100)
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {save_path}")
        
        return report_text
    
    def create_comprehensive_visualizations(
        self, 
        results: Dict, 
        df: pd.DataFrame = None,
        save_dir: Optional[str] = None,
        show_plots: bool = True,
        model_name: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Create comprehensive visualization suite with separate interactive HTML files.
        
        Creates 5 separate Plotly HTML files:
        1. Performance Overview (equity, drawdown, returns, metrics)
        2. Trade Analysis (P&L over time, win/loss, duration, monthly)
        3. Risk Analysis (rolling Sharpe, volatility, underwater, risk-return)
        4. Monthly Heatmap
        5. Trade Distribution (P&L dist, box plot, CDF, Q-Q plot)
        
        Parameters:
        -----------
        results : Dict
            Backtest results
        df : pd.DataFrame, optional
            Original price data
        save_dir : str, optional
            Directory to save plots
        show_plots : bool
            Whether to display plots
        model_name : str, optional
            Model name for titles
            
        Returns:
        --------
        saved_files : Dict[str, str]
            Dictionary of plot names and their file paths
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.express as px
            import webbrowser
            import os
        except ImportError:
            print("⚠ Plotly not installed. Falling back to matplotlib.")
            return self._create_comprehensive_visualizations_matplotlib(results, df, save_dir, show_plots)
        
        saved_files = {}
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        else:
            save_dir = 'backtest_results'
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*70)
        print("CREATING COMPREHENSIVE PLOTLY VISUALIZATIONS")
        print("="*70)
        
        # Extract data
        equity_curve = results.get('equity_curve', self.equity_curve if hasattr(self, 'equity_curve') else None)
        trades = results.get('trades', self.trades if hasattr(self, 'trades') else None)
        
        if equity_curve is None or len(equity_curve) == 0:
            print("  ⚠ No equity curve data available")
            return saved_files
        
        equity_df = pd.DataFrame(equity_curve)
        
        # Create comprehensive figure with all plots stacked vertically
        # Total: 17 plots in 5 rows
        fig = make_subplots(
            rows=9, cols=2,
            row_heights=[0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.08, 0.08],
            subplot_titles=(
                # Row 1: Performance Overview
                'Equity Curve', 'Drawdown (Underwater)',
                # Row 2: Performance Overview continued
                'Returns Distribution', 'Performance Metrics',
                # Row 3: Trade Analysis
                'Cumulative P&L by Trade', 'Win/Loss Distribution',
                # Row 4: Trade Analysis continued
                'Trade Duration Analysis', 'Monthly Performance',
                # Row 5: Risk Analysis
                'Rolling Sharpe Ratio', 'Rolling Volatility',
                # Row 6: Risk Analysis continued
                'Underwater Curve', 'Risk-Return Scatter',
                # Row 7: Trade Distribution
                'P&L Distribution', 'Win/Loss Box Plot',
                # Row 8: Trade Distribution continued
                'Cumulative Distribution', 'Q-Q Plot (Normal)',
                # Row 9: Monthly Heatmap (spans 2 columns)
                'Monthly Returns Heatmap', None
            ),
            specs=[
                # Row 1
                [{'type': 'scatter'}, {'type': 'scatter'}],
                # Row 2
                [{'type': 'histogram'}, {'type': 'table'}],
                # Row 3
                [{'type': 'scatter'}, {'type': 'histogram'}],
                # Row 4
                [{'type': 'box'}, {'type': 'bar'}],
                # Row 5
                [{'type': 'scatter'}, {'type': 'scatter'}],
                # Row 6
                [{'type': 'scatter'}, {'type': 'scatter'}],
                # Row 7
                [{'type': 'histogram'}, {'type': 'box'}],
                # Row 8
                [{'type': 'scatter'}, {'type': 'scatter'}],
                # Row 9
                [{'type': 'heatmap', 'colspan': 2}, None]
            ],
            vertical_spacing=0.04,
            horizontal_spacing=0.08
        )
        
        # ===== ROW 1: PERFORMANCE OVERVIEW =====
        # 1. Equity Curve
        fig.add_trace(
            go.Scatter(
                x=equity_df['timestamp'],
                y=equity_df['equity'],
                mode='lines',
                name='Equity',
                line=dict(color='blue', width=2),
                fill='tozeroy',
                fillcolor='rgba(0,100,255,0.2)',
                showlegend=False
            ),
            row=1, col=1
        )
        initial_capital = results.get('initial_capital', 10000)
        # Add initial capital line
        fig.add_trace(
            go.Scatter(
                x=[equity_df['timestamp'].iloc[0], equity_df['timestamp'].iloc[-1]],
                y=[initial_capital, initial_capital],
                mode='lines',
                line=dict(color='gray', dash='dash', width=1),
                name='Initial Capital',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        # 2. Drawdown
        peak = equity_df['equity'].expanding().max()
        drawdown = ((equity_df['equity'] - peak) / peak * 100)
        fig.add_trace(
            go.Scatter(
                x=equity_df['timestamp'],
                y=drawdown,
                mode='lines',
                name='Drawdown',
                line=dict(color='red', width=2),
                fill='tozeroy',
                fillcolor='rgba(255,0,0,0.2)',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # ===== ROW 2: PERFORMANCE OVERVIEW CONTINUED =====
        # 3. Returns Distribution
        returns = equity_df['equity'].pct_change().dropna() * 100
        fig.add_trace(
            go.Histogram(
                x=returns,
                name='Returns',
                marker_color='skyblue',
                nbinsx=50,
                showlegend=False
            ),
            row=2, col=1
        )
        mean_return = returns.mean()
        # Add mean return line
        fig.add_trace(
            go.Scatter(
                x=[mean_return, mean_return],
                y=[0, len(returns) * 0.1],  # Approximate height for histogram
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name=f'Mean: {mean_return:.2f}%',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )
        
        # 4. Performance Metrics Table
        metrics_data = [
            ['Total Return', f"{results.get('total_return_pct', 0):.2f}%"],
            ['Sharpe Ratio', f"{results.get('sharpe_ratio', 0):.2f}"],
            ['Max Drawdown', f"{results.get('max_drawdown', 0):.2f}%"],
            ['Win Rate', f"{results.get('win_rate', 0):.1f}%"],
            ['Total Trades', f"{results.get('total_trades', 0)}"],
            ['Profit Factor', f"{results.get('profit_factor', 0):.2f}"],
            ['Avg Win', f"${results.get('avg_win', 0):.2f}"],
            ['Avg Loss', f"${results.get('avg_loss', 0):.2f}"]
        ]
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['<b>Metric</b>', '<b>Value</b>'],
                    fill_color='paleturquoise',
                    align='left',
                    font=dict(size=11)
                ),
                cells=dict(
                    values=[[row[0] for row in metrics_data], [row[1] for row in metrics_data]],
                    fill_color='lavender',
                    align='left',
                    font=dict(size=10)
                )
            ),
            row=2, col=2
        )
        
        # ===== ROW 3-4: TRADE ANALYSIS =====
        if trades is not None and len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            
            # 5. Cumulative P&L
            cumulative_pnl = trades_df['net_pnl'].cumsum()
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(cumulative_pnl))),
                    y=cumulative_pnl,
                    mode='lines+markers',
                    name='Cumulative P&L',
                    line=dict(color='blue', width=2),
                    marker=dict(size=3),
                    showlegend=False
                ),
                row=3, col=1
            )
            
            # 6. Win/Loss Distribution
            wins = trades_df[trades_df['net_pnl'] > 0]['net_pnl']
            losses = trades_df[trades_df['net_pnl'] <= 0]['net_pnl']
            fig.add_trace(
                go.Histogram(x=wins, name='Wins', marker_color='green', opacity=0.7, nbinsx=20),
                row=3, col=2
            )
            fig.add_trace(
                go.Histogram(x=losses, name='Losses', marker_color='red', opacity=0.7, nbinsx=20),
                row=3, col=2
            )
            
            # 7. Trade Duration Box Plot
            if 'duration_bars' not in trades_df.columns:
                if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
                    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
                    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
                    trades_df['duration_bars'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 3600
                elif 'bars_held' in trades_df.columns:
                    trades_df['duration_bars'] = trades_df['bars_held']
            
            if 'duration_bars' in trades_df.columns and len(trades_df['duration_bars'].dropna()) > 0:
                fig.add_trace(
                    go.Box(
                        y=trades_df['duration_bars'],
                        name='Duration',
                        marker_color='orange',
                        boxmean='sd',
                        showlegend=False
                    ),
                    row=4, col=1
                )
            
            # 8. Monthly Performance
            if 'exit_time' in trades_df.columns:
                trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
                trades_df['month'] = trades_df['exit_time'].dt.month
                monthly_pnl = trades_df.groupby('month')['net_pnl'].sum()
                colors = ['green' if x > 0 else 'red' for x in monthly_pnl.values]
                fig.add_trace(
                    go.Bar(
                        x=monthly_pnl.index,
                        y=monthly_pnl.values,
                        marker_color=colors,
                        name='Monthly P&L',
                        text=[f'${x:.0f}' for x in monthly_pnl.values],
                        textposition='auto',
                        showlegend=False
                    ),
                    row=4, col=2
                )
        
        # ===== ROW 5-6: RISK ANALYSIS =====
        equity_df['returns'] = equity_df['equity'].pct_change()
        window = min(30, len(equity_df) // 4)
        
        # 9. Rolling Sharpe
        if window > 1:
            rolling_sharpe = (equity_df['returns'].rolling(window).mean() / 
                            equity_df['returns'].rolling(window).std() * np.sqrt(252))
            fig.add_trace(
                go.Scatter(
                    x=equity_df['timestamp'],
                    y=rolling_sharpe,
                    mode='lines',
                    name=f'Sharpe ({window}d)',
                    line=dict(color='purple', width=2),
                    showlegend=False
                ),
                row=5, col=1
            )
        
        # 10. Rolling Volatility
        if window > 1:
            rolling_vol = equity_df['returns'].rolling(window).std() * np.sqrt(252) * 100
            fig.add_trace(
                go.Scatter(
                    x=equity_df['timestamp'],
                    y=rolling_vol,
                    mode='lines',
                    name=f'Volatility ({window}d)',
                    line=dict(color='orange', width=2),
                    showlegend=False
                ),
                row=5, col=2
            )
        
        # 11. Underwater Curve
        fig.add_trace(
            go.Scatter(
                x=equity_df['timestamp'],
                y=drawdown,
                mode='lines',
                name='Drawdown',
                line=dict(color='red', width=2),
                fill='tozeroy',
                fillcolor='rgba(255,0,0,0.2)',
                showlegend=False
            ),
            row=6, col=1
        )
        
        # 12. Risk-Return Scatter
        if trades is not None and len(trades) > 0 and 'duration_bars' in trades_df.columns:
            if len(trades_df['duration_bars'].dropna()) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=trades_df['duration_bars'],
                        y=trades_df['net_pnl'],
                        mode='markers',
                        name='Trades',
                        marker=dict(
                            color=trades_df['net_pnl'],
                            colorscale='RdYlGn',
                            size=6,
                            opacity=0.6,
                            showscale=True,
                            colorbar=dict(title='P&L', len=0.3, y=0.25)
                        ),
                        showlegend=False
                    ),
                    row=6, col=2
                )
        
        # ===== ROW 7-8: TRADE DISTRIBUTION =====
        if trades is not None and len(trades) > 0:
            pnl_col = 'net_pnl'
            
            # 13. P&L Distribution
            fig.add_trace(
                go.Histogram(
                    x=trades_df[pnl_col],
                    name='P&L',
                    marker_color='skyblue',
                    nbinsx=30,
                    showlegend=False
                ),
                row=7, col=1
            )
            mean_pnl = trades_df[pnl_col].mean()
            # Add mean line as a vertical line trace
            fig.add_trace(
                go.Scatter(
                    x=[mean_pnl, mean_pnl],
                    y=[0, trades_df[pnl_col].value_counts().max()],
                    mode='lines',
                    line=dict(color='red', dash='dash', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=7, col=1
            )
            # Add zero line
            fig.add_trace(
                go.Scatter(
                    x=[0, 0],
                    y=[0, trades_df[pnl_col].value_counts().max()],
                    mode='lines',
                    line=dict(color='black', dash='dash', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=7, col=1
            )
            
            # 14. Win/Loss Box Plot
            fig.add_trace(
                go.Box(y=wins, name='Wins', marker_color='green', boxmean='sd'),
                row=7, col=2
            )
            fig.add_trace(
                go.Box(y=losses, name='Losses', marker_color='red', boxmean='sd'),
                row=7, col=2
            )
            
            # 15. Cumulative Distribution
            sorted_pnl = np.sort(trades_df[pnl_col])
            cumulative_prob = np.arange(1, len(sorted_pnl) + 1) / len(sorted_pnl)
            fig.add_trace(
                go.Scatter(
                    x=sorted_pnl,
                    y=cumulative_prob,
                    mode='lines+markers',
                    name='CDF',
                    line=dict(color='blue', width=2),
                    marker=dict(size=2),
                    showlegend=False
                ),
                row=8, col=1
            )
            
            # 16. Q-Q Plot
            try:
                from scipy import stats
                theoretical_quantiles, sample_quantiles = stats.probplot(trades_df[pnl_col], dist="norm")
                fig.add_trace(
                    go.Scatter(
                        x=theoretical_quantiles[0],
                        y=theoretical_quantiles[1],
                        mode='markers',
                        name='Q-Q',
                        marker=dict(color='blue', size=4),
                        showlegend=False
                    ),
                    row=8, col=2
                )
                min_val = min(theoretical_quantiles[0].min(), theoretical_quantiles[1].min())
                max_val = max(theoretical_quantiles[0].max(), theoretical_quantiles[1].max())
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Normal',
                        line=dict(color='red', dash='dash'),
                        showlegend=False
                    ),
                    row=8, col=2
                )
            except:
                pass
            
            # 17. Monthly Heatmap
            if 'exit_time' in trades_df.columns:
                trades_df['year'] = trades_df['exit_time'].dt.year
                monthly_returns = trades_df.groupby(['year', 'month'])['net_pnl'].sum().reset_index()
                heatmap_data = monthly_returns.pivot(index='year', columns='month', values='net_pnl')
                heatmap_data = heatmap_data.fillna(0)
                
                fig.add_trace(
                    go.Heatmap(
                        z=heatmap_data.values,
                        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(heatmap_data.columns)],
                        y=heatmap_data.index,
                        colorscale='RdYlGn',
                        zmid=0,
                        text=heatmap_data.values,
                        texttemplate='$%{text:.0f}',
                        textfont={"size": 9},
                        colorbar=dict(title='P&L', len=0.3, y=0.05),
                        showscale=True
                    ),
                    row=9, col=1
                )
        
        # Update layout
        model_str = f" - {model_name}" if model_name else ""
        fig.update_layout(
            title=dict(
                text=f'Comprehensive Backtest Analysis{model_str}<br><sub>Total Return: {results.get("total_return_pct", 0):.2f}% | Win Rate: {results.get("win_rate", 0):.1f}% | Sharpe: {results.get("sharpe_ratio", 0):.2f}</sub>',
                x=0.5,
                xanchor='center',
                font=dict(size=16)
            ),
            height=3000,  # Tall layout for vertical stacking
            showlegend=True,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Update axes labels (skip row 2 col 2 which is a table)
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_yaxes(title_text="Drawdown (%)", row=1, col=2)
        fig.update_xaxes(title_text="Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        # Row 2, Col 2 is a table - no axes to update
        fig.update_xaxes(title_text="Trade #", row=3, col=1)
        fig.update_yaxes(title_text="Cumulative P&L ($)", row=3, col=1)
        fig.update_xaxes(title_text="P&L ($)", row=3, col=2)
        fig.update_yaxes(title_text="Frequency", row=3, col=2)
        fig.update_yaxes(title_text="Duration (bars)", row=4, col=1)
        fig.update_xaxes(title_text="Month", row=4, col=2)
        fig.update_yaxes(title_text="P&L ($)", row=4, col=2)
        fig.update_xaxes(title_text="Date", row=5, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=5, col=1)
        fig.update_xaxes(title_text="Date", row=5, col=2)
        fig.update_yaxes(title_text="Volatility (%)", row=5, col=2)
        fig.update_xaxes(title_text="Date", row=6, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=6, col=1)
        fig.update_xaxes(title_text="Holding Period (bars)", row=6, col=2)
        fig.update_yaxes(title_text="P&L ($)", row=6, col=2)
        fig.update_xaxes(title_text="P&L ($)", row=7, col=1)
        fig.update_yaxes(title_text="Frequency", row=7, col=1)
        fig.update_yaxes(title_text="P&L ($)", row=7, col=2)
        fig.update_xaxes(title_text="P&L ($)", row=8, col=1)
        fig.update_yaxes(title_text="Cumulative Probability", row=8, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=8, col=2)
        fig.update_yaxes(title_text="Sample Quantiles", row=8, col=2)
        fig.update_xaxes(title_text="Month", row=9, col=1)
        fig.update_yaxes(title_text="Year", row=9, col=1)
        
        # Save HTML
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        model_suffix = f"_{model_name}" if model_name else ""
        filepath = Path(save_dir) / f'comprehensive_analysis{model_suffix}_{timestamp}.html'
        fig.write_html(str(filepath))
        saved_files['comprehensive'] = str(filepath)
        
        print(f"  ✓ Comprehensive analysis saved to: {filepath.name}")
        print(f"  ✓ Total plots: 17 (all in one HTML file)")
        print("="*70)
        print(f"✓ All visualizations saved to: {save_dir}")
        print("="*70)
        
        # Open in browser
        if show_plots:
            webbrowser.open('file://' + os.path.abspath(str(filepath)))
        
        return saved_files
        
        # Extract data from results or self
        equity_curve = results.get('equity_curve', self.equity_curve if hasattr(self, 'equity_curve') else None)
        trades = results.get('trades', self.trades if hasattr(self, 'trades') else None)
        
        # Debug output
        equity_available = False
        trades_available = False
        
        if equity_curve is not None:
            try:
                equity_len = len(equity_curve)
                equity_available = equity_len > 0
                if equity_available:
                    print(f"  Debug - Equity curve points: {equity_len}")
            except:
                equity_available = False
        
        if trades is not None:
            try:
                trades_len = len(trades)
                trades_available = trades_len > 0
                if trades_available:
                    print(f"  Debug - Number of trades: {trades_len}")
                    # Show available columns
                    trades_df_temp = pd.DataFrame(trades)
                    print(f"  Debug - Trades columns: {list(trades_df_temp.columns)}")
            except:
                trades_available = False
        
        print(f"  Debug - Equity curve available: {equity_available}")
        print(f"  Debug - Trades available: {trades_available}")
        
        # 1. Equity Curve
        if equity_curve is not None and equity_available:
            equity_df = pd.DataFrame(equity_curve)
            fig.add_trace(
                go.Scatter(
                    x=equity_df['timestamp'],
                    y=equity_df['equity'],
                    mode='lines',
                    name='Equity',
                    line=dict(color='blue', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(0,100,255,0.2)'
                ),
                row=1, col=1
            )
            
            # Add initial capital line
            initial_capital = results.get('initial_capital', 10000)
            fig.add_hline(
                y=initial_capital,
                line_dash="dash",
                line_color="gray",
                annotation_text="Initial Capital",
                row=1, col=1
            )
        
        # 2. Drawdown (Underwater)
        if equity_curve is not None and equity_available:
            equity_df = pd.DataFrame(equity_curve)
            peak = equity_df['equity'].expanding().max()
            drawdown = ((equity_df['equity'] - peak) / peak * 100)
            
            fig.add_trace(
                go.Scatter(
                    x=equity_df['timestamp'],
                    y=drawdown,
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='red', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(255,0,0,0.2)'
                ),
                row=1, col=2
            )
        
        # 3. Daily Returns Distribution
        if equity_curve is not None and equity_available:
            equity_df = pd.DataFrame(equity_curve)
            returns = equity_df['equity'].pct_change().dropna() * 100
            
            fig.add_trace(
                go.Histogram(
                    x=returns,
                    name='Returns',
                    marker_color='skyblue',
                    nbinsx=50
                ),
                row=1, col=3
            )
            
            # Add mean line
            mean_return = returns.mean()
            fig.add_vline(
                x=mean_return,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_return:.2f}%",
                row=1, col=3
            )
        
        # 4. Cumulative P&L by Trade
        if trades is not None and trades_available:
            trades_df = pd.DataFrame(trades)
            cumulative_pnl = trades_df['net_pnl'].cumsum()
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(cumulative_pnl))),
                    y=cumulative_pnl,
                    mode='lines+markers',
                    name='Cumulative P&L',
                    line=dict(color='blue', width=2),
                    marker=dict(size=4)
                ),
                row=2, col=1
            )
        
        # 5. Win/Loss Distribution (Histogram)
        if trades is not None and trades_available:
            trades_df = pd.DataFrame(trades)
            wins = trades_df[trades_df['net_pnl'] > 0]['net_pnl']
            losses = trades_df[trades_df['net_pnl'] <= 0]['net_pnl']
            
            fig.add_trace(
                go.Histogram(
                    x=wins,
                    name='Wins',
                    marker_color='green',
                    opacity=0.7,
                    nbinsx=20
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Histogram(
                    x=losses,
                    name='Losses',
                    marker_color='red',
                    opacity=0.7,
                    nbinsx=20
                ),
                row=2, col=2
            )
        
        # 6. P&L vs Holding Period (Scatter)
        if trades is not None and trades_available:
            trades_df = pd.DataFrame(trades)
            
            # Calculate duration_bars if not present
            if 'duration_bars' not in trades_df.columns:
                if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
                    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
                    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
                    # Estimate bars (assuming 1 bar = 1 hour for hourly data)
                    trades_df['duration_bars'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 3600
                elif 'bars_held' in trades_df.columns:
                    trades_df['duration_bars'] = trades_df['bars_held']
            
            if 'duration_bars' in trades_df.columns and len(trades_df['duration_bars'].dropna()) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=trades_df['duration_bars'],
                        y=trades_df['net_pnl'],
                        mode='markers',
                        name='Trades',
                        marker=dict(
                            color=trades_df['net_pnl'],
                            colorscale='RdYlGn',
                            size=8,
                            opacity=0.6,
                            colorbar=dict(title='P&L')
                        )
                    ),
                    row=2, col=3
                )
            else:
                print(f"  ⚠ Skipping P&L vs Holding Period plot - duration data not available")
        
        # 7. Monthly P&L (Bar Chart)
        if trades is not None and trades_available:
            trades_df = pd.DataFrame(trades)
            if 'exit_time' in trades_df.columns:
                trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
                trades_df['month'] = trades_df['exit_time'].dt.month
                monthly_pnl = trades_df.groupby('month')['net_pnl'].sum()
                
                colors = ['green' if x > 0 else 'red' for x in monthly_pnl.values]
                fig.add_trace(
                    go.Bar(
                        x=monthly_pnl.index,
                        y=monthly_pnl.values,
                        marker_color=colors,
                        name='Monthly P&L',
                        text=[f'${x:.0f}' for x in monthly_pnl.values],
                        textposition='auto'
                    ),
                    row=3, col=1
                )
        
        # 8. Trade Duration Box Plot
        if trades is not None and trades_available:
            trades_df = pd.DataFrame(trades)
            
            # Calculate duration_bars if not present (same logic as above)
            if 'duration_bars' not in trades_df.columns:
                if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
                    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
                    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
                    trades_df['duration_bars'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 3600
                elif 'bars_held' in trades_df.columns:
                    trades_df['duration_bars'] = trades_df['bars_held']
            
            if 'duration_bars' in trades_df.columns and len(trades_df['duration_bars'].dropna()) > 0:
                fig.add_trace(
                    go.Box(
                        y=trades_df['duration_bars'],
                        name='Duration',
                        marker_color='orange',
                        boxmean='sd'
                    ),
                    row=3, col=2
                )
            else:
                print(f"  ⚠ Skipping Trade Duration Box Plot - duration data not available")
        
        # 9. Rolling Sharpe Ratio
        if equity_curve is not None and equity_available:
            equity_df = pd.DataFrame(equity_curve)
            equity_df['returns'] = equity_df['equity'].pct_change()
            window = min(30, len(equity_df) // 4)
            
            if window > 1:
                rolling_sharpe = (equity_df['returns'].rolling(window).mean() / 
                                equity_df['returns'].rolling(window).std() * np.sqrt(252))
                
                fig.add_trace(
                    go.Scatter(
                        x=equity_df['timestamp'],
                        y=rolling_sharpe,
                        mode='lines',
                        name=f'Sharpe ({window}d)',
                        line=dict(color='purple', width=2)
                    ),
                    row=3, col=3
                )
        
        # 10. Rolling Volatility
        if equity_curve is not None and equity_available:
            equity_df = pd.DataFrame(equity_curve)
            if 'returns' not in equity_df.columns:
                equity_df['returns'] = equity_df['equity'].pct_change()
            window = min(30, len(equity_df) // 4)
            
            if window > 1:
                rolling_vol = equity_df['returns'].rolling(window).std() * np.sqrt(252) * 100
                
                fig.add_trace(
                    go.Scatter(
                        x=equity_df['timestamp'],
                        y=rolling_vol,
                        mode='lines',
                        name=f'Volatility ({window}d)',
                        line=dict(color='orange', width=2)
                    ),
                    row=4, col=1
                )
        
        # 11. Trade P&L Distribution (Overall)
        if trades is not None and trades_available:
            trades_df = pd.DataFrame(trades)
            fig.add_trace(
                go.Histogram(
                    x=trades_df['net_pnl'],
                    name='P&L Distribution',
                    marker_color='steelblue',
                    nbinsx=30
                ),
                row=4, col=2
            )
            
            # Add vertical line at zero
            fig.add_vline(
                x=0,
                line_dash="dash",
                line_color="black",
                row=4, col=2
            )
        
        # 12. Monthly Returns Heatmap
        if trades is not None and trades_available:
            trades_df = pd.DataFrame(trades)
            if 'exit_time' in trades_df.columns:
                trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
                trades_df['year'] = trades_df['exit_time'].dt.year
                trades_df['month'] = trades_df['exit_time'].dt.month
                
                # Group by year and month
                monthly_returns = trades_df.groupby(['year', 'month'])['net_pnl'].sum().reset_index()
                
                # Pivot for heatmap
                heatmap_data = monthly_returns.pivot(index='year', columns='month', values='net_pnl')
                heatmap_data = heatmap_data.fillna(0)
                
                fig.add_trace(
                    go.Heatmap(
                        z=heatmap_data.values,
                        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(heatmap_data.columns)],
                        y=heatmap_data.index,
                        colorscale='RdYlGn',
                        zmid=0,
                        name='Monthly Returns',
                        colorbar=dict(title='P&L')
                    ),
                    row=4, col=3
                )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'Comprehensive Backtest Analysis ({model_name})<br><sub>Total Return: {results.get("total_return_pct", 0):.2f}% | Win Rate: {results.get("win_rate", 0):.1f}% | Sharpe: {results.get("sharpe_ratio", 0):.2f}</sub>',
                x=0.5,
                xanchor='center'
            ),
            height=1600,
            showlegend=True,
            template='plotly_white'
        )
        
        # Update axes labels for all subplots
        # Row 1
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_yaxes(title_text="Drawdown (%)", row=1, col=2)
        fig.update_xaxes(title_text="Daily Return (%)", row=1, col=3)
        fig.update_yaxes(title_text="Frequency", row=1, col=3)
        
        # Row 2
        fig.update_xaxes(title_text="Trade Number", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative P&L ($)", row=2, col=1)
        fig.update_xaxes(title_text="P&L ($)", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        fig.update_xaxes(title_text="Holding Period (bars)", row=2, col=3)
        fig.update_yaxes(title_text="P&L ($)", row=2, col=3)
        
        # Row 3
        fig.update_xaxes(title_text="Month", row=3, col=1)
        fig.update_yaxes(title_text="P&L ($)", row=3, col=1)
        fig.update_yaxes(title_text="Duration (bars)", row=3, col=2)
        fig.update_xaxes(title_text="Date", row=3, col=3)
        fig.update_yaxes(title_text="Sharpe Ratio", row=3, col=3)
        
        # Row 4
        fig.update_xaxes(title_text="Date", row=4, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=4, col=1)
        fig.update_xaxes(title_text="P&L ($)", row=4, col=2)
        fig.update_yaxes(title_text="Frequency", row=4, col=2)
        fig.update_xaxes(title_text="Month", row=4, col=3)
        fig.update_yaxes(title_text="Year", row=4, col=3)
        
        # Save HTML
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        model_name = results.get('model_name', 'backtest')
        filepath = Path(save_dir) / f'comprehensive_analysis_{model_name}_{timestamp}.html'
        fig.write_html(str(filepath))
        saved_files['comprehensive'] = str(filepath)
        
        print(f"  ✓ Comprehensive analysis saved to: {filepath}")
        
        # Open in browser
        if show_plots:
            webbrowser.open('file://' + os.path.abspath(str(filepath)))
        
        return saved_files
    
    def _create_comprehensive_visualizations_matplotlib(
        self,
        results: Dict,
        df: pd.DataFrame = None,
        save_dir: Optional[str] = None,
        show_plots: bool = True
    ) -> Dict[str, str]:
        """
        Fallback matplotlib version of comprehensive visualizations.
        """
        saved_files = {}
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. Performance Overview
        saved_files['performance_overview'] = self._plot_performance_overview(
            results, df, save_dir, show_plots
        )
        
        # 2. Trade Analysis
        if self.trades:
            saved_files['trade_analysis'] = self._plot_trade_analysis(
                results, save_dir, show_plots
            )
        
        # 3. Risk Analysis
        if self.equity_curve:
            saved_files['risk_analysis'] = self._plot_risk_analysis(
                results, save_dir, show_plots
            )
        
        # 4. Monthly Performance Heatmap
        if self.trades:
            saved_files['monthly_heatmap'] = self._plot_monthly_heatmap(
                results, save_dir, show_plots
            )
        
        # 5. Trade Distribution
        if self.trades:
            saved_files['trade_distribution'] = self._plot_trade_distribution(
                results, save_dir, show_plots
            )
        
        return saved_files
    
    def _plot_performance_overview(
        self, 
        results: Dict, 
        df: pd.DataFrame = None,
        save_dir: Optional[str] = None,
        show_plots: bool = True
    ) -> Optional[str]:
        """Create performance overview plot."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance Overview', fontsize=16, fontweight='bold')
        
        # Equity curve
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            ax1 = axes[0, 0]
            ax1.plot(equity_df['timestamp'], equity_df['equity'], 
                    label='Portfolio Value', linewidth=2, color='blue')
            ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', 
                       label='Initial Capital', alpha=0.7)
            
            # Add benchmark if available
            if df is not None and 'close' in df.columns:
                benchmark_value = self.initial_capital * (df['close'] / df['close'].iloc[0])
                ax1.plot(df.index if hasattr(df, 'index') else range(len(df)), 
                        benchmark_value, label='Buy & Hold', alpha=0.7, color='orange')
            
            ax1.set_title('Equity Curve')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Drawdown
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df['peak'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
            
            ax2 = axes[0, 1]
            ax2.fill_between(equity_df['timestamp'], 0, equity_df['drawdown'], 
                           color='red', alpha=0.3)
            ax2.plot(equity_df['timestamp'], equity_df['drawdown'], color='red', linewidth=1)
            ax2.set_title('Drawdown')
            ax2.set_ylabel('Drawdown (%)')
            ax2.grid(True, alpha=0.3)
        
        # Returns distribution
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            returns = equity_df['equity'].pct_change().dropna()
            
            ax3 = axes[1, 0]
            ax3.hist(returns * 100, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.axvline(returns.mean() * 100, color='red', linestyle='--', 
                       label=f'Mean: {returns.mean()*100:.2f}%')
            ax3.set_title('Daily Returns Distribution')
            ax3.set_xlabel('Daily Return (%)')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Performance metrics summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        metrics_text = f"""
        Performance Summary
        
        Total Return: {results.get('total_return_pct', 0):.2f}%
        Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}
        Max Drawdown: {results.get('max_drawdown', 0):.2f}%
        Win Rate: {results.get('win_rate', 0):.2f}%
        Total Trades: {results.get('total_trades', 0)}
        Profit Factor: {results.get('profit_factor', 0):.2f}
        """
        
        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray'))
        
        plt.tight_layout()
        
        file_path = None
        if save_dir:
            file_path = Path(save_dir) / 'performance_overview.png'
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        return str(file_path) if file_path else None
    
    def _plot_trade_analysis(
        self, 
        results: Dict,
        save_dir: Optional[str] = None,
        show_plots: bool = True
    ) -> Optional[str]:
        """Create trade analysis plots."""
        if not self.trades:
            return None
        
        trades_df = pd.DataFrame(self.trades)
        pnl_col = 'net_pnl' if 'net_pnl' in trades_df.columns else 'pnl'
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Trade Analysis', fontsize=16, fontweight='bold')
        
        # Trade P&L over time
        ax1 = axes[0, 0]
        cumulative_pnl = trades_df[pnl_col].cumsum()
        ax1.plot(range(len(cumulative_pnl)), cumulative_pnl, marker='o', linewidth=2)
        ax1.set_title('Cumulative P&L by Trade')
        ax1.set_xlabel('Trade Number')
        ax1.set_ylabel('Cumulative P&L ($)')
        ax1.grid(True, alpha=0.3)
        
        # Win/Loss distribution
        ax2 = axes[0, 1]
        wins = trades_df[trades_df[pnl_col] > 0][pnl_col]
        losses = trades_df[trades_df[pnl_col] <= 0][pnl_col]
        
        ax2.hist([wins, losses], bins=20, label=['Wins', 'Losses'], 
                color=['green', 'red'], alpha=0.7)
        ax2.set_title('Win/Loss Distribution')
        ax2.set_xlabel('P&L ($)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Trade duration analysis
        if 'bars_held' in trades_df.columns:
            ax3 = axes[1, 0]
            ax3.scatter(trades_df['bars_held'], trades_df[pnl_col], alpha=0.6)
            ax3.set_title('P&L vs Holding Period')
            ax3.set_xlabel('Bars Held')
            ax3.set_ylabel('P&L ($)')
            ax3.grid(True, alpha=0.3)
        
        # Monthly performance
        if 'entry_time' in trades_df.columns:
            ax4 = axes[1, 1]
            trades_df['month'] = pd.to_datetime(trades_df['entry_time']).dt.month
            monthly_pnl = trades_df.groupby('month')[pnl_col].sum()
            
            bars = ax4.bar(monthly_pnl.index, monthly_pnl.values, 
                          color=['green' if x > 0 else 'red' for x in monthly_pnl.values])
            ax4.set_title('Monthly P&L')
            ax4.set_xlabel('Month')
            ax4.set_ylabel('P&L ($)')
            ax4.set_xticks(range(1, 13))
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        file_path = None
        if save_dir:
            file_path = Path(save_dir) / 'trade_analysis.png'
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        return str(file_path) if file_path else None
    
    def _plot_risk_analysis(
        self, 
        results: Dict,
        save_dir: Optional[str] = None,
        show_plots: bool = True
    ) -> Optional[str]:
        """Create risk analysis plots."""
        if not self.equity_curve:
            return None
        
        equity_df = pd.DataFrame(self.equity_curve)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Risk Analysis', fontsize=16, fontweight='bold')
        
        # Rolling Sharpe ratio
        equity_df['returns'] = equity_df['equity'].pct_change()
        window = min(30, len(equity_df) // 4)  # 30-day or 1/4 of data
        
        if window > 1:
            ax1 = axes[0, 0]
            rolling_sharpe = equity_df['returns'].rolling(window).mean() / equity_df['returns'].rolling(window).std() * np.sqrt(252)
            ax1.plot(equity_df['timestamp'], rolling_sharpe)
            ax1.set_title(f'Rolling Sharpe Ratio ({window}-period)')
            ax1.set_ylabel('Sharpe Ratio')
            ax1.grid(True, alpha=0.3)
        
        # Volatility analysis
        ax2 = axes[0, 1]
        rolling_vol = equity_df['returns'].rolling(window).std() * np.sqrt(252) * 100
        ax2.plot(equity_df['timestamp'], rolling_vol, color='orange')
        ax2.set_title(f'Rolling Volatility ({window}-period)')
        ax2.set_ylabel('Volatility (%)')
        ax2.grid(True, alpha=0.3)
        
        # Underwater curve (drawdown)
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
        
        ax3 = axes[1, 0]
        ax3.fill_between(equity_df['timestamp'], 0, equity_df['drawdown'], 
                        color='red', alpha=0.3)
        ax3.plot(equity_df['timestamp'], equity_df['drawdown'], color='red')
        ax3.set_title('Underwater Curve')
        ax3.set_ylabel('Drawdown (%)')
        ax3.grid(True, alpha=0.3)
        
        # Risk-Return scatter
        ax4 = axes[1, 1]
        if len(equity_df) > 1:
            total_return = (equity_df['equity'].iloc[-1] / equity_df['equity'].iloc[0] - 1) * 100
            volatility = equity_df['returns'].std() * np.sqrt(252) * 100
            
            ax4.scatter(volatility, total_return, s=100, color='blue')
            ax4.set_xlabel('Volatility (%)')
            ax4.set_ylabel('Total Return (%)')
            ax4.set_title('Risk-Return Profile')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        file_path = None
        if save_dir:
            file_path = Path(save_dir) / 'risk_analysis.png'
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        return str(file_path) if file_path else None
    
    def _plot_monthly_heatmap(
        self, 
        results: Dict,
        save_dir: Optional[str] = None,
        show_plots: bool = True
    ) -> Optional[str]:
        """Create monthly performance heatmap."""
        if not self.trades:
            return None
        
        trades_df = pd.DataFrame(self.trades)
        if 'entry_time' not in trades_df.columns:
            return None
        
        pnl_col = 'net_pnl' if 'net_pnl' in trades_df.columns else 'pnl'
        
        # Create monthly performance matrix
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        trades_df['year'] = trades_df['entry_time'].dt.year
        trades_df['month'] = trades_df['entry_time'].dt.month
        
        monthly_returns = trades_df.groupby(['year', 'month'])[pnl_col].sum().reset_index()
        
        if len(monthly_returns) == 0:
            return None
        
        # Pivot for heatmap
        heatmap_data = monthly_returns.pivot(index='year', columns='month', values=pnl_col)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='RdYlGn', 
                   center=0, ax=ax, cbar_kws={'label': 'P&L ($)'})
        
        ax.set_title('Monthly Performance Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        
        # Set month labels - only for months that exist in the data
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        # Get the actual months present in the data
        actual_months = sorted(heatmap_data.columns)
        actual_month_labels = [month_labels[m-1] for m in actual_months]
        ax.set_xticklabels(actual_month_labels)
        
        plt.tight_layout()
        
        file_path = None
        if save_dir:
            file_path = Path(save_dir) / 'monthly_heatmap.png'
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        return str(file_path) if file_path else None
    
    def _plot_trade_distribution(
        self, 
        results: Dict,
        save_dir: Optional[str] = None,
        show_plots: bool = True
    ) -> Optional[str]:
        """Create trade distribution analysis."""
        if not self.trades:
            return None
        
        trades_df = pd.DataFrame(self.trades)
        pnl_col = 'net_pnl' if 'net_pnl' in trades_df.columns else 'pnl'
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Trade Distribution Analysis', fontsize=16, fontweight='bold')
        
        # P&L distribution
        ax1 = axes[0, 0]
        ax1.hist(trades_df[pnl_col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(trades_df[pnl_col].mean(), color='red', linestyle='--', 
                   label=f'Mean: ${trades_df[pnl_col].mean():.2f}')
        ax1.set_title('P&L Distribution')
        ax1.set_xlabel('P&L ($)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot of wins vs losses
        ax2 = axes[0, 1]
        wins = trades_df[trades_df[pnl_col] > 0][pnl_col]
        losses = trades_df[trades_df[pnl_col] <= 0][pnl_col]
        
        box_data = [wins.values, losses.values]
        ax2.boxplot(box_data, labels=['Wins', 'Losses'])
        ax2.set_title('Win/Loss Box Plot')
        ax2.set_ylabel('P&L ($)')
        ax2.grid(True, alpha=0.3)
        
        # Cumulative distribution
        ax3 = axes[1, 0]
        sorted_pnl = np.sort(trades_df[pnl_col])
        cumulative_prob = np.arange(1, len(sorted_pnl) + 1) / len(sorted_pnl)
        ax3.plot(sorted_pnl, cumulative_prob, marker='o', markersize=3)
        ax3.set_title('Cumulative Distribution Function')
        ax3.set_xlabel('P&L ($)')
        ax3.set_ylabel('Cumulative Probability')
        ax3.grid(True, alpha=0.3)
        
        # Q-Q plot (normal distribution)
        ax4 = axes[1, 1]
        from scipy import stats
        stats.probplot(trades_df[pnl_col], dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot (Normal Distribution)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        file_path = None
        if save_dir:
            file_path = Path(save_dir) / 'trade_distribution.png'
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        return str(file_path) if file_path else None
    
    def export_results(
        self, 
        results: Dict, 
        export_dir: str,
        include_trades: bool = True,
        include_equity_curve: bool = True,
        include_report: bool = True
    ) -> Dict[str, str]:
        """
        Export all results to files.
        
        Parameters:
        -----------
        results : Dict
            Backtest results
        export_dir : str
            Directory to export files
        include_trades : bool
            Whether to export trades data
        include_equity_curve : bool
            Whether to export equity curve data
        include_report : bool
            Whether to export text report
            
        Returns:
        --------
        exported_files : Dict[str, str]
            Dictionary of exported file types and paths
        """
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        # Export trades
        if include_trades and self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_file = export_path / 'trades.csv'
            trades_df.to_csv(trades_file, index=False)
            exported_files['trades'] = str(trades_file)
        
        # Export equity curve
        if include_equity_curve and self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_file = export_path / 'equity_curve.csv'
            equity_df.to_csv(equity_file, index=False)
            exported_files['equity_curve'] = str(equity_file)
        
        # Export results as JSON
        results_file = export_path / 'results.json'
        # Convert numpy types to native Python types for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, (np.integer, np.floating)):
                json_results[key] = value.item()
            elif isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif pd.isna(value):
                json_results[key] = None
            else:
                json_results[key] = value
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        exported_files['results'] = str(results_file)
        
        # Export text report
        if include_report:
            report_file = export_path / 'report.txt'
            report_text = self.generate_detailed_report(results)
            with open(report_file, 'w') as f:
                f.write(report_text)
            exported_files['report'] = str(report_file)
        
        return exported_files
    
    def print_summary(self, results: Dict):
        """
        Print a concise summary of results.
        
        Parameters:
        -----------
        results : Dict
            Backtest results
        """
        print("\n" + "="*60)
        print("BACKTEST SUMMARY")
        print("="*60)
        print(f"Total Return:     {results.get('total_return_pct', 0):>8.2f}%")
        print(f"Sharpe Ratio:     {results.get('sharpe_ratio', 0):>8.2f}")
        print(f"Max Drawdown:     {results.get('max_drawdown', 0):>8.2f}%")
        print(f"Win Rate:         {results.get('win_rate', 0):>8.2f}%")
        print(f"Total Trades:     {results.get('total_trades', 0):>8}")
        print(f"Profit Factor:    {results.get('profit_factor', 0):>8.2f}")
        print("="*60)
