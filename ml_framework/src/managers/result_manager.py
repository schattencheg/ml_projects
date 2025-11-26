"""
ResultManager - Receives and processes results from TrainManager/TestManager/BacktestManager.

Supports both script and Jupyter notebook usage with display methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
from datetime import datetime


def _is_jupyter() -> bool:
    """Detect if running in Jupyter notebook/lab environment."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        shell_name = shell.__class__.__name__
        if 'ZMQInteractiveShell' in shell_name:
            return True
        return False
    except (ImportError, NameError):
        return False


class ResultManager:
    """
    Manages and processes results from various managers.
    
    Features:
    - Collect results from TrainManager, TestManager, BacktestManager
    - Aggregate and compare results
    - Generate comprehensive reports
    - Save results to structured format
    """
    
    def __init__(self):
        """Initialize ResultManager."""
        self.train_results = {}
        self.test_results = {}
        self.backtest_results = {}
        self.metadata = {}
        
    def add_train_results(self, results: Dict[str, Any], metadata: Optional[Dict] = None):
        """
        Add training results.
        
        Args:
            results: Training results dictionary
            metadata: Additional metadata
        """
        self.train_results = results
        if metadata:
            self.metadata.update({'train': metadata})
        print(f"✓ Added training results for {len(results)} models")
    
    def add_test_results(self, results: Dict[str, Any], metadata: Optional[Dict] = None):
        """
        Add test results.
        
        Args:
            results: Test results dictionary
            metadata: Additional metadata
        """
        self.test_results = results
        if metadata:
            self.metadata.update({'test': metadata})
        print(f"✓ Added test results for {len(results)} models")
    
    def add_backtest_results(self, 
                            model_name: str, 
                            results: Dict[str, Any], 
                            metadata: Optional[Dict] = None):
        """
        Add backtest results for a model.
        
        Args:
            model_name: Name of the model
            results: Backtest results dictionary
            metadata: Additional metadata
        """
        self.backtest_results[model_name] = results
        if metadata:
            if 'backtest' not in self.metadata:
                self.metadata['backtest'] = {}
            self.metadata['backtest'][model_name] = metadata
        print(f"✓ Added backtest results for {model_name}")
    
    def get_comprehensive_results(self) -> Dict[str, Any]:
        """
        Get comprehensive results combining all sources.
        
        Returns:
            Dictionary with all results
        """
        return {
            'train': self.train_results,
            'test': self.test_results,
            'backtest': self.backtest_results,
            'metadata': self.metadata
        }
    
    def get_best_model(self, metric: str = 'f1_score', phase: str = 'test') -> Optional[str]:
        """
        Get the best performing model.
        
        Args:
            metric: Metric to use for comparison
            phase: Phase to evaluate ('train' or 'test')
            
        Returns:
            Name of the best model
        """
        if phase == 'train':
            results = self.train_results
            metric = 'train_accuracy' if metric == 'f1_score' else metric
        elif phase == 'test':
            results = self.test_results
        else:
            raise ValueError(f"Unknown phase: {phase}")
        
        if not results:
            return None
        
        # Filter successful results
        valid_results = {name: res for name, res in results.items() 
                        if res.get('status') == 'success' and metric in res}
        
        if not valid_results:
            return None
        
        best_model = max(valid_results.items(), key=lambda x: x[1][metric])[0]
        return best_model
    
    def compare_models(self, phase: str = 'test') -> pd.DataFrame:
        """
        Compare all models across metrics.
        
        Args:
            phase: Phase to compare ('train' or 'test')
            
        Returns:
            DataFrame with model comparison
        """
        if phase == 'train':
            results = self.train_results
            metrics = ['train_accuracy', 'val_accuracy', 'training_time']
        elif phase == 'test':
            results = self.test_results
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        else:
            raise ValueError(f"Unknown phase: {phase}")
        
        if not results:
            return pd.DataFrame()
        
        # Build comparison data
        comparison_data = []
        for model_name, res in results.items():
            if res.get('status') != 'success':
                continue
            
            row = {'model': model_name}
            for metric in metrics:
                row[metric] = res.get(metric, np.nan)
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by primary metric
        if phase == 'test' and 'f1_score' in df.columns:
            df = df.sort_values('f1_score', ascending=False)
        elif phase == 'train' and 'val_accuracy' in df.columns:
            df = df.sort_values('val_accuracy', ascending=False)
        
        return df
    
    def compare_backtests(self) -> pd.DataFrame:
        """
        Compare backtest results across models.
        
        Returns:
            DataFrame with backtest comparison
        """
        if not self.backtest_results:
            return pd.DataFrame()
        
        metrics = ['total_return', 'annualized_return', 'sharpe_ratio', 
                  'max_drawdown', 'win_rate', 'num_trades']
        
        comparison_data = []
        for model_name, results in self.backtest_results.items():
            row = {'model': model_name}
            for metric in metrics:
                row[metric] = results.get(metric, np.nan)
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('sharpe_ratio', ascending=False)
        
        return df
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of all results.
        
        Returns:
            Summary dictionary
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'num_models_trained': len(self.train_results),
            'num_models_tested': len(self.test_results),
            'num_models_backtested': len(self.backtest_results)
        }
        
        # Best models
        best_test_model = self.get_best_model('f1_score', 'test')
        if best_test_model:
            summary['best_test_model'] = best_test_model
            summary['best_test_f1'] = self.test_results[best_test_model].get('f1_score')
        
        # Best backtest
        if self.backtest_results:
            best_backtest = max(self.backtest_results.items(), 
                              key=lambda x: x[1].get('sharpe_ratio', -np.inf))
            summary['best_backtest_model'] = best_backtest[0]
            summary['best_backtest_sharpe'] = best_backtest[1].get('sharpe_ratio')
        
        return summary
    
    def save_results(self, save_dir: Path, format: str = 'json'):
        """
        Save results to files.
        
        Args:
            save_dir: Directory to save results
            format: Format to save ('json' or 'csv')
        """
        save_dir = Path(save_dir)
        reports_dir = save_dir / 'reports'
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"SAVING RESULTS TO: {reports_dir}")
        print(f"{'='*70}")
        
        # Save train results
        if self.train_results:
            train_dir = reports_dir / 'train'
            train_dir.mkdir(exist_ok=True)
            self._save_phase_results(self.train_results, train_dir, 'train', format)
        
        # Save test results
        if self.test_results:
            test_dir = reports_dir / 'test'
            test_dir.mkdir(exist_ok=True)
            self._save_phase_results(self.test_results, test_dir, 'test', format)
        
        # Save backtest results
        if self.backtest_results:
            backtest_dir = reports_dir / 'backtest'
            backtest_dir.mkdir(exist_ok=True)
            self._save_backtest_results(self.backtest_results, backtest_dir, format)
        
        # Save summary
        summary = self.generate_summary()
        summary_path = reports_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"✓ Saved summary")
        
        # Save metadata
        metadata_path = save_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        print(f"✓ Saved metadata")
        
        print(f"{'='*70}\n")
    
    def _save_phase_results(self, 
                           results: Dict[str, Any], 
                           save_dir: Path, 
                           phase: str,
                           format: str):
        """Save results for a specific phase."""
        # Save comparison table
        df_comparison = self.compare_models(phase)
        if not df_comparison.empty:
            if format == 'csv':
                comparison_path = save_dir / f'{phase}_comparison.csv'
                df_comparison.to_csv(comparison_path, index=False)
            else:
                comparison_path = save_dir / f'{phase}_comparison.json'
                df_comparison.to_json(comparison_path, orient='records', indent=2)
            print(f"✓ Saved {phase} comparison")
        
        # Save detailed results
        results_path = save_dir / f'{phase}_results.json'
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, res in results.items():
            serializable_results[model_name] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in res.items()
                if k not in ['predictions', 'confusion_matrix']  # Skip large arrays
            }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        print(f"✓ Saved {phase} detailed results")
    
    def _save_backtest_results(self, 
                               results: Dict[str, Any], 
                               save_dir: Path,
                               format: str):
        """Save backtest results."""
        # Save comparison table
        df_comparison = self.compare_backtests()
        if not df_comparison.empty:
            if format == 'csv':
                comparison_path = save_dir / 'backtest_comparison.csv'
                df_comparison.to_csv(comparison_path, index=False)
            else:
                comparison_path = save_dir / 'backtest_comparison.json'
                df_comparison.to_json(comparison_path, orient='records', indent=2)
            print(f"✓ Saved backtest comparison")
        
        # Save detailed results for each model
        for model_name, res in results.items():
            model_results = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in res.items()
                if k not in ['equity_curve', 'signals', 'strategy_returns']  # Skip large arrays
            }
            
            results_path = save_dir / f'{model_name}_backtest.json'
            with open(results_path, 'w') as f:
                json.dump(model_results, f, indent=2, default=str)
        
        print(f"✓ Saved backtest detailed results")
    
    def prepare_backtest_visualization_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Prepare backtest results with additional data for visualization.
        
        Args:
            df: DataFrame with OHLC data used in backtesting
            
        Returns:
            Dictionary with backtest results enhanced with visualization data
        """
        viz_data = {}
        
        for model_name, results in self.backtest_results.items():
            viz_data[model_name] = results.copy()
            
            # Normalize trades to list of dictionaries
            if 'trades' not in viz_data[model_name]:
                viz_data[model_name]['trades'] = []
            else:
                trades = viz_data[model_name]['trades']
                
                # Convert DataFrame to list of dictionaries
                if isinstance(trades, pd.DataFrame):
                    # Convert DataFrame to list of dicts
                    trades_list = []
                    for idx, row in trades.iterrows():
                        trade_dict = {
                            'entry_idx': int(row.get('EntryBar', idx)) if 'EntryBar' in row else idx,
                            'exit_idx': int(row.get('ExitBar', idx)) if 'ExitBar' in row else idx,
                            'entry_price': float(row.get('EntryPrice', 0)) if 'EntryPrice' in row else 0.0,
                            'exit_price': float(row.get('ExitPrice', 0)) if 'ExitPrice' in row else 0.0,
                            'shares': float(row.get('Size', 0)) if 'Size' in row else 0.0,
                            'pnl': float(row.get('PnL', 0)) if 'PnL' in row else 0.0,
                            'exit_reason': str(row.get('ExitReason', 'signal')) if 'ExitReason' in row else 'signal'
                        }
                        trades_list.append(trade_dict)
                    viz_data[model_name]['trades'] = trades_list
                elif not isinstance(trades, list):
                    # If it's not a list or DataFrame, try to convert to list
                    viz_data[model_name]['trades'] = list(trades) if hasattr(trades, '__iter__') else []
            
            # Add OHLC data reference
            viz_data[model_name]['has_ohlc_data'] = all(
                col in df.columns for col in ['open', 'high', 'low', 'close']
            )
            
            # Ensure equity curve is array-like
            if 'equity_curve' in viz_data[model_name]:
                equity = viz_data[model_name]['equity_curve']
                if not isinstance(equity, (list, np.ndarray)):
                    viz_data[model_name]['equity_curve'] = [equity]
            
            # Add initial capital if not present
            if 'initial_capital' not in viz_data[model_name]:
                if 'metrics' in viz_data[model_name]:
                    final_capital = viz_data[model_name]['metrics'].get('final_capital', 10000)
                    total_return = viz_data[model_name]['metrics'].get('total_return', 0)
                    if total_return != 0:
                        viz_data[model_name]['initial_capital'] = final_capital / (1 + total_return)
                    else:
                        viz_data[model_name]['initial_capital'] = 10000
                else:
                    viz_data[model_name]['initial_capital'] = 10000
        
        return viz_data
    
    def print_summary(self):
        """Print a summary of all results."""
        summary = self.generate_summary()
        
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        
        print("\nOverview:")
        num_models_trained = summary['num_models_trained'] if 'num_models_trained' in summary else -1
        print("  Models trained:    " + f'{num_models_trained}')        
        num_models_tested = summary['num_models_tested'] if 'num_models_tested' in summary else -1
        print("  Models tested:     " + f'{num_models_tested}')        
        num_models_backtested = summary['num_models_backtested'] if 'num_models_backtested' in summary else -1
        print("  Models backtested: " + f'{num_models_backtested}')
        
        if 'best_test_model' in summary:
            print(f"\nBest Test Model:")
            model = summary['best_test_model'] if 'best_test_model' in summary else 'N/A'
            print("  Model:        " + f'{model}')
            f1 = summary['best_test_f1'] if 'best_test_f1' in summary else -0.0
            print("  F1 Score:     " + f'{f1:.4f}')
        
        if 'best_backtest_model' in summary:
            print(f"\nBest Backtest Model:")
            model = summary['best_backtest_model'] if 'best_backtest_model' in summary else 'N/A'
            print("  Model:        " + f'{model}')
            sharpe = summary['best_backtest_sharpe'] if 'best_backtest_sharpe' in summary else -0.0
            sharpe = sharpe if sharpe is not None else -0.0
            print("  Sharpe Ratio: " + f'{sharpe:.4f}')
        
        print("="*70 + "\n")
    
    # =========================================================================
    # JUPYTER-FRIENDLY DISPLAY METHODS
    # =========================================================================
    
    def display_train_comparison(self) -> pd.DataFrame:
        """
        Display training comparison table (Jupyter-friendly).
        
        Returns:
            DataFrame with training comparison (also displays in Jupyter)
        """
        df = self.compare_models('train')
        if _is_jupyter() and not df.empty:
            from IPython.display import display
            display(df.style.highlight_max(axis=0, color='lightgreen'))
        return df
    
    def display_test_comparison(self) -> pd.DataFrame:
        """
        Display test comparison table (Jupyter-friendly).
        
        Returns:
            DataFrame with test comparison (also displays in Jupyter)
        """
        df = self.compare_models('test')
        if _is_jupyter() and not df.empty:
            from IPython.display import display
            display(df.style.highlight_max(axis=0, color='lightgreen'))
        return df
    
    def display_backtest_comparison(self) -> pd.DataFrame:
        """
        Display backtest comparison table (Jupyter-friendly).
        
        Returns:
            DataFrame with backtest comparison (also displays in Jupyter)
        """
        df = self.compare_backtests()
        if _is_jupyter() and not df.empty:
            from IPython.display import display
            # Highlight best values (max for returns/sharpe, min for drawdown)
            styled = df.style
            if 'total_return' in df.columns:
                styled = styled.highlight_max(subset=['total_return'], color='lightgreen')
            if 'sharpe_ratio' in df.columns:
                styled = styled.highlight_max(subset=['sharpe_ratio'], color='lightgreen')
            if 'max_drawdown' in df.columns:
                styled = styled.highlight_min(subset=['max_drawdown'], color='lightgreen')
            if 'win_rate' in df.columns:
                styled = styled.highlight_max(subset=['win_rate'], color='lightgreen')
            display(styled)
        return df
    
    def display_summary(self) -> Dict[str, Any]:
        """
        Display results summary (Jupyter-friendly).
        
        Returns:
            Summary dictionary (also displays formatted in Jupyter)
        """
        summary = self.generate_summary()
        
        if _is_jupyter():
            from IPython.display import display, HTML
            
            html = "<div style='font-family: Arial; padding: 10px;'>"
            html += "<h3>📊 Results Summary</h3>"
            html += "<table style='border-collapse: collapse;'>"
            
            for key, value in summary.items():
                if isinstance(value, float):
                    value_str = f"{value:.4f}"
                else:
                    value_str = str(value)
                html += f"<tr><td style='padding: 5px; font-weight: bold;'>{key}</td>"
                html += f"<td style='padding: 5px;'>{value_str}</td></tr>"
            
            html += "</table></div>"
            display(HTML(html))
        else:
            self.print_summary()
        
        return summary
