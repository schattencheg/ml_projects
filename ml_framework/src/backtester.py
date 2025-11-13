"""
Backtester - Backtests trading strategies using ML predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt


class Backtester:
    """
    Backtests trading strategies using ML model predictions.
    
    Features:
    - Simple buy/hold strategy based on predictions
    - Position sizing
    - Performance metrics (returns, Sharpe ratio, drawdown)
    - Visualization
    """
    
    def __init__(self, initial_capital: float = 10000.0,
                 position_size: float = 1.0,
                 commission: float = 0.001):
        """
        Initialize Backtester.
        
        Args:
            initial_capital: Starting capital
            position_size: Position size as fraction of capital (0-1)
            commission: Commission per trade (as fraction, e.g., 0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.commission = commission
        self.results = None
        
    def run(self, df: pd.DataFrame,
           model: Any,
           scaler: Optional[Any] = None,
           feature_cols: Optional[list] = None,
           price_col: str = 'close') -> Dict[str, Any]:
        """
        Run backtest using ML model predictions.
        
        Args:
            df: DataFrame with OHLCV data and features
            model: Trained ML model
            scaler: Fitted scaler (optional)
            feature_cols: List of feature columns
            price_col: Name of price column
            
        Returns:
            Dictionary with backtest results
        """
        print("\n" + "="*70)
        print("RUNNING BACKTEST")
        print("="*70)
        
        # Prepare features
        if feature_cols is None:
            feature_cols = [col for col in df.columns 
                          if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]
        
        X = df[feature_cols].values
        
        # Scale features if scaler provided
        if scaler is not None:
            X = scaler.transform(X)
        
        # Get predictions
        predictions = model.predict(X)
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)[:, 1]
        else:
            probabilities = predictions.astype(float)
        
        # Run backtest
        df_backtest = df.copy()
        df_backtest['prediction'] = predictions
        df_backtest['probability'] = probabilities
        
        # Calculate positions (1 = long, 0 = no position)
        df_backtest['position'] = predictions
        
        # Calculate returns
        df_backtest['price_return'] = df_backtest[price_col].pct_change()
        df_backtest['strategy_return'] = (
            df_backtest['position'].shift(1) * df_backtest['price_return']
        )
        
        # Apply commission on position changes
        position_changes = df_backtest['position'].diff().abs()
        df_backtest['commission_cost'] = position_changes * self.commission
        df_backtest['strategy_return'] -= df_backtest['commission_cost']
        
        # Calculate cumulative returns
        df_backtest['cumulative_return'] = (1 + df_backtest['strategy_return']).cumprod()
        df_backtest['equity'] = self.initial_capital * df_backtest['cumulative_return']
        
        # Calculate buy & hold returns
        df_backtest['buy_hold_return'] = (1 + df_backtest['price_return']).cumprod()
        df_backtest['buy_hold_equity'] = self.initial_capital * df_backtest['buy_hold_return']
        
        # Calculate metrics
        metrics = self._calculate_metrics(df_backtest)
        
        # Store results
        self.results = {
            'df': df_backtest,
            'metrics': metrics
        }
        
        # Print results
        self._print_results(metrics)
        
        return self.results
    
    def _calculate_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate backtest performance metrics."""
        
        # Total return
        total_return = df['cumulative_return'].iloc[-1] - 1
        buy_hold_return = df['buy_hold_return'].iloc[-1] - 1
        
        # Annualized return (assuming daily data)
        num_days = len(df)
        years = num_days / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Sharpe ratio (assuming daily data, risk-free rate = 0)
        returns = df['strategy_return'].dropna()
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Maximum drawdown
        cumulative = df['cumulative_return']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = (returns > 0).sum()
        total_trades = len(returns[returns != 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Number of trades
        num_trades = df['position'].diff().abs().sum() / 2
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'buy_hold_return': buy_hold_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': num_trades,
            'final_equity': df['equity'].iloc[-1],
        }
    
    def _print_results(self, metrics: Dict[str, float]):
        """Print backtest results."""
        print(f"\nBacktest Configuration:")
        print(f"  Initial capital: ${self.initial_capital:,.2f}")
        print(f"  Position size: {self.position_size*100:.1f}%")
        print(f"  Commission: {self.commission*100:.2f}%")
        
        print(f"\n{'='*70}")
        print("BACKTEST RESULTS")
        print(f"{'='*70}")
        
        print(f"\nPerformance:")
        print(f"  Total return:      {metrics['total_return']*100:>8.2f}%")
        print(f"  Annualized return: {metrics['annualized_return']*100:>8.2f}%")
        print(f"  Buy & Hold return: {metrics['buy_hold_return']*100:>8.2f}%")
        print(f"  Final equity:      ${metrics['final_equity']:>8,.2f}")
        
        print(f"\nRisk Metrics:")
        print(f"  Sharpe ratio:      {metrics['sharpe_ratio']:>8.2f}")
        print(f"  Max drawdown:      {metrics['max_drawdown']*100:>8.2f}%")
        
        print(f"\nTrading:")
        print(f"  Number of trades:  {metrics['num_trades']:>8.0f}")
        print(f"  Win rate:          {metrics['win_rate']*100:>8.2f}%")
        
        print(f"{'='*70}\n")
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot backtest results.
        
        Args:
            save_path: Path to save plot (optional)
        """
        if self.results is None:
            print("No results to plot. Run backtest first.")
            return
        
        df = self.results['df']
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot equity curves
        axes[0].plot(df.index, df['equity'], label='Strategy', linewidth=2)
        axes[0].plot(df.index, df['buy_hold_equity'], 
                    label='Buy & Hold', linewidth=2, alpha=0.7)
        axes[0].set_title('Equity Curve', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Equity ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot drawdown
        cumulative = df['cumulative_return']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        axes[1].fill_between(df.index, drawdown * 100, 0, 
                            alpha=0.3, color='red')
        axes[1].plot(df.index, drawdown * 100, color='red', linewidth=1)
        axes[1].set_title('Drawdown', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].set_xlabel('Date')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        
        plt.show()
    
    def get_results(self) -> Optional[Dict[str, Any]]:
        """Get backtest results."""
        return self.results
