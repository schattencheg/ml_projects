"""
ML-Based Backtesting Module with Trailing Stop Loss

This module provides a backtesting framework that uses trained ML models
to generate trading signals and implements trailing stop loss functionality.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


class MLBacktester:
    """
    Backtester that uses ML model predictions as trading signals.
    Supports trailing stop loss and various position sizing strategies.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        position_size: float = 1.0,
        trailing_stop_pct: float = 2.0,
        take_profit_pct: Optional[float] = None,
        commission: float = 0.001,
        slippage: float = 0.0005,
        use_probability_threshold: bool = True,
        probability_threshold: float = 0.6,
        max_holding_bars: Optional[int] = None
    ):
        """
        Initialize the ML Backtester.
        
        Parameters:
        -----------
        initial_capital : float
            Starting capital for backtesting
        position_size : float
            Fraction of capital to use per trade (0.0 to 1.0)
        trailing_stop_pct : float
            Trailing stop loss percentage (e.g., 2.0 for 2%)
        take_profit_pct : float, optional
            Take profit percentage (e.g., 5.0 for 5%)
        commission : float
            Commission per trade as a fraction (e.g., 0.001 for 0.1%)
        slippage : float
            Slippage per trade as a fraction (e.g., 0.0005 for 0.05%)
        use_probability_threshold : bool
            Whether to use probability threshold for entry
        probability_threshold : float
            Minimum probability to enter trade (0.0 to 1.0)
        max_holding_bars : int, optional
            Maximum number of bars to hold a position
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.trailing_stop_pct = trailing_stop_pct
        self.take_profit_pct = take_profit_pct
        self.commission = commission
        self.slippage = slippage
        self.use_probability_threshold = use_probability_threshold
        self.probability_threshold = probability_threshold
        self.max_holding_bars = max_holding_bars
        
        # Trading state
        self.capital = initial_capital
        self.position = 0  # Number of shares/units
        self.entry_price = 0
        self.highest_price = 0
        self.trailing_stop_price = 0
        self.bars_in_position = 0
        
        # Results tracking
        self.trades = []
        self.equity_curve = []
        self.signals = []
        
    def reset(self):
        """Reset the backtester to initial state."""
        self.capital = self.initial_capital
        self.position = 0
        self.entry_price = 0
        self.highest_price = 0
        self.trailing_stop_price = 0
        self.bars_in_position = 0
        self.trades = []
        self.equity_curve = []
        self.signals = []
    
    def calculate_position_size(self, price: float) -> float:
        """
        Calculate the number of units to buy based on position sizing.
        
        Parameters:
        -----------
        price : float
            Current price
            
        Returns:
        --------
        units : float
            Number of units to buy
        """
        capital_to_use = self.capital * self.position_size
        units = capital_to_use / price
        return units
    
    def enter_long(self, price: float, timestamp: pd.Timestamp, signal_strength: float = 1.0):
        """
        Enter a long position.
        
        Parameters:
        -----------
        price : float
            Entry price
        timestamp : pd.Timestamp
            Entry timestamp
        signal_strength : float
            Strength of the signal (e.g., probability from model)
        """
        if self.position > 0:
            return  # Already in position
        
        # Apply slippage
        entry_price_with_slippage = price * (1 + self.slippage)
        
        # Calculate position size
        units = self.calculate_position_size(entry_price_with_slippage)
        
        # Calculate commission
        commission_cost = units * entry_price_with_slippage * self.commission
        
        # Update state
        self.position = units
        self.entry_price = entry_price_with_slippage
        self.highest_price = entry_price_with_slippage
        self.trailing_stop_price = entry_price_with_slippage * (1 - self.trailing_stop_pct / 100)
        self.capital -= (units * entry_price_with_slippage + commission_cost)
        self.bars_in_position = 0
        
        # Record signal
        self.signals.append({
            'timestamp': timestamp,
            'type': 'ENTRY',
            'price': entry_price_with_slippage,
            'units': units,
            'signal_strength': signal_strength,
            'capital': self.capital
        })
    
    def exit_long(self, price: float, timestamp: pd.Timestamp, reason: str = 'SIGNAL'):
        """
        Exit a long position.
        
        Parameters:
        -----------
        price : float
            Exit price
        timestamp : pd.Timestamp
            Exit timestamp
        reason : str
            Reason for exit ('SIGNAL', 'TRAILING_STOP', 'TAKE_PROFIT', 'MAX_HOLDING')
        """
        if self.position == 0:
            return  # No position to exit
        
        # Apply slippage
        exit_price_with_slippage = price * (1 - self.slippage)
        
        # Calculate commission
        commission_cost = self.position * exit_price_with_slippage * self.commission
        
        # Calculate P&L
        gross_pnl = self.position * (exit_price_with_slippage - self.entry_price)
        net_pnl = gross_pnl - commission_cost
        pnl_pct = (exit_price_with_slippage / self.entry_price - 1) * 100
        
        # Update capital
        self.capital += (self.position * exit_price_with_slippage - commission_cost)
        
        # Record trade
        self.trades.append({
            'entry_time': self.signals[-1]['timestamp'],
            'exit_time': timestamp,
            'entry_price': self.entry_price,
            'exit_price': exit_price_with_slippage,
            'units': self.position,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'pnl_pct': pnl_pct,
            'bars_held': self.bars_in_position,
            'exit_reason': reason,
            'signal_strength': self.signals[-1]['signal_strength']
        })
        
        # Record signal
        self.signals.append({
            'timestamp': timestamp,
            'type': 'EXIT',
            'price': exit_price_with_slippage,
            'units': self.position,
            'reason': reason,
            'capital': self.capital,
            'pnl': net_pnl
        })
        
        # Reset position
        self.position = 0
        self.entry_price = 0
        self.highest_price = 0
        self.trailing_stop_price = 0
        self.bars_in_position = 0
    
    def update_trailing_stop(self, current_price: float):
        """
        Update trailing stop loss based on current price.
        
        Parameters:
        -----------
        current_price : float
            Current market price
        """
        if self.position == 0:
            return
        
        # Update highest price
        if current_price > self.highest_price:
            self.highest_price = current_price
            # Update trailing stop
            self.trailing_stop_price = self.highest_price * (1 - self.trailing_stop_pct / 100)
    
    def check_exit_conditions(self, current_price: float, timestamp: pd.Timestamp) -> bool:
        """
        Check if any exit conditions are met.
        
        Parameters:
        -----------
        current_price : float
            Current market price
        timestamp : pd.Timestamp
            Current timestamp
            
        Returns:
        --------
        exited : bool
            True if position was exited
        """
        if self.position == 0:
            return False
        
        # Check trailing stop
        if current_price <= self.trailing_stop_price:
            self.exit_long(current_price, timestamp, reason='TRAILING_STOP')
            return True
        
        # Check take profit
        if self.take_profit_pct is not None:
            take_profit_price = self.entry_price * (1 + self.take_profit_pct / 100)
            if current_price >= take_profit_price:
                self.exit_long(current_price, timestamp, reason='TAKE_PROFIT')
                return True
        
        # Check max holding period
        if self.max_holding_bars is not None:
            if self.bars_in_position >= self.max_holding_bars:
                self.exit_long(current_price, timestamp, reason='MAX_HOLDING')
                return True
        
        return False
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        model,
        scaler,
        X_columns: List[str],
        close_column: str = 'Close',
        timestamp_column: str = 'Timestamp'
    ) -> Dict:
        """
        Run backtest using ML model predictions.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data and features
        model : sklearn model or similar
            Trained ML model with predict() and optionally predict_proba()
        scaler : sklearn scaler
            Fitted scaler for features
        X_columns : List[str]
            List of feature column names
        close_column : str
            Name of the close price column
        timestamp_column : str
            Name of the timestamp column
            
        Returns:
        --------
        results : Dict
            Dictionary with backtest results and metrics
        """
        self.reset()
        
        # Ensure timestamp is datetime
        if timestamp_column in df.columns:
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        else:
            df[timestamp_column] = pd.to_datetime(df.index)
        
        # Prepare features
        X = df[X_columns].values
        X_scaled = scaler.transform(X)
        
        # Get predictions
        predictions = model.predict(X_scaled)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_scaled)[:, 1]
        elif hasattr(model, 'decision_function'):
            # For models with decision_function, normalize to 0-1
            decision_scores = model.decision_function(X_scaled)
            probabilities = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min())
        else:
            probabilities = predictions.astype(float)
        
        # Run backtest
        for i in range(len(df)):
            current_price = df[close_column].iloc[i]
            timestamp = df[timestamp_column].iloc[i]
            prediction = predictions[i]
            probability = probabilities[i]
            
            # Update bars in position
            if self.position > 0:
                self.bars_in_position += 1
                
                # Update trailing stop
                self.update_trailing_stop(current_price)
                
                # Check exit conditions
                if self.check_exit_conditions(current_price, timestamp):
                    continue
            
            # Check entry signal
            if self.position == 0 and prediction == 1:
                # Check probability threshold
                if self.use_probability_threshold:
                    if probability >= self.probability_threshold:
                        self.enter_long(current_price, timestamp, signal_strength=probability)
                else:
                    self.enter_long(current_price, timestamp, signal_strength=probability)
            
            # Check exit signal (model predicts no increase)
            elif self.position > 0 and prediction == 0:
                self.exit_long(current_price, timestamp, reason='SIGNAL')
            
            # Record equity
            if self.position > 0:
                position_value = self.position * current_price
                total_equity = self.capital + position_value
            else:
                total_equity = self.capital
            
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': total_equity,
                'capital': self.capital,
                'position_value': self.position * current_price if self.position > 0 else 0,
                'in_position': self.position > 0
            })
        
        # Close any open position at the end
        if self.position > 0:
            final_price = df[close_column].iloc[-1]
            final_timestamp = df[timestamp_column].iloc[-1]
            self.exit_long(final_price, final_timestamp, reason='END_OF_DATA')
        
        # Calculate metrics
        results = self.calculate_metrics(df, close_column)
        
        return results
    
    def calculate_metrics(self, df: pd.DataFrame, close_column: str = 'Close') -> Dict:
        """
        Calculate performance metrics.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Original dataframe with price data
        close_column : str
            Name of the close price column
            
        Returns:
        --------
        metrics : Dict
            Dictionary with performance metrics
        """
        if len(self.trades) == 0:
            return {
                'total_trades': 0,
                'final_capital': self.capital,
                'total_return': 0,
                'total_return_pct': 0,
                'buy_and_hold_return_pct': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'trades': [],
                'equity_curve': pd.DataFrame(self.equity_curve)
            }
        
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Basic metrics
        total_trades = len(self.trades)
        final_capital = self.capital
        total_return = final_capital - self.initial_capital
        total_return_pct = (final_capital / self.initial_capital - 1) * 100
        
        # Buy and hold return
        buy_and_hold_return_pct = (df[close_column].iloc[-1] / df[close_column].iloc[0] - 1) * 100
        
        # Win rate
        winning_trades = trades_df[trades_df['net_pnl'] > 0]
        losing_trades = trades_df[trades_df['net_pnl'] <= 0]
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        # Average win/loss
        avg_win = winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades['net_pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['net_pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Max drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
        max_drawdown = equity_df['drawdown'].min()
        
        # Sharpe ratio (annualized, assuming daily data)
        equity_df['returns'] = equity_df['equity'].pct_change()
        sharpe_ratio = equity_df['returns'].mean() / equity_df['returns'].std() * np.sqrt(252) if equity_df['returns'].std() > 0 else 0
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'final_capital': final_capital,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'buy_and_hold_return_pct': buy_and_hold_return_pct,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_bars_held': trades_df['bars_held'].mean(),
            'trades': self.trades,
            'equity_curve': equity_df,
            'signals': self.signals
        }
        
        return metrics
    
    def print_results(self, results: Dict):
        """
        Print formatted backtest results.
        
        Parameters:
        -----------
        results : Dict
            Results dictionary from run_backtest()
        """
        print("\n" + "="*80)
        print("BACKTEST RESULTS")
        print("="*80)
        
        print(f"\nCapital:")
        print(f"  Initial Capital:        ${self.initial_capital:,.2f}")
        print(f"  Final Capital:          ${results['final_capital']:,.2f}")
        print(f"  Total Return:           ${results['total_return']:,.2f}")
        print(f"  Total Return %:         {results['total_return_pct']:.2f}%")
        print(f"  Buy & Hold Return %:    {results['buy_and_hold_return_pct']:.2f}%")
        
        print(f"\nTrades:")
        print(f"  Total Trades:           {results['total_trades']}")
        print(f"  Winning Trades:         {results['winning_trades']}")
        print(f"  Losing Trades:          {results['losing_trades']}")
        print(f"  Win Rate:               {results['win_rate']:.2f}%")
        
        print(f"\nProfit/Loss:")
        print(f"  Average Win:            ${results['avg_win']:,.2f}")
        print(f"  Average Loss:           ${results['avg_loss']:,.2f}")
        print(f"  Profit Factor:          {results['profit_factor']:.2f}")
        
        print(f"\nRisk Metrics:")
        print(f"  Max Drawdown:           {results['max_drawdown']:.2f}%")
        print(f"  Sharpe Ratio:           {results['sharpe_ratio']:.2f}")
        
        print(f"\nHolding Period:")
        print(f"  Avg Bars Held:          {results['avg_bars_held']:.1f}")
        
        print(f"\nStrategy Parameters:")
        print(f"  Position Size:          {self.position_size * 100:.1f}%")
        print(f"  Trailing Stop:          {self.trailing_stop_pct:.2f}%")
        if self.take_profit_pct:
            print(f"  Take Profit:            {self.take_profit_pct:.2f}%")
        if self.use_probability_threshold:
            print(f"  Probability Threshold:  {self.probability_threshold:.2f}")
        if self.max_holding_bars:
            print(f"  Max Holding Bars:       {self.max_holding_bars}")
        
        print("\n" + "="*80)
        
        # Print last 10 trades
        if len(results['trades']) > 0:
            print("\nLast 10 Trades:")
            print("-"*80)
            trades_df = pd.DataFrame(results['trades'])
            print(trades_df[['entry_time', 'exit_time', 'entry_price', 'exit_price', 
                           'net_pnl', 'pnl_pct', 'exit_reason']].tail(10).to_string(index=False))
    
    def plot_results(self, results: Dict, df: pd.DataFrame, close_column: str = 'Close', 
                    timestamp_column: str = 'Timestamp', save_path: Optional[str] = None):
        """
        Plot backtest results.
        
        Parameters:
        -----------
        results : Dict
            Results dictionary from run_backtest()
        df : pd.DataFrame
            Original dataframe with price data
        close_column : str
            Name of the close price column
        timestamp_column : str
            Name of the timestamp column
        save_path : str, optional
            Path to save the plot
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Ensure timestamp is datetime
        if timestamp_column in df.columns:
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        else:
            df[timestamp_column] = pd.to_datetime(df.index)
        
        equity_df = results['equity_curve']
        
        # Plot 1: Price with Entry/Exit signals
        ax1 = axes[0]
        ax1.plot(df[timestamp_column], df[close_column], label='Price', color='black', alpha=0.7)
        
        # Plot entry signals
        entry_signals = [s for s in results['signals'] if s['type'] == 'ENTRY']
        if entry_signals:
            entry_times = [s['timestamp'] for s in entry_signals]
            entry_prices = [s['price'] for s in entry_signals]
            ax1.scatter(entry_times, entry_prices, color='green', marker='^', s=100, 
                       label='Entry', zorder=5)
        
        # Plot exit signals
        exit_signals = [s for s in results['signals'] if s['type'] == 'EXIT']
        if exit_signals:
            exit_times = [s['timestamp'] for s in exit_signals]
            exit_prices = [s['price'] for s in exit_signals]
            exit_colors = ['red' if s['reason'] == 'TRAILING_STOP' else 'orange' 
                          for s in exit_signals]
            ax1.scatter(exit_times, exit_prices, color=exit_colors, marker='v', s=100, 
                       label='Exit', zorder=5)
        
        ax1.set_title('Price Chart with Entry/Exit Signals', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Equity Curve
        ax2 = axes[1]
        ax2.plot(equity_df['timestamp'], equity_df['equity'], label='Portfolio Value', 
                color='blue', linewidth=2)
        ax2.axhline(y=self.initial_capital, color='gray', linestyle='--', 
                   label='Initial Capital', alpha=0.7)
        
        # Shade periods in position
        in_position = equity_df['in_position']
        ax2.fill_between(equity_df['timestamp'], equity_df['equity'].min(), 
                        equity_df['equity'].max(), where=in_position, 
                        alpha=0.1, color='green', label='In Position')
        
        ax2.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Drawdown
        ax3 = axes[2]
        ax3.fill_between(equity_df['timestamp'], 0, equity_df['drawdown'], 
                        color='red', alpha=0.3)
        ax3.plot(equity_df['timestamp'], equity_df['drawdown'], color='red', linewidth=1)
        ax3.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Drawdown (%)')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to: {save_path}")
        
        plt.show()
