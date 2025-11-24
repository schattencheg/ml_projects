"""
BacktestNoLib - Custom backtesting implementation without external libraries.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base_backtest import BaseBacktest


class BacktestNoLib(BaseBacktest):
    """
    Custom backtesting implementation.
    
    Features:
    - Simple and transparent logic
    - Easy to customize
    - No external dependencies
    - Buy/sell signals from ML model predictions
    """
    
    def __init__(self,
                 initial_capital: float = 10000.0,
                 commission: float = 0.001,
                 position_size: float = 1.0,
                 stop_loss: Optional[float] = None,
                 take_profit: Optional[float] = None):
        """
        Initialize NoLib backtest.
        
        Args:
            initial_capital: Starting capital
            commission: Commission rate
            position_size: Position size as fraction of capital
            stop_loss: Stop loss percentage (e.g., 0.02 = 2%)
            take_profit: Take profit percentage (e.g., 0.05 = 5%)
        """
        super().__init__(initial_capital, commission, position_size)
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
    def run(self,
            df: pd.DataFrame,
            model: Any,
            scaler: Any,
            feature_cols: list,
            price_col: str = 'close',
            **kwargs) -> Dict[str, Any]:
        """
        Run backtest using custom logic.
        
        Args:
            df: DataFrame with OHLCV data and features
            model: Trained ML model
            scaler: Fitted scaler
            feature_cols: Feature column names
            price_col: Price column name
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with backtest results
        """
        print(f"\nRunning {self.__class__.__name__} backtest...")
        
        # Prepare features
        X = df[feature_cols].values
        X_scaled = scaler.transform(X)
        
        # Get predictions
        predictions = model.predict(X_scaled)
        
        # Initialize tracking variables
        capital = self.initial_capital
        position = 0  # Number of shares
        entry_price = 0
        equity_curve = []
        trades = []
        
        # Simulate trading
        for i in range(len(df)):
            current_price = df[price_col].iloc[i]
            prediction = predictions[i]
            
            # Check stop loss and take profit
            if position > 0:
                pnl_pct = (current_price - entry_price) / entry_price
                
                if self.stop_loss and pnl_pct <= -self.stop_loss:
                    # Stop loss hit
                    sell_value = position * current_price
                    commission_cost = sell_value * self.commission
                    capital += sell_value - commission_cost
                    
                    trades.append({
                        'entry_idx': entry_idx,
                        'exit_idx': i,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'shares': position,
                        'pnl': sell_value - (position * entry_price) - commission_cost,
                        'exit_reason': 'stop_loss'
                    })
                    
                    position = 0
                    entry_price = 0
                    
                elif self.take_profit and pnl_pct >= self.take_profit:
                    # Take profit hit
                    sell_value = position * current_price
                    commission_cost = sell_value * self.commission
                    capital += sell_value - commission_cost
                    
                    trades.append({
                        'entry_idx': entry_idx,
                        'exit_idx': i,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'shares': position,
                        'pnl': sell_value - (position * entry_price) - commission_cost,
                        'exit_reason': 'take_profit'
                    })
                    
                    position = 0
                    entry_price = 0
            
            # Trading logic
            if prediction == 1 and position == 0:
                # Buy signal
                buy_amount = capital * self.position_size
                commission_cost = buy_amount * self.commission
                shares = (buy_amount - commission_cost) / current_price
                
                if shares > 0:
                    position = shares
                    entry_price = current_price
                    entry_idx = i
                    capital -= buy_amount
                    
            elif prediction == 0 and position > 0:
                # Sell signal
                sell_value = position * current_price
                commission_cost = sell_value * self.commission
                capital += sell_value - commission_cost
                
                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'shares': position,
                    'pnl': sell_value - (position * entry_price) - commission_cost,
                    'exit_reason': 'signal'
                })
                
                position = 0
                entry_price = 0
            
            # Calculate equity
            if position > 0:
                equity = capital + (position * current_price)
            else:
                equity = capital
            
            equity_curve.append(equity)
        
        # Close any open position at the end
        if position > 0:
            final_price = df[price_col].iloc[-1]
            sell_value = position * final_price
            commission_cost = sell_value * self.commission
            capital += sell_value - commission_cost
            
            trades.append({
                'entry_idx': entry_idx,
                'exit_idx': len(df) - 1,
                'entry_price': entry_price,
                'exit_price': final_price,
                'shares': position,
                'pnl': sell_value - (position * entry_price) - commission_cost,
                'exit_reason': 'end_of_data'
            })
            
            position = 0
        
        # Store results
        self.trades = trades
        equity_series = pd.Series(equity_curve, index=df.index)
        
        self.results = {
            'equity_curve': equity_series,
            'trades': trades,
            'final_capital': capital
        }
        
        # Calculate metrics
        self.metrics = self.calculate_metrics()
        
        print(f"✓ Backtest complete: {len(trades)} trades")
        
        return self.results
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Returns:
            Dictionary with metrics
        """
        if self.results is None:
            return {}
        
        equity_curve = self.results['equity_curve']
        trades = self.results['trades']
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        metrics = {
            'initial_capital': self.initial_capital,
            'final_capital': self.results['final_capital'],
            'total_return': self._calculate_returns(equity_curve),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(equity_curve),
            'total_trades': len(trades),
            'win_rate': self._calculate_win_rate(trades) if trades else 0.0,
        }
        
        # Additional metrics
        if trades:
            pnls = [trade['pnl'] for trade in trades]
            metrics['avg_trade_pnl'] = np.mean(pnls)
            metrics['total_pnl'] = sum(pnls)
        
        return metrics
