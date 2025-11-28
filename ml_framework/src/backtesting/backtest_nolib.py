"""
BacktestNoLib - Custom backtesting implementation without external libraries.

Uses RiskManager (from base class) for:
- Position sizing (default 2% of current capital)
- Exit after exactly N bars
- Cooldown: new positions only on the bar after previous closes
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from src.backtesting.base_backtest import BaseBacktest


class BacktestNoLib(BaseBacktest):
    """
    Custom backtesting implementation.
    
    Features:
    - Simple and transparent logic
    - Easy to customize
    - No external dependencies
    - Buy/sell signals from ML model predictions
    - RiskManager controls position sizing and exit timing (inherited from BaseBacktest)
    """
    
    def __init__(self,
                 initial_capital: float = 10000.0,
                 commission: float = 0.001,
                 position_size: float = 0.02,
                 bars_to_hold: int = 15,
                 risk_manager=None):
        """
        Initialize NoLib backtest.
        
        Args:
            initial_capital: Starting capital
            commission: Commission rate
            position_size: Position size as fraction of capital (default 0.02 = 2%)
            bars_to_hold: Number of bars to hold position before exiting (default 15)
            risk_manager: Optional custom RiskManager instance (passed to base class)
        """
        super().__init__(
            initial_capital=initial_capital,
            commission=commission,
            position_size=position_size,
            bars_to_hold=bars_to_hold,
            risk_manager=risk_manager
        )
        
    def run(self,
            df: pd.DataFrame,
            model: Any,
            scaler: Any,
            feature_cols: list,
            price_col: str = 'close',
            **kwargs) -> Dict[str, Any]:
        """
        Run backtest using custom logic with RiskManager.
        
        RiskManager controls:
        - Position sizing (2% of current capital by default)
        - Exit after exactly N bars
        - Cooldown: new positions only on the bar after previous closes
        
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
        print(f"  RiskManager: {self.risk_manager}")
        
        # Reset risk manager state
        self.risk_manager.reset()
        
        # Prepare features
        X = df[feature_cols].values
        X_scaled = scaler.transform(X)
        
        # Get predictions
        predictions = model.predict(X_scaled)
        
        # Debug: show prediction distribution
        unique, counts = np.unique(predictions, return_counts=True)
        print(f"  Predictions distribution: {dict(zip(unique, counts))}")
        print(f"  Model: {model.name}, Total predictions: {len(predictions)}")
        
        # Initialize tracking variables
        capital = self.initial_capital
        position = 0  # Number of shares (positive=long, negative=short)
        position_type = None  # 'long' or 'short'
        entry_price = 0
        entry_idx = 0
        equity_curve = []
        trades = []
        current_position = None  # Track current position for RiskManager
        
        # Simulate trading
        for i in range(len(df)):
            current_price = df[price_col].iloc[i]
            prediction = predictions[i]
            
            # Check exit condition using RiskManager (only if we have a position)
            if position != 0 and current_position is not None:
                should_exit, exit_reason = self.risk_manager.should_exit(
                    position=current_position,
                    current_bar=i,
                    current_price=current_price,
                    df=df
                )
                
                if should_exit:
                    # Calculate P&L based on position type
                    shares = abs(position)
                    if position_type == 'long':
                        pnl = (current_price - entry_price) * shares
                        capital += shares * current_price * (1 - self.commission)
                    else:  # short
                        pnl = (entry_price - current_price) * shares
                        # Close short: buy back shares, return margin
                        capital += (entry_price - current_price) * shares - shares * current_price * self.commission
                    
                    trades.append({
                        'entry_idx': entry_idx,
                        'exit_idx': i,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'shares': shares,
                        'position_type': position_type,
                        'pnl': pnl,
                        'exit_reason': exit_reason
                    })
                    
                    # Notify RiskManager of exit
                    self.risk_manager.on_exit(f"pos_{entry_idx}", i)
                    
                    position = 0
                    position_type = None
                    entry_price = 0
                    current_position = None
            
            # Check entry condition (only if no position AND RiskManager allows)
            # prediction: +1 = long, -1 = short, 0 = no trade
            if position == 0 and prediction != 0:
                # Check if RiskManager allows opening a new position
                if self.risk_manager.can_open_position(i):
                    # Get position size from RiskManager (2% of current capital)
                    shares = self.risk_manager.get_position_size(
                        capital=capital,
                        price=current_price,
                        commission=self.commission
                    )
                    
                    if shares > 0:
                        entry_price = current_price
                        entry_idx = i
                        
                        if prediction == 1:  # Long
                            position_type = 'long'
                            position = shares
                            buy_amount = shares * current_price
                            commission_cost = buy_amount * self.commission
                            capital -= (buy_amount + commission_cost)
                        else:  # prediction == -1, Short
                            position_type = 'short'
                            position = -shares
                            # Short: sell borrowed shares, receive cash minus commission
                            sell_amount = shares * current_price
                            commission_cost = sell_amount * self.commission
                            capital -= commission_cost  # Only pay commission, margin held
                        
                        # Create position dict for RiskManager
                        current_position = {
                            'entry_bar': i,
                            'entry_price': current_price,
                            'shares': shares,
                            'entry_idx': i,
                            'position_type': position_type
                        }
                        
                        # Notify RiskManager of entry
                        self.risk_manager.on_entry(
                            position_id=f"pos_{i}",
                            entry_bar=i,
                            entry_price=current_price,
                            shares=shares,
                            entry_idx=i
                        )
            
            # Calculate equity
            if position > 0:  # Long
                equity = capital + (position * current_price)
            elif position < 0:  # Short
                # Unrealized P&L for short = (entry_price - current_price) * shares
                equity = capital + (entry_price - current_price) * abs(position)
            else:
                equity = capital
            
            equity_curve.append(equity)
        
        # Close any open position at the end
        if position != 0:
            final_price = df[price_col].iloc[-1]
            shares = abs(position)
            
            if position_type == 'long':
                pnl = (final_price - entry_price) * shares
                capital += shares * final_price * (1 - self.commission)
            else:  # short
                pnl = (entry_price - final_price) * shares
                capital += (entry_price - final_price) * shares - shares * final_price * self.commission
            
            trades.append({
                'entry_idx': entry_idx,
                'exit_idx': len(df) - 1,
                'entry_price': entry_price,
                'exit_price': final_price,
                'shares': shares,
                'position_type': position_type,
                'pnl': pnl,
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
