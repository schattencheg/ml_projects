"""
ML-Based Backtesting using backtesting.py Library

This module provides a backtesting strategy class that integrates with the backtesting.py
library to use trained ML models for generating trading signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA


class MLStrategy(Strategy):
    """
    Strategy class for backtesting.py that uses ML model predictions as signals.
    
    This strategy:
    1. Uses ML model predictions to generate buy/sell signals
    2. Implements trailing stop loss
    3. Supports probability thresholds for entry
    4. Allows position sizing based on signal strength
    """
    
    # Strategy parameters (can be optimized)
    probability_threshold = 0.6
    trailing_stop_pct = 2.0
    take_profit_pct = None  # None means no take profit
    position_size_pct = 1.0  # Fraction of equity to use (0.0 to 1.0)
    
    def init(self):
        """
        Initialize the strategy.
        Called once at the start of backtesting.
        """
        # Get ML predictions from data
        # These should be added to the dataframe before backtesting
        self.ml_signal = self.data.ML_Signal  # 1 for buy, 0 for no action
        self.ml_probability = self.data.ML_Probability  # Probability from model
        
        # Track entry price for trailing stop
        self.entry_price = None
        self.highest_price = None
        
    def next(self):
        """
        Called for each bar in the backtest.
        Implements the trading logic.
        """
        # Get current values
        current_price = self.data.Close[-1]
        signal = self.ml_signal[-1]
        probability = self.ml_probability[-1]
        
        # If we're in a position, check exit conditions
        if self.position:
            self._check_exit_conditions(current_price)
        
        # If not in position, check entry conditions
        elif signal == 1 and probability >= self.probability_threshold:
            # Calculate position size
            size = self._calculate_position_size(current_price)
            
            # Enter long position
            self.buy(size=size)
            self.entry_price = current_price
            self.highest_price = current_price
    
    def _check_exit_conditions(self, current_price: float):
        """
        Check if any exit conditions are met.
        
        Parameters:
        -----------
        current_price : float
            Current market price
        """
        if not self.position or self.entry_price is None:
            return
        
        # Update highest price for trailing stop
        if current_price > self.highest_price:
            self.highest_price = current_price
        
        # Calculate trailing stop price
        trailing_stop_price = self.highest_price * (1 - self.trailing_stop_pct / 100)
        
        # Check trailing stop
        if current_price <= trailing_stop_price:
            self.position.close()
            self.entry_price = None
            self.highest_price = None
            return
        
        # Check take profit if enabled
        if self.take_profit_pct is not None:
            take_profit_price = self.entry_price * (1 + self.take_profit_pct / 100)
            if current_price >= take_profit_price:
                self.position.close()
                self.entry_price = None
                self.highest_price = None
                return
        
        # Check for exit signal from ML model
        signal = self.ml_signal[-1]
        if signal == 0:  # Exit signal
            self.position.close()
            self.entry_price = None
            self.highest_price = None
    
    def _calculate_position_size(self, price: float) -> float:
        """
        Calculate position size based on available equity.
        
        Parameters:
        -----------
        price : float
            Current price
            
        Returns:
        --------
        size : float
            Number of units to buy
        """
        equity_to_use = self.equity * self.position_size_pct
        size = equity_to_use / price
        return size


class MLBacktesterPy:
    """
    Wrapper class for backtesting.py that handles ML model integration.
    
    This class:
    1. Prepares data with ML predictions
    2. Runs backtests using the backtesting.py library
    3. Provides easy-to-use interface for ML-based backtesting
    """
    
    def __init__(
        self,
        initial_cash: float = 10000.0,
        commission: float = 0.001,
        margin: float = 1.0,
        trade_on_close: bool = False,
        hedging: bool = False,
        exclusive_orders: bool = True
    ):
        """
        Initialize the ML Backtester for backtesting.py.
        
        Parameters:
        -----------
        initial_cash : float
            Starting capital
        commission : float
            Commission per trade as a fraction (e.g., 0.001 for 0.1%)
        margin : float
            Margin requirement (1.0 = no leverage)
        trade_on_close : bool
            Whether to trade on close prices
        hedging : bool
            Whether to allow hedging (simultaneous long/short)
        exclusive_orders : bool
            Whether orders are exclusive
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.margin = margin
        self.trade_on_close = trade_on_close
        self.hedging = hedging
        self.exclusive_orders = exclusive_orders
        
    def prepare_data(
        self,
        df: pd.DataFrame,
        model,
        scaler,
        X_columns: List[str],
        close_column: str = 'close',
        timestamp_column: str = 'Timestamp'
    ) -> pd.DataFrame:
        """
        Prepare data with ML predictions for backtesting.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data and features
        model : sklearn model
            Trained ML model
        scaler : sklearn scaler
            Fitted scaler
        X_columns : List[str]
            List of feature column names
        close_column : str
            Name of close price column
        timestamp_column : str
            Name of timestamp column
            
        Returns:
        --------
        df_prepared : pd.DataFrame
            DataFrame with ML predictions added
        """
        # Make a copy
        df_prepared = df.copy()
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'Volume']
        for col in required_cols:
            if col not in df_prepared.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Get features
        X = df_prepared[X_columns].values
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Get predictions
        predictions = model.predict(X_scaled)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_scaled)
            # Get probability of positive class
            ml_probability = probabilities[:, 1]
        else:
            # Use decision function or just the prediction
            if hasattr(model, 'decision_function'):
                decision = model.decision_function(X_scaled)
                # Normalize to 0-1 range
                ml_probability = (decision - decision.min()) / (decision.max() - decision.min())
            else:
                ml_probability = predictions.astype(float)
        
        # Add predictions to dataframe
        df_prepared['ML_Signal'] = predictions
        df_prepared['ML_Probability'] = ml_probability
        
        # Set timestamp as index if it exists
        if timestamp_column in df_prepared.columns:
            df_prepared.set_index(timestamp_column, inplace=True)
        
        # Ensure index is datetime
        if not isinstance(df_prepared.index, pd.DatetimeIndex):
            df_prepared.index = pd.to_datetime(df_prepared.index)
        
        return df_prepared
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        model,
        scaler,
        X_columns: List[str],
        probability_threshold: float = 0.6,
        trailing_stop_pct: float = 2.0,
        take_profit_pct: Optional[float] = None,
        position_size_pct: float = 1.0,
        plot: bool = True,
        **kwargs
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Run backtest using ML model predictions.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data and features
        model : sklearn model
            Trained ML model
        scaler : sklearn scaler
            Fitted scaler
        X_columns : List[str]
            List of feature column names
        probability_threshold : float
            Minimum probability to enter trade
        trailing_stop_pct : float
            Trailing stop loss percentage
        take_profit_pct : float, optional
            Take profit percentage
        position_size_pct : float
            Fraction of equity to use per trade
        plot : bool
            Whether to plot results
        **kwargs : dict
            Additional arguments for Backtest
            
        Returns:
        --------
        stats : pd.Series
            Backtest statistics
        trades : pd.DataFrame
            Trade history
        """
        # Prepare data with ML predictions
        df_prepared = self.prepare_data(
            df=df,
            model=model,
            scaler=scaler,
            X_columns=X_columns
        )
        
        # Set strategy parameters
        MLStrategy.probability_threshold = probability_threshold
        MLStrategy.trailing_stop_pct = trailing_stop_pct
        MLStrategy.take_profit_pct = take_profit_pct
        MLStrategy.position_size_pct = position_size_pct
        
        # Create backtest
        bt = Backtest(
            df_prepared,
            MLStrategy,
            cash=self.initial_cash,
            commission=self.commission,
            margin=self.margin,
            trade_on_close=self.trade_on_close,
            hedging=self.hedging,
            exclusive_orders=self.exclusive_orders
        )
        
        # Run backtest
        stats = bt.run()
        
        # Get trades
        trades = stats._trades if hasattr(stats, '_trades') else pd.DataFrame()
        
        # Plot if requested
        if plot:
            bt.plot()
        
        return stats, trades
    
    def optimize(
        self,
        df: pd.DataFrame,
        model,
        scaler,
        X_columns: List[str],
        probability_threshold_range: Tuple[float, float, float] = (0.5, 0.8, 0.05),
        trailing_stop_range: Tuple[float, float, float] = (1.0, 5.0, 0.5),
        maximize: str = 'Return [%]',
        constraint: Optional[callable] = None,
        return_heatmap: bool = True,
        **kwargs
    ) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """
        Optimize strategy parameters.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data and features
        model : sklearn model
            Trained ML model
        scaler : sklearn scaler
            Fitted scaler
        X_columns : List[str]
            List of feature column names
        probability_threshold_range : Tuple[float, float, float]
            (start, end, step) for probability threshold
        trailing_stop_range : Tuple[float, float, float]
            (start, end, step) for trailing stop percentage
        maximize : str
            Metric to maximize
        constraint : callable, optional
            Constraint function
        return_heatmap : bool
            Whether to return optimization heatmap
        **kwargs : dict
            Additional arguments for optimize
            
        Returns:
        --------
        best_stats : pd.Series
            Statistics for best parameters
        heatmap : pd.DataFrame, optional
            Optimization heatmap
        """
        # Prepare data with ML predictions
        df_prepared = self.prepare_data(
            df=df,
            model=model,
            scaler=scaler,
            X_columns=X_columns
        )
        
        # Create backtest
        bt = Backtest(
            df_prepared,
            MLStrategy,
            cash=self.initial_cash,
            commission=self.commission,
            margin=self.margin,
            trade_on_close=self.trade_on_close,
            hedging=self.hedging,
            exclusive_orders=self.exclusive_orders
        )
        
        # Run optimization
        stats = bt.optimize(
            probability_threshold=np.arange(*probability_threshold_range),
            trailing_stop_pct=np.arange(*trailing_stop_range),
            maximize=maximize,
            constraint=constraint,
            return_heatmap=return_heatmap,
            **kwargs
        )
        
        if return_heatmap:
            return stats[0], stats[1]
        else:
            return stats, None
    
    def print_results(self, stats: pd.Series):
        """
        Print formatted backtest results.
        
        Parameters:
        -----------
        stats : pd.Series
            Backtest statistics
        """
        print("\n" + "="*80)
        print("BACKTEST RESULTS (backtesting.py)")
        print("="*80)
        
        print("\nCapital:")
        print(f"  Initial Capital:        ${self.initial_cash:,.2f}")
        print(f"  Final Equity:           ${stats['Equity Final [$]']:,.2f}")
        print(f"  Total Return:           ${stats['Equity Final [$]'] - self.initial_cash:,.2f}")
        print(f"  Total Return %:         {stats['Return [%]']:.2f}%")
        
        if 'Buy & Hold Return [%]' in stats:
            print(f"  Buy & Hold Return %:    {stats['Buy & Hold Return [%]']:.2f}%")
        
        print("\nTrades:")
        print(f"  Total Trades:           {stats['# Trades']}")
        print(f"  Win Rate:               {stats['Win Rate [%]']:.2f}%")
        print(f"  Average Trade:          {stats['Avg. Trade [%]']:.2f}%")
        print(f"  Best Trade:             {stats['Best Trade [%]']:.2f}%")
        print(f"  Worst Trade:            {stats['Worst Trade [%]']:.2f}%")
        
        print("\nRisk Metrics:")
        print(f"  Max Drawdown:           {stats['Max. Drawdown [%]']:.2f}%")
        print(f"  Sharpe Ratio:           {stats['Sharpe Ratio']:.2f}")
        print(f"  Sortino Ratio:          {stats['Sortino Ratio']:.2f}")
        print(f"  Calmar Ratio:           {stats['Calmar Ratio']:.2f}")
        
        print("\nDuration:")
        print(f"  Start:                  {stats['Start']}")
        print(f"  End:                    {stats['End']}")
        print(f"  Duration:               {stats['Duration']}")
        
        print("="*80)


def create_ml_strategy_class(
    model,
    scaler,
    X_columns: List[str],
    probability_threshold: float = 0.6,
    trailing_stop_pct: float = 2.0,
    take_profit_pct: Optional[float] = None,
    position_size_pct: float = 1.0
) -> type:
    """
    Factory function to create a custom MLStrategy class with embedded model.
    
    This is useful when you want to create a strategy class that doesn't require
    external data preparation.
    
    Parameters:
    -----------
    model : sklearn model
        Trained ML model
    scaler : sklearn scaler
        Fitted scaler
    X_columns : List[str]
        List of feature column names
    probability_threshold : float
        Minimum probability to enter trade
    trailing_stop_pct : float
        Trailing stop loss percentage
    take_profit_pct : float, optional
        Take profit percentage
    position_size_pct : float
        Fraction of equity to use per trade
        
    Returns:
    --------
    CustomMLStrategy : type
        Custom strategy class
    """
    
    class CustomMLStrategy(Strategy):
        """Custom ML Strategy with embedded model."""
        
        def init(self):
            """Initialize strategy with ML predictions."""
            # Get features from data
            feature_data = np.column_stack([
                getattr(self.data, col) for col in X_columns
            ])
            
            # Scale and predict
            X_scaled = scaler.transform(feature_data)
            predictions = model.predict(X_scaled)
            
            # Get probabilities
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_scaled)
                ml_probability = probabilities[:, 1]
            else:
                ml_probability = predictions.astype(float)
            
            # Store as indicators
            self.ml_signal = self.I(lambda: predictions)
            self.ml_probability = self.I(lambda: ml_probability)
            
            # Track entry
            self.entry_price = None
            self.highest_price = None
        
        def next(self):
            """Trading logic."""
            current_price = self.data.Close[-1]
            signal = self.ml_signal[-1]
            probability = self.ml_probability[-1]
            
            if self.position:
                self._check_exit_conditions(current_price)
            elif signal == 1 and probability >= probability_threshold:
                size = self.equity * position_size_pct / current_price
                self.buy(size=size)
                self.entry_price = current_price
                self.highest_price = current_price
        
        def _check_exit_conditions(self, current_price: float):
            """Check exit conditions."""
            if not self.position or self.entry_price is None:
                return
            
            if current_price > self.highest_price:
                self.highest_price = current_price
            
            trailing_stop_price = self.highest_price * (1 - trailing_stop_pct / 100)
            
            if current_price <= trailing_stop_price:
                self.position.close()
                self.entry_price = None
                self.highest_price = None
                return
            
            if take_profit_pct is not None:
                take_profit_price = self.entry_price * (1 + take_profit_pct / 100)
                if current_price >= take_profit_price:
                    self.position.close()
                    self.entry_price = None
                    self.highest_price = None
    
    return CustomMLStrategy
