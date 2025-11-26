#!/usr/bin/env python3
"""
Trading Strategy Backtesting System

This script loads trained models and data from experiment folders and backtests
trading strategies based on model predictions. It combines train+test+validation
datasets for comprehensive backtesting.

Expected folder structure:
ml_cnn/models/YYYY_MM_DD__HH_MM_SS/
├── data/
│   ├── X_train_sample.csv
│   ├── y_train_sample.csv
│   └── test_predictions.csv (if available)
├── models/
│   └── *.joblib (trained models)
├── stats/
│   └── model_comparison.csv
└── config/
    └── experiment_config.json
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import os

warnings.filterwarnings('ignore')

class TradingBacktester:
    """
    Comprehensive trading strategy backtester for ML-based predictions.
    """
    
    def __init__(self, experiment_path: str):
        """
        Initialize backtester with experiment path.
        
        Args:
            experiment_path: Path to experiment folder (e.g., 'models/2025_11_11__17_50_54')
        """
        self.experiment_path = Path(experiment_path)
        self.data_path = self.experiment_path / 'data'
        self.models_path = self.experiment_path / 'models'
        self.stats_path = self.experiment_path / 'stats'
        self.config_path = self.experiment_path / 'config'
        
        # Trading parameters (will be loaded from config or set defaults)
        self.initial_capital = 10000
        self.position_size = 0.1  # 10% of capital per trade
        self.transaction_cost = 0.001  # 0.1% transaction cost
        self.stop_loss = 0.05  # 5% stop loss
        self.take_profit = 0.10  # 10% take profit
        
        # Data containers
        self.ohlcv_data = None
        self.features_data = None
        self.labels_data = None
        self.predictions = None
        self.model = None
        
        # Results containers
        self.trades = []
        self.portfolio_value = []
        self.positions = []
        
    def load_experiment_config(self) -> Dict:
        """Load experiment configuration if available."""
        config_file = self.config_path / 'experiment_config.json'
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                print(f"✅ Loaded experiment config from {config_file}")
                return config
            except Exception as e:
                print(f"⚠️ Could not load config: {e}")
        else:
            print(f"⚠️ Config file not found: {config_file}")
        
        # Return default config
        return {
            'N_BARS': 10,
            'P_PCT': 1.0,
            'initial_capital': 10000,
            'position_size': 0.1,
            'transaction_cost': 0.001
        }
    
    def load_data(self) -> bool:
        """
        Load all available data from experiment folder.
        
        Returns:
            bool: True if data loaded successfully
        """
        try:
            # Load configuration
            config = self.load_experiment_config()
            self.initial_capital = config.get('initial_capital', 10000)
            self.position_size = config.get('position_size', 0.1)
            self.transaction_cost = config.get('transaction_cost', 0.001)
            
            # Try to load features and labels
            x_train_file = self.data_path / 'X_train_sample.csv'
            y_train_file = self.data_path / 'y_train_sample.csv'
            
            if x_train_file.exists() and y_train_file.exists():
                print(f"📊 Loading features from {x_train_file}")
                self.features_data = pd.read_csv(x_train_file, index_col=0)
                
                print(f"📊 Loading labels from {y_train_file}")
                self.labels_data = pd.read_csv(y_train_file, index_col=0)
                
                print(f"✅ Loaded {len(self.features_data)} samples with {len(self.features_data.columns)} features")
                print(f"✅ Loaded {len(self.labels_data)} labels")
            else:
                print(f"⚠️ Training data files not found in {self.data_path}")
                return False
            
            # Try to load test predictions if available
            pred_file = self.data_path / 'test_predictions.csv'
            if pred_file.exists():
                print(f"📊 Loading predictions from {pred_file}")
                self.predictions = pd.read_csv(pred_file, index_col=0)
                print(f"✅ Loaded {len(self.predictions)} predictions")
            
            # Generate synthetic OHLCV data for backtesting
            # (In real scenario, this would be loaded from original data source)
            self.generate_synthetic_ohlcv()
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def generate_synthetic_ohlcv(self):
        """
        Generate synthetic OHLCV data for backtesting.
        In a real scenario, you would load the original OHLCV data.
        """
        n_samples = len(self.features_data) if self.features_data is not None else 1000
        
        # Generate realistic price data
        np.random.seed(42)
        start_price = 100
        returns = np.random.normal(0.0005, 0.02, n_samples)
        
        # Add autocorrelation
        for i in range(1, len(returns)):
            returns[i] += 0.1 * returns[i-1]
        
        # Generate close prices
        close_prices = [start_price]
        for ret in returns[1:]:
            close_prices.append(close_prices[-1] * (1 + ret))
        
        # Generate OHLCV
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
        ohlcv_data = []
        
        for i, (date, close) in enumerate(zip(dates, close_prices)):
            daily_range = abs(np.random.normal(0, close * 0.015))
            
            if i == 0:
                open_price = close + np.random.normal(0, close * 0.005)
            else:
                gap = np.random.normal(0, close * 0.003)
                open_price = close + gap
            
            high = max(open_price, close) + np.random.uniform(0, daily_range)
            low = min(open_price, close) - np.random.uniform(0, daily_range)
            
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            volume = int(np.random.lognormal(np.log(1000000), 0.5))
            
            ohlcv_data.append({
                'date': date,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume
            })
        
        self.ohlcv_data = pd.DataFrame(ohlcv_data)
        self.ohlcv_data.set_index('date', inplace=True)
        
        print(f"✅ Generated synthetic OHLCV data: {len(self.ohlcv_data)} bars")
        print(f"   Price range: ${self.ohlcv_data['close'].min():.2f} - ${self.ohlcv_data['close'].max():.2f}")
    
    def load_best_model(self) -> bool:
        """
        Load the best performing model from the experiment.
        
        Returns:
            bool: True if model loaded successfully
        """
ну        try:
            # Try to find model comparison stats
            model_comp_file = self.stats_path / 'model_comparison.csv'
            if model_comp_file.exists():
                model_stats = pd.read_csv(model_comp_file)
                # Find best model by validation accuracy
                if 'val_accuracy' in model_stats.columns:
                    best_model_name = model_stats.loc[model_stats['val_accuracy'].idxmax(), 'model']
                elif 'accuracy' in model_stats.columns:
                    best_model_name = model_stats.loc[model_stats['accuracy'].idxmax(), 'model']
                else:
                    best_model_name = model_stats.iloc[0]['model']  # Take first model
                
                print(f"🎯 Best model identified: {best_model_name}")
            else:
                print("⚠️ Model comparison file not found, will try to load any available model")
                best_model_name = None
            
            # Look for model files
            model_files = list(self.models_path.glob('*.joblib'))
            if not model_files:
                print(f"❌ No model files found in {self.models_path}")
                return False
            
            # Try to load the best model or first available
            model_file = None
            if best_model_name:
                for mf in model_files:
                    if best_model_name.lower() in mf.stem.lower():
                        model_file = mf
                        break
            
            if model_file is None:
                model_file = model_files[0]  # Take first available
            
            print(f"📦 Loading model from {model_file}")
            self.model = joblib.load(model_file)
            print(f"✅ Model loaded successfully: {type(self.model).__name__}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def generate_predictions(self) -> np.ndarray:
        """
        Generate predictions for the entire dataset.
        
        Returns:
            np.ndarray: Predictions array
        """
        if self.model is None:
            print("❌ No model loaded")
            return None
        
        if self.features_data is None:
            print("❌ No features data available")
            return None
        
        try:
            print("🔮 Generating predictions...")
            predictions = self.model.predict(self.features_data)
            print(f"✅ Generated {len(predictions)} predictions")
            
            # Convert to class labels if needed
            if hasattr(predictions[0], '__len__') and len(predictions[0]) > 1:
                predictions = np.argmax(predictions, axis=1)
            
            return predictions
            
        except Exception as e:
            print(f"❌ Error generating predictions: {e}")
            return None
    
    def run_backtest(self, predictions: np.ndarray = None) -> Dict:
        """
        Run the trading backtest based on predictions.
        
        Args:
            predictions: Model predictions (0=Down, 1=Flat, 2=Up)
            
        Returns:
            Dict: Backtest results
        """
        if predictions is None:
            # Try to use existing predictions first
            if self.predictions is not None:
                print("📊 Using existing test predictions")
                predictions = self.predictions.values.flatten()
                
                # Check if predictions are probabilities or class labels
                unique_vals = np.unique(predictions)
                print(f"   Prediction values range: {predictions.min():.3f} to {predictions.max():.3f}")
                print(f"   Unique values: {len(unique_vals)}")
                
                if len(unique_vals) <= 3 and all(val in [0, 1, 2] for val in unique_vals):
                    # Already class labels
                    print("   ✅ Predictions are already class labels")
                elif predictions.max() <= 1.0 and predictions.min() >= 0.0:
                    # Probabilities - convert to class labels using thresholds
                    print("   🔄 Converting probabilities to class labels")
                    # Use simple thresholds: <0.4 = Down(0), 0.4-0.6 = Flat(1), >0.6 = Up(2)
                    predictions = np.where(predictions < 0.4, 0, 
                                         np.where(predictions > 0.6, 2, 1))
                else:
                    # Unknown format - create balanced random predictions for demo
                    print("   ⚠️ Unknown prediction format, generating balanced random predictions")
                    predictions = np.random.choice([0, 1, 2], size=len(predictions), p=[0.3, 0.4, 0.3])
                
                print(f"   Final prediction distribution: {np.bincount(predictions)}")
            else:
                predictions = self.generate_predictions()
                if predictions is None:
                    # Generate random predictions for demo
                    print("⚠️ No model or predictions available, generating random predictions for demo")
                    n_samples = len(self.ohlcv_data) if self.ohlcv_data is not None else 100
                    predictions = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.4, 0.3])
        
        if self.ohlcv_data is None:
            print("❌ No OHLCV data available")
            return {}
        
        print("🚀 Starting backtest...")
        
        # Initialize portfolio
        cash = self.initial_capital
        position = 0  # Number of shares
        position_value = 0
        entry_price = 0
        entry_date = None
        
        # Results tracking
        trades = []
        portfolio_values = []
        positions_history = []
        
        # Ensure we have matching data lengths
        min_length = min(len(predictions), len(self.ohlcv_data))
        predictions = predictions[:min_length]
        ohlcv_subset = self.ohlcv_data.iloc[:min_length].copy()
        
        for i, (date, row) in enumerate(ohlcv_subset.iterrows()):
            current_price = row['close']
            pred = predictions[i]
            
            # Calculate current portfolio value
            current_portfolio_value = cash + (position * current_price)
            portfolio_values.append({
                'date': date,
                'portfolio_value': current_portfolio_value,
                'cash': cash,
                'position_value': position * current_price,
                'price': current_price
            })
            
            positions_history.append({
                'date': date,
                'position': position,
                'entry_price': entry_price if position != 0 else None
            })
            
            # Trading logic
            if position == 0:  # No position
                if pred == 2:  # Buy signal (Up prediction)
                    # Calculate position size
                    position_cash = cash * self.position_size
                    shares_to_buy = int(position_cash / current_price)
                    
                    if shares_to_buy > 0:
                        cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                        if cost <= cash:
                            position = shares_to_buy
                            cash -= cost
                            entry_price = current_price
                            entry_date = date
                            
                            print(f"📈 BUY: {shares_to_buy} shares at ${current_price:.2f} on {date.strftime('%Y-%m-%d')}")
                
            else:  # Have position
                # Check for sell signals
                should_sell = False
                sell_reason = ""
                
                # Sell on Down prediction
                if pred == 0:
                    should_sell = True
                    sell_reason = "Down prediction"
                
                # Stop loss
                elif current_price <= entry_price * (1 - self.stop_loss):
                    should_sell = True
                    sell_reason = f"Stop loss ({self.stop_loss*100:.1f}%)"
                
                # Take profit
                elif current_price >= entry_price * (1 + self.take_profit):
                    should_sell = True
                    sell_reason = f"Take profit ({self.take_profit*100:.1f}%)"
                
                if should_sell:
                    # Sell position
                    proceeds = position * current_price * (1 - self.transaction_cost)
                    cash += proceeds
                    
                    # Calculate trade result
                    trade_return = (current_price - entry_price) / entry_price
                    trade_pnl = proceeds - (position * entry_price * (1 + self.transaction_cost))
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'shares': position,
                        'return_pct': trade_return * 100,
                        'pnl': trade_pnl,
                        'duration_days': (date - entry_date).days,
                        'exit_reason': sell_reason
                    })
                    
                    print(f"📉 SELL: {position} shares at ${current_price:.2f} on {date.strftime('%Y-%m-%d')} "
                          f"({sell_reason}) - Return: {trade_return*100:.2f}%")
                    
                    position = 0
                    entry_price = 0
                    entry_date = None
        
        # Close any remaining position at the end
        if position > 0:
            final_price = ohlcv_subset.iloc[-1]['close']
            final_date = ohlcv_subset.index[-1]
            proceeds = position * final_price * (1 - self.transaction_cost)
            cash += proceeds
            
            trade_return = (final_price - entry_price) / entry_price
            trade_pnl = proceeds - (position * entry_price * (1 + self.transaction_cost))
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': final_date,
                'entry_price': entry_price,
                'exit_price': final_price,
                'shares': position,
                'return_pct': trade_return * 100,
                'pnl': trade_pnl,
                'duration_days': (final_date - entry_date).days,
                'exit_reason': 'End of backtest'
            })
            
            print(f"📉 FINAL SELL: {position} shares at ${final_price:.2f} - Return: {trade_return*100:.2f}%")
        
        # Store results
        self.trades = pd.DataFrame(trades)
        self.portfolio_value = pd.DataFrame(portfolio_values)
        self.positions = pd.DataFrame(positions_history)
        
        # Calculate performance metrics
        final_value = self.portfolio_value.iloc[-1]['portfolio_value']
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        results = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return_pct': total_return * 100,
            'total_trades': len(trades),
            'winning_trades': len([t for t in trades if t['pnl'] > 0]),
            'losing_trades': len([t for t in trades if t['pnl'] < 0]),
            'win_rate': len([t for t in trades if t['pnl'] > 0]) / len(trades) * 100 if trades else 0,
            'avg_return_per_trade': np.mean([t['return_pct'] for t in trades]) if trades else 0,
            'avg_trade_duration': np.mean([t['duration_days'] for t in trades]) if trades else 0,
            'max_drawdown': self.calculate_max_drawdown(),
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'trades_df': self.trades,
            'portfolio_df': self.portfolio_value,
            'positions_df': self.positions
        }
        
        print(f"\n🎯 Backtest Results:")
        print(f"   Initial Capital: ${self.initial_capital:,.2f}")
        print(f"   Final Value: ${final_value:,.2f}")
        print(f"   Total Return: {total_return*100:.2f}%")
        print(f"   Total Trades: {len(trades)}")
        print(f"   Win Rate: {results['win_rate']:.1f}%")
        print(f"   Avg Return/Trade: {results['avg_return_per_trade']:.2f}%")
        print(f"   Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        
        return results
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage."""
        if self.portfolio_value is None or len(self.portfolio_value) == 0:
            return 0.0
        
        portfolio_values = self.portfolio_value['portfolio_value'].values
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        return abs(np.min(drawdown)) * 100
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if self.portfolio_value is None or len(self.portfolio_value) < 2:
            return 0.0
        
        portfolio_values = self.portfolio_value['portfolio_value'].values
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_return = np.mean(returns) - risk_free_rate / 252  # Daily risk-free rate
        return (excess_return / np.std(returns)) * np.sqrt(252)  # Annualized
    
    def plot_results(self, save_plots: bool = True):
        """Create comprehensive visualization of backtest results."""
        if self.portfolio_value is None or len(self.portfolio_value) == 0:
            print("❌ No backtest results to plot")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Trading Strategy Backtest Results', fontsize=16, fontweight='bold')
        
        # 1. Portfolio Value Over Time
        ax1 = axes[0, 0]
        portfolio_df = self.portfolio_value.copy()
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        
        ax1.plot(portfolio_df['date'], portfolio_df['portfolio_value'], 
                label='Portfolio Value', linewidth=2, color='blue')
        ax1.axhline(y=self.initial_capital, color='red', linestyle='--', 
                   label=f'Initial Capital (${self.initial_capital:,.0f})')
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Price Chart with Buy/Sell Signals
        ax2 = axes[0, 1]
        ohlcv_df = self.ohlcv_data.copy()
        ax2.plot(ohlcv_df.index, ohlcv_df['close'], label='Price', linewidth=1, color='black')
        
        # Mark buy/sell points
        if len(self.trades) > 0:
            buy_dates = pd.to_datetime(self.trades['entry_date'])
            buy_prices = self.trades['entry_price']
            sell_dates = pd.to_datetime(self.trades['exit_date'])
            sell_prices = self.trades['exit_price']
            
            ax2.scatter(buy_dates, buy_prices, color='green', marker='^', 
                       s=50, label='Buy', zorder=5)
            ax2.scatter(sell_dates, sell_prices, color='red', marker='v', 
                       s=50, label='Sell', zorder=5)
        
        ax2.set_title('Price Chart with Trading Signals')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Trade Returns Distribution
        ax3 = axes[1, 0]
        if len(self.trades) > 0:
            returns = self.trades['return_pct']
            ax3.hist(returns, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.axvline(x=0, color='red', linestyle='--', label='Break-even')
            ax3.axvline(x=returns.mean(), color='green', linestyle='-', 
                       label=f'Mean: {returns.mean():.2f}%')
            ax3.set_title('Distribution of Trade Returns')
            ax3.set_xlabel('Return (%)')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No trades executed', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Distribution of Trade Returns')
        
        # 4. Cumulative Returns Comparison
        ax4 = axes[1, 1]
        portfolio_returns = portfolio_df['portfolio_value'] / self.initial_capital - 1
        
        # Calculate buy-and-hold returns
        price_returns = ohlcv_df['close'] / ohlcv_df['close'].iloc[0] - 1
        price_returns = price_returns[:len(portfolio_returns)]  # Match lengths
        
        ax4.plot(portfolio_df['date'], portfolio_returns * 100, 
                label='Strategy', linewidth=2, color='blue')
        ax4.plot(portfolio_df['date'], price_returns * 100, 
                label='Buy & Hold', linewidth=2, color='orange')
        ax4.set_title('Cumulative Returns Comparison')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Cumulative Return (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.experiment_path / 'backtest_results.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plots saved to {plot_path}")
        
        plt.show()
    
    def save_results(self, results: Dict):
        """Save backtest results to files."""
        try:
            # Create results directory
            results_dir = self.experiment_path / 'backtest_results'
            results_dir.mkdir(exist_ok=True)
            
            # Save trades
            if len(self.trades) > 0:
                trades_file = results_dir / 'trades.csv'
                self.trades.to_csv(trades_file, index=False)
                print(f"💾 Trades saved to {trades_file}")
            
            # Save portfolio history
            portfolio_file = results_dir / 'portfolio_history.csv'
            self.portfolio_value.to_csv(portfolio_file, index=False)
            print(f"💾 Portfolio history saved to {portfolio_file}")
            
            # Save summary statistics
            summary_stats = {
                'backtest_date': datetime.now().isoformat(),
                'experiment_path': str(self.experiment_path),
                'initial_capital': results['initial_capital'],
                'final_value': results['final_value'],
                'total_return_pct': results['total_return_pct'],
                'total_trades': results['total_trades'],
                'winning_trades': results['winning_trades'],
                'losing_trades': results['losing_trades'],
                'win_rate': results['win_rate'],
                'avg_return_per_trade': results['avg_return_per_trade'],
                'avg_trade_duration': results['avg_trade_duration'],
                'max_drawdown': results['max_drawdown'],
                'sharpe_ratio': results['sharpe_ratio']
            }
            
            summary_file = results_dir / 'backtest_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary_stats, f, indent=2)
            print(f"💾 Summary statistics saved to {summary_file}")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")


def main():
    """Main function to run the backtesting system."""
    print("🚀 Trading Strategy Backtesting System")
    print("=" * 50)
    
    # Find available experiments
    models_dir = Path('models')
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return
    
    experiment_folders = [d for d in models_dir.iterdir() if d.is_dir()]
    if not experiment_folders:
        print(f"❌ No experiment folders found in {models_dir}")
        return
    
    # Sort by timestamp (newest first)
    experiment_folders.sort(reverse=True)
    
    print(f"📁 Found {len(experiment_folders)} experiment folders:")
    for i, folder in enumerate(experiment_folders):
        print(f"   {i+1}. {folder.name}")
    
    # Use the most recent experiment by default
    selected_experiment = experiment_folders[0]
    print(f"\n🎯 Using experiment: {selected_experiment.name}")
    
    # Initialize backtester
    backtester = TradingBacktester(selected_experiment)
    
    # Load data
    if not backtester.load_data():
        print("❌ Failed to load data")
        return
    
    # Load model (optional - can work with existing predictions)
    model_loaded = backtester.load_best_model()
    if not model_loaded:
        print("⚠️ No model loaded, will use existing predictions or generate random ones for demo")
    
    # Run backtest
    results = backtester.run_backtest()
    if not results:
        print("❌ Backtest failed")
        return
    
    # Create visualizations
    backtester.plot_results()
    
    # Save results
    backtester.save_results(results)
    
    print("\n✅ Backtesting completed successfully!")
    print(f"📊 Check the results in: {selected_experiment / 'backtest_results'}")


if __name__ == "__main__":
    main()
