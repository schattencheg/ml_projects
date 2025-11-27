"""
PipelineManager - Orchestrates the complete ML pipeline.
Helps create and manage the workflow for ML tasks.
"""

import pandas as pd
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

from src.managers.model_manager import ModelManager
from src.managers.train_manager import TrainManager
from src.managers.scaler_manager import ScalerManager
from src.managers.mlflow_manager import MLFlowManager
from src.managers.backtest_manager import BacktestManager
from src.managers.result_manager import ResultManager
from src.managers.visualization_manager import VisualizationManager
from src.data_provider import DataProvider
from src.features_generator import FeaturesGenerator


class PipelineManager:
    """
    Orchestrates the complete ML pipeline workflow.
    
    Features:
    - End-to-end pipeline management
    - Automatic directory structure creation
    - Coordinated execution of all managers
    - Comprehensive result tracking
    """
    
    def __init__(self, 
                 results_dir: str = 'results',
                 use_mlflow: bool = False,
                 mlflow_uri: str = "http://localhost:5000"):
        """
        Initialize PipelineManager.
        
        Args:
            results_dir: Base directory for results
            use_mlflow: Whether to use MLFlow tracking
            mlflow_uri: MLFlow tracking server URI
        """
        self.results_dir = Path(results_dir)
        self.use_mlflow = use_mlflow
        self.mlflow_uri = mlflow_uri
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.run_dir = self.results_dir / self.timestamp
        
        # Initialize managers
        self.model_manager = ModelManager(results_dir=str(self.results_dir))
        self.train_manager = TrainManager()
        self.scaler_manager = None
        self.mlflow_manager = None
        self.backtest_manager = None
        self.result_manager = ResultManager()
        self.visualization_manager = VisualizationManager()
        
        # Data components
        self.data_provider = DataProvider()
        self.features_generator = FeaturesGenerator()
        
        # Pipeline state
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.feature_cols = None
        self.target_col = 'target'
        
        print(f"\n{'='*70}")
        print(f"PIPELINE INITIALIZED")
        print(f"{'='*70}")
        print(f"Run directory: {self.run_dir}")
        print(f"{'='*70}\n")
    
    def setup_data(self,
                  ticker: str,
                  start_date: str,
                  end_date: str,
                  timeframe: str = '1d',
                  feature_set: str = 'basic',
                  target_params: Optional[Dict] = None) -> 'PipelineManager':
        """
        Setup data: download, generate features, create target.
        
        Args:
            ticker: Ticker symbol
            start_date: Start date
            end_date: End date
            timeframe: Timeframe (not used with yfinance, placeholder for future)
            feature_set: Feature set to generate
            target_params: Parameters for target creation
            
        Returns:
            Self for chaining
        """
        print("\n" + "="*70)
        print("SETTING UP DATA")
        print("="*70)
        
        # Download data
        print(f"\nDownloading {ticker} from {start_date} to {end_date}...")
        df = self.data_provider.load_yahoo(ticker, start_date, end_date)
        
        # Clean data
        df = self.data_provider.clean_data(df)
        
        # Generate features
        print(f"\nGenerating features (set: {feature_set})...")
        df = self.features_generator.generate_features(df, feature_set=feature_set)
        
        # Create target
        if target_params is None:
            target_params = {'future_bars': 5, 'threshold': 0.02}
        
        print(f"\nCreating target...")
        df = self.features_generator.create_target(df, **target_params)
        
        # Split data
        print(f"\nSplitting data...")
        self.train_data, self.val_data, self.test_data = self.data_provider.split_data(
            df, train_size=0.7, val_size=0.15, test_size=0.15
        )
        
        # Get feature columns
        self.feature_cols = self.features_generator.get_feature_names()
        
        print(f"\n✓ Data setup complete")
        print(f"  Train samples: {len(self.train_data)}")
        print(f"  Val samples:   {len(self.val_data)}")
        print(f"  Test samples:  {len(self.test_data)}")
        print(f"  Features:      {len(self.feature_cols)}")
        print("="*70 + "\n")
        
        return self
    
    def setup_models(self, 
                    model_names: Optional[List[str]] = None,
                    enable_all: bool = False) -> 'PipelineManager':
        """
        Setup models to train.
        
        Args:
            model_names: List of model names to enable (None = use defaults)
            enable_all: Enable all available models
            
        Returns:
            Self for chaining
        """
        if enable_all:
            for model_name in self.model_manager.model_config.keys():
                self.model_manager.enable_model(model_name, True)
        elif model_names:
            # Disable all first
            for model_name in self.model_manager.model_config.keys():
                self.model_manager.enable_model(model_name, False)
            # Enable specified
            for model_name in model_names:
                self.model_manager.enable_model(model_name, True)
        
        self.model_manager.print_config()
        
        return self
    
    def train_models(self, **train_kwargs) -> 'PipelineManager':
        """
        Train all enabled models.
        
        Args:
            **train_kwargs: Additional arguments for training
            
        Returns:
            Self for chaining
        """
        if self.train_data is None:
            raise ValueError("Data not setup. Call setup_data() first.")
        
        # Create models
        models = self.model_manager.create_models()
        
        # Start MLFlow run if enabled
        if self.use_mlflow:
            if self.mlflow_manager is None:
                self.mlflow_manager = MLFlowManager(tracking_uri=self.mlflow_uri)
                self.mlflow_manager.connect()
            self.mlflow_manager.start_run(run_name=self.timestamp)
        
        # Train models
        train_output = self.train_manager.train(
            models=models,
            train_data=self.train_data,
            target_col=self.target_col,
            feature_cols=self.feature_cols,
            val_data=self.val_data,
            **train_kwargs
        )
        
        # Store scaler
        self.scaler_manager = train_output['scaler_manager']
        
        # Add results to result manager
        self.result_manager.add_train_results(train_output['results'])
        
        # Log to MLFlow
        if self.use_mlflow and self.mlflow_manager:
            self.mlflow_manager.log_training_results(
                train_output['results'],
                self.model_manager.get_models()
            )
        
        return self
    
    def test_models(self) -> 'PipelineManager':
        """
        Test all trained models.
        
        Returns:
            Self for chaining
        """
        if self.test_data is None:
            raise ValueError("Data not setup. Call setup_data() first.")
        
        # Test models
        test_results = self.train_manager.test(
            test_data=self.test_data,
            target_col=self.target_col,
            feature_cols=self.feature_cols
        )
        
        # Add results to result manager
        self.result_manager.add_test_results(test_results)
        
        # Log to MLFlow
        if self.use_mlflow and self.mlflow_manager:
            self.mlflow_manager.log_test_results(test_results)
        
        return self
    
    def backtest_models(self, 
                       backend: str = 'nolib',
                       model_names: Optional[List[str]] = None,
                       **backtest_kwargs) -> 'PipelineManager':
        """
        Backtest trained models.
        
        Args:
            backend: Backtesting backend ('nolib', 'backtrader', 'backtesting')
            model_names: List of models to backtest (None = all trained)
            **backtest_kwargs: Additional backtest parameters
            
        Returns:
            Self for chaining
        """
        if self.test_data is None:
            raise ValueError("Data not setup. Call setup_data() first.")
        
        # Initialize backtest manager
        self.backtest_manager = BacktestManager(backend=backend, **backtest_kwargs)
        
        # Get models to backtest
        trained_models = self.train_manager.get_trained_models()
        if model_names:
            models_to_backtest = {name: trained_models[name] for name in model_names 
                                 if name in trained_models}
        else:
            models_to_backtest = trained_models
        
        # Run backtests
        for model_name, model in models_to_backtest.items():
            print(f"\nBacktesting {model_name}...")
            results = self.backtest_manager.run(
                data=self.test_data,
                model=model,
                scaler_manager=self.scaler_manager,
                feature_cols=self.feature_cols
            )
            
            # Add results
            self.result_manager.add_backtest_results(model_name, results)
            
            # Log to MLFlow
            if self.use_mlflow and self.mlflow_manager:
                self.mlflow_manager.log_backtest_results(results, model_name)
        
        return self
    
    def generate_reports(self) -> 'PipelineManager':
        """
        Generate all visualization reports.
        
        Returns:
            Self for chaining
        """
        # Create train report
        if self.result_manager.train_results:
            self.visualization_manager.create_train_report(
                self.result_manager.train_results,
                self.run_dir
            )
        
        # Create test report
        if self.result_manager.test_results:
            self.visualization_manager.create_test_report(
                self.result_manager.test_results,
                self.run_dir
            )
        
        # Create backtest report
        if self.result_manager.backtest_results:
            # Prepare feature info if available
            feature_info = None
            if hasattr(self, 'feature_selector') and self.feature_selector:
                feature_info = {
                    'total_features': len(self.feature_selector.all_features) if hasattr(self.feature_selector, 'all_features') else 0,
                    'selected_features': len(self.feature_selector.selected_features) if hasattr(self.feature_selector, 'selected_features') else 0,
                    'dropped_features': len(self.feature_selector.dropped_features) if hasattr(self.feature_selector, 'dropped_features') else 0,
                    'selection_method': self.feature_selector.method if hasattr(self.feature_selector, 'method') else 'Unknown',
                    'top_features': self.feature_selector.selected_features[:10] if hasattr(self.feature_selector, 'selected_features') else []
                }
            
            self.visualization_manager.create_backtest_report(
                self.result_manager.backtest_results,
                self.run_dir,
                test_results=self.result_manager.test_results,
                train_results=self.result_manager.train_results,
                feature_info=feature_info
            )
        
        return self
    
    def save_results(self) -> 'PipelineManager':
        """
        Save all results and artifacts.
        
        Returns:
            Self for chaining
        """
        print("\n" + "="*70)
        print("SAVING RESULTS AND ARTIFACTS")
        print("="*70)
        
        # Create directory structure
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        trained_models = self.train_manager.get_trained_models()
        if trained_models:
            self.model_manager.save_models(
                models=trained_models,
                save_dir=self.run_dir,
                metadata={
                    'timestamp': self.timestamp,
                    'feature_cols': self.feature_cols,
                    'target_col': self.target_col
                }
            )
        
        # Save scaler
        if self.scaler_manager:
            self.scaler_manager.save(self.run_dir)
        
        # Save results
        self.result_manager.save_results(self.run_dir)
        
        # Log artifacts to MLFlow
        if self.use_mlflow and self.mlflow_manager:
            self.mlflow_manager.log_artifacts(str(self.run_dir))
        
        print(f"\n✓ All results saved to: {self.run_dir}")
        print("="*70 + "\n")
        
        return self
    
    def run_complete_pipeline(self,
                             ticker: str,
                             start_date: str,
                             end_date: str,
                             model_names: Optional[List[str]] = None,
                             feature_set: str = 'basic',
                             backtest: bool = True,
                             backtest_backend: str = 'nolib') -> Dict[str, Any]:
        """
        Run the complete pipeline from start to finish.
        
        Args:
            ticker: Ticker symbol
            start_date: Start date
            end_date: End date
            model_names: Models to train (None = defaults)
            feature_set: Feature set to use
            backtest: Whether to run backtests
            backtest_backend: Backtesting backend
            
        Returns:
            Comprehensive results dictionary
        """
        # Setup data
        self.setup_data(ticker, start_date, end_date, feature_set=feature_set)
        
        # Setup models
        self.setup_models(model_names=model_names)
        
        # Train models
        self.train_models()
        
        # Test models
        self.test_models()
        
        # Backtest (optional)
        if backtest:
            self.backtest_models(backend=backtest_backend)
        
        # Generate reports
        self.generate_reports()
        
        # Save results
        self.save_results()
        
        # Print summary
        self.result_manager.print_summary()
        
        # End MLFlow run
        if self.use_mlflow and self.mlflow_manager:
            self.mlflow_manager.end_run()
        
        return self.result_manager.get_comprehensive_results()
    
    def get_run_directory(self) -> Path:
        """Get the run directory path."""
        return self.run_dir
    
    def get_results(self) -> Dict[str, Any]:
        """Get comprehensive results."""
        return self.result_manager.get_comprehensive_results()
