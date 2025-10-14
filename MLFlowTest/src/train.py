"""Training script for OHLC prediction models."""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.data import DataFetcher, DataPreprocessor
    from src.features import FeatureEngineer
    from src.models import LSTMModel, RandomForestModel, XGBoostModel, EnsembleModel
    from src.mlflow_tracker import MLflowTracker, track_experiment
    from src.utils import config, get_logger, EnvConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    print("Try: python src/train.py --symbol AAPL --model rf")
    sys.exit(1)

logger = get_logger(__name__)

class ModelTrainer:
    """Main training class for OHLC prediction models."""
    
    def __init__(self, symbol: str, model_type: str):
        """Initialize trainer.
        
        Args:
            symbol: Stock symbol to train on
            model_type: Type of model ('lstm', 'rf', 'xgb', 'ensemble')
        """
        self.symbol = symbol
        self.model_type = model_type
        self.config = config
        
        # Initialize components
        self.data_fetcher = DataFetcher()
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.mlflow_tracker = MLflowTracker()
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
    def load_or_fetch_data(self, period: str = "2y", force_fetch: bool = False) -> pd.DataFrame:
        """Load existing data or fetch new data.
        
        Args:
            period: Data period to fetch
            force_fetch: Whether to force fetch new data
            
        Returns:
            Raw market data
        """
        logger.info(f"Loading data for {self.symbol}")
        
        if force_fetch:
            # Force fetch new data
            logger.info(f"Force fetching new data for {self.symbol}")
            self.raw_data = self.data_fetcher.fetch_symbol_data(
                symbol=self.symbol,
                period=period,
                save_to_file=True
            )
        else:
            try:
                # Try to load existing data
                self.raw_data = self.data_fetcher.load_saved_data(self.symbol)
                logger.info(f"Loaded existing data: {self.raw_data.shape}")
            except FileNotFoundError:
                # Fetch new data if no existing data found
                logger.info(f"No existing data found. Fetching new data for {self.symbol}")
                self.raw_data = self.data_fetcher.fetch_symbol_data(
                    symbol=self.symbol,
                    period=period,
                    save_to_file=True
                )
        
        logger.info(f"Data loaded: {self.raw_data.shape}")
        return self.raw_data
    
    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess the raw data.
        
        Returns:
            Preprocessed data
        """
        logger.info("Starting data preprocessing")
        
        # Clean data
        cleaned_data = self.preprocessor.clean_data(self.raw_data)
        
        # Engineer features
        self.processed_data = self.feature_engineer.engineer_features(cleaned_data)
        
        # Remove rows with NaN values (from feature engineering)
        initial_shape = self.processed_data.shape
        self.processed_data = self.processed_data.dropna()
        logger.info(f"Removed {initial_shape[0] - self.processed_data.shape[0]} rows with NaN values")
        
        logger.info(f"Preprocessing completed: {self.processed_data.shape}")
        return self.processed_data
    
    def prepare_model_data(self) -> Tuple[np.ndarray, ...]:
        """Prepare data for model training.
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Preparing data for model training")
        
        # Get target columns
        target_columns = self.config.training_config.get('target_columns', ['Close'])
        
        # Prepare features and targets
        feature_columns = [col for col in self.processed_data.columns 
                          if col not in ['Symbol'] + target_columns]
        
        if self.model_type == 'lstm':
            # For LSTM, create sequences
            sequence_length = self.config.model_config.get('lstm', {}).get('sequence_length', 60)
            
            # Scale data first
            scaled_data = self.preprocessor.scale_data(self.processed_data, fit_scaler=True)
            
            # Create sequences
            X, y = self.preprocessor.create_sequences(
                scaled_data,
                sequence_length=sequence_length,
                target_columns=target_columns,
                prediction_horizon=1
            )
            
            # Split data
            test_size = self.config.training_config.get('test_size', 0.2)
            validation_size = self.config.training_config.get('validation_size', 0.1)
            
            self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = \
                self.preprocessor.split_data(X, y, test_size, validation_size, shuffle=False)
        
        else:
            # For non-sequence models
            X = self.processed_data[feature_columns].values
            y = self.processed_data[target_columns].values
            
            # Scale data
            X_scaled = self.preprocessor.scale_data(
                pd.DataFrame(X, columns=feature_columns), 
                fit_scaler=True
            ).values
            
            # Split data
            test_size = self.config.training_config.get('test_size', 0.2)
            validation_size = self.config.training_config.get('validation_size', 0.1)
            
            self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = \
                self.preprocessor.split_data(X_scaled, y, test_size, validation_size, shuffle=False)
        
        logger.info(f"Data preparation completed:")
        logger.info(f"  Training: X={self.X_train.shape}, y={self.y_train.shape}")
        logger.info(f"  Validation: X={self.X_val.shape}, y={self.y_val.shape}")
        logger.info(f"  Test: X={self.X_test.shape}, y={self.y_test.shape}")
        
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def create_model(self) -> Any:
        """Create model based on type.
        
        Returns:
            Model instance
        """
        logger.info(f"Creating {self.model_type} model")
        
        if self.model_type == 'lstm':
            model_config = self.config.model_config.get('lstm', {})
            model = LSTMModel(model_config)
            
        elif self.model_type == 'rf' or self.model_type == 'random_forest':
            model_config = self.config.model_config.get('random_forest', {})
            model = RandomForestModel(model_config)
            
        elif self.model_type == 'xgb' or self.model_type == 'xgboost':
            model_config = self.config.model_config.get('xgboost', {})
            model = XGBoostModel(model_config)
            
        elif self.model_type == 'ensemble':
            # Create ensemble with multiple models
            ensemble_config = {'method': 'weighted_average'}
            model = EnsembleModel(ensemble_config)
            
            # Add base models
            lstm_config = self.config.model_config.get('lstm', {})
            rf_config = self.config.model_config.get('random_forest', {})
            xgb_config = self.config.model_config.get('xgboost', {})
            
            model.add_model('lstm', LSTMModel(lstm_config))
            model.add_model('random_forest', RandomForestModel(rf_config))
            model.add_model('xgboost', XGBoostModel(xgb_config))
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model
    
    def train_model(self, model: Any) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            model: Model to train
            
        Returns:
            Training results
        """
        logger.info(f"Training {self.model_type} model")
        
        # Train model
        training_results = model.train(
            self.X_train, 
            self.y_train, 
            self.X_val, 
            self.y_val
        )
        
        logger.info("Model training completed")
        return training_results
    
    def evaluate_model(self, model: Any) -> Dict[str, float]:
        """Evaluate the trained model.
        
        Args:
            model: Trained model
            
        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating model")
        
        # Evaluate on test set
        metrics = model.evaluate(self.X_test, self.y_test)
        
        logger.info("Model evaluation completed")
        logger.info(f"Test metrics: {metrics}")
        
        return metrics
    
    def save_model(self, model: Any, model_name: Optional[str] = None) -> str:
        """Save the trained model.
        
        Args:
            model: Trained model
            model_name: Optional model name
            
        Returns:
            Path to saved model
        """
        if model_name is None:
            model_name = f"{self.symbol}_{self.model_type}_model"
        
        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        model_path = models_dir / f"{model_name}.pkl"
        model.save_model(str(model_path))
        
        logger.info(f"Model saved to {model_path}")
        return str(model_path)
    
    def run_training_pipeline(
        self, 
        period: str = "2y", 
        force_fetch: bool = False,
        save_model: bool = True,
        track_mlflow: bool = True
    ) -> Dict[str, Any]:
        """Run the complete training pipeline.
        
        Args:
            period: Data period to fetch
            force_fetch: Whether to force fetch new data
            save_model: Whether to save the trained model
            track_mlflow: Whether to track with MLflow
            
        Returns:
            Training results
        """
        logger.info(f"Starting training pipeline for {self.symbol} with {self.model_type}")
        
        results = {}
        run_id = None
        
        try:
            if track_mlflow:
                # Start MLflow run
                run_name = f"{self.symbol}_{self.model_type}"
                tags = {
                    "symbol": self.symbol,
                    "model_type": self.model_type,
                    "data_period": period
                }
                run_id = self.mlflow_tracker.start_run(run_name=run_name, tags=tags)
            
            # Step 1: Load/fetch data
            self.load_or_fetch_data(period=period, force_fetch=force_fetch)
            
            # Step 2: Preprocess data
            self.preprocess_data()
            
            # Step 3: Prepare model data
            self.prepare_model_data()
            
            # Step 4: Create model
            model = self.create_model()
            
            # Step 5: Train model
            training_results = self.train_model(model)
            results['training'] = training_results
            
            # Step 6: Evaluate model
            evaluation_metrics = self.evaluate_model(model)
            results['evaluation'] = evaluation_metrics
            
            # Step 7: Save model
            if save_model:
                model_path = self.save_model(model)
                results['model_path'] = model_path
            
            # Step 8: MLflow tracking
            if track_mlflow:
                # Log parameters
                params = {
                    'symbol': self.symbol,
                    'model_type': self.model_type,
                    'data_period': period,
                    'data_shape': str(self.processed_data.shape),
                    'n_features': self.X_train.shape[-1] if len(self.X_train.shape) > 1 else self.X_train.shape[1]
                }
                params.update(model.get_model_info())
                self.mlflow_tracker.log_params(params)
                
                # Log metrics
                self.mlflow_tracker.log_metrics(evaluation_metrics)
                
                # Log training history if available
                if 'history' in training_results:
                    self.mlflow_tracker.log_training_history(training_results['history'])
                
                # Log model
                self.mlflow_tracker.log_model(model, self.model_type)
                
                # Log plots
                y_pred = model.predict(self.X_test)
                self.mlflow_tracker.log_predictions_plot(self.y_test, y_pred)
                self.mlflow_tracker.log_residuals_plot(self.y_test, y_pred)
                
                # Log feature importance if available
                if hasattr(model, 'get_feature_importance'):
                    try:
                        importance = model.get_feature_importance()
                        if importance:
                            self.mlflow_tracker.log_feature_importance_plot(importance)
                    except Exception as e:
                        logger.warning(f"Could not log feature importance: {str(e)}")
                
                results['mlflow_run_id'] = run_id
            
            logger.info("Training pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {str(e)}")
            raise
            
        finally:
            if track_mlflow and run_id:
                self.mlflow_tracker.end_run()

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Train OHLC prediction models")
    parser.add_argument("--symbol", required=True, help="Stock symbol")
    parser.add_argument("--model", required=True, 
                       choices=['lstm', 'rf', 'random_forest', 'xgb', 'xgboost', 'ensemble'],
                       help="Model type")
    parser.add_argument("--period", default="2y", help="Data period")
    parser.add_argument("--force-fetch", action="store_true", help="Force fetch new data")
    parser.add_argument("--no-save", action="store_true", help="Don't save model")
    parser.add_argument("--no-mlflow", action="store_true", help="Don't track with MLflow")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = ModelTrainer(args.symbol, args.model)
    
    # Run training
    try:
        results = trainer.run_training_pipeline(
            period=args.period,
            force_fetch=args.force_fetch,
            save_model=not args.no_save,
            track_mlflow=not args.no_mlflow
        )
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Symbol: {args.symbol}")
        print(f"Model: {args.model}")
        print(f"Data Period: {args.period}")
        
        if 'evaluation' in results:
            print("\nEvaluation Metrics:")
            for metric, value in results['evaluation'].items():
                print(f"  {metric}: {value:.4f}")
        
        if 'model_path' in results:
            print(f"\nModel saved to: {results['model_path']}")
        
        if 'mlflow_run_id' in results:
            print(f"MLflow Run ID: {results['mlflow_run_id']}")
            print(f"MLflow UI: {EnvConfig.MLFLOW_TRACKING_URI}")
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
