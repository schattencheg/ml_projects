"""Evaluation script for trained OHLC prediction models."""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data import DataFetcher, DataPreprocessor
from src.features import FeatureEngineer
from src.models import LSTMModel, RandomForestModel, XGBoostModel, EnsembleModel
from src.mlflow_tracker import MLflowTracker
from src.utils import config, get_logger

logger = get_logger(__name__)

class ModelEvaluator:
    """Model evaluation and analysis class."""
    
    def __init__(self, model_path: str, symbol: str):
        """Initialize evaluator.
        
        Args:
            model_path: Path to saved model
            symbol: Stock symbol
        """
        self.model_path = model_path
        self.symbol = symbol
        self.model = None
        self.model_type = None
        
        # Initialize components
        self.data_fetcher = DataFetcher()
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        
    def load_model(self) -> Any:
        """Load the trained model.
        
        Returns:
            Loaded model
        """
        logger.info(f"Loading model from {self.model_path}")
        
        # Determine model type from path
        if 'lstm' in self.model_path.lower():
            self.model = LSTMModel({})
            self.model_type = 'lstm'
        elif 'random_forest' in self.model_path.lower() or 'rf' in self.model_path.lower():
            self.model = RandomForestModel({})
            self.model_type = 'random_forest'
        elif 'xgboost' in self.model_path.lower() or 'xgb' in self.model_path.lower():
            self.model = XGBoostModel({})
            self.model_type = 'xgboost'
        elif 'ensemble' in self.model_path.lower():
            self.model = EnsembleModel({})
            self.model_type = 'ensemble'
        else:
            raise ValueError(f"Cannot determine model type from path: {self.model_path}")
        
        self.model.load_model(self.model_path)
        logger.info(f"Model loaded successfully: {self.model_type}")
        
        return self.model
    
    def prepare_test_data(self, period: str = "6mo") -> tuple:
        """Prepare fresh test data for evaluation.
        
        Args:
            period: Period of data to fetch for testing
            
        Returns:
            Tuple of (X_test, y_test, dates)
        """
        logger.info(f"Preparing test data for {self.symbol}")
        
        # Fetch fresh data
        raw_data = self.data_fetcher.fetch_symbol_data(
            symbol=self.symbol,
            period=period,
            save_to_file=False
        )
        
        # Preprocess data (same pipeline as training)
        cleaned_data = self.preprocessor.clean_data(raw_data)
        processed_data = self.feature_engineer.engineer_features(cleaned_data)
        processed_data = processed_data.dropna()
        
        # Prepare for model
        target_columns = config.training_config.get('target_columns', ['Close'])
        feature_columns = [col for col in processed_data.columns 
                          if col not in ['Symbol'] + target_columns]
        
        if self.model_type == 'lstm':
            # Load scalers and create sequences
            sequence_length = config.model_config.get('lstm', {}).get('sequence_length', 60)
            
            # Scale data (load existing scalers)
            try:
                self.preprocessor.load_scalers('models/scalers.pkl')
                scaled_data = self.preprocessor.scale_data(processed_data, fit_scaler=False)
            except:
                logger.warning("Could not load scalers, fitting new ones")
                scaled_data = self.preprocessor.scale_data(processed_data, fit_scaler=True)
            
            # Create sequences
            X, y = self.preprocessor.create_sequences(
                scaled_data,
                sequence_length=sequence_length,
                target_columns=target_columns,
                prediction_horizon=1
            )
            
            # Get corresponding dates
            dates = processed_data.index[sequence_length:]
            
        else:
            # For non-sequence models
            X = processed_data[feature_columns].values
            y = processed_data[target_columns].values
            dates = processed_data.index
            
            # Scale data
            try:
                self.preprocessor.load_scalers('models/scalers.pkl')
                X_scaled = self.preprocessor.scale_data(
                    pd.DataFrame(X, columns=feature_columns), 
                    fit_scaler=False
                ).values
            except:
                logger.warning("Could not load scalers, fitting new ones")
                X_scaled = self.preprocessor.scale_data(
                    pd.DataFrame(X, columns=feature_columns), 
                    fit_scaler=True
                ).values
            
            X = X_scaled
        
        logger.info(f"Test data prepared: X={X.shape}, y={y.shape}")
        return X, y, dates
    
    def evaluate_model_performance(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model performance")
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = self.model.evaluate(X_test, y_test)
        
        # Additional custom metrics
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            # Multi-output case
            for i, target in enumerate(['Open', 'High', 'Low', 'Close'][:y_test.shape[1]]):
                y_true_i = y_test[:, i]
                y_pred_i = y_pred[:, i]
                
                # Directional accuracy
                true_direction = np.diff(y_true_i) > 0
                pred_direction = np.diff(y_pred_i) > 0
                directional_accuracy = np.mean(true_direction == pred_direction)
                metrics[f'{target}_Directional_Accuracy'] = directional_accuracy
        else:
            # Single output case
            true_direction = np.diff(y_test.flatten()) > 0
            pred_direction = np.diff(y_pred.flatten()) > 0
            directional_accuracy = np.mean(true_direction == pred_direction)
            metrics['Directional_Accuracy'] = directional_accuracy
        
        return metrics
    
    def analyze_predictions(self, X_test: np.ndarray, y_test: np.ndarray, dates: pd.DatetimeIndex) -> Dict[str, Any]:
        """Analyze model predictions in detail.
        
        Args:
            X_test: Test features
            y_test: Test targets
            dates: Corresponding dates
            
        Returns:
            Analysis results
        """
        logger.info("Analyzing predictions")
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        
        # Create results DataFrame
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            target_columns = ['Open', 'High', 'Low', 'Close'][:y_test.shape[1]]
            
            results_df = pd.DataFrame(index=dates)
            for i, col in enumerate(target_columns):
                results_df[f'{col}_Actual'] = y_test[:, i]
                results_df[f'{col}_Predicted'] = y_pred[:, i]
                results_df[f'{col}_Error'] = y_test[:, i] - y_pred[:, i]
                results_df[f'{col}_Error_Pct'] = (y_test[:, i] - y_pred[:, i]) / y_test[:, i] * 100
        else:
            results_df = pd.DataFrame(index=dates)
            results_df['Actual'] = y_test.flatten()
            results_df['Predicted'] = y_pred.flatten()
            results_df['Error'] = y_test.flatten() - y_pred.flatten()
            results_df['Error_Pct'] = (y_test.flatten() - y_pred.flatten()) / y_test.flatten() * 100
        
        # Calculate analysis metrics
        analysis = {
            'results_df': results_df,
            'mean_absolute_error_pct': np.mean(np.abs(results_df.filter(regex='Error_Pct'))),
            'max_error_pct': np.max(np.abs(results_df.filter(regex='Error_Pct'))),
            'prediction_correlation': results_df.filter(regex='Actual').corrwith(results_df.filter(regex='Predicted')).mean()
        }
        
        return analysis
    
    def create_evaluation_plots(self, analysis: Dict[str, Any], save_path: str = "evaluation_plots") -> None:
        """Create evaluation plots.
        
        Args:
            analysis: Analysis results
            save_path: Directory to save plots
        """
        logger.info("Creating evaluation plots")
        
        Path(save_path).mkdir(exist_ok=True)
        results_df = analysis['results_df']
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Time series plot
        plt.figure(figsize=(15, 8))
        if 'Close_Actual' in results_df.columns:
            plt.plot(results_df.index, results_df['Close_Actual'], label='Actual Close', alpha=0.8)
            plt.plot(results_df.index, results_df['Close_Predicted'], label='Predicted Close', alpha=0.8)
        else:
            plt.plot(results_df.index, results_df['Actual'], label='Actual', alpha=0.8)
            plt.plot(results_df.index, results_df['Predicted'], label='Predicted', alpha=0.8)
        
        plt.title(f'{self.symbol} - Actual vs Predicted Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{save_path}/time_series_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Scatter plot
        plt.figure(figsize=(10, 8))
        if 'Close_Actual' in results_df.columns:
            actual = results_df['Close_Actual']
            predicted = results_df['Close_Predicted']
        else:
            actual = results_df['Actual']
            predicted = results_df['Predicted']
        
        plt.scatter(actual, predicted, alpha=0.6)
        
        # Perfect prediction line
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{self.symbol} - Predictions vs Actual')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}/scatter_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Error distribution
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        error_cols = results_df.filter(regex='Error$').columns
        if len(error_cols) > 0:
            for col in error_cols:
                plt.hist(results_df[col], bins=50, alpha=0.7, label=col)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        error_pct_cols = results_df.filter(regex='Error_Pct').columns
        if len(error_pct_cols) > 0:
            for col in error_pct_cols:
                plt.hist(results_df[col], bins=50, alpha=0.7, label=col)
        plt.xlabel('Prediction Error (%)')
        plt.ylabel('Frequency')
        plt.title('Error Percentage Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Feature importance (if available)
        if hasattr(self.model, 'get_feature_importance'):
            try:
                importance = self.model.get_feature_importance()
                if importance:
                    plt.figure(figsize=(12, 8))
                    top_features = dict(list(importance.items())[:20])
                    
                    features = list(top_features.keys())
                    importances = list(top_features.values())
                    
                    y_pos = np.arange(len(features))
                    plt.barh(y_pos, importances)
                    plt.yticks(y_pos, features)
                    plt.xlabel('Importance')
                    plt.title(f'{self.symbol} - Top 20 Feature Importance')
                    plt.gca().invert_yaxis()
                    plt.tight_layout()
                    plt.savefig(f'{save_path}/feature_importance.png', dpi=300, bbox_inches='tight')
                    plt.close()
            except Exception as e:
                logger.warning(f"Could not create feature importance plot: {str(e)}")
        
        logger.info(f"Evaluation plots saved to {save_path}")
    
    def generate_report(self, metrics: Dict[str, float], analysis: Dict[str, Any]) -> str:
        """Generate evaluation report.
        
        Args:
            metrics: Evaluation metrics
            analysis: Analysis results
            
        Returns:
            Report text
        """
        report = f"""
# Model Evaluation Report

## Model Information
- **Symbol**: {self.symbol}
- **Model Type**: {self.model_type}
- **Model Path**: {self.model_path}

## Performance Metrics
"""
        
        for metric, value in metrics.items():
            report += f"- **{metric}**: {value:.4f}\n"
        
        report += f"""

## Analysis Summary
- **Mean Absolute Error (%)**: {analysis['mean_absolute_error_pct']:.2f}%
- **Maximum Error (%)**: {analysis['max_error_pct']:.2f}%
- **Prediction Correlation**: {analysis['prediction_correlation']:.4f}

## Model Insights
"""
        
        # Add model-specific insights
        if hasattr(self.model, 'get_feature_importance'):
            try:
                importance = self.model.get_feature_importance()
                if importance:
                    top_features = list(importance.keys())[:5]
                    report += f"- **Top 5 Important Features**: {', '.join(top_features)}\n"
            except:
                pass
        
        if self.model_type == 'ensemble' and hasattr(self.model, 'get_model_weights'):
            try:
                weights = self.model.get_model_weights()
                report += f"- **Ensemble Weights**: {weights}\n"
            except:
                pass
        
        return report
    
    def run_evaluation(self, period: str = "6mo", save_plots: bool = True) -> Dict[str, Any]:
        """Run complete evaluation pipeline.
        
        Args:
            period: Period of data for evaluation
            save_plots: Whether to save evaluation plots
            
        Returns:
            Evaluation results
        """
        logger.info(f"Starting evaluation for {self.symbol}")
        
        # Load model
        self.load_model()
        
        # Prepare test data
        X_test, y_test, dates = self.prepare_test_data(period)
        
        # Evaluate performance
        metrics = self.evaluate_model_performance(X_test, y_test)
        
        # Analyze predictions
        analysis = self.analyze_predictions(X_test, y_test, dates)
        
        # Create plots
        if save_plots:
            self.create_evaluation_plots(analysis)
        
        # Generate report
        report = self.generate_report(metrics, analysis)
        
        # Save report
        with open('evaluation_report.md', 'w') as f:
            f.write(report)
        
        results = {
            'metrics': metrics,
            'analysis': analysis,
            'report': report,
            'model_info': self.model.get_model_info()
        }
        
        logger.info("Evaluation completed successfully")
        return results

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Evaluate trained OHLC prediction models")
    parser.add_argument("--model-path", required=True, help="Path to saved model")
    parser.add_argument("--symbol", required=True, help="Stock symbol")
    parser.add_argument("--period", default="6mo", help="Evaluation period")
    parser.add_argument("--no-plots", action="store_true", help="Don't save plots")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(args.model_path, args.symbol)
    
    # Run evaluation
    try:
        results = evaluator.run_evaluation(
            period=args.period,
            save_plots=not args.no_plots
        )
        
        print("\n" + "="*50)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Symbol: {args.symbol}")
        print(f"Model: {args.model_path}")
        print(f"Evaluation Period: {args.period}")
        
        print("\nPerformance Metrics:")
        for metric, value in results['metrics'].items():
            print(f"  {metric}: {value:.4f}")
        
        print(f"\nDetailed report saved to: evaluation_report.md")
        if not args.no_plots:
            print("Evaluation plots saved to: evaluation_plots/")
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
