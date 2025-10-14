import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime
from .models.base_model import BaseModel
from .data_loader import DataLoader
from .evaluation import evaluate_model, calculate_metrics
import os


class YearlyModelTrainer:
    """
    Class to handle yearly retraining and evaluation of models
    """
    
    def __init__(self, data_path: str, models: List[BaseModel]):
        """
        Initialize the trainer
        
        Args:
            data_path: Path to the OHLC data
            models: List of model instances to train and evaluate
        """
        self.data_loader = DataLoader(data_path)
        self.models = models
        self.results = []
        self.yearly_results = {}
    
    def train_and_evaluate_yearly(self, lookback_window: int = 10) -> Dict[str, Dict[int, Dict[str, float]]]:
        """
        Train and evaluate models yearly
        
        Args:
            lookback_window: Number of previous days to use for features
            
        Returns:
            Dictionary with results by year and model
        """
        # Load and split data by year
        all_data = self.data_loader.load_data()
        yearly_data = self.data_loader.split_by_year()
        
        # Sort years to process chronologically
        years = sorted(yearly_data.keys())
        
        print(f"Processing {len(years)} years: {years}")
        
        for i, year in enumerate(years):
            print(f"\nProcessing year {year} (year {i+1}/{len(years)})...")
            
            # Get the current year's data
            current_year_data = yearly_data[year]
            
            # Create features for the current year
            current_year_features = self.data_loader.create_features(current_year_data, lookback_window)
            
            # Prepare features and targets for the current year
            X, y = self.data_loader.prepare_data_for_training(current_year_features, ['Next_Close'])
            
            if len(X) == 0 or len(y) == 0:
                print(f"  Skipping {year} - insufficient data for training")
                continue
            
            # Store results for this year
            self.yearly_results[year] = {}
            
            # Train each model on the current year's data
            for model in self.models:
                print(f"  Training {model.name}...")
                
                # Train the model
                trained_model = model.train(X, y)
                
                # Evaluate the model
                # For models that reshape data (like LSTM), we need special handling
                if hasattr(model, 'sequence_length') and model.name == "LSTM":
                    # For LSTM, predictions will be shorter due to sequence reshaping
                    # So we need to adjust y for evaluation
                    y_eval = y[model.sequence_length-1:]
                    y_pred = trained_model.predict(X)
                    metrics = calculate_metrics(y_eval, y_pred)
                    # Add model name to metrics
                    metrics['Model'] = model.name
                else:
                    # For other models, use standard evaluation
                    metrics = evaluate_model(trained_model, X, y, model.name)
                
                metrics['Year'] = year
                metrics['Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Store the results
                self.results.append(metrics)
                self.yearly_results[year][model.name] = metrics
                
                print(f"    {model.name} - RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}")
        
        return self.yearly_results
    
    def get_overall_results(self) -> pd.DataFrame:
        """
        Get overall results as a DataFrame
        
        Returns:
            DataFrame with all results
        """
        if not self.results:
            return pd.DataFrame()
        
        return pd.DataFrame(self.results)
    
    def get_yearly_summary(self) -> pd.DataFrame:
        """
        Get a summary of results by year
        
        Returns:
            DataFrame with yearly summary
        """
        summary_data = []
        
        for year, year_results in self.yearly_results.items():
            for model_name, metrics in year_results.items():
                summary_data.append({
                    'Year': year,
                    'Model': model_name,
                    'RMSE': metrics['RMSE'],
                    'MAE': metrics['MAE'],
                    'R2': metrics['R2'],
                    'MAPE': metrics['MAPE']
                })
        
        return pd.DataFrame(summary_data)
    
    def save_results(self, results_dir: str = "results/metrics"):
        """
        Save results to CSV files
        
        Args:
            results_dir: Directory to save results
        """
        os.makedirs(results_dir, exist_ok=True)
        
        # Save overall results
        overall_df = self.get_overall_results()
        overall_df.to_csv(os.path.join(results_dir, "overall_results.csv"), index=False)
        
        # Save yearly summary
        yearly_df = self.get_yearly_summary()
        yearly_df.to_csv(os.path.join(results_dir, "yearly_summary.csv"), index=False)
        
        print(f"Results saved to {results_dir}/")