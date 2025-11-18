"""
Example: Using the new Pipeline Manager for complete ML workflow.

This example demonstrates the new architecture with:
- PipelineManager for orchestrating the complete workflow
- Automatic timestamped results structure
- Multiple model types (XGBoost, Random Forest, Logistic Regression)
- Comprehensive HTML reports
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src import PipelineManager

def main():
    """Run complete ML pipeline."""
    
    # Initialize pipeline
    pipeline = PipelineManager(
        results_dir='results',
        use_mlflow=False  # Set to True if MLFlow server is running
    )
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(
        ticker='BTC-USD',
        start_date='2020-01-01',
        end_date='2023-12-31',
        model_names=['logistic_regression', 'random_forest', 'xgboost'],
        feature_set='basic',
        backtest=True,
        backtest_backend='nolib'
    )
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {pipeline.get_run_directory()}")
    print("\nCheck the following directories:")
    print(f"  - models/          : Trained models")
    print(f"  - reports/train/   : Training visualizations")
    print(f"  - reports/test/    : Test visualizations")
    print(f"  - reports/backtest/: Backtest visualizations")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
