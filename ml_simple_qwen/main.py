"""
Main Execution Script

This script demonstrates the complete ML training pipeline:
1. Generates synthetic data
2. Sets up multiple ML models
3. Runs continuous training with periodic retraining
4. Collects and displays performance metrics
"""

import numpy as np
import pandas as pd
from src.data_generator import DataGenerator
from src.model_manager import ModelManager
from src.metrics_collector import MetricsCollector
from src.training_loop import ContinuousTrainingLoop

# Import sklearn-based models if available
try:
    from models.linear_model import SklearnLinearModel
    from models.random_forest_model import RandomForestModel
    SKLEARN_AVAILABLE = True
    print("Sklearn models available.")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Sklearn not available. Only basic models will be used.")


def main():
    """
    Main function to demonstrate the continuous ML training pipeline.
    """
    print("="*70)
    print("SIMPLE ML PROJECT - CONTINUOUS TRAINING DEMONSTRATION")
    print("="*70)
    
    # 1. Create required components
    print("\n1. Initializing components...")
    model_manager = ModelManager()
    metrics_collector = MetricsCollector()
    data_generator = DataGenerator()
    
    # 2. Register models
    print("\n2. Registering models...")
    from src.model_manager import LinearModel, DummyModel
    model_manager.register_model('basic_linear', LinearModel)
    model_manager.register_model('dummy', DummyModel)
    
    # Register sklearn models if available
    if SKLEARN_AVAILABLE:
        model_manager.register_model('sklearn_linear', SklearnLinearModel)
        model_manager.register_model('random_forest', RandomForestModel)
    
    # Create instances of all registered models
    model_manager.create_model('basic_linear')
    model_manager.create_model('dummy')
    
    if SKLEARN_AVAILABLE:
        model_manager.create_model('sklearn_linear')
        model_manager.create_model('random_forest', n_estimators=20, max_depth=5)
    
    print(f"Created models: {model_manager.get_trained_models()}")
    
    # 3. Generate synthetic data
    print("\n3. Generating synthetic data...")
    # Using a mix of linear and non-linear patterns to challenge different models
    data = data_generator.generate_linear_data(n_samples=500, n_features=4, noise_level=0.2)
    print(f"Generated data shape: {data.shape}")
    print(f"Features: {list(data.columns[:-1])}")
    print(f"Target: {data.columns[-1]}")
    
    # 4. Set up training parameters
    print("\n4. Setting up training parameters...")
    initial_train_size = 100    # Use first 100 samples for initial training
    retrain_interval = 50       # Retrain every 50 new samples
    batch_size = 10             # Process 10 samples at a time
    
    print(f"  Initial training samples: {initial_train_size}")
    print(f"  Retrain every: {retrain_interval} new samples")
    print(f"  Batch size: {batch_size}")
    
    # 5. Run continuous training
    print(f"\n5. Starting continuous training with {len(model_manager.get_trained_models())} models...")
    training_loop = ContinuousTrainingLoop(model_manager, metrics_collector)
    
    feature_cols = [col for col in data.columns if col != 'target']
    training_loop.run_continuous_training(
        data=data,
        initial_train_size=initial_train_size,
        retrain_interval=retrain_interval,
        batch_size=batch_size,
        feature_cols=feature_cols,
        target_col='target'
    )
    
    # 6. Display results
    print("\n6. Training completed. Displaying results...")
    metrics_collector.print_metrics_summary()
    
    # Show detailed metrics for analysis
    all_metrics = metrics_collector.get_all_metrics()
    print(f"\nTotal metrics records collected: {len(all_metrics)}")
    
    # Show metrics evolution over time for each model
    print("\nMetrics evolution over time:")
    for model_name in all_metrics['model_name'].unique():
        model_metrics = all_metrics[all_metrics['model_name'] == model_name].copy()
        print(f"\n{model_name}:")
        print(f"  Final MSE: {model_metrics['mse'].iloc[-1]:.4f}")
        print(f"  Final MAE: {model_metrics['mae'].iloc[-1]:.4f}")
        print(f"  Final R2:  {model_metrics['r2'].iloc[-1]:.4f}")
        print(f"  Average MSE: {model_metrics['mse'].mean():.4f}")
        print(f"  Average MAE: {model_metrics['mae'].mean():.4f}")
        print(f"  Average R2:  {model_metrics['r2'].mean():.4f}")
    
    # 7. Export results
    print("\n7. Exporting results...")
    metrics_collector.save_metrics('outputs/metrics/model_metrics.csv')
    print("Metrics saved to 'outputs/metrics/model_metrics.csv'")
    
    # Optional: Create a simple visualization if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        # Plot model performance over time
        plt.figure(figsize=(15, 10))
        
        # Plot MSE over time for each model
        plt.subplot(2, 2, 1)
        for model_name in all_metrics['model_name'].unique():
            model_data = all_metrics[all_metrics['model_name'] == model_name]
            plt.plot(model_data['step'], model_data['mse'], label=model_name, marker='o', markersize=4)
        plt.title('MSE Over Time')
        plt.xlabel('Step')
        plt.ylabel('MSE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot MAE over time for each model
        plt.subplot(2, 2, 2)
        for model_name in all_metrics['model_name'].unique():
            model_data = all_metrics[all_metrics['model_name'] == model_name]
            plt.plot(model_data['step'], model_data['mae'], label=model_name, marker='s', markersize=4)
        plt.title('MAE Over Time')
        plt.xlabel('Step')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot R2 over time for each model
        plt.subplot(2, 2, 3)
        for model_name in all_metrics['model_name'].unique():
            model_data = all_metrics[all_metrics['model_name'] == model_name]
            plt.plot(model_data['step'], model_data['r2'], label=model_name, marker='^', markersize=4)
        plt.title('R2 Over Time')
        plt.xlabel('Step')
        plt.ylabel('R2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Bar plot of average metrics by model
        plt.subplot(2, 2, 4)
        summary = all_metrics.groupby('model_name')[['mse', 'mae', 'r2']].mean()
        summary.plot(kind='bar', ax=plt.gca())
        plt.title('Average Metrics by Model')
        plt.xlabel('Model')
        plt.ylabel('Metric Value')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('outputs/visualizations/model_performance.png', dpi=300, bbox_inches='tight')
        print("Performance chart saved to 'outputs/visualizations/model_performance.png'")
        
    except ImportError:
        print("Matplotlib not available, skipping visualization.")
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)


def demo_different_scenarios():
    """
    Demonstrate the system with different data patterns and training scenarios.
    """
    print("\n" + "="*50)
    print("DEMONSTRATING DIFFERENT SCENARIOS")
    print("="*50)
    
    # Scenario 1: Non-linear data with polynomial patterns
    print("\nSCENARIO 1: Non-linear data patterns")
    model_manager = ModelManager()
    metrics_collector = MetricsCollector()
    data_generator = DataGenerator()
    
    # Register models
    from src.model_manager import LinearModel, DummyModel
    model_manager.register_model('linear', LinearModel)
    if SKLEARN_AVAILABLE:
        model_manager.register_model('rf', RandomForestModel)
    
    model_manager.create_model('linear')
    if SKLEARN_AVAILABLE:
        model_manager.create_model('rf', n_estimators=30, max_depth=4)
    
    # Generate non-linear data
    data = data_generator.generate_nonlinear_data(n_samples=300, n_features=3, noise_level=0.1)
    
    # Run training with smaller intervals to see adaptation
    training_loop = ContinuousTrainingLoop(model_manager, metrics_collector)
    feature_cols = [col for col in data.columns if col != 'target']
    
    training_loop.run_continuous_training(
        data=data,
        initial_train_size=80,
        retrain_interval=25,
        batch_size=8,
        feature_cols=feature_cols,
        target_col='target'
    )
    
    print("\nNon-linear scenario results:")
    metrics_collector.print_metrics_summary()
    

if __name__ == "__main__":
    main()
    
    # Run additional scenarios if desired (only in interactive mode)
    try:
        run_scenarios = input("\nRun additional scenarios? (y/n): ")
        if run_scenarios.lower() == 'y':
            demo_different_scenarios()
    except EOFError:
        print("\nNon-interactive mode: skipping additional scenarios")
        print("To run scenarios, execute: python main.py and respond 'y' when prompted")