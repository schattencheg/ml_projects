"""
Example: Refactored ML Workflow

Demonstrates the use of the new class-based architecture:
- ModelsManager: Create and load models
- FeaturesGenerator: Generate features and targets
- Trainer: Train models
- Tester: Test models
- HealthManager: Monitor model health
"""

import pandas as pd
from datetime import datetime

# Import new classes
from src.ModelsManager import ModelsManager
from src.FeaturesGeneratorNew import FeaturesGenerator
from src.Trainer import Trainer
from src.Tester import Tester
from src.HealthManager import HealthManager


def main():
    """Main workflow demonstration."""
    
    print("="*70)
    print("REFACTORED ML WORKFLOW EXAMPLE")
    print("="*70)
    
    # ==================== STEP 1: LOAD DATA ====================
    print("\n" + "="*70)
    print("STEP 1: LOAD DATA")
    print("="*70)
    
    # Load training and test data
    df_train = pd.read_csv("data/hour/btc.csv")
    df_test = pd.read_csv("data/hour/btc_2025.csv")
    
    print(f"✓ Training data: {len(df_train):,} rows")
    print(f"✓ Test data: {len(df_test):,} rows")
    
    # ==================== STEP 2: GENERATE FEATURES ====================
    print("\n" + "="*70)
    print("STEP 2: GENERATE FEATURES")
    print("="*70)
    
    fg = FeaturesGenerator()
    
    # Option 1: Classical features
    print("\nGenerating classical features...")
    df_train_features = fg.generate_features(df_train, method='classical')
    df_test_features = fg.generate_features(df_test, method='classical')
    
    # Option 2: Crypto features (uncomment to use)
    # print("\nGenerating crypto features...")
    # df_train_features = fg.generate_features(df_train, method='crypto', price_change_threshold=0.02)
    # df_test_features = fg.generate_features(df_test, method='crypto', price_change_threshold=0.02)
    
    print(f"✓ Features generated: {len(df_train_features.columns)} columns")
    
    # ==================== STEP 3: CREATE TARGET ====================
    print("\n" + "="*70)
    print("STEP 3: CREATE TARGET")
    print("="*70)
    
    df_train_with_target = fg.create_target(
        df_train_features, 
        target_bars=15, 
        target_pct=3.0, 
        method='binary'
    )
    
    df_test_with_target = fg.create_target(
        df_test_features, 
        target_bars=15, 
        target_pct=3.0, 
        method='binary'
    )
    
    # Drop NaN rows
    df_train_with_target = df_train_with_target.dropna()
    df_test_with_target = df_test_with_target.dropna()
    
    print(f"✓ Target created")
    print(f"  Training samples: {len(df_train_with_target):,}")
    print(f"  Test samples: {len(df_test_with_target):,}")
    
    # Separate features and target
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target', 'pct_change_15']
    feature_cols = [col for col in df_train_with_target.columns if col not in exclude_cols]
    
    X_train = df_train_with_target[feature_cols]
    y_train = df_train_with_target['target']
    X_test = df_test_with_target[feature_cols]
    y_test = df_test_with_target['target']
    
    print(f"✓ Features: {len(feature_cols)}")
    
    # ==================== STEP 4: CREATE MODELS ====================
    print("\n" + "="*70)
    print("STEP 4: CREATE MODELS")
    print("="*70)
    
    models_manager = ModelsManager(models_dir='models')
    models_manager.print_config()
    
    # Create fresh models
    models = models_manager.create_models(enabled_only=True)
    
    # ==================== STEP 5: TRAIN MODELS ====================
    print("\n" + "="*70)
    print("STEP 5: TRAIN MODELS")
    print("="*70)
    
    trainer = Trainer(use_smote=True, optimize_threshold=True, use_scaler=True)
    
    trained_models, scaler, train_results, best_model_name = trainer.train(
        models=models,
        X_train=X_train,
        y_train=y_train
    )
    
    trainer.print_results()
    
    # ==================== STEP 6: SAVE MODELS ====================
    print("\n" + "="*70)
    print("STEP 6: SAVE MODELS")
    print("="*70)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    saved_paths = models_manager.save_models(
        models=trained_models,
        scaler=scaler,
        suffix=timestamp
    )
    
    # ==================== STEP 7: TEST MODELS ====================
    print("\n" + "="*70)
    print("STEP 7: TEST MODELS")
    print("="*70)
    
    tester = Tester(scaler=scaler)
    
    # Extract optimal thresholds from training
    optimal_thresholds = {
        name: results['optimal_threshold'] 
        for name, results in train_results.items()
    }
    
    test_results = tester.test(
        models=trained_models,
        X_test=X_test,
        y_test=y_test,
        optimal_thresholds=optimal_thresholds
    )
    
    tester.print_results()
    
    # Print detailed report for best model
    best_test_model = tester.get_best_model_name()
    tester.print_detailed_report(best_test_model, y_test, target_names=['No Rise', 'Rise'])
    
    # ==================== STEP 8: MONITOR HEALTH ====================
    print("\n" + "="*70)
    print("STEP 8: MONITOR MODEL HEALTH")
    print("="*70)
    
    health_manager = HealthManager(
        performance_threshold=0.05,  # 5% degradation threshold
        time_threshold_days=30       # Retrain after 30 days
    )
    
    # Set baseline for best model
    best_model_metrics = test_results[best_test_model]['metrics']
    health_manager.set_baseline(
        model_name=best_test_model,
        metrics=best_model_metrics,
        timestamp=datetime.now()
    )
    
    # Simulate checking health later (for demonstration)
    # In production, you would check this periodically with new data
    simulated_current_metrics = {
        'accuracy': best_model_metrics['accuracy'] - 0.03,  # Simulate 3% drop
        'f1': best_model_metrics['f1'] - 0.02
    }
    
    health_report = health_manager.check_health(
        model_name=best_test_model,
        current_metrics=simulated_current_metrics
    )
    
    health_manager.print_health_report(health_report)
    
    # ==================== STEP 9: LOAD SAVED MODELS (OPTIONAL) ====================
    print("\n" + "="*70)
    print("STEP 9: LOAD SAVED MODELS (DEMONSTRATION)")
    print("="*70)
    
    # List all saved model versions
    print("\nAvailable model versions:")
    versions = models_manager.list_saved_models()
    for suffix, metadata in versions[:5]:  # Show first 5
        print(f"  • {suffix}: {len(metadata['models'])} models")
    
    # Load the latest models
    print("\nLoading latest models...")
    loaded_models, loaded_scaler, loaded_metadata = models_manager.load_models(suffix='latest')
    
    if loaded_models:
        print(f"✓ Loaded {len(loaded_models)} models")
        print(f"  Models: {', '.join(loaded_models.keys())}")
    
    # ==================== SUMMARY ====================
    print("\n" + "="*70)
    print("WORKFLOW COMPLETE")
    print("="*70)
    print(f"\n✓ Best training model: {best_model_name}")
    print(f"✓ Best test model: {best_test_model}")
    print(f"✓ Models saved to: {models_manager.models_dir}")
    print(f"✓ Health monitoring active")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Use trained models for predictions")
    print("2. Monitor model health regularly")
    print("3. Retrain when health check recommends")
    print("4. Backtest with MLBacktester module")
    print("5. Deploy best model to production")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
