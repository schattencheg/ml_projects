"""
Main Training and Testing Script - Refactored

Uses the new class-based architecture:
- ModelsManager: Model lifecycle
- FeaturesGenerator: Feature generation
- Trainer: Training logic
- Tester: Testing logic
- ReportManager: Report generation
- HealthManager: Health monitoring
"""

import os
import pandas as pd
from datetime import datetime

# Import new classes
from src.ModelsManager import ModelsManager
from src.FeaturesGenerator import FeaturesGenerator
from src.Trainer import Trainer
from src.Tester import Tester
from src.ReportManager import ReportManager
from src.HealthManager import HealthManager


# Data paths
PATH_TRAIN = "data/hour/btc.csv"
PATH_TEST = "data/hour/btc_2025.csv"
PATH_MODELS = "models"
os.environ["PATH_TRAIN"] = PATH_TRAIN
os.environ["PATH_TEST"] = PATH_TEST
os.environ["PATH_MODELS"] = PATH_MODELS


def main_train():
    """Main execution function."""
    
    print("="*80)
    print("ML PREDICTION SYSTEM - TRAINING & TESTING")
    print("="*80)
    
    # ==================== STEP 1: LOAD DATA ====================
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    
    print(f"Loading training data from: {PATH_TRAIN}")
    df_train = pd.read_csv(PATH_TRAIN)
    print(f"✓ Training data loaded: {len(df_train):,} rows")
    
    print(f"Loading test data from: {PATH_TEST}")
    df_test = pd.read_csv(PATH_TEST)
    print(f"✓ Test data loaded: {len(df_test):,} rows")
    
    # ==================== STEP 2: GENERATE FEATURES ====================
    print("\n" + "="*80)
    print("STEP 2: GENERATING FEATURES")
    print("="*80)
    
    fg = FeaturesGenerator()
    
    print("Generating classical features (SMA, RSI, Bollinger, Stochastic)...")
    df_train_features = fg.generate_features(df_train, method='classical')
    df_test_features = fg.generate_features(df_test, method='classical')
    
    print(f"✓ Features generated: {len(df_train_features.columns)} columns")
    
    # ==================== STEP 3: CREATE TARGET ====================
    print("\n" + "="*80)
    print("STEP 3: CREATING TARGET")
    print("="*80)
    
    # Parameters:
    # - target_bars=15: Look ahead 15 bars (15 hours for hourly data)
    # - target_pct=3.0: Predict if price increases by 3% or more
    # - method='binary': Two classes (0=No Rise, 1=Rise)
    
    print("Creating target variable...")
    print(f"  Look-ahead period: 15 bars")
    print(f"  Price change threshold: 3.0%")
    print(f"  Method: binary classification")
    
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
    
    print(f"✓ Features prepared: {len(feature_cols)} features")
    
    # ==================== STEP 4: CREATE MODELS ====================
    print("\n" + "="*80)
    print("STEP 4: CREATING MODELS")
    print("="*80)
    
    models_manager = ModelsManager(models_dir='models')
    
    # Show configuration
    print("\nEnabled models:")
    enabled = models_manager.get_enabled_models()
    for model_name in enabled:
        print(f"  ✓ {model_name}")
    
    # Create models
    models = models_manager.create_models(enabled_only=True)
    
    # ==================== STEP 5: TRAIN MODELS ====================
    print("\n" + "="*80)
    print("STEP 5: TRAINING MODELS")
    print("="*80)
    
    trainer = Trainer(
        use_smote=True,           # Apply SMOTE for imbalanced data
        optimize_threshold=True,  # Optimize probability threshold
        use_scaler=True          # Scale features
    )
    
    trained_models, scaler, train_results, best_model_name = trainer.train(
        models=models,
        X_train=X_train,
        y_train=y_train
    )
    
    # Print training results
    trainer.print_results()
    
    # ==================== STEP 6: SAVE MODELS ====================
    print("\n" + "="*80)
    print("STEP 6: SAVING MODELS")
    print("="*80)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    saved_paths = models_manager.save_models(
        models=trained_models,
        scaler=scaler,
        suffix=timestamp
    )
    
    print(f"\n✓ Models saved with timestamp: {timestamp}")
    
    # ==================== STEP 7: TEST MODELS ====================
    print("\n" + "="*80)
    print("STEP 7: TESTING MODELS")
    print("="*80)
    
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
    
    # Print test results
    tester.print_results()
    
    # Print detailed report for best model
    best_test_model = tester.get_best_model_name()
    print(f"\nDetailed report for best model: {best_test_model}")
    tester.print_detailed_report(best_test_model, y_test, target_names=['No Rise', 'Rise'])
    
    # ==================== STEP 8: GENERATE REPORTS ====================
    print("\n" + "="*80)
    print("STEP 8: GENERATING REPORTS")
    print("="*80)
    
    report_manager = ReportManager(output_dir='reports')
    
    # Create full report with all visualizations
    full_report = report_manager.export_full_report(
        train_results=train_results,
        test_results=test_results,
        y_test=y_test,
        filename=f"ml_report_{timestamp}",
        target_names=['No Rise', 'Rise']
    )
    
    # ==================== STEP 9: MONITOR HEALTH ====================
    print("\n" + "="*80)
    print("STEP 9: SETTING UP HEALTH MONITORING")
    print("="*80)
    
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
    
    print(f"\n✓ Health baseline set for {best_test_model}")
    print(f"  Performance threshold: 5%")
    print(f"  Time threshold: 30 days")
    
    # ==================== SUMMARY ====================
    print("\n" + "="*80)
    print("TRAINING AND TESTING COMPLETE!")
    print("="*80)
    
    print(f"\n📊 Training Summary:")
    print(f"  • Best training model: {best_model_name}")
    print(f"  • Models trained: {len(trained_models)}")
    print(f"  • Total training time: {trainer.training_time:.2f}s ({trainer.training_time/60:.2f} min)")
    print(f"  • Average time per model: {trainer.training_time/len(trained_models):.2f}s")
    
    print(f"\n📈 Testing Summary:")
    print(f"  • Best test model: {best_test_model}")
    print(f"  • Test Accuracy: {best_model_metrics['accuracy']:.4f}")
    print(f"  • Test F1 Score: {best_model_metrics['f1']:.4f}")
    print(f"  • Test Precision: {best_model_metrics['precision']:.4f}")
    print(f"  • Test Recall: {best_model_metrics['recall']:.4f}")
    
    print(f"\n💾 Saved Files:")
    print(f"  • Models: models/ (timestamp: {timestamp})")
    print(f"  • Reports: reports/")
    print(f"    - Training report (CSV + PNG)")
    print(f"    - Test report (CSV + PNG)")
    print(f"    - Comparison report (CSV + PNG)")
    
    print(f"\n🏥 Health Monitoring:")
    print(f"  • Baseline set for {best_test_model}")
    print(f"  • Monitor regularly for performance degradation")
    print(f"  • Retrain when health check recommends")
    
    print(f"\n🚀 Next Steps:")
    print(f"  1. Review reports in 'reports/' folder")
    print(f"  2. Check visualizations (PNG files)")
    print(f"  3. Run backtesting:")
    print(f"     python examples/backtest_quick_start.py")
    print(f"  4. Monitor model health regularly")
    print(f"  5. Use best model for predictions")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
