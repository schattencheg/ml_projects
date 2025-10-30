"""
Complete ML Workflow Example with ReportManager

Demonstrates the full workflow:
1. Load data
2. Generate features
3. Create target
4. Create models
5. Train models
6. Test models
7. Generate comprehensive reports
8. Monitor health

Run from project root: python examples/example_complete_workflow.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from datetime import datetime

# Import classes
from src.ModelsManager import ModelsManager
from src.FeaturesGeneratorNew import FeaturesGenerator
from src.Trainer import Trainer
from src.Tester import Tester
from src.HealthManager import HealthManager
from src.ReportManager import ReportManager


def main():
    """Complete ML workflow with reporting."""
    
    print("="*70)
    print("COMPLETE ML WORKFLOW WITH REPORTING")
    print("="*70)
    
    # ==================== STEP 1: LOAD DATA ====================
    print("\n" + "="*70)
    print("STEP 1: LOAD DATA")
    print("="*70)
    
    df_train = pd.read_csv("data/hour/btc.csv")
    df_test = pd.read_csv("data/hour/btc_2025.csv")
    
    print(f"✓ Training data: {len(df_train):,} rows")
    print(f"✓ Test data: {len(df_test):,} rows")
    
    # ==================== STEP 2: GENERATE FEATURES ====================
    print("\n" + "="*70)
    print("STEP 2: GENERATE FEATURES")
    print("="*70)
    
    fg = FeaturesGenerator()
    
    print("\nGenerating classical features...")
    df_train_features = fg.generate_features(df_train, method='classical')
    df_test_features = fg.generate_features(df_test, method='classical')
    
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
    
    # Extract optimal thresholds
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
    
    # ==================== STEP 8: GENERATE REPORTS ====================
    print("\n" + "="*70)
    print("STEP 8: GENERATE COMPREHENSIVE REPORTS")
    print("="*70)
    
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
    print("\n" + "="*70)
    print("STEP 9: MONITOR MODEL HEALTH")
    print("="*70)
    
    health_manager = HealthManager(
        performance_threshold=0.05,
        time_threshold_days=30
    )
    
    # Set baseline for best model
    best_test_model = tester.get_best_model_name()
    best_model_metrics = test_results[best_test_model]['metrics']
    
    health_manager.set_baseline(
        model_name=best_test_model,
        metrics=best_model_metrics,
        timestamp=datetime.now()
    )
    
    print(f"\n✓ Health baseline set for {best_test_model}")
    
    # ==================== SUMMARY ====================
    print("\n" + "="*70)
    print("WORKFLOW COMPLETE - SUMMARY")
    print("="*70)
    print(f"\n📊 Training:")
    print(f"  • Best model: {best_model_name}")
    print(f"  • Models trained: {len(trained_models)}")
    print(f"  • Training time: {trainer.training_time:.2f}s")
    
    print(f"\n📈 Testing:")
    print(f"  • Best model: {best_test_model}")
    print(f"  • Best F1 Score: {best_model_metrics['f1']:.4f}")
    print(f"  • Best Accuracy: {best_model_metrics['accuracy']:.4f}")
    
    print(f"\n💾 Saved:")
    print(f"  • Models: {models_manager.models_dir}/")
    print(f"  • Reports: {report_manager.output_dir}/")
    
    print(f"\n📋 Reports Generated:")
    print(f"  • Training report (CSV + PNG)")
    print(f"  • Test report (CSV + PNG)")
    print(f"  • Comparison report (CSV + PNG)")
    
    print(f"\n🏥 Health Monitoring:")
    print(f"  • Baseline set for {best_test_model}")
    print(f"  • Performance threshold: 5%")
    print(f"  • Time threshold: 30 days")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Review reports in 'reports/' folder")
    print("2. Check visualizations (PNG files)")
    print("3. Monitor model health regularly")
    print("4. Retrain when health check recommends")
    print("5. Use best model for predictions/backtesting")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
