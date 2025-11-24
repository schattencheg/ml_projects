"""
BTCUSDT Framework Example - Using Enhanced Framework Components

This example demonstrates:
1. Smart data loading with caching (loads existing data, downloads only missing parts)
2. Using LogisticRegressionModel and RandomForestModel classes
3. Feature generation
4. Model training and testing
5. Model persistence

Author: ML Framework
Date: 2024-11-24
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_provider import DataProvider
from src.features_generator import FeaturesGenerator
from src.models_lib.linear_model import LogisticRegressionModel, RandomForestModel


def main():
    """Run complete ML workflow for BTCUSDT using framework components."""
    
    print("\n" + "="*80)
    print("BTCUSDT ML FRAMEWORK - ENHANCED EXAMPLE")
    print("Using LogisticRegressionModel & RandomForestModel with Smart Caching")
    print("="*80)
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    TICKER = 'BTC-USD'
    START_DATE = '2020-01-01'
    END_DATE = '2024-11-24'
    INTERVAL = '1d'
    
    FUTURE_BARS = 5
    THRESHOLD = 0.02
    
    print(f"\nConfiguration:")
    print(f"  Ticker: {TICKER}")
    print(f"  Period: {START_DATE} to {END_DATE}")
    print(f"  Interval: {INTERVAL}")
    print(f"  Target: Predict {THRESHOLD*100}% change in {FUTURE_BARS} bars")
    
    # ========================================================================
    # STEP 1: Smart Data Loading with Caching
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 1] SMART DATA LOADING (with caching)")
    print("="*80)
    
    data_provider = DataProvider(data_dir='data')
    
    # First call: Downloads data and caches it
    # Subsequent calls: Loads from cache, downloads only missing data
    df = data_provider.load_yahoo(
        ticker=TICKER,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL,
        use_cache=True  # Enable smart caching
    )
    
    print(f"\n✓ Data ready: {len(df)} rows")
    print(f"✓ Date range: {df.index[0].date()} to {df.index[-1].date()}")
    
    # Validate and clean
    data_provider.validate_data(df)
    df = data_provider.clean_data(df, method='drop')
    
    # ========================================================================
    # STEP 2: Generate Features
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 2] GENERATING FEATURES")
    print("="*80)
    
    features_gen = FeaturesGenerator()
    
    # Generate technical indicators
    df_features = features_gen.generate_features(df, feature_set='advanced')
    
    # Create target variable
    df_features = features_gen.create_target(
        df_features,
        target_type='classification',
        future_bars=FUTURE_BARS,
        threshold=THRESHOLD
    )
    
    # Remove NaN
    df_features = df_features.dropna()
    
    print(f"\n✓ Features generated: {len(df_features.columns)} columns")
    print(f"✓ Dataset ready: {len(df_features)} rows")
    
    # Check class distribution
    target_counts = df_features['target'].value_counts()
    print(f"\nClass Distribution:")
    for label, count in target_counts.items():
        percentage = (count / len(df_features)) * 100
        print(f"  Class {label}: {count} samples ({percentage:.1f}%)")
    
    # ========================================================================
    # STEP 3: Prepare Data
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 3] PREPARING DATA")
    print("="*80)
    
    # Split data
    train_df, val_df, test_df = data_provider.split_data(
        df_features,
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    # Get feature columns
    feature_cols = features_gen.get_feature_names()
    
    # Prepare X and y
    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['target'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values
    
    print(f"\n✓ Feature columns: {len(feature_cols)}")
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Validation samples: {len(X_val)}")
    print(f"✓ Test samples: {len(X_test)}")
    
    # ========================================================================
    # STEP 4: Create and Train Models using Framework Classes
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 4] TRAINING MODELS (Framework Classes)")
    print("="*80)
    
    # Create model instances
    models = {
        'LogisticRegression': LogisticRegressionModel(
            name='LogisticRegression',
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        ),
        'RandomForest': RandomForestModel(
            name='RandomForest',
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    }
    
    print(f"\nTraining {len(models)} models...\n")
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Predict on validation set
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print(f"  ✓ Accuracy: {accuracy:.4f} | F1: {f1:.4f} | "
              f"Precision: {precision:.4f} | Recall: {recall:.4f}")
        
        # Show model-specific info
        if isinstance(model, LogisticRegressionModel):
            coef = model.get_coefficients()
            if coef is not None:
                print(f"  ℹ Coefficients shape: {coef.shape}")
        elif isinstance(model, RandomForestModel):
            importance = model.get_feature_importance()
            if importance is not None:
                print(f"  ℹ Feature importance shape: {importance.shape}")
                # Show top 5 features
                top_indices = np.argsort(importance)[-5:][::-1]
                print(f"  ℹ Top 5 features:")
                for idx in top_indices:
                    print(f"      {feature_cols[idx]}: {importance[idx]:.4f}")
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['f1_score'])
    best_model = results[best_model_name]['model']
    
    print(f"\n✓ Best model: {best_model_name} (F1: {results[best_model_name]['f1_score']:.4f})")
    
    # ========================================================================
    # STEP 5: Test on Test Set
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 5] TESTING ON TEST SET")
    print("="*80)
    
    print(f"\nTesting best model ({best_model_name})...\n")
    
    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)
    
    from sklearn.metrics import classification_report
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    
    print(f"Test Set Results:")
    print(f"  Accuracy:  {test_accuracy:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  F1 Score:  {test_f1:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_test_pred, 
                                target_names=['No Increase', 'Increase']))
    
    # ========================================================================
    # STEP 6: Save Models
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 6] SAVING MODELS")
    print("="*80)
    
    import joblib
    from datetime import datetime
    
    # Create models directory
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = models_dir / timestamp
    save_dir.mkdir(exist_ok=True)
    
    # Save all models
    for name, result in results.items():
        model_path = save_dir / f"{name.lower()}.joblib"
        joblib.dump(result['model'], model_path)
        print(f"✓ Saved {name} to {model_path}")
    
    # Save best model separately
    best_model_path = save_dir / f"{best_model_name.lower()}_best.joblib"
    joblib.dump(best_model, best_model_path)
    print(f"✓ Saved best model to {best_model_path}")
    
    # Save metadata
    metadata = {
        'ticker': TICKER,
        'start_date': START_DATE,
        'end_date': END_DATE,
        'interval': INTERVAL,
        'best_model': best_model_name,
        'best_f1_score': results[best_model_name]['f1_score'],
        'feature_cols': feature_cols,
        'future_bars': FUTURE_BARS,
        'threshold': THRESHOLD,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'test_accuracy': test_accuracy,
        'test_f1': test_f1
    }
    
    metadata_path = save_dir / 'metadata.joblib'
    joblib.dump(metadata, metadata_path)
    print(f"✓ Saved metadata to {metadata_path}")
    
    # ========================================================================
    # STEP 7: Demonstrate Loading Saved Models
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 7] LOADING SAVED MODELS (Demonstration)")
    print("="*80)
    
    # Load best model
    loaded_model = joblib.load(best_model_path)
    print(f"\n✓ Loaded model from {best_model_path}")
    print(f"  Model type: {type(loaded_model).__name__}")
    print(f"  Model name: {loaded_model.name}")
    
    # Verify it works
    test_predictions = loaded_model.predict(X_test[:5])
    print(f"\n✓ Test predictions (first 5): {test_predictions}")
    
    # Load metadata
    loaded_metadata = joblib.load(metadata_path)
    print(f"\n✓ Loaded metadata:")
    print(f"  Ticker: {loaded_metadata['ticker']}")
    print(f"  Best model: {loaded_metadata['best_model']}")
    print(f"  Test F1: {loaded_metadata['test_f1']:.4f}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("WORKFLOW COMPLETE - SUMMARY")
    print("="*80)
    
    print(f"\n📊 Data:")
    print(f"  ✓ Ticker: {TICKER}")
    print(f"  ✓ Total samples: {len(df_features)}")
    print(f"  ✓ Features: {len(feature_cols)}")
    print(f"  ✓ Smart caching: Enabled (data cached to data/ folder)")
    
    print(f"\n🤖 Models:")
    print(f"  ✓ Models trained: {len(models)}")
    print(f"  ✓ Model classes used:")
    for name in models.keys():
        print(f"      - {name}")
    print(f"  ✓ Best model: {best_model_name}")
    print(f"  ✓ Validation F1: {results[best_model_name]['f1_score']:.4f}")
    print(f"  ✓ Test F1: {test_f1:.4f}")
    
    print(f"\n💾 Saved:")
    print(f"  ✓ Models directory: {save_dir}")
    print(f"  ✓ Cached data: data/{TICKER.replace('-', '_')}_{INTERVAL}.csv")
    
    print(f"\n🎯 Key Features Demonstrated:")
    print(f"  ✓ Smart data caching (loads existing, downloads only missing)")
    print(f"  ✓ LogisticRegressionModel class with coefficients")
    print(f"  ✓ RandomForestModel class with feature importance")
    print(f"  ✓ Model persistence with joblib")
    print(f"  ✓ Timestamped model saves")
    
    print("\n" + "="*80)
    print("🎉 Enhanced framework workflow completed successfully!")
    print("="*80 + "\n")
    
    print("💡 Try running this script again - it will load data from cache!")
    print("💡 Try extending the date range - it will download only new data!")
    print("\n")
    
    return {
        'models': results,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'save_dir': save_dir,
        'metadata': metadata
    }


if __name__ == '__main__':
    results = main()
