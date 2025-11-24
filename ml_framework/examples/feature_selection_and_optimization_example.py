"""
Example: Feature Selection and Hyperparameter Optimization with GPU Support

This example demonstrates:
- Feature importance tracking and selection BEFORE training
- Hyperparameter optimization for ML models
- GPU acceleration for XGBoost, CatBoost, and CNN models
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src import (
    DataProvider,
    FeaturesGenerator,
    FeatureSelector,
    ModelManager,
    TrainManager,
    HyperparameterOptimizer
)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


def main():
    """Run complete pipeline with feature selection and optimization."""
    
    print("\n" + "="*70)
    print("FEATURE SELECTION & HYPERPARAMETER OPTIMIZATION EXAMPLE")
    print("="*70)
    
    # ========================================================================
    # 1. Load and prepare data
    # ========================================================================
    print("\n[1/6] Loading data...")
    data_provider = DataProvider()
    df = data_provider.load_yahoo('BTC-USD', '2020-01-01', '2023-12-31')
    df = data_provider.clean_data(df)
    
    # Generate features
    print("\n[2/6] Generating features...")
    features_gen = FeaturesGenerator()
    df = features_gen.generate_features(df, feature_set='basic')
    df = features_gen.create_target(df, future_bars=5, threshold=0.02)
    
    # Split data
    train_df, val_df, test_df = data_provider.split_data(df, train_size=0.7, val_size=0.15)
    
    # Get initial features
    initial_features = features_gen.get_feature_names()
    print(f"\nInitial features: {len(initial_features)}")
    
    # ========================================================================
    # 2. Feature Selection (BEFORE training)
    # ========================================================================
    print("\n[3/6] Performing feature selection...")
    
    # Try different methods
    feature_selector = FeatureSelector(method='tree')  # Options: 'tree', 'mutual_info', 'correlation', 'rfe', 'lasso'
    
    # Fit selector on training data
    feature_selector.fit(
        X=train_df[initial_features],
        y=train_df['target'],
        n_features=None,  # None = auto-select based on importance
        threshold=None    # None = auto-threshold
    )
    
    # Print summary
    feature_selector.print_summary()
    
    # Get selected features
    selected_features = feature_selector.get_selected_features()
    print(f"\nSelected {len(selected_features)} features for training")
    
    # Transform datasets
    train_X = feature_selector.transform(train_df[initial_features])
    val_X = feature_selector.transform(val_df[initial_features])
    test_X = feature_selector.transform(test_df[initial_features])
    
    # Add target back
    train_df_selected = train_X.copy()
    train_df_selected['target'] = train_df['target'].values
    
    val_df_selected = val_X.copy()
    val_df_selected['target'] = val_df['target'].values
    
    test_df_selected = test_X.copy()
    test_df_selected['target'] = test_df['target'].values
    
    # ========================================================================
    # 3. Hyperparameter Optimization
    # ========================================================================
    print("\n[4/6] Optimizing hyperparameters...")
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(
        method='random',  # Options: 'grid', 'random', 'bayesian'
        cv=3,
        n_jobs=-1
    )
    
    # Define models to optimize
    models_to_optimize = {
        'xgboost': {
            'model_class': xgb.XGBClassifier,
            'param_space': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
        },
        'random_forest': {
            'model_class': RandomForestClassifier,
            'param_space': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        }
    }
    
    # Optimize
    optimization_results = optimizer.optimize_multiple(
        models_config=models_to_optimize,
        X_train=train_df_selected[selected_features].values,
        y_train=train_df_selected['target'].values,
        scoring='f1_weighted',
        n_iter=20  # Number of random search iterations
    )
    
    # ========================================================================
    # 4. Train with optimized parameters and GPU support
    # ========================================================================
    print("\n[5/6] Training models with optimized parameters and GPU...")
    
    # Create model manager with GPU support
    model_manager = ModelManager(use_gpu=True)  # Enable GPU for XGBoost, CatBoost, CNNs
    
    # Enable models
    model_manager.enable_model('xgboost', True)
    model_manager.enable_model('random_forest', True)
    
    # Create models (will use GPU if available)
    models = model_manager.create_models()
    
    # Update models with optimized parameters
    for model_name in models.keys():
        if model_name in optimizer.best_params:
            best_params = optimizer.best_params[model_name]
            print(f"\nApplying optimized params to {model_name}:")
            for param, value in best_params.items():
                print(f"  {param}: {value}")
            
            # Recreate model with optimized params
            models[model_name] = model_manager.create_model(model_name, **best_params)
    
    # Train models
    train_manager = TrainManager(use_scaler=True)
    train_output = train_manager.train(
        models=models,
        train_data=train_df_selected,
        target_col='target',
        feature_cols=selected_features,
        val_data=val_df_selected
    )
    
    # ========================================================================
    # 5. Test models
    # ========================================================================
    print("\n[6/6] Testing models...")
    test_results = train_manager.test(
        test_data=test_df_selected,
        target_col='target',
        feature_cols=selected_features
    )
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nFeature Selection:")
    print(f"  Initial features: {len(initial_features)}")
    print(f"  Selected features: {len(selected_features)}")
    print(f"  Reduction: {(1 - len(selected_features)/len(initial_features))*100:.1f}%")
    
    print(f"\nHyperparameter Optimization:")
    for model_name, score in optimizer.best_scores.items():
        print(f"  {model_name}: {score:.4f}")
    
    print(f"\nTest Results:")
    for model_name, results in test_results.items():
        if results.get('status') == 'success':
            print(f"  {model_name}: F1={results['f1_score']:.4f}, Acc={results['accuracy']:.4f}")
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    
    # Optional: Save feature selector for future use
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = Path('results') / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    feature_selector.save(run_dir)
    optimizer.save_results(run_dir)
    
    print(f"\nResults saved to: {run_dir}")


if __name__ == '__main__':
    main()
